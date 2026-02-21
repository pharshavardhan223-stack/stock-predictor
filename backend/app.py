import os
import uuid
import joblib
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, session, flash, send_file
from werkzeug.utils import secure_filename
from backend.utils.data_handler import load_csv, clean_data, preview_table
from backend.utils.predictor import predict_series
from backend.utils.stock_api import fetch_stock_data
from backend.utils.email_utils import send_report

from backend.utils.live_stock import get_live_stock
from config import *
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.colors import black, lightgrey
from backend.utils.email_sender import send_report

from flask import jsonify
import yfinance as yf
import datetime
from reportlab.platypus import Image
from reportlab.lib import colors
from reportlab.lib.units import inch
import datetime
import matplotlib.pyplot as plt
from werkzeug.utils import secure_filename

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Spacer, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from reportlab.lib.colors import lightgrey
import datetime
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Table, TableStyle,
    Spacer, PageBreak, Image
)

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import inch
import matplotlib
matplotlib.use("Agg")   # IMPORTANT for Flask (no GUI)

import matplotlib.pyplot as plt
from reportlab.platypus import Image
from reportlab.lib.units import inch
import datetime

os.makedirs("data", exist_ok=True)
# Login users (Username & Password)
USERS = {
    "admin": "admin123",
    "user": "1234"
}


# ==========================
# APP SETUP
# ==========================

app = Flask(__name__)
app.secret_key = SECRET_KEY
app.jinja_env.globals.update(zip=zip)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["PROFILE_UPLOAD_FOLDER"] = "static/profile_pics"
os.makedirs(app.config["PROFILE_UPLOAD_FOLDER"], exist_ok=True)


# Create required folders
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs("models", exist_ok=True)


# ==========================
# LOAD ML MODEL
# ==========================

MODEL_PATH = "models/linear_model.pkl"

model = None

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully.")
else:
    print("Warning: Model file not found. Train model first.")


# ==========================
# HELPERS
# ==========================

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS



def predict_with_model(series, days=7):

    global model

    y = np.array(series).reshape(-1, 1)
    x = np.arange(len(y)).reshape(-1, 1)

    # Train model if not exists
    if model is None:

        from sklearn.linear_model import LinearRegression

        model = LinearRegression()
        model.fit(x, y)

        joblib.dump(model, MODEL_PATH)

    future_x = np.arange(len(y), len(y) + days).reshape(-1, 1)

    predictions = model.predict(future_x)

    return predictions.flatten()


def generate_recommendation(last_actual, predicted_avg):

    if predicted_avg > last_actual:
        return "BUY", "Uptrend detected"
    elif predicted_avg < last_actual:
        return "SELL", "Downtrend detected"
    else:
        return "HOLD", "Stable movement"


# ==========================
# AUTH ROUTES
# ==========================
@app.route("/", methods=["GET", "POST"])
def login():

    if request.method == "POST":

        username = request.form.get("username")
        password = request.form.get("password")

        print("LOGIN TRY:", username, password)   # Debug
        print("USERS:", USERS)                    # Debug

        if username in USERS and USERS[username] == password:

            
          if username in USERS and USERS[username] == password:
            session["user"] = username   # ✅ THIS LINE MUST BE HERE
            return redirect(url_for("dashboard"))
  

        else:
            flash("Invalid username or password")

    return render_template("index.html")


# ==========================
# STOCK API INPUT
# ==========================

@app.route("/stock")
def stock_input():

    if "user" not in session:
        return redirect(url_for("login"))

    return render_template("stock_input.html")


@app.route("/fetch_stock", methods=["POST"])
def fetch_stock():

    if "user" not in session:
        return redirect(url_for("login"))

    symbol = request.form.get("symbol")

    if not symbol:
        flash("Please enter stock symbol")
        return redirect(url_for("stock_input"))


    # Fetch data from API
    filepath = fetch_stock_data(symbol.upper())


    if not filepath:
        flash("Invalid stock symbol or no data found")
        return redirect(url_for("stock_input"))


    # Save path in session (like upload)
    session["file_path"] = filepath


    return redirect(url_for("preview"))


# ==========================
# PREVIEW
# ==========================

@app.route("/preview")
def preview():

    if "user" not in session:
        return redirect(url_for("login"))

    if "file_path" not in session:
        return redirect(url_for("upload"))

    df = load_csv(session["file_path"], MAX_ROWS)
    df = clean_data(df)

    table_html = preview_table(df)

    columns = df.select_dtypes(include=["float64", "int64"]).columns.tolist()

    return render_template(
        "preview.html",
        table=table_html,
        columns=columns
    )
# ==========================
# GRAPH SETTINGS
# ==========================

@app.route("/graph-settings")
def graph_settings():

    if "user" not in session:
        return redirect(url_for("login"))

    if "file_path" not in session:
        return redirect(url_for("upload"))

    df = load_csv(session["file_path"])

    columns = df.columns.tolist()

    return render_template(
        "graph_settings.html",
        columns=columns
    )


# ==========================
# COLUMN SELECTION
# ==========================

@app.route("/select", methods=["POST"])
def select_columns():

    if "user" not in session:
        return redirect(url_for("login"))

    # Get selected columns from form
    selected_columns = request.form.getlist("columns")

    if not selected_columns:
        flash("Please select at least one column")
        return redirect(url_for("preview"))

    # Save to session
    session["columns"] = selected_columns

    return redirect(url_for("charts"))


# ==========================
# CHARTS
# ==========================
@app.route("/charts", methods=["GET", "POST"])
def charts():

    if "user" not in session:
        return redirect(url_for("login"))

    if "file_path" not in session or "columns" not in session:
        return redirect(url_for("upload"))

    df = load_csv(session["file_path"])

    columns = session.get("columns")

    # Default chart
    chart_type = session.get("chart_type", "line")

    # Form submit
    if request.method == "POST":
        chart_type = request.form.get("chart_type", "line")
        session["chart_type"] = chart_type


    # Prepare data
    data = {}

    for col in columns:
        data[col] = df[col].astype(float).tolist()


    return render_template(
        "graphs.html",
        data=data,
        columns=columns,
        chart_type=chart_type
    )

# ==========================
# PREDICTION (LEVEL 6.1)
# ==========================

# ==========================
# PREDICTION
# ==========================

# ==========================
# PREDICTION
# ==========================

import matplotlib.pyplot as plt
import csv
import datetime
@app.route("/predict")
def predict():

    if "user" not in session:
        return redirect(url_for("login"))

    if "file_path" not in session or "columns" not in session:
        return redirect(url_for("upload"))

    import csv
    import datetime

    # Load and clean data
    df = load_csv(session["file_path"])
    df = clean_data(df)

    columns = session["columns"]

    results = {}
    predictions = {}
    analytics = {}

    # History file setup
    history_file = "data/history.csv"
    os.makedirs("data", exist_ok=True)

    # Create file if not exists
    if not os.path.exists(history_file):
        with open(history_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["username", "stock", "action", "confidence", "date"])

    # =========================
    # LOOP THROUGH COLUMNS
    # =========================
    for col in columns:

        series = df[col].astype(float).tolist()

        output = predict_series(series)

        # Save predictions
        predictions[col] = output["predictions"]

        # Recommendation
        if output["future_avg"] > output["last_actual"]:
            action = "BUY"
        elif output["future_avg"] < output["last_actual"]:
            action = "SELL"
        else:
            action = "HOLD"

        # AI Confidence (R2 → %)
        confidence = round(output["r2"] * 100, 2)

        # Risk Level
        if confidence >= 80:
            risk = "Low Risk"
        elif confidence >= 60:
            risk = "Medium Risk"
        else:
            risk = "High Risk"

        # Save main result
        results[col] = {
            "last": output["last_actual"],
            "avg": output["future_avg"],
            "action": action,
            "reason": f"Trend is {output['trend_percent']}%",
            "confidence": confidence,
            "risk": risk
        }

        # Save analytics
        analytics[col] = output

        # =========================
        # SAVE ACTIVITY TO CSV
        # =========================
        with open(history_file, "a", newline="") as f:
            writer = csv.writer(f)

            writer.writerow([
                session.get("user"),
                col,
                action,
                confidence,
                datetime.datetime.now().strftime("%d-%m-%Y %H:%M")
            ])


    # =========================
    # SAVE FOR PDF
    # =========================
    session["results"] = results
    session["predictions"] = predictions
    session["analytics"] = analytics


    return render_template(
        "predictions.html",
        results=results,
        predictions=predictions,
        analytics=analytics
    )



# ==========================
# RESULT PAGE
# ==========================

@app.route("/predictions")
def show_predictions():

    if "user" not in session:
        return redirect(url_for("login"))

    return render_template(
        "predictions.html",
        predictions=session.get("predictions"),
        results=session.get("results")
    )
# ==========================
# DASHBOARD (LIVE DATA)
# ==========================

@app.route("/dashboard")
def dashboard():

    if "user" not in session:
        return redirect(url_for("login"))

    # REAL DATA USING YFINANCE
    import yfinance as yf
    import datetime

    symbol = "AAPL"

    stock = yf.Ticker(symbol)

    data = stock.history(period="7d")

    last7 = []

    chart_labels = []
    chart_data = []

    for date, row in data.iterrows():

        d = date.strftime("%b %d")
        price = round(float(row["Close"]),2)

        last7.append((d, price))
        chart_labels.append(d)
        chart_data.append(price)

    # Latest price
    live_price = chart_data[-1] if chart_data else "N/A"

    return render_template(
        "dashboard.html",

        live_price=live_price,
        last7=last7,

        chart_labels=chart_labels,
        chart_data=chart_data
    )


# ==========================
# PROFILE
# ==========================

@app.route("/profile")
def profile():

    if "user" not in session:
        return redirect(url_for("login"))

    username = session["user"]

    history_file = "data/history.csv"

    total_predictions = 0
    last_activity = "N/A"

    activities = []   # 👈 NEW


    if os.path.exists(history_file):

        df = pd.read_csv(history_file)

        user_data = df[df["username"] == username]

        total_predictions = len(user_data)

        if not user_data.empty:

            last_activity = user_data.iloc[-1]["date"]

            # Build activity list
            for _, row in user_data.tail(5).iterrows():

                activities.append({
                    "action": row["action"],
                    "time": row["date"]
                })


    return render_template(
        "profile.html",
        username=username,
        total=total_predictions,
        last=last_activity,
        activities=activities   # 👈 SEND TO HTML
    )
@app.route("/update-profile", methods=["POST"])
def update_profile():

    if "user" not in session:
        return redirect(url_for("login"))

    name = request.form.get("name")
    email = request.form.get("email")

    # Save in session (simple method)
    session["profile_name"] = name
    session["profile_email"] = email

    flash("Profile Updated Successfully")

    return redirect(url_for("profile"))
@app.route("/upload-avatar", methods=["POST"])
def upload_avatar():

    if "user" not in session:
        return redirect(url_for("login"))

    if "avatar" not in request.files:
        flash("No file selected")
        return redirect(url_for("profile"))

    file = request.files["avatar"]

    if file.filename == "":
        flash("No file selected")
        return redirect(url_for("profile"))

    # Save file
    filename = secure_filename(file.filename)

    save_path = os.path.join("static/uploads", filename)

    file.save(save_path)

    # Save path in session
    session["avatar"] = "/" + save_path.replace("\\","/")

    flash("Profile photo updated")

    return redirect(url_for("profile"))
# ==========================
# EDIT PROFILE PAGE
# ==========================
@app.route("/edit-profile", methods=["GET", "POST"])
def edit_profile():

    if "user" not in session:
        return redirect(url_for("login"))

    if request.method == "POST":

        name = request.form.get("name")
        email = request.form.get("email")

        # Update session values
        session["user"] = name
        session["email"] = email

        # Handle profile photo
        if "photo" in request.files:
            file = request.files["photo"]

            if file.filename != "":
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config["PROFILE_UPLOAD_FOLDER"], filename)
                file.save(filepath)

                session["profile_photo"] = filename

        flash("Profile updated successfully!")
        return redirect(url_for("profile"))

    return render_template(
        "edit_profile.html",
        username=session.get("user"),
        email=session.get("email"),
        photo=session.get("profile_photo")
    )



# ==========================
# UPLOAD
# ==========================

@app.route("/upload", methods=["GET", "POST"])
def upload_file():

    if "user" not in session:
        return redirect(url_for("login"))

    if request.method == "POST":

        if "file" not in request.files:
            flash("No file selected")
            return redirect(request.url)

        file = request.files["file"]

        if file.filename == "":
            flash("No file selected")
            return redirect(request.url)

        if not allowed_file(file.filename):
            flash("Only CSV files allowed")
            return redirect(request.url)

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)

        file.save(filepath)

        session["file_path"] = filepath

        return redirect(url_for("preview"))

    return render_template("upload.html")

# ==========================
# LIVE STOCK FETCH
# ==========================

@app.route("/live", methods=["GET", "POST"])
def live_stock():

    if "user" not in session:
        return redirect(url_for("login"))


    if request.method == "POST":

        symbol = request.form.get("symbol")

        if not symbol:
            flash("Enter stock symbol")
            return redirect("/live")


        file_path = get_live_stock(symbol.upper())


        if not file_path:
            flash("Invalid symbol")
            return redirect("/live")


        session["file_path"] = file_path

        return redirect("/preview")


    return render_template("live_stock.html")


# ==========================
# RUN
# ==========================
# ==========================
# DOWNLOAD PROFESSIONAL PDF
# ==========================
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Table,
    TableStyle,
    Spacer,
    Image,
    PageBreak
)
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib import colors
import datetime
import os
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Table,
    TableStyle,
    Spacer,
    Image,
    PageBreak
)
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib import colors

# ======================
# PDF WATERMARK
# ======================

def add_watermark(canvas, doc):

    canvas.saveState()

    canvas.setFont("Helvetica-Bold", 45)
    canvas.setFillColor(colors.lightgrey)

    # Position + rotation
    canvas.translate(200, 200)
    canvas.rotate(45)

    # Text
    canvas.drawString(0, 0, "DATA MARKERS")

    canvas.restoreState()

@app.route("/download_report")
def download_report():

    # ======================
    # LOGIN CHECK
    # ======================
    if "user" not in session:
        return redirect(url_for("login"))

    # ======================
    # DATA CHECK
    # ======================
    if "results" not in session:
        flash("No report available")
        return redirect(url_for("dashboard"))

    results = session.get("results", {})
    charts = session.get("charts", {})

    # ======================
    # FILE SETUP
    # ======================
    filename = "StockAI_Report.pdf"
    filepath = os.path.join("data", filename)

    os.makedirs("data", exist_ok=True)

    # ======================
    # PDF SETUP
    # ======================
    doc = SimpleDocTemplate(
        filepath,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()
    elements = []


    # ======================
    # LOGO (SAFE LOAD)
    # ======================
    logo_path = "static/images/logo.png"

    try:
        if os.path.exists(logo_path):
            logo = Image(logo_path, width=2*inch, height=0.8*inch)
            elements.append(logo)
            elements.append(Spacer(1, 15))
    except:
        pass


    # ======================
    # TITLE
    # ======================
    elements.append(
        Paragraph("STOCK AI ANALYTICS REPORT", styles["Title"])
    )

    elements.append(Spacer(1, 10))


    # ======================
    # USER + DATE
    # ======================
    today = datetime.datetime.now().strftime("%d-%m-%Y %H:%M")

    elements.append(
        Paragraph(f"<b>User:</b> {session.get('user','N/A')}", styles["Normal"])
    )

    elements.append(
        Paragraph(f"<b>Date:</b> {today}", styles["Normal"])
    )

    elements.append(Spacer(1, 25))


    # ==================================================
    # EXECUTIVE SUMMARY PAGE
    # ==================================================

    elements.append(
        Paragraph("EXECUTIVE SUMMARY", styles["Heading1"])
    )

    elements.append(Spacer(1, 20))


    total_stocks = len(results)

    conf_list = [r.get("confidence", 0) for r in results.values()]
    avg_conf = round(sum(conf_list)/len(conf_list),2) if conf_list else 0


    buy = sum(1 for r in results.values() if r.get("action")=="BUY")
    sell = sum(1 for r in results.values() if r.get("action")=="SELL")
    hold = sum(1 for r in results.values() if r.get("action")=="HOLD")


    low = sum(1 for r in results.values() if r.get("risk")=="Low Risk")
    med = sum(1 for r in results.values() if r.get("risk")=="Medium Risk")
    high = sum(1 for r in results.values() if r.get("risk")=="High Risk")


    summary_data = [

        ["Metric", "Value"],

        ["Total Stocks", str(total_stocks)],
        ["Avg Confidence", f"{avg_conf} %"],

        ["BUY Signals", str(buy)],
        ["SELL Signals", str(sell)],
        ["HOLD Signals", str(hold)],

        ["Low Risk", str(low)],
        ["Medium Risk", str(med)],
        ["High Risk", str(high)],
    ]


    summary_table = Table(summary_data, colWidths=[220,180])

    summary_table.setStyle(TableStyle([

        ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#0d47a1")),
        ("TEXTCOLOR",(0,0),(-1,0),colors.white),

        ("GRID",(0,0),(-1,-1),0.6,colors.grey),

        ("ALIGN",(0,0),(-1,-1),"CENTER"),

        ("BOTTOMPADDING",(0,0),(-1,-1),8),
        ("TOPPADDING",(0,0),(-1,-1),8),

    ]))


    elements.append(summary_table)

    elements.append(Spacer(1,30))


    elements.append(
        Paragraph(
            "This report summarizes AI-driven financial forecasting and investment insights.",
            styles["Normal"]
        )
    )


    elements.append(PageBreak())


    # ==================================================
    # STOCK ANALYSIS PAGES
    # ==================================================

    for col, res in results.items():

        # Section Title
        elements.append(
            Paragraph(f"{col} Analysis", styles["Heading2"])
        )

        elements.append(Spacer(1,10))


        # Table
        table_data = [

            ["Metric","Value"],

            ["Last Value", str(res.get("last",""))],
            ["Predicted Avg", str(res.get("avg",""))],

            ["Action", res.get("action","")],
            ["Reason", res.get("reason","")],

            ["Confidence", str(res.get("confidence",""))+" %"],
            ["Risk Level", res.get("risk","")]

        ]


        table = Table(table_data, colWidths=[200,200])

        table.setStyle(TableStyle([

            ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#1976d2")),
            ("TEXTCOLOR",(0,0),(-1,0),colors.white),

            ("GRID",(0,0),(-1,-1),0.6,colors.grey),

            ("ALIGN",(0,0),(-1,-1),"CENTER"),

            ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),

            ("BOTTOMPADDING",(0,0),(-1,-1),8),
            ("TOPPADDING",(0,0),(-1,-1),8),

        ]))


        elements.append(table)
        elements.append(Spacer(1,15))


        # ======================
        # ADD CHART IMAGE
        # ======================

        chart_path = charts.get(col)

        if chart_path and os.path.exists(chart_path):

            try:
                chart_img = Image(
                    chart_path,
                    width=5.5*inch,
                    height=3.5*inch
                )

                elements.append(chart_img)
                elements.append(Spacer(1,20))

            except:
                pass


        elements.append(PageBreak())


    # ======================
    # FOOTER
    # ======================

    elements.append(
        Paragraph(
            "This report is generated automatically by Stock AI Analytics System.",
            styles["Normal"]
        )
    )


    # ======================
    # BUILD PDF
    # ======================

    doc.build(elements)


    # ======================
    # RETURN FILE
    # ======================

    return send_file(filepath, as_attachment=True)



@app.route("/history")
def history():

    if "user" not in session:
        return redirect(url_for("login"))

    history_file = "data/history.csv"

    if not os.path.exists(history_file):
        return render_template("history.html", records=[])

    df = pd.read_csv(history_file)

    # Filter only current user
    user_df = df[df["username"] == session["user"]]

    records = user_df.to_dict(orient="records")

    return render_template(
        "history.html",
        records=records
    )

@app.route("/live-data/<symbol>")
def live_data(symbol):

    try:
        stock = yf.Ticker(symbol)

        data = stock.history(period="1d", interval="1m")

        if data.empty:
            return jsonify({
                "price": "N/A",
                "time": "N/A"
            })

        latest = data.iloc[-1]

        price = round(float(latest["Close"]), 2)

        time = datetime.datetime.now().strftime("%H:%M:%S")

        return jsonify({
            "price": price,
            "time": time
        })

    except Exception as e:

        print("Live API Error:", e)

        return jsonify({
            "price": "Error",
            "time": "N/A"
        })
    
@app.route("/admin")
def admin_panel():

    if "user" not in session:
        return redirect(url_for("login"))

    if session.get("role") != "admin":
        return "Access Denied", 403


    # Read users (example file)
    users = []

    if os.path.exists("data/users.txt"):
        with open("data/users.txt") as f:
            users = f.readlines()


    # Read history
    history = []

    if os.path.exists("data/history.csv"):
        with open("data/history.csv") as f:
            history = f.readlines()[1:]


    return render_template(
        "admin.html",
        users=users,
        history=history
    )

if __name__ == "__main__":

    app.run(debug=DEBUG, host=HOST, port=PORT)



