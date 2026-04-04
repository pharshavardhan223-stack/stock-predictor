import os
import uuid
import joblib
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, session, flash, send_file
from werkzeug.utils import secure_filename
from werkzeug.security import check_password_hash
from backend.utils.data_handler import load_csv, clean_data, preview_table
from backend.utils.predictor import predict_series
from backend.utils.stock_api import fetch_stock_data

from backend.config import *
from flask import jsonify
import yfinance as yf
import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sqlite3
import csv

from backend.utils.auto_train    import start_auto_train
from backend.utils.email_alerts  import init_mail, start_alert_checker

from reportlab.pdfgen        import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.colors    import HexColor
import sqlite3
import os
from flask import request, redirect, url_for, session, flash, render_template
from backend.routes.advanced_analysis import advanced_analysis_bp
from backend.routes.notifications import add_price_alert_notification
from backend.routes.notifications import add_watchlist_notification
from backend.routes.notifications import add_prediction_notification
# ==========================
# RAG DATABASE CLASS
# ==========================

class RAGDatabase:
    def __init__(self, db_path="data/rag_knowledge.db"):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # User trust score table
        c.execute('''CREATE TABLE IF NOT EXISTS user_trust (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            trust_score REAL DEFAULT 50.0,
            total_predictions INTEGER DEFAULT 0,
            correct_predictions INTEGER DEFAULT 0,
            accuracy REAL DEFAULT 0.0,
            risk_profile TEXT DEFAULT 'moderate',
            last_active TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )''')
        
        # Prediction history for RAG retrieval
        c.execute('''CREATE TABLE IF NOT EXISTS prediction_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            stock_symbol TEXT,
            prediction_date TIMESTAMP,
            predicted_action TEXT,
            actual_action TEXT,
            predicted_price REAL,
            actual_price REAL,
            confidence REAL,
            was_correct BOOLEAN,
            accuracy_score REAL
        )''')
        
        # User conversation context
        c.execute('''CREATE TABLE IF NOT EXISTS conversation_context (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            session_id TEXT,
            query TEXT,
            response TEXT,
            intent TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )''')
        
        # User feedback
        c.execute('''CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            query TEXT,
            response TEXT,
            rating INTEGER,
            helpful BOOLEAN,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )''')
        
        conn.commit()
        conn.close()
    
    def update_trust_score(self, username, prediction_accuracy):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute("SELECT trust_score, total_predictions, correct_predictions FROM user_trust WHERE username = ?", (username,))
        result = c.fetchone()
        
        if result:
            trust_score, total, correct = result
            total += 1
            if prediction_accuracy > 0.7:
                correct += 1
                trust_score = min(100, trust_score + (prediction_accuracy * 5))
            else:
                trust_score = max(0, trust_score - 10)
            
            accuracy = (correct / total) * 100 if total > 0 else 0
            
            c.execute('''UPDATE user_trust 
                        SET trust_score = ?, total_predictions = ?, correct_predictions = ?, 
                            accuracy = ?, last_active = CURRENT_TIMESTAMP
                        WHERE username = ?''',
                     (trust_score, total, correct, accuracy, username))
        else:
            trust_score = 50 + (prediction_accuracy * 20)
            c.execute('''INSERT INTO user_trust (username, trust_score, total_predictions, correct_predictions, accuracy)
                        VALUES (?, ?, 1, ?, ?)''',
                     (username, trust_score, 1 if prediction_accuracy > 0.7 else 0, prediction_accuracy * 100))
        
        conn.commit()
        conn.close()
        return trust_score
    
    def get_user_context(self, username, limit=10):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute("SELECT trust_score, accuracy, risk_profile FROM user_trust WHERE username = ?", (username,))
        trust_data = c.fetchone()
        
        c.execute('''SELECT stock_symbol, predicted_action, was_correct, confidence
                    FROM prediction_history 
                    WHERE username = ? 
                    ORDER BY prediction_date DESC 
                    LIMIT ?''', (username, limit))
        predictions = c.fetchall()
        
        c.execute('''SELECT query, response, intent 
                    FROM conversation_context 
                    WHERE username = ? 
                    ORDER BY timestamp DESC 
                    LIMIT 5''', (username,))
        conversations = c.fetchall()
        
        conn.close()
        
        return {
            "trust_score": trust_data[0] if trust_data else 50,
            "accuracy": trust_data[1] if trust_data else 0,
            "risk_profile": trust_data[2] if trust_data else "moderate",
            "recent_predictions": predictions,
            "recent_conversations": conversations,
            "total_predictions": trust_data[1] if trust_data else 0,
            "correct_predictions": int((trust_data[1] if trust_data else 0) * (trust_data[2] if trust_data else 0) / 100) if trust_data else 0
        }
    
    def add_prediction_record(self, username, stock, predicted_action, actual_action, 
                              predicted_price, actual_price, confidence, was_correct):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        accuracy_score = 1.0 if was_correct else 0.0
        
        c.execute('''INSERT INTO prediction_history 
                    (username, stock_symbol, prediction_date, predicted_action, actual_action, 
                     predicted_price, actual_price, confidence, was_correct, accuracy_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                 (username, stock, datetime.datetime.now(), predicted_action, actual_action,
                  predicted_price, actual_price, confidence, was_correct, accuracy_score))
        
        self.update_trust_score(username, accuracy_score)
        
        conn.commit()
        conn.close()
    
    def add_conversation(self, username, session_id, query, response, intent):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute('''INSERT INTO conversation_context 
                    (username, session_id, query, response, intent)
                    VALUES (?, ?, ?, ?, ?)''',
                 (username, session_id, query, response, intent))
        
        conn.commit()
        conn.close()
    
    def add_feedback(self, username, query, response, rating, helpful):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute('''INSERT INTO feedback (username, query, response, rating, helpful)
                    VALUES (?, ?, ?, ?, ?)''',
                 (username, query, response, rating, helpful))
        
        conn.commit()
        conn.close()
    
    def get_similar_predictions(self, stock_symbol, limit=5):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute('''SELECT predicted_action, was_correct, confidence, accuracy_score
                    FROM prediction_history 
                    WHERE stock_symbol = ?
                    ORDER BY prediction_date DESC 
                    LIMIT ?''', (stock_symbol, limit))
        
        results = c.fetchall()
        conn.close()
        
        return [{"action": r[0], "correct": r[1], "confidence": r[2], "accuracy": r[3]} for r in results]


# ==========================
# RAG CHATBOT CLASS
# ==========================

class RAGChatbot:
    def __init__(self, api_key=None):
        self.db = RAGDatabase()
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
        self.use_claude = bool(self.api_key)
    
    # ========== NEW METHODS ==========
    
    def generate_daily_tip(self):
        return """💡 **Daily Trading Tip** (Risk Management)

Always use stop-loss orders to protect your capital. Never risk more than 2% of your account on a single trade.

_Pro tip: Set stop-loss at key support levels._"""
    
    def generate_market_summary(self, username, market_data=None):
        from datetime import datetime
        hour = datetime.now().hour
        greeting = "Morning" if hour < 12 else "Afternoon" if hour < 17 else "Evening"
        market_status = "🟢 OPEN" if 9 <= hour < 15 else "🔴 CLOSED"
        return f"""📊 **Good {greeting}, {username.title()}!**

{market_status}

📈 **Market Overview:**
• NIFTY 50: ▲ +0.42%
• SENSEX: ▲ +0.38%

_Type 'help' for commands!_"""
    
    def suggest_follow_up_questions(self, history):
        return ["📊 Trust score?", "🔮 Predict stock?", "📈 Trading advice?"]
    
    def get_user_preferences(self, username):
        return self.db.get_user_preferences(username)
    
    def update_user_preferences(self, username, prefs):
        self.db.update_user_preferences(username, prefs)
    
    # ========== EXISTING METHODS ==========
    
    def retrieve_context(self, username, query, stock_symbol=None):
        context = {
            "user_context": self.db.get_user_context(username),
            "relevant_predictions": []
        }
        
        if stock_symbol:
            context["relevant_predictions"] = self.db.get_similar_predictions(stock_symbol)
        
        predictions = context["user_context"].get("recent_predictions", [])
        stock_specific = [p for p in predictions if stock_symbol and p[0] == stock_symbol]
        
        if stock_specific:
            correct_count = sum(1 for p in stock_specific if p[2])
            context["user_accuracy_on_stock"] = {
                "total": len(stock_specific),
                "correct": correct_count,
                "accuracy": (correct_count / len(stock_specific)) * 100 if stock_specific else 0
            }
        
        return context
    
    def generate_rag_response(self, username, query, stock_symbol=None, market_data=None, results=None, analytics=None, preds=None):
        context = self.retrieve_context(username, query, stock_symbol)
        user_context = context["user_context"]
        
        trust_score = user_context['trust_score']
        accuracy = user_context['accuracy']
        
        if trust_score >= 70:
            trust_prefix = "Based on your strong track record"
            confidence_multiplier = 1.2
            level = "Expert"
        elif trust_score >= 40:
            trust_prefix = "Based on your trading history"
            confidence_multiplier = 1.0
            level = "Learning"
        else:
            trust_prefix = "To help you build experience"
            confidence_multiplier = 0.8
            level = "Beginner"
        
        # Check if this is a greeting
        if any(w in query.lower() for w in ["hello", "hi", "hey", "good morning", "good evening"]):
            return f"{trust_prefix} How can I help you today?"
        
        # Check for market summary request
        if any(w in query.lower() for w in ["market summary", "market overview", "today's market"]):
            return self.generate_market_summary(username, market_data)
        
        # Check for daily tip request
        if any(w in query.lower() for w in ["daily tip", "tip of the day", "trading tip"]):
            return self.generate_daily_tip()
        
        # Check if this is a prediction-related query
        if any(w in query.lower() for w in ["buy", "sell", "invest", "trade", "predict", "forecast"]):
            if results and stock_symbol and stock_symbol in results:
                res = results[stock_symbol]
                action = res.get("action", "HOLD")
                confidence = res.get("confidence", 0) * confidence_multiplier
                risk = res.get("risk", "Medium")
                
                stock_history = context.get("user_accuracy_on_stock", {})
                history_text = ""
                if stock_history.get("total", 0) > 0:
                    history_text = f"\n\n📊 **Your history with {stock_symbol}:** {stock_history['correct']}/{stock_history['total']} correct predictions ({stock_history['accuracy']:.0f}% accuracy)"
                
                response = f"""{trust_prefix}, here's my analysis for **{stock_symbol}**:

🔮 **AI Prediction:** {action}
📈 **Confidence:** {confidence:.1f}%
⚠️ **Risk Level:** {risk}

📊 **Technical Indicators:**
• RSI: {analytics.get(stock_symbol, {}).get('rsi', 'N/A')}
• MACD: {analytics.get(stock_symbol, {}).get('macd', 'N/A')}
• Trend: {analytics.get(stock_symbol, {}).get('trend_percent', 0):.1f}%{history_text}

💡 **Recommendation:** {"Consider entering with proper stop-loss" if action == "BUY" else "Consider reducing exposure" if action == "SELL" else "Wait for clearer signals"}

🎯 **Suggested Stop-Loss:** {res.get('last', 0) * 0.98:.2f}
🎯 **Target Price:** {res.get('avg', 0):.2f}

_Your trust score: {trust_score:.0f}/100 ({level} Trader)_"""
            else:
                response = f"{trust_prefix}, I need more data to analyze {stock_symbol}. Try running a prediction first or ask about a stock in your watchlist."
        
        elif "trust" in query.lower() or "score" in query.lower():
            level_text = "Expert Trader" if trust_score >= 80 else "Experienced Trader" if trust_score >= 60 else "Learning Trader" if trust_score >= 40 else "New Trader"
            response = f"""🔒 **Your Trust Score: {trust_score:.0f}/100**

📊 **Performance Metrics:**
• Prediction Accuracy: {accuracy:.1f}%
• Total Predictions: {user_context.get('total_predictions', 0)}
• Trust Level: {level_text}

💡 **How to Improve Your Trust Score:**
1. ✅ Make more predictions (each correct prediction increases score)
2. 📊 Use our AI prediction tools
3. 🎯 Set realistic targets and stop-losses

Higher trust scores unlock more confident recommendations and advanced features!"""
        
        else:
            response = f"""🤖 **AI Trading Assistant**

{trust_prefix}, I can help you with:

📈 **Trading Advice** - "Should I buy AAPL?"
🔮 **Price Predictions** - "Predict NVDA for next week"
📊 **Portfolio Analysis** - "Review my portfolio risk"
🎯 **Trust Score** - "What's my trust rating?"

Your current trust score: {trust_score:.0f}/100 ({'Expert' if trust_score > 70 else 'Learning' if trust_score > 40 else 'Beginner'})

_Type 'help' for more commands or ask about a specific stock!_"""
        
        if trust_score >= 80:
            footer = "\n\n---\n⭐ **Trusted Trader** - Your excellent track record gives high confidence in this analysis."
        elif trust_score >= 60:
            footer = "\n\n---\n✅ **Verified Trader** - Good history. Continue building your track record!"
        elif trust_score >= 40:
            footer = "\n\n---\n📚 **Learning Trader** - Consider paper trading to build confidence first."
        else:
            footer = "\n\n---\n🎓 **New Trader** - Start with paper trading and small positions to build experience."
        
        self.db.add_conversation(username, "session_rag", query, response, "trading_advice")
        
        return response + footer


def generate_daily_tip(self):
    """Generate a daily trading tip"""
    return """💡 **Daily Trading Tip** (Risk Management)

Always use stop-loss orders to protect your capital. Never risk more than 2% of your account on a single trade.

_Pro tip: Set stop-loss at key support levels._"""

def generate_market_summary(self, username, market_data=None):
    """Generate personalized market summary for the user"""
    from datetime import datetime
    current_hour = datetime.now().hour
    
    if 9 <= current_hour < 15:
        market_status = "🟢 Market is OPEN"
    else:
        market_status = "🔴 Market is CLOSED"
    
    if current_hour < 12:
        greeting = "Morning"
    elif current_hour < 17:
        greeting = "Afternoon"
    else:
        greeting = "Evening"
    
    return f"""📊 **Good {greeting}, {username.title()}!**

{market_status}

📈 **Market Overview:**
• NIFTY 50: ▲ +0.42%
• SENSEX: ▲ +0.38%
• Market Sentiment: 🟢 Bullish

💡 **Today's Focus:**
Check your dashboard for real-time updates and predictions.

_Type 'help' for commands!_"""
def suggest_follow_up_questions(self, conversation_history):
    """Generate smart follow-up questions"""
    return [
        "📊 What's my trust score?",
        "🔮 Predict a stock for me",
        "📈 Should I buy AAPL?",
        "📰 Show me market news"
    ]

def get_conversation_summary(self, username, limit=5):
    """Get conversation summary"""
    return self.db.get_conversation_summary(username, limit)

def get_user_preferences(self, username):
    """Get user preferences"""
    return self.db.get_user_preferences(username)

def update_user_preferences(self, username, preferences):
    """Update user preferences"""
    self.db.update_user_preferences(username, preferences)
# ==========================
# APP SETUP
# ==========================

app = Flask(__name__)
app.secret_key = SECRET_KEY
app.config["SESSION_COOKIE_NAME"] = "stockai_v2"
app.jinja_env.globals.update(zip=zip)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["PROFILE_UPLOAD_FOLDER"] = "static/profile_pics"

os.makedirs(app.config["PROFILE_UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)

# ==========================
# REGISTER BLUEPRINTS & NEW ROUTES
# ==========================

from backend.routes.auth_routes import auth, init_auth_db 
app.register_blueprint(auth)
init_auth_db()

from backend.routes.new_routes import register_new_routes
register_new_routes(app)

from backend.routes.advanced_analysis import advanced_analysis_bp
app.register_blueprint(advanced_analysis_bp)

from backend.routes.notifications import notifications_bp
app.register_blueprint(notifications_bp)


# ==========================
# AUTO-TRAIN + EMAIL ALERTS
# ==========================

start_auto_train()
init_mail(app)
start_alert_checker()

# ==========================
# INITIALIZE RAG CHATBOT
# ==========================

rag_chatbot = RAGChatbot(api_key=os.getenv("ANTHROPIC_API_KEY", ""))

# ==========================
# LOAD ML MODEL
# ==========================

MODEL_PATH = "models/linear_model.pkl"
model = None

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    print("✅ Model loaded successfully.")
else:
    print("⚠️ Warning: Model file not found. Train model first.")


# ==========================
# ENSURE WATCHLIST + ALERTS TABLES
# ==========================

def _ensure_tables():
    try:
        conn = sqlite3.connect("data/users.db")
        c = conn.cursor()
        c.execute("""CREATE TABLE IF NOT EXISTS watchlist (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            symbol TEXT NOT NULL,
            added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(username, symbol)
        )""")
        c.execute("""CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            symbol TEXT NOT NULL,
            target_price REAL NOT NULL,
            direction TEXT NOT NULL DEFAULT 'above',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )""")
        conn.commit()
        conn.close()
    except Exception as e:
        print("_ensure_tables error:", e)

_ensure_tables()


# ==========================
# HELPERS
# ==========================

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def predict_with_model(series, days=7):
    global model
    y = np.array(series).reshape(-1, 1)
    x = np.arange(len(y)).reshape(-1, 1)
    if model is None:
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(x, y)
        joblib.dump(model, MODEL_PATH)
    future_x = np.arange(len(y), len(y) + days).reshape(-1, 1)
    predictions = model.predict(future_x)
    return predictions.flatten()


def generate_recommendation(last_actual, predicted_avg):
    if predicted_avg > last_actual:   return "BUY", "Uptrend detected"
    elif predicted_avg < last_actual: return "SELL", "Downtrend detected"
    else: return "HOLD", "Stable movement"


def fetch_last_one_year(symbol):
    try:
        stock = yf.Ticker(symbol.upper())
        data = stock.history(period="1y")
        if data is None or data.empty: return None
        return data
    except Exception as e:
        print("Error fetching stock:", e)
        return None


def filter_by_month(data, year, month):
    try:
        start_date = f"{year}-{int(month):02d}-01"
        end_date = pd.to_datetime(start_date) + pd.offsets.MonthEnd(0)
        filtered = data.loc[start_date:end_date]
        return None if filtered.empty else filtered
    except Exception as e:
        print("Error filtering data:", e)
        return None


def prepare_chart_data(filtered_df):
    chart_labels, chart_data = [], []
    for date, row in filtered_df.iterrows():
        chart_labels.append(date.strftime("%b %d"))
        chart_data.append(round(float(row["Close"]), 2))
    return chart_labels, chart_data


def prepare_preview_table(filtered_df):
    preview_df = filtered_df.reset_index()
    return preview_df.to_html(classes="table table-striped table-bordered", index=False)


# ==========================
# LOGIN
# ==========================

@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "GET":
        session.clear()
        return render_template("index.html")

    username = request.form.get("username", "").strip()
    password = request.form.get("password", "")

    if not username or not password:
        flash("Please enter both username and password", "error")
        return render_template("index.html")

    try:
        conn = sqlite3.connect("data/users.db")
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("SELECT * FROM users WHERE username = ?", (username,))
        user = cur.fetchone()
        conn.close()

        if user is None:
            flash("Invalid username or password", "error")
            return render_template("index.html")

        if check_password_hash(user["password"], password):
            session.clear()
            session["user"] = user["username"]
            session["role"] = user["role"]
            session["user_id"] = user["id"]
            return redirect(url_for("dashboard"))
        else:
            flash("Invalid username or password", "error")

    except Exception as e:
        print("Login error:", e)
        flash("Login error. Please try again.", "error")

    return render_template("index.html")


# ==========================
# LOGOUT
# ==========================

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# ==========================
# STOCK INPUT
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

    symbol = request.form.get("symbol", "").strip().upper()
    days = int(request.form.get("days", 7))
    model_type = request.form.get("model", "linear")

    if not symbol:
        flash("Please enter a stock symbol")
        return redirect(url_for("stock_input"))

    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="2y")
        if hist is None or hist.empty:
            flash(f"No data found for '{symbol}'. Use .NS for NSE e.g. INFY.NS")
            return redirect(url_for("stock_input"))
        
        os.makedirs("data/uploads", exist_ok=True)
        filepath = f"data/uploads/{symbol}_data.csv"
        
        # Save only the Close prices as a single column
        close_series = hist['Close'].copy()
        close_df = pd.DataFrame({symbol: close_series})
        close_df.to_csv(filepath)
        
        print(f"✅ Data saved for {symbol}: {len(close_df)} rows")
        
    except Exception as e:
        flash(f"Failed to fetch data for '{symbol}': {e}")
        return redirect(url_for("stock_input"))

    try:
        # Load and clean data
        df = pd.read_csv(filepath)
        
        # Set date as index if present
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
        
        # Ensure numeric data
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.dropna()
        
        if df.empty:
            flash(f"No valid data found for '{symbol}'. Please try another symbol.")
            return redirect(url_for("stock_input"))
        
        columns = df.columns.tolist()
        results = {}
        predictions = {}
        analytics = {}
        lstm_preds = {}
        history_file = "data/history.csv"
        os.makedirs("data", exist_ok=True)

        if not os.path.exists(history_file):
            with open(history_file, "w", newline="") as f:
                csv.writer(f).writerow(["username", "stock", "action", "confidence", "date"])

        for col in columns:
            series = df[col].dropna().tolist()
            
            if len(series) < 10:
                flash(f"Not enough data for {col}. Need at least 10 data points.")
                return redirect(url_for("stock_input"))
            
            output = predict_series(series)
            predictions[col] = output["predictions"]
            if output.get("lstm_preds"):
                lstm_preds[col] = output["lstm_preds"]

            action = ("BUY" if output["future_avg"] > output["last_actual"] else
                      "SELL" if output["future_avg"] < output["last_actual"] else "HOLD")
            confidence = round(output["r2"] * 100, 2)
            add_prediction_notification(session.get("user"), col, action, confidence)
            risk = ("Low Risk" if confidence >= 80 else
                    "Medium Risk" if confidence >= 60 else "High Risk")

            results[col] = {"last": output["last_actual"], "avg": output["future_avg"],
                            "action": action, "reason": f"Trend is {output['trend_percent']}%",
                            "confidence": confidence, "risk": risk}
            analytics[col] = output

            with open(history_file, "a", newline="") as f:
                csv.writer(f).writerow([session.get("user"), col, action, confidence,
                                        datetime.datetime.now().strftime("%d-%m-%Y %H:%M")])
            
            try:
                from backend.utils.db_handler import save_prediction
                save_prediction(username=session.get("user", ""), stock=col, action=action,
                                confidence=confidence, last_price=output.get("last_actual", 0),
                                predicted_price=output.get("future_avg", 0), risk=risk)
            except Exception as _e:
                print("save_prediction error:", _e)
            
            try:
                rag_chatbot.db.add_prediction_record(
                    username=session.get("user"),
                    stock=col,
                    predicted_action=action,
                    actual_action="pending",
                    predicted_price=output.get("future_avg", 0),
                    actual_price=0,
                    confidence=confidence,
                    was_correct=False
                )
            except Exception as rag_e:
                print(f"RAG record error: {rag_e}")

        session.update({"file_path": filepath, "columns": columns, "results": results,
                        "predictions": predictions, "analytics": analytics,
                        "lstm_preds": lstm_preds, "symbol": symbol, "days": days, "model": model_type})

        return render_template("predictions.html", results=results, predictions=predictions,
                               analytics=analytics, lstm_preds=lstm_preds)

    except Exception as e:
        import traceback
        traceback.print_exc()
        flash(f"Prediction failed: {str(e)}")
        return redirect(url_for("stock_input"))

# ==========================
# PREVIEW
# ==========================

@app.route("/preview")
def preview():
    if "user" not in session:
        return redirect(url_for("login"))
    if "file_path" not in session:
        return redirect(url_for("upload_file"))
    df = load_csv(session["file_path"], MAX_ROWS)
    df = clean_data(df)
    table_html = preview_table(df)
    columns = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
    return render_template("preview.html", table=table_html, columns=columns)


# ==========================
# GRAPH SETTINGS
# ==========================

@app.route("/graph-settings")
def graph_settings():
    if "user" not in session:
        return redirect(url_for("login"))
    if "file_path" not in session:
        return redirect(url_for("upload_file"))
    df = load_csv(session["file_path"])
    columns = df.columns.tolist()
    return render_template("graph_settings.html", columns=columns)


# ==========================
# COLUMN SELECTION
# ==========================

@app.route("/select", methods=["POST"])
def select_columns():
    if "user" not in session:
        return redirect(url_for("login"))
    
    selected_columns = request.form.getlist("columns")
    if not selected_columns:
        flash("Please select at least one column")
        return redirect(url_for("preview"))
    
    # Store for basic charts (keep for compatibility)
    session["columns"] = selected_columns
    
    # Store for advanced analysis
    session["advanced_columns"] = selected_columns
    session["advanced_selected_column"] = selected_columns[0] if selected_columns else None
    session["advanced_file_path"] = session.get("file_path")
    
    # Get file info for display
    file_path = session.get("file_path")
    if file_path and os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path)
            session["dataset_rows"] = len(df)
            session["dataset_cols"] = len(df.columns)
        except:
            pass
    
    # DIRECT REDIRECT TO ADVANCED ANALYSIS
    return redirect("/advanced/analysis")

@app.route("/analysis-choice")
def analysis_choice():
    """Page to choose between Basic Charts and Advanced Analysis"""
    if "user" not in session:
        return redirect(url_for("login"))
    
    selected_column = session.get("advanced_selected_column", "Not selected")
    rows = session.get("dataset_rows", 0)
    cols = session.get("dataset_cols", 0)
    username = session.get("user", "User")
    
    return render_template("analysis_choice.html", 
                         selected_column=selected_column,
                         rows=rows,
                         cols=cols,
                         username=username)

# ==========================
# CHARTS
# ==========================

@app.route("/charts", methods=["GET", "POST"])
def charts():
    if "user" not in session:
        return redirect(url_for("login"))
    if "file_path" not in session or "columns" not in session:
        return redirect(url_for("upload_file"))
    df = load_csv(session["file_path"])
    columns = session.get("columns")
    chart_type = session.get("chart_type", "line")
    if request.method == "POST":
        chart_type = request.form.get("chart_type", "line")
        session["chart_type"] = chart_type
    data = {col: df[col].astype(float).tolist() for col in columns}
    return render_template("graphs.html", data=data, columns=columns, chart_type=chart_type)


# ==========================
# CHATBOT API
# ==========================

@app.route("/api/chatbot", methods=["POST"])
def chatbot():
    if "user" not in session:
        return jsonify({"error": "Not logged in"}), 401
    try:
        body = request.get_json()
        message = body.get("message", "").lower().strip()
        results = session.get("results", {})
        analytics = session.get("analytics", {})
        preds = session.get("predictions", {})
        username = session.get("user", "Investor")

        def advisor_score(col):
            res = results.get(col, {})
            an = analytics.get(col, {})
            pr = preds.get(col, [])
            score = 50
            reasons = []
            warnings = []
            action = res.get("action", "HOLD")
            conf = float(res.get("confidence", 50))
            risk = res.get("risk", "Medium Risk")
            rsi = float(an.get("rsi", 50))
            trend = float(an.get("trend_percent", 0))
            vol = float(an.get("volatility", 0))
            ma7 = float(an.get("ma7", 0))
            ma14 = float(an.get("ma14", 0))
            if action == "BUY":
                score += 20
                reasons.append("📈 Model signals **BUY** — upward trend detected")
            elif action == "SELL":
                score -= 20
                warnings.append("📉 Model signals **SELL** — downward trend detected")
            else:
                reasons.append("◆ Model signals **HOLD** — sideways movement")
            if rsi < 25:
                score += 20
                reasons.append(f"💡 RSI = {rsi:.1f} → **Strongly Oversold** — BUY zone")
            elif rsi < 35:
                score += 12
                reasons.append(f"📊 RSI = {rsi:.1f} → **Oversold** — potential BUY")
            elif rsi > 75:
                score -= 20
                warnings.append(f"🔥 RSI = {rsi:.1f} → **Strongly Overbought** — consider SELL")
            elif rsi > 65:
                score -= 10
                warnings.append(f"⚠️ RSI = {rsi:.1f} → **Approaching Overbought**")
            else:
                reasons.append(f"⚖️ RSI = {rsi:.1f} → Neutral zone")
            if conf >= 85:
                score += 15
                reasons.append(f"✅ Very High Confidence ({conf}%)")
            elif conf >= 70:
                score += 8
                reasons.append(f"✅ Good Confidence ({conf}%)")
            elif conf >= 55:
                score -= 5
                warnings.append(f"⚠️ Moderate Confidence ({conf}%)")
            else:
                score -= 15
                warnings.append(f"❌ Low Confidence ({conf}%) — use caution")
            if trend > 5:
                score += 10
                reasons.append(f"📈 Strong upward trend ({trend}%)")
            elif trend > 1:
                score += 5
                reasons.append(f"📊 Mild upward trend ({trend}%)")
            elif trend < -5:
                score -= 10
                warnings.append(f"📉 Strong downward trend ({trend}%)")
            elif trend < -1:
                score -= 5
                warnings.append(f"📉 Mild downward trend ({trend}%)")
            if ma7 > 0 and ma14 > 0:
                if ma7 > ma14:
                    score += 8
                    reasons.append(f"🟢 MA7 ({ma7:.2f}) > MA14 ({ma14:.2f}) → **Bullish crossover**")
                else:
                    score -= 8
                    warnings.append(f"🔴 MA7 ({ma7:.2f}) < MA14 ({ma14:.2f}) → **Bearish crossover**")
            if vol > 5:
                score -= 8
                warnings.append(f"⚡ High Volatility ({vol}%)")
            elif vol < 1.5:
                score += 5
                reasons.append(f"😌 Low Volatility ({vol}%)")
            if "Low" in risk:
                score += 5
                reasons.append("🛡️ Low Risk classification")
            elif "High" in risk:
                score -= 10
                warnings.append("🚨 High Risk classification")
            if len(pr) >= 2:
                if pr[-1] > pr[0]:
                    pct = ((pr[-1] - pr[0]) / pr[0] * 100) if pr[0] != 0 else 0
                    reasons.append(f"📅 7-day forecast: {pct:.1f}% upside")
                else:
                    pct = ((pr[0] - pr[-1]) / pr[0] * 100) if pr[0] != 0 else 0
                    warnings.append(f"📅 7-day forecast: {pct:.1f}% downside risk")
            return max(0, min(100, score)), reasons, warnings

        def verdict(score):
            if score >= 80:
                return "🟢 STRONG BUY", "#22c55e"
            if score >= 65:
                return "🟢 BUY", "#4ade80"
            if score >= 50:
                return "🟡 WEAK BUY", "#fbbf24"
            if score >= 40:
                return "🟡 HOLD", "#fde047"
            if score >= 25:
                return "🔴 WEAK SELL", "#fb7185"
            return "🔴 STRONG SELL", "#ef4444"

        def full_advice(col):
            score, reasons, warnings = advisor_score(col)
            verd, _ = verdict(score)
            res = results.get(col, {})
            lines = [f"## 📊 Stock Analysis: **{col}**", "", f"**Advisor Score: {score}/100 → {verd}**", ""]
            if reasons:
                lines += ["**✅ Positive Signals:**"] + [f"  {r}" for r in reasons] + [""]
            if warnings:
                lines += ["**⚠️ Risk Factors:**"] + [f"  {w}" for w in warnings] + [""]
            lines.append(f"**📋 Summary:** Confidence {res.get('confidence', '?')}% | {res.get('risk', '?')}")
            lines.append("")
            if score >= 65:
                lines.append("**💼 Recommendation:** Indicators align positively. Consider entering with proper stop-loss.")
            elif score >= 45:
                lines.append("**💼 Recommendation:** Mixed signals. Wait for a clearer trend.")
            else:
                lines.append("**💼 Recommendation:** Multiple risk factors. Consider reducing exposure.")
            lines += ["", "_⚠️ AI-generated analysis only. Not financial advice._"]
            return "\n".join(lines)

        def best_stock():
            if not results:
                return "No prediction data. Please run a prediction first."
            scored = {col: advisor_score(col)[0] for col in results}
            best = max(scored, key=scored.get)
            score = scored[best]
            verd, _ = verdict(score)
            _, reasons, _ = advisor_score(best)
            return (f"🏆 **Best Stock: {best}**\n\n**Score: {score}/100 → {verd}**\n\n"
                    f"**Top reason:** {reasons[0] if reasons else 'Strong overall signal'}\n\nType **'advise {best}'** for full analysis.")

        def worst_stock():
            if not results:
                return "No prediction data available."
            scored = {col: advisor_score(col)[0] for col in results}
            worst = min(scored, key=scored.get)
            score = scored[worst]
            verd, _ = verdict(score)
            _, _, warnings = advisor_score(worst)
            return (f"⚠️ **Weakest Stock: {worst}**\n\n**Score: {score}/100 → {verd}**\n\n"
                    f"**Main concern:** {warnings[0] if warnings else 'Weak overall signal'}\n\nType **'advise {worst}'** for full analysis.")

        def compare_all():
            if not results:
                return "No prediction data. Run a prediction first."
            scored = {col: advisor_score(col)[0] for col in results}
            sorted_stocks = sorted(scored.items(), key=lambda x: x[1], reverse=True)
            lines = ["## 📊 Stock Ranking (Best → Worst)\n"]
            for rank, (col, score) in enumerate(sorted_stocks, 1):
                verd, _ = verdict(score)
                res = results.get(col, {})
                rsi = float(analytics.get(col, {}).get("rsi", 50))
                lines.append(f"**#{rank} {col}** — Score: {score}/100 → {verd}\n"
                             f"   Action: {res.get('action', '?')} | Confidence: {res.get('confidence', '?')}% | RSI: {rsi:.1f} | Risk: {res.get('risk', '?')}\n")
            lines.append("\n_Type **'advise [name]'** for detailed analysis._")
            return "\n".join(lines)

        def should_buy_today():
            if not results:
                return "No prediction data available."
            scored = {col: advisor_score(col)[0] for col in results}
            buys = [(col, s) for col, s in scored.items() if s >= 60]
            sells = [(col, s) for col, s in scored.items() if s < 40]
            holds = [(col, s) for col, s in scored.items() if 40 <= s < 60]
            lines = ["## 📅 Today's Advisor Decision\n"]
            if buys:
                lines.append("**✅ CONSIDER BUYING:**")
                for col, s in sorted(buys, key=lambda x: x[1], reverse=True):
                    verd, _ = verdict(s)
                    lines.append(f"  • **{col}** (Score: {s}/100) → {verd}")
                lines.append("")
            if holds:
                lines.append("**◆ HOLD / WATCH:**")
                for col, s in holds:
                    lines.append(f"  • **{col}** (Score: {s}/100) → Mixed signals")
                lines.append("")
            if sells:
                lines.append("**❌ AVOID / CONSIDER SELLING:**")
                for col, s in sorted(sells, key=lambda x: x[1]):
                    verd, _ = verdict(s)
                    lines.append(f"  • **{col}** (Score: {s}/100) → {verd}")
                lines.append("")
            if not buys and not sells:
                lines.append("All stocks in HOLD territory — no strong signals.")
            lines.append("_⚠️ Educational analysis only. Not financial advice._")
            return "\n".join(lines)

        def portfolio_risk():
            if not results:
                return "No prediction data available."
            scores = [advisor_score(col)[0] for col in results]
            avg_s = sum(scores) / len(scores)
            high_r = sum(1 for col in results if "High" in results[col].get("risk", ""))
            med_r = sum(1 for col in results if "Medium" in results[col].get("risk", ""))
            low_r = sum(1 for col in results if "Low" in results[col].get("risk", ""))
            sells = sum(1 for col in results if results[col].get("action") == "SELL")
            avg_rsi = sum(float(analytics.get(col, {}).get("rsi", 50)) for col in analytics) / max(len(analytics), 1)
            verdict_str = "🟢 HEALTHY" if avg_s >= 65 else "🟡 MODERATE" if avg_s >= 45 else "🔴 RISKY"
            return (f"## 🗂️ Portfolio Risk Assessment\n\n**Overall Score: {avg_s:.0f}/100**\n"
                    f"**Verdict: {verdict_str}**\n\n"
                    f"**Risk Breakdown:**\n  • 🟢 Low Risk: {low_r}\n  • 🟡 Medium Risk: {med_r}\n  • 🔴 High Risk: {high_r}\n\n"
                    f"**Signals:**\n  • SELL signals: {sells}/{len(results)}\n"
                    f"  • Average RSI: {avg_rsi:.1f} ({'Overbought' if avg_rsi > 70 else 'Oversold' if avg_rsi < 30 else 'Neutral'})\n\n"
                    f"_Type **'compare all'** to see rankings._")

        def highest_confidence():
            if not results:
                return "No prediction data available."
            best = max(results, key=lambda col: float(results[col].get("confidence", 0)))
            r = results[best]
            return (f"🎯 **Highest Confidence: {best}**\n\n• Confidence: **{r.get('confidence', '?')}%**\n"
                    f"• Action: **{r.get('action', '?')}**\n• Risk: **{r.get('risk', '?')}**\n\nType **'advise {best}'** for full analysis.")

        def simple_forecast(col):
            pr = preds.get(col, [])
            res = results.get(col, {})
            if not pr:
                return f"No forecast data for {col}."
            start = pr[0]
            end = pr[-1]
            chg = ((end - start) / start * 100) if start != 0 else 0
            direction = "rise" if chg > 0 else "fall"
            emoji = "📈" if chg > 0 else "📉"
            return (f"{emoji} **Simple Forecast for {col}:**\n\n"
                    f"Next 7 days: price will **{direction} by {abs(chg):.1f}%**\n\n"
                    f"• Day 1: **{start:.2f}**  →  Day 7: **{end:.2f}**\n"
                    f"• Signal: **{res.get('action', '?')}** | Confidence: **{res.get('confidence', '?')}%**\n\n"
                    f"{'✅ Upward momentum.' if chg > 0 else '⚠️ Downward momentum.'}")

        advise_match = None
        for col in results:
            if col.lower() in message:
                advise_match = col
                break

        if any(w in message for w in ["advise", "advice", "analyse", "analyze", "tell me about", "what about", "review"]) and advise_match:
            reply = full_advice(advise_match)
        elif any(w in message for w in ["advise", "advice", "analyse", "analyze", "full analysis"]) and not advise_match:
            if len(results) == 1:
                reply = full_advice(list(results.keys())[0])
            else:
                reply = "Which stock?\n\n" + "\n".join([f"• {c}" for c in results]) + "\n\nType **'advise [name]'**"
        elif any(w in message for w in ["best stock", "best to buy", "top stock", "which stock", "recommend", "should i invest", "invest in"]):
            reply = best_stock()
        elif any(w in message for w in ["worst", "avoid", "weakest", "bad stock"]):
            reply = worst_stock()
        elif any(w in message for w in ["compare", "compare all", "rank", "ranking", "versus", "vs"]):
            reply = compare_all()
        elif any(w in message for w in ["should i buy", "buy today", "sell today", "buy or sell", "what to do today", "today"]):
            reply = should_buy_today()
        elif any(w in message for w in ["portfolio risk", "overall risk", "total risk", "portfolio"]):
            reply = portfolio_risk()
        elif any(w in message for w in ["highest confidence", "most confident", "most reliable", "most accurate"]):
            reply = highest_confidence()
        elif any(w in message for w in ["simple forecast", "explain forecast", "simple prediction", "in simple", "plain english", "easy"]):
            if advise_match:
                reply = simple_forecast(advise_match)
            elif len(results) == 1:
                reply = simple_forecast(list(results.keys())[0])
            else:
                reply = "\n\n---\n\n".join([simple_forecast(col) for col in results])
        elif any(w in message for w in ["buy signal", "buy"]):
            buys = {col: r for col, r in results.items() if r.get("action") == "BUY"}
            if buys:
                lines = [f"• **{col}** → {verdict(advisor_score(col)[0])[0]} | Confidence: {r.get('confidence')}%" for col, r in buys.items()]
                reply = "📈 **BUY Signals:**\n\n" + "\n".join(lines)
            else:
                reply = "No BUY signals in current session."
        elif any(w in message for w in ["sell signal", "sell"]):
            sells = {col: r for col, r in results.items() if r.get("action") == "SELL"}
            if sells:
                lines = [f"• **{col}** → {verdict(advisor_score(col)[0])[0]} | Confidence: {r.get('confidence')}%" for col, r in sells.items()]
                reply = "📉 **SELL Signals:**\n\n" + "\n".join(lines)
            else:
                reply = "No SELL signals in current session."
        elif any(w in message for w in ["rsi", "relative strength", "overbought", "oversold"]):
            if not analytics:
                reply = "No RSI data. Run a prediction first."
            else:
                lines = []
                for col, a in analytics.items():
                    rsi = float(a.get("rsi", 50))
                    note = "→ ⚠️ OVERSOLD — BUY zone" if rsi < 30 else "→ 🔥 OVERBOUGHT — caution" if rsi > 70 else "→ ⚖️ Neutral"
                    lines.append(f"• **{col}**: RSI = **{rsi:.1f}** {note}")
                reply = "📊 **RSI Analysis:**\n\n" + "\n".join(lines)
        elif any(w in message for w in ["forecast", "7 day", "next week", "future price", "prediction"]):
            parts = [simple_forecast(col) for col in preds]
            reply = "\n\n---\n\n".join(parts) if parts else "No forecast data available."
        elif any(w in message for w in ["risk level", "risk", "risky", "safe"]):
            reply = portfolio_risk() if len(results) > 1 else "\n".join([f"• **{col}**: {r.get('risk', '?')} | {r.get('confidence', '?')}%" for col, r in results.items()])
        elif any(w in message for w in ["confidence", "accurate", "reliable"]):
            if not results:
                reply = "No confidence data available."
            else:
                lines = []
                for col, r in results.items():
                    conf = float(r.get("confidence", 0))
                    bar = "█" * int(conf // 10) + "░" * (10 - int(conf // 10))
                    note = "Very reliable" if conf >= 85 else "Reliable" if conf >= 70 else "Moderate" if conf >= 55 else "Low"
                    lines.append(f"• **{col}**: {conf}% [{bar}] → {note}")
                reply = "🎯 **Confidence Scores:**\n\n" + "\n".join(lines)
        elif any(w in message for w in ["model", "best model", "r2", "mae", "rmse", "performance"]):
            if not analytics:
                reply = "No model data available."
            else:
                lines = [f"• **{col}**: Best={a.get('best_model', '?')} | R²={a.get('r2', '?')} | MAE={a.get('mae', '?')} | RMSE={a.get('rmse', '?')}" for col, a in analytics.items()]
                reply = "🏆 **Model Performance:**\n\n" + "\n".join(lines)
        elif any(w in message for w in ["summary", "overview", "all stocks", "show all", "report"]):
            reply = compare_all()
        elif any(w in message for w in ["hello", "hi", "hey", "start", "begin"]):
            stock_list = ", ".join(results.keys()) if results else "none loaded yet"
            reply = (f"👋 Hello **{username.title()}**!\n\n📊 **Stocks loaded:** {stock_list}\n\n"
                     "Try: **'Should I buy today?'** | **'Best stock'** | **'Compare all'** | **'Help'**")
        elif any(w in message for w in ["help", "commands", "what can you"]):
            reply = ("🤖 **Commands:**\n\n💼 Should I buy today? | Best stock | Worst stock | Advise [name]\n\n"
                     "📊 Compare all | Portfolio risk | Highest confidence | RSI | 7-day forecast\n\n"
                     "📈 BUY signals | SELL signals | Model performance | Confidence scores")
        elif any(w in message for w in ["thank", "thanks", "great", "awesome", "good job"]):
            reply = "😊 Happy to help! Type **'help'** anytime."
        else:
            if results:
                if len(results) == 1:
                    col = list(results.keys())[0]
                    score, reasons, _ = advisor_score(col)
                    verd, _ = verdict(score)
                    reply = (f"🤔 Quick take on **{col}**: **{score}/100 → {verd}**\n\n"
                             f"Top signal: {reasons[0] if reasons else 'No clear signal'}\n\nType **'advise {col}'** or **'help'**")
                else:
                    reply = "🤔 Not sure what you meant. Try **'help'** for all commands."
            else:
                reply = "No prediction data loaded. Please upload a CSV or fetch a stock first!"

        return jsonify({"reply": reply})

    except Exception as e:
        print("Chatbot error:", e)
        return jsonify({"reply": f"⚠️ Error: {str(e)}"}), 500


@app.route("/test-chatbot")
def test_chatbot():
    return "✅ Stock AI Advisor is ready — no API key needed!"


# ==========================
# RAG CHATBOT API ENDPOINTS
# ==========================

@app.route("/api/rag-chatbot", methods=["POST"])
def rag_chatbot_api():
    """Enhanced RAG-powered chatbot with user trust tracking"""
    if "user" not in session:
        return jsonify({"error": "Not logged in"}), 401
    
    try:
        data = request.get_json()
        message = data.get("message", "").lower().strip()
        username = session.get("user", "User")
        
        results = session.get("results", {})
        analytics = session.get("analytics", {})
        preds = session.get("predictions", {})
        
        stock_symbol = None
        symbols = ["AAPL", "TSLA", "NVDA", "MSFT", "GOOGL", "AMZN", "META", 
                   "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS"]
        for sym in symbols:
            if sym.lower() in message or sym.split('.')[0].lower() in message:
                stock_symbol = sym
                break
        
        market_data = None
        if stock_symbol:
            try:
                ticker = yf.Ticker(stock_symbol)
                info = ticker.info
                market_data = {
                    "price": info.get("regularMarketPrice", 0),
                    "change_pct": info.get("regularMarketChangePercent", 0)
                }
            except:
                pass
        
        response = rag_chatbot.generate_rag_response(
            username=username,
            query=message,
            stock_symbol=stock_symbol,
            market_data=market_data,
            results=results,
            analytics=analytics,
            preds=preds
        )
        
        return jsonify({"reply": response, "rag_enabled": True})
        
    except Exception as e:
        print("RAG Chatbot error:", e)
        return jsonify({"reply": f"⚠️ Error: {str(e)}"}), 500


@app.route("/api/trust-score", methods=["GET"])
def get_trust_score():
    """Get user's trust score and performance metrics"""
    if "user" not in session:
        return jsonify({"error": "Not logged in"}), 401
    
    username = session.get("user")
    user_context = rag_chatbot.db.get_user_context(username)
    
    if user_context["trust_score"] >= 80:
        level = "Expert Trader"
    elif user_context["trust_score"] >= 60:
        level = "Experienced Trader"
    elif user_context["trust_score"] >= 40:
        level = "Learning Trader"
    else:
        level = "New Trader"
    
    return jsonify({
        "trust_score": user_context["trust_score"],
        "accuracy": user_context["accuracy"],
        "risk_profile": user_context["risk_profile"],
        "level": level,
        "total_predictions": user_context.get("total_predictions", 0),
        "correct_predictions": user_context.get("correct_predictions", 0),
        "message": f"You are a {level}. Keep making predictions to improve your score!"
    })


@app.route("/api/record-prediction-outcome", methods=["POST"])
def record_prediction_outcome():
    """Record prediction outcome for RAG learning"""
    if "user" not in session:
        return jsonify({"error": "Not logged in"}), 401
    
    data = request.get_json()
    username = session.get("user")
    
    rag_chatbot.db.add_prediction_record(
        username=username,
        stock=data.get("stock"),
        predicted_action=data.get("predicted_action"),
        actual_action=data.get("actual_action"),
        predicted_price=data.get("predicted_price", 0),
        actual_price=data.get("actual_price", 0),
        confidence=data.get("confidence", 50),
        was_correct=data.get("was_correct", False)
    )
    
    return jsonify({"success": True, "message": "Prediction recorded for future reference"})


@app.route("/api/rag-feedback", methods=["POST"])
def rag_feedback():
    """Submit feedback on RAG responses"""
    if "user" not in session:
        return jsonify({"error": "Not logged in"}), 401
    
    data = request.get_json()
    username = session.get("user")
    
    rag_chatbot.db.add_feedback(
        username=username,
        query=data.get("query"),
        response=data.get("response"),
        rating=data.get("rating", 0),
        helpful=data.get("helpful", True)
    )
    
    return jsonify({"success": True, "message": "Thank you for your feedback!"})


# ==========================
# PREDICT
# ==========================

@app.route("/predict")
def predict():
    if "user" not in session:
        return redirect(url_for("login"))
    if "file_path" not in session or "columns" not in session:
        return redirect(url_for("upload_file"))

    df = load_csv(session["file_path"])
    df = clean_data(df)
    columns = session["columns"]
    results = {}
    predictions = {}
    analytics = {}
    lstm_preds = {}
    history_file = "data/history.csv"
    os.makedirs("data", exist_ok=True)

    if not os.path.exists(history_file):
        with open(history_file, "w", newline="") as f:
            csv.writer(f).writerow(["username", "stock", "action", "confidence", "date"])

    for col in columns:
        series = df[col].astype(float).tolist()
        output = predict_series(series)
        predictions[col] = output["predictions"]
        if output.get("lstm_preds"):
            lstm_preds[col] = output["lstm_preds"]
        action = ("BUY" if output["future_avg"] > output["last_actual"] else
                  "SELL" if output["future_avg"] < output["last_actual"] else "HOLD")
        confidence = round(output["r2"] * 100, 2)
        risk = "Low Risk" if confidence >= 80 else "Medium Risk" if confidence >= 60 else "High Risk"
        results[col] = {"last": output["last_actual"], "avg": output["future_avg"], "action": action,
                        "reason": f"Trend is {output['trend_percent']}%", "confidence": confidence, "risk": risk}
        analytics[col] = output
        with open(history_file, "a", newline="") as f:
            csv.writer(f).writerow([session.get("user"), col, action, confidence,
                                    datetime.datetime.now().strftime("%d-%m-%Y %H:%M")])
        try:
            from backend.utils.db_handler import save_prediction
            save_prediction(username=session.get("user", ""), stock=col, action=action, confidence=confidence,
                            last_price=output.get("last_actual", 0), predicted_price=output.get("future_avg", 0), risk=risk)
        except Exception as _e:
            print("save_prediction error:", _e)
    add_prediction_notification(session.get("user"), col, action, confidence)
    session.update({"results": results, "predictions": predictions, "analytics": analytics, "lstm_preds": lstm_preds})
    return render_template("predictions.html", results=results, predictions=predictions, analytics=analytics, lstm_preds=lstm_preds)


# ==========================
# PREDICTIONS PAGE
# ==========================

@app.route("/predictions")
def show_predictions():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("predictions.html",
                           predictions=session.get("predictions"),
                           results=session.get("results"),
                           analytics=session.get("analytics", {}),
                           lstm_preds=session.get("lstm_preds", {}))


# ==========================
# DASHBOARD
# ==========================

@app.route("/dashboard")
def dashboard():
    if "user" not in session:
        return redirect(url_for("login"))
    last7 = []
    chart_labels = []
    chart_data = []
    try:
        stock = yf.Ticker("AAPL")
        data = stock.history(period="7d")
        if data is not None and not data.empty:
            for date, row in data.iterrows():
                d = date.strftime("%b %d")
                price = round(float(row["Close"]), 2)
                last7.append((d, price))
                chart_labels.append(d)
                chart_data.append(price)
    except Exception as e:
        print(f"Dashboard AAPL fetch error: {e}")
        import datetime as _dt
        for i in range(7):
            day = (_dt.date.today() - _dt.timedelta(days=6 - i)).strftime("%b %d")
            chart_labels.append(day)
            chart_data.append(0)
            last7.append((day, 0))
    return render_template("dashboard.html", live_price=chart_data[-1] if chart_data else "N/A",
                           last7=last7, chart_labels=chart_labels, chart_data=chart_data)


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
    activities = []
    if os.path.exists(history_file):
        df = pd.read_csv(history_file)
        user_data = df[df["username"] == username]
        total_predictions = len(user_data)
        if not user_data.empty:
            last_activity = user_data.iloc[-1]["date"]
            for _, row in user_data.tail(5).iterrows():
                activities.append({"action": row["action"], "time": row["date"]})
    return render_template("profile.html", username=username, total=total_predictions,
                           last=last_activity, activities=activities)


@app.route("/update-profile", methods=["POST"])
def update_profile():
    if "user" not in session:
        return redirect(url_for("login"))
    session["profile_name"] = request.form.get("name")
    session["profile_email"] = request.form.get("email")
    flash("Profile Updated Successfully", "success")
    return redirect(url_for("profile"))


@app.route("/upload-avatar", methods=["POST"])
def upload_avatar():
    if "user" not in session:
        return redirect(url_for("login"))
    if "avatar" not in request.files:
        flash("No file selected", "error")
        return redirect(url_for("profile"))
    file = request.files["avatar"]
    if file.filename == "":
        flash("No file selected", "error")
        return redirect(url_for("profile"))
    filename = secure_filename(file.filename)
    save_path = os.path.join("static/uploads", filename)
    file.save(save_path)
    session["avatar"] = "/" + save_path.replace("\\", "/")
    flash("Profile photo updated", "success")
    return redirect(url_for("profile"))


@app.route("/edit-profile", methods=["GET", "POST"])
def edit_profile():
    if "user" not in session:
        return redirect(url_for("login"))
    if request.method == "POST":
        session["user"] = request.form.get("name")
        session["email"] = request.form.get("email")
        if "photo" in request.files:
            file = request.files["photo"]
            if file.filename != "":
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config["PROFILE_UPLOAD_FOLDER"], filename)
                file.save(filepath)
                session["profile_photo"] = filename
        flash("Profile updated successfully!", "success")
        return redirect(url_for("profile"))
    return render_template("edit_profile.html", username=session.get("user"),
                           email=session.get("email"), photo=session.get("profile_photo"))


# ==========================
# UPLOAD
# ==========================

@app.route("/upload", methods=["GET", "POST"])
def upload_file():
    if "user" not in session:
        return redirect(url_for("login"))
    if request.method == "POST":
        if "file" not in request.files:
            flash("No file selected", "error")
            return redirect(request.url)
        file = request.files["file"]
        if file.filename == "":
            flash("No file selected", "error")
            return redirect(request.url)
        if not allowed_file(file.filename):
            flash("Only CSV files allowed", "error")
            return redirect(request.url)
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)
        session["file_path"] = filepath
        return redirect(url_for("preview"))
    return render_template("upload.html")


# ==========================
# LIVE STOCK
# ==========================

@app.route("/live", methods=["GET", "POST"])
def live_stock():
    if "user" not in session:
        return redirect(url_for("login"))
    symbol = request.args.get("symbol", "AAPL").upper()
    if request.method == "POST":
        symbol = request.form.get("symbol", "AAPL").upper()
    return render_template("live_stock.html", symbol=symbol)


@app.route("/live-data/<symbol>")
def live_data(symbol):
    try:
        period = request.args.get("period", "1d")
        interval = request.args.get("interval", "5m")
        valid = {"1d": ["1m", "2m", "5m", "15m", "30m"], "5d": ["5m", "15m", "30m", "1h"],
                 "1mo": ["1h", "1d"], "3mo": ["1d"], "6mo": ["1d"], "1y": ["1d"], "2y": ["1wk"]}
        if period not in valid:
            period = "1d"
        if interval not in valid[period]:
            interval = valid[period][0]
        stock = yf.Ticker(symbol.upper())
        data = stock.history(period=period, interval=interval)
        if data is None or data.empty:
            return jsonify({"error": "No data", "candles": [], "price": "N/A"})
        data.dropna(inplace=True)
        candles = []
        volumes = []
        for ts, row in data.iterrows():
            t = int(ts.timestamp())
            candles.append({"time": t, "open": round(float(row["Open"]), 2), "high": round(float(row["High"]), 2),
                            "low": round(float(row["Low"]), 2), "close": round(float(row["Close"]), 2)})
            volumes.append({"time": t, "value": int(row["Volume"]),
                            "color": "rgba(34,197,94,.5)" if row["Close"] >= row["Open"] else "rgba(248,113,113,.5)"})
        closes = data["Close"]
        latest = float(closes.iloc[-1])
        prev = float(closes.iloc[0])
        ma20 = closes.rolling(20).mean().dropna()
        ma20_series = [{"time": int(ts.timestamp()), "value": round(float(v), 2)} for ts, v in ma20.items()]
        delta = closes.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, float("nan"))
        rsi_s = (100 - 100 / (1 + rs)).dropna()
        rsi_latest = round(float(rsi_s.iloc[-1]), 1) if not rsi_s.empty else 50.0
        change = round(latest - prev, 2)
        change_pct = round((change / prev) * 100, 2) if prev else 0
        currency = "₹" if symbol.upper().endswith((".NS", ".BO")) else "$"
        return jsonify({"price": round(latest, 2), "change": change, "change_pct": change_pct,
                        "currency": currency, "rsi": rsi_latest, "candles": candles, "volumes": volumes,
                        "ma20": ma20_series, "time": datetime.datetime.now().strftime("%H:%M:%S"),
                        "symbol": symbol.upper(), "period": period, "interval": interval})
    except Exception as e:
        print("Live API Error:", e)
        return jsonify({"error": str(e), "candles": [], "price": "N/A"})


# ==========================
# HISTORY
# ==========================

@app.route("/history")
def history():
    if "user" not in session:
        return redirect(url_for("login"))
    history_file = "data/history.csv"
    if not os.path.exists(history_file):
        return render_template("history.html", records=[])
    df = pd.read_csv(history_file)
    user_df = df[df["username"] == session["user"]]
    return render_template("history.html", records=user_df.to_dict(orient="records"))


# ==========================
# ADMIN PANEL
# ==========================

@app.route("/admin")
def admin_panel():
    if "user" not in session:
        return redirect(url_for("login"))
    if session.get("role") != "admin":
        return "Access Denied", 403
    users = []
    try:
        conn = sqlite3.connect("data/users.db")
        c = conn.cursor()
        c.execute("SELECT id, username, email, role FROM users ORDER BY id")
        users = [{"id": r[0], "username": r[1], "email": r[2] or "—", "role": r[3] or "user"} for r in c.fetchall()]
        conn.close()
    except Exception as e:
        print("Admin users error:", e)
    history = []
    if os.path.exists("data/history.csv"):
        try:
            df = pd.read_csv("data/history.csv")
            history = df.to_dict(orient="records")
        except Exception as e:
            print("Admin history error:", e)
    total_predictions = len(history)
    buy_count = sum(1 for h in history if str(h.get("action", "")).upper() == "BUY")
    sell_count = sum(1 for h in history if str(h.get("action", "")).upper() == "SELL")
    hold_count = sum(1 for h in history if str(h.get("action", "")).upper() == "HOLD")
    train_info = {}
    try:
        from backend.utils.auto_train import get_last_train_info
        train_info = get_last_train_info()
    except Exception:
        pass
    return render_template("admin.html", users=users, history=history, total_predictions=total_predictions,
                           buy_count=buy_count, sell_count=sell_count, hold_count=hold_count, train_info=train_info)


# ==========================
# STOCK ANALYSIS
# ==========================

@app.route("/stock-analysis", methods=["GET", "POST"])
def stock_analysis():
    if request.method == "POST":
        symbol = request.form["symbol"]
        month = int(request.form["month"])
        year = int(request.form["year"])
        return redirect(url_for("stock_result", symbol=symbol, month=month, year=year))
    return render_template("stock_analysis.html")


@app.route("/stock-result")
def stock_result():
    symbol = request.args.get("symbol")
    month = int(request.args.get("month"))
    year = int(request.args.get("year"))
    data = yf.download(symbol, period="5y")
    if data.empty:
        flash("Invalid symbol or no data found.")
        return redirect(url_for("stock_analysis"))
    filtered = data[(data.index.month == month) & (data.index.year == year)]
    if filtered.empty:
        flash("No data found for selected period.")
        return redirect(url_for("stock_analysis"))
    close_series = filtered["Close"]
    if isinstance(close_series, pd.DataFrame):
        close_series = close_series.iloc[:, 0]
    close_series = close_series.astype(float)
    overview = {"total_days": len(close_series), "highest": round(float(close_series.max()), 2),
                "lowest": round(float(close_series.min()), 2), "volatility": round(float(close_series.std()), 2),
                "return": round(float(((close_series.iloc[-1] - close_series.iloc[0]) / close_series.iloc[0]) * 100), 2)}
    preview_table_html = filtered.tail(10).to_html(classes="table table-striped", index=True)
    chart_labels = filtered.index.strftime("%d %b").tolist()
    chart_data = close_series.round(2).values.tolist()
    df_pred = pd.DataFrame({"Close": close_series.values, "Day": np.arange(len(close_series))})
    from sklearn.linear_model import LinearRegression
    lr_model = LinearRegression()
    lr_model.fit(df_pred[["Day"]], df_pred["Close"].values.ravel())
    prediction = float(lr_model.predict([[len(df_pred)]]).flatten()[0])
    last_price = float(close_series.iloc[-1])
    summary = {"last_price": round(last_price, 2), "mean_price": round(float(close_series.mean()), 2),
               "prediction": round(prediction, 2), "trend": "Uptrend" if prediction > last_price else "Downtrend"}
    return render_template("stock_result.html", symbol=symbol, month=month, year=year,
                           table=preview_table_html, chart_labels=chart_labels,
                           chart_data=chart_data, summary=summary, overview=overview)


# ==========================
# DOWNLOAD REPORT (PDF)
# ==========================

@app.route("/download_report")
def download_report():
    if "user" not in session:
        return redirect(url_for("login"))
    if "results" not in session:
        flash("⚠️ No report available yet. Run a prediction on a stock first, then download the PDF report.", "warning")
        return redirect(url_for("history"))

    from reportlab.pdfgen import canvas as rl_canvas
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.colors import HexColor, white, black

    results = session.get("results", {})
    analytics = session.get("analytics", {})
    preds_raw = session.get("predictions", {})
    lstm_preds = session.get("lstm_preds", {})
    username = session.get("user", "N/A")
    filename = "StockAI_Report.pdf"
    filepath = os.path.join("data", filename)
    os.makedirs("data", exist_ok=True)

    NAVY = HexColor("#003366")
    BLUE = HexColor("#0055a5")
    LIGHTBLUE = HexColor("#e8f0f8")
    GOLD = HexColor("#b8860b")
    BUY_C = HexColor("#006633")
    SELL_C = HexColor("#990000")
    HOLD_C = HexColor("#8b6914")
    BLACK = HexColor("#1a1a1a")
    GREY = HexColor("#555555")
    LGREY = HexColor("#888888")
    ALTROW = HexColor("#f2f6fc")
    BORDER = HexColor("#ccd9e8")
    PAGE_BG = HexColor("#ffffff")
    PW, PH = letter
    today = datetime.datetime.now().strftime("%B %d, %Y  %H:%M")
    total_pages = 1 + len(results)
    total = len(results)
    confs = [float(r.get("confidence", 0)) for r in results.values()]
    avg_c = round(sum(confs) / len(confs), 1) if confs else 0
    buy_c = sum(1 for r in results.values() if r.get("action") == "BUY")
    sell_c = sum(1 for r in results.values() if r.get("action") == "SELL")
    hold_c = sum(1 for r in results.values() if r.get("action") == "HOLD")
    low_c = sum(1 for r in results.values() if "Low" in str(r.get("risk", "")))
    med_c = sum(1 for r in results.values() if "Medium" in str(r.get("risk", "")))
    hi_c = sum(1 for r in results.values() if "High" in str(r.get("risk", "")))
    dominant = "BUY" if buy_c >= sell_c and buy_c >= hold_c else "SELL" if sell_c >= hold_c else "HOLD"
    risk_full = "Low Risk" if low_c > med_c and low_c > hi_c else "High Risk" if hi_c > med_c else "Medium Risk"
    cv = rl_canvas.Canvas(filepath, pagesize=letter)

    def page_bg():
        cv.setFillColor(PAGE_BG)
        cv.rect(0, 0, PW, PH, fill=1, stroke=0)
        import os as _os
        _base = _os.path.dirname(__file__)
        _candidates = [_os.path.join(_base, "..", "static", "images", "logo.png"),
                       _os.path.join(_base, "static", "images", "logo.png"),
                       _os.path.join(_base, "..", "..", "static", "images", "logo.png")]
        _logo_path = next((_p for _p in _candidates if _os.path.exists(_os.path.normpath(_p))), None)
        if _logo_path:
            from reportlab.lib.utils import ImageReader as _IR
            try:
                _img = _IR(_os.path.normpath(_logo_path))
                _sz = 250
                cv.saveState()
                cv.setFillAlpha(0.05)
                cv.drawImage(_img, (PW - _sz) / 2, (PH - _sz) / 2, width=_sz, height=_sz, mask="auto", preserveAspectRatio=True)
                cv.restoreState()
            except Exception:
                pass

    def top_bar(subtitle=""):
        cv.setFillColor(NAVY)
        cv.rect(0, PH - 72, PW, 72, fill=1, stroke=0)
        cv.setFillColor(GOLD)
        cv.rect(0, PH - 74, PW, 2.5, fill=1, stroke=0)
        cv.setFont("Helvetica", 7)
        cv.setFillColor(HexColor("#7a9abf"))
        cv.drawString(40, PH - 14, "EQUITY RESEARCH  |  AI ANALYTICS DIVISION")
        cv.drawRightString(PW - 40, PH - 14, today)
        cv.setStrokeColor(HexColor("#1a4070"))
        cv.setLineWidth(0.4)
        cv.line(40, PH - 22, PW - 40, PH - 22)
        cv.setFont("Helvetica-Bold", 12)
        cv.setFillColor(HexColor("#ffffff"))
        cv.drawString(40, PH - 42, "DATAMARKERS")
        cv.setFont("Helvetica", 9.5)
        cv.setFillColor(HexColor("#aabbd4"))
        cv.drawString(148, PH - 42, subtitle if subtitle else "AI-Powered Forecasting Platform")
        cv.setFont("Helvetica-Bold", 8)
        cv.setFillColor(GOLD)
        cv.drawRightString(PW - 40, PH - 42, "CONFIDENTIAL")
        cv.setFont("Helvetica", 6.5)
        cv.setFillColor(HexColor("#5a7a9f"))
        cv.drawString(40, PH - 60, "AI-Generated Equity Research  |  For informational purposes only  |  Not Financial Advice")

    def footer(pg):
        cv.setFillColor(HexColor("#f5f7fb"))
        cv.rect(0, 0, PW, 54, fill=1, stroke=0)
        cv.setFillColor(NAVY)
        cv.rect(0, 53, PW, 2, fill=1, stroke=0)
        cv.setFillColor(GOLD)
        cv.rect(0, 53, 90, 2, fill=1, stroke=0)
        cv.setFont("Helvetica", 6.5)
        cv.setFillColor(LGREY)
        cv.drawString(40, 38, "This report is generated by Stock AI Analytics System. For informational and educational purposes only.")
        cv.drawString(40, 26, "NOT financial advice. Past performance does not guarantee future results.")
        cv.drawString(40, 14, f"© {datetime.datetime.now().year} Datamarkers  |  AI Analytics Division  |  Confidential")
        cv.setFont("Helvetica-Bold", 10)
        cv.setFillColor(NAVY)
        cv.drawRightString(PW - 40, 36, f"Page {pg} / {total_pages}")
        cv.setFont("Helvetica", 7)
        cv.setFillColor(LGREY)
        cv.drawRightString(PW - 40, 23, today)

    def section_rule(y, title):
        cv.setFont("Helvetica-Bold", 9.5)
        cv.setFillColor(NAVY)
        cv.drawString(40, y + 2, title.upper())
        cv.setFillColor(NAVY)
        cv.rect(40, y - 2, PW - 80, 1.5, fill=1, stroke=0)
        cv.setFillColor(GOLD)
        cv.rect(40, y - 2, min(len(title) * 6.2, 200), 1.5, fill=1, stroke=0)
        return y - 18

    def metric_card(bx, by, bw, bh, label, value, sub="", vc=NAVY):
        cv.setFillColor(PAGE_BG)
        cv.rect(bx, by, bw, bh, fill=1, stroke=0)
        cv.setStrokeColor(BORDER)
        cv.setLineWidth(0.5)
        cv.rect(bx, by, bw, bh, fill=0, stroke=1)
        cv.setFillColor(vc)
        cv.rect(bx, by + bh - 4, bw, 4, fill=1, stroke=0)
        cv.setFont("Helvetica", 7)
        cv.setFillColor(LGREY)
        cv.drawCentredString(bx + bw / 2, by + bh - 16, label.upper())
        fs = 15 if len(str(value)) <= 6 else 10
        cv.setFont("Helvetica-Bold", fs)
        cv.setFillColor(vc)
        cv.drawCentredString(bx + bw / 2, by + bh // 2 - 4, str(value))
        if sub:
            cv.setFont("Helvetica", 6.5)
            cv.setFillColor(LGREY)
            cv.drawCentredString(bx + bw / 2, by + 6, sub)

    def pro_table(y, headers, rows, col_widths):
        tw = sum(col_widths)
        rh = 19
        hh = 23
        cv.setFillColor(NAVY)
        cv.rect(40, y - hh, tw, hh, fill=1, stroke=0)
        cv.setFont("Helvetica-Bold", 8)
        cv.setFillColor(HexColor("#ffffff"))
        x = 40
        for hdr, cw in zip(headers, col_widths):
            cv.drawCentredString(x + cw / 2, y - hh + 8, hdr)
            x += cw
        y -= hh
        for ri, row_data in enumerate(rows):
            bg = ALTROW if ri % 2 == 0 else PAGE_BG
            cv.setFillColor(bg)
            cv.rect(40, y - rh, tw, rh, fill=1, stroke=0)
            cv.setStrokeColor(BORDER)
            cv.setLineWidth(0.3)
            cv.line(40, y - rh, 40 + tw, y - rh)
            x = 40
            for ci, (cell, cw) in enumerate(zip(row_data, col_widths)):
                s = str(cell)
                fc = BLACK
                if s in ("BUY", "STRONG BUY", "★ BEST"):
                    fc = BUY_C
                elif s in ("SELL", "STRONG SELL"):
                    fc = SELL_C
                elif s == "HOLD":
                    fc = HOLD_C
                elif s.startswith("+"):
                    fc = BUY_C
                elif s.startswith("-") and s != "--":
                    fc = SELL_C
                elif s in ("▲", "✓"):
                    fc = BUY_C
                elif s == "▼":
                    fc = SELL_C
                font = "Helvetica-Bold" if (ci == 0 or fc != BLACK) else "Helvetica"
                cv.setFont(font, 8)
                cv.setFillColor(fc)
                cv.drawCentredString(x + cw / 2, y - rh + 6, s[:28])
                x += cw
            y -= rh
        cv.setStrokeColor(NAVY)
        cv.setLineWidth(0.8)
        cv.rect(40, y, tw, hh + rh * len(rows), fill=0, stroke=1)
        return y - 10

    def spark(sx, sy, sw, sh, vals, color=BLUE):
        if not vals or len(vals) < 2:
            return
        mn, mx = min(vals), max(vals)
        rng = mx - mn if mx != mn else 1
        pts = [(sx + i * (sw / (len(vals) - 1)), sy + (v - mn) / rng * sh) for i, v in enumerate(vals)]
        cv.setFillColor(HexColor("#ddeeff"))
        p = cv.beginPath()
        p.moveTo(pts[0][0], sy)
        for px, py in pts:
            p.lineTo(px, py)
        p.lineTo(pts[-1][0], sy)
        p.close()
        cv.drawPath(p, fill=1, stroke=0)
        cv.setStrokeColor(color)
        cv.setLineWidth(1.6)
        p2 = cv.beginPath()
        p2.moveTo(*pts[0])
        for px, py in pts[1:]:
            p2.lineTo(px, py)
        cv.drawPath(p2, fill=0, stroke=1)
        cv.setFillColor(color)
        cv.circle(pts[-1][0], pts[-1][1], 2.5, fill=1, stroke=0)
        cv.setFont("Helvetica", 6.5)
        cv.setFillColor(LGREY)
        cv.drawString(pts[0][0], sy - 9, f"{vals[0]:.2f}")
        cv.drawRightString(pts[-1][0], sy - 9, f"{vals[-1]:.2f}")

    def info_box(y, lines_data):
        total_h = len(lines_data) * 13 + 20
        cv.setFillColor(LIGHTBLUE)
        cv.rect(40, y - total_h, PW - 80, total_h, fill=1, stroke=0)
        cv.setStrokeColor(BLUE)
        cv.setLineWidth(0.5)
        cv.rect(40, y - total_h, PW - 80, total_h, fill=0, stroke=1)
        cv.setFillColor(NAVY)
        cv.rect(40, y - total_h, 4, total_h, fill=1, stroke=0)
        for i, (lbl, val, bold) in enumerate(lines_data):
            ty = y - 14 - i * 13
            cv.setFont("Helvetica-Bold" if bold else "Helvetica", 8.5)
            cv.setFillColor(NAVY if bold else BLACK)
            cv.drawString(52, ty, lbl)
            cv.setFont("Helvetica-Bold", 8.5)
            cv.setFillColor(BLACK)
            cv.drawRightString(PW - 52, ty, str(val))
        return y - total_h - 10

    def disclaimer_box(y):
        h = 42
        cv.setFillColor(HexColor("#fff8e8"))
        cv.rect(40, y - h, PW - 80, h, fill=1, stroke=0)
        cv.setStrokeColor(GOLD)
        cv.setLineWidth(0.8)
        cv.rect(40, y - h, PW - 80, h, fill=0, stroke=1)
        cv.setFillColor(GOLD)
        cv.rect(40, y - h, 4, h, fill=1, stroke=0)
        cv.setFont("Helvetica-Bold", 8)
        cv.setFillColor(HOLD_C)
        cv.drawString(52, y - 14, "IMPORTANT DISCLAIMER")
        cv.setFont("Helvetica", 7.5)
        cv.setFillColor(BLACK)
        cv.drawString(52, y - 26, "All predictions are AI-generated for informational and educational purposes only.")
        cv.drawString(52, y - 38, "This is NOT financial advice. Always consult a licensed financial advisor.")
        return y - h - 10

    page_bg()
    top_bar()
    footer(1)
    y = PH - 92
    cv.setFont("Helvetica-Bold", 22)
    cv.setFillColor(NAVY)
    cv.drawString(40, y, "STOCK AI ANALYTICS REPORT")
    y -= 5
    cv.setFillColor(GOLD)
    cv.rect(40, y, 340, 2, fill=1, stroke=0)
    y -= 14
    cv.setFont("Helvetica", 10)
    cv.setFillColor(GREY)
    cv.drawString(40, y, f"AI-Powered Equity Research   |   {today}   |   Prepared for: {username.upper()}")
    y -= 28
    bw = (PW - 80) / 6 - 4
    bh = 60
    rsk_c = BUY_C if "Low" in risk_full else SELL_C if "High" in risk_full else HOLD_C
    cards = [("Total Stocks", str(total), "Analyzed", NAVY), ("Avg Confidence", f"{avg_c}%", "AI Score", BLUE),
             ("BUY Signals", str(buy_c), "Positive", BUY_C), ("SELL Signals", str(sell_c), "Negative", SELL_C),
             ("HOLD Signals", str(hold_c), "Neutral", HOLD_C), ("Portfolio Risk", risk_full.replace(" Risk", ""), "Level", rsk_c)]
    for i, (lbl, val, sub, vc) in enumerate(cards):
        metric_card(40 + i * (bw + 4), y - bh, bw, bh, lbl, val, sub, vc)
    y -= bh + 24
    y = section_rule(y, "Executive Summary")
    y -= 4
    para_lines = [("Report Coverage:", f"{total} stock(s) analyzed using 4 AI/ML models", True),
                  ("Models Used:", "Linear Regression  |  Random Forest  |  Gradient Boosting  |  LSTM (Deep Learning)", False),
                  ("Dominant Signal:", f"{dominant}  —  Average AI Confidence: {avg_c}%", True),
                  ("Portfolio Risk:", f"{risk_full}  —  Low: {low_c}  |  Medium: {med_c}  |  High: {hi_c}", False),
                  ("Model Selection:", "Auto-selected per stock using composite R² and RMSE scoring", False),
                  ("Technical Indicators:", "RSI, MACD, EMA 12/26, Bollinger Bands, MA 7/14, Volatility", False)]
    y = info_box(y, para_lines)
    y = section_rule(y, "Report Statistics")
    y -= 4
    stat_rows = [["Total Stocks Analyzed", str(total), "ML Models Used", "LR / RF / GBM / LSTM"],
                 ["Average AI Confidence", f"{avg_c}%", "Best Model Selection", "Auto (R² + RMSE)"],
                 ["BUY Signals", str(buy_c), "Low Risk Stocks", str(low_c)],
                 ["SELL Signals", str(sell_c), "Medium Risk Stocks", str(med_c)],
                 ["HOLD Signals", str(hold_c), "High Risk Stocks", str(hi_c)]]
    y = pro_table(y, ["METRIC", "VALUE", "METRIC", "VALUE"], stat_rows, [(PW - 80) / 4] * 4)
    y = section_rule(y, "AI Models Overview")
    y -= 4
    model_rows = [["Linear Regression", "Baseline Trend", "Fast", "Low", "—"],
                  ["Random Forest", "Ensemble", "Medium", "Medium", "—"],
                  ["Gradient Boosting", "Ensemble (Best)", "Medium", "Medium", "★ BEST"],
                  ["LSTM (Deep Learning)", "Neural Network", "Slow", "High", "—"]]
    y = pro_table(y, ["MODEL", "TYPE", "SPEED", "COMPLEXITY", "SELECTION"], model_rows, [160, 100, 60, 80, 70])
    y -= 8
    cv.setFillColor(LIGHTBLUE)
    cv.rect(40, y - 28, PW - 80, 28, fill=1, stroke=0)
    cv.setStrokeColor(BLUE)
    cv.setLineWidth(0.5)
    cv.rect(40, y - 28, PW - 80, 28, fill=0, stroke=1)
    cv.setFillColor(NAVY)
    cv.rect(40, y - 28, 4, 28, fill=1, stroke=0)
    cv.setFont("Helvetica-Bold", 8.5)
    cv.setFillColor(NAVY)
    cv.drawString(52, y - 11, "Stocks in this report:")
    cv.setFont("Helvetica", 8.5)
    cv.setFillColor(BLACK)
    cv.drawString(190, y - 11, "  |  ".join(results.keys())[:80])
    y -= 38
    disclaimer_box(y)

    for pg_idx, (stock_col, res) in enumerate(results.items(), start=2):
        cv.showPage()
        page_bg()
        top_bar(f"{stock_col} — Stock Analysis Report")
        footer(pg_idx)
        y = PH - 92
        act = res.get("action", "HOLD")
        an = analytics.get(stock_col, {})
        preds = preds_raw.get(stock_col, [])
        lstm_p = lstm_preds.get(stock_col, [])
        conf_val = float(res.get("confidence", 0))
        last_v = res.get("last", "--")
        avg_v = res.get("avg", "--")
        trend_v = an.get("trend_percent", "--")
        act_c = BUY_C if act == "BUY" else SELL_C if act == "SELL" else HOLD_C
        cv.setFont("Helvetica-Bold", 17)
        cv.setFillColor(NAVY)
        cv.drawString(40, y, f"{stock_col}  —  Detailed Stock Analysis")
        y -= 5
        cv.setFillColor(GOLD)
        cv.rect(40, y, 320, 2, fill=1, stroke=0)
        y -= 14
        _fg = act_c
        _bg_c = HexColor("#e6f4ec") if act == "BUY" else HexColor("#fce8e8") if act == "SELL" else HexColor("#fef9e7")
        _sw2 = 160
        _sh2 = 40
        cv.setFillColor(_bg_c)
        cv.roundRect(40, y - _sh2, _sw2, _sh2, 5, fill=1, stroke=0)
        cv.setStrokeColor(_fg)
        cv.setLineWidth(2)
        cv.roundRect(40, y - _sh2, _sw2, _sh2, 5, fill=0, stroke=1)
        cv.setLineWidth(0.7)
        cv.roundRect(44, y - _sh2 + 3, _sw2 - 8, _sh2 - 6, 4, fill=0, stroke=1)
        cv.setFont("Helvetica-Bold", 14)
        cv.setFillColor(_fg)
        cv.drawCentredString(40 + _sw2 / 2, y - _sh2 + 15, f"RATING:  {act}")
        cv.setFont("Helvetica", 6.5)
        cv.drawCentredString(40 + _sw2 / 2, y - _sh2 + 6, "AI ANALYST SIGNAL")
        y -= _sh2 + 12
        _bar_h = 38
        cv.setFillColor(NAVY)
        cv.rect(40, y - _bar_h, PW - 80, _bar_h, fill=1, stroke=0)
        cv.setStrokeColor(GOLD)
        cv.setLineWidth(1.2)
        cv.rect(40, y - _bar_h, PW - 80, _bar_h, fill=0, stroke=1)
        _bar_items = [("LAST PRICE", str(last_v)), ("7-DAY TARGET", str(avg_v)),
                      ("CONFIDENCE", f"{conf_val:.0f}%"), ("RSI (14)", str(an.get("rsi", "--"))),
                      ("MACD", str(an.get("macd", "--"))), ("VOLATILITY", f"{an.get('volatility', '--')}%"),
                      ("BEST MODEL", str(an.get("best_model", "--"))[:13]), ("RISK LEVEL", str(res.get("risk", "--")))]
        _slot = (PW - 80) / len(_bar_items)
        for _i, (_lbl, _val) in enumerate(_bar_items):
            _x = 40 + _i * _slot
            if _i > 0:
                cv.setStrokeColor(HexColor("#1a4070"))
                cv.setLineWidth(0.5)
                cv.line(_x, y - _bar_h + 6, _x, y - 6)
            cv.setFont("Helvetica", 6)
            cv.setFillColor(HexColor("#8ab0d4"))
            cv.drawCentredString(_x + _slot / 2, y - _bar_h + 10, _lbl)
            cv.setFont("Helvetica-Bold", 9)
            cv.setFillColor(HexColor("#ffffff"))
            cv.drawCentredString(_x + _slot / 2, y - _bar_h + 23, _val)
        y -= _bar_h + 16
        bw2 = (PW - 80) / 4 - 4
        bh2 = 52
        try:
            tv = float(trend_v or 0)
        except:
            tv = 0
        key_cards = [("Last Price", f"{last_v}", "Actual", NAVY), ("7-Day Target", f"{avg_v}", "AI Forecast", act_c),
                     ("AI Confidence", f"{conf_val:.0f}%", "Score", BLUE), ("Trend", f"{trend_v}%", "7-Day", BUY_C if tv > 0 else SELL_C)]
        for i, (lbl, val, sub, vc) in enumerate(key_cards):
            metric_card(40 + i * (bw2 + 4), y - bh2, bw2, bh2, lbl, val, sub, vc)
        y -= bh2 + 18
        y = section_rule(y, "7-Day Price Forecast")
        y -= 10
        cw_chart = (PW - 100) / 2
        ch_h = 46
        if len(preds) >= 2:
            spark(40, y - ch_h, cw_chart, ch_h - 8, preds, BLUE)
            cv.setFont("Helvetica-Bold", 8)
            cv.setFillColor(NAVY)
            cv.drawString(40, y, f"Best Model  ({an.get('best_model', 'ML Model')})")
            cv.setFont("Helvetica", 7)
            cv.setFillColor(LGREY)
            cv.drawString(40, y - ch_h - 10, "Forecast trajectory")
        if len(lstm_p) >= 2:
            spark(50 + cw_chart, y - ch_h, cw_chart, ch_h - 8, lstm_p, HexColor("#6633cc"))
            cv.setFont("Helvetica-Bold", 8)
            cv.setFillColor(HexColor("#6633cc"))
            cv.drawString(50 + cw_chart, y, "LSTM  (Deep Learning)")
            cv.setFont("Helvetica", 7)
            cv.setFillColor(LGREY)
            cv.drawString(50 + cw_chart, y - ch_h - 10, "Neural network trajectory")
        y -= ch_h + 22
        if preds:
            best_name = an.get("best_model", "Best Model")[:14]
            has_lstm = bool(lstm_p)
            f_rows = []
            for i, val in enumerate(preds[:7]):
                chg = val - preds[i - 1] if i > 0 else 0
                chg_s = f"+{chg:.2f}" if chg > 0 else f"{chg:.2f}" if chg < 0 else "Base"
                lv = f"{lstm_p[i]:.2f}" if has_lstm and i < len(lstm_p) else "--"
                direction = "▲" if chg > 0 else ("▼" if chg < 0 else "—")
                if has_lstm:
                    f_rows.append([f"Day {i + 1}", f"{val:.2f}", lv, chg_s, direction])
                else:
                    f_rows.append([f"Day {i + 1}", f"{val:.2f}", chg_s, direction])
            if has_lstm:
                y = pro_table(y, ["DAY", best_name.upper(), "LSTM", "CHANGE", "DIR"], f_rows, [50, 120, 110, 80, 50])
            else:
                y = pro_table(y, ["DAY", best_name.upper(), "CHANGE", "DIR"], f_rows, [50, 180, 110, 70])
        if an:
            y = section_rule(y, "Technical Indicators")
            y -= 4
            rsi_v = float(an.get("rsi", 50))
            macd_v = float(an.get("macd") or 0)
            all_ind = [["RSI (14)", f"{rsi_v:.1f}", "Oversold–BUY" if rsi_v < 30 else "Overbought–SELL" if rsi_v > 70 else "Neutral", "▲" if rsi_v < 30 else ("▼" if rsi_v > 70 else "—")],
                       ["MACD", f"{macd_v:.2f}", "Bullish" if macd_v > 0 else "Bearish", "▲" if macd_v > 0 else "▼"],
                       ["EMA 12", str(an.get("ema12", "--")), "Short-term EMA", "—"], ["EMA 26", str(an.get("ema26", "--")), "Long-term EMA", "—"],
                       ["MA 7", str(an.get("ma7", "--")), "7-day moving avg", "—"], ["MA 14", str(an.get("ma14", "--")), "14-day moving avg", "—"],
                       ["Bollinger Upper", str(an.get("bb_upper", "--")), "Resistance band", "—"], ["Bollinger Lower", str(an.get("bb_lower", "--")), "Support band", "—"],
                       ["Volatility", f"{an.get('volatility', '--')}%", "Price stability", "✓" if float(an.get("volatility") or 99) < 3 else "▼"],
                       ["Trend (7-day)", f"{an.get('trend_percent', '--')}%", "Forecast dir", "▲" if float(an.get("trend_percent") or 0) > 0 else "▼"]]
            _half = 5
            _ileft = all_ind[:_half]
            _iright = all_ind[_half:]
            _cw2 = (PW - 80) / 2 - 3
            _irh = 18
            _ihh = 22
            for _xo in [40, 40 + _cw2 + 6]:
                cv.setFillColor(NAVY)
                cv.rect(_xo, y - _ihh, _cw2, _ihh, fill=1, stroke=0)
                cv.setFont("Helvetica-Bold", 7.5)
                cv.setFillColor(HexColor("#ffffff"))
                cv.drawString(_xo + 6, y - _ihh + 8, "INDICATOR")
                cv.drawString(_xo + _cw2 * 0.44, y - _ihh + 8, "VALUE")
                cv.drawString(_xo + _cw2 * 0.60, y - _ihh + 8, "INTERPRETATION")
                cv.drawRightString(_xo + _cw2 - 6, y - _ihh + 8, "SIG")
            y -= _ihh
            for _ri in range(max(len(_ileft), len(_iright))):
                _bg = ALTROW if _ri % 2 == 0 else PAGE_BG
                for _lst, _xo in [(_ileft, 40), (_iright, 40 + _cw2 + 6)]:
                    if _ri < len(_lst):
                        _r = _lst[_ri]
                        _fc = BUY_C if _r[3] in ("▲", "✓") else (SELL_C if _r[3] == "▼" else BLACK)
                        cv.setFillColor(_bg)
                        cv.rect(_xo, y - _irh, _cw2, _irh, fill=1, stroke=0)
                        cv.setStrokeColor(BORDER)
                        cv.setLineWidth(0.3)
                        cv.line(_xo, y - _irh, _xo + _cw2, y - _irh)
                        cv.setFont("Helvetica-Bold", 7.5)
                        cv.setFillColor(NAVY)
                        cv.drawString(_xo + 6, y - _irh + 5, _r[0])
                        cv.setFont("Helvetica", 7.5)
                        cv.setFillColor(_fc)
                        cv.drawString(_xo + _cw2 * 0.44, y - _irh + 5, str(_r[1]))
                        cv.setFont("Helvetica", 6.8)
                        cv.setFillColor(LGREY)
                        cv.drawString(_xo + _cw2 * 0.60, y - _irh + 5, str(_r[2])[:18])
                        cv.setFont("Helvetica-Bold", 9)
                        cv.setFillColor(_fc)
                        cv.drawRightString(_xo + _cw2 - 6, y - _irh + 5, _r[3])
                y -= _irh
            for _xo in [40, 40 + _cw2 + 6]:
                cv.setStrokeColor(NAVY)
                cv.setLineWidth(0.8)
                cv.rect(_xo, y, _cw2, _ihh + _irh * max(len(_ileft), len(_iright)), fill=0, stroke=1)
            y -= 12
        mdls = an.get("models", {})
        best_m = an.get("best_model", "")
        if mdls and y > 100:
            y = section_rule(y, "Model Performance Comparison")
            y -= 4
            m_rows = [[mname, str(mdata.get("r2", "--")), str(mdata.get("mae", "--")), str(mdata.get("rmse", "--")),
                       "★ BEST" if mname == best_m else "LSTM" if "LSTM" in mname else "—"] for mname, mdata in mdls.items()]
            y = pro_table(y, ["MODEL", "R² SCORE", "MAE", "RMSE", "SELECTION"], m_rows, [180, 70, 70, 70, 80])

    cv.save()
    from flask import make_response
    response = make_response(send_file(os.path.abspath(filepath), as_attachment=True,
                                       download_name=f"StockAI_Report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                                       mimetype="application/pdf"))
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


# ==========================
# TRADER TERMINAL
# ==========================

@app.route('/trader')
def trader():
    if 'user' not in session:
        return redirect(url_for('login'))

    username = session['user']
    try:
        conn = sqlite3.connect('data/users.db')
        c = conn.cursor()
        c.execute("SELECT symbol, added_at FROM watchlist WHERE username = ?", (username,))
        rows = c.fetchall()
        conn.close()
        watchlist = [{'symbol': row[0], 'added_at': row[1]} for row in rows]
    except Exception as e:
        print(f"Trader watchlist error: {e}")
        watchlist = []

    return render_template('trader.html', watchlist=watchlist)


# ==========================
# WATCHLIST JSON API
# ==========================

@app.route("/api/watchlist-prices")
def api_watchlist_prices():
    if "user" not in session:
        return jsonify([]), 401
    username = session["user"]
    try:
        conn = sqlite3.connect("data/users.db")
        c = conn.cursor()
        c.execute("SELECT symbol FROM watchlist WHERE username = ?", (username,))
        symbols = [row[0] for row in c.fetchall()]
        conn.close()
    except Exception as e:
        print("watchlist_prices DB error:", e)
        return jsonify([])

    if not symbols:
        return jsonify([])

    result = []
    for sym in symbols:
        try:
            ticker = yf.Ticker(sym)
            info = ticker.info or {}
            price = (info.get("regularMarketPrice") or
                     info.get("currentPrice") or
                     info.get("previousClose"))
            prev = info.get("previousClose") or info.get("regularMarketPreviousClose")
            if price and prev and float(prev) != 0:
                change_pct = ((float(price) - float(prev)) / float(prev)) * 100
            else:
                hist = ticker.history(period="2d")
                if hist is not None and len(hist) >= 2:
                    price = float(hist["Close"].iloc[-1])
                    prev_price = float(hist["Close"].iloc[-2])
                    change_pct = ((price - prev_price) / prev_price) * 100
                else:
                    price = None
                    change_pct = 0.0
            result.append({"symbol": sym,
                           "price": round(float(price), 2) if price else "N/A",
                           "change": round(change_pct, 2)})
        except Exception as e:
            print(f"Price fetch error [{sym}]:", e)
            result.append({"symbol": sym, "price": "N/A", "change": 0})

    return jsonify(result)


@app.route("/api/watchlist/add", methods=["POST"])
def api_watchlist_add():
    if "user" not in session:
        return jsonify({"error": "Not logged in"}), 401
    data = request.get_json() or {}
    symbol = (data.get("symbol") or "").strip().upper()
    if not symbol:
        return jsonify({"error": "Symbol required"}), 400
    try:
        conn = sqlite3.connect("data/users.db")
        c = conn.cursor()
        c.execute("INSERT OR IGNORE INTO watchlist (username, symbol) VALUES (?, ?)",
                  (session["user"], symbol))
        conn.commit()
        conn.close()
        add_watchlist_notification(session.get("user"), symbol, "add")
        return jsonify({"success": True, "symbol": symbol})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/watchlist/remove", methods=["POST"])
def api_watchlist_remove():
    if "user" not in session:
        return jsonify({"error": "Not logged in"}), 401
    data = request.get_json() or {}
    symbol = (data.get("symbol") or "").strip().upper()
    if not symbol:
        return jsonify({"error": "Symbol required"}), 400
    try:
        conn = sqlite3.connect("data/users.db")
        c = conn.cursor()
        c.execute("DELETE FROM watchlist WHERE username = ? AND symbol = ?",
                  (session["user"], symbol))
        conn.commit()
        conn.close()
        
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==========================
# NEWS FEED
# ==========================

@app.route("/news")
def news_feed():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("news.html")


@app.route("/api/news/<symbol>")
def get_news(symbol):
    try:
        stock = yf.Ticker(symbol.upper())
        news = stock.news
        results_list = []
        for item in news[:10]:
            content = item.get("content", {})
            title = content.get("title", "No title")
            summary = content.get("summary", "")
            pub_raw = content.get("pubDate", "")
            provider = content.get("provider", {})
            source = provider.get("displayName", "Yahoo Finance") if isinstance(provider, dict) else "Yahoo Finance"
            click_url = content.get("clickThroughUrl", {})
            url = click_url.get("url", "#") if isinstance(click_url, dict) else "#"
            if url == "#":
                canon = content.get("canonicalUrl", {})
                url = canon.get("url", "#") if isinstance(canon, dict) else "#"
            try:
                dt = datetime.datetime.strptime(pub_raw[:19], "%Y-%m-%dT%H:%M:%S")
                pub = dt.strftime("%d %b %Y %I:%M %p")
            except Exception:
                pub = pub_raw[:10] if pub_raw else "Recent"
            try:
                from backend.utils.sentiment import analyze as _sa
                _api_key = os.getenv("ANTHROPIC_API_KEY", "")
                _sent = _sa(title + ". " + summary[:150], use_claude=bool(_api_key), api_key=_api_key)
            except Exception:
                _sent = {"label": "neutral", "score": 0, "emoji": "🟡", "badge": "Neutral", "impact": "0.00", "summary": "", "engine": "none"}
            results_list.append({"title": title, "summary": summary[:200] + "..." if len(summary) > 200 else summary,
                                 "source": source, "url": url, "time": pub, "sentiment": _sent})
        return jsonify({"symbol": symbol.upper(), "news": results_list})
    except Exception as e:
        print("News error:", e)
        return jsonify({"symbol": symbol.upper(), "news": []})


@app.route("/api/market-news")
def get_market_news():
    try:
        symbols = ["SPY", "QQQ", "AAPL", "TSLA", "NVDA"]
        all_news = []
        seen = set()
        for sym in symbols:
            try:
                stock = yf.Ticker(sym)
                news = stock.news
                for item in news[:4]:
                    content = item.get("content", {})
                    title = content.get("title", "")
                    if not title or title in seen:
                        continue
                    seen.add(title)
                    summary = content.get("summary", "")
                    pub_raw = content.get("pubDate", "")
                    provider = content.get("provider", {})
                    source = provider.get("displayName", "Yahoo Finance") if isinstance(provider, dict) else "Yahoo Finance"
                    click_url = content.get("clickThroughUrl", {})
                    url = click_url.get("url", "#") if isinstance(click_url, dict) else "#"
                    if url == "#":
                        canon = content.get("canonicalUrl", {})
                        url = canon.get("url", "#") if isinstance(canon, dict) else "#"
                    try:
                        dt = datetime.datetime.strptime(pub_raw[:19], "%Y-%m-%dT%H:%M:%S")
                        pub = dt.strftime("%d %b %Y %I:%M %p")
                    except Exception:
                        pub = pub_raw[:10] if pub_raw else "Recent"
                    all_news.append({"title": title, "summary": summary[:200] + "..." if len(summary) > 200 else summary,
                                     "source": source, "url": url, "time": pub, "tag": sym})
            except Exception:
                continue
        return jsonify({"news": all_news[:20]})
    except Exception as e:
        print("Market news error:", e)
        return jsonify({"news": []})


# ==========================
# SENTIMENT ANALYSIS ROUTES
# ==========================

@app.route("/api/sentiment/<symbol>")
def sentiment_for_symbol(symbol):
    if "user" not in session:
        return jsonify({"error": "Not logged in"}), 401
    try:
        from backend.utils.sentiment import analyze, sentiment_summary
        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        use_claude = bool(api_key)
        stock = yf.Ticker(symbol.upper())
        news = stock.news or []
        results = []
        for item in news[:12]:
            content = item.get("content", {})
            title = content.get("title", "")
            summary = content.get("summary", "")
            if not title:
                continue
            text = title + ". " + summary[:200]
            sentiment = analyze(text, use_claude=use_claude, api_key=api_key)
            pub_raw = content.get("pubDate", "")
            provider = content.get("provider", {})
            source = provider.get("displayName", "Yahoo Finance") if isinstance(provider, dict) else "Yahoo Finance"
            click = content.get("clickThroughUrl", {})
            url = click.get("url", "#") if isinstance(click, dict) else "#"
            if url == "#":
                canon = content.get("canonicalUrl", {})
                url = canon.get("url", "#") if isinstance(canon, dict) else "#"
            try:
                dt = datetime.datetime.strptime(pub_raw[:19], "%Y-%m-%dT%H:%M:%S")
                pub = dt.strftime("%d %b %Y %I:%M %p")
            except Exception:
                pub = pub_raw[:10] if pub_raw else "Recent"
            results.append({"title": title, "summary": summary[:200] + "..." if len(summary) > 200 else summary,
                            "source": source, "url": url, "time": pub, "sentiment": sentiment})
        overall = sentiment_summary([r["sentiment"] for r in results])
        return jsonify({"symbol": symbol.upper(), "news": results, "overall": overall})
    except Exception as e:
        print("Sentiment error:", e)
        return jsonify({"symbol": symbol.upper(), "news": [], "overall": {}})


@app.route("/api/sentiment/market")
def sentiment_market():
    if "user" not in session:
        return jsonify({"error": "Not logged in"}), 401
    try:
        from backend.utils.sentiment import analyze, sentiment_summary
        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        use_claude = bool(api_key)
        symbols = ["SPY", "AAPL", "TSLA", "NVDA", "MSFT", "GOOGL", "AMZN"]
        all_sent = []
        per_symbol = {}
        for sym in symbols:
            try:
                news = yf.Ticker(sym).news or []
                sents = []
                for item in news[:5]:
                    title = item.get("content", {}).get("title", "")
                    if title:
                        sents.append(analyze(title, use_claude=use_claude, api_key=api_key))
                if sents:
                    sm = sentiment_summary(sents)
                    per_symbol[sym] = sm
                    all_sent.extend(sents)
            except Exception:
                continue
        overall = sentiment_summary(all_sent)
        return jsonify({"overall": overall, "per_symbol": per_symbol})
    except Exception as e:
        print("Market sentiment error:", e)
        return jsonify({"overall": {}, "per_symbol": {}})


# ==========================
# COMPARE PAGE
# ==========================

@app.route("/compare")
def compare():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("compare.html")


@app.route("/api/compare-full/<sym>")
def api_compare_full(sym):
    try:
        period = request.args.get("period", "1mo")
        ticker = yf.Ticker(sym)
        hist = ticker.history(period=period)
        if hist.empty:
            return jsonify({"error": f"No data for {sym}"})
        closes = hist["Close"].dropna().tolist()
        dates = [d.strftime("%Y-%m-%d") for d in hist.index]
        info = {}
        try:
            info = ticker.info or {}
        except Exception:
            pass
        price = (info.get("regularMarketPrice") or info.get("currentPrice") or
                 info.get("previousClose") or (closes[-1] if closes else None))
        prev_close = info.get("previousClose") or info.get("regularMarketPreviousClose")
        if price and prev_close and prev_close != 0:
            change_pct = ((float(price) - float(prev_close)) / float(prev_close)) * 100
        elif len(closes) >= 2:
            change_pct = ((closes[-1] - closes[-2]) / closes[-2]) * 100
        else:
            change_pct = 0.0
        period_return = ((closes[-1] - closes[0]) / closes[0]) * 100 if len(closes) >= 2 else 0.0
        high52 = info.get("fiftyTwoWeekHigh") or info.get("regularMarketDayHigh")
        low52 = info.get("fiftyTwoWeekLow") or info.get("regularMarketDayLow")
        mc_raw = info.get("marketCap")
        if mc_raw:
            if mc_raw >= 1e12:
                mc_fmt = f"${mc_raw / 1e12:.2f}T"
            elif mc_raw >= 1e9:
                mc_fmt = f"${mc_raw / 1e9:.2f}B"
            elif mc_raw >= 1e6:
                mc_fmt = f"${mc_raw / 1e6:.2f}M"
            else:
                mc_fmt = f"${mc_raw:,.0f}"
        else:
            mc_fmt = "N/A"
        vol_raw = None
        if "Volume" in hist.columns and len(hist) > 0:
            vol_raw = hist["Volume"].iloc[-1]
        vol_raw = vol_raw or info.get("regularMarketVolume") or info.get("volume")
        if vol_raw and float(vol_raw) > 0:
            v = float(vol_raw)
            vol_fmt = f"{v / 1e6:.2f}M" if v >= 1e6 else f"{v / 1e3:.1f}K" if v >= 1e3 else str(int(v))
        else:
            vol_fmt = "N/A"
        s = pd.Series(closes)
        ma7 = float(s.rolling(7).mean().iloc[-1]) if len(s) >= 7 else None
        ma14 = float(s.rolling(14).mean().iloc[-1]) if len(s) >= 14 else None
        avg_price = float(s.mean())
        returns = s.pct_change().dropna()
        volatility = float(returns.std() * 100) if len(returns) > 1 else 0.0
        rsi = None
        if len(returns) >= 14:
            gains = returns.clip(lower=0)
            losses = (-returns).clip(lower=0)
            avg_gain = gains.rolling(14).mean().iloc[-1]
            avg_loss = losses.rolling(14).mean().iloc[-1]
            rsi = round(100 - (100 / (1 + avg_gain / avg_loss)), 2) if avg_loss != 0 else 100.0
        macd = None
        if len(s) >= 26:
            ema12 = s.ewm(span=12, adjust=False).mean().iloc[-1]
            ema26 = s.ewm(span=26, adjust=False).mean().iloc[-1]
            macd = round(float(ema12 - ema26), 4)
        bb_upper = bb_lower = None
        if len(s) >= 20:
            ma20 = s.rolling(20).mean().iloc[-1]
            std20 = s.rolling(20).std().iloc[-1]
            bb_upper = round(float(ma20 + 2 * std20), 4)
            bb_lower = round(float(ma20 - 2 * std20), 4)
        signal = "HOLD"
        score = 0
        if rsi is not None:
            score += 2 if rsi < 35 else (-2 if rsi > 65 else 0)
        if macd is not None:
            score += 1 if macd > 0 else -1
        if period_return > 3:
            score += 1
        elif period_return < -3:
            score -= 1
        if score >= 2:
            signal = "BUY"
        elif score <= -2:
            signal = "SELL"
        return jsonify({"price": round(float(price), 4) if price else None, "change_pct": round(change_pct, 4),
                        "period_return": round(period_return, 4), "avg_price": round(avg_price, 4),
                        "high52": round(float(high52), 4) if high52 else None,
                        "low52": round(float(low52), 4) if low52 else None,
                        "market_cap_fmt": mc_fmt, "volume": vol_fmt,
                        "pe_ratio": round(float(info.get("trailingPE") or 0), 2) or None,
                        "beta": round(float(info.get("beta") or 0), 2) or None,
                        "company_name": info.get("longName") or info.get("shortName") or sym,
                        "sector": info.get("sector") or "", "volatility": round(volatility, 4),
                        "rsi": rsi, "macd": macd,
                        "ma7": round(ma7, 4) if ma7 else None, "ma14": round(ma14, 4) if ma14 else None,
                        "bb_upper": bb_upper, "bb_lower": bb_lower, "signal": signal,
                        "closes": [round(c, 4) for c in closes], "dates": dates})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/compare-history/<symbol>")
def compare_history(symbol):
    try:
        period = request.args.get("period", "1mo")
        tk = yf.Ticker(symbol)
        hist = tk.history(period=period)
        if hist is None or hist.empty:
            return jsonify({"error": f"No data for {symbol}"}), 404
        dates = [str(d.date()) for d in hist.index]
        closes = [round(float(v), 2) for v in hist["Close"]]
        period_return = round(((closes[-1] - closes[0]) / closes[0]) * 100, 2) if len(closes) > 1 else 0
        avg_price = round(sum(closes) / len(closes), 2) if closes else 0
        volatility = round(float(hist["Close"].pct_change().std() * 100), 2) if len(closes) > 1 else 0
        return jsonify({"dates": dates, "closes": closes, "period_return": period_return,
                        "avg_price": avg_price, "volatility": volatility})
    except Exception as e:
        print("Compare history error:", e)
        return jsonify({"dates": [], "closes": [], "period_return": 0, "avg_price": 0, "volatility": 0})


# ==========================
# PORTFOLIO ADVISOR
# ==========================

@app.route("/portfolio-advisor")
def portfolio_advisor():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("portfolio_advisor.html")


@app.route("/api/portfolio-advisor", methods=["POST"])
def api_portfolio_advisor():
    import json
    import traceback
    try:
        data = request.get_json()
        amount = float(data.get("amount", 100000))
        risk_level = data.get("risk", "medium").lower()
        horizon_yrs = float(data.get("horizon", 1))
        market = data.get("market", "IN")
        custom_syms = [s.strip().upper() for s in data.get("symbols", []) if s.strip()]
        from scipy.optimize import minimize
        if custom_syms:
            symbols = custom_syms
        elif market == "IN":
            symbols = ["RELIANCE.NS", "INFY.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "ITC.NS", "WIPRO.NS",
                       "LT.NS", "SBIN.NS", "AXISBANK.NS", "BAJFINANCE.NS", "HINDUNILVR.NS", "TATAMOTORS.NS", "SUNPHARMA.NS", "ONGC.NS"]
        else:
            symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "JPM", "JNJ", "V", "PG", "UNH", "META", "TSLA", "BRK-B", "HD", "XOM"]
        RISK_PARAMS = {"low": {"max_alloc": 0.25, "min_stocks": 6, "max_vol": 0.18},
                       "medium": {"max_alloc": 0.35, "min_stocks": 5, "max_vol": 0.28},
                       "high": {"max_alloc": 0.45, "min_stocks": 4, "max_vol": 0.99}}
        rp = RISK_PARAMS.get(risk_level, RISK_PARAMS["medium"])
        rf_rate = 0.065 if market == "IN" else 0.045
        period = "2y" if horizon_yrs >= 1 else "1y"
        raw = yf.download(symbols, period=period, auto_adjust=True, progress=False)["Close"]
        raw = raw.dropna(axis=1, thresh=int(len(raw) * 0.85)).dropna()
        if raw.shape[1] < 3:
            return json.dumps({"error": "Not enough valid stock data. Try different symbols."}), 400
        available = list(raw.columns)
        returns = raw.pct_change().dropna()
        ann_returns = returns.mean() * 252
        ann_vol = returns.std() * np.sqrt(252)
        cov_matrix = returns.cov() * 252
        eligible = [s for s in available if ann_vol[s] <= rp["max_vol"]] if risk_level == "low" else available
        if len(eligible) < 3:
            eligible = available
        ret_e = ann_returns[eligible]
        vol_e = ann_vol[eligible]
        cov_e = cov_matrix.loc[eligible, eligible].values
        n = len(eligible)

        def neg_sharpe(w):
            return -(np.dot(w, ret_e.values) - rf_rate) / (np.sqrt(w @ cov_e @ w) + 1e-9)

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
        bounds = [(0.03, rp["max_alloc"])] * n
        best_result = None
        for _ in range(30):
            w0 = np.random.dirichlet(np.ones(n))
            r = minimize(neg_sharpe, w0, method="SLSQP", bounds=bounds, constraints=constraints, options={"maxiter": 500, "ftol": 1e-10})
            if r.success and (best_result is None or r.fun < best_result.fun):
                best_result = r
        if best_result is None:
            sharpes = (ret_e - rf_rate) / vol_e
            top = sharpes.nlargest(5).index.tolist()
            weights = {s: round(1 / len(top), 4) for s in top}
        else:
            raw_w = best_result.x
            raw_w[raw_w < 0.02] = 0
            if raw_w.sum() > 0:
                raw_w /= raw_w.sum()
            idx_sorted = np.argsort(raw_w)[::-1]
            keep = [i for i in idx_sorted if raw_w[i] >= 0.04][:10]
            if len(keep) < rp["min_stocks"]:
                keep = list(idx_sorted[:rp["min_stocks"]])
            final_w = np.zeros(n)
            for i in keep:
                final_w[i] = raw_w[i]
            final_w /= final_w.sum()
            weights = {eligible[i]: round(float(final_w[i]), 4) for i in range(n) if final_w[i] > 0}
        w_arr = np.array([weights.get(s, 0) for s in eligible])
        port_ret = float(np.dot(w_arr, ret_e.values))
        port_vol = float(np.sqrt(w_arr @ cov_e @ w_arr))
        sharpe = round((port_ret - rf_rate) / (port_vol + 1e-9), 2)
        port_ret_pct = round(port_ret * 100, 1)
        port_vol_pct = round(port_vol * 100, 1)
        allocations = []
        for sym, w in sorted(weights.items(), key=lambda x: -x[1]):
            alloc = round(amount * w, 2)
            r_pct = round(float(ann_returns.get(sym, 0)) * 100, 1)
            v_pct = round(float(ann_vol.get(sym, 0)) * 100, 1)
            sh = round((float(ann_returns.get(sym, 0)) - rf_rate) / (float(ann_vol.get(sym, 0.01)) + 1e-9), 2)
            last_price = float(raw[sym].iloc[-1]) if sym in raw.columns else 0
            shares = int(alloc / last_price) if last_price > 0 else 0
            beta_approx = round(float(returns[sym].cov(returns.mean(axis=1))) / float(returns.mean(axis=1).var() + 1e-9), 2) if sym in returns.columns else 1.0
            allocations.append({"symbol": sym.replace(".NS", ""), "full_symbol": sym, "weight_pct": round(w * 100, 1),
                                "amount": alloc, "exp_return": r_pct, "volatility": v_pct, "sharpe": sh,
                                "last_price": round(last_price, 2), "shares": shares, "beta": beta_approx})
        projected_value = round(amount * ((1 + port_ret) ** horizon_yrs), 2)
        projected_gain = round(projected_value - amount, 2)
        gain_pct = round((projected_gain / amount) * 100, 1)
        actual_risk = "Low" if port_vol_pct < 15 else "Medium" if port_vol_pct < 25 else "High"
        sector_map = {"RELIANCE.NS": "Energy", "INFY.NS": "IT", "TCS.NS": "IT", "HDFCBANK.NS": "Finance",
                      "ICICIBANK.NS": "Finance", "ITC.NS": "FMCG", "WIPRO.NS": "IT", "LT.NS": "Infra",
                      "SBIN.NS": "Finance", "AXISBANK.NS": "Finance", "BAJFINANCE.NS": "Finance",
                      "HINDUNILVR.NS": "FMCG", "TATAMOTORS.NS": "Auto", "SUNPHARMA.NS": "Pharma",
                      "ONGC.NS": "Energy", "AAPL": "Tech", "MSFT": "Tech", "GOOGL": "Tech", "AMZN": "Tech",
                      "NVDA": "Tech", "JPM": "Finance", "JNJ": "Healthcare", "V": "Finance", "PG": "Consumer",
                      "UNH": "Healthcare", "META": "Tech", "TSLA": "Auto", "BRK-B": "Finance", "HD": "Consumer", "XOM": "Energy"}
        sector_alloc = {}
        for a in allocations:
            sec = sector_map.get(a["full_symbol"], "Other")
            sector_alloc[sec] = round(sector_alloc.get(sec, 0) + a["weight_pct"], 1)
        top3 = [a["symbol"] for a in allocations[:3]]
        reasoning = (f"Portfolio optimized using Markowitz Mean-Variance with Sharpe Ratio maximization. "
                     f"Analyzed {len(available)} stocks over {period} of real market data. "
                     f"Top holdings — {', '.join(top3)} — selected for highest risk-adjusted returns. "
                     f"Diversified across {len(sector_alloc)} sector(s): "
                     f"{', '.join(f'{k} {v}%' for k, v in sorted(sector_alloc.items(), key=lambda x: -x[1])[:4])}. "
                     f"Risk-free rate: {rf_rate * 100:.1f}%. Rebalancing recommended every {'quarter' if horizon_yrs <= 1 else '6 months'}.")
        return json.dumps({"success": True, "allocations": allocations, "portfolio_return": port_ret_pct,
                           "portfolio_vol": port_vol_pct, "sharpe_ratio": sharpe, "projected_value": projected_value,
                           "projected_gain": projected_gain, "gain_pct": gain_pct, "actual_risk": actual_risk,
                           "sector_alloc": sector_alloc, "reasoning": reasoning, "stocks_analyzed": len(available),
                           "market": market, "amount": amount, "horizon": horizon_yrs})
    except ImportError:
        return json.dumps({"error": "Missing package. Run: pip install yfinance scipy"}), 500
    except Exception as e:
        return json.dumps({"error": str(e), "trace": traceback.format_exc()}), 500


# ==========================
# PERSONAL FINANCE TRACKER
# ==========================

@app.route("/finance-tracker")
def finance_tracker():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("finance_tracker.html", username=session.get("user", "User"))


@app.route("/api/finance-plan", methods=["POST"])
def api_finance_plan():
    if "user" not in session:
        return jsonify({"error": "Not logged in"}), 401
    try:
        data = request.get_json()

        salary = float(data.get("salary", 0))
        expenses = float(data.get("expenses", 0))
        savings_goal = float(data.get("savings_goal", 0))
        risk_level = data.get("risk", "medium").lower()
        age = int(data.get("age", 25))
        dependents = int(data.get("dependents", 0))
        has_loan = bool(data.get("has_loan", False))
        loan_emi = float(data.get("loan_emi", 0))
        goal = data.get("goal", "wealth").lower()

        if salary <= 0:
            return jsonify({"error": "Salary must be greater than 0"}), 400

        total_deductions = expenses + (loan_emi if has_loan else 0)
        disposable = salary - total_deductions
        investable = max(0, disposable - savings_goal)

        emergency_fund_needed = expenses * (6 if dependents > 0 else 3)

        base_equity = max(20, min(80, 100 - age))
        if risk_level == "low":
            base_equity = max(20, base_equity - 20)
        if risk_level == "high":
            base_equity = min(80, base_equity + 15)

        debt_ratio = 100 - base_equity

        if investable <= 0:
            investment_plan = []
            monthly_invest = 0
        else:
            equity_amt = round(investable * base_equity / 100, 2)
            debt_amt = round(investable * debt_ratio / 100, 2)

            if risk_level == "low":
                mf_pct, idx_pct, stock_pct = 60, 35, 5
            elif risk_level == "high":
                mf_pct, idx_pct, stock_pct = 30, 30, 40
            else:
                mf_pct, idx_pct, stock_pct = 50, 35, 15

            gold_pct = 10 if risk_level == "low" else 5
            gold_adj = investable * gold_pct / 100
            equity_adj = equity_amt - gold_adj if equity_amt > gold_adj else equity_amt
            debt_adj = debt_amt

            mf_amt = round(equity_adj * mf_pct / 100, 2)
            idx_amt = round(equity_adj * idx_pct / 100, 2)
            stock_amt = round(equity_adj * stock_pct / 100, 2)

            investment_plan = [
                {"name": "Mutual Funds (SIP)", "amount": mf_amt, "pct": round(mf_amt / investable * 100, 1),
                 "color": "#00ff88", "icon": "chart-pie", "description": "Diversified equity mutual funds via SIP.",
                 "risk": "Medium", "horizon": "3–10 years"},
                {"name": "Index Funds / ETFs", "amount": idx_amt, "pct": round(idx_amt / investable * 100, 1),
                 "color": "#00d4ff", "icon": "arrow-trend-up", "description": "Nifty 50 / Sensex index funds.",
                 "risk": "Medium", "horizon": "5+ years"},
                {"name": "Direct Stocks / Trading", "amount": stock_amt, "pct": round(stock_amt / investable * 100, 1),
                 "color": "#ffaa00", "icon": "bolt", "description": "Direct equity trading using AI tools.",
                 "risk": "High", "horizon": "1–5 years"},
                {"name": "Debt / Bonds / FD", "amount": debt_adj, "pct": round(debt_adj / investable * 100, 1),
                 "color": "#9d4edd", "icon": "shield-halved", "description": "Debt funds, bonds, or FDs for stability.",
                 "risk": "Low", "horizon": "1–3 years"},
                {"name": "Gold / Sovereign Bonds", "amount": round(gold_adj, 2), "pct": round(gold_adj / investable * 100, 1),
                 "color": "#ffd700", "icon": "coins", "description": "Hedge against inflation.",
                 "risk": "Low", "horizon": "3–7 years"},
            ]
            monthly_invest = investable

        health_score = 100
        if total_deductions > salary * 0.7:
            health_score -= 30
        if investable <= 0:
            health_score -= 40
        if dependents > 0 and investable < emergency_fund_needed / 12:
            health_score -= 15
        if has_loan and loan_emi > salary * 0.4:
            health_score -= 15
        health_score = max(0, min(100, health_score))

        if health_score >= 75:
            health_label = "Excellent 🟢"
        elif health_score >= 50:
            health_label = "Good 🟡"
        elif health_score >= 30:
            health_label = "Fair 🟠"
        else:
            health_label = "Needs Work 🔴"

        def project(monthly, rate_annual, years):
            r = rate_annual / 12 / 100
            n = years * 12
            if r == 0:
                return round(monthly * n, 0)
            return round(monthly * ((1 + r) ** n - 1) / r * (1 + r), 0)

        avg_return = {"low": 8, "medium": 12, "high": 15}.get(risk_level, 12)

        projections = [
            {"years": 1, "value": project(monthly_invest, avg_return, 1)},
            {"years": 3, "value": project(monthly_invest, avg_return, 3)},
            {"years": 5, "value": project(monthly_invest, avg_return, 5)},
            {"years": 10, "value": project(monthly_invest, avg_return, 10)},
        ] if monthly_invest > 0 else []

        tips = []
        if investable <= 0:
            tips.append("⚠️ Your expenses exceed your disposable income. Focus on reducing monthly costs first.")
        if loan_emi > salary * 0.4:
            tips.append("💳 Your EMI is above 40% of salary — try to prepay high-interest loans.")
        if stock_amt > 0 and stock_amt > investable * 0.3:
            tips.append("⚡ You're allocating >30% to direct stocks. Make sure to use stop-losses.")
        if age < 30:
            tips.append("🚀 You're young — maximise equity exposure now. Time is your biggest asset.")
        if age > 50:
            tips.append("🛡️ Shift gradually to debt/bonds to protect your corpus as retirement nears.")
        if dependents > 0:
            tips.append(f"👨‍👩‍👧 With {dependents} dependent(s), ensure ₹{emergency_fund_needed:,.0f} emergency fund first.")
        if monthly_invest > 0:
            tips.append(f"📈 At {avg_return}% avg returns, ₹{monthly_invest:,.0f}/month grows to ₹{project(monthly_invest, avg_return, 10):,.0f} in 10 years.")

        return jsonify({
            "salary": salary,
            "expenses": expenses + (loan_emi if has_loan else 0),
            "disposable": disposable,
            "monthly_invest": round(monthly_invest, 2),
            "savings_reserve": savings_goal,
            "investment_plan": investment_plan,
            "health_score": health_score,
            "health_label": health_label,
            "emergency_fund_needed": emergency_fund_needed,
            "equity_pct": base_equity,
            "debt_pct": debt_ratio,
            "avg_return": avg_return,
            "projections": projections,
            "tips": tips,
            "risk": risk_level,
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# Add this to app.py for real-time username availability check

@app.route("/api/check-username", methods=["GET"])
def check_username():
    """Check if username is available for registration"""
    username = request.args.get("username", "").strip()
    
    if not username or len(username) < 3:
        return jsonify({"available": False, "message": "Username too short"})
    
    try:
        conn = sqlite3.connect("data/users.db")
        c = conn.cursor()
        c.execute("SELECT id FROM users WHERE username = ?", (username,))
        exists = c.fetchone() is not None
        conn.close()
        
        return jsonify({"available": not exists, "message": "Username available" if not exists else "Username taken"})
    except Exception as e:
        print(f"Username check error: {e}")
        return jsonify({"available": True, "message": "Error checking username"})
    
# Add to app.py - Memory enhancement endpoints

@app.route("/api/daily-tip", methods=["GET"])
def get_daily_tip():
    """Get a daily trading tip"""
    if "user" not in session:
        return jsonify({"error": "Not logged in"}), 401
    
    tip = rag_chatbot.generate_daily_tip()
    return jsonify({"tip": tip})

@app.route("/api/market-summary", methods=["GET"])
def get_market_summary():
    """Get personalized market summary"""
    if "user" not in session:
        return jsonify({"error": "Not logged in"}), 401
    
    username = session.get("user")
    summary = rag_chatbot.generate_market_summary(username)
    return jsonify({"summary": summary})

@app.route("/api/suggest-questions", methods=["GET"])
def suggest_questions():
    """Get suggested follow-up questions"""
    if "user" not in session:
        return jsonify({"error": "Not logged in"}), 401
    
    username = session.get("user")
    conversation_history = rag_chatbot.db.get_conversation_summary(username, limit=5)
    suggestions = rag_chatbot.suggest_follow_up_questions(conversation_history)
    return jsonify({"suggestions": suggestions})

@app.route("/api/user-preferences", methods=["GET", "POST"])
def user_preferences():
    """Get or update user preferences"""
    if "user" not in session:
        return jsonify({"error": "Not logged in"}), 401
    
    username = session.get("user")
    
    if request.method == "GET":
        prefs = rag_chatbot.db.get_user_preferences(username)
        return jsonify(prefs)
    
    else:
        data = request.get_json()
        rag_chatbot.db.update_user_preferences(username, data)
        return jsonify({"success": True, "message": "Preferences updated"})
    
# ==========================
# CHOOSE ANALYSIS PAGE - NEW ROUTE NAME
# ==========================

# ==========================
# CHOOSE ANALYSIS PAGE
# ==========================

@app.route("/choose-analysis")
def choose_analysis():
    """Page to choose between Basic Charts and Advanced Analysis"""
    if "user" not in session:
        return redirect(url_for("login"))
    
    selected_column = session.get("advanced_selected_column", "Not selected")
    rows = session.get("dataset_rows", 0)
    cols = session.get("dataset_cols", 0)
    username = session.get("user", "User")
    
    return render_template("analysis_choice.html", 
                         selected_column=selected_column,
                         rows=rows,
                         cols=cols,
                         username=username)
# Add this route in app.py

@app.route("/notifications")
def notifications_page():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("notifications.html", username=session.get("user"))
# ==========================
# RUN
# ==========================

if __name__ == "__main__":
    app.run(debug=DEBUG, host=HOST, port=PORT)