"""
NEW FILE → Place at: backend/routes/new_routes.py

How to use:
1. Keep your old app.py exactly as it is
2. Add these 3 lines at the bottom of your app.py (before if __name__ == "__main__"):

    from backend.routes.new_routes import register_new_routes
    register_new_routes(app)

That's it! All new features will be added automatically.
"""

from flask import request, redirect, url_for, session, flash, render_template, jsonify
from backend.utils.db_handler import (
    add_to_watchlist, get_watchlist, remove_from_watchlist,
    add_alert, get_user_alerts, remove_alert,
    save_prediction, get_user_stats
)
import yfinance as yf


def register_new_routes(app):
    """
    Call this function in your app.py to register all new routes.
    """

    # ==============================
    # WATCHLIST ROUTES
    # ==============================

    @app.route("/watchlist")
    def watchlist():
        if "user" not in session:
            return redirect(url_for("login"))

        username = session["user"]
        wl       = get_watchlist(username)
        alerts   = get_user_alerts(username)

        return render_template("watchlist.html", watchlist=wl, alerts=alerts)


    @app.route("/watchlist/add", methods=["POST"])
    def watchlist_add():
        if "user" not in session:
            return redirect(url_for("login"))

        symbol = request.form.get("symbol", "").strip().upper()

        import re
        if not symbol or len(symbol) > 15 or not re.match(r'^[A-Z0-9.]{1,15}$', symbol):
            flash("Invalid stock symbol (use A-Z, 0-9, dot only)", "error")
            return redirect(url_for("watchlist"))

        # Validate symbol exists
        try:
            ticker = yf.Ticker(symbol)
            info   = ticker.history(period="1d")
            if info.empty:
                flash(f"Symbol '{symbol}' not found", "error")
                return redirect(url_for("watchlist"))
        except Exception:
            flash("Could not verify symbol", "error")
            return redirect(url_for("watchlist"))

        success = add_to_watchlist(session["user"], symbol)

        if success:
            flash(f"✓ {symbol} added to watchlist", "success")
        else:
            flash(f"{symbol} is already in your watchlist", "error")

        return redirect(url_for("watchlist"))


    @app.route("/watchlist/remove", methods=["POST"])
    def watchlist_remove():
        if "user" not in session:
            return redirect(url_for("login"))

        symbol = request.form.get("symbol", "").strip().upper()
        remove_from_watchlist(session["user"], symbol)
        flash(f"✓ {symbol} removed from watchlist", "success")

        return redirect(url_for("watchlist"))


    # ==============================
    # ALERT ROUTES
    # ==============================

    @app.route("/alerts/add", methods=["POST"])
    def alert_add():
        if "user" not in session:
            return redirect(url_for("login"))

        symbol       = request.form.get("symbol", "").strip().upper()
        target_price = request.form.get("target_price", "")
        direction    = request.form.get("direction", "above")

        if not symbol or not target_price:
            flash("Please fill all alert fields", "error")
            return redirect(url_for("watchlist"))

        try:
            target_price = float(target_price)
        except ValueError:
            flash("Invalid price value", "error")
            return redirect(url_for("watchlist"))

        add_alert(session["user"], symbol, target_price, direction)
        flash(f"✓ Alert set for {symbol} @ ${target_price}", "success")

        return redirect(url_for("watchlist"))


    @app.route("/alerts/remove", methods=["POST"])
    def alert_remove():
        if "user" not in session:
            return redirect(url_for("login"))

        alert_id = request.form.get("alert_id", "")
        try:
            alert_id = int(alert_id)
        except (ValueError, TypeError):
            flash("Invalid alert ID", "error")
            return redirect(url_for("watchlist"))

        remove_alert(session["user"], alert_id)
        flash("✓ Alert deleted", "success")
        return redirect(url_for("watchlist"))


    # ==============================
    # REGISTER ROUTE
    # ==============================

    @app.route("/register", methods=["GET", "POST"])
    def register():
        if "user" in session:
            return redirect(url_for("dashboard"))

        if request.method == "POST":
            import sqlite3
            from werkzeug.security import generate_password_hash

            username         = request.form.get("username", "").strip()
            password         = request.form.get("password", "")
            confirm_password = request.form.get("confirm_password", "")
            email            = request.form.get("email", "").strip()

            # Validation
            if not username or not password:
                flash("Username and password are required", "error")
                return render_template("register.html")

            if len(username) < 3:
                flash("Username must be at least 3 characters", "error")
                return render_template("register.html")

            if len(password) < 6:
                flash("Password must be at least 6 characters", "error")
                return render_template("register.html")

            if password != confirm_password:
                flash("Passwords do not match", "error")
                return render_template("register.html")

            try:
                import sqlite3
                DB_PATH = "data/users.db"
                import os
                os.makedirs("data", exist_ok=True)

                conn = sqlite3.connect(DB_PATH)
                c    = conn.cursor()

                c.execute("""
                    CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        username TEXT UNIQUE NOT NULL,
                        password TEXT NOT NULL,
                        email TEXT,
                        role TEXT DEFAULT 'user'
                    )
                """)

                c.execute(
                    "INSERT INTO users (username, password, email) VALUES (?, ?, ?)",
                    (username, generate_password_hash(password), email)
                )
                conn.commit()
                conn.close()

                flash("Account created successfully! Please login.", "success")
                return redirect(url_for("login"))

            except sqlite3.IntegrityError:
                flash("Username already exists. Please choose another.", "error")

        return render_template("register.html")


    # ==============================
    # ENHANCED STATS API
    # ==============================

    @app.route("/api/user-stats")
    def user_stats_api():
        if "user" not in session:
            return jsonify({"error": "Not logged in"}), 401

        stats = get_user_stats(session["user"])

        # ── Fallback: also count rows in history.csv so old predictions show up ──
        # New predictions go to both CSV + SQLite, but existing ones may only be in CSV
        try:
            import os, csv as _csv
            csv_path = "data/history.csv"
            csv_count = 0
            if os.path.exists(csv_path):
                with open(csv_path, newline="") as f:
                    reader = _csv.DictReader(f)
                    for row in reader:
                        if row.get("username") == session["user"]:
                            csv_count += 1
            # Use whichever count is higher (CSV may include older records)
            if csv_count > stats.get("total", 0):
                stats["total"] = csv_count
        except Exception as _e:
            pass  # if CSV read fails, SQLite count is still returned

        return jsonify(stats)


    @app.route("/api/watchlist-prices")
    def watchlist_prices_api():
        if "user" not in session:
            return jsonify({"error": "Not logged in"}), 401

        wl     = get_watchlist(session["user"])
        result = {}

        for item in wl:
            sym = item["symbol"]
            try:
                ticker = yf.Ticker(sym)
                data   = ticker.history(period="2d")
                if not data.empty and len(data) >= 2:
                    curr  = round(float(data["Close"].iloc[-1]), 2)
                    prev  = round(float(data["Close"].iloc[-2]), 2)
                    chg   = round(((curr - prev) / prev) * 100, 2)
                    result[sym] = {"price": curr, "change": chg}
                elif not data.empty:
                    result[sym] = {"price": round(float(data["Close"].iloc[-1]), 2), "change": 0}
            except Exception:
                result[sym] = {"price": "N/A", "change": 0}

        return jsonify(result)

    print("✅ New routes registered: /watchlist, /register, /alerts")