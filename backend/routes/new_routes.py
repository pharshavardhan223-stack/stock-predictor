"""
backend/routes/new_routes.py  ← REPLACE YOUR EXISTING FILE WITH THIS

CHANGES IN THIS VERSION:
  ✅ Admin registration requires a secret passkey (ADMIN_SECRET_KEY)
  ✅ Passkey is validated server-side before saving as admin
  ✅ Wrong passkey → saved as 'user' role with error flash
  ✅ All previous register bugs fixed (role saved, first_name handled, real errors printed)
  ✅ All watchlist / alerts / /api/user-stats routes intact
"""

from flask import request, redirect, url_for, session, flash, render_template, jsonify
import sqlite3, os, yfinance as yf

DB_PATH = "data/users.db"

# ── Admin passkey: change this to your own secret ──────────────────────────
# You can also put this in config.py or .env as ADMIN_SECRET_KEY
ADMIN_SECRET_KEY = "STOCKAI@ADMIN2024"


def _ensure_users_table():
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS users (
        id       INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT    UNIQUE NOT NULL,
        password TEXT    NOT NULL,
        email    TEXT,
        role     TEXT    DEFAULT 'user'
    )""")
    conn.commit()
    conn.close()


def _get_watchlist(username):
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT symbol, added_at FROM watchlist WHERE username=? ORDER BY added_at DESC",
                  (username,))
        rows = c.fetchall()
        conn.close()
        return [{"symbol": r[0], "added_at": r[1]} for r in rows]
    except Exception as e:
        print(f"[new_routes] get_watchlist error: {e}")
        return []


def _get_alerts(username):
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("""SELECT id, symbol, target_price, direction, created_at
                     FROM alerts WHERE username=? ORDER BY created_at DESC""",
                  (username,))
        rows = c.fetchall()
        conn.close()
        return [{"id":r[0],"symbol":r[1],"target_price":r[2],
                 "direction":r[3],"created_at":r[4]} for r in rows]
    except Exception as e:
        print(f"[new_routes] get_alerts error: {e}")
        return []


def register_new_routes(app):

    # ── /watchlist ──────────────────────────────────────────────────
    @app.route("/watchlist")
    def watchlist():
        if "user" not in session:
            return redirect(url_for("login"))
        username = session["user"]
        wl      = _get_watchlist(username)
        alerts  = _get_alerts(username)
        return render_template("watchlist.html", watchlist=wl, alerts=alerts)

    @app.route("/watchlist/add", methods=["POST"])
    def watchlist_add():
        if "user" not in session:
            return redirect(url_for("login"))
        symbol = request.form.get("symbol", "").strip().upper()
        if not symbol:
            flash("Please enter a stock symbol.", "error")
            return redirect(url_for("watchlist"))
        clean = symbol.replace(".", "").replace("-", "")
        if len(symbol) > 15 or not clean.isalnum():
            flash(f"Invalid symbol: {symbol}", "error")
            return redirect(url_for("watchlist"))
        try:
            hist = yf.Ticker(symbol).history(period="1d")
            if hist.empty:
                flash(f"Symbol '{symbol}' not found. Use .NS for NSE e.g. INFY.NS", "error")
                return redirect(url_for("watchlist"))
        except Exception:
            flash("Could not verify symbol. Try again.", "error")
            return redirect(url_for("watchlist"))
        try:
            conn = sqlite3.connect(DB_PATH)
            c    = conn.cursor()
            c.execute("INSERT OR IGNORE INTO watchlist (username, symbol) VALUES (?,?)",
                      (session["user"], symbol))
            conn.commit()
            conn.close()
            flash(f"✓ {symbol} added to watchlist", "success")
        except Exception as e:
            print(f"[new_routes] watchlist_add DB error: {e}")
            flash("Could not add symbol.", "error")
        return redirect(url_for("watchlist"))

    @app.route("/watchlist/remove", methods=["POST"])
    def watchlist_remove():
        if "user" not in session:
            return redirect(url_for("login"))
        symbol = request.form.get("symbol", "").strip().upper()
        try:
            conn = sqlite3.connect(DB_PATH)
            c    = conn.cursor()
            c.execute("DELETE FROM watchlist WHERE username=? AND symbol=?",
                      (session["user"], symbol))
            conn.commit()
            conn.close()
            flash(f"✓ {symbol} removed", "success")
        except Exception as e:
            print(f"[new_routes] watchlist_remove error: {e}")
        return redirect(url_for("watchlist"))

    # ── /alerts ─────────────────────────────────────────────────────
    @app.route("/alerts/add", methods=["POST"])
    def alert_add():
        if "user" not in session:
            return redirect(url_for("login"))
        symbol       = request.form.get("symbol", "").strip().upper()
        target_price = request.form.get("target_price", "")
        direction    = request.form.get("direction", "above")
        if not symbol or not target_price:
            flash("Please fill all alert fields.", "error")
            return redirect(url_for("watchlist"))
        try:
            target_price = float(target_price)
        except ValueError:
            flash("Invalid price value.", "error")
            return redirect(url_for("watchlist"))
        if direction not in ("above", "below"):
            direction = "above"
        try:
            conn = sqlite3.connect(DB_PATH)
            c    = conn.cursor()
            c.execute("""INSERT INTO alerts (username, symbol, target_price, direction)
                         VALUES (?,?,?,?)""",
                      (session["user"], symbol, target_price, direction))
            conn.commit()
            conn.close()
            flash(f"✓ Alert set for {symbol} @ {target_price}", "success")
        except Exception as e:
            print(f"[new_routes] alert_add error: {e}")
            flash("Could not set alert.", "error")
        return redirect(url_for("watchlist"))

    @app.route("/alerts/remove", methods=["POST"])
    def alert_remove():
        if "user" not in session:
            return redirect(url_for("login"))
        alert_id = request.form.get("alert_id", "")
        try:
            conn = sqlite3.connect(DB_PATH)
            c    = conn.cursor()
            c.execute("DELETE FROM alerts WHERE id=? AND username=?",
                      (alert_id, session["user"]))
            conn.commit()
            conn.close()
            flash("✓ Alert removed", "success")
        except Exception as e:
            print(f"[new_routes] alert_remove error: {e}")
        return redirect(url_for("watchlist"))

    # ── /register ──────────────────────────────────────────────────
    @app.route("/register", methods=["GET", "POST"])
    def register():
        if "user" in session:
            return redirect(url_for("dashboard"))

        if request.method == "POST":
            from werkzeug.security import generate_password_hash

            username         = request.form.get("username", "").strip()
            password         = request.form.get("password", "")
            confirm_password = request.form.get("confirm_password", "")
            email            = request.form.get("email", "").strip()
            role             = request.form.get("role", "user").strip()
            first_name       = request.form.get("first_name", "").strip()

            # Admin passkey field (only required when role == "admin")
            admin_key        = request.form.get("admin_passkey", "").strip()

            # ── Validation ─────────────────────────────────────────
            if not username or not password:
                flash("Username and password are required.", "error")
                return render_template("register.html")

            if len(username) < 3:
                flash("Username must be at least 3 characters.", "error")
                return render_template("register.html")

            if not all(ch.isalnum() or ch == "_" for ch in username):
                flash("Username: only letters, numbers and underscore allowed.", "error")
                return render_template("register.html")

            if len(password) < 6:
                flash("Password must be at least 6 characters.", "error")
                return render_template("register.html")

            if password != confirm_password:
                flash("Passwords do not match.", "error")
                return render_template("register.html")

            if role not in ("user", "admin"):
                role = "user"

            # ── Admin passkey check ────────────────────────────────
            if role == "admin":
                if not admin_key:
                    flash("Admin registration requires a security passkey.", "error")
                    return render_template("register.html")
                if admin_key != ADMIN_SECRET_KEY:
                    flash("Invalid admin security key. Your account has been created as a standard user instead.", "error")
                    role = "user"   # demote silently to user

            # ── Save to DB ─────────────────────────────────────────
            try:
                _ensure_users_table()
                conn = sqlite3.connect(DB_PATH)
                c    = conn.cursor()
                hashed = generate_password_hash(password)
                c.execute(
                    "INSERT INTO users (username, password, email, role) VALUES (?,?,?,?)",
                    (username, hashed, email, role)
                )
                conn.commit()
                conn.close()
                display = first_name if first_name else username
                role_label = "Admin" if role == "admin" else "User"
                flash(f"Account created! Welcome, {display}. You joined as {role_label}. Please sign in.", "success")
                return redirect(url_for("login"))

            except sqlite3.IntegrityError:
                flash("Username already taken. Please choose another.", "error")
                return render_template("register.html")

            except Exception as e:
                import traceback
                print(f"[new_routes] register EXCEPTION: {type(e).__name__}: {e}")
                traceback.print_exc()
                flash(f"Registration failed: {type(e).__name__} — {e}", "error")
                return render_template("register.html")

        return render_template("register.html")

    # ── /api/user-stats ────────────────────────────────────────────
    @app.route("/api/user-stats")
    def user_stats_api():
        if "user" not in session:
            return jsonify({"error": "Not logged in"}), 401
        try:
            from backend.utils.db_handler import get_user_stats
            stats = get_user_stats(session["user"])
            return jsonify(stats)
        except Exception as e:
            return jsonify({"total": 0, "buy_count": 0, "sell_count": 0, "hold_count": 0})

    print("✅ new_routes registered: /watchlist /alerts /register /api/user-stats")