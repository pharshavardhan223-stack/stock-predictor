"""
FILE → Place at: backend/utils/email_alerts.py

Checks all un-triggered price alerts every 5 minutes.
Sends an email when a stock crosses its target price.
Runs in a background thread — never blocks Flask.

─── SETUP ────────────────────────────────────────────────────────────────────
1. Install Flask-Mail:
       pip install Flask-Mail

2. Add to your config.py (or .env):
       MAIL_SERVER   = "smtp.gmail.com"
       MAIL_PORT     = 587
       MAIL_USERNAME = "your@gmail.com"
       MAIL_PASSWORD = "your-app-password"   ← Gmail App Password, not login pw
       MAIL_USE_TLS  = True
       ALERT_FROM    = "your@gmail.com"

   For Gmail: https://support.google.com/accounts/answer/185833
   (Generate an App Password under Security → 2-Step Verification)

3. Add to app.py (after app is created):
       from backend.utils.email_alerts import init_mail, start_alert_checker
       init_mail(app)
       start_alert_checker()
──────────────────────────────────────────────────────────────────────────────
"""

import os
import time
import threading
import datetime
import sqlite3
import yfinance as yf

# ── DB path (same as db_handler.py) ──────────────────────────────────────────
DB_PATH = "data/stockai.db"

# ── Flask-Mail instance (set by init_mail) ────────────────────────────────────
_mail       = None
_app        = None
_mail_from  = None
_check_secs = 300   # check every 5 minutes


# ── Init ──────────────────────────────────────────────────────────────────────
def init_mail(app):
    """
    Call once in app.py after the Flask app is created.
    Configures Flask-Mail using app.config values.
    """
    global _mail, _app, _mail_from

    try:
        from flask_mail import Mail

        app.config.setdefault("MAIL_SERVER",   os.getenv("MAIL_SERVER",   "smtp.gmail.com"))
        app.config.setdefault("MAIL_PORT",     int(os.getenv("MAIL_PORT", 587)))
        app.config.setdefault("MAIL_USE_TLS",  True)
        app.config.setdefault("MAIL_USERNAME", os.getenv("MAIL_USERNAME", ""))
        app.config.setdefault("MAIL_PASSWORD", os.getenv("MAIL_PASSWORD", ""))
        app.config.setdefault("MAIL_DEFAULT_SENDER", os.getenv("MAIL_USERNAME", ""))

        _mail      = Mail(app)
        _app       = app
        _mail_from = os.getenv("ALERT_FROM", app.config.get("MAIL_USERNAME", ""))
        print("✅ Flask-Mail initialised")

    except ImportError:
        print("⚠️  Flask-Mail not installed — run: pip install Flask-Mail")
    except Exception as e:
        print(f"⚠️  Mail init error: {e}")


# ── Helpers ───────────────────────────────────────────────────────────────────
def _get_live_price(symbol: str) -> float | None:
    try:
        tk   = yf.Ticker(symbol)
        data = tk.history(period="1d")
        if data.empty:
            return None
        return round(float(data["Close"].iloc[-1]), 2)
    except Exception:
        return None


def _get_user_email(username: str) -> str | None:
    """Look up email from users.db."""
    try:
        conn = sqlite3.connect("data/users.db")
        c    = conn.cursor()
        c.execute("SELECT email FROM users WHERE username = ?", (username,))
        row = c.fetchone()
        conn.close()
        return row[0] if row and row[0] else None
    except Exception:
        return None


def _mark_triggered(alert_id: int):
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()
    c.execute("UPDATE alerts SET triggered = 1 WHERE id = ?", (alert_id,))
    conn.commit()
    conn.close()


def _log_sent(username: str, symbol: str, price: float, direction: str, target: float):
    """Record sent alert in DB for audit trail."""
    try:
        conn = sqlite3.connect(DB_PATH)
        c    = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS alert_sent_log (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                username   TEXT,
                symbol     TEXT,
                price      REAL,
                target     REAL,
                direction  TEXT,
                sent_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        c.execute(
            "INSERT INTO alert_sent_log (username, symbol, price, target, direction) VALUES (?,?,?,?,?)",
            (username, symbol, price, target, direction)
        )
        conn.commit()
        conn.close()
    except Exception:
        pass


# ── Email sending ─────────────────────────────────────────────────────────────
def _send_alert_email(to_email: str, username: str,
                      symbol: str, current_price: float,
                      target_price: float, direction: str):
    """Send a styled HTML alert email."""
    if _mail is None or _app is None:
        print(f"  ⚠ Mail not configured — skipping email to {to_email}")
        return False

    arrow     = "▲" if direction == "above" else "▼"
    color     = "#22c55e" if direction == "above" else "#f87171"
    cur_sym   = "₹" if (".NS" in symbol or ".BO" in symbol) else "$"
    subject   = f"🔔 Stock Alert: {symbol} {arrow} {cur_sym}{target_price}"

    html_body = f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <style>
    body      {{ font-family: 'Inter', Arial, sans-serif; background:#0f172a; color:#e5e7eb; margin:0; padding:0; }}
    .wrap     {{ max-width:520px; margin:30px auto; background:#0d1b2a; border:1px solid rgba(56,189,248,.2); border-radius:18px; overflow:hidden; }}
    .header   {{ background:linear-gradient(135deg,#0288d1,#0097a7); padding:28px 32px; text-align:center; }}
    .header h1{{ margin:0; font-size:22px; color:#fff; letter-spacing:1px; }}
    .header p {{ margin:6px 0 0; color:rgba(255,255,255,.8); font-size:13px; }}
    .body     {{ padding:28px 32px; }}
    .badge    {{ display:inline-block; background:{color}22; border:1px solid {color}55;
                 color:{color}; border-radius:8px; padding:6px 16px;
                 font-size:13px; font-weight:700; margin-bottom:20px; }}
    .price-row{{ display:flex; justify-content:space-between; margin:12px 0;
                 padding:14px 18px; background:rgba(255,255,255,.04);
                 border-radius:12px; border:1px solid rgba(255,255,255,.07); }}
    .price-row .label {{ font-size:12px; color:#64748b; text-transform:uppercase; letter-spacing:.5px; }}
    .price-row .value {{ font-size:20px; font-weight:800; color:#f1f5f9; font-family:monospace; }}
    .price-row .value.green {{ color:#22c55e; }}
    .price-row .value.red   {{ color:#f87171; }}
    .cta      {{ display:block; text-align:center; background:linear-gradient(135deg,#0288d1,#0097a7);
                 color:#fff; text-decoration:none; border-radius:12px;
                 padding:14px 28px; font-weight:700; font-size:15px; margin:24px 0 0; }}
    .footer   {{ text-align:center; padding:16px 32px; font-size:11px; color:#334155;
                 border-top:1px solid rgba(255,255,255,.06); }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="header">
      <h1>📈 Stock AI — Price Alert</h1>
      <p>Hi {username}, your alert has been triggered!</p>
    </div>
    <div class="body">
      <div class="badge">{arrow} {symbol} is now {direction.upper()} your target</div>

      <div class="price-row">
        <div><div class="label">Symbol</div><div class="value">{symbol}</div></div>
      </div>
      <div class="price-row">
        <div><div class="label">Current Price</div><div class="value {'green' if direction=='above' else 'red'}">{cur_sym}{current_price}</div></div>
        <div style="text-align:right"><div class="label">Your Target</div><div class="value">{cur_sym}{target_price}</div></div>
      </div>
      <div class="price-row">
        <div><div class="label">Condition</div><div class="value" style="font-size:14px;color:{color};">{arrow} Price went {direction} {cur_sym}{target_price}</div></div>
      </div>

      <a href="http://127.0.0.1:5000/watchlist" class="cta">View Watchlist →</a>
    </div>
    <div class="footer">
      Stock AI · Alert sent {datetime.datetime.now().strftime("%Y-%m-%d %H:%M")} ·
      <a href="http://127.0.0.1:5000/watchlist" style="color:#38bdf8;">Manage Alerts</a>
    </div>
  </div>
</body>
</html>
"""

    try:
        from flask_mail import Message
        with _app.app_context():
            msg = Message(
                subject    = subject,
                recipients = [to_email],
                html       = html_body,
                sender     = _mail_from
            )
            _mail.send(msg)
        print(f"  ✉ Alert email sent → {to_email} ({symbol} {direction} {cur_sym}{target_price})")
        return True
    except Exception as e:
        print(f"  ❌ Email send error: {e}")
        return False


# ── Checker loop ──────────────────────────────────────────────────────────────
def _check_alerts():
    """Runs forever in background thread, checking every 5 minutes."""
    print("🔔 Alert checker started (interval: 5 min)")

    while True:
        try:
            conn = sqlite3.connect(DB_PATH)
            c    = conn.cursor()
            c.execute("""
                SELECT id, username, symbol, target_price, direction
                FROM   alerts
                WHERE  triggered = 0
            """)
            pending = c.fetchall()
            conn.close()

            if pending:
                print(f"🔔 Checking {len(pending)} pending alert(s) …")

            for alert_id, username, symbol, target_price, direction in pending:
                price = _get_live_price(symbol)
                if price is None:
                    continue

                triggered = (
                    (direction == "above" and price >= target_price) or
                    (direction == "below" and price <= target_price)
                )

                if triggered:
                    print(f"  🚨 Alert #{alert_id}: {symbol} @ {price} — {direction} {target_price}")
                    email = _get_user_email(username)

                    if email:
                        sent = _send_alert_email(
                            to_email      = email,
                            username      = username,
                            symbol        = symbol,
                            current_price = price,
                            target_price  = target_price,
                            direction     = direction
                        )
                        if sent:
                            _log_sent(username, symbol, price, direction, target_price)
                    else:
                        print(f"  ⚠ No email on file for user '{username}' — alert marked triggered anyway")

                    _mark_triggered(alert_id)

        except Exception as e:
            print(f"  ❌ Alert checker error: {e}")

        time.sleep(_check_secs)


# ── Public entry-point ────────────────────────────────────────────────────────
def start_alert_checker():
    """
    Call once in app.py after init_mail().
    Launches a daemon thread that checks alerts every 5 minutes.
    """
    t = threading.Thread(target=_check_alerts, daemon=True, name="AlertChecker")
    t.start()