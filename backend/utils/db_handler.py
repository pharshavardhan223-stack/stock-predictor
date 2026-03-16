"""
NEW FILE → Place at: backend/utils/db_handler.py
Replaces data/history.csv with proper SQLite database.
Your old CSV code still works - this is additive only.
"""

import sqlite3
import os
import datetime

DB_PATH = "data/stockai.db"


# ==========================
# INIT ALL TABLES
# ==========================

def init_db():
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Prediction history
    c.execute("""
        CREATE TABLE IF NOT EXISTS prediction_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            stock TEXT NOT NULL,
            action TEXT,
            confidence REAL,
            last_price REAL,
            predicted_price REAL,
            risk TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Watchlist
    c.execute("""
        CREATE TABLE IF NOT EXISTS watchlist (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            symbol TEXT NOT NULL,
            added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(username, symbol)
        )
    """)

    # Price alerts
    c.execute("""
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            symbol TEXT NOT NULL,
            target_price REAL NOT NULL,
            direction TEXT NOT NULL,
            triggered INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()


# ==========================
# PREDICTION HISTORY
# ==========================

def save_prediction(username, stock, action, confidence, last_price=0, predicted_price=0, risk=""):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        INSERT INTO prediction_history
        (username, stock, action, confidence, last_price, predicted_price, risk)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (username, stock, action, confidence, last_price, predicted_price, risk))
    conn.commit()
    conn.close()


def get_user_history(username, limit=50):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        SELECT stock, action, confidence, last_price, predicted_price, risk, created_at
        FROM prediction_history
        WHERE username = ?
        ORDER BY created_at DESC
        LIMIT ?
    """, (username, limit))
    rows = c.fetchall()
    conn.close()

    return [
        {
            "stock":           r[0],
            "action":          r[1],
            "confidence":      r[2],
            "last_price":      r[3],
            "predicted_price": r[4],
            "risk":            r[5],
            "date":            r[6]
        }
        for r in rows
    ]


def get_user_stats(username):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("SELECT COUNT(*) FROM prediction_history WHERE username = ?", (username,))
    total = c.fetchone()[0]

    c.execute("""
        SELECT action, COUNT(*) as cnt
        FROM prediction_history
        WHERE username = ?
        GROUP BY action
    """, (username,))
    action_counts = dict(c.fetchall())

    c.execute("""
        SELECT created_at FROM prediction_history
        WHERE username = ?
        ORDER BY created_at DESC LIMIT 1
    """, (username,))
    last = c.fetchone()
    last_activity = last[0] if last else "N/A"

    conn.close()

    return {
        "total":         total,
        "buy_count":     action_counts.get("BUY", 0),
        "sell_count":    action_counts.get("SELL", 0),
        "hold_count":    action_counts.get("HOLD", 0),
        "last_activity": last_activity
    }


# ==========================
# WATCHLIST
# ==========================

def add_to_watchlist(username, symbol):
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute(
            "INSERT OR IGNORE INTO watchlist (username, symbol) VALUES (?, ?)",
            (username, symbol.upper())
        )
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print("Watchlist error:", e)
        return False


def get_watchlist(username):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "SELECT symbol, added_at FROM watchlist WHERE username = ? ORDER BY added_at DESC",
        (username,)
    )
    rows = c.fetchall()
    conn.close()
    return [{"symbol": r[0], "added_at": r[1]} for r in rows]


def remove_from_watchlist(username, symbol):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "DELETE FROM watchlist WHERE username = ? AND symbol = ?",
        (username, symbol.upper())
    )
    conn.commit()
    conn.close()


# ==========================
# PRICE ALERTS
# ==========================

def add_alert(username, symbol, target_price, direction):
    """direction: 'above' or 'below'"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        INSERT INTO alerts (username, symbol, target_price, direction)
        VALUES (?, ?, ?, ?)
    """, (username, symbol.upper(), target_price, direction))
    conn.commit()
    conn.close()


def get_user_alerts(username):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        SELECT id, symbol, target_price, direction, triggered, created_at
        FROM alerts
        WHERE username = ? AND triggered = 0
        ORDER BY created_at DESC
    """, (username,))
    rows = c.fetchall()
    conn.close()
    return [
        {
            "id":           r[0],
            "symbol":       r[1],
            "target_price": r[2],
            "direction":    r[3],
            "triggered":    r[4],
            "created_at":   r[5]
        }
        for r in rows
    ]




def remove_alert(username, alert_id):
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()
    c.execute(
        "DELETE FROM alerts WHERE id = ? AND username = ?",
        (int(alert_id), username)
    )
    conn.commit()
    conn.close()


def get_last_train_info():
    try:
        conn = sqlite3.connect(DB_PATH)
        c    = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS train_log (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                trained_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                samples    INTEGER,
                symbols    TEXT,
                status     TEXT
            )
        """)
        conn.commit()
        c.execute("SELECT trained_at, samples, symbols, status FROM train_log ORDER BY id DESC LIMIT 1")
        row = c.fetchone()
        conn.close()
        if row:
            return {"trained_at": row[0], "samples": row[1], "symbols": row[2], "status": row[3]}
    except Exception:
        pass
    return {"trained_at": "Never", "samples": 0, "symbols": "", "status": "N/A"}

# Auto-init when imported
init_db()