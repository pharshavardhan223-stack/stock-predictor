import sqlite3
import os
from datetime import datetime
from flask import Blueprint, request, session, jsonify

notifications_bp = Blueprint("notifications", __name__, url_prefix="/api/notifications")
DB_PATH = os.path.join("data", "users.db")

def get_user_id(username):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id FROM users WHERE username = ?", (username,))
    result = c.fetchone()
    conn.close()
    return result[0] if result else None

def add_notification(username, n_type, title, message, link):
    user_id = get_user_id(username)
    if not user_id:
        return False
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''INSERT INTO notifications (user_id, username, type, title, message, link, is_read, created_at)
                VALUES (?, ?, ?, ?, ?, ?, 0, ?)''',
              (user_id, username, n_type, title, message, link, datetime.now()))
    conn.commit()
    conn.close()
    return True

# ========== HELPER FUNCTIONS FOR APP.PY ==========
def add_price_alert_notification(username, symbol, price, target_price, direction):
    """Add notification for price alert"""
    if direction == "above":
        title = f"📈 Price Alert: {symbol}"
        message = f"{symbol} has crossed ₹{target_price} (Current: ₹{price})"
    else:
        title = f"📉 Price Alert: {symbol}"
        message = f"{symbol} has dropped below ₹{target_price} (Current: ₹{price})"
    
    link = f"/live?symbol={symbol}"
    return add_notification(username, "price_alert", title, message, link)

def add_prediction_notification(username, symbol, action, confidence):
    """Add notification when prediction is complete"""
    title = f"🔮 Prediction Ready: {symbol}"
    message = f"AI predicts {action} with {confidence}% confidence"
    link = "/predictions"
    return add_notification(username, "prediction", title, message, link)

def add_watchlist_notification(username, symbol, action):
    """Add notification for watchlist changes"""
    if action == "add":
        title = f"⭐ Added to Watchlist: {symbol}"
        message = f"{symbol} has been added to your watchlist"
    else:
        title = f"🗑️ Removed from Watchlist: {symbol}"
        message = f"{symbol} has been removed from your watchlist"
    
    link = "/watchlist"
    return add_notification(username, "watchlist", title, message, link)

def add_trust_score_notification(username, old_score, new_score):
    """Add notification for trust score change"""
    if new_score > old_score:
        title = "🎉 Trust Score Increased!"
        message = f"Your trust score increased from {old_score:.0f} to {new_score:.0f}"
    else:
        title = "📊 Trust Score Updated"
        message = f"Your trust score is now {new_score:.0f}"
    
    link = "/profile"
    return add_notification(username, "trust_score", title, message, link)

# ========== API ENDPOINTS ==========

@notifications_bp.route("", methods=["GET"])
def get_notifications():
    if "user" not in session:
        return jsonify({"error": "Not logged in"}), 401
    username = session.get("user")
    user_id = get_user_id(username)
    if not user_id:
        return jsonify({"notifications": []})
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''SELECT id, type, title, message, link, is_read, created_at 
                FROM notifications WHERE user_id = ? ORDER BY created_at DESC LIMIT 100''', (user_id,))
    results = c.fetchall()
    conn.close()
    notifications = [{"id": r[0], "type": r[1], "title": r[2], "message": r[3], "link": r[4], "is_read": bool(r[5]), "created_at": r[6]} for r in results]
    return jsonify({"notifications": notifications})

@notifications_bp.route("/unread-count", methods=["GET"])
def get_unread_count():
    if "user" not in session:
        return jsonify({"count": 0}), 401
    username = session.get("user")
    user_id = get_user_id(username)
    if not user_id:
        return jsonify({"count": 0})
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM notifications WHERE user_id = ? AND is_read = 0", (user_id,))
    count = c.fetchone()[0]
    conn.close()
    return jsonify({"count": count})

@notifications_bp.route("/mark-read", methods=["POST"])
def mark_as_read():
    if "user" not in session:
        return jsonify({"error": "Not logged in"}), 401
    data = request.get_json()
    notification_id = data.get("notification_id")
    if not notification_id:
        return jsonify({"error": "Notification ID required"}), 400
    username = session.get("user")
    user_id = get_user_id(username)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("UPDATE notifications SET is_read = 1 WHERE id = ? AND user_id = ?", (notification_id, user_id))
    conn.commit()
    conn.close()
    return jsonify({"success": True})

@notifications_bp.route("/mark-all-read", methods=["POST"])
def mark_all_as_read():
    if "user" not in session:
        return jsonify({"error": "Not logged in"}), 401
    username = session.get("user")
    user_id = get_user_id(username)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("UPDATE notifications SET is_read = 1 WHERE user_id = ?", (user_id,))
    conn.commit()
    conn.close()
    return jsonify({"success": True})

@notifications_bp.route("/delete-all", methods=["POST"])
def delete_all_notifications():
    """Delete all notifications for the logged-in user"""
    if "user" not in session:
        return jsonify({"error": "Not logged in"}), 401
    
    username = session.get("user")
    user_id = get_user_id(username)
    
    if not user_id:
        return jsonify({"error": "User not found"}), 404
    
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM notifications WHERE user_id = ?", (user_id,))
    deleted_count = c.rowcount
    conn.commit()
    conn.close()
    
    return jsonify({"success": True, "deleted_count": deleted_count})

@notifications_bp.route("/add", methods=["POST"])
def add_notification_api():
    if "user" not in session:
        return jsonify({"error": "Not logged in"}), 401
    data = request.get_json()
    username = session.get("user")
    n_type = data.get("type", "system")
    title = data.get("title", "Notification")
    message = data.get("message", "")
    link = data.get("link", "#")
    success = add_notification(username, n_type, title, message, link)
    return jsonify({"success": success})