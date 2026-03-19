from flask import Blueprint, render_template, request, redirect, url_for, session, flash
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3
import os

# ==========================
# BLUEPRINT
# ==========================

auth = Blueprint("auth", __name__)

DB_PATH = "data/users.db"


# ==========================
# INIT DATABASE
# ==========================

def init_auth_db():
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            username   TEXT UNIQUE NOT NULL,
            password   TEXT NOT NULL,
            email      TEXT,
            role       TEXT DEFAULT 'user',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Seed default users
    try:
        c.execute(
            "INSERT INTO users (username, password, role) VALUES (?, ?, ?)",
            ("admin", generate_password_hash("admin123"), "admin")
        )
    except sqlite3.IntegrityError:
        pass

    try:
        c.execute(
            "INSERT INTO users (username, password, role) VALUES (?, ?, ?)",
            ("user", generate_password_hash("1234"), "user")
        )
    except sqlite3.IntegrityError:
        pass

    conn.commit()
    conn.close()
    print("✅ Auth DB ready.")


# ==========================
# HELPER
# ==========================

def get_user(username):
    try:
        conn = sqlite3.connect(DB_PATH)
        c    = conn.cursor()
        c.execute("SELECT * FROM users WHERE username = ?", (username,))
        user = c.fetchone()
        conn.close()
        return user
    except Exception as e:
        print("DB Error:", e)
        return None


# ==========================
# REGISTER ROUTE
# ==========================

@auth.route("/register", methods=["GET", "POST"])
def register():

    if "user" in session:
        return redirect(url_for("dashboard"))

    if request.method == "POST":
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
            os.makedirs("data", exist_ok=True)
            conn = sqlite3.connect(DB_PATH)
            c    = conn.cursor()

            c.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    username   TEXT UNIQUE NOT NULL,
                    password   TEXT NOT NULL,
                    email      TEXT,
                    role       TEXT DEFAULT 'user',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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

        except Exception as e:
            print("Register error:", e)
            flash("Registration failed. Try again.", "error")

    return render_template("register.html")


# ==========================
# CHANGE PASSWORD
# ==========================

@auth.route("/change-password", methods=["GET", "POST"])
def change_password():

    if "user" not in session:
        return redirect(url_for("login"))

    if request.method == "POST":
        current = request.form.get("current_password", "")
        new_pwd = request.form.get("new_password", "")
        confirm = request.form.get("confirm_password", "")

        if not current or not new_pwd or not confirm:
            flash("All fields are required", "error")
            return render_template("change_password.html")

        if new_pwd != confirm:
            flash("New passwords do not match", "error")
            return render_template("change_password.html")

        if len(new_pwd) < 6:
            flash("Password must be at least 6 characters", "error")
            return render_template("change_password.html")

        user = get_user(session["user"])

        if not user or not check_password_hash(user[2], current):
            flash("Current password is incorrect", "error")
            return render_template("change_password.html")

        try:
            conn = sqlite3.connect(DB_PATH)
            c    = conn.cursor()
            c.execute(
                "UPDATE users SET password = ? WHERE username = ?",
                (generate_password_hash(new_pwd), session["user"])
            )
            conn.commit()
            conn.close()

            flash("Password changed successfully!", "success")
            return redirect(url_for("profile"))

        except Exception as e:
            print("Change password error:", e)
            flash("Failed to change password. Try again.", "error")

    return render_template("change_password.html")



