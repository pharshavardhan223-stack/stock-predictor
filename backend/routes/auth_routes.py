"""
FILE: backend/routes/auth_routes.py

Exports:
    auth        — Blueprint (registered in app.py via app.register_blueprint(auth))
    auth_bp     — same Blueprint (alias)
    init_auth_db — creates users table on startup

Routes:
    /register   (GET + POST)

NOTE: /login and /logout are defined in app.py — do NOT add them here.
"""

import os
import sqlite3
import traceback
from flask import Blueprint, request, redirect, url_for, session, flash, render_template
from werkzeug.security import generate_password_hash

# Export as BOTH names so app.py can use either
auth    = Blueprint("auth", __name__)
auth_bp = auth

DB_PATH = os.path.join("data", "users.db")

# Admin security key (change this to your desired key)
ADMIN_SECURITY_KEY = "STOCKAI_ADMIN_2024"  # Change this to a strong key


def init_auth_db():
    """Create the users table if it doesn't exist. Called once at app startup."""
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Check if users table exists
    c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='users'")
    table_exists = c.fetchone()
    
    if not table_exists:
        # Create new table with all columns
        c.execute("""
            CREATE TABLE users (
                id       INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT    UNIQUE NOT NULL,
                password TEXT    NOT NULL,
                email    TEXT,
                role     TEXT    DEFAULT 'user',
                first_name TEXT,
                last_name TEXT
            )
        """)
        print("✅ Users table created with all columns.")
    else:
        # Check if first_name column exists, if not add it
        c.execute("PRAGMA table_info(users)")
        columns = [col[1] for col in c.fetchall()]
        
        if 'first_name' not in columns:
            c.execute("ALTER TABLE users ADD COLUMN first_name TEXT")
            print("✅ Added first_name column")
        
        if 'last_name' not in columns:
            c.execute("ALTER TABLE users ADD COLUMN last_name TEXT")
            print("✅ Added last_name column")
    
    conn.commit()
    conn.close()
    print("✅ Auth DB initialised.")


# Run on import so the table always exists
init_auth_db()


@auth.route("/register", methods=["GET", "POST"])
def register():
    # Already logged in → go to dashboard
    if "user" in session:
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        try:
            username = request.form.get("username", "").strip()
            password = request.form.get("password", "")
            confirm_password = request.form.get("confirm_password", "")
            email = request.form.get("email", "").strip()
            role = request.form.get("role", "user").strip()
            admin_passkey = request.form.get("admin_passkey", "").strip()

            first_name = request.form.get("first_name", "").strip()
            last_name = request.form.get("last_name", "").strip()
            display_name = f"{first_name} {last_name}".strip() or username

            # Print debug info
            print(f"Registration attempt for username: {username}")
            print(f"Role selected: {role}")
            print(f"Admin passkey provided: {'Yes' if admin_passkey else 'No'}")

            # ── Validation ──────────────────────────────────────────
            if not username or not password:
                flash("Username and password are required.", "error")
                return render_template("register.html", first_name=first_name, last_name=last_name, 
                                       email=email, username=username, role=role)

            if len(username) < 3:
                flash("Username must be at least 3 characters.", "error")
                return render_template("register.html", first_name=first_name, last_name=last_name, 
                                       email=email, username=username, role=role)

            if not all(c.isalnum() or c == "_" for c in username):
                flash("Username: only letters, numbers, and underscore allowed.", "error")
                return render_template("register.html", first_name=first_name, last_name=last_name, 
                                       email=email, username=username, role=role)

            if len(password) < 6:
                flash("Password must be at least 6 characters.", "error")
                return render_template("register.html", first_name=first_name, last_name=last_name, 
                                       email=email, username=username, role=role)

            if password != confirm_password:
                flash("Passwords do not match.", "error")
                return render_template("register.html", first_name=first_name, last_name=last_name, 
                                       email=email, username=username, role=role)

            if not first_name:
                flash("First name is required.", "error")
                return render_template("register.html", first_name=first_name, last_name=last_name, 
                                       email=email, username=username, role=role)

            # Role validation with security key
            if role == "admin":
                # Verify admin security key
                if not admin_passkey:
                    flash("Admin security key is required to register as admin.", "error")
                    return render_template("register.html", first_name=first_name, last_name=last_name, 
                                           email=email, username=username, role="admin")
                
                if admin_passkey != ADMIN_SECURITY_KEY:
                    flash("Invalid admin security key. Account will be created as standard user.", "warning")
                    role = "user"  # Downgrade to user if wrong key
                else:
                    flash("Admin account privileges granted!", "success")
            elif role not in ("user", "admin"):
                role = "user"

            # ── Save to DB ──────────────────────────────────────────
            try:
                conn = sqlite3.connect(DB_PATH)
                c = conn.cursor()
                hashed = generate_password_hash(password)
                
                # Insert with all columns
                c.execute("""
                    INSERT INTO users (username, password, email, role, first_name, last_name) 
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (username, hashed, email, role, first_name, last_name))
                
                conn.commit()
                conn.close()
                
                print(f"✅ User {username} created successfully with role: {role}")
                flash(f"Account created! Welcome, {display_name}. Please sign in.", "success")
                return redirect(url_for("login"))

            except sqlite3.IntegrityError as e:
                print(f"IntegrityError: {e}")
                flash("Username already taken. Please choose another.", "error")
                return render_template("register.html", first_name=first_name, last_name=last_name, 
                                       email=email, username=username, role=role)

            except Exception as e:
                print(f"Database error: {e}")
                print(traceback.format_exc())
                flash("Something went wrong. Please try again.", "error")
                return render_template("register.html", first_name=first_name, last_name=last_name, 
                                       email=email, username=username, role=role)

        except Exception as e:
            print(f"Unexpected error in register POST: {e}")
            print(traceback.format_exc())
            flash("Something went wrong. Please try again.", "error")
            return render_template("register.html")

    return render_template("register.html")