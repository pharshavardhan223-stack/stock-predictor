# fix_database.py
import sqlite3
import os

DB_PATH = "data/users.db"

print("=" * 50)
print("FIXING DATABASE...")
print("=" * 50)

# Check if database exists
if not os.path.exists(DB_PATH):
    print(f"\nDatabase not found at {DB_PATH}")
    print("Creating new database...")
    os.makedirs("data", exist_ok=True)

# Connect to database
conn = sqlite3.connect(DB_PATH)
c = conn.cursor()

# Check what columns currently exist
c.execute("PRAGMA table_info(users)")
columns = c.fetchall()

if not columns:
    print("\nNo users table found. Creating new table...")
    # Create fresh table with all columns
    c.execute("""
        CREATE TABLE users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            email TEXT,
            role TEXT DEFAULT 'user',
            first_name TEXT,
            last_name TEXT
        )
    """)
    print("✅ Users table created with all columns!")
else:
    print("\nCurrent columns in users table:")
    existing_columns = [col[1] for col in columns]
    for col in columns:
        print(f"  - {col[1]}")
    
    # Add first_name column if missing
    if 'first_name' not in existing_columns:
        print("\n✅ Adding 'first_name' column...")
        c.execute("ALTER TABLE users ADD COLUMN first_name TEXT")
        print("   'first_name' column added successfully!")
    else:
        print("\n✅ 'first_name' column already exists")
    
    # Add last_name column if missing
    if 'last_name' not in existing_columns:
        print("✅ Adding 'last_name' column...")
        c.execute("ALTER TABLE users ADD COLUMN last_name TEXT")
        print("   'last_name' column added successfully!")
    else:
        print("✅ 'last_name' column already exists")

# Save changes
conn.commit()
conn.close()

print("\n" + "=" * 50)
print("DATABASE FIXED SUCCESSFULLY!")
print("=" * 50)
print("\nYou can now restart your Flask app and try registering again.")