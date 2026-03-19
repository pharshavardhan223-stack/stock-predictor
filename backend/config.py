import os

# ==========================
# BASE DIRECTORY
# ==========================

BASE_DIR = os.path.abspath(os.path.dirname(__file__))


# ==========================
# SECURITY
# ==========================

SECRET_KEY = os.getenv("SECRET_KEY", "stock_prediction_secret_key_2026")


# ==========================
# ANTHROPIC API
# ==========================

# ⚠️  IMPORTANT: Never hardcode API keys in source code.
# Set this as an environment variable instead:
#   Windows:  set ANTHROPIC_API_KEY=sk-ant-api03-...
#   Linux/Mac: export ANTHROPIC_API_KEY=sk-ant-api03-...
# Or create a .env file and load with python-dotenv.
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")


# ==========================
# UPLOAD SETTINGS
# ==========================

UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")

ALLOWED_EXTENSIONS = {"csv", "xlsx", "xls"}

MAX_FILE_SIZE = 5 * 1024 * 1024      # 5 MB
MAX_ROWS      = 10000


# ==========================
# SESSION SETTINGS
# ==========================

SESSION_PERMANENT = False
SESSION_TYPE      = "filesystem"


# ==========================
# ML SETTINGS
# ==========================

PREDICTION_DAYS = 7

# Auto-train: retrain model if older than this many hours
AUTO_TRAIN_INTERVAL_HOURS = 24

# Stocks used for auto-training the model on startup
AUTO_TRAIN_SYMBOLS = [
    "AAPL", "TSLA", "MSFT", "GOOGL", "AMZN",
    "NVDA", "META", "NFLX", "BABA",  "TCS.NS"
]


# ==========================
# APP SETTINGS
# ==========================

DEBUG = os.getenv("FLASK_DEBUG", "true").lower() == "true"
HOST  = os.getenv("FLASK_HOST", "127.0.0.1")
PORT  = int(os.getenv("FLASK_PORT", 5000))


# ==========================
# DEMO USERS  (fallback only)
# ==========================

# These are only used if the users.db SQLite database doesn't exist yet.
# Real users are stored in data/users.db via auth.py.
USERS = {
    "admin": "admin123",
    "user":  "1234",
}


# ==========================
# EMAIL / ALERT SETTINGS
# ==========================

# Gmail SMTP  (recommended)
# Steps to get an App Password:
#   1. Go to myaccount.google.com → Security
#   2. Enable 2-Step Verification
#   3. Search "App Passwords" → generate one for "Mail"
#   4. Paste the 16-char password as MAIL_PASSWORD below (or in env)

MAIL_SERVER        = os.getenv("MAIL_SERVER",   "smtp.gmail.com")
MAIL_PORT          = int(os.getenv("MAIL_PORT", 587))
MAIL_USE_TLS       = True
MAIL_USE_SSL       = False
MAIL_USERNAME      = os.getenv("mymailforn8n@gmail.com", "")   # your@gmail.com
MAIL_PASSWORD      = os.getenv("pharshavardhan184", "")   # Gmail App Password
MAIL_DEFAULT_SENDER= os.getenv("MAIL_USERNAME", "")
ALERT_FROM         = os.getenv("MAIL_USERNAME", "")

# How often the alert checker runs (seconds).  Default: 5 minutes.
ALERT_CHECK_INTERVAL = int(os.getenv("ALERT_CHECK_INTERVAL", 300))


# ==========================
# DATABASE PATHS
# ==========================

DB_USERS   = os.path.join(BASE_DIR, "..", "data", "users.db")
DB_STOCKAI = os.path.join(BASE_DIR, "..", "data", "stockai.db")


# ==========================
# MODEL PATHS
# ==========================

MODEL_PATH      = os.path.join(BASE_DIR, "..", "models", "linear_model.pkl")
TRAIN_LOG_PATH  = os.path.join(BASE_DIR, "..", "data",   "auto_train.log")