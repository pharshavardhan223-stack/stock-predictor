import os


# ==========================
# BASE DIRECTORY
# ==========================

BASE_DIR = os.path.abspath(os.path.dirname(__file__))


# ==========================
# SECURITY
# ==========================

SECRET_KEY = "stock_prediction_secret_key_2026"


# ==========================
# UPLOAD SETTINGS
# ==========================

UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")

ALLOWED_EXTENSIONS = {"csv", "xlsx", "xls"}

MAX_FILE_SIZE = 5 * 1024 * 1024      # 5 MB
MAX_ROWS = 10000


# ==========================
# SESSION SETTINGS
# ==========================

SESSION_PERMANENT = False
SESSION_TYPE = "filesystem"


# ==========================
# ML SETTINGS
# ==========================

PREDICTION_DAYS = 7


# ==========================
# APP SETTINGS
# ==========================

DEBUG = True

HOST = "127.0.0.1"
PORT = 5000


# ==========================
# DEMO USERS
# ==========================

USERS = {
    "admin": "admin123",
    "user1": "pass1",
    "user2": "pass2"
}
