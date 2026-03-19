# ── Flask ──
SECRET_KEY=your-super-secret-key-change-this-now
DEBUG=False
HOST=0.0.0.0
PORT=5000

# ── Upload ──
UPLOAD_FOLDER=uploads
MAX_ROWS=1000
ALLOWED_EXTENSIONS=csv

# ── Database ──
DB_PATH=data/stockai.db
USERS_DB=data/users.db
HISTORY_FILE=data/history.csv

# ── Email (Gmail) ──
MAIL_SERVER=smtp.gmail.com
MAIL_PORT=587
MAIL_USERNAME=your-email@gmail.com
MAIL_PASSWORD=your-app-password
MAIL_USE_TLS=True

# ── Stock ──
DEFAULT_STOCK=AAPL
DEFAULT_PERIOD=7d
LIVE_REFRESH_SEC=30

# ── Security ──
MAX_LOGIN_ATTEMPTS=5
SESSION_TIMEOUT=3600
PASSWORD_MIN_LEN=6