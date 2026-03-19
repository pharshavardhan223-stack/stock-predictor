"""
FILE → Place at: backend/utils/auto_train.py

Auto-trains the Linear Regression model on startup using the latest
1-year data from 10 major stocks.  Runs in a background thread so it
never blocks Flask from starting.

Usage — add these 2 lines near the top of app.py (after app is created):
    from backend.utils.auto_train import start_auto_train
    start_auto_train()
"""

import os
import threading
import datetime
import joblib
import numpy  as np
import pandas as pd

# ── paths ────────────────────────────────────────────────────────────────────
MODEL_PATH     = "models/linear_model.pkl"
TRAIN_LOG_PATH = "data/auto_train.log"
LOCK_FILE      = "data/.train_lock"

# Stocks used to build a broad training dataset
TRAIN_SYMBOLS = [
    "AAPL", "TSLA", "MSFT", "GOOGL", "AMZN",
    "NVDA", "META",  "NFLX",  "BABA",  "TCS.NS"
]

# ── helpers ───────────────────────────────────────────────────────────────────
def _log(msg: str):
    os.makedirs("data", exist_ok=True)
    ts  = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}\n"
    with open(TRAIN_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line)
    print(f"🤖 AutoTrain | {msg}")


def _should_retrain() -> bool:
    """Retrain if model is missing OR older than 24 hours."""
    if not os.path.exists(MODEL_PATH):
        return True
    age = datetime.datetime.now() - datetime.datetime.fromtimestamp(
        os.path.getmtime(MODEL_PATH)
    )
    return age.total_seconds() > 86_400   # 24 h


def _fetch_closes(symbol: str) -> np.ndarray | None:
    """Download 1-year daily closes for symbol. Returns numpy array or None."""
    try:
        import yfinance as yf
        tk   = yf.Ticker(symbol)
        hist = tk.history(period="1y")
        if hist.empty or len(hist) < 30:
            return None
        return hist["Close"].astype(float).values
    except Exception as e:
        _log(f"  ⚠ fetch failed for {symbol}: {e}")
        return None


# ── main training function ────────────────────────────────────────────────────
def _run_training():
    """Collect data, train model, save .pkl.  Runs inside a daemon thread."""

    # Prevent concurrent retraining
    if os.path.exists(LOCK_FILE):
        _log("Lock file found — skipping (another train is running)")
        return

    try:
        open(LOCK_FILE, "w").write("1")

        if not _should_retrain():
            _log("Model is up-to-date (< 24 h old) — skipping retrain")
            return

        _log("Starting model retrain …")

        from sklearn.linear_model    import LinearRegression
        from sklearn.preprocessing   import MinMaxScaler
        from sklearn.pipeline        import Pipeline

        X_all, y_all = [], []

        for sym in TRAIN_SYMBOLS:
            _log(f"  Fetching {sym} …")
            closes = _fetch_closes(sym)
            if closes is None:
                continue

            # Build supervised dataset: window of 10 days → predict next day
            WINDOW = 10
            for i in range(WINDOW, len(closes)):
                X_all.append(closes[i - WINDOW : i])
                y_all.append(closes[i])

        if len(X_all) < 100:
            _log("Not enough data collected — aborting retrain")
            return

        X = np.array(X_all)
        y = np.array(y_all)

        # Drop rows with any NaN values (causes LinearRegression to crash)
        mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X, y = X[mask], y[mask]

        if len(X) < 50:
            _log("Not enough clean data after NaN removal — aborting")
            return

        _log(f"  Training on {len(X):,} samples (after NaN cleanup) …")

        pipeline = Pipeline([
            ("scaler", MinMaxScaler()),
            ("model",  LinearRegression())
        ])
        pipeline.fit(X, y)

        os.makedirs("models", exist_ok=True)
        joblib.dump(pipeline, MODEL_PATH)

        # Also save a simple flat model (compatible with old predict_with_model)
        flat_model = LinearRegression()
        flat_X = np.arange(len(y)).reshape(-1, 1)
        flat_model.fit(flat_X, y)
        joblib.dump(flat_model, MODEL_PATH)          # overwrites with flat model
        # (keeps backward-compat with the existing predict_with_model helper)

        _log(f"✅ Retrain complete — model saved to {MODEL_PATH}")

        # ── record training stats ────────────────────────────────────────────
        import sqlite3
        try:
            DB_PATH = "data/stockai.db"
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
            c.execute(
                "INSERT INTO train_log (samples, symbols, status) VALUES (?, ?, ?)",
                (len(X), ",".join(TRAIN_SYMBOLS), "success")
            )
            conn.commit()
            conn.close()
        except Exception as db_err:
            _log(f"  DB log failed: {db_err}")

    except Exception as e:
        _log(f"❌ Retrain error: {e}")

    finally:
        if os.path.exists(LOCK_FILE):
            os.remove(LOCK_FILE)


# ── public entry-point ────────────────────────────────────────────────────────
def start_auto_train():
    """
    Call this once at app startup.
    Launches a daemon thread so Flask is never blocked.
    """
    t = threading.Thread(target=_run_training, daemon=True, name="AutoTrain")
    t.start()
    _log("AutoTrain thread launched (runs in background)")


# ── last-train status (for Admin panel) ──────────────────────────────────────
def get_last_train_info() -> dict:
    """Returns info about the last training run (used by admin panel)."""
    info = {
        "model_exists":   os.path.exists(MODEL_PATH),
        "last_trained":   "Never",
        "model_age_hrs":  None,
        "log_tail":       [],
    }

    if os.path.exists(MODEL_PATH):
        mtime = datetime.datetime.fromtimestamp(os.path.getmtime(MODEL_PATH))
        age   = datetime.datetime.now() - mtime
        info["last_trained"]  = mtime.strftime("%Y-%m-%d %H:%M:%S")
        info["model_age_hrs"] = round(age.total_seconds() / 3600, 1)

    if os.path.exists(TRAIN_LOG_PATH):
        with open(TRAIN_LOG_PATH) as f:
            lines = f.readlines()
        info["log_tail"] = [l.strip() for l in lines[-15:]]

    return info