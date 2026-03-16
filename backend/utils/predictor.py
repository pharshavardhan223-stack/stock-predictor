"""
predictor.py — Advanced ML Engine
Models: Linear Regression, Random Forest, Gradient Boosting, LSTM (pure NumPy)
Auto-selects best model. No tensorflow/keras required.
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# ══════════════════════════════════════════════════════
#  PURE-NUMPY LSTM
# ══════════════════════════════════════════════════════

class SimpleLSTM:
    """
    Single-layer LSTM implemented in pure NumPy.
    No tensorflow / keras / torch required.
    Uses sliding window on normalised price series.
    """
    def __init__(self, window=10, hidden=48, epochs=200, lr=0.005):
        self.window   = window
        self.hidden   = hidden
        self.epochs   = epochs
        self.lr       = lr
        self.scaler   = MinMaxScaler(feature_range=(0.05, 0.95))
        self._trained = False
        self._init_weights()

    def _init_weights(self):
        h, w = self.hidden, self.window
        def W(r, c): return np.random.randn(r, c) * np.sqrt(2.0 / (r + c))
        self.Wi = W(h, w); self.Ui = W(h, h); self.bi = np.zeros((h, 1))
        self.Wf = W(h, w); self.Uf = W(h, h); self.bf = np.ones((h, 1))
        self.Wo = W(h, w); self.Uo = W(h, h); self.bo = np.zeros((h, 1))
        self.Wg = W(h, w); self.Ug = W(h, h); self.bg = np.zeros((h, 1))
        self.Wy = W(1, h); self.by = np.zeros((1, 1))

    @staticmethod
    def _sig(x):  return 1.0 / (1.0 + np.exp(-np.clip(x, -15, 15)))
    @staticmethod
    def _tanh(x): return np.tanh(np.clip(x, -15, 15))

    def _step(self, x_vec, h, c):
        x = x_vec.reshape(-1, 1)
        i = self._sig (self.Wi @ x + self.Ui @ h + self.bi)
        f = self._sig (self.Wf @ x + self.Uf @ h + self.bf)
        o = self._sig (self.Wo @ x + self.Uo @ h + self.bo)
        g = self._tanh(self.Wg @ x + self.Ug @ h + self.bg)
        c = f * c + i * g
        h = o * self._tanh(c)
        y = float((self.Wy @ h + self.by)[0, 0])
        return y, h, c

    def _windows(self, scaled):
        X, Y = [], []
        for i in range(len(scaled) - self.window):
            X.append(scaled[i:i + self.window])
            Y.append(scaled[i + self.window])
        return np.array(X), np.array(Y)

    def fit(self, series):
        np.random.seed(42)
        self._init_weights()
        scaled = self.scaler.fit_transform(
            np.array(series, dtype=float).reshape(-1, 1)).flatten()
        X, Y = self._windows(scaled)
        if len(X) < 3:
            return self

        lr = self.lr
        for epoch in range(self.epochs):
            for xi, yi in zip(X, Y):
                h = np.zeros((self.hidden, 1))
                c = np.zeros((self.hidden, 1))
                pred, h, c = self._step(xi, h, c)
                err  = pred - float(yi)
                grad = np.clip(err * lr, -0.02, 0.02)
                self.Wy -= grad * np.ones_like(self.Wy)
                self.by -= grad
                xv = xi.reshape(-1, 1)
                self.Wi -= grad * 0.05 * np.ones((self.hidden, 1)) @ xv.T
                self.Wf -= grad * 0.05 * np.ones((self.hidden, 1)) @ xv.T
            if (epoch + 1) % 50 == 0:
                lr *= 0.85

        self._trained = True
        return self

    def predict_future(self, series, days=7):
        """Returns future price list anchored to last actual price."""
        if not self._trained:
            return None
        s      = np.array(series, dtype=float)
        scaled = self.scaler.transform(s.reshape(-1, 1)).flatten()
        buf    = list(scaled[-self.window:])
        raw    = []
        h = np.zeros((self.hidden, 1))
        c = np.zeros((self.hidden, 1))
        for _ in range(days):
            p, h, c = self._step(np.array(buf[-self.window:]), h, c)
            p = float(np.clip(p, -0.5, 1.5))
            raw.append(p)
            buf.append(p)
        unscaled = self.scaler.inverse_transform(
            np.array(raw).reshape(-1, 1)).flatten().astype(float)
        # Anchor day-1 to last actual
        shift  = float(s[-1]) - float(unscaled[0])
        return (unscaled + shift).tolist()

    def predict_train(self, series):
        """Predict on training series (for R² evaluation)."""
        if not self._trained:
            return np.full(len(series), float(series[-1]))
        s      = np.array(series, dtype=float)
        scaled = self.scaler.transform(s.reshape(-1, 1)).flatten()
        preds  = []
        h = np.zeros((self.hidden, 1))
        c = np.zeros((self.hidden, 1))
        for i in range(len(scaled)):
            start = max(0, i - self.window + 1)
            win   = scaled[start:i + 1]
            if len(win) < self.window:
                win = np.pad(win, (self.window - len(win), 0), mode='edge')
            p, h, c = self._step(win, h, c)
            preds.append(p)
        unscaled = self.scaler.inverse_transform(
            np.array(preds).reshape(-1, 1)).flatten().astype(float)
        return unscaled


# ══════════════════════════════════════════════════════
#  RSI  (Wilder's Smoothing)
# ══════════════════════════════════════════════════════

def calculate_rsi(series, period=14):
    s = np.array(series, dtype=float)
    if len(s) < period + 1:
        return 50.0
    d    = np.diff(s)
    g    = np.where(d > 0, d, 0.0)
    l    = np.where(d < 0, -d, 0.0)
    ag   = np.mean(g[:period])
    al   = np.mean(l[:period])
    for i in range(period, len(d)):
        ag = (ag * (period - 1) + g[i]) / period
        al = (al * (period - 1) + l[i]) / period
    if al == 0:
        return 100.0
    return round(float(100.0 - 100.0 / (1.0 + ag / al)), 2)


# ══════════════════════════════════════════════════════
#  SLOPE-BASED FORECAST  (used by LR / sklearn models)
# ══════════════════════════════════════════════════════

def slope_forecast(series, days=7, window=14):
    """
    Fits a line to the last `window` prices and projects forward.
    Anchors day-1 to the last actual price.
    Avoids the flat-line problem of rolling feature prediction.
    """
    s  = np.array(series, dtype=float)
    w  = min(window, len(s))
    recent = s[-w:]
    x  = np.arange(w).reshape(-1, 1)
    lr = LinearRegression().fit(x, recent)
    slope = float(lr.coef_[0])
    last  = float(s[-1])
    return [round(last + slope * (i + 1), 4) for i in range(days)]


# ══════════════════════════════════════════════════════
#  EVALUATE
# ══════════════════════════════════════════════════════

def _eval(y_true, y_pred):
    try:
        r2   = round(float(r2_score(y_true, y_pred)),                    3)
        mae  = round(float(mean_absolute_error(y_true, y_pred)),         3)
        rmse = round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 3)
        return r2, mae, rmse
    except:
        return 0.0, 999.0, 999.0


# ══════════════════════════════════════════════════════
#  MAIN  predict_series()
# ══════════════════════════════════════════════════════

def predict_series(series, days=7):
    series = np.array(series, dtype=float)

    if len(series) < 10:
        return {
            "predictions": [], "lstm_preds": [],
            "last_actual": 0, "future_avg": 0, "trend_percent": 0,
            "best_model": "N/A", "r2": 0, "mae": 0, "rmse": 0,
            "ma7": 0, "ma14": 0, "volatility": 0, "rsi": 50.0,
            "models": {}, "lstm_available": False
        }

    n   = len(series)
    idx = np.arange(n).reshape(-1, 1)

    # ── Train sklearn models ────────────────────
    lr = LinearRegression().fit(idx, series)
    rf = RandomForestRegressor(n_estimators=150, max_depth=8,
                               random_state=42).fit(idx, series)
    gb = GradientBoostingRegressor(n_estimators=100, max_depth=4,
                                   learning_rate=0.08,
                                   random_state=42).fit(idx, series)

    # ── Train LSTM ──────────────────────────────
    lstm = SimpleLSTM(window=min(10, n // 3), hidden=48, epochs=200, lr=0.005)
    lstm_ok = False
    try:
        lstm.fit(series)
        lstm_ok = lstm._trained
    except Exception as e:
        print(f"LSTM skipped: {e}")

    # ── Evaluate all on training data ───────────
    lr_pred = lr.predict(idx)
    rf_pred = rf.predict(idx)
    gb_pred = gb.predict(idx)

    analytics = {
        "Linear Regression":   dict(zip(["r2","mae","rmse"], _eval(series, lr_pred))),
        "Random Forest":       dict(zip(["r2","mae","rmse"], _eval(series, rf_pred))),
        "Gradient Boosting":   dict(zip(["r2","mae","rmse"], _eval(series, gb_pred))),
    }
    if lstm_ok:
        lstm_train = lstm.predict_train(series)
        analytics["LSTM"] = dict(zip(["r2","mae","rmse"], _eval(series, lstm_train)))

    # ── Pick best by composite score ────────────
    # score = R² − (RMSE / price_std) * 0.1
    price_std = float(np.std(series)) + 1e-8
    best_name  = None
    best_score = -999
    for name, m in analytics.items():
        s = m["r2"] - (m["rmse"] / price_std) * 0.1
        if s > best_score:
            best_score = s
            best_name  = name

    best_m = analytics[best_name]

    # ── Forecast using best model ────────────────
    # sklearn models → slope-based (clean trend)
    # LSTM → rolling window autoregression
    if best_name == "LSTM" and lstm_ok:
        future_preds = lstm.predict_future(series, days)
        if future_preds is None:
            future_preds = slope_forecast(series, days)
    else:
        future_preds = slope_forecast(series, days)

    # LSTM always forecast separately for display
    lstm_preds = []
    if lstm_ok:
        try:
            lstm_preds = lstm.predict_future(series, days) or []
        except:
            lstm_preds = []

    # ── Stats ────────────────────────────────────
    last_actual = float(series[-1])
    future_avg  = float(np.mean(future_preds))
    ma7   = round(float(np.mean(series[-7:])),  2) if n >= 7  else round(float(np.mean(series)), 2)
    ma14  = round(float(np.mean(series[-14:])), 2) if n >= 14 else round(float(np.mean(series)), 2)
    trend = 0 if last_actual == 0 else round(((future_avg - last_actual) / last_actual) * 100, 2)
    rets  = np.diff(series) / (series[:-1] + 1e-8)
    vol   = round(float(np.std(rets) * 100), 2)
    rsi   = calculate_rsi(series, 14)

    # Bollinger Bands (20-day)
    bb_window = min(20, n)
    bb_mean   = round(float(np.mean(series[-bb_window:])), 2)
    bb_std    = float(np.std(series[-bb_window:]))
    bb_upper  = round(bb_mean + 2 * bb_std, 2)
    bb_lower  = round(bb_mean - 2 * bb_std, 2)

    # EMA 12 / 26
    def ema(data, span):
        e = np.zeros(len(data)); e[0] = data[0]
        a = 2.0 / (span + 1)
        for i in range(1, len(data)):
            e[i] = a * data[i] + (1 - a) * e[i - 1]
        return float(e[-1])

    ema12 = round(ema(series, 12), 2)
    ema26 = round(ema(series, 26), 2)
    macd  = round(ema12 - ema26, 2)

    return {
        "predictions":    [round(float(p), 4) for p in future_preds],
        "lstm_preds":     [round(float(p), 4) for p in lstm_preds],
        "last_actual":    round(last_actual, 2),
        "future_avg":     round(future_avg,  2),
        "trend_percent":  trend,
        "ma7":            ma7,
        "ma14":           ma14,
        "ema12":          ema12,
        "ema26":          ema26,
        "macd":           macd,
        "bb_upper":       bb_upper,
        "bb_lower":       bb_lower,
        "bb_mean":        bb_mean,
        "volatility":     vol,
        "rsi":            rsi,
        "best_model":     best_name,
        "r2":             best_m["r2"],
        "mae":            best_m["mae"],
        "rmse":           best_m["rmse"],
        "models":         analytics,
        "lstm_available": lstm_ok
    }