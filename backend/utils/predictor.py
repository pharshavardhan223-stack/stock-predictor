import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# ==========================
# TRAIN MODELS
# ==========================

def train_models(x, y):

    models = {}

    # Linear Regression
    lr = LinearRegression()
    lr.fit(x, y)

    models["Linear Regression"] = lr

    # Random Forest
    rf = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )
    rf.fit(x, y.ravel())

    models["Random Forest"] = rf

    return models


# ==========================
# EVALUATE MODEL
# ==========================

def evaluate_model(model, x, y):

    preds = model.predict(x)

    r2 = r2_score(y, preds)
    mae = mean_absolute_error(y, preds)
    rmse = np.sqrt(mean_squared_error(y, preds))

    return r2, mae, rmse


# ==========================
# MAIN PREDICTION FUNCTION
# ==========================

def predict_series(series, days=7):

    # Convert to numpy (safety)
    series = np.array(series, dtype=float)

    # Minimum data check
    if len(series) < 5:
        return {
            "predictions": [],
            "last_actual": 0,
            "future_avg": 0,
            "trend_percent": 0,
            "best_model": "N/A",

            "r2": 0,
            "mae": 0,
            "rmse": 0,

            "ma7": 0,
            "ma14": 0,

            "volatility": 0,

            "models": {}
        }


    # ==========================
    # PREPARE DATA
    # ==========================

    y = series.reshape(-1, 1)
    x = np.arange(len(y)).reshape(-1, 1)


    # ==========================
    # TRAIN MODELS
    # ==========================

    models = train_models(x, y)

    best_model = None
    best_score = -999
    best_name = None

    best_r2 = 0
    best_mae = 0
    best_rmse = 0

    analytics = {}


    # ==========================
    # EVALUATE MODELS
    # ==========================

    for name, model in models.items():

        r2, mae, rmse = evaluate_model(model, x, y)

        r2 = round(float(r2), 3)
        mae = round(float(mae), 3)
        rmse = round(float(rmse), 3)

        analytics[name] = {
            "r2": r2,
            "mae": mae,
            "rmse": rmse
        }

        # Select best model
        if r2 > best_score:

            best_score = r2
            best_model = model
            best_name = name

            best_r2 = r2
            best_mae = mae
            best_rmse = rmse


    # Safety: if no model selected
    if best_model is None:
        return {
            "predictions": [],
            "last_actual": 0,
            "future_avg": 0,
            "trend_percent": 0,
            "best_model": "N/A",

            "r2": 0,
            "mae": 0,
            "rmse": 0,

            "ma7": 0,
            "ma14": 0,

            "volatility": 0,

            "models": analytics
        }


    # ==========================
    # FUTURE PREDICTION
    # ==========================

    future_x = np.arange(len(y), len(y) + days).reshape(-1, 1)

    future_preds = best_model.predict(future_x).flatten()


    # ==========================
    # STATS
    # ==========================

    last_actual = float(y[-1][0])
    future_avg = float(np.mean(future_preds))


    # Moving averages
    ma7 = round(float(np.mean(series[-7:])), 2) if len(series) >= 7 else round(float(np.mean(series)), 2)
    ma14 = round(float(np.mean(series[-14:])), 2) if len(series) >= 14 else round(float(np.mean(series)), 2)


    # Trend %
    if last_actual == 0:
        trend = 0
    else:
        trend = round(
            ((future_avg - last_actual) / last_actual) * 100,
            2
        )


    # ==========================
    # VOLATILITY (LEVEL 9.1)
    # ==========================

    returns = np.diff(series) / series[:-1]
    volatility = round(float(np.std(returns) * 100), 2)


    # ==========================
    # RETURN RESULT
    # ==========================

    return {

        # Predictions
        "predictions": future_preds.tolist(),

        # Main values
        "last_actual": round(last_actual, 2),
        "future_avg": round(future_avg, 2),
        "trend_percent": trend,

        # Moving averages
        "ma7": ma7,
        "ma14": ma14,

        # Volatility
        "volatility": volatility,

        # Best model
        "best_model": best_name,

        # Metrics
        "r2": best_r2,
        "mae": best_mae,
        "rmse": best_rmse,

        # All models analytics
        "models": analytics
    }
