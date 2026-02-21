import numpy as np


# ==========================
# GENERATE RECOMMENDATION
# ==========================

def get_recommendation(last_value, predictions):

    avg_prediction = np.mean(predictions)

    change_percent = ((avg_prediction - last_value) / last_value) * 100


    if change_percent > 3:
        action = "BUY"
        reason = "Uptrend detected with positive growth"

    elif change_percent < -3:
        action = "SELL"
        reason = "Downtrend detected with negative growth"

    else:
        action = "HOLD"
        reason = "Stable market condition"


    return {
        "action": action,
        "reason": reason,
        "change": round(change_percent, 2),
        "avg": round(avg_prediction, 2)
    }
