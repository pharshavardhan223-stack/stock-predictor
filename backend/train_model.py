# ==========================
# train_model.py
# Run this ONCE to generate models/linear_model.pkl
# Command: python -m backend.utils.train_model
# ==========================

import os
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression

def train_and_save():

    os.makedirs("models", exist_ok=True)

    # Generate synthetic training data
    # (simple upward trend — just to initialize the model)
    x = np.arange(100).reshape(-1, 1)
    y = (x * 1.5 + np.random.randn(100, 1) * 2).ravel()

    model = LinearRegression()
    model.fit(x, y)

    model_path = "models/linear_model.pkl"
    joblib.dump(model, model_path)

    print(f"✅ Model trained and saved to {model_path}")
    print(f"   Coefficients : {model.coef_}")
    print(f"   Intercept    : {model.intercept_:.4f}")

if __name__ == "__main__":
    train_and_save()