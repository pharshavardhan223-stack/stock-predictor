import pandas as pd
import numpy as np
import joblib
import os

from sklearn.linear_model import LinearRegression


# =========================
# SETTINGS
# =========================

DATA_FILE = "uploads/sample.csv"   # Put any CSV here
MODEL_PATH = "models/linear_model.pkl"
TARGET_COLUMN = "Close"            # Change based on your CSV


# =========================
# CREATE FOLDER
# =========================

os.makedirs("models", exist_ok=True)


# =========================
# LOAD DATA
# =========================

print("Loading dataset...")

df = pd.read_csv(DATA_FILE)

# Keep only numeric values
df = df[[TARGET_COLUMN]].dropna()

y = df[TARGET_COLUMN].values.reshape(-1, 1)
x = np.arange(len(y)).reshape(-1, 1)


# =========================
# TRAIN MODEL
# =========================

print("Training model...")

model = LinearRegression()
model.fit(x, y)


# =========================
# SAVE MODEL
# =========================

joblib.dump(model, MODEL_PATH)

print("Model saved at:", MODEL_PATH)
