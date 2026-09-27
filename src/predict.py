"""
predict.py
----------
Loads the saved model + scaler and makes predictions on new transaction(s).

Usage as a script (predicts on a sample from data/creditcard.csv):
    python src/predict.py

Usage as a module (used by app.py):
    from predict import load_artifacts, predict_transaction
"""

import os
import joblib
import pandas as pd
import numpy as np

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def load_artifacts():
    """Load the trained model, fitted scaler, and expected feature order."""
    model = joblib.load(os.path.join(MODELS_DIR, "fraud_detection_model.pkl"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    feature_columns = joblib.load(os.path.join(MODELS_DIR, "feature_columns.pkl"))
    return model, scaler, feature_columns


def predict_transaction(input_dict, model=None, scaler=None, feature_columns=None):
    """
    input_dict: a dict with keys Time, V1..V28, Amount (matching the
    original dataset's columns).

    Returns: (prediction, fraud_probability)
        prediction: 0 (legitimate) or 1 (fraud)
        fraud_probability: model's confidence that this is fraud (0-1)
    """
    if model is None or scaler is None or feature_columns is None:
        model, scaler, feature_columns = load_artifacts()

    row = pd.DataFrame([input_dict])[feature_columns]

    scale_cols = [c for c in ("Time", "Amount") if c in row.columns]
    row[scale_cols] = scaler.transform(row[scale_cols])

    prediction = model.predict(row)[0]
    probability = model.predict_proba(row)[0][1]

    return int(prediction), float(probability)


def predict_batch(df, model=None, scaler=None, feature_columns=None):
    """Same as predict_transaction but for a whole DataFrame of transactions."""
    if model is None or scaler is None or feature_columns is None:
        model, scaler, feature_columns = load_artifacts()

    df = df[feature_columns].copy()
    scale_cols = [c for c in ("Time", "Amount") if c in df.columns]
    df[scale_cols] = scaler.transform(df[scale_cols])

    predictions = model.predict(df)
    probabilities = model.predict_proba(df)[:, 1]
    return predictions, probabilities


if __name__ == "__main__":
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "creditcard.csv")
    if not os.path.exists(data_path):
        print("data/creditcard.csv not found. See README.md for download instructions.")
        raise SystemExit(1)

    model, scaler, feature_columns = load_artifacts()
    df = pd.read_csv(data_path)
    sample = df.sample(5, random_state=1)

    y_true = sample["Class"].values
    X_sample = sample.drop(columns=["Class"])

    preds, probs = predict_batch(X_sample, model, scaler, feature_columns)

    for i in range(len(sample)):
        label = "FRAUD" if preds[i] == 1 else "Legitimate"
        actual = "FRAUD" if y_true[i] == 1 else "Legitimate"
        print(
            f"Transaction {i+1}: Predicted = {label} "
            f"(confidence {probs[i]:.2%})  |  Actual = {actual}"
        )
