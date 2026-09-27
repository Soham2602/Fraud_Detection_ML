"""
predict.py
----------
Inference module for SENTINEL — Fraud Intelligence Platform.
Preserves full backward compatibility with the original function signatures
while adding robust validation and risk scoring through ModelService.

Usage as a script:
    python src/predict.py

Usage as a module:
    from predict import load_artifacts, predict_transaction, predict_batch
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np

# Ensure src is on sys.path
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import CREDITCARD_CSV, SAMPLE_PRESETS_JSON, DEFAULT_THRESHOLD
from model_service import ModelService


def load_artifacts():
    """Load model, scaler, and expected feature columns (backward-compatible)."""
    service = ModelService.get_instance()
    service.ensure_loaded()
    return service.model, service.scaler, service.feature_columns


def predict_transaction(
    input_dict: Dict[str, Any],
    model=None,
    scaler=None,
    feature_columns=None,
    threshold: float = DEFAULT_THRESHOLD,
) -> Tuple[int, float]:
    """
    Predict a single transaction.
    Backward-compatible signature: returns (prediction: int, fraud_probability: float).
    """
    service = ModelService.get_instance()
    # Override artifacts if custom instances were passed
    if model is not None:
        service.model = model
    if scaler is not None:
        service.scaler = scaler
    if feature_columns is not None:
        service.feature_columns = feature_columns

    result = service.predict_single(input_dict, threshold=threshold)
    return result["prediction"], result["fraud_probability"]


def predict_batch(
    df: pd.DataFrame,
    model=None,
    scaler=None,
    feature_columns=None,
    threshold: float = DEFAULT_THRESHOLD,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Batch prediction.
    Backward-compatible signature: returns (predictions: np.ndarray, probabilities: np.ndarray).
    """
    service = ModelService.get_instance()
    if model is not None:
        service.model = model
    if scaler is not None:
        service.scaler = scaler
    if feature_columns is not None:
        service.feature_columns = feature_columns

    enriched, _ = service.predict_batch(df, threshold=threshold)
    return enriched["Predicted_Class"].values, enriched["Fraud_Probability"].values


if __name__ == "__main__":
    import json
    service = ModelService.get_instance()
    service.ensure_loaded()

    # Try presets first, fallback to creditcard.csv
    if SAMPLE_PRESETS_JSON.exists():
        print(f"Loading sample transactions from {SAMPLE_PRESETS_JSON.name}...")
        with open(SAMPLE_PRESETS_JSON, "r") as f:
            samples = json.load(f)

        for i, s in enumerate(samples[:6]):
            res = service.predict_single(s["features"])
            label = "[FRAUD]" if res["prediction"] == 1 else "[LEGIT]"
            actual = "FRAUD" if s.get("actual_class") == 1 else "Legitimate"
            print(
                f"[{i+1}] {s['name'][:35]:<35} | "
                f"Pred: {label:<8} | Score: {res['risk_score']:>3}/100 ({res['risk_level']:<8}) | "
                f"Actual: {actual}"
            )
    elif CREDITCARD_CSV.exists():
        print(f"Sampling transactions from {CREDITCARD_CSV.name}...")
        df = pd.read_csv(CREDITCARD_CSV)
        sample = df.sample(5, random_state=42)
        y_true = sample["Class"].values
        X_sample = sample.drop(columns=["Class"])

        preds, probs = predict_batch(X_sample)
        for i in range(len(sample)):
            label = "FRAUD" if preds[i] == 1 else "Legitimate"
            actual = "FRAUD" if y_true[i] == 1 else "Legitimate"
            print(
                f"Transaction {i+1}: Predicted = {label:<10} "
                f"(confidence: {probs[i]:.2%})  |  Actual = {actual}"
            )
    else:
        print("No sample presets or dataset found. Please run preprocessing first.")
