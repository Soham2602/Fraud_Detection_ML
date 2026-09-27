"""
train.py
--------
Automated model training, cross-validation, evaluation, and experiment tracking
for SENTINEL — Fraud Intelligence Platform.

Avoids data leakage:
- Stratified train/test split happens first.
- Scaler is fit ONLY on the training split.
- Imbalance handling (SMOTE or class weights) is applied strictly to training data.
- Cross-validation uses imblearn.pipeline.Pipeline so SMOTE is applied inside each CV fold.
- Results are saved to experiments/results.csv and models/model_metadata.json.
"""

import os
import sys
import json
import datetime
from pathlib import Path
from typing import Dict, Any, Tuple
import joblib
import pandas as pd
import numpy as np
import sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

from config import (
    CREDITCARD_CSV,
    MODELS_DIR,
    EXPERIMENTS_DIR,
    MODEL_PKL,
    SCALER_PKL,
    FEATURES_PKL,
    METADATA_JSON,
    EXPERIMENT_RESULTS_CSV,
    FEATURE_COLUMNS,
    DEFAULT_THRESHOLD,
)
from preprocessing import (
    load_data,
    clean_data,
    split_data,
    scale_features,
    balance_with_smote,
    get_class_weights,
)
from evaluate import evaluate_model, compare_models


def build_models(class_weights: Dict[int, float] = None) -> Dict[str, Any]:
    """Return dictionary of classifier baselines."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight=class_weights, random_state=42
        ),
        "Decision Tree": DecisionTreeClassifier(
            class_weight=class_weights, random_state=42, max_depth=10
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, class_weight=class_weights, random_state=42, n_jobs=-1
        ),
    }


def tune_random_forest_pipeline(X_train: pd.DataFrame, y_train: pd.Series) -> Any:
    """
    Hyperparameter tuning using an imbalanced-learn Pipeline.
    SMOTE is performed INSIDE each cross-validation fold, eliminating
    data leakage across validation folds.
    """
    pipeline = ImbPipeline([
        ("smote", SMOTE(random_state=42)),
        ("rf", RandomForestClassifier(random_state=42, n_jobs=-1)),
    ])

    param_dist = {
        "rf__n_estimators": [100, 200],
        "rf__max_depth": [10, 16, None],
        "rf__min_samples_split": [2, 5],
        "rf__min_samples_leaf": [1, 2],
    }

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=6,
        scoring="f1",
        cv=3,
        random_state=42,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_train, y_train)
    print("\nBest CV params:", search.best_params_)
    print(f"Best CV F1 score: {search.best_score_:.4f}")
    return search.best_estimator_


def run_training_suite(quick_mode: bool = False) -> Tuple[Any, Any, pd.DataFrame]:
    """Run full training suite, compare strategies, select deployment model, and save artifacts."""
    print("=" * 70)
    print("SENTINEL ML PIPELINE — TRAINING & COMPARISON")
    print("=" * 70)

    print("\n[Step 1/6] Loading and cleaning raw dataset...")
    df = load_data()
    raw_shape = df.shape
    df = clean_data(df)

    print(f"\n[Step 2/6] Stratified splitting (Train=80%, Test=20%)...")
    X_train, X_test, y_train, y_test = split_data(df, test_size=0.2, random_state=42)
    print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")
    print(f"Train fraud count: {int(y_train.sum())}, Test fraud count: {int(y_test.sum())}")

    print("\n[Step 3/6] Scaling Time and Amount (fit on train only)...")
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    results_records = []
    trained_models = {}

    # Approach A: Class Weights
    print("\n[Step 4a/6] Approach A: Class Weights (Cost-sensitive learning)...")
    weights = get_class_weights(y_train)
    cw_models = build_models(weights)
    for name, m in cw_models.items():
        tag = f"{name} [class-weight]"
        print(f"Fitting {tag}...")
        m.fit(X_train_scaled, y_train)
        metrics = evaluate_model(m, X_test_scaled, y_test, model_name=tag, save_plots=True)
        trained_models[tag] = {"model": m, "metrics": metrics, "strategy": "class-weight"}
        results_records.append({
            "model": name,
            "strategy": "class-weight",
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "roc_auc": metrics["roc_auc"],
            "average_precision": metrics["average_precision"],
            "brier_score": metrics["brier_score"],
        })

    # Approach B: SMOTE
    print("\n[Step 4b/6] Approach B: SMOTE (Resampling training fold only)...")
    X_train_res, y_train_res = balance_with_smote(X_train_scaled, y_train)
    smote_models = build_models(None)
    for name, m in smote_models.items():
        tag = f"{name} [SMOTE]"
        print(f"Fitting {tag}...")
        m.fit(X_train_res, y_train_res)
        metrics = evaluate_model(m, X_test_scaled, y_test, model_name=tag, save_plots=True)
        trained_models[tag] = {"model": m, "metrics": metrics, "strategy": "SMOTE"}
        results_records.append({
            "model": name,
            "strategy": "SMOTE",
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "roc_auc": metrics["roc_auc"],
            "average_precision": metrics["average_precision"],
            "brier_score": metrics["brier_score"],
        })

    # Step 5: Hyperparameter search or tuned model
    print("\n[Step 5/6] Tuning candidate model...")
    if not quick_mode:
        tuned_pipeline = tune_random_forest_pipeline(X_train_scaled, y_train)
        best_model = tuned_pipeline.named_steps["rf"]
    else:
        best_model = trained_models["Random Forest [SMOTE]"]["model"]

    tuned_tag = "Tuned Random Forest [SMOTE]"
    best_metrics = evaluate_model(best_model, X_test_scaled, y_test, model_name=tuned_tag, save_plots=True)
    trained_models[tuned_tag] = {"model": best_model, "metrics": best_metrics, "strategy": "SMOTE+Tuning"}
    results_records.append({
        "model": "Tuned Random Forest",
        "strategy": "SMOTE+Tuning",
        "accuracy": best_metrics["accuracy"],
        "precision": best_metrics["precision"],
        "recall": best_metrics["recall"],
        "f1": best_metrics["f1"],
        "roc_auc": best_metrics["roc_auc"],
        "average_precision": best_metrics["average_precision"],
        "brier_score": best_metrics["brier_score"],
    })

    # Save experiments/results.csv
    results_df = pd.DataFrame(results_records)
    results_df.to_csv(EXPERIMENT_RESULTS_CSV, index=False)
    print(f"\nSaved experiment results to {EXPERIMENT_RESULTS_CSV}")

    # Step 6: Select best model and save metadata
    print("\n[Step 6/6] Selecting deployment candidate based on F1-score & PR-AUC...")
    best_row = results_df.sort_values(by=["f1", "average_precision"], ascending=False).iloc[0]
    best_choice_tag = f"{best_row['model']} [{best_row['strategy']}]"
    if best_choice_tag not in trained_models:
        best_choice_tag = tuned_tag
    selected_entry = trained_models[best_choice_tag]
    selected_model = selected_entry["model"]

    # Save artifacts
    joblib.dump(selected_model, MODEL_PKL)
    joblib.dump(scaler, SCALER_PKL)
    feature_cols = list(X_train.columns)
    joblib.dump(feature_cols, FEATURES_PKL)

    metadata = {
        "platform": "SENTINEL — Fraud Intelligence Platform",
        "model_name": str(best_row["model"]),
        "strategy": str(best_row["strategy"]),
        "training_date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_shape": {
            "raw_rows": int(raw_shape[0]),
            "cleaned_rows": int(df.shape[0]),
            "train_rows": int(X_train.shape[0]),
            "test_rows": int(X_test.shape[0]),
            "train_fraud_samples": int(y_train.sum()),
            "test_fraud_samples": int(y_test.sum()),
        },
        "feature_count": len(feature_cols),
        "feature_list": feature_cols,
        "default_threshold": DEFAULT_THRESHOLD,
        "metrics": {
            "accuracy": float(best_row["accuracy"]),
            "precision": float(best_row["precision"]),
            "recall": float(best_row["recall"]),
            "f1": float(best_row["f1"]),
            "roc_auc": float(best_row["roc_auc"]),
            "average_precision": float(best_row["average_precision"]),
            "brier_score": float(best_row["brier_score"]),
        },
        "confusion_matrix": selected_entry["metrics"]["confusion_matrix"],
        "hyperparameters": getattr(selected_model, "get_params", lambda: {})(),
        "library_versions": {
            "python": sys.version.split()[0],
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }

    with open(METADATA_JSON, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved model metadata to {METADATA_JSON}")

    return selected_model, scaler, results_df


if __name__ == "__main__":
    run_training_suite()
