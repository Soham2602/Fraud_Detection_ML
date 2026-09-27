"""
train.py
--------
Trains multiple classification models (Logistic Regression, Decision Tree,
Random Forest) on the fraud detection dataset, compares two imbalance
handling strategies (class weights vs SMOTE), performs a light
hyperparameter search on the best-performing model family, and saves the
final chosen model + scaler to disk with joblib.

Run with:  python src/train.py
"""

import os
import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

from preprocessing import (
    load_data,
    clean_data,
    split_data,
    scale_features,
    balance_with_smote,
    get_class_weights,
)
from evaluate import evaluate_model


MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def build_models(class_weights=None):
    """Return a dict of untrained model instances."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight=class_weights, random_state=42
        ),
        "Decision Tree": DecisionTreeClassifier(
            class_weight=class_weights, random_state=42, max_depth=10
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, class_weight=class_weights, random_state=42, n_jobs=-1
        ),
    }


def train_and_compare(X_train, y_train, X_test, y_test, class_weights=None, tag=""):
    """Train every model in build_models() and print/store evaluation metrics."""
    models = build_models(class_weights)
    results = {}

    for name, model in models.items():
        print(f"\nTraining {name} [{tag}] ...")
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test, model_name=f"{name} [{tag}]")
        results[name] = {"model": model, "metrics": metrics}

    return results


def tune_random_forest(X_train, y_train):
    """
    Light RandomizedSearchCV over Random Forest hyperparameters.
    Kept intentionally small (few candidates, 3-fold CV) so it runs in a
    reasonable time on a laptop — this is a college mini-project, not a
    production tuning pipeline.
    """
    param_dist = {
        "n_estimators": [100, 200, 300],
        "max_depth": [8, 12, 16, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    }

    base_model = RandomForestClassifier(random_state=42, n_jobs=-1)

    search = RandomizedSearchCV(
        base_model,
        param_distributions=param_dist,
        n_iter=8,
        scoring="f1",
        cv=3,
        random_state=42,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_train, y_train)
    print("Best params found:", search.best_params_)
    print("Best CV F1 score:", search.best_score_)
    return search.best_estimator_


def main():
    print("=" * 60)
    print("STEP 1: Load and clean data")
    print("=" * 60)
    df = load_data()
    df = clean_data(df)

    print("\n" + "=" * 60)
    print("STEP 2: Train/test split (before any scaling/resampling)")
    print("=" * 60)
    X_train, X_test, y_train, y_test = split_data(df)

    print("\n" + "=" * 60)
    print("STEP 3: Scale Time/Amount (fit on train only)")
    print("=" * 60)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    print("\n" + "=" * 60)
    print("STEP 4a: Approach A — Class weights (no synthetic data)")
    print("=" * 60)
    weights = get_class_weights(y_train)
    results_weights = train_and_compare(
        X_train_scaled, y_train, X_test_scaled, y_test,
        class_weights=weights, tag="class-weight",
    )

    print("\n" + "=" * 60)
    print("STEP 4b: Approach B — SMOTE (oversample training data only)")
    print("=" * 60)
    X_train_res, y_train_res = balance_with_smote(X_train_scaled, y_train)
    results_smote = train_and_compare(
        X_train_res, y_train_res, X_test_scaled, y_test,
        class_weights=None, tag="SMOTE",
    )

    print("\n" + "=" * 60)
    print("STEP 5: Hyperparameter tuning on Random Forest + SMOTE data")
    print("=" * 60)
    best_rf = tune_random_forest(X_train_res, y_train_res)
    evaluate_model(best_rf, X_test_scaled, y_test, model_name="Tuned Random Forest [SMOTE]")

    print("\n" + "=" * 60)
    print("STEP 6: Save final model + scaler")
    print("=" * 60)
    joblib.dump(best_rf, os.path.join(MODELS_DIR, "fraud_detection_model.pkl"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
    joblib.dump(list(X_train.columns), os.path.join(MODELS_DIR, "feature_columns.pkl"))
    print(f"Model saved to {MODELS_DIR}/fraud_detection_model.pkl")
    print(f"Scaler saved to {MODELS_DIR}/scaler.pkl")

    print("\nAll done. See PROJECT_REPORT.md for full metric comparisons and explanations.")


if __name__ == "__main__":
    main()
