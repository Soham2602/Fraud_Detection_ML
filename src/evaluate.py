"""
evaluate.py
-----------
Model evaluation utilities. Deliberately avoids relying on accuracy alone
since accuracy is a misleading metric on an imbalanced dataset like this
one (see PROJECT_REPORT.md for a full explanation with numbers).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
)

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


def evaluate_model(model, X_test, y_test, model_name="model", save_plots=True):
    """
    Compute the full metric suite for a trained classifier and optionally
    save a confusion-matrix plot and ROC-curve plot to the figures/ folder.
    Returns a dict of metrics for easy comparison across models.
    """
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_proba = y_pred  # fallback, e.g. for models without probability output

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n--- {model_name} ---")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}  (of predicted frauds, how many were real fraud)")
    print(f"Recall   : {rec:.4f}  (of real frauds, how many were caught)")
    print(f"F1-score : {f1:.4f}")
    print(f"ROC-AUC  : {auc:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    if save_plots:
        _plot_confusion_matrix(cm, model_name)
        _plot_roc_curve(y_test, y_proba, model_name, auc)

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": auc,
        "confusion_matrix": cm,
    }


def _plot_confusion_matrix(cm, model_name):
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Legit (0)", "Fraud (1)"],
        yticklabels=["Legit (0)", "Fraud (1)"],
    )
    plt.title(f"Confusion Matrix — {model_name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    safe_name = model_name.replace(" ", "_").replace("[", "").replace("]", "")
    path = os.path.join(FIGURES_DIR, f"confusion_matrix_{safe_name}.png")
    plt.savefig(path)
    plt.close()


def _plot_roc_curve(y_test, y_proba, model_name, auc):
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve — {model_name}")
    plt.legend()
    plt.tight_layout()
    safe_name = model_name.replace(" ", "_").replace("[", "").replace("]", "")
    path = os.path.join(FIGURES_DIR, f"roc_curve_{safe_name}.png")
    plt.savefig(path)
    plt.close()


def compare_models(results_dict):
    """
    results_dict: {model_name: metrics_dict}
    Prints a simple side-by-side comparison table — handy for the final
    slide of a presentation.
    """
    print(f"\n{'Model':<30}{'Precision':<12}{'Recall':<12}{'F1':<10}{'ROC-AUC':<10}")
    print("-" * 74)
    for name, metrics in results_dict.items():
        print(
            f"{name:<30}{metrics['precision']:<12.4f}{metrics['recall']:<12.4f}"
            f"{metrics['f1']:<10.4f}{metrics['roc_auc']:<10.4f}"
        )
