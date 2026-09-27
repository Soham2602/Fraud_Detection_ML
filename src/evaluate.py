"""
evaluate.py
-----------
Comprehensive model evaluation suite for SENTINEL — Fraud Intelligence Platform.
Deliberately focuses on Precision-Recall curves, Average Precision (PR-AUC),
F1-score, and threshold trade-offs to properly evaluate severely imbalanced fraud data.
"""

import os
import sys
from pathlib import Path
_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
    brier_score_loss,
)

from config import FIGURES_DIR

def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "model",
    save_plots: bool = True,
    threshold: float = 0.50,
) -> Dict[str, Any]:
    """
    Compute comprehensive metrics for a trained classifier:
    - Accuracy, Precision, Recall, F1
    - ROC-AUC, Average Precision (PR-AUC)
    - Confusion Matrix
    - Threshold performance profile
    - Optional figure generation (CM, ROC curve, PR curve)
    """
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X_test)
        y_proba = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
    else:
        y_proba = model.predict(X_test).astype(float)

    # Apply decision threshold
    y_pred = (y_proba >= threshold).astype(int)

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    auc = float(roc_auc_score(y_test, y_proba))
    pr_auc = float(average_precision_score(y_test, y_proba))
    brier = float(brier_score_loss(y_test, y_proba))
    cm = confusion_matrix(y_test, y_pred)

    tn, fp, fn, tp = cm.ravel()

    print(f"\n--- Evaluation: {model_name} (Threshold = {threshold:.2f}) ---")
    print(f"Accuracy         : {acc:.4f}")
    print(f"Precision        : {prec:.4f}  (Of flagged transactions, {prec:.1%} were real fraud)")
    print(f"Recall           : {rec:.4f}  (Of all real fraud cases, {rec:.1%} were detected)")
    print(f"F1-score         : {f1:.4f}")
    print(f"ROC-AUC          : {auc:.4f}")
    print(f"PR-AUC (Avg Prec): {pr_auc:.4f}")
    print(f"Brier Score      : {brier:.4f}")
    print(f"Confusion Matrix : TP={tp}, FP={fp}, FN={fn}, TN={tn}")

    if save_plots:
        _plot_confusion_matrix(cm, model_name)
        _plot_roc_curve(y_test, y_proba, model_name, auc)
        _plot_precision_recall_curve(y_test, y_proba, model_name, pr_auc)

    threshold_table = compute_threshold_analysis(y_test, y_proba)

    return {
        "model_name": model_name,
        "threshold": threshold,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": auc,
        "average_precision": pr_auc,
        "brier_score": brier,
        "confusion_matrix": cm.tolist(),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "threshold_analysis": threshold_table,
    }


def compute_threshold_analysis(
    y_test: pd.Series, y_proba: np.ndarray, thresholds: Optional[List[float]] = None
) -> List[Dict[str, Any]]:
    """Compute precision, recall, F1, TP, FP, FN at varying decision thresholds."""
    if thresholds is None:
        thresholds = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]

    records = []
    y_true = np.asarray(y_test)

    for t in thresholds:
        preds = (y_proba >= t).astype(int)
        p = float(precision_score(y_true, preds, zero_division=0))
        r = float(recall_score(y_true, preds, zero_division=0))
        f = float(f1_score(y_true, preds, zero_division=0))
        cm = confusion_matrix(y_true, preds)
        tn, fp, fn, tp = cm.ravel()
        records.append({
            "threshold": round(t, 2),
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f, 4),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
        })

    return records


def _plot_confusion_matrix(cm: np.ndarray, model_name: str) -> None:
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
    path = FIGURES_DIR / f"confusion_matrix_{safe_name}.png"
    plt.savefig(path, dpi=150)
    plt.close()


def _plot_roc_curve(y_test: pd.Series, y_proba: np.ndarray, model_name: str, auc: float) -> None:
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, color="#2563eb", lw=2, label=f"ROC (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="#94a3b8")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve — {model_name}")
    plt.legend(loc="lower right")
    plt.tight_layout()
    safe_name = model_name.replace(" ", "_").replace("[", "").replace("]", "")
    path = FIGURES_DIR / f"roc_curve_{safe_name}.png"
    plt.savefig(path, dpi=150)
    plt.close()


def _plot_precision_recall_curve(
    y_test: pd.Series, y_proba: np.ndarray, model_name: str, pr_auc: float
) -> None:
    prec, rec, _ = precision_recall_curve(y_test, y_proba)
    plt.figure(figsize=(5, 4))
    plt.plot(rec, prec, color="#10b981", lw=2, label=f"PR Curve (AP = {pr_auc:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve — {model_name}")
    plt.legend(loc="lower left")
    plt.tight_layout()
    safe_name = model_name.replace(" ", "_").replace("[", "").replace("]", "")
    path = FIGURES_DIR / f"pr_curve_{safe_name}.png"
    plt.savefig(path, dpi=150)
    plt.close()


def compare_models(results_dict: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """Print and return a clean comparison table across all trained models."""
    rows = []
    print(f"\n{'Model':<35}{'Precision':<12}{'Recall':<12}{'F1':<10}{'ROC-AUC':<10}{'PR-AUC':<10}")
    print("-" * 89)
    for name, r in results_dict.items():
        metrics = r.get("metrics", r)
        print(
            f"{name:<35}{metrics['precision']:<12.4f}{metrics['recall']:<12.4f}"
            f"{metrics['f1']:<10.4f}{metrics['roc_auc']:<10.4f}{metrics.get('average_precision', 0.0):<10.4f}"
        )
        rows.append({
            "model": name,
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "roc_auc": metrics["roc_auc"],
            "average_precision": metrics.get("average_precision", 0.0),
            "accuracy": metrics.get("accuracy", 0.0),
        })
    return pd.DataFrame(rows)
