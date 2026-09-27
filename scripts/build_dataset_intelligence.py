"""
scripts/build_dataset_intelligence.py
------------------------------------
Extracts 100% ground-truth dataset statistics and model evaluation metrics from
the real Kaggle creditcard.csv dataset (284,807 transactions) and test fold.
Outputs:
  - data/dataset_summary.json (Kaggle EDA & intelligence bundle)
  - models/evaluation_bundle.json (High-resolution curves & threshold sweep)
"""

import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    precision_recall_curve,
    roc_auc_score,
    average_precision_score,
)
import joblib

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
SRC_DIR = BASE_DIR / "src"

sys.path.insert(0, str(SRC_DIR))
from preprocessing import clean_data, split_data, scale_features
from config import FEATURE_COLUMNS

CSV_PATH = DATA_DIR / "creditcard.csv"
MODEL_PKL = MODELS_DIR / "fraud_detection_model.pkl"
SCALER_PKL = MODELS_DIR / "scaler.pkl"


def compute_dataset_summary(df_in: pd.DataFrame) -> dict:
    df_raw = df_in.copy()
    print(f"Analyzing raw dataset: {df_raw.shape} rows...")

    total_rows = int(len(df_raw))
    duplicate_rows = int(df_raw.duplicated().sum())
    missing_values = int(df_raw.isnull().sum().sum())

    df_clean = df_raw.drop_duplicates().reset_index(drop=True)
    clean_rows = int(len(df_clean))

    class_counts = df_raw["Class"].value_counts().to_dict()
    fraud_count = int(class_counts.get(1, 0))
    legit_count = int(class_counts.get(0, 0))
    fraud_rate_pct = round((fraud_count / total_rows) * 100, 4)

    # Amount statistics
    def get_stats(series: pd.Series):
        return {
            "count": int(len(series)),
            "mean": round(float(series.mean()), 2),
            "std": round(float(series.std()), 2),
            "min": round(float(series.min()), 2),
            "p25": round(float(series.quantile(0.25)), 2),
            "median": round(float(series.median()), 2),
            "p75": round(float(series.quantile(0.75)), 2),
            "p90": round(float(series.quantile(0.90)), 2),
            "p99": round(float(series.quantile(0.99)), 2),
            "max": round(float(series.max()), 2),
        }

    amount_overall = get_stats(df_raw["Amount"])
    amount_legit = get_stats(df_raw[df_raw["Class"] == 0]["Amount"])
    amount_fraud = get_stats(df_raw[df_raw["Class"] == 1]["Amount"])

    # Amount binned distribution comparison
    bins = [0, 10, 50, 100, 200, 500, 1000, 2500, 30000]
    bin_labels = ["$0–10", "$10–50", "$50–100", "$100–200", "$200–500", "$500–1K", "$1K–2.5K", "$2.5K+"]

    df_raw["amount_bin"] = pd.cut(df_raw["Amount"], bins=bins, labels=bin_labels, right=True, include_lowest=True)
    legit_bins = df_raw[df_raw["Class"] == 0]["amount_bin"].value_counts().reindex(bin_labels).fillna(0).to_dict()
    fraud_bins = df_raw[df_raw["Class"] == 1]["amount_bin"].value_counts().reindex(bin_labels).fillna(0).to_dict()

    binned_distribution = []
    for label in bin_labels:
        l_cnt = int(legit_bins.get(label, 0))
        f_cnt = int(fraud_bins.get(label, 0))
        binned_distribution.append({
            "bin": label,
            "legit_count": l_cnt,
            "fraud_count": f_cnt,
            "legit_pct": round((l_cnt / legit_count) * 100, 2),
            "fraud_pct": round((f_cnt / fraud_count) * 100, 2),
        })

    # Time distribution (2 days = 48 hours = 172,800 seconds)
    df_raw["hour_overall"] = (df_raw["Time"] // 3600).astype(int).clip(0, 47)
    df_raw["hour_of_day"] = (df_raw["hour_overall"] % 24).astype(int)

    time_span_hours = round(float(df_raw["Time"].max() / 3600), 2)

    hourly_48h = []
    for h in range(48):
        sub = df_raw[df_raw["hour_overall"] == h]
        f_c = int((sub["Class"] == 1).sum())
        l_c = int((sub["Class"] == 0).sum())
        hourly_48h.append({
            "hour": h,
            "fraud_count": f_c,
            "legit_count": l_c,
            "total": f_c + l_c,
            "fraud_rate_pct": round((f_c / (f_c + l_c)) * 100, 3) if (f_c + l_c) > 0 else 0.0,
        })

    cyclic_24h = []
    for h in range(24):
        sub = df_raw[df_raw["hour_of_day"] == h]
        f_c = int((sub["Class"] == 1).sum())
        l_c = int((sub["Class"] == 0).sum())
        cyclic_24h.append({
            "hour_of_day": h,
            "fraud_count": f_c,
            "legit_count": l_c,
            "total": f_c + l_c,
            "fraud_rate_pct": round((f_c / (f_c + l_c)) * 100, 3) if (f_c + l_c) > 0 else 0.0,
        })

    # Correlation with Class
    corr_series = df_raw.drop(columns=["amount_bin", "hour_overall", "hour_of_day"]).corr()["Class"].drop("Class")
    top_pos_corr = corr_series.sort_values(ascending=False).head(8).to_dict()
    top_neg_corr = corr_series.sort_values(ascending=True).head(8).to_dict()

    correlations = {
        "top_positive": [{"feature": k, "correlation": round(float(v), 4)} for k, v in top_pos_corr.items()],
        "top_negative": [{"feature": k, "correlation": round(float(v), 4)} for k, v in top_neg_corr.items()],
    }

    # Feature profiles for all 30 features
    feature_profiles = {}
    for col in FEATURE_COLUMNS:
        s_all = df_raw[col]
        s_legit = df_raw[df_raw["Class"] == 0][col]
        s_fraud = df_raw[df_raw["Class"] == 1][col]
        corr_val = float(corr_series.get(col, 0.0))
        
        feature_profiles[col] = {
            "feature": col,
            "min": round(float(s_all.min()), 4),
            "max": round(float(s_all.max()), 4),
            "mean": round(float(s_all.mean()), 4),
            "median": round(float(s_all.median()), 4),
            "std": round(float(s_all.std()), 4),
            "fraud_mean": round(float(s_fraud.mean()), 4),
            "fraud_median": round(float(s_fraud.median()), 4),
            "fraud_std": round(float(s_fraud.std()), 4),
            "legit_mean": round(float(s_legit.mean()), 4),
            "legit_median": round(float(s_legit.median()), 4),
            "legit_std": round(float(s_legit.std()), 4),
            "correlation_with_class": round(corr_val, 4),
        }

    # Top features correlation matrix for Plotly Heatmap
    top_corr_features = ["V17", "V14", "V12", "V10", "V16", "V3", "V7", "V11", "V4", "V2", "Amount", "Time", "Class"]
    corr_matrix_df = df_raw[top_corr_features].corr().round(4)
    correlation_heatmap_data = {
        "features": top_corr_features,
        "z": corr_matrix_df.values.tolist(),
    }

    # Curated 3D sample: all 492 frauds + 1008 random legits (total 1500 points)
    frauds_df = df_raw[df_raw["Class"] == 1]
    legits_sample = df_raw[df_raw["Class"] == 0].sample(n=1008, random_state=42)
    combined_sample = pd.concat([frauds_df, legits_sample]).sample(frac=1.0, random_state=42)
    
    sample_3d = {
        "index": combined_sample.index.tolist(),
        "Class": combined_sample["Class"].astype(int).tolist(),
        "Amount": combined_sample["Amount"].round(2).tolist(),
        "Time": combined_sample["Time"].round(0).tolist(),
        "V14": combined_sample["V14"].round(3).tolist(),
        "V10": combined_sample["V10"].round(3).tolist(),
        "V12": combined_sample["V12"].round(3).tolist(),
        "V17": combined_sample["V17"].round(3).tolist(),
        "V4": combined_sample["V4"].round(3).tolist(),
        "V11": combined_sample["V11"].round(3).tolist(),
        "V16": combined_sample["V16"].round(3).tolist(),
        "V2": combined_sample["V2"].round(3).tolist(),
        "V7": combined_sample["V7"].round(3).tolist(),
        "V3": combined_sample["V3"].round(3).tolist(),
    }

    summary = {
        "overview": {
            "total_transactions": total_rows,
            "clean_transactions": clean_rows,
            "duplicate_transactions": duplicate_rows,
            "missing_values": missing_values,
            "fraud_count": fraud_count,
            "legitimate_count": legit_count,
            "fraud_rate_pct": fraud_rate_pct,
            "time_span_hours": time_span_hours,
            "dataset_citation": "Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson, Gianluca Bontempi. Calibrating Probability with Undersampling for Fraud Detection in Credit Card Data. IEEE CIDM 2015.",
        },
        "amount_statistics": {
            "overall": amount_overall,
            "legitimate": amount_legit,
            "fraudulent": amount_fraud,
        },
        "amount_distribution": binned_distribution,
        "time_hourly_48h": hourly_48h,
        "time_cyclic_24h": cyclic_24h,
        "correlations": correlations,
        "feature_profiles": feature_profiles,
        "correlation_heatmap": correlation_heatmap_data,
        "sample_3d": sample_3d,
    }

    return summary


def compute_evaluation_bundle(df_clean: pd.DataFrame, model, scaler) -> dict:
    print("Evaluating model on test fold...")
    X_train, X_test, y_train, y_test = split_data(df_clean, test_size=0.2, random_state=42)
    X_test = X_test[FEATURE_COLUMNS].copy()

    X_test_scaled = X_test.copy()
    X_test_scaled[["Time", "Amount"]] = scaler.transform(X_test[["Time", "Amount"]])

    probs = model.predict_proba(X_test_scaled)[:, 1]
    y_true = y_test.values

    # Overall metrics at default threshold 0.50
    preds_50 = (probs >= 0.50).astype(int)
    cm_50 = confusion_matrix(y_true, preds_50)
    tn, fp, fn, tp = int(cm_50[0, 0]), int(cm_50[0, 1]), int(cm_50[1, 0]), int(cm_50[1, 1])

    prec_50 = float(precision_score(y_true, preds_50, zero_division=0))
    rec_50 = float(recall_score(y_true, preds_50, zero_division=0))
    f1_50 = float(f1_score(y_true, preds_50, zero_division=0))
    roc_auc = float(roc_auc_score(y_true, probs))
    pr_auc = float(average_precision_score(y_true, probs))
    fpr_50 = round(fp / (fp + tn), 6)
    fnr_50 = round(fn / (fn + tp), 6)

    # 99-step High-Resolution Threshold Sweep
    threshold_sweep = []
    for t_val in np.linspace(0.01, 0.99, 99):
        t_float = round(float(t_val), 2)
        preds_t = (probs >= t_val).astype(int)
        cm_t = confusion_matrix(y_true, preds_t)
        tn_t, fp_t, fn_t, tp_t = int(cm_t[0, 0]), int(cm_t[0, 1]), int(cm_t[1, 0]), int(cm_t[1, 1])
        p_t = float(precision_score(y_true, preds_t, zero_division=0))
        r_t = float(recall_score(y_true, preds_t, zero_division=0))
        f_t = float(f1_score(y_true, preds_t, zero_division=0))
        fpr_t = round(fp_t / (fp_t + tn_t), 6) if (fp_t + tn_t) > 0 else 0.0
        fnr_t = round(fn_t / (fn_t + tp_t), 6) if (fn_t + tp_t) > 0 else 0.0

        threshold_sweep.append({
            "threshold": t_float,
            "precision": round(p_t, 4),
            "recall": round(r_t, 4),
            "f1": round(f_t, 4),
            "tp": tp_t,
            "fp": fp_t,
            "fn": fn_t,
            "tn": tn_t,
            "fpr": fpr_t,
            "fnr": fnr_t,
        })

    # ROC curve points (sampled to ~60 points for light JSON payload)
    fpr_raw, tpr_raw, thresh_roc = roc_curve(y_true, probs)
    step_roc = max(1, len(fpr_raw) // 60)
    roc_sampled = []
    for i in range(0, len(fpr_raw), step_roc):
        t_val = float(thresh_roc[i]) if i < len(thresh_roc) else 0.0
        if np.isinf(t_val) or np.isnan(t_val):
            t_val = 1.0
        roc_sampled.append({
            "fpr": round(float(fpr_raw[i]), 5),
            "tpr": round(float(tpr_raw[i]), 5),
            "threshold": round(t_val, 4),
        })
    # Ensure end point (1, 1) is present
    roc_sampled.append({"fpr": 1.0, "tpr": 1.0, "threshold": 0.0})

    # PR curve points (sampled to ~60 points)
    prec_raw, rec_raw, thresh_pr = precision_recall_curve(y_true, probs)
    step_pr = max(1, len(prec_raw) // 60)
    pr_sampled = []
    for i in range(0, len(prec_raw), step_pr):
        t_val = float(thresh_pr[i]) if i < len(thresh_pr) else 0.0
        if np.isinf(t_val) or np.isnan(t_val):
            t_val = 1.0
        pr_sampled.append({
            "recall": round(float(rec_raw[i]), 5),
            "precision": round(float(prec_raw[i]), 5),
            "threshold": round(t_val, 4),
        })
    # Ensure end point
    pr_sampled.append({"recall": 0.0, "precision": 1.0, "threshold": 1.0})

    # Feature importances
    feature_names = [f"V{i}" for i in range(1, 29)] + ["Time", "Amount"]
    if hasattr(model, "feature_importances_"):
        importances = [
            {"feature": name, "importance": round(float(imp), 5)}
            for name, imp in sorted(zip(X_test.columns, model.feature_importances_), key=lambda x: x[1], reverse=True)
        ]
    else:
        importances = []

    # Risk score distribution on test fold
    # Bin scores into 10-point bands
    risk_scores = np.clip(np.round(probs * 100), 0, 100).astype(int)
    bands = ["0–10", "11–20", "21–30", "31–40", "41–50", "51–60", "61–70", "71–80", "81–90", "91–100"]
    score_bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    legit_mask = (y_true == 0)
    fraud_mask = (y_true == 1)

    legit_scores = risk_scores[legit_mask]
    fraud_scores = risk_scores[fraud_mask]

    legit_binned, _ = np.histogram(legit_scores, bins=score_bins)
    fraud_binned, _ = np.histogram(fraud_scores, bins=score_bins)

    risk_distribution = []
    for b_label, l_c, f_c in zip(bands, legit_binned, fraud_binned):
        risk_distribution.append({
            "band": b_label,
            "legit_count": int(l_c),
            "fraud_count": int(f_c),
            "legit_pct": round(float(l_c / len(legit_scores)) * 100, 2),
            "fraud_pct": round(float(f_c / len(fraud_scores)) * 100, 2),
        })

    bundle = {
        "model_name": "Tuned Random Forest",
        "strategy": "SMOTE Inside Fold (Leak-Free)",
        "test_set_size": int(len(y_true)),
        "test_legit_count": int(np.sum(y_true == 0)),
        "test_fraud_count": int(np.sum(y_true == 1)),
        "default_metrics": {
            "threshold": 0.50,
            "accuracy": round(float(np.mean(preds_50 == y_true)), 6),
            "precision": round(prec_50, 4),
            "recall": round(rec_50, 4),
            "f1": round(f1_50, 4),
            "roc_auc": round(roc_auc, 4),
            "average_precision": round(pr_auc, 4),
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "fpr": fpr_50,
            "fnr": fnr_50,
        },
        "threshold_sweep": threshold_sweep,
        "roc_curve": roc_sampled,
        "pr_curve": pr_sampled,
        "feature_importances": importances,
        "risk_distribution": risk_distribution,
    }

    return bundle


def main():
    if not CSV_PATH.exists():
        print(f"Error: {CSV_PATH} not found.")
        sys.exit(1)

    df_raw = pd.read_csv(CSV_PATH)
    summary = compute_dataset_summary(df_raw)

    out_summary = DATA_DIR / "dataset_summary.json"
    with open(out_summary, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"-> Saved dataset intelligence to {out_summary}")

    df_clean = clean_data(df_raw)
    model = joblib.load(MODEL_PKL)
    scaler = joblib.load(SCALER_PKL)

    bundle = compute_evaluation_bundle(df_clean, model, scaler)
    out_bundle = MODELS_DIR / "evaluation_bundle.json"
    with open(out_bundle, "w") as f:
        json.dump(bundle, f, indent=2)
    print(f"-> Saved evaluation bundle to {out_bundle}")
    print("Dataset intelligence build complete!")


if __name__ == "__main__":
    main()
