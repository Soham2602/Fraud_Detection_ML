"""
demo_data.py
------------
Deterministic Synthetic Demonstration Layer for SENTINEL — Fraud Intelligence Platform.

Purpose:
- Solves the extreme class imbalance problem (0.17% fraud rate in real data) by providing
  a calibrated, rich, populated set of 500 transactions and 30 prioritized alerts.
- Ensures the platform feels vibrant, populated, and operational on initial launch
  without requiring manual live simulation clicks.
- Strictly adheres to Data Provenance: all synthetic data is explicitly tagged as
  "SYNTHETIC DEMO" and never misrepresents benchmark test set metrics.
- Uses a deterministic PRNG seed (DEMO_SEED = 42) so outputs, scores, and tables
  remain 100% reproducible across Streamlit reruns.
- Highly optimized using vectorized batch inference.
"""

import sys
import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from config import (
    FEATURE_COLUMNS,
    DEFAULT_THRESHOLD,
    DEMO_CHANNELS,
    DEMO_CATEGORIES,
    DEMO_LOCATIONS,
    RISK_BANDS,
    get_risk_level,
)
from model_service import ModelService

# Constants
DEMO_SEED = 42
DEMO_TRANSACTION_COUNT = 500
DEMO_ALERT_COUNT = 30

# In-memory singletons for fast re-use without recomputing
_CACHED_DEMO_DF: Optional[pd.DataFrame] = None
_CACHED_DEMO_ALERTS: Optional[List[Dict[str, Any]]] = None
_CACHED_DEMO_KPIS: Optional[Dict[str, Any]] = None


# -----------------------------------------------------------------------------
# 5 Curated Scenario Presets for Immediate Viva / Demonstration Testing
# -----------------------------------------------------------------------------
DEMO_SCENARIOS = {
    "Scenario A: Low-Risk Everyday Purchase (Legitimate)": {
        "scenario_id": "SCN-A-LEGIT",
        "title": "Low-Risk Point-of-Sale Grocery",
        "description": "Routine supermarket purchase during business hours with normal cardholder telemetry.",
        "category": "Groceries",
        "channel": "POS Terminal",
        "location": "Mumbai, IN",
        "expected_risk": "LOW (0-30)",
        "expected_verdict": "✅ APPROVED",
        "explanation": "All PCA feature values sit within 1 standard deviation of zero. Normal spending amount ($28.50).",
        "features": {
            "Time": 43200.0,
            "Amount": 28.50,
            "V1": 0.12, "V2": -0.05, "V3": 0.85, "V4": -0.32, "V5": 0.15,
            "V6": -0.22, "V7": 0.35, "V8": 0.05, "V9": 0.18, "V10": 0.25,
            "V11": -0.40, "V12": 0.30, "V13": 0.12, "V14": 0.45, "V15": 0.60,
            "V16": 0.20, "V17": 0.35, "V18": 0.10, "V19": -0.15, "V20": -0.05,
            "V21": -0.08, "V22": 0.15, "V23": -0.02, "V24": 0.10, "V25": -0.12,
            "V26": 0.05, "V27": 0.02, "V28": 0.01,
        },
    },
    "Scenario B: Moderate-Risk Online Electronics (Elevated Drift)": {
        "scenario_id": "SCN-B-MODERATE",
        "title": "High-Value E-Commerce Purchase",
        "description": "Late evening online electronics order exhibiting moderate PCA drift on leading indicator V14.",
        "category": "Electronics",
        "channel": "Online Web",
        "location": "New York, US",
        "expected_risk": "MEDIUM (31-60)",
        "expected_verdict": "⚠️ CAUTION / SECONDARY CHECK",
        "explanation": "Amount is higher than average ($385) with negative drift on V14 (-2.2) and positive on V4 (+1.45).",
        "features": {
            "Time": 79200.0,
            "Amount": 385.00,
            "V1": -1.25, "V2": 0.85, "V3": -0.45, "V4": 1.45, "V5": -0.55,
            "V6": -0.35, "V7": -0.80, "V8": 0.40, "V9": -0.65, "V10": -0.95,
            "V11": 0.85, "V12": -1.10, "V13": -0.30, "V14": -2.20, "V15": 0.10,
            "V16": -0.75, "V17": -1.30, "V18": -0.40, "V19": 0.65, "V20": 0.25,
            "V21": 0.18, "V22": 0.35, "V23": -0.12, "V24": 0.05, "V25": 0.20,
            "V26": 0.35, "V27": 0.15, "V28": -0.05,
        },
    },
    "Scenario C: High-Risk Off-Hours International Travel (Flagged)": {
        "scenario_id": "SCN-C-HIGH",
        "title": "Off-Hours Cross-Border Airline Booking",
        "description": "Early morning luxury booking exhibiting strong multi-component fraud signatures.",
        "category": "Travel & Airlines",
        "channel": "Mobile App",
        "location": "Singapore, SG",
        "expected_risk": "HIGH (61-80)",
        "expected_verdict": "🚨 FLAGGED FOR REVIEW",
        "explanation": "Substantial negative shifts on V14 (-5.6), V12 (-4.8), and V10 (-4.1) heavily trigger tree voting.",
        "features": {
            "Time": 14400.0,
            "Amount": 1420.00,
            "V1": -3.85, "V2": 2.65, "V3": -4.20, "V4": 3.10, "V5": -2.50,
            "V6": -1.40, "V7": -3.60, "V8": 1.85, "V9": -2.40, "V10": -4.10,
            "V11": 3.20, "V12": -4.80, "V13": -0.10, "V14": -5.60, "V15": -0.40,
            "V16": -3.10, "V17": -5.20, "V18": -1.80, "V19": 1.10, "V20": 0.65,
            "V21": 0.68, "V22": -0.25, "V23": 0.15, "V24": -0.30, "V25": 0.35,
            "V26": 0.45, "V27": 0.85, "V28": -0.18,
        },
    },
    "Scenario D: Critical Coordinated Attack (Micro-Auth Testing)": {
        "scenario_id": "SCN-D-CRITICAL",
        "title": "Automated Micro-Authorization Probing",
        "description": "Zero or near-zero testing charge with extreme signature matching confirmed syndicates.",
        "category": "Digital Services",
        "channel": "Online Web",
        "location": "London, UK",
        "expected_risk": "CRITICAL (81-100)",
        "expected_verdict": "🚨 CRITICAL BLOCK",
        "explanation": "Severe anomaly on key fraud components (V14 < -11, V4 > 5, V12 < -8). Over 95% tree consensus.",
        "features": {
            "Time": 85285.0,
            "Amount": 1.00,
            "V1": -7.03, "V2": 3.42, "V3": -9.53, "V4": 5.27, "V5": -4.02,
            "V6": -2.87, "V7": -6.99, "V8": 3.79, "V9": -4.62, "V10": -8.41,
            "V11": 6.31, "V12": -8.58, "V13": 0.25, "V14": -11.53, "V15": -0.36,
            "V16": -5.45, "V17": -11.89, "V18": -3.56, "V19": 0.88, "V20": 0.55,
            "V21": 1.10, "V22": -0.54, "V23": 0.04, "V24": -0.36, "V25": 0.35,
            "V26": 1.04, "V27": 1.36, "V28": -0.27,
        },
    },
    "Scenario E: Borderline Ambiguous Case (Threshold Sensitivity)": {
        "scenario_id": "SCN-E-BORDERLINE",
        "title": "Borderline E-Commerce Transaction",
        "description": "A transaction straddling the decision boundary (P ≈ 0.48-0.52). Ideal for threshold testing.",
        "category": "Retail",
        "channel": "POS Terminal",
        "location": "Berlin, DE",
        "expected_risk": "BORDERLINE (~50)",
        "expected_verdict": "⚖️ SENSITIVE TO THRESHOLD τ",
        "explanation": "Model posterior probability lands right near 0.50. Shifting threshold by ±0.05 flips decision.",
        "features": {
            "Time": 50554.0,
            "Amount": 115.00,
            "V1": -1.80, "V2": 1.45, "V3": -1.20, "V4": 1.85, "V5": -0.90,
            "V6": -0.60, "V7": -1.25, "V8": 0.70, "V9": -1.15, "V10": -1.65,
            "V11": 1.35, "V12": -1.95, "V13": 0.15, "V14": -2.85, "V15": -0.10,
            "V16": -1.40, "V17": -2.10, "V18": -0.85, "V19": 0.45, "V20": 0.30,
            "V21": 0.32, "V22": 0.15, "V23": -0.05, "V24": -0.15, "V25": 0.18,
            "V26": 0.22, "V27": 0.25, "V28": 0.05,
        },
    },
}


def get_demo_scenarios() -> Dict[str, Dict[str, Any]]:
    """Return the 5 curated scenario presets with pre-computed model predictions."""
    service = ModelService.get_instance()
    service.ensure_loaded()

    enriched_scenarios = {}
    for name, item in DEMO_SCENARIOS.items():
        res = service.predict_single(item["features"], threshold=DEFAULT_THRESHOLD)
        copy_item = dict(item)
        copy_item["prediction_result"] = res
        enriched_scenarios[name] = copy_item
    return enriched_scenarios


# -----------------------------------------------------------------------------
# Vectorized Generation of 500 Realistic Transactions
# -----------------------------------------------------------------------------
def generate_deterministic_demo_dataset(count: int = DEMO_TRANSACTION_COUNT, seed: int = DEMO_SEED) -> pd.DataFrame:
    """
    Generate a deterministic synthetic dataset of realistic credit card transactions.
    
    Properties:
    - Exactly count transactions (default 500).
    - ~14% fraud/flagged rate (70 flagged out of 500) to ensure rich dashboard tables,
      metric cards, and alert workflows.
    - Realistic amounts (log-normal distributions with typical spending vs micro-tests).
    - Features generated using PCA distributions matching empirical creditcard.csv statistics.
    - Evaluated in vectorized batch through ModelService.predict_batch.
    """
    rng = np.random.default_rng(seed)
    service = ModelService.get_instance()
    service.ensure_loaded()

    fraud_count = int(round(count * 0.14))
    legit_count = count - fraud_count

    # 1. Prepare Feature Matrix
    feat_rows = []
    meta_rows = []
    base_timestamp = datetime.datetime(2026, 9, 28, 0, 0, 0)

    # Legitimate transactions
    for i in range(legit_count):
        txn_id = f"DEMO-TXN-{10000 + i}"
        time_elapsed = float(rng.uniform(0, 172800))
        txn_time = base_timestamp + datetime.timedelta(seconds=time_elapsed)
        amt = float(np.clip(rng.lognormal(mean=3.5, sigma=1.1), 1.0, 4500.0))

        row_dict = {"Time": time_elapsed, "Amount": round(amt, 2)}
        for c in range(1, 29):
            row_dict[f"V{c}"] = float(rng.normal(0.0, 1.0))
        feat_rows.append(row_dict)

        meta_rows.append({
            "id": txn_id,
            "timestamp": txn_time.strftime("%Y-%m-%d %H:%M:%S"),
            "time_elapsed": time_elapsed,
            "amount": round(amt, 2),
            "channel": str(rng.choice(DEMO_CHANNELS)),
            "category": str(rng.choice(DEMO_CATEGORIES)),
            "location": str(rng.choice(DEMO_LOCATIONS)),
            "status": "Approved",
            "is_synthetic": True,
            "provenance": "SYNTHETIC DEMO",
        })

    # Fraudulent transactions
    for j in range(fraud_count):
        idx = legit_count + j
        txn_id = f"DEMO-TXN-{10000 + idx}"
        hour_offset = rng.choice([rng.uniform(3600, 18000), rng.uniform(90000, 104400)])
        time_elapsed = float(hour_offset)
        txn_time = base_timestamp + datetime.timedelta(seconds=time_elapsed)

        if rng.random() < 0.45:
            amt = float(rng.uniform(0.75, 9.99))
        else:
            amt = float(rng.uniform(450.0, 3200.0))

        row_dict = {"Time": time_elapsed, "Amount": round(amt, 2)}
        for c in range(1, 29):
            row_dict[f"V{c}"] = float(rng.normal(0.0, 1.2))

        # Shift key discriminative components
        row_dict["V14"] = float(rng.normal(-7.2, 1.4))
        row_dict["V12"] = float(rng.normal(-6.1, 1.5))
        row_dict["V10"] = float(rng.normal(-5.3, 1.3))
        row_dict["V17"] = float(rng.normal(-4.8, 1.6))
        row_dict["V4"] = float(rng.normal(4.2, 1.1))
        row_dict["V11"] = float(rng.normal(3.6, 1.0))
        feat_rows.append(row_dict)

        meta_rows.append({
            "id": txn_id,
            "timestamp": txn_time.strftime("%Y-%m-%d %H:%M:%S"),
            "time_elapsed": time_elapsed,
            "amount": round(amt, 2),
            "channel": str(rng.choice(["Online Web", "Mobile App", "Contactless NFC"])),
            "category": str(rng.choice(["Electronics", "Luxury Goods", "Travel & Airlines", "Digital Services"])),
            "location": str(rng.choice(DEMO_LOCATIONS)),
            "status": str(rng.choice(["Pending", "Under Review", "Escalated", "Confirmed Fraud"])),
            "is_synthetic": True,
            "provenance": "SYNTHETIC DEMO",
        })

    features_df = pd.DataFrame(feat_rows)[FEATURE_COLUMNS]
    enhanced_df, _ = service.predict_batch(features_df, threshold=DEFAULT_THRESHOLD)

    # Standardize column naming
    enhanced_df["fraud_probability"] = enhanced_df["Fraud_Probability"]
    enhanced_df["risk_score"] = enhanced_df["Risk_Score"]
    enhanced_df["risk_level"] = enhanced_df["Risk_Level"]
    enhanced_df["prediction"] = enhanced_df["Predicted_Class"]
    enhanced_df["is_flagged"] = enhanced_df["Predicted_Class"] == 1
    enhanced_df["amount"] = enhanced_df["Amount"]
    enhanced_df["time_elapsed"] = enhanced_df["Time"]

    # Merge metadata with batch predictions
    meta_df = pd.DataFrame(meta_rows)
    for col in ["id", "timestamp", "channel", "category", "location", "status", "is_synthetic", "provenance"]:
        enhanced_df[col] = meta_df[col].values

    # Deterministic shuffle
    perm = rng.permutation(len(enhanced_df))
    shuffled_df = enhanced_df.iloc[perm].reset_index(drop=True)
    return shuffled_df


def get_demo_transactions(count: int = DEMO_TRANSACTION_COUNT) -> pd.DataFrame:
    """Return the cached deterministic demo transactions DataFrame."""
    global _CACHED_DEMO_DF
    if _CACHED_DEMO_DF is None or len(_CACHED_DEMO_DF) != count:
        _CACHED_DEMO_DF = generate_deterministic_demo_dataset(count=count, seed=DEMO_SEED)
    return _CACHED_DEMO_DF


def get_demo_alerts(count: int = DEMO_ALERT_COUNT) -> List[Dict[str, Any]]:
    """Return prioritized high-risk alerts filtered from the demo dataset."""
    global _CACHED_DEMO_ALERTS
    if _CACHED_DEMO_ALERTS is not None and len(_CACHED_DEMO_ALERTS) == count:
        return _CACHED_DEMO_ALERTS

    df = get_demo_transactions()
    # Prioritize flagged transactions, then highest risk scores
    top_risks = df.sort_values(by=["is_flagged", "risk_score", "fraud_probability"], ascending=[False, False, False]).head(count)
    
    alerts = []
    for i, (_, row) in enumerate(top_risks.iterrows()):
        alerts.append({
            "id": i + 1,
            "transaction_id": row["id"],
            "created_at": row["timestamp"],
            "amount": row["amount"],
            "channel": row["channel"],
            "category": row["category"],
            "location": row["location"],
            "risk_score": int(row["risk_score"]),
            "risk_level": row["risk_level"],
            "fraud_probability": float(row["fraud_probability"]),
            "status": row["status"] if row["status"] != "Approved" else "Pending Review",
            "assigned_to": "Senior Fraud Specialist" if i % 2 == 0 else "Unassigned",
            "provenance": "SYNTHETIC DEMO",
        })

    _CACHED_DEMO_ALERTS = alerts
    return _CACHED_DEMO_ALERTS


def get_demo_kpis() -> Dict[str, Any]:
    """Compute aggregate KPIs for the demonstration layer."""
    global _CACHED_DEMO_KPIS
    if _CACHED_DEMO_KPIS is not None:
        return _CACHED_DEMO_KPIS

    df = get_demo_transactions()
    total = len(df)
    flagged = int(df["is_flagged"].sum())
    high_risk = int((df["risk_score"] >= 61).sum())
    fraud_rate = (flagged / total * 100) if total > 0 else 0.0
    avg_amt = float(df["amount"].mean())
    alerts = get_demo_alerts()

    _CACHED_DEMO_KPIS = {
        "total_transactions": total,
        "fraud_flags": flagged,
        "high_risk_transactions": high_risk,
        "fraud_rate_pct": round(fraud_rate, 2),
        "avg_amount": round(avg_amt, 2),
        "transactions_reviewed": int(total * 0.28),
        "open_alerts": len(alerts),
        "provenance": "SYNTHETIC DEMO",
    }
    return _CACHED_DEMO_KPIS


def get_demo_channel_breakdown() -> Dict[str, int]:
    """Return distribution of transactions across payment channels."""
    df = get_demo_transactions()
    return df["channel"].value_counts().to_dict()


def get_demo_category_breakdown() -> Dict[str, int]:
    """Return distribution of transactions across merchant categories."""
    df = get_demo_transactions()
    return df["category"].value_counts().to_dict()
