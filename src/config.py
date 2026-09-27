"""
config.py
---------
Centralized configuration for SENTINEL — Fraud Intelligence Platform.
Manages file paths, model parameters, risk score bands, and simulation settings.
"""

import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
FIGURES_DIR = BASE_DIR / "figures"
EXPERIMENTS_DIR = BASE_DIR / "experiments"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"

# Data Files
CREDITCARD_CSV = DATA_DIR / "creditcard.csv"
SAMPLE_PRESETS_JSON = DATA_DIR / "sample_presets.json"
DATASET_SUMMARY_JSON = DATA_DIR / "dataset_summary.json"

# Model Artifacts
MODEL_PKL = MODELS_DIR / "fraud_detection_model.pkl"
SCALER_PKL = MODELS_DIR / "scaler.pkl"
FEATURES_PKL = MODELS_DIR / "feature_columns.pkl"
METADATA_JSON = MODELS_DIR / "model_metadata.json"
EVALUATION_BUNDLE_JSON = MODELS_DIR / "evaluation_bundle.json"
EXPERIMENT_RESULTS_CSV = EXPERIMENTS_DIR / "results.csv"

# Database Path - Supports Vercel Serverless /tmp fallback
IS_VERCEL = bool(
    os.environ.get("VERCEL")
    or os.environ.get("AWS_LAMBDA_FUNCTION_NAME")
    or os.environ.get("LAMBDA_TASK_ROOT")
    or not os.access(str(BASE_DIR), os.W_OK)
)

if IS_VERCEL:
    DB_PATH = Path("/tmp") / "sentinel.db"
else:
    DB_PATH = DATA_DIR / "sentinel.db"

# Ensure runtime directories exist only if filesystem is writable
if not IS_VERCEL:
    for folder in [MODELS_DIR, FIGURES_DIR, EXPERIMENTS_DIR, DATA_DIR]:
        try:
            folder.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass

# Feature Schema
FEATURE_COLUMNS = [
    "Time",
    "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10",
    "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20",
    "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28",
    "Amount",
]
SCALE_COLUMNS = ["Time", "Amount"]
TARGET_COLUMN = "Class"

# Risk Thresholds and Scoring
DEFAULT_THRESHOLD = 0.50
RISK_BANDS = [
    {"label": "LOW", "min": 0, "max": 30, "color": "#10b981", "badge": "bg-emerald-500/10 text-emerald-400"},
    {"label": "MEDIUM", "min": 31, "max": 60, "color": "#f59e0b", "badge": "bg-amber-500/10 text-amber-400"},
    {"label": "HIGH", "min": 61, "max": 80, "color": "#f97316", "badge": "bg-orange-500/10 text-orange-400"},
    {"label": "CRITICAL", "min": 81, "max": 100, "color": "#ef4444", "badge": "bg-rose-500/10 text-rose-400"},
]

def get_risk_level(risk_score: int) -> dict:
    """Return risk band metadata for a given score (0-100)."""
    for band in RISK_BANDS:
        if band["min"] <= risk_score <= band["max"]:
            return band
    return RISK_BANDS[-1]

# Demo/Simulation Channels and Categories (Clearly marked as synthetic/demo metadata)
DEMO_CHANNELS = ["Online Web", "POS Terminal", "Mobile App", "Contactless NFC", "ATM"]
DEMO_CATEGORIES = ["Electronics", "Groceries", "Luxury Goods", "Travel & Airlines", "Dining", "Digital Services", "Jewelry"]
DEMO_LOCATIONS = ["Mumbai, IN", "London, UK", "New York, US", "Singapore, SG", "Berlin, DE", "Toronto, CA", "Sydney, AU"]
