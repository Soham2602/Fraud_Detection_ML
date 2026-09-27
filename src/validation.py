"""
validation.py
-------------
Defensive data validation for SENTINEL — Fraud Intelligence Platform.
Validates single transaction payloads and batch DataFrames before inference.
Provides human-readable validation results without leaking Python stack traces.
"""

import sys
from pathlib import Path
_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from config import FEATURE_COLUMNS, SCALE_COLUMNS

class ValidationError(Exception):
    """Custom exception containing structured human-readable errors."""
    def __init__(self, message: str, errors: Optional[List[str]] = None):
        super().__init__(message)
        self.message = message
        self.errors = errors or [message]


def validate_single_transaction(data: Dict[str, Any]) -> Tuple[bool, List[str], Optional[Dict[str, float]]]:
    """
    Validate a single transaction dictionary.
    
    Checks:
    1. Input is a dictionary
    2. Required columns are present (Time, V1..V28, Amount)
    3. Values are castable to float
    4. No NaN, None, or infinite values
    5. Amount >= 0, Time >= 0
    
    Returns:
        (is_valid, error_list, cleaned_dict)
    """
    errors: List[str] = []
    cleaned: Dict[str, float] = {}

    if not isinstance(data, dict):
        return False, ["Input must be a JSON object / dictionary of features."], None

    # Check missing columns
    missing = [c for c in FEATURE_COLUMNS if c not in data]
    if missing:
        errors.append(f"Missing required features ({len(missing)} total): {', '.join(missing[:6])}{'...' if len(missing) > 6 else ''}")

    for col in FEATURE_COLUMNS:
        if col not in data:
            continue
        val = data[col]

        if val is None:
            errors.append(f"Feature '{col}' cannot be null/empty.")
            continue

        try:
            fval = float(val)
        except (ValueError, TypeError):
            errors.append(f"Feature '{col}' must be a valid number, got: {val!r}")
            continue

        if np.isnan(fval) or np.isinf(fval):
            errors.append(f"Feature '{col}' contains an invalid NaN or Infinite value.")
            continue

        cleaned[col] = fval

    # Logical value checks
    if "Amount" in cleaned and cleaned["Amount"] < 0:
        errors.append("Transaction Amount cannot be negative.")

    if "Time" in cleaned and cleaned["Time"] < 0:
        errors.append("Transaction Time cannot be negative.")

    is_valid = len(errors) == 0
    return is_valid, errors, cleaned if is_valid else None


def validate_batch_dataframe(df: pd.DataFrame) -> Tuple[bool, List[str], Optional[pd.DataFrame]]:
    """
    Validate a batch DataFrame prior to running batch inference.
    
    Checks:
    1. Input is a non-empty DataFrame
    2. All expected columns exist
    3. Types are numeric
    4. NaN / null / infinite values
    5. Logical bounds
    
    Returns:
        (is_valid, error_list, cleaned_df)
    """
    errors: List[str] = []

    if df is None or not isinstance(df, pd.DataFrame):
        return False, ["Uploaded data is not a valid DataFrame."], None

    if len(df) == 0:
        return False, ["The uploaded dataset contains zero rows."], None

    # Column presence check
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        errors.append(f"Missing required columns ({len(missing)}): {', '.join(missing[:8])}{'...' if len(missing) > 8 else ''}")
        return False, errors, None

    # Clean subset in expected column order
    df_clean = df[FEATURE_COLUMNS].copy()

    # Check non-numeric types
    non_numeric = []
    for col in FEATURE_COLUMNS:
        if not pd.api.types.is_numeric_dtype(df_clean[col]):
            try:
                df_clean[col] = pd.to_numeric(df_clean[col])
            except Exception:
                non_numeric.append(col)

    if non_numeric:
        errors.append(f"Columns containing non-numeric values: {', '.join(non_numeric)}")

    # Check null / NaN / inf
    nan_counts = df_clean.isna().sum()
    cols_with_nans = nan_counts[nan_counts > 0]
    if len(cols_with_nans) > 0:
        errors.append(f"Found missing / NaN values in columns: {dict(cols_with_nans)}")

    inf_counts = np.isinf(df_clean.select_dtypes(include=[np.number])).sum()
    cols_with_inf = inf_counts[inf_counts > 0]
    if len(cols_with_inf) > 0:
        errors.append(f"Found infinite values in columns: {dict(cols_with_inf)}")

    if "Amount" in df_clean.columns:
        neg_amounts = (df_clean["Amount"] < 0).sum()
        if neg_amounts > 0:
            errors.append(f"Found {neg_amounts} rows with negative Transaction Amount.")

    is_valid = len(errors) == 0
    return is_valid, errors, df_clean if is_valid else None
