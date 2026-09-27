"""
test_validation.py
------------------
Unit tests for data validation module.
"""

import pytest
import pandas as pd
import numpy as np

from src.config import FEATURE_COLUMNS
from src.validation import (
    validate_single_transaction,
    validate_batch_dataframe,
    ValidationError,
)


@pytest.fixture
def valid_transaction():
    data = {f"V{i}": 0.0 for i in range(1, 29)}
    data["Time"] = 1000.0
    data["Amount"] = 50.0
    return data


def test_valid_single_transaction(valid_transaction):
    is_valid, errors, cleaned = validate_single_transaction(valid_transaction)
    assert is_valid is True
    assert len(errors) == 0
    assert cleaned["Amount"] == 50.0
    assert cleaned["Time"] == 1000.0
    assert len(cleaned) == 30


def test_missing_feature(valid_transaction):
    del valid_transaction["V14"]
    is_valid, errors, cleaned = validate_single_transaction(valid_transaction)
    assert is_valid is False
    assert any("V14" in err for err in errors)
    assert cleaned is None


def test_negative_amount(valid_transaction):
    valid_transaction["Amount"] = -10.5
    is_valid, errors, cleaned = validate_single_transaction(valid_transaction)
    assert is_valid is False
    assert any("negative" in err.lower() for err in errors)


def test_nan_value(valid_transaction):
    valid_transaction["V1"] = float("nan")
    is_valid, errors, cleaned = validate_single_transaction(valid_transaction)
    assert is_valid is False
    assert any("nan" in err.lower() for err in errors)


def test_infinite_value(valid_transaction):
    valid_transaction["V2"] = float("inf")
    is_valid, errors, cleaned = validate_single_transaction(valid_transaction)
    assert is_valid is False
    assert any("infinite" in err.lower() for err in errors)


def test_non_numeric_value(valid_transaction):
    valid_transaction["V5"] = "invalid_string"
    is_valid, errors, cleaned = validate_single_transaction(valid_transaction)
    assert is_valid is False
    assert any("valid number" in err.lower() for err in errors)


def test_valid_batch_dataframe(valid_transaction):
    df = pd.DataFrame([valid_transaction, valid_transaction])
    is_valid, errors, df_clean = validate_batch_dataframe(df)
    assert is_valid is True
    assert len(errors) == 0
    assert len(df_clean) == 2


def test_empty_dataframe():
    df = pd.DataFrame()
    is_valid, errors, df_clean = validate_batch_dataframe(df)
    assert is_valid is False
    assert any("zero rows" in err.lower() for err in errors)


def test_missing_batch_columns(valid_transaction):
    df = pd.DataFrame([valid_transaction])
    df = df.drop(columns=["V28"])
    is_valid, errors, df_clean = validate_batch_dataframe(df)
    assert is_valid is False
    assert any("missing" in err.lower() for err in errors)
