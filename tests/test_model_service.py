"""
test_model_service.py
---------------------
Unit tests for ModelService and inference logic.
"""

import json
import pytest
import pandas as pd
import numpy as np

from src.config import SAMPLE_PRESETS_JSON
from src.model_service import ModelService, ValidationError


@pytest.fixture(scope="module")
def model_service():
    service = ModelService.get_instance()
    service.ensure_loaded()
    return service


@pytest.fixture(scope="module")
def presets():
    assert SAMPLE_PRESETS_JSON.exists(), "sample_presets.json must exist"
    with open(SAMPLE_PRESETS_JSON, "r") as f:
        return json.load(f)


def test_service_initialization(model_service):
    assert model_service.is_loaded is True
    assert model_service.model is not None
    assert model_service.scaler is not None
    assert len(model_service.feature_columns) == 30


def test_predict_single_legitimate(model_service, presets):
    legit_samples = [p for p in presets if p["type"] == "legitimate"]
    assert len(legit_samples) > 0

    sample = legit_samples[0]
    result = model_service.predict_single(sample["features"])

    assert "risk_score" in result
    assert 0 <= result["risk_score"] <= 100
    assert result["risk_level"] in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    assert result["prediction"] in [0, 1]
    # Legitimate transactions should have low risk
    assert result["risk_score"] <= 50
    assert result["prediction"] == 0


def test_predict_single_fraud(model_service, presets):
    fraud_samples = [p for p in presets if p["type"] == "fraud"]
    assert len(fraud_samples) > 0

    sample = fraud_samples[0]
    result = model_service.predict_single(sample["features"])

    assert 0 <= result["risk_score"] <= 100
    # Confirmed real fraud sample should have high risk score
    assert result["risk_score"] >= 60
    assert result["is_flagged"] is True


def test_predict_batch(model_service, presets):
    rows = [p["features"] for p in presets[:5]]
    df = pd.DataFrame(rows)

    enriched, summary = model_service.predict_batch(df)

    assert "Predicted_Class" in enriched.columns
    assert "Fraud_Probability" in enriched.columns
    assert "Risk_Score" in enriched.columns
    assert "Risk_Level" in enriched.columns

    assert summary["total_transactions"] == 5
    assert "flagged_transactions" in summary
    assert "fraud_rate_pct" in summary
    assert "risk_distribution" in summary


def test_custom_threshold(model_service, presets):
    fraud_sample = next(p for p in presets if p["type"] == "fraud")
    res_sensitive = model_service.predict_single(fraud_sample["features"], threshold=0.10)
    assert res_sensitive["threshold_used"] == 0.10

    res_strict = model_service.predict_single(fraud_sample["features"], threshold=0.95)
    assert res_strict["threshold_used"] == 0.95


def test_invalid_input_raises_validation_error(model_service):
    with pytest.raises(ValidationError):
        model_service.predict_single({"invalid": 123})
