"""
test_explainability.py
----------------------
Unit tests for XAI / SHAP attribution engine.
"""

import json
import pytest
from src.config import SAMPLE_PRESETS_JSON
from src.explainability import ExplainabilityEngine


@pytest.fixture(scope="module")
def explainer():
    return ExplainabilityEngine.get_instance()


@pytest.fixture(scope="module")
def presets():
    with open(SAMPLE_PRESETS_JSON, "r") as f:
        return json.load(f)


def test_explainability_initialization(explainer):
    assert explainer.service.is_loaded is True


def test_explain_single_transaction(explainer, presets):
    sample = presets[0]["features"]
    res = explainer.explain_transaction(sample, top_k=6)

    assert "top_features" in res
    assert len(res["top_features"]) == 6
    assert "narrative" in res
    assert len(res["narrative"]) > 0
    assert "method" in res

    first_feat = res["top_features"][0]
    assert "feature" in first_feat
    assert "contribution" in first_feat
    assert "raw_value" in first_feat


def test_global_feature_importance(explainer):
    importances = explainer.get_global_feature_importance()
    assert len(importances) == 30
    assert importances[0]["importance"] >= importances[-1]["importance"]
