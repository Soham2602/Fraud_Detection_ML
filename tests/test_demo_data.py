"""
test_demo_data.py
------------------
Unit tests for deterministic synthetic demonstration data module (src/demo_data.py).
"""

import pytest
import pandas as pd
from src.demo_data import (
    DEMO_SEED,
    DEMO_TRANSACTION_COUNT,
    DEMO_ALERT_COUNT,
    DEMO_SCENARIOS,
    get_demo_transactions,
    get_demo_alerts,
    get_demo_kpis,
    get_demo_scenarios,
    get_demo_channel_breakdown,
    get_demo_category_breakdown,
)
from src.config import FEATURE_COLUMNS


def test_demo_scenarios_structure():
    """Verify that all 5 curated scenarios exist and contain full 30-feature vectors."""
    scenarios = get_demo_scenarios()
    assert len(scenarios) == 5

    required_scenarios = [
        "Scenario A: Low-Risk Everyday Purchase (Legitimate)",
        "Scenario B: Moderate-Risk Online Electronics (Elevated Drift)",
        "Scenario C: High-Risk Off-Hours International Travel (Flagged)",
        "Scenario D: Critical Coordinated Attack (Micro-Auth Testing)",
        "Scenario E: Borderline Ambiguous Case (Threshold Sensitivity)",
    ]
    for s_name in required_scenarios:
        assert s_name in scenarios
        sc_data = scenarios[s_name]
        assert "features" in sc_data
        assert "prediction_result" in sc_data
        for col in FEATURE_COLUMNS:
            assert col in sc_data["features"], f"Missing {col} in {s_name}"

        # Prediction result structure
        pred = sc_data["prediction_result"]
        assert "fraud_probability" in pred
        assert "risk_score" in pred
        assert "risk_level" in pred
        assert "is_flagged" in pred


def test_demo_transactions_deterministic_shape():
    """Verify that get_demo_transactions returns exactly 500 rows with required columns."""
    df1 = get_demo_transactions(500)
    assert len(df1) == 500
    assert isinstance(df1, pd.DataFrame)

    required_cols = [
        "id", "timestamp", "time_elapsed", "amount", "channel",
        "category", "location", "fraud_probability", "risk_score",
        "risk_level", "prediction", "is_flagged", "status",
        "is_synthetic", "provenance"
    ]
    for col in required_cols:
        assert col in df1.columns

    # Verify deterministic reproducibility
    df2 = get_demo_transactions(500)
    pd.testing.assert_frame_equal(df1, df2)


def test_demo_alerts():
    """Verify that get_demo_alerts returns 30 prioritized alerts."""
    alerts = get_demo_alerts(30)
    assert len(alerts) == 30
    for a in alerts:
        assert "id" in a
        assert "transaction_id" in a
        assert "risk_score" in a
        assert "risk_level" in a
        assert "fraud_probability" in a
        assert a["provenance"] == "SYNTHETIC DEMO"


def test_demo_kpis():
    """Verify that get_demo_kpis calculates realistic aggregate metrics."""
    kpis = get_demo_kpis()
    assert kpis["total_transactions"] == 500
    assert kpis["fraud_flags"] > 0
    assert kpis["high_risk_transactions"] > 0
    assert kpis["fraud_rate_pct"] > 0
    assert kpis["avg_amount"] > 0
    assert kpis["provenance"] == "SYNTHETIC DEMO"


def test_demo_breakdowns():
    """Verify channels and category distribution lookups."""
    channels = get_demo_channel_breakdown()
    assert len(channels) >= 3
    assert sum(channels.values()) == 500

    categories = get_demo_category_breakdown()
    assert len(categories) >= 3
    assert sum(categories.values()) == 500
