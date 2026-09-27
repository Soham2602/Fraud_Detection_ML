"""
test_database.py
----------------
Unit tests for SQLite database persistence, review workflows, and KPI calculations.
"""

import pytest
import tempfile
from pathlib import Path

from src.database import (
    init_db,
    save_transaction,
    get_recent_transactions,
    get_transaction_by_id,
    add_review_action,
    get_alerts,
    get_dashboard_kpis,
)


@pytest.fixture(autouse=True)
def setup_test_db(monkeypatch, tmp_path):
    """Point DB_PATH to a temporary file for isolated test execution."""
    test_db = tmp_path / "test_sentinel.db"
    monkeypatch.setattr("src.database.DB_PATH", test_db)
    init_db()


def test_save_and_retrieve_legitimate_transaction():
    record = {
        "transaction_id": "TXN-TEST-001",
        "amount": 42.50,
        "time": 3600.0,
        "prediction": 0,
        "is_flagged": False,
        "fraud_probability": 0.05,
        "risk_score": 5,
        "risk_level": "LOW",
        "threshold_used": 0.50,
        "features_raw": {"Amount": 42.50, "Time": 3600.0, "V1": 0.1},
        "demo_channel": "Online Web",
        "demo_category": "Retail",
        "demo_location": "Mumbai, IN",
    }
    txn_id = save_transaction(record)
    assert txn_id == "TXN-TEST-001"

    txns = get_recent_transactions(limit=10)
    assert len(txns) == 1
    assert txns[0]["id"] == "TXN-TEST-001"
    assert txns[0]["is_flagged"] == 0
    assert txns[0]["risk_score"] == 5

    # Should have no alerts for legitimate transaction
    alerts = get_alerts()
    assert len(alerts) == 0


def test_save_flagged_transaction_creates_alert():
    record = {
        "transaction_id": "TXN-TEST-002",
        "amount": 1250.00,
        "time": 7200.0,
        "prediction": 1,
        "is_flagged": True,
        "fraud_probability": 0.92,
        "risk_score": 92,
        "risk_level": "CRITICAL",
        "threshold_used": 0.50,
        "features_raw": {"Amount": 1250.00, "Time": 7200.0, "V14": -8.5},
    }
    save_transaction(record)

    alerts = get_alerts()
    assert len(alerts) == 1
    assert alerts[0]["transaction_id"] == "TXN-TEST-002"
    assert alerts[0]["risk_score"] == 92
    assert alerts[0]["status"] == "Open"


def test_review_action_workflow():
    record = {
        "transaction_id": "TXN-TEST-003",
        "amount": 800.00,
        "time": 1000.0,
        "prediction": 1,
        "is_flagged": True,
        "fraud_probability": 0.85,
        "risk_score": 85,
        "risk_level": "HIGH",
        "threshold_used": 0.50,
        "features_raw": {"Amount": 800.00, "Time": 1000.0},
    }
    save_transaction(record)

    # Perform analyst action: Mark reviewed
    success = add_review_action(
        "TXN-TEST-003",
        action="Mark as reviewed",
        reviewer="Analyst Jane",
        notes="Customer verified transaction via phone verification.",
    )
    assert success is True

    details = get_transaction_by_id("TXN-TEST-003")
    assert details is not None
    assert details["status"] == "Reviewed"
    assert len(details["reviews"]) == 1
    assert details["reviews"][0]["reviewer"] == "Analyst Jane"
    assert "phone verification" in details["reviews"][0]["notes"]


def test_dashboard_kpis():
    # Save 1 legit and 1 fraud
    save_transaction({
        "transaction_id": "T1", "amount": 100.0, "time": 1.0, "prediction": 0,
        "is_flagged": False, "fraud_probability": 0.1, "risk_score": 10, "risk_level": "LOW"
    })
    save_transaction({
        "transaction_id": "T2", "amount": 900.0, "time": 2.0, "prediction": 1,
        "is_flagged": True, "fraud_probability": 0.9, "risk_score": 90, "risk_level": "CRITICAL"
    })

    kpis = get_dashboard_kpis()
    assert kpis["total_transactions"] == 2
    assert kpis["fraud_flags"] == 1
    assert kpis["high_risk_transactions"] == 1
    assert kpis["fraud_rate_pct"] == 50.0
    assert kpis["avg_amount"] == 500.0
