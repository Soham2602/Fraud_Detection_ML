"""
test_api.py
------------
Unit tests for FastAPI REST API endpoints and data provenance contracts.
"""

import pytest
from fastapi.testclient import TestClient

from src.api import app
from src.config import SAMPLE_PRESETS_JSON
from src.model_service import ModelService
import json


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


@pytest.fixture(scope="module")
def presets():
    with open(SAMPLE_PRESETS_JSON, "r") as f:
        return json.load(f)


def test_root_endpoint(client):
    # Test JSON metadata request
    response = client.get("/", headers={"accept": "application/json"})
    assert response.status_code == 200
    data = response.json()
    assert "SENTINEL" in data["platform"]

    # Test HTML dashboard request
    html_resp = client.get("/", headers={"accept": "text/html"})
    assert html_resp.status_code == 200
    assert "SENTINEL" in html_resp.text


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_predict_single_endpoint(client, presets):
    sample = presets[0]["features"]
    response = client.post("/predict", json=sample)
    assert response.status_code == 200
    data = response.json()
    assert "risk_score" in data
    assert "risk_level" in data
    assert 0 <= data["risk_score"] <= 100


def test_predict_batch_endpoint(client, presets):
    batch = [p["features"] for p in presets[:3]]
    response = client.post("/predict/batch", json={"transactions": batch})
    assert response.status_code == 200
    data = response.json()
    assert "summary" in data
    assert "results" in data
    assert len(data["results"]) == 3


def test_simulate_endpoint(client):
    response = client.post("/simulate?count=2&fraud_bias=0.2")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 2
    assert "risk_score" in data[0]


def test_kpis_endpoint(client):
    response = client.get("/kpis")
    assert response.status_code == 200
    data = response.json()
    assert "total_transactions" in data
    assert "fraud_flags" in data


def test_dataset_intelligence_endpoint(client):
    response = client.get("/dataset-intelligence")
    assert response.status_code == 200
    data = response.json()
    assert "overview" in data
    assert data["overview"]["total_transactions"] == 284807
    assert data["overview"]["fraud_count"] == 492
    assert "amount_statistics" in data
    assert "amount_distribution" in data


def test_evaluation_bundle_endpoint(client):
    response = client.get("/evaluation-bundle")
    assert response.status_code == 200
    data = response.json()
    assert "default_metrics" in data
    assert data["default_metrics"]["precision"] >= 0.90
    assert data["default_metrics"]["recall"] >= 0.70
    assert "threshold_sweep" in data
    assert len(data["threshold_sweep"]) >= 50


def test_threshold_analysis_endpoint(client):
    r_low = client.get("/threshold-analysis?threshold=0.20").json()
    r_high = client.get("/threshold-analysis?threshold=0.80").json()

    assert r_low["recall"] >= r_high["recall"], "Lower threshold must have >= recall"
    assert r_low["operational_mode"] == "HIGH_SENSITIVITY"
    assert r_high["operational_mode"] == "CONSERVATIVE"


def test_risk_score_determinism(presets):
    service = ModelService.get_instance()
    sample = presets[0]["features"]
    scores = [service.predict_single(sample)["risk_score"] for _ in range(5)]
    assert len(set(scores)) == 1, f"Risk scoring must be strictly deterministic across calls: {scores}"


def test_demo_api_endpoints(client):
    # Test /scenarios
    resp = client.get("/scenarios")
    assert resp.status_code == 200
    scenarios = resp.json()
    assert len(scenarios) == 5

    # Test /demo-transactions
    resp_tx = client.get("/demo-transactions?limit=10")
    assert resp_tx.status_code == 200
    txns = resp_tx.json()
    assert len(txns) == 10
    assert "risk_score" in txns[0]

    # Test /demo-alerts
    resp_al = client.get("/demo-alerts?limit=5")
    assert resp_al.status_code == 200
    alerts = resp_al.json()
    assert len(alerts) == 5

    # Test /demo-kpis
    resp_kp = client.get("/demo-kpis")
    assert resp_kp.status_code == 200
    kpis = resp_kp.json()
    assert kpis["total_transactions"] == 500

