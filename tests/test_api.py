"""
test_api.py
------------
Unit tests for FastAPI REST API endpoints.
"""

import pytest
from fastapi.testclient import TestClient

from src.api import app
from src.config import SAMPLE_PRESETS_JSON
import json


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


@pytest.fixture(scope="module")
def presets():
    with open(SAMPLE_PRESETS_JSON, "r") as f:
        return json.load(f)


def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "SENTINEL" in data["platform"]


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
