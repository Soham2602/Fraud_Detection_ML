"""
test_simulation.py
------------------
Unit tests for live simulation and What-If sensitivity engine.
"""

import json
import pytest
from src.config import SAMPLE_PRESETS_JSON
from src.simulation import SimulationEngine


@pytest.fixture(scope="module")
def sim_engine():
    return SimulationEngine()


@pytest.fixture(scope="module")
def presets():
    with open(SAMPLE_PRESETS_JSON, "r") as f:
        return json.load(f)


def test_generate_single_simulated_transaction(sim_engine):
    res = sim_engine.generate_single_simulated_transaction(fraud_bias=0.0, persist=False)
    assert res["is_simulated"] is True
    assert "demo_channel" in res
    assert "demo_category" in res
    assert "demo_location" in res
    assert "risk_score" in res
    assert 0 <= res["risk_score"] <= 100


def test_what_if_analysis(sim_engine, presets):
    base = presets[0]["features"]
    # Change amount from normal to large
    mods = {"Amount": 12000.0}
    res = sim_engine.run_what_if_analysis(base, mods)

    assert "original" in res
    assert "modified" in res
    assert "delta" in res
    assert "assessment" in res["delta"]
    assert "disclaimer" in res
    assert res["modified"]["amount"] == 12000.0
