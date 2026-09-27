"""
simulation.py
-------------
Live transaction simulation and What-If sensitivity engine for
SENTINEL — Fraud Intelligence Platform.

Generates realistic synthetic transaction streams, supports burst-attack simulations,
and provides What-If sensitivity analysis comparing original vs modified risk scores.
Strictly labeled as SIMULATION data — does not connect to live banking networks.
"""

import sys
import random
import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from config import (
    FEATURE_COLUMNS,
    DEMO_CHANNELS,
    DEMO_CATEGORIES,
    DEMO_LOCATIONS,
    DEFAULT_THRESHOLD,
)
from model_service import ModelService
from database import save_transaction


class SimulationEngine:
    """Generates synthetic transactions and runs sensitivity analysis."""

    def __init__(self):
        self.service = ModelService.get_instance()
        self.service.ensure_loaded()
        self.rng = random.Random()
        self.np_rng = np.random.default_rng()

    def generate_single_simulated_transaction(
        self,
        fraud_bias: float = 0.15,
        persist: bool = True,
    ) -> Dict[str, Any]:
        """
        Generate one synthetic transaction, run ML evaluation, and optionally save to DB.
        
        Args:
            fraud_bias: probability that this simulated transaction will attempt a fraud pattern.
            persist: whether to store the record in SQLite transactions table.
        """
        now = datetime.datetime.now()
        txn_id = f"SIM-{now.strftime('%H%M%S')}-{self.rng.randint(1000, 9999)}"

        is_fraud_scenario = self.rng.random() < fraud_bias

        # Baseline PCA features sampled from standard normal
        features: Dict[str, float] = {}
        for i in range(1, 29):
            col = f"V{i}"
            features[col] = float(self.np_rng.normal(0.0, 1.0))

        # Time elapsed (seconds since midnight or arbitrary counter)
        seconds_today = (now.hour * 3600) + (now.minute * 60) + now.second
        features["Time"] = float(seconds_today)

        if is_fraud_scenario:
            # Inject classic fraud vector characteristics on high-correlation PCA components
            features["V14"] = float(self.np_rng.normal(-7.0, 1.5))
            features["V12"] = float(self.np_rng.normal(-6.0, 1.5))
            features["V10"] = float(self.np_rng.normal(-5.0, 1.2))
            features["V17"] = float(self.np_rng.normal(-4.5, 1.5))
            features["V4"] = float(self.np_rng.normal(4.0, 1.0))
            features["V11"] = float(self.np_rng.normal(3.5, 1.0))
            # Fraud amounts tend to either be small probing charges or large luxury purchases
            if self.rng.random() < 0.6:
                amount = round(self.rng.uniform(150.0, 2400.0), 2)
            else:
                amount = round(self.rng.uniform(1.0, 10.0), 2)
        else:
            # Legitimate vector characteristics
            features["V14"] = float(self.np_rng.normal(0.2, 0.8))
            features["V12"] = float(self.np_rng.normal(0.1, 0.8))
            features["V10"] = float(self.np_rng.normal(0.0, 0.8))
            features["V17"] = float(self.np_rng.normal(0.1, 0.8))
            amount = round(float(self.np_rng.exponential(scale=65.0) + 2.0), 2)

        features["Amount"] = float(min(amount, 15000.0))

        # Run inference through ML pipeline
        result = self.service.predict_single(features, transaction_id=txn_id)

        # Attach simulation metadata
        result["is_simulated"] = True
        result["demo_channel"] = self.rng.choice(DEMO_CHANNELS)
        result["demo_category"] = self.rng.choice(DEMO_CATEGORIES)
        result["demo_location"] = self.rng.choice(DEMO_LOCATIONS)
        result["card_mask"] = f"**** {self.rng.randint(1000, 9999)}"

        if persist:
            save_transaction(result, is_simulated=True)

        return result

    def generate_batch(self, count: int = 10, fraud_bias: float = 0.15) -> List[Dict[str, Any]]:
        """Generate multiple synthetic transactions in sequence."""
        return [
            self.generate_single_simulated_transaction(fraud_bias=fraud_bias, persist=True)
            for _ in range(count)
        ]

    def run_what_if_analysis(
        self,
        base_features: Dict[str, Any],
        modifications: Dict[str, Any],
        threshold: float = DEFAULT_THRESHOLD,
    ) -> Dict[str, Any]:
        """
        Model sensitivity simulation:
        Evaluates a baseline transaction, applies hypothetical modifications (e.g. Amount, Time, V-features),
        and calculates the delta impact on model risk score.
        
        Strict disclaimer:
        This reflects statistical sensitivity of the trained machine learning model,
        not a causal guarantee of real-world financial fraud probability.
        """
        # Original evaluation
        orig_res = self.service.predict_single(base_features, threshold=threshold)

        # Modified evaluation
        modified_features = dict(base_features)
        modified_features.update(modifications)
        new_res = self.service.predict_single(modified_features, threshold=threshold)

        score_delta = new_res["risk_score"] - orig_res["risk_score"]
        prob_delta = new_res["fraud_probability"] - orig_res["fraud_probability"]

        if score_delta > 15:
            assessment = "Significant Risk Increase"
            badge_color = "#ef4444"
        elif score_delta > 5:
            assessment = "Moderate Risk Increase"
            badge_color = "#f97316"
        elif score_delta < -15:
            assessment = "Significant Risk Reduction"
            badge_color = "#10b981"
        elif score_delta < -5:
            assessment = "Moderate Risk Reduction"
            badge_color = "#059669"
        else:
            assessment = "Negligible Score Shift"
            badge_color = "#64748b"

        return {
            "disclaimer": "Model sensitivity simulation. Demonstrates mathematical model response to input perturbations, not real-world causality.",
            "original": {
                "amount": orig_res["amount"],
                "time": orig_res["time"],
                "risk_score": orig_res["risk_score"],
                "risk_level": orig_res["risk_level"],
                "fraud_probability": orig_res["fraud_probability"],
                "is_flagged": orig_res["is_flagged"],
            },
            "modified": {
                "amount": new_res["amount"],
                "time": new_res["time"],
                "risk_score": new_res["risk_score"],
                "risk_level": new_res["risk_level"],
                "fraud_probability": new_res["fraud_probability"],
                "is_flagged": new_res["is_flagged"],
            },
            "delta": {
                "score_delta": int(score_delta),
                "prob_delta": round(float(prob_delta), 4),
                "assessment": assessment,
                "badge_color": badge_color,
                "modifications_applied": modifications,
            },
        }
