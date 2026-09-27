"""
explainability.py
-----------------
Explainable AI (XAI) engine for SENTINEL — Fraud Intelligence Platform.
Computes local feature attributions using SHAP TreeExplainer for single transactions
("Why was this transaction flagged?"), phrases explanations strictly in terms of
PCA components and scaled features without semantic hallucinations, and provides
global feature importances.
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd

_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from model_service import ModelService
from config import SCALE_COLUMNS

# Optional SHAP import with defensive fallback
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


class ExplainabilityEngine:
    """Manages model explainability, feature contributions, and attribution narratives."""

    _instance: Optional["ExplainabilityEngine"] = None

    def __init__(self):
        self.service = ModelService.get_instance()
        self.explainer = None
        self._init_explainer()

    @classmethod
    def get_instance(cls) -> "ExplainabilityEngine":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _init_explainer(self) -> None:
        """Initialize SHAP TreeExplainer if SHAP is installed."""
        if not HAS_SHAP:
            return
        try:
            self.service.ensure_loaded()
            if self.service.model is not None:
                # TreeExplainer works natively on RandomForestClassifier
                self.explainer = shap.TreeExplainer(self.service.model)
        except Exception as e:
            print(f"[ExplainabilityEngine] Notice: Could not initialize SHAP TreeExplainer ({e}). Using tree attribution fallback.")
            self.explainer = None

    def explain_transaction(
        self,
        transaction_dict: Dict[str, Any],
        top_k: int = 8,
    ) -> Dict[str, Any]:
        """
        Explain why a transaction received its fraud risk score.
        
        Returns:
            - top_features: list of dicts with feature, value, impact (+ / -), contribution_score
            - fraud_drivers: features increasing fraud risk
            - legit_drivers: features decreasing fraud risk
            - explanation_text: clear, technically honest summary
        """
        self.service.ensure_loaded()

        # Step 1: Prepare row and scale Time/Amount
        cols = self.service.feature_columns
        row_df = pd.DataFrame([transaction_dict])[cols].copy()
        raw_values = {col: float(row_df[col].iloc[0]) for col in cols}

        scale_cols = [c for c in SCALE_COLUMNS if c in row_df.columns]
        row_scaled = row_df.copy()
        row_scaled[scale_cols] = self.service.scaler.transform(row_df[scale_cols])

        # Step 2: Compute SHAP values or fallback
        contributions: Dict[str, float] = {}
        method_used = "SHAP TreeExplainer"

        if self.explainer is not None:
            try:
                shap_raw = self.explainer.shap_values(row_scaled)
                # Parse output based on SHAP shape
                if isinstance(shap_raw, list):
                    # List of [class_0_array, class_1_array]
                    class_1_vals = shap_raw[1][0]
                elif isinstance(shap_raw, np.ndarray):
                    if shap_raw.ndim == 3:  # (n_samples, n_features, n_classes)
                        class_1_vals = shap_raw[0, :, 1]
                    elif shap_raw.ndim == 2:
                        class_1_vals = shap_raw[0]
                    else:
                        class_1_vals = shap_raw.ravel()
                else:
                    class_1_vals = np.array(shap_raw).ravel()

                for col, val in zip(cols, class_1_vals):
                    contributions[col] = float(val)
            except Exception as e:
                print(f"[ExplainabilityEngine] SHAP calculation warning: {e}. Falling back.")
                contributions = self._compute_heuristic_attribution(row_scaled.iloc[0].to_dict())
                method_used = "Feature Importance & Deviation Attribution"
        else:
            contributions = self._compute_heuristic_attribution(row_scaled.iloc[0].to_dict())
            method_used = "Feature Importance & Deviation Attribution"

        # Step 3: Sort by magnitude
        sorted_features = sorted(
            contributions.items(), key=lambda x: abs(x[1]), reverse=True
        )

        all_attributions = []
        for feat, val in sorted_features:
            is_pushing_fraud = val > 0
            raw_v = raw_values.get(feat, 0.0)
            all_attributions.append({
                "feature": feat,
                "raw_value": round(raw_v, 4),
                "contribution": round(val, 4),
                "direction": "Pushes towards Fraud" if is_pushing_fraud else "Pushes towards Legitimate",
                "impact": "+" if is_pushing_fraud else "-",
                "magnitude": abs(round(val, 4)),
            })

        # Separate into fraud drivers vs legit drivers
        fraud_drivers = [a for a in all_attributions if a["contribution"] > 0][:top_k]
        legit_drivers = [a for a in all_attributions if a["contribution"] < 0][:top_k]

        # Generate technically honest narrative
        summary_sentences = []
        if fraud_drivers:
            top_f = fraud_drivers[0]
            summary_sentences.append(
                f"Component {top_f['feature']} (value: {top_f['raw_value']}) had the highest influence in increasing the fraud risk score (+{top_f['magnitude']:.3f})."
            )
            if len(fraud_drivers) > 1:
                secondary = [f"{d['feature']} (+{d['magnitude']:.3f})" for d in fraud_drivers[1:4]]
                summary_sentences.append(f"Other primary risk elevators: {', '.join(secondary)}.")
        else:
            summary_sentences.append("No prominent features significantly elevated the fraud risk score.")

        if legit_drivers:
            top_l = legit_drivers[0]
            summary_sentences.append(
                f"Conversely, component {top_l['feature']} (value: {top_l['raw_value']}) strongly supported a legitimate classification (-{top_l['magnitude']:.3f})."
            )

        explanation_narrative = " ".join(summary_sentences)

        return {
            "method": method_used,
            "top_features": all_attributions[:top_k],
            "fraud_drivers": fraud_drivers,
            "legit_drivers": legit_drivers,
            "all_attributions": all_attributions,
            "narrative": explanation_narrative,
        }

    def _compute_heuristic_attribution(self, scaled_dict: Dict[str, float]) -> Dict[str, float]:
        """
        Fast tree attribution fallback:
        Multiplies tree feature importances by the standardized feature deviation.
        """
        importances = self.service.get_feature_importances()
        imp_map = {item["feature"]: item["importance"] for item in importances}

        contributions = {}
        for col, val in scaled_dict.items():
            imp = imp_map.get(col, 0.01)
            # High deviation on influential fraud PCA features (e.g. V14, V12, V10, V17) pushes score
            contributions[col] = float(val * imp * -1.0 if col in ["V14", "V12", "V10", "V17", "V3", "V7"] else val * imp)

        return contributions

    def get_global_feature_importance(self) -> List[Dict[str, Any]]:
        """Return global feature importances for all 30 features."""
        return self.service.get_feature_importances()
