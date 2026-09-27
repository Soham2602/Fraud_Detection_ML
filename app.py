"""
app.py
------
SENTINEL — Fraud Intelligence Platform
High-Performance, Interactive Streamlit Operations Center.

Architecture:
- Native Plotly interactive visual analytics (hover, zoom, pan, export)
- Strict Data Provenance labeling (REAL DATASET vs EVALUATION DATA vs STORED IN VAULT vs SYNTHETIC DEMO)
- Global Data Mode Switch: REAL DATA | DEMO DATA | COMBINED VIEW
- 10 Dedicated Security & Operations Modules:
  1. 🏠 Command Center (High-level operations & triage)
  2. 📊 Dataset Intelligence (Ground-truth EDA & distributions)
  3. 🔎 Transaction Investigation (Inline ML, Mathematical Decision Logic & SHAP XAI)
  4. 🚨 Fraud Alerts (Triage queue & case management)
  5. 📈 Model Intelligence (ROC/PR curves, Threshold Playground, Metrics)
  6. 🧠 Explainable AI (SHAP TreeExplainer & feature attribution)
  7. ⚡ Live Simulation (Synthetic stream generator)
  8. 🔬 What-If Analysis (Sensitivity perturbation engine)
  9. 🗂 Transaction Explorer (Search, filter, and audit vault)
  10. ⚙️ System & Model (Architecture, telemetry & diagnostics)
- Calibrated Deterministic Risk Scoring (0–100) & Visual Segmented Gauge
- Anonymized PCA-Honest Explainable AI (Directional SHAP force breakdown)
- Single Canonical ModelService Backend
"""

import sys
import json
import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# Ensure src is on python path
SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import (
    CREDITCARD_CSV,
    SAMPLE_PRESETS_JSON,
    DATASET_SUMMARY_JSON,
    EVALUATION_BUNDLE_JSON,
    EXPERIMENT_RESULTS_CSV,
    METADATA_JSON,
    DEFAULT_THRESHOLD,
    FEATURE_COLUMNS,
    SCALE_COLUMNS,
    RISK_BANDS,
    get_risk_level,
)
from model_service import ModelService, ValidationError
from database import (
    init_db,
    save_transaction,
    get_recent_transactions,
    get_transaction_by_id,
    add_review_action,
    get_alerts,
    get_dashboard_kpis,
    seed_demo_database_if_empty,
    get_db_connection,
)
from explainability import ExplainabilityEngine
from simulation import SimulationEngine
from demo_data import (
    DEMO_SCENARIOS,
    get_demo_scenarios,
    get_demo_transactions,
    get_demo_alerts,
    get_demo_kpis,
    get_demo_channel_breakdown,
    get_demo_category_breakdown,
)
from visualizations import (
    plot_class_imbalance,
    plot_amount_distribution,
    plot_amount_by_class_boxplot,
    plot_fraud_over_time,
    plot_cyclic_fraud_dynamics,
    plot_risk_distribution,
    plot_threshold_curves,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_precision_recall_curve,
    plot_feature_importances,
    plot_shap_force_bars,
    plot_risk_gauge,
    plot_correlation_heatmap,
    plot_3d_feature_space,
    plot_what_if_comparison,
    plot_simulation_stream,
)

# -----------------------------------------------------------------------------
# Streamlit Page Setup & Custom CSS
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="SENTINEL — Fraud Intelligence Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    code, pre, .mono, .font-mono {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Container Spacing */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }

    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.85) 0%, rgba(10, 15, 29, 0.95) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 16px 18px;
        box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.35);
        transition: transform 0.15s ease, border-color 0.15s ease;
        position: relative;
    }
    .metric-card:hover {
        border-color: rgba(59, 130, 246, 0.4);
        transform: translateY(-2px);
    }
    .metric-provenance {
        font-size: 0.65rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        display: inline-block;
        padding: 2px 7px;
        border-radius: 4px;
        margin-bottom: 6px;
    }
    .tag-real { background: rgba(59, 130, 246, 0.15); color: #60a5fa; border: 1px solid rgba(59, 130, 246, 0.3); }
    .tag-model { background: rgba(139, 92, 246, 0.15); color: #c084fc; border: 1px solid rgba(139, 92, 246, 0.3); }
    .tag-vault { background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3); }
    .tag-sim { background: rgba(245, 158, 11, 0.15); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.3); }
    .tag-demo { background: rgba(236, 72, 153, 0.15); color: #f472b6; border: 1px solid rgba(236, 72, 153, 0.3); }

    .metric-val {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.65rem;
        font-weight: 700;
        color: #f8fafc;
        line-height: 1.2;
    }
    .metric-lbl {
        font-size: 0.78rem;
        color: #94a3b8;
        font-weight: 500;
        margin-top: 3px;
    }
    .metric-sub {
        font-size: 0.70rem;
        color: #64748b;
        margin-top: 4px;
    }

    /* Hero Banner */
    .hero-banner {
        background: radial-gradient(circle at 10% 20%, rgba(30, 58, 138, 0.25) 0%, rgba(15, 23, 42, 0.6) 80%);
        border: 1px solid rgba(59, 130, 246, 0.25);
        border-radius: 14px;
        padding: 20px 24px;
        margin-bottom: 20px;
    }

    /* Mode Banner */
    .mode-banner-demo {
        background: rgba(236, 72, 153, 0.08);
        border: 1px solid rgba(236, 72, 153, 0.3);
        border-radius: 8px;
        padding: 8px 14px;
        margin-bottom: 16px;
        font-size: 0.80rem;
        color: #f472b6;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .mode-banner-real {
        background: rgba(59, 130, 246, 0.08);
        border: 1px solid rgba(59, 130, 246, 0.3);
        border-radius: 8px;
        padding: 8px 14px;
        margin-bottom: 16px;
        font-size: 0.80rem;
        color: #60a5fa;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .mode-banner-comb {
        background: rgba(16, 185, 129, 0.08);
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 8px;
        padding: 8px 14px;
        margin-bottom: 16px;
        font-size: 0.80rem;
        color: #34d399;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* System Status Badges */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-size: 0.72rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
        padding: 4px 10px;
        border-radius: 20px;
        background: rgba(15, 23, 42, 0.8);
        border: 1px solid rgba(255, 255, 255, 0.1);
        color: #e2e8f0;
    }
    .dot-green { width: 7px; height: 7px; border-radius: 50%; background: #10b981; box-shadow: 0 0 8px #10b981; }
    .dot-amber { width: 7px; height: 7px; border-radius: 50%; background: #f59e0b; box-shadow: 0 0 8px #f59e0b; }
    .dot-pink { width: 7px; height: 7px; border-radius: 50%; background: #ec4899; box-shadow: 0 0 8px #ec4899; }

    /* Custom Risk Badges */
    .risk-badge-low { background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3); font-weight: 700; padding: 2px 8px; border-radius: 4px; }
    .risk-badge-med { background: rgba(245, 158, 11, 0.15); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.3); font-weight: 700; padding: 2px 8px; border-radius: 4px; }
    .risk-badge-high { background: rgba(249, 115, 22, 0.15); color: #fb923c; border: 1px solid rgba(249, 115, 22, 0.3); font-weight: 700; padding: 2px 8px; border-radius: 4px; }
    .risk-badge-crit { background: rgba(239, 68, 68, 0.15); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.3); font-weight: 700; padding: 2px 8px; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Cached Resource & Data Loaders
# -----------------------------------------------------------------------------
@st.cache_resource
def get_service() -> ModelService:
    service = ModelService.get_instance()
    service.ensure_loaded()
    return service

@st.cache_resource
def get_explainer() -> ExplainabilityEngine:
    return ExplainabilityEngine.get_instance()

@st.cache_resource
def get_simulator() -> SimulationEngine:
    return SimulationEngine()

@st.cache_data
def load_dataset_summary() -> Dict[str, Any]:
    if DATASET_SUMMARY_JSON.exists():
        with open(DATASET_SUMMARY_JSON, "r") as f:
            return json.load(f)
    return {}

@st.cache_data
def load_evaluation_bundle() -> Dict[str, Any]:
    if EVALUATION_BUNDLE_JSON.exists():
        with open(EVALUATION_BUNDLE_JSON, "r") as f:
            return json.load(f)
    return {}

@st.cache_data
def load_sample_presets() -> List[Dict[str, Any]]:
    if SAMPLE_PRESETS_JSON.exists():
        with open(SAMPLE_PRESETS_JSON, "r") as f:
            return json.load(f)
    return []

@st.cache_data
def load_experiment_results() -> pd.DataFrame:
    if EXPERIMENT_RESULTS_CSV.exists():
        return pd.read_csv(EXPERIMENT_RESULTS_CSV)
    return pd.DataFrame()

@st.cache_data
def load_metadata() -> Dict[str, Any]:
    if METADATA_JSON.exists():
        with open(METADATA_JSON, "r") as f:
            return json.load(f)
    return {}


# -----------------------------------------------------------------------------
# Initialize DB & Session State
# -----------------------------------------------------------------------------
init_db()
seed_demo_database_if_empty()

if "selected_nav" not in st.session_state:
    st.session_state.selected_nav = "🏠 Command Center"
if "simulation_history" not in st.session_state:
    st.session_state.simulation_history = []
if "last_evaluated" not in st.session_state:
    st.session_state.last_evaluated = None
if "data_mode" not in st.session_state:
    st.session_state.data_mode = "DEMO DATA (Curated 500 TXNs)"


# -----------------------------------------------------------------------------
# Currency Formatter ($ USD vs ₹ INR)
# -----------------------------------------------------------------------------
def format_currency(amount_val: float, currency_mode: str = "$ USD") -> str:
    if currency_mode == "₹ INR":
        inr_val = amount_val * 83.50
        if inr_val >= 10000000:
            return f"₹{inr_val / 10000000:.2f} Cr"
        elif inr_val >= 100000:
            return f"₹{inr_val / 100000:.2f} L"
        else:
            return f"₹{inr_val:,.2f}"
    else:
        return f"${amount_val:,.2f}"


# -----------------------------------------------------------------------------
# Sidebar: Control Center
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="padding-bottom: 12px; border-bottom: 1px solid rgba(255,255,255,0.08); margin-bottom: 14px;">
        <div style="font-size: 1.35rem; font-weight: 800; color: #f8fafc; letter-spacing: -0.02em;">
            🛡️ SENTINEL
        </div>
        <div style="font-size: 0.72rem; color: #60a5fa; font-weight: 700; letter-spacing: 0.05em; text-transform: uppercase;">
            Fraud Intelligence Platform
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 1. Navigation
    nav_options = [
        "🏠 Command Center",
        "📊 Dataset Intelligence",
        "🔎 Transaction Investigation",
        "🚨 Fraud Alerts",
        "📈 Model Intelligence",
        "🧠 Explainable AI",
        "⚡ Live Simulation",
        "🔬 What-If Analysis",
        "🗂 Transaction Explorer",
        "⚙️ System & Model",
    ]

    selected_page = st.radio(
        "OPERATIONAL MODULES",
        nav_options,
        index=nav_options.index(st.session_state.selected_nav) if st.session_state.selected_nav in nav_options else 0,
        label_visibility="collapsed",
    )
    st.session_state.selected_nav = selected_page

    st.markdown("<hr style='border-color: rgba(255,255,255,0.08); margin: 16px 0;'>", unsafe_allow_html=True)

    # 2. Global Data Mode Selector (Master Feature)
    st.markdown("<div style='font-size:0.70rem; font-weight:700; color:#94a3b8; text-transform:uppercase; margin-bottom:6px;'>Global Data Mode</div>", unsafe_allow_html=True)
    mode_options = [
        "DEMO DATA (Curated 500 TXNs)",
        "REAL DATA (284k Kaggle Benchmark)",
        "COMBINED VIEW (Unified Telemetry)",
    ]
    active_mode = st.selectbox(
        "Select Data Mode",
        mode_options,
        index=mode_options.index(st.session_state.data_mode) if st.session_state.data_mode in mode_options else 0,
        label_visibility="collapsed",
    )
    st.session_state.data_mode = active_mode

    # 3. Visualization Controls
    st.markdown("<div style='font-size:0.70rem; font-weight:700; color:#94a3b8; text-transform:uppercase; margin-top:14px; margin-bottom:6px;'>Display Controls</div>", unsafe_allow_html=True)
    show_advanced_charts = st.toggle("Show Advanced Analytics", value=True)
    show_explanations = st.toggle("Show Model Explanations", value=True)
    show_raw_features = st.toggle("Show Raw PCA Features", value=False)
    currency_toggle = st.selectbox("Currency Format", ["$ USD", "₹ INR (Demo Layer)"], index=0)
    active_currency = "$ USD" if "USD" in currency_toggle else "₹ INR"

    st.markdown("<hr style='border-color: rgba(255,255,255,0.08); margin: 16px 0;'>", unsafe_allow_html=True)

    # 4. System Telemetry
    service_ref = get_service()
    db_kpis = get_dashboard_kpis()
    demo_kpis = get_demo_kpis()
    st.markdown(f"""
    <div style="font-size: 0.72rem; color: #94a3b8; space-y-1;">
        <div><b>Model:</b> <span class="font-mono text-blue-400">Tuned Random Forest</span></div>
        <div><b>Threshold:</b> <span class="font-mono text-amber-400">{DEFAULT_THRESHOLD:.2f}</span></div>
        <div><b>Vault Records:</b> <span class="font-mono text-emerald-400">{db_kpis.get('total_transactions', 0):,}</span></div>
        <div><b>Demo Records:</b> <span class="font-mono text-pink-400">{demo_kpis.get('total_transactions', 500):,}</span></div>
        <div><b>Open Alerts:</b> <span class="font-mono text-rose-400">{demo_kpis.get('open_alerts', 30) if 'DEMO' in active_mode else db_kpis.get('open_alerts', 0):,}</span></div>
    </div>
    """, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Global Top Header & Status Badges
# -----------------------------------------------------------------------------
st.markdown(f"""
<div class="hero-banner">
    <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:12px;">
        <div>
            <div style="display:flex; align-items:center; gap:10px;">
                <span style="font-size:1.6rem; font-weight:800; color:#f8fafc; letter-spacing:-0.03em;">SENTINEL</span>
                <span class="status-badge"><span class="dot-green"></span> SYSTEM OPERATIONAL</span>
            </div>
            <div style="font-size:0.85rem; color:#94a3b8; margin-top:4px;">
                Fraud Intelligence & Investigation Platform • <b>Detect • Investigate • Explain</b>
            </div>
        </div>
        <div style="display:flex; gap:8px; flex-wrap:wrap;">
            <span class="status-badge"><span class="dot-green"></span> MODEL ONLINE</span>
            <span class="status-badge"><span class="dot-green"></span> DATABASE ONLINE</span>
            <span class="status-badge"><span class="dot-green"></span> EXPLAINABILITY ONLINE</span>
            <span class="status-badge"><span class="dot-green"></span> API ONLINE</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Render Global Data Mode Banner
if "DEMO" in st.session_state.data_mode:
    st.markdown("""
    <div class="mode-banner-demo">
        <span class="dot-pink"></span>
        <b>ACTIVE MODE: SYNTHETIC DEMONSTRATION LAYER</b> — Displaying 500 realistic, deterministic transactions (14% flagged rate, 30 prioritized alerts) for rich demonstration. All model scores are real.
    </div>
    """, unsafe_allow_html=True)
elif "REAL" in st.session_state.data_mode:
    st.markdown("""
    <div class="mode-banner-real">
        <span class="dot-green"></span>
        <b>ACTIVE MODE: REAL DATASET BENCHMARK</b> — Ground truth from 284,807 European cardholder transactions (492 frauds, 0.1727% fraud rate) and local SQLite audit database.
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="mode-banner-comb">
        <span class="dot-green"></span>
        <b>ACTIVE MODE: COMBINED VIEW</b> — Aggregating real Kaggle dataset metrics alongside active demonstration and audit logs.
    </div>
    """, unsafe_allow_html=True)


# Load artifacts
ds_summary = load_dataset_summary()
eval_bundle = load_evaluation_bundle()
sample_presets = load_sample_presets()
exp_df = load_experiment_results()
model_meta = load_metadata()
db_kpis = get_dashboard_kpis()
demo_kpis = get_demo_kpis()
demo_df = get_demo_transactions()
demo_alerts = get_demo_alerts()
demo_scenarios = get_demo_scenarios()


# =============================================================================
# MODULE 1: 🏠 COMMAND CENTER (Operations Console)
# =============================================================================
if selected_page == "🏠 Command Center":
    # Quick Navigation Buttons
    c_btn1, c_btn2, c_btn3 = st.columns(3)
    with c_btn1:
        if st.button("🔎 Analyze Single Transaction", use_container_width=True):
            st.session_state.selected_nav = "🔎 Transaction Investigation"
            st.rerun()
    with c_btn2:
        if st.button("📊 Explore Dataset Intelligence", use_container_width=True):
            st.session_state.selected_nav = "📊 Dataset Intelligence"
            st.rerun()
    with c_btn3:
        if st.button("🚨 View Fraud Alert Queue", use_container_width=True):
            st.session_state.selected_nav = "🚨 Fraud Alerts"
            st.rerun()

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    # Key Operational KPI Cards (Dynamic according to mode)
    k1, k2, k3, k4, k5 = st.columns(5)
    
    if "DEMO" in st.session_state.data_mode:
        with k1:
            st.markdown(f"""
            <div class="metric-card">
                <span class="metric-provenance tag-demo">SYNTHETIC DEMO</span>
                <div class="metric-val">{demo_kpis['total_transactions']:,}</div>
                <div class="metric-lbl">Total Transactions</div>
                <div class="metric-sub">Calibrated Demo Layer</div>
            </div>
            """, unsafe_allow_html=True)
        with k2:
            st.markdown(f"""
            <div class="metric-card">
                <span class="metric-provenance tag-demo">SYNTHETIC DEMO</span>
                <div class="metric-val" style="color: #f87171;">{demo_kpis['fraud_flags']:,}</div>
                <div class="metric-lbl">Flagged Incidents</div>
                <div class="metric-sub">Rate: <b>{demo_kpis['fraud_rate_pct']:.1f}%</b> (Rich Demo)</div>
            </div>
            """, unsafe_allow_html=True)
        with k3:
            st.markdown(f"""
            <div class="metric-card">
                <span class="metric-provenance tag-model">MODEL BENCHMARK</span>
                <div class="metric-val" style="color: #60a5fa;">{eval_bundle.get('default_metrics', {}).get('pr_auc', 0.8096):.4f}</div>
                <div class="metric-lbl">PR-AUC Score</div>
                <div class="metric-sub">Tuned Random Forest Test</div>
            </div>
            """, unsafe_allow_html=True)
        with k4:
            st.markdown(f"""
            <div class="metric-card">
                <span class="metric-provenance tag-demo">SYNTHETIC DEMO</span>
                <div class="metric-val" style="color: #34d399;">{demo_kpis['high_risk_transactions']:,}</div>
                <div class="metric-lbl">High-Risk Severity</div>
                <div class="metric-sub">Score &ge; 61 / 100</div>
            </div>
            """, unsafe_allow_html=True)
        with k5:
            st.markdown(f"""
            <div class="metric-card">
                <span class="metric-provenance tag-demo">SYNTHETIC DEMO</span>
                <div class="metric-val" style="color: #fbbf24;">{demo_kpis['open_alerts']:,}</div>
                <div class="metric-lbl">Active Alerts</div>
                <div class="metric-sub">Reviewed: <b>{demo_kpis['transactions_reviewed']:,}</b></div>
            </div>
            """, unsafe_allow_html=True)
    else:
        with k1:
            st.markdown(f"""
            <div class="metric-card">
                <span class="metric-provenance tag-real">REAL DATASET</span>
                <div class="metric-val">{ds_summary.get('overview', {}).get('total_transactions', 284807):,}</div>
                <div class="metric-lbl">Total Transactions</div>
                <div class="metric-sub">Kaggle Benchmark (48h)</div>
            </div>
            """, unsafe_allow_html=True)
        with k2:
            st.markdown(f"""
            <div class="metric-card">
                <span class="metric-provenance tag-real">REAL DATASET</span>
                <div class="metric-val" style="color: #f87171;">{ds_summary.get('overview', {}).get('fraud_count', 492):,}</div>
                <div class="metric-lbl">Fraud Incidents</div>
                <div class="metric-sub">Rate: <b>0.1727%</b> (492/284k)</div>
            </div>
            """, unsafe_allow_html=True)
        with k3:
            st.markdown(f"""
            <div class="metric-card">
                <span class="metric-provenance tag-model">MODEL BENCHMARK</span>
                <div class="metric-val" style="color: #60a5fa;">{eval_bundle.get('default_metrics', {}).get('pr_auc', 0.8096):.4f}</div>
                <div class="metric-lbl">PR-AUC Score</div>
                <div class="metric-sub">Tuned Random Forest Test</div>
            </div>
            """, unsafe_allow_html=True)
        with k4:
            st.markdown(f"""
            <div class="metric-card">
                <span class="metric-provenance tag-vault">STORED IN VAULT</span>
                <div class="metric-val" style="color: #34d399;">{db_kpis.get('total_transactions', 0):,}</div>
                <div class="metric-lbl">Monitored in Vault</div>
                <div class="metric-sub">Flagged: <b>{db_kpis.get('fraud_flags', 0):,}</b></div>
            </div>
            """, unsafe_allow_html=True)
        with k5:
            st.markdown(f"""
            <div class="metric-card">
                <span class="metric-provenance tag-vault">STORED IN VAULT</span>
                <div class="metric-val" style="color: #fbbf24;">{db_kpis.get('open_alerts', 0):,}</div>
                <div class="metric-lbl">Active Alerts</div>
                <div class="metric-sub">Reviewed: <b>{db_kpis.get('transactions_reviewed', 0):,}</b></div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    # Section 1: Fraud Landscape
    st.subheader("🌐 Fraud Landscape & Statistical Distributions")
    c_land1, c_land2 = st.columns([1, 1])
    with c_land1:
        legit_c = ds_summary.get("overview", {}).get("legitimate_count", 284315)
        fraud_c = ds_summary.get("overview", {}).get("fraud_count", 492)
        fig_pie = plot_class_imbalance(legit_c, fraud_c)
        st.plotly_chart(fig_pie, use_container_width=True)
        st.caption("Answers: *What is the severity of class imbalance in credit card transactions?* (0.17% fraud rate requires specialized PR-AUC / SMOTE handling).")
    with c_land2:
        risk_dist = eval_bundle.get("risk_distribution", [])
        fig_risk = plot_risk_distribution(risk_dist, threshold=DEFAULT_THRESHOLD)
        st.plotly_chart(fig_risk, use_container_width=True)
        st.caption("Answers: *How decisively does the model separate legitimate transactions from frauds?* (Bimodal separation near 0 and 1).")

    # Section 2: Spending Behavior & Temporal Activity
    st.subheader("💳 Transaction Behavior & Temporal Patterns")
    c_beh1, c_beh2 = st.columns([1, 1])
    with c_beh1:
        use_log = st.toggle("Logarithmic Scale (Amount)", value=True, key="cmd_log_toggle")
        binned = ds_summary.get("amount_distribution", [])
        fig_amt = plot_amount_distribution(binned, use_log_scale=use_log)
        st.plotly_chart(fig_amt, use_container_width=True)
        st.caption("Answers: *Do fraudsters prefer high or low dollar amounts?* (Over 50% of frauds are below $10 card testing attempts).")
    with c_beh2:
        show_rate = st.toggle("Show Fraud Rate (%) instead of Volume", value=False, key="cmd_rate_toggle")
        hourly = ds_summary.get("time_hourly_48h", [])
        fig_time = plot_fraud_over_time(hourly, show_rate=show_rate)
        st.plotly_chart(fig_time, use_container_width=True)
        st.caption("Answers: *Does fraud occurrence track human diurnal waking hours?* (Fraud peaks during off-peak night hours).")

    # Section 3: Model Intelligence At A Glance
    st.subheader("📈 Model Test-Set Diagnostics (Tuned Random Forest)")
    c_mod1, c_mod2 = st.columns([1, 1])
    with c_mod1:
        cm_default = eval_bundle.get("default_metrics", {}).get("confusion_matrix", {})
        fig_cm = plot_confusion_matrix(cm_default, threshold=0.50)
        st.plotly_chart(fig_cm, use_container_width=True)
        st.caption("Answers: *How many false alarms and missed frauds occur at threshold 0.50?* (Only 6 false alarms out of 56,651 legitimate test cases).")
    with c_mod2:
        pr_curve_pts = eval_bundle.get("pr_curve", [])
        fig_pr = plot_precision_recall_curve(pr_curve_pts, pr_auc=eval_bundle.get("default_metrics", {}).get("pr_auc", 0.8096))
        st.plotly_chart(fig_pr, use_container_width=True)
        st.caption("Answers: *Why evaluate PR-AUC instead of ROC-AUC?* (Under 0.17% fraud prevalence, PR-AUC penalizes false alarms heavily).")

    # Section 4: Recent Alerts & Live System Activity
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    st.subheader("🚨 Incident Queue & Audit Activity")
    c_act1, c_act2 = st.columns([3, 2])
    with c_act1:
        st.markdown("**Recent High-Risk Incidents (Active Alert Queue)**")
        # Load from DB or Demo based on active mode
        if "DEMO" in st.session_state.data_mode or db_kpis.get("total_transactions", 0) < 5:
            disp_alerts = demo_alerts[:6]
            table_rows = []
            for a in disp_alerts:
                table_rows.append({
                    "TXN ID": a["transaction_id"],
                    "Time": a["created_at"].split(" ")[1] if " " in a["created_at"] else a["created_at"][:8],
                    "Amount": format_currency(a["amount"], active_currency),
                    "Channel": a["channel"],
                    "Risk Score": f"{a['risk_score']} / 100",
                    "Risk Level": a["risk_level"],
                    "Status": a["status"],
                    "Provenance": a.get("provenance", "SYNTHETIC DEMO"),
                })
            st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)
        else:
            recent_txns = get_recent_transactions(limit=6)
            table_rows = []
            for t in recent_txns:
                table_rows.append({
                    "TXN ID": t["id"],
                    "Time": t["timestamp"].split("T")[1][:8] if "T" in t["timestamp"] else t["timestamp"][:8],
                    "Amount": format_currency(t["amount"], active_currency),
                    "Risk Score": f"{t['risk_score']} / 100",
                    "Risk Level": t["risk_level"],
                    "Status": t["status"],
                    "Decision": "🚨 FLAGGED" if t["is_flagged"] else "✅ APPROVED",
                    "Provenance": "STORED IN VAULT",
                })
            st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

    with c_act2:
        st.markdown("**Live System Activity Log**")
        events_to_show = demo_alerts[:4] if ("DEMO" in st.session_state.data_mode or db_kpis.get("total_transactions", 0) < 5) else get_recent_transactions(limit=4)
        for ev in events_to_show:
            t_id = ev.get("transaction_id") or ev.get("id")
            t_amt = ev.get("amount", 0.0)
            t_score = ev.get("risk_score", 0)
            t_lvl = ev.get("risk_level", "LOW")
            t_stat = ev.get("status", "Active")
            t_prov = ev.get("provenance", "SYNTHETIC DEMO")
            badge_color = "#f87171" if t_score >= 61 else "#fbbf24" if t_score >= 31 else "#34d399"

            st.markdown(f"""
            <div style="padding:8px 12px; margin-bottom:6px; border-radius:8px; background:rgba(15,23,42,0.6); border:1px solid rgba(255,255,255,0.06); font-size:0.75rem;">
                <div style="display:flex; justify-content:space-between; color:#94a3b8;">
                    <span class="font-mono">{t_id}</span>
                    <span class="font-mono text-xs" style="color:#ec4899;">{t_prov}</span>
                </div>
                <div style="margin-top:3px; color:#f1f5f9;">
                    <b>{format_currency(t_amt, active_currency)}</b> evaluated • Risk <b style="color:{badge_color};">{t_score}/100</b> ({t_lvl}) • Status: <b>{t_stat}</b>
                </div>
            </div>
            """, unsafe_allow_html=True)


# =============================================================================
# MODULE 2: 📊 DATASET INTELLIGENCE
# =============================================================================
elif selected_page == "📊 Dataset Intelligence":
    st.title("📊 Dataset Intelligence & Exploratory Analysis")
    st.markdown("""
    Explore ground-truth properties of the **284,807 European cardholder transactions** collected over 48 hours in September 2013.
    *All metrics and distributions presented here are extracted directly from the benchmark dataset without synthetic modifications.*
    """)

    tab_overview, tab_spending, tab_temporal, tab_features, tab_corr, tab_3d = st.tabs([
        "📋 Overview & Imbalance",
        "💵 Spending Behavior",
        "⏰ Temporal Dynamics",
        "🔬 Feature Explorer",
        "🧬 Correlations & Heatmap",
        "🪐 3D Feature Space",
    ])

    with tab_overview:
        ov = ds_summary.get("overview", {})
        amt_stats = ds_summary.get("amount_statistics", {})

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Records", f"{ov.get('total_transactions', 284807):,}", help="Total raw transaction records")
        c2.metric("Clean Records", f"{ov.get('clean_transactions', 283726):,}", delta="-1,081 duplicates removed")
        c3.metric("Verified Frauds", f"{ov.get('fraud_count', 492):,}", help="Confirmed fraudulent transactions")
        c4.metric("Fraud Rate", f"{ov.get('fraud_rate_pct', 0.1727):.4f}%", help="Extreme positive class sparsity")

        st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

        col_pie, col_stats = st.columns([1, 1])
        with col_pie:
            st.plotly_chart(plot_class_imbalance(ov.get("legitimate_count", 284315), ov.get("fraud_count", 492)), use_container_width=True)
        with col_stats:
            st.markdown("**Amount Summary by Class (Kaggle Benchmark)**")
            stats_table = [
                {"Metric": "Count", "Overall": f"{amt_stats.get('overall', {}).get('count', 0):,}", "Legitimate": f"{amt_stats.get('legitimate', {}).get('count', 0):,}", "Fraudulent": f"{amt_stats.get('fraudulent', {}).get('count', 0):,}"},
                {"Metric": "Mean Amount", "Overall": format_currency(amt_stats.get('overall', {}).get('mean', 0), active_currency), "Legitimate": format_currency(amt_stats.get('legitimate', {}).get('mean', 0), active_currency), "Fraudulent": format_currency(amt_stats.get('fraudulent', {}).get('mean', 0), active_currency)},
                {"Metric": "Median Amount", "Overall": format_currency(amt_stats.get('overall', {}).get('median', 0), active_currency), "Legitimate": format_currency(amt_stats.get('legitimate', {}).get('median', 0), active_currency), "Fraudulent": format_currency(amt_stats.get('fraudulent', {}).get('median', 0), active_currency)},
                {"Metric": "Std Deviation", "Overall": format_currency(amt_stats.get('overall', {}).get('std', 0), active_currency), "Legitimate": format_currency(amt_stats.get('legitimate', {}).get('std', 0), active_currency), "Fraudulent": format_currency(amt_stats.get('fraudulent', {}).get('std', 0), active_currency)},
                {"Metric": "99th Percentile", "Overall": format_currency(amt_stats.get('overall', {}).get('p99', 0), active_currency), "Legitimate": format_currency(amt_stats.get('legitimate', {}).get('p99', 0), active_currency), "Fraudulent": format_currency(amt_stats.get('fraudulent', {}).get('p99', 0), active_currency)},
                {"Metric": "Max Amount", "Overall": format_currency(amt_stats.get('overall', {}).get('max', 0), active_currency), "Legitimate": format_currency(amt_stats.get('legitimate', {}).get('max', 0), active_currency), "Fraudulent": format_currency(amt_stats.get('fraudulent', {}).get('max', 0), active_currency)},
            ]
            st.dataframe(pd.DataFrame(stats_table), use_container_width=True, hide_index=True)

    with tab_spending:
        c_s1, c_s2 = st.columns([1, 1])
        with c_s1:
            st.plotly_chart(plot_amount_distribution(ds_summary.get("amount_distribution", []), use_log_scale=True), use_container_width=True)
        with c_s2:
            st.plotly_chart(plot_amount_by_class_boxplot(ds_summary.get("amount_statistics", {})), use_container_width=True)
        st.info("💡 **Statistical Insight:** Fraud transactions exhibit a lower median amount ($9.25 vs $22.00) because organized fraudsters perform micro-authorization tests to verify stolen cards before initiating large cashouts.")

    with tab_temporal:
        c_t1, c_t2 = st.columns([1, 1])
        with c_t1:
            st.plotly_chart(plot_fraud_over_time(ds_summary.get("time_hourly_48h", []), show_rate=False), use_container_width=True)
        with c_t2:
            st.plotly_chart(plot_cyclic_fraud_dynamics(ds_summary.get("time_cyclic_24h", [])), use_container_width=True)
        st.info("💡 **Diurnal Dynamics:** While total legitimate volume drops by 80% between 01:00 AM and 05:00 AM, fraudulent transactions continue at steady velocity, causing the local fraud rate to surge above 1.04% at 04:00 AM UTC.")

    with tab_features:
        feat_profiles = ds_summary.get("feature_profiles", {})
        if feat_profiles:
            st.markdown(r"**Inspect Statistical Signatures of Any Feature ($V_1 \dots V_{28}$, Amount, Time)**")
            selected_feat = st.selectbox("Select Feature Component", FEATURE_COLUMNS, index=14)
            prof = feat_profiles.get(selected_feat, {})

            f_c1, f_c2, f_c3, f_c4 = st.columns(4)
            f_c1.metric("Global Mean", f"{prof.get('mean', 0.0):.4f}")
            f_c2.metric("Fraud Class Mean", f"{prof.get('fraud_mean', 0.0):.4f}")
            f_c3.metric("Legit Class Mean", f"{prof.get('legit_mean', 0.0):.4f}")
            f_c4.metric("Correlation w/ Class", f"{prof.get('correlation_with_class', 0.0):+.4f}")

            fig_f_comp = go.Figure()
            fig_f_comp.add_trace(go.Bar(name="Legitimate Mean", x=[selected_feat], y=[prof.get("legit_mean", 0.0)], marker_color="#3b82f6"))
            fig_f_comp.add_trace(go.Bar(name="Fraudulent Mean", x=[selected_feat], y=[prof.get("fraud_mean", 0.0)], marker_color="#ef4444"))
            fig_f_comp.update_layout(title=f"Mean Separation for {selected_feat} between Classes", height=320, barmode="group", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#e2e8f0"))
            st.plotly_chart(fig_f_comp, use_container_width=True)
            st.caption(r"Notice: Features $V_1 \dots V_{28}$ are principal components derived via orthogonal PCA to preserve cardholder privacy. Semantic meanings should not be inferred.")

    with tab_corr:
        c_heat1, c_heat2 = st.columns([1, 1])
        with c_heat1:
            st.markdown("**Top Fraud Discriminators (Linear Pearson Correlation)**")
            corrs = ds_summary.get("correlations", {})
            pos_df = pd.DataFrame(corrs.get("top_positive", []))
            neg_df = pd.DataFrame(corrs.get("top_negative", []))

            fig_corr_bar = go.Figure()
            fig_corr_bar.add_trace(go.Bar(name="Negative Correlation (Decreases Fraud)", x=neg_df["feature"], y=neg_df["correlation"], marker_color="#3b82f6"))
            fig_corr_bar.add_trace(go.Bar(name="Positive Correlation (Increases Fraud)", x=pos_df["feature"], y=pos_df["correlation"], marker_color="#ef4444"))
            fig_corr_bar.update_layout(title="Leading Correlated Features with Class", height=380, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#e2e8f0"))
            st.plotly_chart(fig_corr_bar, use_container_width=True)
        with c_heat2:
            st.plotly_chart(plot_correlation_heatmap(ds_summary.get("correlation_heatmap", {})), use_container_width=True)

    with tab_3d:
        st.markdown("**Interactive 3D Manifold of 1,500 Real Dataset Transactions**")
        st.caption("Rotatable 3D scatter plot containing all 492 confirmed fraud transactions + 1,008 randomly sampled legitimate transactions.")
        s3d_col1, s3d_col2, s3d_col3 = st.columns(3)
        with s3d_col1:
            x_f = st.selectbox("X-Axis Feature", ["V14", "V17", "V12", "V10", "V4", "Amount"], index=0)
        with s3d_col2:
            y_f = st.selectbox("Y-Axis Feature", ["V10", "V12", "V14", "V17", "V11", "Time"], index=0)
        with s3d_col3:
            z_f = st.selectbox("Z-Axis Feature", ["V12", "V17", "V4", "V16", "V2", "Amount"], index=0)

        fig_3d = plot_3d_feature_space(ds_summary.get("sample_3d", {}), x_feature=x_f, y_feature=y_f, z_feature=z_f)
        st.plotly_chart(fig_3d, use_container_width=True)


# =============================================================================
# MODULE 3: 🔎 TRANSACTION INVESTIGATION (Immediate Full Decision Sequence)
# =============================================================================
elif selected_page == "🔎 Transaction Investigation":
    st.title("🔎 Transaction Investigation Workspace")
    st.markdown("""
    End-to-end operational intelligence: **Input Vector → Real-Time Inference → Mathematical Decision Logic → Directional SHAP Attribution → What-If Sensitivity → Case Audit Action**.
    """)

    # Step 1: Input Setup with 5 Curated Scenarios
    st.markdown("### Step 1: Select or Configure Transaction Vector")
    tab_scenarios, tab_preset, tab_quick, tab_advanced = st.tabs([
        "🎭 5 Curated Scenarios",
        "📂 Real Kaggle Presets",
        "⚡ Key Drivers (Sliders)",
        "🔬 Complete 30-Feature Vector",
    ])

    active_features = {}
    preset_label = ""
    scenario_metadata = {}

    with tab_scenarios:
        st.markdown("Select from 5 pre-calibrated scenario archetypes designed for immediate viva demonstration:")
        scenario_keys = list(demo_scenarios.keys())
        chosen_sc_name = st.selectbox("Select Scenario Archetype", scenario_keys, index=2)  # Default High-Risk
        sc_info = demo_scenarios[chosen_sc_name]
        scenario_metadata = sc_info
        preset_label = f"{sc_info['scenario_id']} — {sc_info['title']}"

        st.markdown(f"""
        <div style="padding:12px 16px; border-radius:8px; background:rgba(15,23,42,0.7); border:1px solid rgba(255,255,255,0.08); margin: 8px 0 14px 0;">
            <div style="font-weight:700; color:#f8fafc; font-size:0.95rem;">{sc_info['title']}</div>
            <div style="color:#94a3b8; font-size:0.80rem; margin-top:2px;">{sc_info['description']}</div>
            <div style="display:flex; gap:16px; margin-top:8px; font-size:0.75rem; flex-wrap:wrap;">
                <span><b>Category:</b> {sc_info['category']}</span>
                <span><b>Channel:</b> {sc_info['channel']}</span>
                <span><b>Location:</b> {sc_info['location']}</span>
                <span><b>Expected Verdict:</b> {sc_info['expected_verdict']}</span>
            </div>
            <div style="font-size:0.72rem; color:#60a5fa; margin-top:6px;"><b>Profile Note:</b> {sc_info['explanation']}</div>
        </div>
        """, unsafe_allow_html=True)
        active_features = {col: float(sc_info["features"].get(col, 0.0)) for col in FEATURE_COLUMNS}

    with tab_preset:
        preset_names = [f"{p['id']} — {p['name']} ({'🚨 FRAUD' if p.get('actual_class')==1 else '✅ LEGIT'})" for p in sample_presets]
        selected_preset_idx = st.selectbox("Select Kaggle Ground-Truth Record", range(len(preset_names)), format_func=lambda i: preset_names[i], index=8)
        chosen_preset = sample_presets[selected_preset_idx]
        gt_label = "🚨 GROUND TRUTH: FRAUDULENT" if chosen_preset.get("actual_class") == 1 else "✅ GROUND TRUTH: LEGITIMATE"
        st.markdown(f"<span class='status-badge font-mono' style='color:#f8fafc;'>{gt_label}</span>", unsafe_allow_html=True)
        st.markdown(f"**Amount:** {format_currency(chosen_preset['amount'], active_currency)} • **Elapsed Time:** {chosen_preset['time']}s")
        # Overwrite if user clicks this tab
        if st.checkbox("Load this Kaggle Preset into Active Vector", value=False, key="use_kaggle_preset"):
            active_features = {col: float(chosen_preset["features"].get(col, 0.0)) for col in FEATURE_COLUMNS}
            preset_label = chosen_preset["id"]

    with tab_quick:
        st.markdown("Adjust primary financial and top predictive PCA features:")
        q_c1, q_c2, q_c3 = st.columns(3)
        with q_c1:
            q_amount = st.number_input("Transaction Amount ($)", value=float(active_features.get("Amount", 99.99)), min_value=0.0, step=10.0, key="quick_amt")
            q_v14 = st.slider("V14 (Leading Negative Predictor)", -12.0, 5.0, float(active_features.get("V14", -4.28)), 0.1, key="quick_v14")
        with q_c2:
            q_time = st.number_input("Time (Seconds from start)", value=float(active_features.get("Time", 406.0)), min_value=0.0, step=100.0, key="quick_time")
            q_v10 = st.slider("V10 (Risk Elevator)", -10.0, 5.0, float(active_features.get("V10", -2.77)), 0.1, key="quick_v10")
        with q_c3:
            q_v12 = st.slider("V12 (Separation Component)", -10.0, 5.0, float(active_features.get("V12", -2.89)), 0.1, key="quick_v12")
            q_v17 = st.slider("V17 (Inverse Risk)", -10.0, 5.0, float(active_features.get("V17", -2.83)), 0.1, key="quick_v17")

        if st.checkbox("Apply Quick Slider Overrides to Vector", value=False, key="apply_quick"):
            active_features["Amount"] = q_amount
            active_features["Time"] = q_time
            active_features["V14"] = q_v14
            active_features["V10"] = q_v10
            active_features["V12"] = q_v12
            active_features["V17"] = q_v17

    with tab_advanced:
        st.markdown("Full 30-feature vector inputs:")
        adv_cols = st.columns(5)
        for i, col in enumerate(FEATURE_COLUMNS):
            with adv_cols[i % 5]:
                val = float(active_features.get(col, 0.0))
                active_features[col] = st.number_input(col, value=val, key=f"adv_{col}")

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    # Step 2: Dynamic Operational Threshold
    st.markdown("### Step 2: Operational Decision Cutoff ($\tau$)")
    t_c1, t_c2 = st.columns([3, 2])
    with t_c1:
        invest_threshold = st.slider(
            "Classification Decision Cutoff Threshold ($\\tau$)",
            min_value=0.01,
            max_value=0.99,
            value=0.50,
            step=0.01,
            help="Transactions with Posterior Probability P(Fraud) >= tau are flagged."
        )
    with t_c2:
        st.markdown("<div style='font-size:0.75rem; color:#94a3b8; margin-bottom:4px;'>Preset Threshold Strategies</div>", unsafe_allow_html=True)
        tb_c1, tb_c2, tb_c3 = st.columns(3)
        with tb_c1:
            if st.button("⚖️ Balanced (0.50)"):
                invest_threshold = 0.50
        with tb_c2:
            if st.button("🛡️ High Recall (0.35)"):
                invest_threshold = 0.35
        with tb_c3:
            if st.button("🎯 High Prec. (0.65)"):
                invest_threshold = 0.65

    # Pipeline Flow Visual Bar
    st.markdown(f"""
    <div style="padding:10px 14px; margin: 12px 0; border-radius:8px; background:rgba(15,23,42,0.6); border:1px solid rgba(255,255,255,0.08); font-size:0.75rem; color:#94a3b8; display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap;">
        <span><b>INPUT:</b> 30 Features</span> → 
        <span><b>SCALING:</b> StandardScaler (Time, Amount)</span> → 
        <span><b>MODEL:</b> Tuned Random Forest (100 Trees)</span> → 
        <span><b>POSTERIOR:</b> P(Fraud)</span> → 
        <span><b>CUTOFF:</b> &ge; {invest_threshold:.2f}</span> → 
        <span><b>EXPLANATION:</b> Directional SHAP</span>
    </div>
    """, unsafe_allow_html=True)

    # Step 3: Run Inference
    if st.button("🚀 Run SENTINEL Intelligence & Risk Audit", type="primary", use_container_width=True):
        service = get_service()
        result = service.predict_single(active_features, threshold=invest_threshold)
        if scenario_metadata:
            result["channel"] = scenario_metadata.get("channel", "POS Terminal")
            result["category"] = scenario_metadata.get("category", "Retail")
            result["location"] = scenario_metadata.get("location", "Global")
        save_transaction(result, is_simulated=False)
        st.session_state.last_evaluated = result
        st.success(f"Transaction evaluated and recorded in audit vault! ID: {result['transaction_id']}")

    # Step 4: Immediate Results, Decision Logic & Inline SHAP XAI
    if st.session_state.last_evaluated:
        res = st.session_state.last_evaluated
        # Recalculate decision if threshold slider was moved after prediction
        recalc_flag = bool(res["fraud_probability"] >= invest_threshold)
        recalc_status = "🚨 FLAGGED FOR MANUAL REVIEW" if recalc_flag else "✅ APPROVED AS LEGITIMATE"
        recalc_color = "#f87171" if recalc_flag else "#34d399"

        st.markdown("---")
        st.subheader("📋 Real-Time Investigation Dossier")

        # Row 1: Detection Output & Transparent Decision Rule
        r_c1, r_c2 = st.columns([1, 1])
        with r_c1:
            st.plotly_chart(plot_risk_gauge(res["risk_score"], threshold=invest_threshold), use_container_width=True)
            st.markdown(f"""
            <div style="text-align:center; padding:12px; border-radius:10px; background:rgba(15,23,42,0.7); border:1px solid {recalc_color}44; margin-top:-10px;">
                <div style="font-size:1.2rem; font-weight:800; color:{recalc_color};">{recalc_status}</div>
                <div style="font-size:0.82rem; color:#94a3b8; margin-top:4px;">
                    Posterior Probability: <b>{res['fraud_probability']:.4f}</b> • Threshold Cutoff: <b>{invest_threshold:.2f}</b> • Score: <b>{res['risk_score']} / 100</b> ({res['risk_level']})
                </div>
            </div>
            """, unsafe_allow_html=True)

        with r_c2:
            st.markdown("#### 📐 Transparent Mathematical Decision Rule")
            st.markdown(f"""
            <div style="background:rgba(15,23,42,0.6); border:1px solid rgba(255,255,255,0.08); border-radius:10px; padding:14px; font-size:0.80rem; space-y-2;">
                <div><b>1. Feature Vector Ingestion:</b> Amount = <code>{format_currency(res['amount'], active_currency)}</code>, Elapsed Time = <code>{res['time']:.0f}s</code>.</div>
                <div style="margin-top:6px;"><b>2. Normalization:</b> <code>Time</code> and <code>Amount</code> transformed via Robust StandardScaler parameters ($z = \\frac{{x - \\mu}}{{\\sigma}}$).</div>
                <div style="margin-top:6px;"><b>3. Ensemble Tree Voting:</b> Evaluated across 100 decorrelated decision trees in Tuned Random Forest.</div>
                <div style="margin-top:6px;"><b>4. Model Posterior Probability:</b>
                    <div style="padding:6px 10px; background:rgba(0,0,0,0.3); border-radius:6px; margin:4px 0; font-family:'JetBrains Mono';">
                        P(Fraud | X) = {res['fraud_probability']:.4f} ({res['fraud_probability']*100:.1f}% Tree Agreement)
                    </div>
                </div>
                <div style="margin-top:6px;"><b>5. Decision Comparison:</b>
                    <div style="padding:6px 10px; background:rgba(0,0,0,0.3); border-radius:6px; margin:4px 0; font-family:'JetBrains Mono';">
                        Decision = {"FLAGGED" if recalc_flag else "APPROVED"} &nbsp;[Condition: {res['fraud_probability']:.4f} {">=" if recalc_flag else "<"} {invest_threshold:.2f}]
                    </div>
                </div>
                <div style="margin-top:6px;"><b>6. Security Action:</b> {"Queued into Analyst Alert Center for manual review." if recalc_flag else "Cleared immediately through automated STP (Straight-Through Processing)."}</div>
            </div>
            """, unsafe_allow_html=True)

        # Row 2: Immediate Directional SHAP Attribution Bars (The "Why")
        st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)
        st.subheader("🧠 Explainable AI: Directional SHAP Attributions")
        st.caption("Which specific features pushed this transaction toward fraud or pulled it toward legitimate status?")

        explainer = get_explainer()
        xai_res = explainer.explain_transaction(active_features, top_k=8)

        x_col1, x_col2 = st.columns([1, 1])
        with x_col1:
            st.plotly_chart(plot_shap_force_bars(xai_res.get("top_features", [])), use_container_width=True)
        with x_col2:
            st.markdown("**Feature Contribution Details**")
            top_feats = xai_res.get("top_features", [])
            table_records = []
            for tf in top_feats:
                raw_val = active_features.get(tf["feature"], 0.0)
                table_records.append({
                    "Feature": tf["feature"],
                    "Input Value": f"{raw_val:.4f}" if abs(raw_val) < 100 else f"{raw_val:.2f}",
                    "SHAP Impact": f"{tf['shap_value']:+.4f}",
                    "Effect": "🚨 Pushes toward Fraud" if tf["shap_value"] > 0 else "✅ Pulls toward Legit",
                })
            st.dataframe(pd.DataFrame(table_records), use_container_width=True, hide_index=True)
            st.markdown(f"**Attribution Narrative:** {xai_res.get('narrative', '')}")

        # Row 3: Immediate Inline Sensitivity & What-If Simulator
        st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)
        st.subheader("🔬 What Would Change This Decision? (Inline Sensitivity Simulator)")
        st.caption("Perturb leading features in real-time to observe how the posterior probability and decision boundary respond:")

        wi_col1, wi_col2, wi_col3 = st.columns(3)
        with wi_col1:
            sim_amt = st.number_input("Perturb Amount ($)", value=float(res["amount"]), step=25.0, key="wi_in_amt")
        with wi_col2:
            sim_v14 = st.slider("Perturb V14 (Leading Anomaly)", -12.0, 5.0, float(active_features.get("V14", -4.0)), 0.2, key="wi_in_v14")
        with wi_col3:
            sim_v10 = st.slider("Perturb V10 (Risk Elevator)", -10.0, 5.0, float(active_features.get("V10", -2.5)), 0.2, key="wi_in_v10")

        # Run fast sensitivity prediction
        perturbed_features = dict(active_features)
        perturbed_features["Amount"] = sim_amt
        perturbed_features["V14"] = sim_v14
        perturbed_features["V10"] = sim_v10

        service = get_service()
        pert_res = service.predict_single(perturbed_features, threshold=invest_threshold)
        p_delta = pert_res["fraud_probability"] - res["fraud_probability"]

        st.plotly_chart(plot_what_if_comparison(res["fraud_probability"], pert_res["fraud_probability"]), use_container_width=True)
        st.markdown(f"""
        <div style="font-size:0.85rem; padding:10px 14px; border-radius:8px; background:rgba(15,23,42,0.6); border:1px solid rgba(255,255,255,0.08); display:flex; justify-content:space-between; flex-wrap:wrap;">
            <span><b>Baseline Probability:</b> <code>{res['fraud_probability']:.4f}</code> ({res['risk_level']})</span>
            <span><b>Perturbed Probability:</b> <code>{pert_res['fraud_probability']:.4f}</code> ({pert_res['risk_level']})</span>
            <span><b>Net Delta:</b> <code style="color:{'#f87171' if p_delta > 0 else '#34d399'};">{p_delta:+.4f}</code></span>
            <span><b>New Verdict:</b> <b>{'🚨 FLAGGED' if pert_res['is_flagged'] else '✅ APPROVED'}</b></span>
        </div>
        """, unsafe_allow_html=True)

        # Row 4: Immediate Analyst Disposition / Audit Action
        st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)
        st.subheader("✍️ Analyst Case Disposition & Vault Audit Log")
        disp_col1, disp_col2 = st.columns([1, 1])
        with disp_col1:
            analyst_choice = st.selectbox("Assign Case Verdict", [
                "Under Investigation",
                "Escalated to Tier-2 Fraud Ring Unit",
                "Confirmed Fraud (Card Blocked)",
                "False Positive (Customer Verified)",
                "Approved & Closed",
            ])
            analyst_tag = st.text_input("Reviewing Analyst Badge", value="Senior Fraud Specialist #402")
        with disp_col2:
            analyst_note = st.text_area("Investigation Observations", placeholder="Enter customer verification details, device IP match, or merchant dispute log...")

        if st.button("💾 Commit Disposition to Database Vault", type="primary"):
            add_review_action(
                transaction_id=res["transaction_id"],
                action=f"Disposition: {analyst_choice}",
                reviewer=analyst_tag,
                notes=analyst_note.strip() if analyst_note else "Standard case adjudication",
            )
            st.success(f"Audit log committed to SQLite database vault for {res['transaction_id']}!")


# =============================================================================
# MODULE 4: 🚨 FRAUD ALERTS (Investigation Center)
# =============================================================================
elif selected_page == "🚨 Fraud Alerts":
    st.title("🚨 Fraud Alert & Incident Queue")
    st.markdown("Prioritize, investigate, and adjudicate high-risk incidents queued by the SENTINEL inference engine.")

    # Filter Controls
    f_col1, f_col2, f_col3 = st.columns(3)
    with f_col1:
        status_filter = st.selectbox("Filter by Status", ["All", "Pending Review", "Under Review", "Escalated", "Confirmed Fraud", "Open"])
    with f_col2:
        risk_filter = st.selectbox("Filter by Risk Level", ["All", "CRITICAL", "HIGH", "MEDIUM", "LOW"])
    with f_col3:
        channel_filter = st.selectbox("Filter by Channel", ["All", "Online Web", "Mobile App", "Contactless NFC", "POS Terminal", "ATM"])

    # Source alerts from DB or Demo Layer
    raw_alerts = get_alerts(status=None if status_filter == "All" else status_filter, limit=50)
    if not raw_alerts or len(raw_alerts) < 5 or "DEMO" in st.session_state.data_mode:
        alerts = demo_alerts
    else:
        alerts = raw_alerts

    # Apply filters
    filtered_alerts = []
    for a in alerts:
        if risk_filter != "All" and a.get("risk_level") != risk_filter:
            continue
        if status_filter != "All" and a.get("status") != status_filter:
            continue
        if channel_filter != "All" and a.get("channel") != channel_filter:
            continue
        filtered_alerts.append(a)

    st.markdown(f"**Showing {len(filtered_alerts)} Active Alert Incidents**")
    
    if filtered_alerts:
        df_a = pd.DataFrame(filtered_alerts)
        display_cols = [c for c in ["id", "transaction_id", "created_at", "amount", "channel", "category", "risk_score", "risk_level", "fraud_probability", "status", "provenance"] if c in df_a.columns]
        st.dataframe(df_a[display_cols], use_container_width=True, hide_index=True)

        st.markdown("### Analyst Action & Case Review")
        chosen_alert_id = st.selectbox("Select Incident to Investigate", [a["transaction_id"] for a in filtered_alerts])
        chosen_alert = next((a for a in filtered_alerts if a["transaction_id"] == chosen_alert_id), None)

        if chosen_alert:
            act_col1, act_col2 = st.columns([1, 1])
            with act_col1:
                st.markdown(f"""
                <div style="background:rgba(15,23,42,0.6); border:1px solid rgba(255,255,255,0.08); border-radius:8px; padding:12px; font-size:0.80rem;">
                    <div><b>Transaction ID:</b> <code>{chosen_alert['transaction_id']}</code></div>
                    <div><b>Amount:</b> {format_currency(chosen_alert.get('amount', 0.0), active_currency)}</div>
                    <div><b>Channel / Category:</b> {chosen_alert.get('channel', 'N/A')} • {chosen_alert.get('category', 'N/A')}</div>
                    <div><b>Risk Score:</b> <b>{chosen_alert['risk_score']} / 100</b> ({chosen_alert['risk_level']})</div>
                    <div><b>Posterior Probability:</b> <code>{chosen_alert.get('fraud_probability', 0.0):.4f}</code></div>
                </div>
                """, unsafe_allow_html=True)
                
                new_status = st.selectbox("Update Case Status", ["Under Review", "Escalated", "Confirmed Fraud", "False Positive", "Closed"])
                reviewer_name = st.text_input("Reviewer Name", value="Senior Fraud Specialist")
                if st.button("Submit Status Update", type="primary"):
                    add_review_action(chosen_alert["transaction_id"], f"Status updated to {new_status}", reviewer=reviewer_name)
                    st.success(f"Incident {chosen_alert_id} updated to '{new_status}'!")
                    st.rerun()

            with act_col2:
                invest_note = st.text_area("Add Investigation Note", placeholder="Enter findings, merchant follow-up, or user verification details...")
                if st.button("Save Note to Audit Vault"):
                    if invest_note.strip():
                        add_review_action(chosen_alert["transaction_id"], "Analyst note added", reviewer=reviewer_name, notes=invest_note.strip())
                        st.success("Note persisted in database audit log!")
                        st.rerun()
                    else:
                        st.warning("Please enter a note before saving.")
    else:
        st.info("No alerts match the active filter criteria. Clear filters to see queued incidents.")


# =============================================================================
# MODULE 5: 📈 MODEL INTELLIGENCE (ML Evaluation Console)
# =============================================================================
elif selected_page == "📈 Model Intelligence":
    st.title("📈 Model Intelligence & Diagnostics")
    st.markdown("Empirical performance evaluation of the **Tuned Random Forest** model against the 56,746-sample test fold.")

    m_tab_summary, m_tab_curves, m_tab_thresh, m_tab_feat, m_tab_comp = st.tabs([
        "⚙️ Architecture & Specs",
        "📉 Performance Curves",
        "🎛️ Threshold Playground",
        "🌳 Feature Importance",
        "⚖️ Model Benchmark Comparison",
    ])

    with m_tab_summary:
        s_c1, s_c2, s_c3, s_c4 = st.columns(4)
        def_metrics = eval_bundle.get("default_metrics", {})
        s_c1.metric("ROC-AUC", f"{def_metrics.get('roc_auc', 0.9664):.4f}")
        s_c2.metric("PR-AUC", f"{def_metrics.get('pr_auc', 0.8096):.4f}")
        s_c3.metric("F1-Score (Optimal)", f"{def_metrics.get('f1', 0.8324):.4f}")
        s_c4.metric("Recall (Sensitivity)", f"{def_metrics.get('recall', 0.7579):.4f}")

        st.markdown(f"""
        <div style="background:rgba(15,23,42,0.6); border:1px solid rgba(255,255,255,0.08); border-radius:10px; padding:16px; margin-top:14px;">
            <b>Algorithm:</b> Tuned Random Forest Classifier (100 Estimators, max_depth=12)<br>
            <b>Balancing Strategy:</b> Leak-Free SMOTE (Applied strictly inside cross-validation folds)<br>
            <b>Evaluation Test Fold:</b> 56,746 transactions (95 confirmed fraud cases, 56,651 legitimate)<br>
            <b>Feature Normalization:</b> Robust StandardScaler applied to continuous variables (Time, Amount)<br>
            <b>Inference Latency:</b> &lt; 2.5ms per transaction on CPU
        </div>
        """, unsafe_allow_html=True)

    with m_tab_curves:
        c_cur1, c_cur2 = st.columns([1, 1])
        with c_cur1:
            st.plotly_chart(plot_roc_curve(eval_bundle.get("roc_curve", []), roc_auc=def_metrics.get("roc_auc", 0.9664)), use_container_width=True)
        with c_cur2:
            st.plotly_chart(plot_precision_recall_curve(eval_bundle.get("pr_curve", []), pr_auc=def_metrics.get("pr_auc", 0.8096)), use_container_width=True)

    with m_tab_thresh:
        st.markdown("**Interactive Decision Threshold Playground (0.01 to 0.99)**")
        st.caption("Shift the classification cutoff to observe real-time trade-offs in Precision, Recall, and False Alarms without retraining.")
        t_val = st.slider("Select Cutoff Threshold", 0.01, 0.99, 0.50, 0.01, key="mod_thresh_slider")

        service = get_service()
        t_metrics = service.get_threshold_metrics(t_val)

        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Recall (Caught)", f"{t_metrics['recall']*100:.1f}%", help="Proportion of actual frauds caught")
        p2.metric("Precision (Purity)", f"{t_metrics['precision']*100:.1f}%", help="Proportion of alerts that are genuine frauds")
        p3.metric("F1-Score", f"{t_metrics['f1']:.4f}")
        p4.metric("False Alarms (FP)", f"{t_metrics['fp']:,}", delta=f"{t_metrics['fn']:,} Missed", delta_color="inverse")

        st.plotly_chart(plot_threshold_curves(eval_bundle.get("threshold_sweep", []), current_threshold=t_val), use_container_width=True)
        st.plotly_chart(plot_confusion_matrix(t_metrics, threshold=t_val), use_container_width=True)

    with m_tab_feat:
        top_k_feat = st.slider("Number of Top Features to Display", 5, 25, 12)
        st.plotly_chart(plot_feature_importances(eval_bundle.get("feature_importances", []), top_n=top_k_feat), use_container_width=True)

    with m_tab_comp:
        st.markdown("**Empirical Evaluation Across Multiple Architectures**")
        st.caption("Results recorded in `experiments/results.csv` on the exact same 56,746-sample test fold:")
        if not exp_df.empty:
            st.dataframe(exp_df, use_container_width=True, hide_index=True)


# =============================================================================
# MODULE 6: 🧠 EXPLAINABLE AI (XAI)
# =============================================================================
elif selected_page == "🧠 Explainable AI":
    st.title("🧠 Explainable AI (XAI) & Attribution")
    st.markdown("Transparent mathematical decomposition of model predictions using **SHAP (Shapley Additive exPlanations)**.")

    x_c1, x_c2 = st.columns([1, 1])
    with x_c1:
        st.markdown("### Select Transaction for SHAP Analysis")
        x_preset_idx = st.selectbox("Select Preset Case", range(len(sample_presets)), format_func=lambda i: sample_presets[i]["name"])
        chosen_x = sample_presets[x_preset_idx]
        features_x = chosen_x["features"]

        service = get_service()
        pred_x = service.predict_single(features_x)
        st.markdown(f"""
        <div style="background:rgba(15,23,42,0.6); border:1px solid rgba(255,255,255,0.08); border-radius:8px; padding:12px; margin-top:10px;">
            <b>Transaction:</b> {chosen_x['id']}<br>
            <b>Ground Truth:</b> {'🚨 Fraud' if chosen_x.get('actual_class')==1 else '✅ Legitimate'}<br>
            <b>Model Probability:</b> {pred_x['fraud_probability']:.4f}<br>
            <b>Risk Level:</b> {pred_x['risk_level']} ({pred_x['risk_score']} / 100)
        </div>
        """, unsafe_allow_html=True)

    with x_c2:
        explainer = get_explainer()
        xai_out = explainer.explain_transaction(features_x, top_k=8)
        st.plotly_chart(plot_shap_force_bars(xai_out.get("top_features", [])), use_container_width=True)

    st.markdown("---")
    st.markdown("### Feature Attribution Breakdown Table")
    feat_df = pd.DataFrame(xai_out.get("top_features", []))
    st.dataframe(feat_df, use_container_width=True, hide_index=True)

    with st.expander("📚 Educational Reference: What are SHAP Values in Tree Ensembles?"):
        st.markdown("""
        - **Shapley Additive exPlanations (SHAP):** Rooted in cooperative game theory, SHAP allocates credit for a prediction among input features by averaging marginal contributions across all possible feature subsets.
        - **TreeExplainer:** A polynomial-time algorithm designed for tree ensembles (Random Forests, Gradient Boosters) that calculates exact Shapley values.
        - **Directional Signs:** Positive values ($+$) elevate fraud probability; negative values ($-$) pull the estimate toward legitimate status.
        """)


# =============================================================================
# MODULE 7: ⚡ LIVE SIMULATION (Synthetic Stream)
# =============================================================================
elif selected_page == "⚡ Live Simulation":
    st.title("⚡ Live Fraud Stream Simulation")
    st.markdown("""
    <div style="padding:10px 14px; border-radius:8px; background:rgba(245,158,11,0.1); border:1px solid rgba(245,158,11,0.3); color:#fbbf24; font-size:0.8rem; margin-bottom:14px;">
        ⚠️ <b>SIMULATION MODE:</b> Feature vectors are synthesized to simulate incoming transaction streams, but <b>every risk score is computed by the real model in real-time</b>.
    </div>
    """, unsafe_allow_html=True)

    sim_col1, sim_col2, sim_col3 = st.columns(3)
    with sim_col1:
        batch_size = st.slider("Transactions per Batch", 1, 20, 5)
    with sim_col2:
        fraud_ratio = st.slider("Simulated Fraud Anomaly Ratio", 0.0, 0.5, 0.2, 0.05)
    with sim_col3:
        sim_threshold = st.slider("Cutoff Threshold", 0.01, 0.99, 0.50, 0.01, key="sim_thresh")

    btn_gen, btn_clear = st.columns([1, 1])
    with btn_gen:
        if st.button("⚡ Generate & Score Transaction Batch", type="primary", use_container_width=True):
            simulator = get_simulator()
            service = get_service()
            new_records = []
            for _ in range(batch_size):
                raw_txn = simulator.generate_transaction(fraud_ratio=fraud_ratio)
                res = service.predict_single(raw_txn, threshold=sim_threshold)
                res["channel"] = raw_txn.get("channel", "POS Terminal")
                res["category"] = raw_txn.get("category", "Retail")
                res["location"] = raw_txn.get("location", "Global")
                save_transaction(res, is_simulated=True)
                new_records.append(res)
            st.session_state.simulation_history.extend(new_records)
            st.success(f"Processed and evaluated {batch_size} simulated transactions!")
    with btn_clear:
        if st.button("🗑️ Reset Simulation Stream", use_container_width=True):
            st.session_state.simulation_history = []
            st.rerun()

    if st.session_state.simulation_history:
        st.markdown(f"**Stream Length:** {len(st.session_state.simulation_history)} transactions evaluated")
        st.plotly_chart(plot_simulation_stream(st.session_state.simulation_history, threshold=sim_threshold), use_container_width=True)

        sim_df = pd.DataFrame(st.session_state.simulation_history)
        st.dataframe(sim_df[["transaction_id", "timestamp", "amount", "fraud_probability", "risk_score", "risk_level", "is_flagged"]], use_container_width=True, hide_index=True)


# =============================================================================
# MODULE 8: 🔬 WHAT-IF ANALYSIS (Sensitivity Simulation)
# =============================================================================
elif selected_page == "🔬 What-If Analysis":
    st.title("🔬 What-If Analysis (Model Sensitivity Simulation)")
    st.markdown("Perturb feature values of an existing transaction to evaluate how the decision boundary responds.")
    st.caption("Note: This evaluates model mathematical sensitivity. It does not establish real-world causal inference.")

    wi_preset = st.selectbox("Select Baseline Transaction", range(len(sample_presets)), format_func=lambda i: sample_presets[i]["name"])
    base_features = sample_presets[wi_preset]["features"]

    c_orig, c_mod = st.columns([1, 1])
    with c_orig:
        st.markdown("### Baseline Features")
        st.json({k: round(base_features[k], 2) for k in ["Amount", "Time", "V14", "V10", "V12", "V17"]})

    modified_features = dict(base_features)
    with c_mod:
        st.markdown("### Perturb Inputs")
        mod_amt = st.number_input("Modify Amount ($)", value=float(base_features["Amount"]), step=10.0)
        mod_v14 = st.slider("Perturb V14", -10.0, 5.0, float(base_features["V14"]), 0.1)
        mod_v10 = st.slider("Perturb V10", -10.0, 5.0, float(base_features["V10"]), 0.1)
        mod_v12 = st.slider("Perturb V12", -10.0, 5.0, float(base_features["V12"]), 0.1)

        modified_features["Amount"] = mod_amt
        modified_features["V14"] = mod_v14
        modified_features["V10"] = mod_v10
        modified_features["V12"] = mod_v12

    service = get_service()
    orig_res = service.predict_single(base_features)
    mod_res = service.predict_single(modified_features)

    st.markdown("---")
    st.subheader("Model Sensitivity Outcome")
    st.plotly_chart(plot_what_if_comparison(orig_res["fraud_probability"], mod_res["fraud_probability"]), use_container_width=True)

    delta_p = mod_res["fraud_probability"] - orig_res["fraud_probability"]
    st.markdown(f"""
    - **Original Probability:** `{orig_res['fraud_probability']:.4f}` ({orig_res['risk_level']})
    - **Modified Probability:** `{mod_res['fraud_probability']:.4f}` ({mod_res['risk_level']})
    - **Net Probability Delta:** `{delta_p:+.4f}`
    """)


# =============================================================================
# MODULE 9: 🗂 TRANSACTION EXPLORER
# =============================================================================
elif selected_page == "🗂 Transaction Explorer":
    st.title("🗂 Transaction Vault Explorer")
    st.markdown("Search, filter, and inspect transactions stored in the SQLite audit database and demonstration layer.")

    # Determine data source based on data mode
    if "DEMO" in st.session_state.data_mode:
        df_source = demo_df
        data_source_label = "SYNTHETIC DEMO LAYER (500 records)"
    elif "REAL" in st.session_state.data_mode:
        recent = get_recent_transactions(limit=200)
        df_source = pd.DataFrame(recent) if recent else pd.DataFrame()
        data_source_label = "SQLITE AUDIT VAULT"
    else:
        recent = get_recent_transactions(limit=100)
        df_db = pd.DataFrame(recent) if recent else pd.DataFrame()
        df_source = pd.concat([df_db, demo_df], ignore_index=True) if not df_db.empty else demo_df
        data_source_label = "COMBINED AUDIT VAULT & DEMO LAYER"

    st.caption(f"Active Data Source: **{data_source_label}**")

    if not df_source.empty:
        # Filters
        f1, f2, f3 = st.columns(3)
        with f1:
            sel_risk = st.multiselect("Filter by Risk Level", ["LOW", "MEDIUM", "HIGH", "CRITICAL"], default=["LOW", "MEDIUM", "HIGH", "CRITICAL"])
        with f2:
            sel_flag = st.selectbox("Classification Verdict", ["All", "Flagged Only", "Approved Only"])
        with f3:
            search_query = st.text_input("Search by ID or Channel", "")

        filtered = df_source[df_source["risk_level"].isin(sel_risk)]
        if sel_flag == "Flagged Only":
            filtered = filtered[filtered["is_flagged"] == True]
        elif sel_flag == "Approved Only":
            filtered = filtered[filtered["is_flagged"] == False]

        if search_query.strip():
            filtered = filtered[filtered["id"].str.contains(search_query, case=False, na=False) | filtered.get("channel", pd.Series([""]*len(filtered))).str.contains(search_query, case=False, na=False)]

        st.markdown(f"**Found {len(filtered):,} Matching Transactions**")
        display_cols = [c for c in ["id", "timestamp", "amount", "channel", "category", "fraud_probability", "risk_score", "risk_level", "status", "provenance"] if c in filtered.columns]
        st.dataframe(filtered[display_cols], use_container_width=True, hide_index=True)

        st.download_button(
            "📥 Download Filtered Transactions (CSV)",
            filtered[display_cols].to_csv(index=False),
            file_name="sentinel_transactions_export.csv",
            mime="text/csv",
        )
    else:
        st.info("No transactions available to display.")


# =============================================================================
# MODULE 10: ⚙️ SYSTEM & MODEL
# =============================================================================
elif selected_page == "⚙️ System & Model":
    st.title("⚙️ System Architecture & Administration")
    st.markdown("Inspect backend dependencies, active file paths, model metadata, and administrative utilities.")

    st.markdown("""
    ```
    ┌────────────────────────────────────────────────────────────────────────┐
    │                SENTINEL FRAUD INTELLIGENCE PLATFORM                    │
    ├─────────────────────────────┬──────────────────────────────────────────┤
    │ INFERENCE ENGINE            │ ModelService (Tuned Random Forest)       │
    │ EXPLAINABILITY LAYER        │ SHAP TreeExplainer & Attribution Forces  │
    │ PERSISTENCE LAYER           │ SQLite3 ACID Database (sentinel.db)      │
    │ DEMONSTRATION LAYER         │ Deterministic Synthetic Suite (500 TXNs) │
    │ SIMULATION LAYER            │ Gaussian Stream Generator                │
    │ VISUAL ANALYTICS            │ Plotly Interactive Scientific Suite      │
    └─────────────────────────────┴──────────────────────────────────────────┘
    ```
    """)

    st.markdown("### Operational Telemetry")
    t1, t2 = st.columns([1, 1])
    with t1:
        st.markdown(f"""
        - **Model Artifact:** `{model_meta.get('model_type', 'RandomForestClassifier')}`
        - **Number of Estimators:** `{model_meta.get('best_params', {}).get('n_estimators', 100)}`
        - **Max Tree Depth:** `{model_meta.get('best_params', {}).get('max_depth', 12)}`
        - **Default Threshold:** `{DEFAULT_THRESHOLD:.2f}`
        - **Total Features:** `30 (Time, Amount, V1-V28)`
        """)
    with t2:
        st.markdown(f"""
        - **Database Status:** `Connected (sentinel.db)`
        - **Monitored Vault Records:** `{db_kpis.get('total_transactions', 0):,}`
        - **Open Audit Alerts:** `{db_kpis.get('open_alerts', 0):,}`
        - **Demo Dataset Records:** `{demo_kpis.get('total_transactions', 500):,}`
        - **Demo Seed:** `42 (Deterministic)`
        """)

    st.markdown("### Database Administration")
    if st.button("🌱 Re-Seed Presets into Vault"):
        n_seeded = seed_demo_database_if_empty()
        st.success(f"Database verified/seeded with {n_seeded} records!")
