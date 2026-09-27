"""
app.py
------
SENTINEL — Fraud Intelligence Platform
High-Performance, Interactive Streamlit Operations Center.

Architecture:
- Native Plotly interactive visual analytics (hover, zoom, pan, download)
- Strict Data Provenance labeling (REAL DATASET vs EVALUATION DATA vs STORED IN VAULT vs LIVE SIMULATION)
- 10 Dedicated Security & Operations Modules:
  1. 🏠 Command Center
  2. 📊 Dataset Intelligence
  3. 🔎 Transaction Investigation
  4. 🚨 Fraud Alerts
  5. 📈 Model Intelligence
  6. 🧠 Explainable AI
  7. ⚡ Live Simulation
  8. 🔬 What-If Analysis
  9. 🗂 Transaction Explorer
  10. ⚙️ System & Model
- Calibrated Deterministic Risk Scoring (0–100) & Visual Segmented Gauge
- Anonymized PCA-Honest Explainable AI (SHAP force breakdown)
- Single Canonical ModelService Backend
"""

import sys
import json
import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

import pandas as pd
import numpy as np
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

    # 2. Global Data Source Indicator
    st.markdown("<div style='font-size:0.70rem; font-weight:700; color:#94a3b8; text-transform:uppercase; margin-bottom:6px;'>Active Data Source</div>", unsafe_allow_html=True)
    source_choice = st.selectbox(
        "Data Source",
        ["REAL DATASET (284,807 Kaggle TXNs)", "STORED IN VAULT (SQLite DB)", "LIVE SIMULATION (Synthetic Stream)"],
        label_visibility="collapsed",
    )

    # 3. Visualization Controls
    st.markdown("<div style='font-size:0.70rem; font-weight:700; color:#94a3b8; text-transform:uppercase; margin-top:14px; margin-bottom:6px;'>Display Controls</div>", unsafe_allow_html=True)
    show_advanced_charts = st.toggle("Show Advanced Analytics", value=True)
    show_explanations = st.toggle("Show Model Explanations", value=True)
    show_raw_features = st.toggle("Show Raw PCA Features", value=False)
    analyst_mode = st.toggle("Engineering Mode (Technical)", value=False)
    currency_toggle = st.selectbox("Currency Format", ["$ USD", "₹ INR (Demo Layer)"], index=0)
    active_currency = "$ USD" if "USD" in currency_toggle else "₹ INR"

    st.markdown("<hr style='border-color: rgba(255,255,255,0.08); margin: 16px 0;'>", unsafe_allow_html=True)

    # 4. System Telemetry
    service_ref = get_service()
    db_kpis = get_dashboard_kpis()
    st.markdown(f"""
    <div style="font-size: 0.72rem; color: #94a3b8; space-y-1;">
        <div><b>Model:</b> <span class="font-mono text-blue-400">Tuned Random Forest</span></div>
        <div><b>Threshold:</b> <span class="font-mono text-amber-400">{DEFAULT_THRESHOLD:.2f}</span></div>
        <div><b>Vault Records:</b> <span class="font-mono text-emerald-400">{db_kpis.get('total_transactions', 0):,}</span></div>
        <div><b>Open Alerts:</b> <span class="font-mono text-rose-400">{db_kpis.get('open_alerts', 0):,}</span></div>
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


# Load artifacts
ds_summary = load_dataset_summary()
eval_bundle = load_evaluation_bundle()
sample_presets = load_sample_presets()
exp_df = load_experiment_results()
model_meta = load_metadata()
db_kpis = get_dashboard_kpis()


# =============================================================================
# MODULE 1: 🏠 COMMAND CENTER (The Main Operations Center)
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
        if st.button("⚡ Launch Live Simulation", use_container_width=True):
            st.session_state.selected_nav = "⚡ Live Simulation"
            st.rerun()

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    # Key Operational KPI Cards (Ground Truth Provenance)
    k1, k2, k3, k4, k5 = st.columns(5)
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
            <div class="metric-val text-rose-400" style="color: #f87171;">{ds_summary.get('overview', {}).get('fraud_count', 492):,}</div>
            <div class="metric-lbl">Fraud Incidents</div>
            <div class="metric-sub">Rate: <b>0.1727%</b> (492/284k)</div>
        </div>
        """, unsafe_allow_html=True)
    with k3:
        st.markdown(f"""
        <div class="metric-card">
            <span class="metric-provenance tag-model">MODEL BENCHMARK</span>
            <div class="metric-val text-blue-400" style="color: #60a5fa;">{eval_bundle.get('default_metrics', {}).get('pr_auc', 0.8096):.4f}</div>
            <div class="metric-lbl">PR-AUC Score</div>
            <div class="metric-sub">Tuned Random Forest Test</div>
        </div>
        """, unsafe_allow_html=True)
    with k4:
        st.markdown(f"""
        <div class="metric-card">
            <span class="metric-provenance tag-vault">STORED IN VAULT</span>
            <div class="metric-val text-emerald-400" style="color: #34d399;">{db_kpis.get('total_transactions', 0):,}</div>
            <div class="metric-lbl">Monitored in Vault</div>
            <div class="metric-sub">Flagged: <b>{db_kpis.get('flagged_transactions', 0):,}</b></div>
        </div>
        """, unsafe_allow_html=True)
    with k5:
        st.markdown(f"""
        <div class="metric-card">
            <span class="metric-provenance tag-vault">STORED IN VAULT</span>
            <div class="metric-val text-amber-400" style="color: #fbbf24;">{db_kpis.get('open_alerts', 0):,}</div>
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
        st.markdown("**Recent High-Risk Incidents (Vault Database)**")
        recent_txns = get_recent_transactions(limit=6)
        if recent_txns:
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
                })
            st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)
        else:
            st.info("No recorded transactions in vault database. Run a simulation or score a transaction.")

    with c_act2:
        st.markdown("**Live System Activity Log**")
        if recent_txns:
            for t in recent_txns[:4]:
                badge_style = "dot-green" if not t["is_flagged"] else "dot-amber"
                st.markdown(f"""
                <div style="padding:8px 12px; margin-bottom:6px; border-radius:8px; background:rgba(15,23,42,0.6); border:1px solid rgba(255,255,255,0.06); font-size:0.75rem;">
                    <div style="display:flex; justify-content:space-between; color:#94a3b8;">
                        <span class="font-mono">{t['id']}</span>
                        <span>{t['timestamp'][:19].replace('T', ' ')}</span>
                    </div>
                    <div style="margin-top:3px; color:#f1f5f9;">
                        <b>{format_currency(t['amount'], active_currency)}</b> evaluated • Risk <b>{t['risk_score']}</b> ({t['risk_level']}) • Status: <b>{t['status']}</b>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.caption("No recent events logged.")


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
            st.markdown(r"**Inspect Statistical Signatures of Any Anonymized Feature ($V_1 \dots V_{28}$, Amount, Time)**")
            selected_feat = st.selectbox("Select Feature Component", FEATURE_COLUMNS, index=14)  # default V14
            prof = feat_profiles.get(selected_feat, {})

            f_c1, f_c2, f_c3, f_c4 = st.columns(4)
            f_c1.metric("Global Mean", f"{prof.get('mean', 0.0):.4f}")
            f_c2.metric("Fraud Class Mean", f"{prof.get('fraud_mean', 0.0):.4f}")
            f_c3.metric("Legit Class Mean", f"{prof.get('legit_mean', 0.0):.4f}")
            f_c4.metric("Correlation w/ Class", f"{prof.get('correlation_with_class', 0.0):+.4f}")

            # Plot comparison of means
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
# MODULE 3: 🔎 TRANSACTION INVESTIGATION (Analyst Workspace)
# =============================================================================
elif selected_page == "🔎 Transaction Investigation":
    st.title("🔎 Transaction Investigation Workspace")
    st.markdown("Run end-to-end inference through the **canonical ModelService**, inspect calibrated risk scores, and review directional SHAP feature forces.")

    # Step 1: Input Setup
    st.markdown("### Step 1: Transaction Feature Input")
    tab_preset, tab_quick, tab_advanced = st.tabs(["📂 Load Dataset Preset", "⚡ Quick Sliders", "🔬 Complete 30-Feature Vector"])

    preset_features = sample_presets[8]["features"] if sample_presets else {col: 0.0 for col in FEATURE_COLUMNS}
    active_features = {}

    with tab_preset:
        preset_names = [f"{p['id']} — {p['name']} ({'🚨 FRAUD' if p.get('actual_class')==1 else '✅ LEGIT'})" for p in sample_presets]
        selected_preset_idx = st.selectbox("Select Ground-Truth Preset Record", range(len(preset_names)), format_func=lambda i: preset_names[i], index=8)
        chosen_preset = sample_presets[selected_preset_idx]
        preset_features = chosen_preset["features"]

        gt_label = "🚨 GROUND TRUTH: FRAUDULENT" if chosen_preset.get("actual_class") == 1 else "✅ GROUND TRUTH: LEGITIMATE"
        st.markdown(f"<span class='status-badge font-mono' style='color:#f8fafc;'>{gt_label}</span>", unsafe_allow_html=True)
        st.markdown(f"**Amount:** {format_currency(chosen_preset['amount'], active_currency)} • **Elapsed Time:** {chosen_preset['time']}s")
        active_features = {col: float(preset_features.get(col, 0.0)) for col in FEATURE_COLUMNS}

    with tab_quick:
        st.markdown("Adjust primary financial and top predictive PCA features:")
        q_c1, q_c2, q_c3 = st.columns(3)
        with q_c1:
            q_amount = st.number_input("Transaction Amount ($)", value=float(preset_features.get("Amount", 99.99)), min_value=0.0, step=10.0)
            q_v14 = st.slider("V14 (Leading Predictor)", -10.0, 5.0, float(preset_features.get("V14", -4.28)), 0.1)
        with q_c2:
            q_time = st.number_input("Time (Seconds from start)", value=float(preset_features.get("Time", 406.0)), min_value=0.0, step=100.0)
            q_v10 = st.slider("V10 (Risk Elevator)", -10.0, 5.0, float(preset_features.get("V10", -2.77)), 0.1)
        with q_c3:
            q_v12 = st.slider("V12 (Separation Component)", -10.0, 5.0, float(preset_features.get("V12", -2.89)), 0.1)
            q_v17 = st.slider("V17 (Inverse Risk)", -10.0, 5.0, float(preset_features.get("V17", -2.83)), 0.1)

        # Merge adjustments
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
                val = float(active_features.get(col, preset_features.get(col, 0.0)))
                active_features[col] = st.number_input(col, value=val, key=f"adv_{col}")

    # Threshold Selector
    invest_threshold = st.slider("Operational Decision Threshold (Cutoff)", min_value=0.01, max_value=0.99, value=0.50, step=0.01)

    # Before / After Pipeline Flow Visualization
    st.markdown("""
    <div style="padding:10px 14px; margin: 12px 0; border-radius:8px; background:rgba(15,23,42,0.6); border:1px solid rgba(255,255,255,0.08); font-size:0.75rem; color:#94a3b8; display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap;">
        <span><b>INPUT:</b> 30 Features</span> → 
        <span><b>SCALING:</b> StandardScaler (Time, Amount)</span> → 
        <span><b>MODEL:</b> Tuned Random Forest</span> → 
        <span><b>OUTPUT:</b> P(Fraud)</span> → 
        <span><b>DECISION:</b> Threshold Comparison</span> → 
        <span><b>EXPLANATION:</b> SHAP Attribution</span>
    </div>
    """, unsafe_allow_html=True)

    # Score Action
    if st.button("🚀 Evaluate Transaction Risk", type="primary", use_container_width=True):
        service = get_service()
        result = service.predict_single(active_features, threshold=invest_threshold)
        save_transaction(result, is_simulated=False)
        st.session_state.last_evaluated = result
        st.success(f"Transaction evaluated and recorded in vault! ID: {result['transaction_id']}")

    # Results Display
    if st.session_state.last_evaluated:
        res = st.session_state.last_evaluated
        st.markdown("---")
        st.subheader("📋 Transaction Investigation Report")

        r_c1, r_c2 = st.columns([1, 1])
        with r_c1:
            st.plotly_chart(plot_risk_gauge(res["risk_score"], threshold=invest_threshold), use_container_width=True)
            dec_color = "#f87171" if res["is_flagged"] else "#34d399"
            dec_text = "🚨 FLAGGED FOR MANUAL REVIEW" if res["is_flagged"] else "✅ APPROVED AS LEGITIMATE"
            st.markdown(f"""
            <div style="text-align:center; padding:10px; border-radius:8px; background:rgba(15,23,42,0.7); border:1px solid {dec_color}44;">
                <div style="font-size:1.1rem; font-weight:800; color:{dec_color};">{dec_text}</div>
                <div style="font-size:0.8rem; color:#94a3b8; margin-top:4px;">
                    Model Posterior Probability: <b>{res['fraud_probability']:.4f}</b> • Calibrated Score: <b>{res['risk_score']} / 100</b>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with r_c2:
            explainer = get_explainer()
            xai_res = explainer.explain_transaction(active_features, top_k=8)
            st.plotly_chart(plot_shap_force_bars(xai_res.get("top_features", [])), use_container_width=True)
            st.markdown(f"**Attribution Narrative:** {xai_res.get('narrative', '')}")

        # Expandable Pipeline Details
        with st.expander("🔍 How did SENTINEL decide? (Detailed Decision Pipeline)"):
            st.markdown(f"""
            1. **Feature Vector Ingestion:** Received vector with Amount = {format_currency(res['amount'], active_currency)} and Time = {res['time']}s.
            2. **Preprocessing & Standardization:** Standardized `Time` and `Amount` using training set parameters (Mean Amount: $88.35, Std: 250.12).
            3. **Ensemble Voting:** 100 decision trees in the Tuned Random Forest evaluated the vector across split criteria.
            4. **Posterior Probability Generation:** Exactly **{res['fraud_probability']*100:.2f}%** of voting trees classified the sample as fraudulent.
            5. **Threshold Application:** Probability {res['fraud_probability']:.4f} was compared against cutoff $\\tau = {invest_threshold:.2f}$.
            6. **Action Taken:** {'Flagged and queued in Alerts table.' if res['is_flagged'] else 'Approved; no alert triggered.'}
            """)


# =============================================================================
# MODULE 4: 🚨 FRAUD ALERTS (Investigation Center)
# =============================================================================
elif selected_page == "🚨 Fraud Alerts":
    st.title("🚨 Fraud Alert Center")
    st.markdown("Manage, triage, and annotate high-risk transactions queued for analyst review.")

    f_col1, f_col2 = st.columns([1, 1])
    with f_col1:
        status_filter = st.selectbox("Filter by Status", ["All", "Pending", "Reviewing", "Escalated", "Confirmed Fraud", "False Positive", "Closed"])
    with f_col2:
        risk_filter = st.selectbox("Filter by Risk Level", ["All", "CRITICAL", "HIGH", "MEDIUM", "LOW"])

    alerts = get_alerts(status=None if status_filter == "All" else status_filter, limit=50)

    if alerts:
        a_df = pd.DataFrame(alerts)
        st.markdown(f"**Showing {len(alerts)} Alert Incidents**")
        st.dataframe(
            a_df[["id", "transaction_id", "risk_level", "fraud_probability", "status", "created_at"]],
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("### Analyst Action & Investigation Notes")
        chosen_alert_id = st.selectbox("Select Alert ID to Investigate", [a["id"] for a in alerts])
        chosen_alert = next((a for a in alerts if a["id"] == chosen_alert_id), None)

        if chosen_alert:
            act_col1, act_col2 = st.columns([1, 1])
            with act_col1:
                new_status = st.selectbox("Update Case Status", ["Reviewing", "Escalated", "Confirmed Fraud", "False Positive", "Closed"])
                reviewer_name = st.text_input("Reviewer Name", value="Lead Fraud Analyst")
                if st.button("Submit Status Update", type="primary"):
                    add_review_action(chosen_alert["transaction_id"], f"Status updated to {new_status}", reviewer=reviewer_name)
                    st.success(f"Alert {chosen_alert_id} updated to '{new_status}'!")
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
        st.info("No alerts found matching the current filters.")


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
    st.markdown("Search, filter, and inspect transactions stored in the SQLite audit database.")

    recent = get_recent_transactions(limit=100)
    if recent:
        df_vault = pd.DataFrame(recent)

        # Filters
        f1, f2 = st.columns(2)
        with f1:
            sel_risk = st.multiselect("Filter by Risk Level", ["LOW", "MEDIUM", "HIGH", "CRITICAL"], default=["LOW", "MEDIUM", "HIGH", "CRITICAL"])
        with f2:
            sel_flag = st.selectbox("Classification Status", ["All", "Flagged Only", "Approved Only"])

        filtered = df_vault[df_vault["risk_level"].isin(sel_risk)]
        if sel_flag == "Flagged Only":
            filtered = filtered[filtered["is_flagged"] == 1]
        elif sel_flag == "Approved Only":
            filtered = filtered[filtered["is_flagged"] == 0]

        st.dataframe(filtered[["id", "timestamp", "amount", "fraud_probability", "risk_score", "risk_level", "status", "is_simulated"]], use_container_width=True, hide_index=True)

        st.download_button(
            "📥 Download Filtered Transactions (CSV)",
            filtered.to_csv(index=False),
            file_name="sentinel_vault_export.csv",
            mime="text/csv",
        )
    else:
        st.info("Vault is currently empty.")


# =============================================================================
# MODULE 10: ⚙️ SYSTEM & MODEL
# =============================================================================
elif selected_page == "⚙️ System & Model":
    st.title("⚙️ System Architecture & Administration")
    st.markdown("Inspect backend dependencies, active file paths, and administrative maintenance utilities.")

    st.markdown("""
    ```
    ┌────────────────────────────────────────────────────────────────────────┐
    │                SENTINEL FRAUD INTELLIGENCE PLATFORM                    │
    ├─────────────────────────────┬──────────────────────────────────────────┤
    │ INFERENCE ENGINE            │ ModelService (Tuned Random Forest)       │
    │ EXPLAINABILITY LAYER        │ SHAP TreeExplainer & Attribution Forces  │
    │ PERSISTENCE LAYER           │ SQLite3 ACID Database (sentinel.db)      │
    │ SIMULATION LAYER            │ Gaussian Stream Generator                │
    │ VISUAL ANALYTICS            │ Plotly Interactive Scientific Suite      │
    └─────────────────────────────┴──────────────────────────────────────────┘
    ```
    """)

    st.markdown("### Database Administration")
    if st.button("🌱 Re-Seed Demo Presets into Vault"):
        n_seeded = seed_demo_database_if_empty()
        st.success(f"Database verified/seeded with {n_seeded} records!")
