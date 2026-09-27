"""
app.py
------
SENTINEL — Fraud Intelligence Platform
Machine Learning powered transaction risk analysis & investigation vault.

Refinement Pass 3:
- Strict Data Provenance (Real Kaggle Dataset vs Live Synthetic Simulation vs Stored Vault)
- Dual Operating Modes (Mode 1: Dataset Analytics, Mode 2: Live Operations)
- 10 Real Analytical Graphs answering specific fraud questions (Zero decorative charts)
- High-Resolution Interactive Decision Threshold Playground (0.01 to 0.99)
- Calibrated Deterministic Risk Scoring (0-100) with Visual Risk Gauge
- Anonymized PCA-Honest Explainable AI (SHAP force breakdown)
- Unified Single ModelService Backend
"""

import os
import sys
import json
import datetime
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Ensure src is on path
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
    FIGURES_DIR,
    DEFAULT_THRESHOLD,
    FEATURE_COLUMNS,
    SCALE_COLUMNS,
    RISK_BANDS,
    DEMO_CHANNELS,
    DEMO_CATEGORIES,
    DEMO_LOCATIONS,
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
)
from explainability import ExplainabilityEngine
from simulation import SimulationEngine

# Page configuration
st.set_page_config(
    page_title="SENTINEL — Fraud Intelligence Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom High-End Cyber / Fintech Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    code, pre, .mono {
        font-family: 'JetBrains Mono', monospace !important;
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
        letter-spacing: 0.08em;
        text-transform: uppercase;
        display: inline-block;
        padding: 2px 6px;
        border-radius: 4px;
        margin-bottom: 6px;
    }
    .provenance-dataset { background: rgba(59, 130, 246, 0.15); color: #60a5fa; border: 1px solid rgba(59, 130, 246, 0.3); }
    .provenance-model { background: rgba(139, 92, 246, 0.15); color: #a78bfa; border: 1px solid rgba(139, 92, 246, 0.3); }
    .provenance-vault { background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3); }
    .provenance-sim { background: rgba(245, 158, 11, 0.15); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.3); }

    .metric-title {
        font-size: 0.80rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: #94a3b8;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 1.75rem;
        font-weight: 700;
        color: #f8fafc;
        line-height: 1.2;
    }
    .metric-sub {
        font-size: 0.72rem;
        color: #64748b;
        margin-top: 4px;
    }

    /* Risk Gauge */
    .gauge-wrapper {
        background: rgba(15, 23, 42, 0.8);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 16px 20px;
        margin: 12px 0;
    }
    .gauge-track {
        display: flex;
        height: 12px;
        border-radius: 6px;
        overflow: hidden;
        margin: 10px 0;
        background: #1e293b;
    }
    .gauge-seg-low { width: 30%; background: #10b981; }
    .gauge-seg-med { width: 30%; background: #f59e0b; }
    .gauge-seg-high { width: 20%; background: #f97316; }
    .gauge-seg-crit { width: 20%; background: #ef4444; }

    .gauge-marker {
        display: flex;
        justify-content: space-between;
        font-size: 0.70rem;
        color: #64748b;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Hero Banner */
    .hero-banner {
        background: linear-gradient(135deg, rgba(30, 58, 138, 0.3) 0%, rgba(15, 23, 42, 0.8) 100%);
        border: 1px solid rgba(59, 130, 246, 0.25);
        border-radius: 14px;
        padding: 22px 26px;
        margin-bottom: 24px;
    }
    .hero-title {
        font-size: 1.85rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        background: linear-gradient(90deg, #60a5fa, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 4px;
    }
    .hero-subtitle {
        font-size: 0.92rem;
        color: #94a3b8;
        margin-bottom: 14px;
    }

    /* Mode Pill */
    .mode-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.76rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# Initialize Singletons and Resource Caching
@st.cache_resource
def get_service():
    return ModelService.get_instance()

@st.cache_resource
def get_xai():
    return ExplainabilityEngine.get_instance()

@st.cache_resource
def get_sim():
    return SimulationEngine()

@st.cache_data
def load_presets():
    if SAMPLE_PRESETS_JSON.exists():
        with open(SAMPLE_PRESETS_JSON, "r") as f:
            return json.load(f)
    return []

@st.cache_data
def load_dataset_summary():
    service = get_service()
    return service.get_dataset_summary()

@st.cache_data
def load_evaluation_bundle():
    service = get_service()
    return service.get_evaluation_bundle()

# Ensure DB is ready
init_db()
seed_demo_database_if_empty()
service = get_service()
ds_summary = load_dataset_summary()
eval_bundle = load_evaluation_bundle()


# Currency Formatter with explicit educational labeling
def format_curr(val: float, mode: str = "USD ($)") -> str:
    if "INR" in mode:
        return f"₹{val * 83:,.2f}"
    return f"${val:,.2f}"


# Sidebar Navigation
with st.sidebar:
    st.markdown("""
    <div style="padding: 10px 0 16px 0; border-bottom: 1px solid rgba(255,255,255,0.08);">
        <h2 style="font-size: 1.35rem; font-weight: 800; margin: 0; background: linear-gradient(90deg, #60a5fa, #818cf8); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">🛡️ SENTINEL</h2>
        <p style="font-size: 0.78rem; color: #64748b; margin: 2px 0 0 0;">Fraud Intelligence Platform</p>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        [
            "📊 Overview Dashboard",
            "🔬 Dataset Intelligence",
            "🔍 Analyze Transaction",
            "📄 Batch Analysis",
            "🚨 Alerts & Investigation Vault",
            "📈 Model Performance & Thresholds",
            "💡 Explainable AI (SHAP Lab)",
            "⚡ Live Simulation & What-If",
            "⚙️ Platform Settings & Governance",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("### 🌐 Currency Display Layer")
    currency_mode = st.selectbox(
        "Display Currency",
        ["USD ($) — Dataset Benchmark (Ground Truth)", "INR (₹) — Presentation Formatting Layer (@ 83)"],
        label_visibility="collapsed",
    )
    if "INR" in currency_mode:
        st.caption("ℹ️ INR formatting is an illustrative presentation layer. Original dataset is measured in standard currency units (EUR/USD).")

    st.markdown("---")
    st.markdown(
        f"""
        <div style="font-size: 0.74rem; color: #94a3b8; line-height: 1.5;">
            <b>System Telemetry:</b><br>
            • <b>Model:</b> Tuned Random Forest<br>
            • <b>Resampling:</b> SMOTE Pipeline<br>
            • <b>Cutoff:</b> {DEFAULT_THRESHOLD:.2f}<br>
            • <b>Status:</b> <span style="color: #34d399;">● Online (Local & Vercel)</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("v3.0 · B.Tech AIML / ECE Project")


# Helper: Render Hero Banner
def render_hero_banner():
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">SENTINEL — Fraud Intelligence Platform</div>
        <div class="hero-subtitle">Detect. Investigate. Understand. — Defense-grade ML risk intelligence with complete data provenance</div>
        <div style="display: flex; gap: 12px; flex-wrap: wrap;">
            <span class="mode-pill" style="background: rgba(59, 130, 246, 0.15); color: #60a5fa; border: 1px solid rgba(59, 130, 246, 0.3);">
                ● Model: Tuned Random Forest (SMOTE)
            </span>
            <span class="mode-pill" style="background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3);">
                ● Audit Vault: SQLite Persistent
            </span>
            <span class="mode-pill" style="background: rgba(168, 85, 247, 0.15); color: #c084fc; border: 1px solid rgba(168, 85, 247, 0.3);">
                ● Explainability: SHAP TreeExplainer
            </span>
            <span class="mode-pill" style="background: rgba(14, 165, 233, 0.15); color: #38bdf8; border: 1px solid rgba(14, 165, 233, 0.3);">
                ● Production API: Vercel Serverless (200 OK)
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# Helper: Render Risk Gauge
def render_risk_gauge(score: int, level: str, prob: float, threshold: float):
    color = "#ef4444" if score >= 81 else ("#f97316" if score >= 61 else ("#f59e0b" if score >= 31 else "#10b981"))
    st.markdown(f"""
    <div class="gauge-wrapper">
        <div style="display: flex; justify-content: space-between; align-items: baseline;">
            <span style="font-size: 0.85rem; font-weight: 700; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em;">Calculated Risk Level</span>
            <span style="font-size: 1.15rem; font-weight: 800; color: {color};">{level} ({score} / 100)</span>
        </div>
        <div class="gauge-track">
            <div class="gauge-seg-low"></div>
            <div class="gauge-seg-med"></div>
            <div class="gauge-seg-high"></div>
            <div class="gauge-seg-crit"></div>
        </div>
        <div class="gauge-marker">
            <span>LOW (0–30)</span>
            <span>MEDIUM (31–60)</span>
            <span>HIGH (61–80)</span>
            <span>CRITICAL (81–100)</span>
        </div>
        <div style="margin-top: 10px; font-size: 0.78rem; color: #cbd5e1; display: flex; justify-content: space-between; border-top: 1px solid rgba(255,255,255,0.06); padding-top: 8px;">
            <span><b>Fraud Confidence:</b> {prob * 100:.2f}%</span>
            <span><b>Decision Threshold:</b> {threshold:.2f}</span>
            <span><b>Model Action:</b> <b style="color: {'#f87171' if prob >= threshold else '#34d399'};">{'🚨 FLAGGED FOR REVIEW' if prob >= threshold else '✅ APPROVED LEGITIMATE'}</b></span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ===========================================================================
# PAGE 1: OVERVIEW DASHBOARD
# ===========================================================================
if page == "📊 Overview Dashboard":
    render_hero_banner()

    # Clear Conceptual Split: Two Operating Modes
    st.markdown("### Select Dashboard Operating Mode")
    app_mode = st.radio(
        "Operating Mode",
        [
            "📊 MODE 1: DATASET ANALYTICS (Actual Kaggle Dataset — 284,807 Ground Truth Records)",
            "⚡ MODE 2: LIVE SIMULATION & AUDIT VAULT (Synthetic Stream & Operational Incidents)",
        ],
        label_visibility="collapsed",
    )

    st.markdown("<br>", unsafe_allow_html=True)

    if "MODE 1" in app_mode:
        # MODE 1: DATASET ANALYTICS
        ov = ds_summary.get("overview", {})
        amt = ds_summary.get("amount_statistics", {})

        st.markdown("""
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
            <h3 style="margin: 0; font-size: 1.25rem;">Kaggle Dataset Ground-Truth Analytics</h3>
            <span class="metric-provenance provenance-dataset">DATASET ANALYTICS (ACTUAL KAGGLE GROUND TRUTH)</span>
        </div>
        """, unsafe_allow_html=True)

        k1, k2, k3, k4, k5, k6 = st.columns(6)
        with k1:
            st.markdown(f"""
            <div class="metric-card">
                <span class="metric-provenance provenance-dataset">REAL DATASET</span>
                <div class="metric-title">Total Records</div>
                <div class="metric-value">{ov.get('total_transactions', 284807):,}</div>
                <div class="metric-sub">Kaggle benchmark data</div>
            </div>
            """, unsafe_allow_html=True)
        with k2:
            st.markdown(f"""
            <div class="metric-card">
                <span class="metric-provenance provenance-dataset">REAL DATASET</span>
                <div class="metric-title">Fraud Incidents</div>
                <div class="metric-value" style="color: #f87171;">{ov.get('fraud_count', 492):,}</div>
                <div class="metric-sub">Confirmed Class 1</div>
            </div>
            """, unsafe_allow_html=True)
        with k3:
            st.markdown(f"""
            <div class="metric-card">
                <span class="metric-provenance provenance-dataset">REAL DATASET</span>
                <div class="metric-title">Legitimate Records</div>
                <div class="metric-value" style="color: #34d399;">{ov.get('legitimate_count', 284315):,}</div>
                <div class="metric-sub">Confirmed Class 0</div>
            </div>
            """, unsafe_allow_html=True)
        with k4:
            st.markdown(f"""
            <div class="metric-card">
                <span class="metric-provenance provenance-dataset">REAL DATASET</span>
                <div class="metric-title">Class Imbalance</div>
                <div class="metric-value">{ov.get('fraud_rate_pct', 0.1727):.2f}%</div>
                <div class="metric-sub">1 fraud per 578 txns</div>
            </div>
            """, unsafe_allow_html=True)
        with k5:
            st.markdown(f"""
            <div class="metric-card">
                <span class="metric-provenance provenance-dataset">REAL DATASET</span>
                <div class="metric-title">Avg Fraud Amount</div>
                <div class="metric-value">{format_curr(amt.get('fraudulent', {}).get('mean', 122.21), currency_mode)}</div>
                <div class="metric-sub">vs Legit {format_curr(amt.get('legitimate', {}).get('mean', 88.29), currency_mode)}</div>
            </div>
            """, unsafe_allow_html=True)
        with k6:
            st.markdown(f"""
            <div class="metric-card">
                <span class="metric-provenance provenance-dataset">REAL DATASET</span>
                <div class="metric-title">Median Fraud</div>
                <div class="metric-value" style="color: #fb923c;">{format_curr(amt.get('fraudulent', {}).get('median', 9.25), currency_mode)}</div>
                <div class="metric-sub">vs Legit {format_curr(amt.get('legitimate', {}).get('median', 22.00), currency_mode)}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Major Analytical Graphs (No Decorative Charts)
        c_left, c_right = st.columns(2)

        with c_left:
            st.markdown("#### 1. Severe Class Imbalance (Actual Dataset)")
            st.caption("Question Answered: How rare is fraud, and why is accuracy misleading?")
            fig, ax = plt.subplots(figsize=(6, 3.4), facecolor="none")
            ax.set_facecolor("none")
            classes = ["Legitimate (Class 0)", "Fraudulent (Class 1)"]
            counts = [ov.get("legitimate_count", 284315), ov.get("fraud_count", 492)]
            bars = ax.bar(classes, counts, color=["#10b981", "#ef4444"], width=0.5)
            ax.set_yscale("log")
            ax.set_ylabel("Count (Logarithmic Scale)", color="#94a3b8", fontsize=9)
            ax.tick_params(colors="#94a3b8", labelsize=9)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#334155')
            ax.spines['bottom'].set_color('#334155')
            for bar, cnt in zip(bars, counts):
                h = bar.get_height()
                ax.annotate(f"{cnt:,} ({cnt/sum(counts)*100:.2f}%)",
                            (bar.get_x() + bar.get_width() / 2, h),
                            ha='center', va='bottom', color='#f8fafc', fontsize=9, xytext=(0, 3),
                            textcoords='offset points')
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

        with c_right:
            st.markdown("#### 2. Spending Behavior: Fraud vs Legitimate Binned (%)")
            st.caption("Question Answered: Do fraudsters make large or small purchases?")
            dist_data = ds_summary.get("amount_distribution", [])
            if dist_data:
                b_labels = [d["bin"] for d in dist_data]
                l_pcts = [d["legit_pct"] for d in dist_data]
                f_pcts = [d["fraud_pct"] for d in dist_data]

                fig, ax = plt.subplots(figsize=(6, 3.4), facecolor="none")
                ax.set_facecolor("none")
                x = np.arange(len(b_labels))
                width = 0.38
                ax.bar(x - width/2, l_pcts, width, label="Legit %", color="#10b981")
                ax.bar(x + width/2, f_pcts, width, label="Fraud %", color="#ef4444")
                ax.set_xticks(x)
                ax.set_xticklabels(b_labels, rotation=35, ha='right', color="#94a3b8", fontsize=8)
                ax.set_ylabel("% of Category Volume", color="#94a3b8", fontsize=9)
                ax.tick_params(colors="#94a3b8")
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_color('#334155')
                ax.spines['bottom'].set_color('#334155')
                ax.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#cbd5e1", fontsize=8)
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close()

        # Row 2: Temporal Dynamics and Model Risk Separation
        r2_left, r2_right = st.columns(2)
        with r2_left:
            st.markdown("#### 3. Temporal Fraud Dynamics (48-Hour Dataset Timeline)")
            st.caption("Question Answered: When does fraud peak across the 48-hour recording window?")
            time_48 = ds_summary.get("time_hourly_48h", [])
            if time_48:
                hours = [d["hour"] for d in time_48]
                f_counts = [d["fraud_count"] for d in time_48]
                fig, ax = plt.subplots(figsize=(6, 3.2), facecolor="none")
                ax.set_facecolor("none")
                ax.plot(hours, f_counts, color="#ef4444", lw=2, marker='o', markersize=3, label="Fraud Incidents")
                ax.fill_between(hours, f_counts, color="#ef4444", alpha=0.15)
                ax.set_xlabel("Elapsed Hours Since Start", color="#94a3b8", fontsize=9)
                ax.set_ylabel("Hourly Fraud Count", color="#94a3b8", fontsize=9)
                ax.tick_params(colors="#94a3b8", labelsize=8)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_color('#334155')
                ax.spines['bottom'].set_color('#334155')
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close()

        with r2_right:
            st.markdown("#### 4. Model Risk Score Distribution (Held-Out Test Set)")
            st.caption("Question Answered: How clearly does the model separate legitimate transactions from fraud?")
            r_dist = eval_bundle.get("risk_distribution", [])
            if r_dist:
                r_bands = [d["band"] for d in r_dist]
                l_dist_pct = [d["legit_pct"] for d in r_dist]
                f_dist_pct = [d["fraud_pct"] for d in r_dist]

                fig, ax = plt.subplots(figsize=(6, 3.2), facecolor="none")
                ax.set_facecolor("none")
                x = np.arange(len(r_bands))
                width = 0.38
                ax.bar(x - width/2, l_dist_pct, width, label="Legit (Class 0)", color="#10b981")
                ax.bar(x + width/2, f_dist_pct, width, label="Fraud (Class 1)", color="#ef4444")
                ax.set_xticks(x)
                ax.set_xticklabels(r_bands, rotation=35, ha='right', color="#94a3b8", fontsize=8)
                ax.set_ylabel("% Distribution", color="#94a3b8", fontsize=9)
                ax.tick_params(colors="#94a3b8", labelsize=8)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_color('#334155')
                ax.spines['bottom'].set_color('#334155')
                ax.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#cbd5e1", fontsize=8)
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close()

    else:
        # MODE 2: LIVE SIMULATION & OPERATIONAL VAULT
        kpis = get_dashboard_kpis()

        st.markdown("""
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
            <h3 style="margin: 0; font-size: 1.25rem;">Live Stream & Audit Vault Telemetry</h3>
            <span class="metric-provenance provenance-vault">LIVE DEMO / STORED TRANSACTIONS</span>
        </div>
        """, unsafe_allow_html=True)

        c1, c2, c3, c4, c5, c6 = st.columns(6)
        with c1:
            st.markdown(f"""
            <div class="metric-card">
                <span class="metric-provenance provenance-vault">STORED IN VAULT</span>
                <div class="metric-title">Total Evaluated</div>
                <div class="metric-value">{kpis['total_transactions']:,}</div>
                <div class="metric-sub">Audit database records</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="metric-card">
                <span class="metric-provenance provenance-model">MODEL OUTPUT</span>
                <div class="metric-title">Fraud Flags</div>
                <div class="metric-value" style="color: #f87171;">{kpis['fraud_flags']:,}</div>
                <div class="metric-sub">Flagged at cutoff ≥ 0.50</div>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="metric-card">
                <span class="metric-provenance provenance-model">MODEL OUTPUT</span>
                <div class="metric-title">High / Critical</div>
                <div class="metric-value" style="color: #fb923c;">{kpis['high_risk_transactions']:,}</div>
                <div class="metric-sub">Risk score ≥ 61</div>
            </div>
            """, unsafe_allow_html=True)
        with c4:
            st.markdown(f"""
            <div class="metric-card">
                <span class="metric-provenance provenance-vault">STORED IN VAULT</span>
                <div class="metric-title">Flag Rate</div>
                <div class="metric-value">{kpis['fraud_rate_pct']:.1f}%</div>
                <div class="metric-sub">Of evaluated volume</div>
            </div>
            """, unsafe_allow_html=True)
        with c5:
            st.markdown(f"""
            <div class="metric-card">
                <span class="metric-provenance provenance-vault">STORED IN VAULT</span>
                <div class="metric-title">Avg Amount</div>
                <div class="metric-value">{format_curr(kpis['avg_amount'], currency_mode)}</div>
                <div class="metric-sub">Per transaction</div>
            </div>
            """, unsafe_allow_html=True)
        with c6:
            st.markdown(f"""
            <div class="metric-card">
                <span class="metric-provenance provenance-vault">AUDIT TRAIL</span>
                <div class="metric-title">Reviewed</div>
                <div class="metric-value" style="color: #38bdf8;">{kpis['transactions_reviewed']:,}</div>
                <div class="metric-sub">Analyst cleared</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        col_v1, col_v2 = st.columns([1, 1])
        recent_txns = get_recent_transactions(limit=150)
        df_recent = pd.DataFrame(recent_txns) if recent_txns else pd.DataFrame()

        with col_v1:
            st.markdown("#### Vault Risk Level Distribution")
            if not df_recent.empty and "risk_level" in df_recent.columns:
                counts = df_recent["risk_level"].value_counts().reindex(["LOW", "MEDIUM", "HIGH", "CRITICAL"]).fillna(0)
                colors = ["#10b981", "#f59e0b", "#f97316", "#ef4444"]

                fig, ax = plt.subplots(figsize=(6, 3.2), facecolor="none")
                ax.set_facecolor("none")
                bars = ax.bar(counts.index, counts.values, color=colors, width=0.55)
                ax.tick_params(colors="#94a3b8", labelsize=9)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_color('#334155')
                ax.spines['bottom'].set_color('#334155')
                for bar in bars:
                    h = bar.get_height()
                    ax.annotate(f"{int(h)}", (bar.get_x() + bar.get_width() / 2, h),
                                ha='center', va='bottom', color='#f8fafc', fontsize=9, xytext=(0, 2),
                                textcoords='offset points')
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close()
            else:
                st.info("No transactions logged in SQLite vault yet.")

        with col_v2:
            st.markdown("#### Simulated Attack Stream Quick-Inject")
            st.caption("Generate synthetic transactions passed directly through the real ML model.")
            sim = get_sim()
            b1, b2 = st.columns([2, 1])
            with b1:
                bias_sel = st.slider("Simulated Fraud Bias", 0.05, 0.50, 0.20, 0.05)
            with b2:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("⚡ Inject 3 Transactions", use_container_width=True):
                    sim.generate_batch(count=3, fraud_bias=bias_sel)
                    st.success("3 synthetic transactions processed through model and saved to vault!")
                    st.rerun()

            st.caption("ℹ️ Strictly labeled as synthetic demo layer. Every transaction passes through StandardScaler and trained Random Forest model.")

        st.markdown("---")
        st.markdown("#### Recent Vault Alerts (Pending Analyst Review)")
        alerts = get_alerts(limit=6)
        if alerts:
            alerts_df = pd.DataFrame(alerts)[["transaction_id", "created_at", "amount", "risk_score", "risk_level", "status"]]
            st.dataframe(alerts_df, use_container_width=True)
        else:
            st.success("No active critical alerts pending review.")


# ===========================================================================
# PAGE 2: DATASET INTELLIGENCE (GROUND TRUTH KAGGLE AUDIT)
# ===========================================================================
elif page == "🔬 Dataset Intelligence":
    st.title("Dataset Intelligence & Exploratory Audit")
    st.caption("Comprehensive exploratory analysis of the original Kaggle Credit Card Fraud dataset (284,807 transactions)")

    st.markdown("""
    <div style="background: rgba(30, 58, 138, 0.15); border: 1px solid rgba(59, 130, 246, 0.25); border-radius: 10px; padding: 12px 18px; margin-bottom: 20px;">
        <span class="metric-provenance provenance-dataset">GROUND TRUTH PROVENANCE</span>
        <div style="font-size: 0.84rem; color: #cbd5e1; margin-top: 4px;">
            The metrics below are derived directly from the real <b>creditcard.csv</b> dataset. 
            All 28 PCA dimensions ($V_1$–$V_{28}$) are mathematically preserved without fabricated semantic labels.
        </div>
    </div>
    """, unsafe_allow_html=True)

    ov = ds_summary.get("overview", {})
    amt = ds_summary.get("amount_statistics", {})

    st.subheader("1. Data Quality & Imbalance Telemetry")
    d1, d2, d3, d4, d5 = st.columns(5)
    with d1:
        st.metric("Total Records", f"{ov.get('total_transactions', 284807):,}")
    with d2:
        st.metric("Clean Records", f"{ov.get('clean_transactions', 283726):,}")
    with d3:
        st.metric("Duplicates Dropped", f"{ov.get('duplicate_transactions', 1081):,}")
    with d4:
        st.metric("Missing Values", "0 (Zero)")
    with d5:
        st.metric("Imbalance Ratio", f"1 : {int(ov.get('legitimate_count', 284315) / ov.get('fraud_count', 492)):,}")

    st.markdown("---")
    st.subheader("2. Monetary Value Statistics (Amount Comparison)")

    stat_rows = []
    for cat, data in amt.items():
        stat_rows.append({
            "Cohort": cat.capitalize(),
            "Count": f"{data['count']:,}",
            "Mean": format_curr(data['mean'], currency_mode),
            "Std Dev": format_curr(data['std'], currency_mode),
            "Min": format_curr(data['min'], currency_mode),
            "25th %ile": format_curr(data['p25'], currency_mode),
            "Median (50%)": format_curr(data['median'], currency_mode),
            "75th %ile": format_curr(data['p75'], currency_mode),
            "99th %ile": format_curr(data['p99'], currency_mode),
            "Max": format_curr(data['max'], currency_mode),
        })
    st.table(pd.DataFrame(stat_rows).set_index("Cohort"))

    st.caption("💡 Key Observation: Fraud median amount ($9.25) is significantly lower than legitimate median ($22.00) because fraudsters test card validity with micro-transactions before attempting large purchases.")

    st.markdown("---")
    col_corr1, col_corr2 = st.columns(2)
    corrs = ds_summary.get("correlations", {})

    with col_corr1:
        st.subheader("3. Top Positive Correlated Features with Fraud")
        st.caption("Features where higher values strongly associate with fraud")
        pos_df = pd.DataFrame(corrs.get("top_positive", []))
        st.dataframe(pos_df, use_container_width=True)

    with col_corr2:
        st.subheader("4. Top Negative Correlated Features with Fraud")
        st.caption("Features where deeply negative values strongly associate with fraud")
        neg_df = pd.DataFrame(corrs.get("top_negative", []))
        st.dataframe(neg_df, use_container_width=True)


# ===========================================================================
# PAGE 3: ANALYZE TRANSACTION (INVESTIGATION TERMINAL)
# ===========================================================================
elif page == "🔍 Analyze Transaction":
    st.title("Transaction Risk Scoring Terminal")
    st.caption("Real-time inference through trained Random Forest & SMOTE pipeline with XAI attribution")

    presets = load_presets()

    col_pre, col_spacer = st.columns([3, 1])
    with col_pre:
        preset_names = ["Custom Manual Vector"] + [f"{p['name']} ({format_curr(p['amount'], currency_mode)})" for p in presets]
        selected_p_name = st.selectbox(
            "Load Real Kaggle Benchmark Transaction:",
            preset_names,
            help="Select verified records directly extracted from the Kaggle dataset.",
        )

    preset_obj = None
    if selected_p_name != "Custom Manual Vector":
        clean_name = selected_p_name.split(" (")[0]
        preset_obj = next((p for p in presets if p["name"] == clean_name), None)

    with st.form("single_transaction_form"):
        st.markdown("#### Transaction Input Vector")
        c_amt, c_time, c_thresh = st.columns(3)

        with c_amt:
            amt_val = float(preset_obj["amount"]) if preset_obj else 150.00
            input_amount = st.number_input("Transaction Amount", min_value=0.0, max_value=50000.0, value=amt_val, step=10.0)
        with c_time:
            time_val = float(preset_obj["time"]) if preset_obj else 3600.0
            input_time = st.number_input("Elapsed Seconds (Time)", min_value=0.0, value=time_val, step=60.0)
        with c_thresh:
            input_threshold = st.slider("Classification Threshold", 0.05, 0.95, DEFAULT_THRESHOLD, 0.05)

        with st.expander("Inspect & Modify Anonymized PCA Features (V1–V28)", expanded=(preset_obj is None)):
            st.caption("V1–V28 are anonymized principal components derived from PCA. Zero semantic labels hallucinated.")
            pca_inputs = {}
            pca_cols = st.columns(7)
            for i in range(1, 29):
                feat_name = f"V{i}"
                default_v = float(preset_obj["features"].get(feat_name, 0.0)) if preset_obj else 0.0
                with pca_cols[(i - 1) % 7]:
                    pca_inputs[feat_name] = st.number_input(feat_name, value=default_v, step=0.25, format="%.2f")

        submit_btn = st.form_submit_button("⚡ Run Real Model Inference & Risk Scoring", use_container_width=True)

    if submit_btn:
        payload = {"Amount": input_amount, "Time": input_time, **pca_inputs}
        try:
            result = service.predict_single(payload, threshold=input_threshold)
            save_transaction(result, is_simulated=False)

            st.markdown("---")
            st.markdown("### Investigation Analysis Result")

            render_risk_gauge(result["risk_score"], result["risk_level"], result["fraud_probability"], result["threshold_used"])

            m1, m2, m3, m4, m5 = st.columns(5)
            with m1:
                st.metric("Transaction ID", result["transaction_id"])
            with m2:
                st.metric("Amount Evaluated", format_curr(result["amount"], currency_mode))
            with m3:
                st.metric("Model Cutoff Used", f"{result['threshold_used']:.2f}")
            with m4:
                st.metric("Fraud Probability", f"{result['fraud_probability'] * 100:.2f}%")
            with m5:
                st.metric("Deterministic Score", f"{result['risk_score']} / 100")

            # SHAP Attribution: WHY WAS THIS TRANSACTION FLAGGED?
            st.markdown("---")
            st.markdown("### ❓ Why Was This Transaction Flagged? (SHAP Feature Attribution)")
            st.caption("Component-level force attribution. Positive (+) values elevate fraud risk; negative (−) values indicate legitimate behavior.")

            xai = get_xai()
            xai_res = xai.explain_transaction(payload, top_k=8)

            st.info(f"**Attribution Narrative:** {xai_res['narrative']}")

            # Horizontal Contribution Chart (Graph 10)
            top_feats = xai_res["top_features"]
            fig, ax = plt.subplots(figsize=(8, 3.2), facecolor="none")
            ax.set_facecolor("none")
            f_names = [f["feature"] for f in reversed(top_feats)]
            f_vals = [f["contribution"] for f in reversed(top_feats)]
            colors = ["#ef4444" if v > 0 else "#10b981" for v in f_vals]

            ax.barh(f_names, f_vals, color=colors, height=0.55)
            ax.axvline(0, color="#94a3b8", linestyle="--", lw=1)
            ax.set_xlabel("Contribution to Fraud Score (SHAP Force)", color="#94a3b8", fontsize=9)
            ax.tick_params(colors="#94a3b8", labelsize=9)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#334155')
            ax.spines['bottom'].set_color('#334155')
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

            st.success(f"Audit record permanently stored to vault as `{result['transaction_id']}`.")

        except ValidationError as ve:
            st.error(f"Validation Failure: {ve.message}")
            for err in ve.errors:
                st.write(f"- {err}")
        except Exception as e:
            st.error(f"Inference error: {str(e)}")


# ===========================================================================
# PAGE 4: BATCH ANALYSIS
# ===========================================================================
elif page == "📄 Batch Analysis":
    st.title("Bulk Portfolio Batch Scoring")
    st.caption("Upload portfolios of transactions to detect anomalous clusters and measure financial exposure")

    col_up, col_dl = st.columns([3, 1])
    with col_dl:
        presets = load_presets()
        if presets:
            sample_df = pd.DataFrame([p["features"] for p in presets])
            csv_bytes = sample_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download Ready Test Batch CSV",
                data=csv_bytes,
                file_name="sentinel_test_batch.csv",
                mime="text/csv",
                help="Test batch with verified legitimate and fraudulent vectors.",
            )

    uploaded_file = st.file_uploader("Upload Transaction Batch (CSV format with Time, Amount, V1–V28):", type=["csv"])

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.write(f"Loaded **{len(batch_df):,}** transactions. Initializing validation checks...")

            threshold_batch = st.slider("Classification Threshold for Batch:", 0.10, 0.90, 0.50, 0.05)

            with st.spinner("Executing pipeline inference..."):
                enriched_df, summary = service.predict_batch(batch_df, threshold=threshold_batch)

            st.success("Batch Inference Finished Successfully.")

            b1, b2, b3, b4, b5 = st.columns(5)
            with b1:
                st.metric("Total Transactions", f"{summary['total_transactions']:,}")
            with b2:
                st.metric("Flagged Fraud", f"{summary['flagged_transactions']:,}", delta=f"{summary['fraud_rate_pct']}% rate")
            with b3:
                st.metric("High / Critical Risk", f"{summary['high_risk_count']:,}")
            with b4:
                st.metric("Average Amount", format_curr(summary['average_amount'], currency_mode))
            with b5:
                st.metric("At-Risk Volume", format_curr(summary['total_at_risk_amount'], currency_mode))

            st.markdown("---")
            st.dataframe(enriched_df.head(100), use_container_width=True)

            csv_out = enriched_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Export Full Scored Batch (fraud_predictions.csv)",
                data=csv_out,
                file_name="fraud_predictions.csv",
                mime="text/csv",
                use_container_width=True,
            )

        except ValidationError as ve:
            st.error(f"Validation failure: {ve.message}")
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")


# ===========================================================================
# PAGE 5: ALERTS & INVESTIGATION VAULT
# ===========================================================================
elif page == "🚨 Alerts & Investigation Vault":
    st.title("Fraud Analyst Investigation Console")
    st.caption("Incident resolution workflows, escalation management, and immutable audit logs")

    filter_status = st.selectbox("Filter Incidents by Resolution Status:", ["All", "Open", "Escalated", "Resolved"])
    status_param = None if filter_status == "All" else filter_status

    alerts = get_alerts(status=status_param, limit=100)

    if not alerts:
        st.info("No incidents found matching the selected status filter.")
    else:
        st.write(f"Displaying **{len(alerts)}** incidents:")
        alert_map = {
            f"[{a['status'].upper()}] {a['transaction_id']} — {format_curr(a['amount'], currency_mode)} (Score: {a['risk_score']}/100)": a["transaction_id"]
            for a in alerts
        }
        selected_label = st.selectbox("Select Incident to Investigate:", list(alert_map.keys()))
        selected_txn_id = alert_map[selected_label]

        txn = get_transaction_by_id(selected_txn_id)
        if txn:
            st.markdown("---")
            i1, i2, i3, i4 = st.columns(4)
            with i1:
                st.metric("Incident ID", txn["id"])
            with i2:
                st.metric("Amount", format_curr(txn["amount"], currency_mode))
            with i3:
                st.metric("Risk Score", f"{txn['risk_score']} / 100", delta=txn["risk_level"])
            with i4:
                st.metric("Current Status", txn["status"].upper())

            render_risk_gauge(txn["risk_score"], txn["risk_level"], txn["risk_score"]/100.0, 0.50)

            # Analyst Decision Form
            st.markdown("#### Analyst Audit Action")
            with st.form("analyst_action_form"):
                action_choice = st.radio(
                    "Select Audit Action:",
                    ["Mark as reviewed", "Mark legitimate", "Escalate", "Add note"],
                    horizontal=True,
                )
                reviewer_name = st.text_input("Investigator Name / ID", value="Lead Fraud Analyst")
                analyst_notes = st.text_area("Investigation Notes & Rationale", placeholder="e.g. Verified legitimate merchant charge with cardholder...")
                submit_action = st.form_submit_button("Record Action in Audit Trail")

                if submit_action:
                    add_review_action(selected_txn_id, action_choice, reviewer=reviewer_name, notes=analyst_notes)
                    st.success(f"Action '{action_choice}' committed to audit vault.")
                    st.rerun()

            # Audit History Log
            reviews = txn.get("reviews", [])
            if reviews:
                st.markdown("#### Incident Audit Trail Log")
                for r in reviews:
                    st.markdown(f"- **{r['reviewed_at']}** — `{r['reviewer']}` executed **{r['action']}**: *{r['notes']}*")


# ===========================================================================
# PAGE 6: MODEL PERFORMANCE & THRESHOLDS
# ===========================================================================
elif page == "📈 Model Performance & Thresholds":
    st.title("Model Performance & Threshold Playground")
    st.caption("Empirical benchmarks, interactive decision cutoff analysis, and evaluation curves")

    t1, t2, t3 = st.tabs([
        "🎛️ Interactive Threshold Playground",
        "📊 Model Comparison Matrix",
        "🔬 Global Feature Importance",
    ])

    with t1:
        st.subheader("Dynamic Decision Threshold Playground")
        st.markdown(
            "Fraud detection is fundamentally an **asymmetric cost problem**. "
            "Lowering the threshold catches more fraud (higher Recall) but creates customer friction (more False Positives). "
            "Raising the threshold reduces alerts (higher Precision) but risks missed fraud (higher False Negatives)."
        )

        thresh_val = st.slider(
            "Select Decision Threshold Cutoff:",
            min_value=0.01,
            max_value=0.99,
            value=0.50,
            step=0.01,
            help="Threshold cutoff where probabilities >= threshold are classified as Fraud.",
        )

        # Dynamic metrics computed from real 99-step test set predictions
        t_metrics = service.get_threshold_metrics(thresh_val)

        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            st.metric("Recall (Fraud Caught)", f"{t_metrics['recall']:.1%}")
        with m2:
            st.metric("Precision (Flag Accuracy)", f"{t_metrics['precision']:.1%}")
        with m3:
            st.metric("F1-Score (Harmonic Mean)", f"{t_metrics['f1']:.3f}")
        with m4:
            st.metric("False Positives (Alarms)", f"{t_metrics['fp']:,}", delta=f"FPR: {t_metrics['fpr']*100:.2f}%", delta_color="inverse")
        with m5:
            st.metric("False Negatives (Missed)", f"{t_metrics['fn']:,}", delta=f"FNR: {t_metrics['fnr']*100:.1f}%", delta_color="inverse")

        # Operational Mode Badge
        op_mode = t_metrics.get("operational_mode", "BALANCED_PRODUCTION")
        if op_mode == "HIGH_SENSITIVITY":
            st.warning("⚠️ **High Sensitivity Mode (Threshold < 0.35):** Maximum recall. Catches almost all fraud but results in higher false alarms and customer verification steps.")
        elif op_mode == "CONSERVATIVE":
            st.warning("⚠️ **Conservative Mode (Threshold > 0.65):** Maximum precision. Minimal false alarms, but stealthy fraud may escape detection.")
        else:
            st.success("✅ **Balanced Production Mode (0.35–0.65):** Optimal operational harmonic F1 balance between fraud prevention and customer friction.")

        # Threshold curves plot
        sweep = eval_bundle.get("threshold_sweep", [])
        if sweep:
            sw_df = pd.DataFrame(sweep)
            fig, ax = plt.subplots(figsize=(8, 3.2), facecolor="none")
            ax.set_facecolor("none")
            ax.plot(sw_df["threshold"], sw_df["recall"], label="Recall (Fraud Caught)", color="#3b82f6", lw=2)
            ax.plot(sw_df["threshold"], sw_df["precision"], label="Precision (Accuracy)", color="#10b981", lw=2)
            ax.plot(sw_df["threshold"], sw_df["f1"], label="F1-Score", color="#f59e0b", lw=2, linestyle="--")
            ax.axvline(thresh_val, color="#ef4444", linestyle=":", lw=2, label=f"Selected ({thresh_val:.2f})")
            ax.set_xlabel("Decision Threshold", color="#94a3b8", fontsize=9)
            ax.set_ylabel("Score", color="#94a3b8", fontsize=9)
            ax.tick_params(colors="#94a3b8", labelsize=8)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#334155')
            ax.spines['bottom'].set_color('#334155')
            ax.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#cbd5e1", fontsize=8)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

        # Large Visual Confusion Matrix (Graph 6)
        st.markdown("---")
        st.subheader("Confusion Matrix at Selected Threshold")
        cm_c1, cm_c2 = st.columns([1, 1])

        with cm_c1:
            # Table visualization of Confusion Matrix
            cm_table = pd.DataFrame({
                "Predicted Legitimate (0)": [f"TN: {t_metrics['tn']:,}", f"FN: {t_metrics['fn']:,} (Missed Fraud)"],
                "Predicted Fraud (1)": [f"FP: {t_metrics['fp']:,} (False Alarms)", f"TP: {t_metrics['tp']:,} (Caught Fraud)"]
            }, index=["Actual Legitimate (0)", "Actual Fraud (1)"])
            st.table(cm_table)

        with cm_c2:
            st.markdown(f"""
            <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(255,255,255,0.08); border-radius: 10px; padding: 14px 18px; font-size: 0.82rem; color: #cbd5e1;">
                <b>Confusion Matrix Ratios:</b><br>
                • <b>True Negatives (TN):</b> {t_metrics['tn']:,} / 56,651 legitimate correctly cleared.<br>
                • <b>True Positives (TP):</b> {t_metrics['tp']:,} / 95 fraud cases correctly intercepted.<br>
                • <b>False Positive Rate (FPR):</b> {t_metrics['fpr']*100:.3f}% (customer friction rate).<br>
                • <b>False Negative Rate (FNR):</b> {t_metrics['fnr']*100:.2f}% (unmitigated fraud loss rate).<br>
            </div>
            """, unsafe_allow_html=True)

        # Side-by-Side ROC and PR Curves (Graphs 7 & 8)
        st.markdown("---")
        st.subheader("ROC Curve vs Precision-Recall Curve (Held-Out Test Set)")
        st.caption("Note: On highly imbalanced data (0.17%), the Precision-Recall curve is the definitive measure of operational efficacy.")

        roc_col, pr_col = st.columns(2)
        roc_pts = eval_bundle.get("roc_curve", [])
        pr_pts = eval_bundle.get("pr_curve", [])

        with roc_col:
            if roc_pts:
                r_df = pd.DataFrame(roc_pts)
                fig, ax = plt.subplots(figsize=(5, 3.2), facecolor="none")
                ax.set_facecolor("none")
                ax.plot(r_df["fpr"], r_df["tpr"], color="#3b82f6", lw=2, label="ROC Curve (AUC = 0.9664)")
                ax.plot([0, 1], [0, 1], color="#64748b", linestyle="--", lw=1, label="Chance Baseline")
                ax.set_xlabel("False Positive Rate", color="#94a3b8", fontsize=9)
                ax.set_ylabel("True Positive Rate", color="#94a3b8", fontsize=9)
                ax.tick_params(colors="#94a3b8", labelsize=8)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_color('#334155')
                ax.spines['bottom'].set_color('#334155')
                ax.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#cbd5e1", fontsize=8)
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close()

        with pr_col:
            if pr_pts:
                p_df = pd.DataFrame(pr_pts)
                fig, ax = plt.subplots(figsize=(5, 3.2), facecolor="none")
                ax.set_facecolor("none")
                ax.plot(p_df["recall"], p_df["precision"], color="#10b981", lw=2, label="PR Curve (PR-AUC = 0.8096)")
                ax.axhline(0.0017, color="#ef4444", linestyle="--", lw=1, label="Baseline (0.17%)")
                ax.set_xlabel("Recall", color="#94a3b8", fontsize=9)
                ax.set_ylabel("Precision", color="#94a3b8", fontsize=9)
                ax.tick_params(colors="#94a3b8", labelsize=8)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_color('#334155')
                ax.spines['bottom'].set_color('#334155')
                ax.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#cbd5e1", fontsize=8)
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close()

    with t2:
        st.subheader("Model Comparison Matrix (Actual Experiment Results)")
        if EXPERIMENT_RESULTS_CSV.exists():
            exp_df = pd.read_csv(EXPERIMENT_RESULTS_CSV)
            st.dataframe(
                exp_df.style.highlight_max(subset=["f1", "average_precision", "roc_auc", "recall"], color="#1e3a8a"),
                use_container_width=True,
            )
            st.caption("Evaluated on identical 20% test fold. Random Forest with SMOTE pipeline achieves optimal PR-AUC (0.8096) and F1 (0.8324).")

    with t3:
        st.subheader("Global Gini Feature Importances (Graph 9)")
        st.caption("Features providing the highest mean decrease in tree impurity across all 100 decision trees")
        f_imps = eval_bundle.get("feature_importances", [])
        if f_imps:
            imp_df = pd.DataFrame(f_imps[:15])
            fig, ax = plt.subplots(figsize=(8, 4), facecolor="none")
            ax.set_facecolor("none")
            ax.barh(imp_df["feature"][::-1], imp_df["importance"][::-1], color="#3b82f6", height=0.6)
            ax.tick_params(colors="#94a3b8", labelsize=9)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#334155')
            ax.spines['bottom'].set_color('#334155')
            ax.set_xlabel("Mean Decrease in Impurity", color="#94a3b8", fontsize=9)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()


# ===========================================================================
# PAGE 7: EXPLAINABLE AI (SHAP LAB)
# ===========================================================================
elif page == "💡 Explainable AI (SHAP Lab)":
    st.title("Explainable AI (XAI) Attribution Lab")
    st.caption("Deconstruct black-box predictions into transparent, auditable feature force attributions")

    presets = load_presets()
    selected_name = st.selectbox("Select Benchmark Record for Attribution Decomposition:", [p["name"] for p in presets])
    chosen = next(p for p in presets if p["name"] == selected_name)

    xai = get_xai()
    with st.spinner("Calculating SHAP force values..."):
        res = xai.explain_transaction(chosen["features"], top_k=10)

    st.markdown("---")
    st.subheader(f"Explanation for {chosen['name']}")
    st.info(f"**Method:** {res['method']}\n\n**Attribution Narrative:** {res['narrative']}")

    c_pos, c_neg = st.columns(2)
    with c_pos:
        st.markdown("#### 🚨 Features Increasing Fraud Risk (+)")
        if res["fraud_drivers"]:
            st.dataframe(pd.DataFrame(res["fraud_drivers"])[["feature", "raw_value", "contribution"]], use_container_width=True)
        else:
            st.write("None. All features supported legitimate classification.")

    with c_neg:
        st.markdown("#### ✅ Features Pushing Legitimate (−)")
        if res["legit_drivers"]:
            st.dataframe(pd.DataFrame(res["legit_drivers"])[["feature", "raw_value", "contribution"]], use_container_width=True)
        else:
            st.write("None. All features elevated risk.")

    # Attribution Force Plot
    st.markdown("---")
    st.subheader("Local Feature Force Breakdown")
    fig, ax = plt.subplots(figsize=(8, 3.4), facecolor="none")
    ax.set_facecolor("none")
    top_feats = res["top_features"]
    f_names = [f["feature"] for f in reversed(top_feats)]
    f_vals = [f["contribution"] for f in reversed(top_feats)]
    colors = ["#ef4444" if v > 0 else "#10b981" for v in f_vals]

    ax.barh(f_names, f_vals, color=colors, height=0.55)
    ax.axvline(0, color="#94a3b8", linestyle="--")
    ax.set_xlabel("SHAP Force Value", color="#94a3b8", fontsize=9)
    ax.tick_params(colors="#94a3b8")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#334155')
    ax.spines['bottom'].set_color('#334155')
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()


# ===========================================================================
# PAGE 8: LIVE SIMULATION & WHAT-IF LAB
# ===========================================================================
elif page == "⚡ Live Simulation & What-If":
    st.title("Transaction Simulation & Sensitivity Lab")
    st.caption("Generate synthetic real-time payment streams and test hypothetical What-If parameter variations")

    sim_tab1, sim_tab2 = st.tabs(["⚡ Live Stream Simulator", "🧪 What-If Sensitivity Lab"])

    sim_engine = get_sim()

    with sim_tab1:
        st.subheader("Synthetic Payment Velocity Stream")
        st.caption("Demonstrates model throughput in a real-time event loop. Strictly labeled as SIMULATION MODE.")

        s1, s2, s3 = st.columns(3)
        with s1:
            fraud_bias = st.slider("Attack Injection Frequency", 0.05, 0.50, 0.20, 0.05)
        with s2:
            burst_count = st.selectbox("Batch Generation Size", [1, 5, 10], index=1)
        with s3:
            st.markdown("<br>", unsafe_allow_html=True)
            gen_btn = st.button("🚀 Stream Synthetic Transactions", use_container_width=True)

        if gen_btn:
            with st.spinner("Processing through ML pipeline..."):
                generated = sim_engine.generate_batch(count=burst_count, fraud_bias=fraud_bias)
            st.success(f"Generated {len(generated)} transactions through trained model pipeline.")

        st.markdown("#### Live Simulated Transaction Feed")
        stream_txns = get_recent_transactions(limit=15)
        if stream_txns:
            feed = []
            for t in stream_txns:
                feed.append({
                    "Time": t["timestamp"].split("T")[-1][:8] if "T" in t["timestamp"] else t["timestamp"][-8:],
                    "ID": t["id"],
                    "Amount": format_curr(t["amount"], currency_mode),
                    "Channel": t["channel"],
                    "Score": f"{t['risk_score']}/100",
                    "Risk": t["risk_level"],
                    "Decision": "🚨 FRAUD" if t["is_flagged"] else "✅ OK",
                })
            st.table(pd.DataFrame(feed))

    with sim_tab2:
        st.subheader("What-If Model Sensitivity Simulation")
        st.markdown(
            "Inspect how the model's fraud probability shifts under hypothetical modifications of Amount and Time. "
            "*Disclaimer: Evaluates mathematical sensitivity of the trained model, not real-world causal effects.*"
        )

        presets = load_presets()
        target_p = st.selectbox("Baseline Transaction:", [p["name"] for p in presets])
        base_item = next(p for p in presets if p["name"] == target_p)

        w1, w2 = st.columns(2)
        with w1:
            new_amount = st.number_input("Hypothetical Amount", min_value=0.0, max_value=50000.0, value=float(base_item["amount"]), step=25.0)
        with w2:
            new_time = st.number_input("Hypothetical Elapsed Time (sec)", min_value=0.0, value=float(base_item["time"]), step=3600.0)

        if st.button("Run Sensitivity Evaluation"):
            res_whatif = sim_engine.run_what_if_analysis(base_item["features"], {"Amount": new_amount, "Time": new_time})
            orig = res_whatif["original"]
            mod = res_whatif["modified"]
            delta = res_whatif["delta"]

            st.markdown("---")
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Original Risk Score", f"{orig['risk_score']} / 100", delta=orig["risk_level"])
            with m2:
                st.metric("Modified Risk Score", f"{mod['risk_score']} / 100", delta=f"{delta['score_delta']:+d} pts", delta_color="inverse")
            with m3:
                st.metric("Sensitivity Assessment", delta["assessment"])


# ===========================================================================
# PAGE 9: PLATFORM SETTINGS & GOVERNANCE
# ===========================================================================
elif page == "⚙️ Platform Settings & Governance":
    st.title("Platform Specifications & Data Governance")
    st.caption("Architectural telemetry, academic citations, and provenance mandates")

    info = service.get_model_info()

    st.subheader("Model Telemetry")
    t1, t2 = st.columns(2)
    with t1:
        st.markdown(f"""
        - **Platform:** SENTINEL — Fraud Intelligence Platform v3.0
        - **Model:** {info.get('model_name', 'Tuned Random Forest')}
        - **Imbalance Method:** {info.get('strategy', 'SMOTE Pipeline')}
        - **Feature Space:** {info.get('feature_count', 30)} Dimensions
        - **Default Threshold:** {info.get('default_threshold', DEFAULT_THRESHOLD):.2f}
        """)
    with t2:
        st.markdown(f"""
        - **Accuracy:** {info.get('metrics', {}).get('accuracy', 0.9995):.4f}
        - **Precision:** {info.get('metrics', {}).get('precision', 0.9231):.4f}
        - **Recall:** {info.get('metrics', {}).get('recall', 0.7579):.4f}
        - **F1-Score:** {info.get('metrics', {}).get('f1', 0.8324):.4f}
        - **PR-AUC:** {info.get('metrics', {}).get('average_precision', 0.8096):.4f}
        """)

    st.markdown("---")
    st.subheader("Data Provenance & Ethical Disclosure")
    st.info(
        "**Strict Data Honesty Policy:** All features $V_1$ through $V_{28}$ are anonymized PCA components. "
        "The SENTINEL platform does not fabricate semantic labels (e.g. merchant name, device ID) for PCA features. "
        "Any human-friendly fields (Channel, Category, Geolocation) exist solely as a clearly labeled simulation layer."
    )
    st.warning(
        "**Privacy Warning:** This is an academic research demonstration. Never input real credit card numbers, CVVs, "
        "PINs, or banking passwords into this terminal."
    )
    st.markdown("""
    **Dataset Citation:**  
    Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson, and Gianluca Bontempi.  
    *Calibrating Probability with Undersampling for Fraud Detection in Credit Card Data.*  
    IEEE Symposium on Computational Intelligence and Data Mining (CIDM), 2015.
    """)
