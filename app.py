"""
app.py
------
SENTINEL — Fraud Intelligence Platform
Machine Learning powered transaction risk analysis

A multi-page fraud monitoring and investigation platform designed for
academic rigor, technical honesty, and professional cybersecurity UX.
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
import seaborn as sns

# Ensure src is on path
SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import (
    CREDITCARD_CSV,
    SAMPLE_PRESETS_JSON,
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
    page_title="SENTINEL — Fraud Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom High-End Fintech / Cyber Styling
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
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.8) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 18px 20px;
        box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.25);
        transition: transform 0.15s ease, border-color 0.15s ease;
    }
    .metric-card:hover {
        border-color: rgba(59, 130, 246, 0.4);
        transform: translateY(-2px);
    }
    .metric-title {
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 1.85rem;
        font-weight: 700;
        color: #f8fafc;
        line-height: 1.2;
    }
    .metric-sub {
        font-size: 0.78rem;
        color: #64748b;
        margin-top: 4px;
    }

    /* Badges */
    .badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.02em;
    }
    .badge-low { background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3); }
    .badge-medium { background: rgba(245, 158, 11, 0.15); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.3); }
    .badge-high { background: rgba(249, 115, 22, 0.15); color: #fb923c; border: 1px solid rgba(249, 115, 22, 0.3); }
    .badge-critical { background: rgba(239, 68, 68, 0.15); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.3); }

    /* Risk Score Gauge */
    .risk-banner {
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }

    /* Sidebar Branding */
    .sidebar-brand {
        padding: 10px 0 20px 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        margin-bottom: 15px;
    }
    .sidebar-brand h2 {
        font-size: 1.25rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: 0.05em;
        background: linear-gradient(90deg, #60a5fa, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sidebar-brand p {
        font-size: 0.72rem;
        color: #94a3b8;
        margin: 2px 0 0 0;
    }
</style>
""", unsafe_allow_html=True)


# Initialize Singletons
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

# Ensure DB is ready
init_db()
seed_demo_database_if_empty()
service = get_service()


# Sidebar Navigation
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <h2>🛡️ SENTINEL</h2>
        <p>Fraud Intelligence Platform</p>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        [
            "📊 Overview",
            "🔍 Analyze Transaction",
            "📄 Batch Analysis",
            "🚨 Alerts & Investigation",
            "📈 Model Performance",
            "💡 Explainable AI",
            "⚡ Live Simulation & What-If",
            "⚙️ Settings / Model Info",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown(
        f"""
        <div style="font-size: 0.76rem; color: #94a3b8;">
            <b>Model:</b> Tuned Random Forest<br>
            <b>Balancing:</b> SMOTE Pipeline<br>
            <b>Threshold:</b> {DEFAULT_THRESHOLD:.2f}<br>
            <b>Status:</b> <span style="color: #34d399;">● Online</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("v2.0 · B.Tech AIML / ECE Project")


# ===========================================================================
# PAGE 1: OVERVIEW DASHBOARD
# ===========================================================================
if page == "📊 Overview":
    st.title("Fraud Operations Command Center")
    st.caption("Real-time intelligence feed and portfolio risk telemetry")

    kpis = get_dashboard_kpis()

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Total Processed</div>
            <div class="metric-value">{kpis['total_transactions']:,}</div>
            <div class="metric-sub">Logged transactions</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Fraud Flags</div>
            <div class="metric-value" style="color: #f87171;">{kpis['fraud_flags']:,}</div>
            <div class="metric-sub">Model detected</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">High Risk</div>
            <div class="metric-value" style="color: #fb923c;">{kpis['high_risk_transactions']:,}</div>
            <div class="metric-sub">Risk score ≥ 61</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Fraud Rate</div>
            <div class="metric-value">{kpis['fraud_rate_pct']:.2f}%</div>
            <div class="metric-sub">Of total volume</div>
        </div>
        """, unsafe_allow_html=True)
    with c5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Avg Amount</div>
            <div class="metric-value">${kpis['avg_amount']:,.2f}</div>
            <div class="metric-sub">Per transaction</div>
        </div>
        """, unsafe_allow_html=True)
    with c6:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Reviewed</div>
            <div class="metric-value" style="color: #38bdf8;">{kpis['transactions_reviewed']:,}</div>
            <div class="metric-sub">Analyst cleared</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_chart1, col_chart2 = st.columns(2)

    recent_txns = get_recent_transactions(limit=200)
    df_recent = pd.DataFrame(recent_txns) if recent_txns else pd.DataFrame()

    with col_chart1:
        st.subheader("Risk Level Distribution")
        if not df_recent.empty and "risk_level" in df_recent.columns:
            counts = df_recent["risk_level"].value_counts().reindex(["LOW", "MEDIUM", "HIGH", "CRITICAL"]).fillna(0)
            colors = ["#10b981", "#f59e0b", "#f97316", "#ef4444"]

            fig, ax = plt.subplots(figsize=(6, 3.5), facecolor="none")
            bars = ax.bar(counts.index, counts.values, color=colors, width=0.55, edgecolor="none")
            ax.set_facecolor("none")
            ax.tick_params(colors="#94a3b8", labelsize=10)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#334155')
            ax.spines['bottom'].set_color('#334155')
            for bar in bars:
                h = bar.get_height()
                ax.annotate(f"{int(h)}", (bar.get_x() + bar.get_width() / 2, h),
                            ha='center', va='bottom', color='#cbd5e1', fontsize=9, xytext=(0, 2),
                            textcoords='offset points')
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()
        else:
            st.info("No transaction data available yet. Generate simulated transactions or analyze a sample.")

    with col_chart2:
        st.subheader("Transaction Volume by Channel (Demo Layer)")
        if not df_recent.empty and "channel" in df_recent.columns:
            channel_counts = df_recent["channel"].value_counts()
            fig, ax = plt.subplots(figsize=(6, 3.5), facecolor="none")
            ax.pie(
                channel_counts.values,
                labels=channel_counts.index,
                autopct="%1.1f%%",
                colors=["#3b82f6", "#8b5cf6", "#ec4899", "#10b981", "#f59e0b"],
                textprops={'color': '#cbd5e1', 'fontsize': 9},
            )
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()
        else:
            st.info("Channel data will appear as transactions are evaluated.")

    st.markdown("---")
    st.subheader("Recent System Alerts")
    alerts = get_alerts(limit=5)
    if alerts:
        alerts_df = pd.DataFrame(alerts)[["transaction_id", "created_at", "amount", "risk_score", "risk_level", "status"]]
        st.dataframe(alerts_df, use_container_width=True)
    else:
        st.success("No active critical alerts pending review.")


# ===========================================================================
# PAGE 2: ANALYZE TRANSACTION
# ===========================================================================
elif page == "🔍 Analyze Transaction":
    st.title("Transaction Risk Analyzer")
    st.caption("Score incoming transactions with ML-powered multi-vector classification")

    presets = load_presets()

    # Preset Selector Section
    col_preset, col_btn = st.columns([3, 1])
    with col_preset:
        preset_options = ["Custom Input"] + [f"{p['name']} (${p['amount']:.2f})" for p in presets]
        selected_preset_label = st.selectbox(
            "Load Curated Benchmark Preset (Real Kaggle Dataset Records):",
            preset_options,
            help="Select verified real records from Kaggle credit card dataset without loading 150MB CSV.",
        )

    preset_data = None
    if selected_preset_label != "Custom Input":
        idx = preset_options.index(selected_preset_label) - 1
        preset_data = presets[idx]

    # Preload values into session state if preset changed
    if preset_data and st.session_state.get("last_selected_preset") != selected_preset_label:
        st.session_state["in_Amount"] = float(preset_data["amount"])
        st.session_state["in_Time"] = float(preset_data["time"])
        for c in FEATURE_COLUMNS:
            st.session_state[f"in_{c}"] = float(preset_data["features"].get(c, 0.0))
        st.session_state["last_selected_preset"] = selected_preset_label

    # Human-centric Form
    with st.form("analyze_txn_form"):
        st.subheader("1. Transaction Context")
        c1, c2, c3 = st.columns(3)
        with c1:
            amount_val = st.number_input(
                "Transaction Amount ($)",
                min_value=0.0,
                max_value=100000.0,
                value=float(st.session_state.get("in_Amount", 124.50)),
                step=10.0,
            )
        with c2:
            time_val = st.number_input(
                "Elapsed Time (seconds)",
                min_value=0.0,
                value=float(st.session_state.get("in_Time", 42000.0)),
                help="Seconds elapsed since the initial reference timestamp.",
            )
        with c3:
            threshold_val = st.slider(
                "Decision Threshold",
                min_value=0.10,
                max_value=0.90,
                value=float(DEFAULT_THRESHOLD),
                step=0.05,
                help="Model classification decision boundary.",
            )

        st.caption("ℹ️ Demo Simulation Fields (Educational Metadata Layer — Not in raw dataset):")
        d1, d2, d3 = st.columns(3)
        with d1:
            demo_channel = st.selectbox("Transaction Channel", DEMO_CHANNELS, index=0)
        with d2:
            demo_category = st.selectbox("Merchant Category", DEMO_CATEGORIES, index=0)
        with d3:
            demo_location = st.selectbox("Transaction Origin", DEMO_LOCATIONS, index=0)

        # Advanced V1-V28 expander
        with st.expander("🛠️ Advanced Model Features (Anonymized PCA V1–V28)", expanded=False):
            st.caption(
                "V1–V28 represent orthogonal PCA projections created by the dataset owners for confidentiality. "
                "Manual typing is non-intuitive; benchmark presets provide authentic feature distributions."
            )
            v_inputs = {}
            v_cols = st.columns(4)
            for i in range(1, 29):
                col_name = f"V{i}"
                with v_cols[(i - 1) % 4]:
                    v_inputs[col_name] = st.number_input(
                        col_name,
                        value=float(st.session_state.get(f"in_{col_name}", 0.0)),
                        format="%.4f",
                        key=f"v_input_{col_name}",
                    )

        submit_btn = st.form_submit_button("⚡ Evaluate Transaction Risk", use_container_width=True)

    if submit_btn:
        payload = {
            "Amount": amount_val,
            "Time": time_val,
            **v_inputs,
        }

        try:
            result = service.predict_single(payload, threshold=threshold_val)
            result["demo_channel"] = demo_channel
            result["demo_category"] = demo_category
            result["demo_location"] = demo_location

            # Automatically log to SQLite database
            save_transaction(result, is_simulated=False)

            st.markdown("---")
            st.subheader("Intelligence Assessment")

            # Risk Gauge Card
            risk_color = result["risk_color"]
            risk_level = result["risk_level"]
            score = result["risk_score"]
            prob = result["fraud_probability"]
            is_flagged = result["is_flagged"]

            bg_alpha = f"{risk_color}1a"
            border_color = risk_color

            status_banner = f"""
            <div class="risk-banner" style="background: {bg_alpha}; border-color: {border_color};">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <span class="badge badge-{risk_level.lower()}">{risk_level} RISK LEVEL</span>
                        <h2 style="margin: 8px 0 4px 0; color: #f8fafc; font-size: 1.8rem;">
                            Risk Score: {score} <span style="font-size: 1rem; color: #94a3b8;">/ 100</span>
                        </h2>
                        <p style="margin: 0; color: #cbd5e1; font-size: 0.9rem;">
                            Fraud Probability: <b>{prob:.2%}</b> &nbsp;|&nbsp; 
                            Decision Cutoff: <b>{threshold_val:.2f}</b> &nbsp;|&nbsp; 
                            Outcome: <b>{'🚨 FLAGGED FOR REVIEW' if is_flagged else '✅ CLEARED LEGITIMATE'}</b>
                        </p>
                    </div>
                    <div style="text-align: right;">
                        <div style="font-size: 0.78rem; color: #94a3b8;">TXN ID</div>
                        <div class="mono" style="font-size: 0.95rem; font-weight: 600; color: #e2e8f0;">{result['transaction_id']}</div>
                    </div>
                </div>
            </div>
            """
            st.markdown(status_banner, unsafe_allow_html=True)

            # Explainability preview
            xai = get_xai()
            explanation = xai.explain_transaction(payload, top_k=5)

            st.markdown(f"**Attribution Analysis:** {explanation['narrative']}")

            # Horizontal contribution chart
            top_feats = explanation["top_features"]
            if top_feats:
                fig, ax = plt.subplots(figsize=(8, 2.5), facecolor="none")
                f_names = [f['feature'] for f in reversed(top_feats)]
                f_contribs = [f['contribution'] for f in reversed(top_feats)]
                bar_colors = ["#ef4444" if c > 0 else "#10b981" for c in f_contribs]

                ax.barh(f_names, f_contribs, color=bar_colors, height=0.55)
                ax.axvline(0, color="#94a3b8", linestyle="--", alpha=0.7)
                ax.set_facecolor("none")
                ax.tick_params(colors="#94a3b8", labelsize=9)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_color('#334155')
                ax.spines['bottom'].set_color('#334155')
                ax.set_xlabel("Contribution to Fraud Score (SHAP / Tree Attribution)", color="#94a3b8", fontsize=9)
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close()

            st.success(f"Transaction recorded to local audit vault as `{result['transaction_id']}`.")

        except ValidationError as ve:
            st.error(f"Validation Failure: {ve.message}")
            for err in ve.errors:
                st.write(f"- {err}")
        except Exception as e:
            st.error(f"Inference error: {str(e)}")


# ===========================================================================
# PAGE 3: BATCH ANALYSIS
# ===========================================================================
elif page == "📄 Batch Analysis":
    st.title("Bulk Transaction Portfolio Analysis")
    st.caption("Upload batch CSV portfolios to identify anomalous clusters and calculate systemic exposure")

    st.markdown("""
    **Expected Schema:** CSV file containing columns `Time`, `Amount`, and `V1` through `V28`.
    """)

    # Option to download a sample test CSV if user doesn't have one
    col_upload, col_sample = st.columns([3, 1])
    with col_sample:
        presets = load_presets()
        if presets:
            sample_df = pd.DataFrame([p["features"] for p in presets])
            csv_sample_bytes = sample_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download Test Batch CSV",
                data=csv_sample_bytes,
                file_name="sentinel_sample_batch.csv",
                mime="text/csv",
                help="Download a ready-to-test CSV with verified transactions.",
            )

    uploaded_file = st.file_uploader("Upload Batch CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.write(f"Loaded **{len(batch_df):,}** transactions. Initializing validation checks...")

            threshold_batch = st.slider("Classification Threshold for Batch:", 0.10, 0.90, 0.50, 0.05)

            with st.spinner("Processing batch inference through tuned pipeline..."):
                enriched_df, summary = service.predict_batch(batch_df, threshold=threshold_batch)

            st.success("Batch Analysis Complete.")

            # Summary KPIs
            b1, b2, b3, b4, b5 = st.columns(5)
            with b1:
                st.metric("Total Transactions", f"{summary['total_transactions']:,}")
            with b2:
                st.metric("Flagged Fraud", f"{summary['flagged_transactions']:,}", delta=f"{summary['fraud_rate_pct']}% rate")
            with b3:
                st.metric("High / Critical Risk", f"{summary['high_risk_count']:,}")
            with b4:
                st.metric("Average Amount", f"${summary['average_amount']:,.2f}")
            with b5:
                st.metric("At-Risk Volume", f"${summary['total_at_risk_amount']:,.2f}")

            # Filters for table
            st.markdown("---")
            f1, f2 = st.columns([2, 2])
            with f1:
                risk_filter = st.multiselect("Filter by Risk Level", ["LOW", "MEDIUM", "HIGH", "CRITICAL"], default=["HIGH", "CRITICAL"])
            with f2:
                pred_filter = st.selectbox("Filter by Model Decision", ["All", "Only Flagged Fraud (1)", "Only Legitimate (0)"])

            filtered_df = enriched_df.copy()
            if risk_filter:
                filtered_df = filtered_df[filtered_df["Risk_Level"].isin(risk_filter)]
            if pred_filter == "Only Flagged Fraud (1)":
                filtered_df = filtered_df[filtered_df["Predicted_Class"] == 1]
            elif pred_filter == "Only Legitimate (0)":
                filtered_df = filtered_df[filtered_df["Predicted_Class"] == 0]

            st.dataframe(filtered_df.head(100), use_container_width=True)

            # Download CSV
            csv_out = enriched_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Full Analyzed Results (fraud_predictions.csv)",
                data=csv_out,
                file_name="fraud_predictions.csv",
                mime="text/csv",
                use_container_width=True,
            )

        except ValidationError as ve:
            st.error(f"Batch Validation Error: {ve.message}")
            for err in ve.errors:
                st.write(f"- {err}")
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")


# ===========================================================================
# PAGE 4: ALERTS & INVESTIGATION
# ===========================================================================
elif page == "🚨 Alerts & Investigation":
    st.title("Transaction Investigation Vault")
    st.caption("Investigate flagged incidents, review feature attributions, and manage resolution workflows")

    filter_status = st.selectbox("Filter Alerts by Status", ["All", "Open", "Escalated", "Resolved"])
    status_param = None if filter_status == "All" else filter_status

    alerts = get_alerts(status=status_param, limit=100)

    if not alerts:
        st.info("No alerts matching the selected filter.")
    else:
        st.write(f"Displaying **{len(alerts)}** incidents:")
        
        # Display as selectable list
        alert_options = {
            f"[{a['status'].upper()}] {a['transaction_id']} — ${a['amount']:,.2f} (Score: {a['risk_score']})": a["transaction_id"]
            for a in alerts
        }
        selected_label = st.selectbox("Select Incident to Investigate:", list(alert_options.keys()))
        selected_txn_id = alert_options[selected_label]

        txn_details = get_transaction_by_id(selected_txn_id)

        if txn_details:
            st.markdown("---")
            i1, i2, i3, i4 = st.columns(4)
            with i1:
                st.metric("Incident ID", txn_details["id"])
            with i2:
                st.metric("Amount", f"${txn_details['amount']:,.2f}")
            with i3:
                st.metric("Risk Score", f"{txn_details['risk_score']} / 100")
            with i4:
                st.metric("Current Status", txn_details["status"])

            st.markdown("#### Incident Feature Attribution")
            xai = get_xai()
            feat_dict = txn_details.get("features", {})
            if feat_dict:
                explanation = xai.explain_transaction(feat_dict, top_k=6)
                st.info(f"**Analysis Narrative:** {explanation['narrative']}")

                st.write("**Top Influential Model Components:**")
                contrib_df = pd.DataFrame(explanation["top_features"])[["feature", "raw_value", "contribution", "direction"]]
                st.table(contrib_df)

            # Review Workflow Action Form
            st.markdown("#### Analyst Review Decision")
            with st.form("review_form"):
                action_choice = st.radio(
                    "Select Action:",
                    ["Mark as reviewed", "Mark legitimate", "Escalate", "Add note"],
                    horizontal=True,
                )
                reviewer_name = st.text_input("Investigator Name / ID", value="Lead Fraud Analyst")
                analyst_notes = st.text_area("Investigation Notes & Rationale", placeholder="e.g. Cardholder confirmed legitimate travel transaction...")
                
                review_submit = st.form_submit_button("Record Action in Audit Trail")

                if review_submit:
                    success = add_review_action(selected_txn_id, action_choice, reviewer=reviewer_name, notes=analyst_notes)
                    if success:
                        st.success(f"Action '{action_choice}' committed to immutable audit trail.")
                        st.experimental_rerun() if hasattr(st, "experimental_rerun") else st.rerun()

            # Audit History
            reviews = txn_details.get("reviews", [])
            if reviews:
                st.markdown("#### Audit History Log")
                for r in reviews:
                    st.markdown(f"- **{r['reviewed_at']}** — `{r['reviewer']}` performed **{r['action']}**: *{r['notes']}*")


# ===========================================================================
# PAGE 5: MODEL PERFORMANCE & MONITORING
# ===========================================================================
elif page == "📈 Model Performance":
    st.title("Model Performance & Threshold Playground")
    st.caption("Empirical cross-model comparative benchmarks and dynamic threshold sensitivity")

    tab_eval1, tab_eval2, tab_eval3 = st.tabs(["📊 Model Comparison", "🎛️ Threshold Tradeoff Playground", "🔬 Global Feature Importance"])

    with tab_eval1:
        st.subheader("Supervised Classification Benchmarks")
        if EXPERIMENT_RESULTS_CSV.exists():
            exp_df = pd.read_csv(EXPERIMENT_RESULTS_CSV)
            st.dataframe(
                exp_df.style.highlight_max(subset=["f1", "average_precision", "roc_auc", "recall"], color="#1e3a8a"),
                use_container_width=True,
            )
            st.caption(
                "All metrics evaluated strictly on held-out test fold (20% stratified split). "
                "SMOTE applied inside training folds to guarantee zero data leakage."
            )
        else:
            st.info("Run `python src/train.py` to regenerate the full experiment benchmark suite.")

        st.markdown("---")
        st.subheader("Performance Curves (Saved Evaluation Artifacts)")
        p1, p2, p3 = st.columns(3)
        with p1:
            cm_img = FIGURES_DIR / "confusion_matrix_Tuned_Random_Forest_SMOTE.png"
            if cm_img.exists():
                st.image(str(cm_img), caption="Confusion Matrix (Tuned Random Forest)")
        with p2:
            roc_img = FIGURES_DIR / "roc_curve_Tuned_Random_Forest_SMOTE.png"
            if roc_img.exists():
                st.image(str(roc_img), caption="ROC Curve (AUC = 0.966)")
        with p3:
            pr_img = FIGURES_DIR / "pr_curve_Tuned_Random_Forest_SMOTE.png"
            if pr_img.exists():
                st.image(str(pr_img), caption="Precision-Recall Curve (PR-AUC = 0.810)")

    with tab_eval2:
        st.subheader("Interactive Threshold Tradeoff Analyzer")
        st.markdown(
            "Fraud detection systems operate under asymmetric risk costs. Adjust the classification threshold "
            "to inspect how Precision, Recall, False Positives (customer friction), and False Negatives (financial loss) shift."
        )

        thresh_slider = st.slider("Select Fraud Classification Cutoff:", 0.05, 0.95, 0.50, 0.05)

        # Load metadata threshold analysis table
        if METADATA_JSON.exists():
            with open(METADATA_JSON, "r") as f:
                meta = json.load(f)
            t_data = meta.get("threshold_analysis", [])
            if t_data:
                tdf = pd.DataFrame(t_data)
                closest_row = tdf.iloc[(tdf["threshold"] - thresh_slider).abs().argsort()[:1]].iloc[0]

                t1, t2, t3, t4 = st.columns(4)
                with t1:
                    st.metric("Recall (Fraud Caught)", f"{closest_row['recall']:.1%}")
                with t2:
                    st.metric("Precision (Flag Accuracy)", f"{closest_row['precision']:.1%}")
                with t3:
                    st.metric("False Positives (Alarms)", f"{int(closest_row['fp']):,}")
                with t4:
                    st.metric("False Negatives (Missed)", f"{int(closest_row['fn']):,}")

                # Tradeoff explanation
                if thresh_slider < 0.35:
                    st.warning("⚠️ High Sensitivity Mode: Captures maximum fraud (high recall) but increases false alerts and analyst workload.")
                elif thresh_slider > 0.65:
                    st.warning("⚠️ Conservative Mode: Minimizes false alerts (high precision) but risks letting stealthy fraud slip through.")
                else:
                    st.success("Balanced Operational Mode: Optimal harmonic F1 trade-off for standard production fraud queuing.")

                fig, ax = plt.subplots(figsize=(8, 3), facecolor="none")
                ax.plot(tdf["threshold"], tdf["recall"], label="Recall", color="#3b82f6", lw=2)
                ax.plot(tdf["threshold"], tdf["precision"], label="Precision", color="#10b981", lw=2)
                ax.plot(tdf["threshold"], tdf["f1"], label="F1-Score", color="#f59e0b", lw=2, linestyle="--")
                ax.axvline(thresh_slider, color="#ef4444", linestyle=":", label=f"Selected ({thresh_slider:.2f})")
                ax.set_facecolor("none")
                ax.tick_params(colors="#94a3b8")
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_color('#334155')
                ax.spines['bottom'].set_color('#334155')
                ax.set_xlabel("Decision Threshold", color="#94a3b8")
                ax.set_ylabel("Metric Score", color="#94a3b8")
                ax.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="#e2e8f0")
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close()

    with tab_eval3:
        st.subheader("Global Tree Feature Importances (Gini Impurity)")
        importances = service.get_feature_importances()
        if importances:
            imp_df = pd.DataFrame(importances[:15])
            fig, ax = plt.subplots(figsize=(8, 4), facecolor="none")
            ax.barh(imp_df["feature"][::-1], imp_df["importance"][::-1], color="#3b82f6", height=0.6)
            ax.set_facecolor("none")
            ax.tick_params(colors="#94a3b8")
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#334155')
            ax.spines['bottom'].set_color('#334155')
            ax.set_xlabel("Mean Decrease in Impurity (Feature Importance)", color="#94a3b8")
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()
            st.caption("Top 15 features globally. Note that components V14, V12, V10, V17 dominate tree split purity.")


# ===========================================================================
# PAGE 6: EXPLAINABLE AI
# ===========================================================================
elif page == "💡 Explainable AI":
    st.title("Explainable AI (XAI) Attribution Lab")
    st.caption("Local feature contributions via SHAP TreeExplainer and technical attribution narratives")

    presets = load_presets()
    selected_name = st.selectbox(
        "Select Sample Transaction to Explain:",
        [p["name"] for p in presets],
        help="Select a benchmark transaction to decompose into component-level SHAP attributions.",
    )

    chosen_preset = next(p for p in presets if p["name"] == selected_name)
    sample_feat = chosen_preset["features"]

    xai = get_xai()
    with st.spinner("Computing local SHAP explanations..."):
        xai_res = xai.explain_transaction(sample_feat, top_k=10)

    st.markdown("---")
    st.subheader(f"Explanation Breakdown — {chosen_preset['name']}")
    st.info(f"**Methodology:** {xai_res['method']}\n\n**Attribution Narrative:** {xai_res['narrative']}")

    col_pos, col_neg = st.columns(2)
    with col_pos:
        st.markdown("#### 🚨 Features Increasing Fraud Risk (+)")
        if xai_res["fraud_drivers"]:
            pos_df = pd.DataFrame(xai_res["fraud_drivers"])[["feature", "raw_value", "contribution"]]
            st.dataframe(pos_df, use_container_width=True)
        else:
            st.write("None. All prominent features contributed towards a legitimate classification.")

    with col_neg:
        st.markdown("#### ✅ Features Pushing Legitimate (-)")
        if xai_res["legit_drivers"]:
            neg_df = pd.DataFrame(xai_res["legit_drivers"])[["feature", "raw_value", "contribution"]]
            st.dataframe(neg_df, use_container_width=True)
        else:
            st.write("None. All prominent features pushed towards fraud.")

    # Attribution Plot
    st.markdown("---")
    st.subheader("Local Feature Force Breakdown")
    fig, ax = plt.subplots(figsize=(9, 4), facecolor="none")
    top_10 = xai_res["top_features"]
    f_labels = [f['feature'] for f in reversed(top_10)]
    f_vals = [f['contribution'] for f in reversed(top_10)]
    colors = ["#ef4444" if v > 0 else "#10b981" for v in f_vals]

    ax.barh(f_labels, f_vals, color=colors, height=0.55)
    ax.axvline(0, color="#94a3b8", linestyle="--")
    ax.set_facecolor("none")
    ax.tick_params(colors="#94a3b8")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#334155')
    ax.spines['bottom'].set_color('#334155')
    ax.set_xlabel("SHAP Contribution Value", color="#94a3b8")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()


# ===========================================================================
# PAGE 7: LIVE SIMULATION & WHAT-IF
# ===========================================================================
elif page == "⚡ Live Simulation & What-If":
    st.title("Transaction Simulation & Sensitivity Lab")
    st.caption("Generate synthetic real-time transaction streams and test hypothetical What-If parameter variations")

    tab_sim1, tab_sim2 = st.tabs(["⚡ Live Stream Simulator", "🧪 What-If Sensitivity Lab"])

    sim_engine = get_sim()

    with tab_sim1:
        st.subheader("Synthetic Real-Time Stream")
        st.caption("Simulates real-world payment stream velocity. Strictly synthetic simulation data.")

        s1, s2, s3 = st.columns(3)
        with s1:
            fraud_bias = st.slider("Simulated Attack Frequency", 0.05, 0.50, 0.15, 0.05, help="Probability of injecting fraudulent vector signatures.")
        with s2:
            burst_count = st.selectbox("Batch Generation Size", [1, 5, 10], index=1)
        with s3:
            st.markdown("<br>", unsafe_allow_html=True)
            gen_btn = st.button("🚀 Generate Simulated Stream", use_container_width=True)

        if gen_btn:
            with st.spinner("Generating synthetic transactions..."):
                generated = sim_engine.generate_batch(count=burst_count, fraud_bias=fraud_bias)
            st.success(f"Generated {len(generated)} transactions through pipeline.")

        # Show live stream feed
        st.markdown("#### Live Simulated Transaction Feed")
        stream_txns = get_recent_transactions(limit=15)
        if stream_txns:
            feed_data = []
            for t in stream_txns:
                feed_data.append({
                    "Time": t["timestamp"].split("T")[-1][:8] if "T" in t["timestamp"] else t["timestamp"][-8:],
                    "ID": t["id"],
                    "Channel": t["channel"],
                    "Category": t["category"],
                    "Amount": f"${t['amount']:,.2f}",
                    "Score": f"{t['risk_score']} / 100",
                    "Risk": t["risk_level"],
                    "Decision": "🚨 FRAUD" if t["is_flagged"] else "✅ OK",
                })
            st.table(pd.DataFrame(feed_data))
        else:
            st.info("Click 'Generate Simulated Stream' to trigger transactions.")

    with tab_sim2:
        st.subheader("What-If Model Sensitivity Simulation")
        st.markdown(
            "Test model sensitivity to hypothetical parameter shifts (e.g. escalating transaction amounts or altered time windows). "
            "*Disclaimer: Reflects statistical sensitivity of the trained model, not causal real-world guarantees.*"
        )

        presets = load_presets()
        target_preset = st.selectbox(
            "Select Baseline Transaction for Sensitivity Testing:",
            [p["name"] for p in presets],
            index=0,
        )
        base_item = next(p for p in presets if p["name"] == target_preset)

        w1, w2 = st.columns(2)
        with w1:
            new_amount = st.number_input(
                "Hypothetical Amount ($)",
                min_value=0.0,
                max_value=50000.0,
                value=float(base_item["amount"]),
                step=50.0,
            )
        with w2:
            new_time = st.number_input(
                "Hypothetical Elapsed Time (sec)",
                min_value=0.0,
                value=float(base_item["time"]),
                step=3600.0,
            )

        what_if_run = st.button("Run Sensitivity Evaluation")
        if what_if_run:
            modifications = {"Amount": new_amount, "Time": new_time}
            res_whatif = sim_engine.run_what_if_analysis(base_item["features"], modifications)

            orig = res_whatif["original"]
            mod = res_whatif["modified"]
            delta = res_whatif["delta"]

            st.markdown("---")
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Original Risk Score", f"{orig['risk_score']} / 100", delta=f"{orig['risk_level']}")
            with m2:
                st.metric("Simulated Risk Score", f"{mod['risk_score']} / 100", delta=f"{delta['score_delta']:+d} pts", delta_color="inverse")
            with m3:
                st.metric("Sensitivity Assessment", delta["assessment"])

            st.caption(f"ℹ️ {res_whatif['disclaimer']}")


# ===========================================================================
# PAGE 8: SETTINGS & MODEL INFO
# ===========================================================================
elif page == "⚙️ Settings / Model Info":
    st.title("Platform Specifications & System Health")
    st.caption("Architecture telemetry, academic citations, and data governance policies")

    info = service.get_model_info()

    st.subheader("Model Telemetry")
    t1, t2 = st.columns(2)
    with t1:
        st.markdown(f"""
        - **Platform:** SENTINEL — Fraud Intelligence Platform v2.0
        - **Algorithm:** {info.get('model_name', 'Random Forest Classifier')}
        - **Balancing Strategy:** {info.get('strategy', 'SMOTE')}
        - **Feature Space:** {info.get('feature_count', 30)} Dimensions
        - **Default Cutoff:** {info.get('default_threshold', DEFAULT_THRESHOLD):.2f}
        """)
    with t2:
        st.markdown(f"""
        - **Training Split Date:** {info.get('training_date', '2026-09-27')}
        - **Accuracy:** {info.get('metrics', {}).get('accuracy', 0.9995):.4f}
        - **Precision:** {info.get('metrics', {}).get('precision', 0.9231):.4f}
        - **Recall:** {info.get('metrics', {}).get('recall', 0.7579):.4f}
        - **PR-AUC:** {info.get('metrics', {}).get('average_precision', 0.8096):.4f}
        """)

    st.markdown("---")
    st.subheader("Critical Data Honesty & Privacy Mandate")
    st.warning(
        "**Privacy Warning:** This is an academic intelligence demonstration. Do not input real credit card numbers, "
        "CVV, PIN, or real banking authentication credentials. All synthetic features are processed locally."
    )
    st.info(
        "**Feature Anonymization Disclosure:** The original dataset contains anonymized PCA components (V1–V28). "
        "Human-friendly metadata fields (channel, merchant category, geolocation) are provided as a clearly labeled "
        "simulation/demo layer for UI demonstration and are not present in the underlying Kaggle PCA feature vectors."
    )
