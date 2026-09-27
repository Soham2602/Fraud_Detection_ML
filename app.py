"""
app.py
------
Streamlit web interface for the Fraud Detection project.

Run with:
    streamlit run app.py

Two modes:
1. Manual entry — type in Time, Amount, and V1..V28 (or use a random
   sample from the dataset as a starting point) and get an instant
   prediction.
2. Batch upload — upload a CSV of transactions and get predictions for
   all of them, downloadable as a results file.
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
import streamlit as st

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from predict import load_artifacts, predict_transaction, predict_batch  # noqa: E402

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "creditcard.csv")

st.set_page_config(page_title="Fraud Detection Demo", page_icon="💳", layout="wide")


@st.cache_resource
def get_artifacts():
    return load_artifacts()


def artifacts_available():
    return os.path.exists(os.path.join(MODELS_DIR, "fraud_detection_model.pkl"))


st.title("💳 Credit Card Fraud Detection")
st.caption(
    "B.Tech Mini-Project demo — a Random Forest model trained on the Kaggle "
    "Credit Card Fraud Detection dataset, tuned with RandomizedSearchCV and "
    "trained on SMOTE-balanced data."
)

if not artifacts_available():
    st.error(
        "No trained model found in `models/`. Run `python src/train.py` "
        "first (after placing `creditcard.csv` in `data/`), then relaunch this app."
    )
    st.stop()

model, scaler, feature_columns = get_artifacts()

tab1, tab2 = st.tabs(["🔍 Single Transaction", "📄 Batch Upload"])

# ---------------------------------------------------------------------------
# TAB 1: Single transaction
# ---------------------------------------------------------------------------
with tab1:
    st.subheader("Check a single transaction")

    col_a, col_b = st.columns([1, 2])

    with col_a:
        if os.path.exists(DATA_PATH):
            if st.button("🎲 Fill with a random sample from the dataset"):
                sample = pd.read_csv(DATA_PATH).sample(1).iloc[0]
                for col in feature_columns:
                    st.session_state[f"in_{col}"] = float(sample[col])
        else:
            st.info("Place `creditcard.csv` in `data/` to enable random sampling.")

    with st.form("single_txn_form"):
        st.write("Enter transaction details (or use the random-sample button above):")

        c1, c2 = st.columns(2)
        with c1:
            time_val = st.number_input(
                "Time (seconds since first transaction in dataset)",
                value=st.session_state.get("in_Time", 0.0),
            )
            amount_val = st.number_input(
                "Amount ($)", value=st.session_state.get("in_Amount", 0.0), min_value=0.0
            )

        with c2:
            st.write("PCA components V1–V28")
            st.caption(
                "These come from a PCA transformation applied by the original "
                "dataset creators to anonymize the real features. Manually "
                "entering meaningful values isn't realistic — use the random "
                "sample button, or expand below to edit them."
            )

        with st.expander("Edit V1–V28 manually"):
            v_values = {}
            cols = st.columns(4)
            for i in range(1, 29):
                col_name = f"V{i}"
                with cols[(i - 1) % 4]:
                    v_values[col_name] = st.number_input(
                        col_name,
                        value=st.session_state.get(f"in_{col_name}", 0.0),
                        key=f"widget_{col_name}",
                        format="%.4f",
                    )

        submitted = st.form_submit_button("Predict")

    if submitted:
        input_dict = {"Time": time_val, "Amount": amount_val, **v_values}
        prediction, probability = predict_transaction(
            input_dict, model, scaler, feature_columns
        )

        if prediction == 1:
            st.error(f"🚨 Predicted: FRAUDULENT transaction (confidence: {probability:.2%})")
        else:
            st.success(f"✅ Predicted: Legitimate transaction (fraud confidence: {probability:.2%})")

# ---------------------------------------------------------------------------
# TAB 2: Batch upload
# ---------------------------------------------------------------------------
with tab2:
    st.subheader("Upload a CSV of transactions")
    st.caption("The file must contain the same columns as the training data: Time, V1–V28, Amount.")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded is not None:
        batch_df = pd.read_csv(uploaded)
        missing_cols = [c for c in feature_columns if c not in batch_df.columns]

        if missing_cols:
            st.error(f"Missing required columns: {missing_cols}")
        else:
            preds, probs = predict_batch(batch_df, model, scaler, feature_columns)
            result_df = batch_df.copy()
            result_df["Predicted_Class"] = preds
            result_df["Fraud_Probability"] = probs

            n_fraud = int(preds.sum())
            st.write(f"**{n_fraud} of {len(preds)}** transactions flagged as fraud.")
            st.dataframe(result_df.head(50))

            csv_out = result_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download full results as CSV",
                data=csv_out,
                file_name="fraud_predictions.csv",
                mime="text/csv",
            )

st.divider()
st.caption(
    "Model: Random Forest (tuned) · Metrics and full methodology in PROJECT_REPORT.md"
)
