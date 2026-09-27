# SENTINEL — Fraud Intelligence Platform
### Machine Learning Powered Transaction Risk Analysis & Investigation Vault

[![Python Version](https://img.shields.io/badge/Python-3.10%20|%203.11%20|%203.12%20|%203.14-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-30%20Passed%20%E2%9C%93-brightgreen.svg)](tests/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688.svg)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)](https://streamlit.io)
[![Vercel Ready](https://img.shields.io/badge/Deploy-Vercel%20Serverless-black.svg)](vercel.json)

---

## 1. Executive Summary & Overview

**SENTINEL** is an end-to-end fraud monitoring and investigation platform powered by machine learning. Designed as a rigorous final-year B.Tech (AIML / ECE) project, it transforms conventional binary classification into a defense-grade fintech intelligence suite.

Rather than a simplistic page that outputs "Fraud or Legit", SENTINEL provides:
- **Calibrated Risk Scoring (0–100)** with dynamic operational risk bands (LOW, MEDIUM, HIGH, CRITICAL).
- **Explainable AI (XAI)** powered by SHAP TreeExplainer decomposing single-transaction predictions into component-level force attributions.
- **Dynamic Threshold Playground** modeling real-world asymmetric fraud costs (precision vs. recall trade-offs).
- **Live Stream Simulation** generating synthetic high-velocity payment traffic with burst-attack injections.
- **What-If Sensitivity Simulator** testing hypothetical parameter shifts without retraining.
- **Analyst Investigation Vault & SQLite Audit Trail** supporting review workflows (Mark Reviewed, Clear Legitimate, Escalate).
- **Dual Deployment Architecture**: Full multi-page Streamlit operations center + headless FastAPI REST backend deployable on **Vercel Serverless**.

---

## 2. System Architecture

```text
                           ┌────────────────────────────────────────┐
                           │      SENTINEL OPERATIONS CENTER        │
                           └──────────────────┬─────────────────────┘
                                              │
                     ┌────────────────────────┴────────────────────────┐
                     ▼                                                 ▼
       ┌───────────────────────────┐                     ┌───────────────────────────┐
       │   Streamlit Multi-Page    │                     │     FastAPI REST API      │
       │   Interactive Dashboard   │                     │  Vercel Serverless / ASGI │
       │         (app.py)          │                     │       (api/index.py)      │
       └─────────────┬─────────────┘                     └─────────────┬─────────────┘
                     │                                                 │
                     └────────────────────────┬────────────────────────┘
                                              ▼
                           ┌─────────────────────────────────────┐
                           │    ModelService & Validation Core   │
                           │       (src/model_service.py)        │
                           └──────────────────┬──────────────────┘
                                              │
         ┌───────────────────┬────────────────┼────────────────────┬───────────────────┐
         ▼                   ▼                ▼                    ▼                   ▼
┌─────────────────┐ ┌─────────────────┐ ┌───────────┐    ┌───────────────────┐ ┌──────────────┐
│  Scikit-Learn   │ │  Imb-Pipeline   │ │   SHAP    │    │  SQLite Database  │ │  Simulation  │
│  Random Forest  │ │  SMOTE + Scaler │ │ Explainer │    │  Vault & Audit    │ │  & What-If   │
│  (Tuned Model)  │ │  (Leak-Free)    │ │   (XAI)   │    │  (sentinel.db)    │ │  Engine      │
└─────────────────┘ └─────────────────┘ └───────────┘    └───────────────────┘ └──────────────┘
```

---

## 3. Empirical Model Performance & Benchmarks

All models were evaluated on the held-out 20% test fold (56,746 legitimate, 95 fraudulent transactions) using strict stratified splitting before scaling or resampling:

| Model | Imbalance Strategy | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC (Avg Prec) |
|---|---|---|---|---|---|---|---|
| Logistic Regression | Class-Weight | 0.9752 | 0.0562 | **0.8737** | 0.1057 | 0.9658 | 0.6719 |
| Logistic Regression | SMOTE | 0.9737 | 0.0530 | **0.8737** | 0.1000 | 0.9619 | 0.6769 |
| Decision Tree | Class-Weight | 0.9953 | 0.2320 | 0.7789 | 0.3575 | 0.8886 | 0.5035 |
| Decision Tree | SMOTE | 0.9866 | 0.0917 | 0.7895 | 0.1643 | 0.8424 | 0.4147 |
| Random Forest | Class-Weight | 0.9995 | **0.9452** | 0.7263 | 0.8214 | 0.9391 | 0.8012 |
| **Tuned Random Forest** | **SMOTE Pipeline** | **0.9995** | **0.9231** | **0.7579** | **0.8324** | **0.9664** | **0.8096** |

> **Key Academic Takeaway:** On severely imbalanced tabular data (0.17% fraud rate), Plain Accuracy (99.95%) is uninformative. **PR-AUC (0.8096)** and **F1-Score (0.8324)** prove that Tuned Random Forest with SMOTE pipeline achieves the optimal trade-off between catching fraud and minimizing customer friction.

---

## 4. Repository Structure

```text
Fraud_Detection_ML/
├── api/
│   └── index.py               ← Vercel serverless entrypoint + web portal
├── data/
│   ├── sample_presets.json    ← Curated real benchmark records (legit, borderline, fraud)
│   ├── creditcard.csv         ← Full Kaggle dataset (150 MB, download via data/README.md)
│   └── sentinel.db            ← SQLite persistence vault (auto-generated)
├── experiments/
│   └── results.csv            ← Measured benchmark results across all 6 models
├── figures/                   ← Confusion matrices, ROC curves, PR curves (.png)
├── models/
│   ├── fraud_detection_model.pkl  ← Tuned Random Forest (compressed, 13.4 MB)
│   ├── scaler.pkl                 ← Fitted StandardScaler
│   ├── feature_columns.pkl        ← Feature schema order
│   └── model_metadata.json        ← Comprehensive training telemetry & hyperparameters
├── notebooks/
│   └── fraud_detection.ipynb  ← Exploratory Data Analysis & modeling walkthrough
├── src/
│   ├── config.py              ← Paths, schemas, risk bands, environment detection
│   ├── validation.py          ← Defensive input sanitization & human-readable errors
│   ├── preprocessing.py       ← Zero-leakage data cleaning, splitting, and scaling
│   ├── evaluate.py            ← PR-AUC, ROC-AUC, threshold analysis, curve plotting
│   ├── train.py               ← Automated cross-validation and pipeline tuning
│   ├── model_service.py       ← Core inference engine & risk scoring (0-100)
│   ├── database.py            ← SQLite persistence, alert queues & review logs
│   ├── explainability.py      ← SHAP TreeExplainer & feature attribution engine
│   ├── simulation.py          ← Live stream generator & What-If sensitivity simulator
│   └── api.py                 ← FastAPI REST backend
├── tests/                     ← Comprehensive test suite (30 pytest unit tests)
│   ├── test_validation.py
│   ├── test_model_service.py
│   ├── test_database.py
│   ├── test_explainability.py
│   ├── test_simulation.py
│   └── test_api.py
├── app.py                     ← SENTINEL Multi-Page Streamlit Operations Center
├── requirements.txt           ← Pinned Python dependencies
├── vercel.json                ← Vercel Serverless configuration
├── pytest.ini                 ← Pytest environment configuration
├── PROJECT_REPORT.md          ← Comprehensive academic project documentation
├── PRESENTATION.md            ← 12-slide presentation deck with speaker notes
└── VIVA.md                    ← 40+ rigorous viva examination questions & answers
```

---

## 5. Quickstart Guide (Local Setup)

### Step 1: Clone & Create Virtual Environment
```bash
git clone https://github.com/Soham2602/Fraud_Detection_ML.git
cd Fraud_Detection_ML

# Create virtual environment
python -m venv venv

# Activate environment
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Run Unit Test Suite
```bash
pytest -v
```
*Expected: 30 passed in ~8s.*

### Step 3: Launch SENTINEL Streamlit Operations Center
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

### Step 4: Run FastAPI REST Backend (Optional)
```bash
uvicorn src.api:app --reload --port 8000
```
Interactive Swagger UI documentation: [http://localhost:8000/docs](http://localhost:8000/docs).

---

## 6. Deployment to Vercel

SENTINEL is pre-configured for **1-click Vercel Serverless Deployment**:

1. Push your repository to GitHub.
2. Sign in to [Vercel](https://vercel.com) and click **"Add New Project"**.
3. Import `Fraud_Detection_ML`.
4. Leave build settings as default (Vercel automatically detects `vercel.json` and Python ASGI runtime).
5. Click **Deploy**.
6. Visit your live deployment URL — it serves the interactive SENTINEL Web Portal connected to the serverless FastAPI backend!

---

## 7. Data Honesty & Academic Governance

- **Anonymized PCA Features:** The dataset features $V_1$ through $V_{28}$ are anonymized principal components. SENTINEL does **not** hallucinate semantic labels (e.g. "suspicious merchant name" or "unusual IP address") for $V_i$. All SHAP explanations explicitly refer to PCA component vectors.
- **Simulation Layer:** Human-friendly fields (Transaction Channel, Merchant Category, Location) are clearly labeled as a **demo simulation layer** and never conflated with original dataset features.
- **Zero Data Leakage:** Train/test splitting strictly precedes feature scaling and SMOTE resampling. Resampling during cross-validation is performed inside an `imblearn.pipeline.Pipeline`.
- **Privacy Notice:** Never enter real credit card numbers, CVVs, or bank credentials.

---

## 8. License & Acknowledgements

- **Dataset:** Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson, and Gianluca Bontempi. *Calibrating Probability with Undersampling for Fraud Detection in Credit Card Data*. IEEE Symposium on Computational Intelligence and Data Mining (CIDM), 2015.
- **License:** Open-source MIT License. Built for B.Tech AIML / ECE Final Mini-Project submission.
