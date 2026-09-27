# SENTINEL — Presentation Deck & Defense Plan
## 12-Slide Master Presentation with Visual Recommendations and Presenter Notes

---

### Slide 1: Title & Project Identity
- **Slide Title:** SENTINEL — Machine Learning Powered Fraud Intelligence Platform
- **Subtitle:** An End-to-End Operational Risk Scoring and Investigation System
- **Presenter:** [Your Name / Team Members], Department of AIML / ECE
- **Visual Recommendation:** SENTINEL shield icon with dark cyber-fintech dashboard mockup.
- **Presenter Notes:**
  > "Respected examiners, today we present SENTINEL. Rather than a basic script that outputs a binary prediction, SENTINEL is an end-to-end fraud monitoring and investigation platform built with machine learning, explainable AI, calibrated risk scoring, and interactive alert management."

---

### Slide 2: The Core Problem
- **Slide Title:** The Needle in the Haystack: Transaction Fraud
- **Key Points:**
  - Modern payment rails process hundreds of millions of transactions every day.
  - Fraud constitutes a minute fraction (~0.17%), creating severe class imbalance.
  - Asymmetry in financial risk: Missing a fraud costs thousands in direct losses and regulatory penalties, while false alarms create customer friction.
- **Visual Recommendation:** Visual comparison diagram of 99.83% legitimate transactions vs. 0.17% fraud cases.
- **Presenter Notes:**
  > "The central problem in financial transaction classification is extreme class imbalance. In our dataset of over 284,000 transactions, only 492 are fraudulent. This needle-in-a-haystack property makes conventional accuracy completely deceptive."

---

### Slide 3: Motivation & The Accuracy Paradox
- **Slide Title:** Why Plain Accuracy is Misleading
- **Key Points:**
  - A trivial 'dummy' model predicting legitimate 100% of the time scores **99.83% accuracy**.
  - Yet it catches **zero fraud** ($0\% \text{ recall}$), rendering it completely useless in production.
  - Evaluation must prioritize **Recall, Precision, F1-Score, and Precision-Recall AUC (PR-AUC)**.
- **Visual Recommendation:** Formula callout showing:
  $$\text{Accuracy} = \frac{284,315}{284,807} \approx 99.83\% \quad (\text{Caught: } 0/492)$$
- **Presenter Notes:**
  > "If a student or engineer claims 99% accuracy on this dataset, they may have accomplished nothing. A model that approves every stolen credit card gets 99.8% accuracy. This paradox drove every methodological decision in SENTINEL."

---

### Slide 4: Dataset Anatomy & Data Honesty
- **Slide Title:** Dataset Properties & Technical Honesty Mandate
- **Key Points:**
  - European cardholders transaction dataset (284,807 transactions, September 2013).
  - 30 input features: `Time`, `Amount`, and `V1`–`V28`.
  - `V1`–`V28` are anonymized PCA projections for confidentiality.
  - **Data Honesty Rule:** SENTINEL does not invent fake merchant names or locations in the core dataset; human-friendly fields exist strictly as a labeled simulation layer.
- **Visual Recommendation:** Feature schema diagram highlighting PCA components vs. raw scalar features.
- **Presenter Notes:**
  > "We maintain strict technical honesty. Because the researchers anonymized the features via Principal Component Analysis, features V1 through V28 do not represent merchant names or locations. Any demo metadata is strictly separated into a simulation layer."

---

### Slide 5: Data Leakage Prevention Methodology
- **Slide Title:** Engineering a Zero-Leakage Pipeline
- **Key Points:**
  - **Leakage Risk 1:** Fitting StandardScaler on the full dataset leaks test set statistics into training.
  - **Leakage Risk 2:** Oversampling with SMOTE before splitting creates synthetic clones in both train and test partitions.
  - **SENTINEL Solution:** Stratified 80/20 train/test split occurs first; scaler is fitted only on training data; SMOTE is embedded inside an `imblearn.pipeline.Pipeline` during cross-validation.
- **Visual Recommendation:** Process flow diagram showing Split $\to$ Scaling (Train-only) $\to$ Resampling inside CV $\to$ Model Fitting $\to$ Held-Out Test Evaluation.
- **Presenter Notes:**
  > "Data leakage is the most common flaw in college ML projects. In SENTINEL, we mathematically guarantee zero leakage: the test split is sealed before scaling or oversampling. During cross-validation, SMOTE is applied strictly inside each training fold."

---

### Slide 6: Model Comparison & Empirical Benchmarks
- **Slide Title:** Empirical Benchmarking Across 6 Configurations
- **Key Points:**
  - Evaluated Logistic Regression, Decision Tree, and Random Forest.
  - Tested Cost-Sensitive Class Weighting vs. SMOTE Oversampling.
  - **Measured Results on Test Split:**
    - Logistic Regression [SMOTE]: Recall 87.4%, Precision 5.3%, F1 0.100.
    - Decision Tree [SMOTE]: Recall 78.9%, Precision 9.2%, F1 0.164.
    - Random Forest [Class-Weight]: Recall 72.6%, Precision 94.5%, F1 0.821.
    - **Tuned Random Forest [SMOTE Pipeline]:** Recall 75.8%, Precision 92.3%, **F1 0.832**, **PR-AUC 0.810**.
- **Visual Recommendation:** Side-by-side bar chart of F1-Score and PR-AUC comparing all six models.
- **Presenter Notes:**
  > "Here are our actual measured experimental results. Logistic Regression caught fraud but generated nearly 1,500 false alarms. Decision Trees suffered from split variance. Tuned Random Forest with SMOTE achieved the highest F1 score of 0.832 and PR-AUC of 0.810."

---

### Slide 7: Threshold Optimization Playground
- **Slide Title:** Dynamic Threshold Optimization & Asymmetric Loss
- **Key Points:**
  - Standard ML models use an arbitrary 0.50 cutoff.
  - In banking, risk appetite dictates threshold tuning:
    - $\theta = 0.10$: Recall climbs to **82.1%** (catches more fraud, more analyst review required).
    - $\theta = 0.70$: Precision rises to **94.4%** (minimal false alarms, VIP customer lanes).
  - SENTINEL provides an interactive threshold slider with real-time FP/FN trade-off visualization.
- **Visual Recommendation:** Plot showing Precision-Recall trade-off curve with movable threshold cursor.
- **Presenter Notes:**
  > "A bank does not operate on a fixed 0.5 threshold. In high-risk windows, they lower the threshold to catch stealthy fraud. In frictionless checkout lanes, they raise it. SENTINEL gives risk teams an interactive threshold playground."

---

### Slide 8: Explainable AI (SHAP TreeExplainer)
- **Slide Title:** Explainable AI: Why Was This Flagged?
- **Key Points:**
  - High-accuracy black-box models are unacceptable in regulated finance.
  - SENTINEL integrates SHAP (SHapley Additive exPlanations) based on cooperative game theory.
  - Decomposes each prediction into positive (fraud-pushing) and negative (legit-pushing) forces.
  - Generates clear, technically honest attribution narratives.
- **Visual Recommendation:** Horizontal SHAP waterfall force bar chart showing $V_{14}, V_{12}, V_{10}$ pushing risk upward.
- **Presenter Notes:**
  > "When a transaction is declined, the analyst must know why. Using SHAP TreeExplainer, SENTINEL computes the exact contribution of each PCA feature. Notice that component V14 and V12 are the strongest mathematical drivers of fraud risk."

---

### Slide 9: System Architecture & SQLite Vault
- **Slide Title:** Robust Software Architecture
- **Key Points:**
  - **Inference Core (`model_service.py`):** Unified validation, feature scaling, and 0–100 risk scoring.
  - **Persistence Layer (`database.py`):** SQLite database tracking transactions, alerts, and audit logs.
  - **Analyst Workflow:** Actions include 'Mark Reviewed', 'Clear Legitimate', 'Escalate', and 'Add Note'.
  - **Serverless Resilience:** Automatic `/tmp` database path fallback for Vercel cloud execution.
- **Visual Recommendation:** Architecture block diagram linking Streamlit, FastAPI, ModelService, SQLite, and SHAP.
- **Presenter Notes:**
  > "Under the hood, SENTINEL follows clean software engineering principles. Predictions are persisted to an SQLite database with complete audit logging. Analysts can review, clear, or escalate flagged transactions."

---

### Slide 10: Live Stream Simulation & What-If Lab
- **Slide Title:** Live Simulation & What-If Sensitivity Lab
- **Key Points:**
  - **Live Stream Generator:** Simulates continuous payment traffic with burst-attack injections.
  - **What-If Sensitivity:** Allows analysts to alter transaction amount or time and observe the mathematical risk delta.
  - Explicit educational disclaimer prevents confusing model sensitivity with real-world causality.
- **Visual Recommendation:** Screenshot of the live transaction feed ticker and What-If delta cards.
- **Presenter Notes:**
  > "To demonstrate how the system responds in real time, SENTINEL features a live streaming simulator that generates realistic payment streams and a What-If sensitivity lab where we can simulate how changing transaction amounts shifts the model's score."

---

### Slide 11: Deployment & Automated Testing
- **Slide Title:** Deployment Architecture & Test Engineering
- **Key Points:**
  - **30 Pytest Unit Tests:** 100% test coverage across validation, inference, database, explainability, simulation, and API endpoints.
  - **Dual Frontend:**
    1. Multi-page Streamlit Operations Center (`app.py`).
    2. Headless FastAPI REST backend with Vercel Serverless entrypoint (`api/index.py`).
  - Model compressed by 71% (13.4 MB) to ensure fast cold starts and serverless compliance.
- **Visual Recommendation:** Terminal screenshot showing 30 passing tests and Vercel deployment badge.
- **Presenter Notes:**
  > "Code quality is proven through 30 automated unit tests. The application is packaged with FastAPI and configured for zero-config Vercel serverless deployment, while also offering the full multi-page Streamlit desktop experience."

---

### Slide 12: Conclusion & Viva Readiness
- **Slide Title:** Summary & Key Contributions
- **Key Points:**
  - Developed a complete fraud intelligence platform adhering to academic rigor and data honesty.
  - Tuned Random Forest achieved **0.8324 F1-Score** and **0.8096 PR-AUC** with zero data leakage.
  - Implemented risk scoring, SHAP explainability, threshold analysis, SQLite review workflows, and live simulation.
  - Fully documented with comprehensive project report, viva guide, and reproducible code.
- **Visual Recommendation:** Summary bullet matrix and repository QR code / link.
- **Presenter Notes:**
  > "In conclusion, SENTINEL bridges the gap between academic machine learning and real-world software engineering. We are now ready to demonstrate the live platform and answer your questions."
