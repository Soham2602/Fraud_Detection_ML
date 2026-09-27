# SENTINEL: Machine Learning Powered Fraud Intelligence Platform
## Comprehensive Academic Project Report

**Degree Programme:** Bachelor of Technology (B.Tech) in Artificial Intelligence & Machine Learning / Electronics & Communication Engineering  
**Project Title:** SENTINEL — An End-to-End Fraud Intelligence, Investigation, and Risk Scoring Platform  
**Dataset:** European Cardholders Transaction Dataset (284,807 transactions, September 2013)  

---

### Abstract

Financial institutions process millions of electronic payment transactions daily, of which only a minute fraction (~0.17%) represent fraudulent activity. Detecting fraudulent transactions presents two fundamental machine learning challenges: extreme class imbalance and asymmetric operational error costs. 

This project designs and implements **SENTINEL**, an end-to-end fraud intelligence platform. Moving beyond isolated prediction scripts, SENTINEL implements:
1. A leak-free data preprocessing and oversampling pipeline (`imblearn.pipeline.Pipeline` with SMOTE);
2. Empirical benchmarking of six distinct model/strategy combinations across Accuracy, Precision, Recall, F1-Score, ROC-AUC, and Precision-Recall AUC (PR-AUC);
3. Calibrated risk scoring (0–100) mapped to four discrete risk tiers;
4. Explainable AI (XAI) using SHAP TreeExplainer for local feature attribution without semantic hallucinations;
5. An interactive threshold optimization playground analyzing false positives vs. false negatives;
6. A local persistence vault (SQLite) supporting analyst review workflows (Mark Reviewed, Cleared Legitimate, Escalated);
7. A dual-interface architecture featuring a multi-page Streamlit command center and a Vercel-ready FastAPI REST service.

Tuned Random Forest trained inside an imbalanced-learn SMOTE pipeline achieved the superior trade-off with an **F1-Score of 0.8324**, **PR-AUC of 0.8096**, **ROC-AUC of 0.9664**, **Precision of 92.31%**, and **Recall of 75.79%** on the held-out test fold.

---

### 1. Problem Statement & Motivation

Card-not-present (CNP) and point-of-sale fraud cause billions of dollars in global annual losses. Modern transaction fraud detection is characterized by:
- **Severe Class Imbalance:** Fraud represents only 492 of 284,807 transactions (0.172%). A naive baseline predicting legitimate for all records achieves 99.83% accuracy while detecting zero fraud.
- **Asymmetric Misclassification Costs:** False Negatives (FN — missed fraud) lead to direct chargebacks, regulatory fines, and reputational erosion. False Positives (FP — false alarms) cause customer friction, transaction abandonment, and operational analyst overload.
- **Explainability Deficits:** Regulators and risk analysts require transparent justification for declined or queued transactions under Fair Lending and consumer protection mandates.
- **Data Anonymization:** Public financial datasets anonymize sensitive cardholder attributes into Principal Component Analysis (PCA) projections, requiring rigorous technical honesty in feature attribution.

---

### 2. Dataset & Exploratory Data Analysis

The dataset comprises credit card transactions made by European cardholders in September 2013 over a two-day period.

#### 2.1 Feature Schema
- **`Time` (Numeric):** Number of seconds elapsed between the transaction and the initial transaction in the dataset (range: 0 to 172,792 seconds, reflecting ~48 hours).
- **`V1` through `V28` (Numeric):** Anonymized numerical features obtained via Principal Component Analysis (PCA) due to privacy and non-disclosure requirements.
- **`Amount` (Numeric):** Transaction amount in Euros (mean: €88.35, maximum: €25,691.16).
- **`Class` (Binary Target):** Ground-truth label where `0` denotes a legitimate transaction and `1` denotes confirmed fraud.

#### 2.2 Exploratory Findings
1. **Class Distribution:** 284,315 legitimate transactions (99.827%) vs. 492 fraudulent transactions (0.173%).
2. **Duplicate Handling:** 1,081 exact duplicate rows were detected in the raw dataset. To prevent synthetic inflation or leakage, these duplicates were defensively dropped, leaving 283,726 unique transactions.
3. **Amount Distribution:** Most transactions are small (median ~€22.00). Fraudulent transactions cluster predominantly at smaller probing amounts (€1 to €100) with occasional high-value spikes, showing that `Amount` alone is insufficient to identify fraud.
4. **Time Cyclicality:** Plotting transactions across elapsed time revealed distinct diurnal activity dips corresponding to nocturnal hours.

---

### 3. Methodology & Zero-Leakage Pipeline

#### 3.1 Data Leakage Prevention
Data leakage occurs when information from outside the training partition improperly influences model fitting, producing artificially inflated test scores that fail in production. SENTINEL guarantees zero leakage by enforcing the following sequence:

```text
Raw Data (283,726 rows)
   │
   ▼
[1] Stratified Train/Test Split (80% Train, 20% Held-Out Test)
   │
   ├──────────────────────────────┬──────────────────────────────┐
   ▼                              ▼                              ▼
Training Fold (226,980 rows)                             Test Fold (56,746 rows)
   │                                                             │
[2] StandardScaler.fit_transform(Time, Amount)            [3] StandardScaler.transform()
   │                                                             │
[4] SMOTE Resampling (Train Only)                                │
   │                                                             │
[5] Classifier Training & CV Pipeline                            │
   │                                                             │
   └──────────────────────────────┬──────────────────────────────┘
                                  ▼
                   [6] Unbiased Model Evaluation
```

#### 3.2 Imbalance Strategies Compared
1. **Cost-Sensitive Class Weighting:** Re-weights the loss function during model optimization inversely proportional to class frequencies:
   $$W_0 = \frac{N}{2 \cdot N_0}, \quad W_1 = \frac{N}{2 \cdot N_1}$$
   Misclassifying a fraud sample is penalized ~290 times more heavily than misclassifying a legitimate sample.
2. **SMOTE (Synthetic Minority Over-sampling Technique):** Synthesizes new fraud examples by selecting $k$-nearest neighbors in minority feature space and interpolating:
   $$\vec{x}_{\text{new}} = \vec{x}_i + \lambda (\vec{x}_{zi} - \vec{x}_i), \quad \lambda \sim U(0, 1)$$
   To avoid data leakage across cross-validation folds, SMOTE was embedded inside an `imblearn.pipeline.Pipeline`.

---

### 4. Empirical Results & Comparative Analysis

All six model/strategy configurations were trained and evaluated on the identical stratified test partition (56,651 legitimate, 95 fraudulent). The resulting empirical metrics:

| Model | Strategy | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC (AP) | Brier Score |
|---|---|---|---|---|---|---|---|---|
| Logistic Regression | Class-Weight | 0.9752 | 0.0562 | **0.8737** | 0.1057 | 0.9658 | 0.6719 | 0.0224 |
| Logistic Regression | SMOTE | 0.9737 | 0.0530 | **0.8737** | 0.1000 | 0.9619 | 0.6769 | 0.0239 |
| Decision Tree | Class-Weight | 0.9953 | 0.2320 | 0.7789 | 0.3575 | 0.8886 | 0.5035 | 0.0044 |
| Decision Tree | SMOTE | 0.9866 | 0.0917 | 0.7895 | 0.1643 | 0.8424 | 0.4147 | 0.0103 |
| Random Forest | Class-Weight | 0.9995 | **0.9452** | 0.7263 | 0.8214 | 0.9391 | 0.8012 | 0.0005 |
| **Tuned Random Forest** | **SMOTE Pipeline** | **0.9995** | **0.9231** | **0.7579** | **0.8324** | **0.9664** | **0.8096** | **0.0005** |

#### 4.1 Evaluation Takeaways
- **Logistic Regression** achieved high recall (87.37%) but unacceptable precision (5.3% to 5.6%), generating nearly 1,400 false alarms on the test set.
- **Decision Trees** showed moderate performance (F1 ~0.16–0.36) but suffered from variance and split greediness.
- **Tuned Random Forest + SMOTE Pipeline** attained the best overall performance with **PR-AUC of 0.8096**, **F1 of 0.8324**, **ROC-AUC of 0.9664**, catching 72 of 95 frauds with only 6 false alarms.

---

### 5. Threshold Optimization & Asymmetric Loss

Standard classifiers apply an arbitrary decision threshold of $0.50$. In fraud risk management, decision thresholds are dynamically tuned to align with institution risk appetite:

$$\hat{y} = \begin{cases} 1 & \text{if } P(\text{Fraud} \mid \vec{x}) \ge \theta \\ 0 & \text{otherwise} \end{cases}$$

Empirical threshold trade-off analysis on the test fold:

| Threshold ($\theta$) | Recall | Precision | F1-Score | False Positives (Friction) | False Negatives (Loss) |
|---|---|---|---|---|---|
| **0.10** | 82.1% | 76.5% | 0.792 | 24 | 17 |
| **0.20** | 78.9% | 83.3% | 0.810 | 15 | 20 |
| **0.30** | 77.9% | 88.1% | 0.827 | 10 | 21 |
| **0.50 (Default)** | 75.8% | 92.3% | **0.832** | 6 | 23 |
| **0.70** | 71.6% | 94.4% | 0.814 | 4 | 27 |
| **0.90** | 58.9% | 98.2% | 0.736 | 1 | 39 |

- Lowering $\theta$ to $0.10$ catches 6 additional fraud cases (recall 82.1%) at the cost of 18 additional false alarms.
- Raising $\theta$ to $0.90$ eliminates nearly all false alarms (1 FP) but misses 39 fraud cases.

---

### 6. Explainable AI (SHAP TreeExplainer)

To comply with technical honesty guidelines, SENTINEL uses SHAP (SHapley Additive exPlanations) based on cooperative game theory:

$$f(x) = \phi_0 + \sum_{i=1}^{M} \phi_i(x)$$

Where $\phi_i(x)$ is the marginal contribution of feature $i$ to the log-odds of fraud.
- **Top Fraud Enablers:** Components $V_{14}$, $V_{12}$, $V_{10}$, and $V_{17}$ exhibit strong negative values on fraudulent vectors, resulting in large positive SHAP forces (+0.25 to +0.45).
- **Academic Transparency:** Because $V_1$ through $V_{28}$ are anonymized PCA components, SENTINEL strictly explains attributions as mathematical component forces rather than inventing fictional semantic attributes.

---

### 7. Software Architecture & Engineering

SENTINEL is built with modular separation of concerns:
- **`src/model_service.py`:** Singleton orchestrator handling validation, scaling, inference, and risk scoring (0–100).
- **`src/database.py`:** SQLite relational layer tracking transactions, alert queues, and analyst actions (`sentinel.db`, with `/tmp` fallback for serverless).
- **`src/simulation.py`:** Synthetic transaction generator with configurable attack rates and What-If sensitivity delta computation.
- **`src/api.py`:** FastAPI REST API exposing `/predict`, `/predict/batch`, `/alerts`, `/reviews`, `/simulate`, and `/health`.
- **`api/index.py` & `vercel.json`:** Serverless entry point enabling deployment to Vercel.
- **`app.py`:** Multi-page Streamlit dashboard featuring custom typography, metric cards, and interactive charts.
- **`tests/`:** 30 unit tests using pytest verifying all endpoints, validations, and services.

---

### 8. Limitations & Future Scope

1. **Dataset Recency:** The Kaggle dataset dates to 2013; modern fraud vectors (SIM swap, token manipulation, crypto on-ramps) exhibit different signatures.
2. **Concept Drift Monitoring:** In production, continuous distribution tracking (e.g., Population Stability Index, Wasserstein distance) should monitor input feature drift.
3. **Graph Neural Networks (GNNs):** Real-world fraud rings operate via shared device IDs, mule accounts, and card clusters; integrating graph-based relational embeddings represents a promising next step.

---

### 9. Conclusion

SENTINEL successfully elevates a classroom ML exercise into a professional, technically honest fraud intelligence platform. By integrating zero-leakage ML pipelines, calibrated risk scoring, SHAP explainability, SQLite persistence, and dual Vercel/Streamlit deployment, the project demonstrates both theoretical AIML rigor and practical software engineering capability.
