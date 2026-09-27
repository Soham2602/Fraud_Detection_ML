# Credit Card Fraud Detection — B.Tech ML Mini-Project

A complete, end-to-end machine learning system that classifies credit card
transactions as **legitimate (0)** or **fraudulent (1)**, built on the
Kaggle Credit Card Fraud Detection dataset.

## What's inside

```text
Fraud_Detection_ML/
│
├── data/
│   └── README.md              ← where to download creditcard.csv
│
├── notebooks/
│   └── fraud_detection.ipynb  ← full EDA + modeling walkthrough
│
├── src/
│   ├── preprocessing.py       ← cleaning, scaling, SMOTE, class weights
│   ├── train.py                ← trains + compares models, tunes, saves
│   ├── evaluate.py             ← metrics, confusion matrix, ROC curve
│   └── predict.py              ← load model and predict new transactions
│
├── models/                     ← trained model + scaler saved here (.pkl)
│
├── app.py                      ← Streamlit web demo
│
├── requirements.txt
├── README.md                   ← you are here
└── PROJECT_REPORT.md           ← full write-up, metric explanations, viva Q&A
```

## 1. Setup

```bash
# from inside Fraud_Detection_ML/
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Get the dataset

Follow the instructions in `data/README.md` to download `creditcard.csv`
from Kaggle and place it at `data/creditcard.csv`.

## 3. Explore the data (optional but recommended)

```bash
jupyter notebook notebooks/fraud_detection.ipynb
```

This notebook walks through data inspection, EDA charts, and a first pass
at modeling — useful for understanding the project before presenting it.

## 4. Train the models

```bash
cd src
python train.py
```

This will:
- Load and clean the data
- Split into train/test (stratified, before any scaling/resampling)
- Scale `Time` and `Amount`
- Train Logistic Regression, Decision Tree, and Random Forest under
  **both** class-weighting and SMOTE strategies
- Print a full metrics report for every model/strategy combination
- Run a small `RandomizedSearchCV` hyperparameter search on Random Forest
- Save the best model, the fitted scaler, and the feature column order to
  `models/`
- Save confusion matrix and ROC curve plots to `figures/`

## 5. Try a quick prediction from the command line

```bash
python src/predict.py
```

Loads 5 random transactions from the dataset and shows the model's
prediction vs. the actual label for each.

## 6. Run the web demo

```bash
streamlit run app.py
```

Opens a browser tab where you can:
- Fill in a transaction manually (or auto-fill from a random real sample)
  and get an instant prediction
- Upload a CSV of transactions and download predictions for all of them

## Notes on methodology

- **No data leakage**: scaling and SMOTE are fit only on the training
  split, never on the test set.
- **Accuracy is not the headline metric** — with fraud making up ~0.17%
  of transactions, a model that predicts "legitimate" every single time
  would still be >99.8% accurate while catching zero fraud. Precision,
  Recall, F1, and ROC-AUC are used instead. See `PROJECT_REPORT.md` for
  a full explanation.

## For presentations / viva

See `PROJECT_REPORT.md` — it includes a plain-language explanation of
every step suitable for a slide deck, plus a set of likely viva questions
with answers.
