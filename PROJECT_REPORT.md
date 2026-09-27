# Project Report — Credit Card Fraud Detection

## 1. Problem Statement

Financial institutions process millions of transactions per day, and only
a tiny fraction of them are fraudulent. The goal of this project is to
build a machine learning classifier that flags a transaction as fraud (1)
or legitimate (0), using anonymized transaction data.

This is a **binary classification problem with severe class imbalance** —
the single biggest challenge in the project, and the reason most of the
design decisions below exist.

## 2. Dataset

- **Source**: Kaggle "Credit Card Fraud Detection" dataset, collected from
  European cardholders over two days in September 2013.
- **Size**: 284,807 transactions, 31 columns.
- **Features**: `Time`, `Amount`, and `V1`–`V28` (the result of a PCA
  transformation applied by the original researchers to anonymize the raw
  features — this is why the V columns don't have human-readable names).
- **Target**: `Class` — 0 (legitimate) or 1 (fraud).
- **Class balance**: only **492 transactions (~0.17%)** are fraud. This is
  an extremely imbalanced dataset, which drives almost every methodology
  choice in this project.

## 3. Exploratory Data Analysis — what each chart tells us

| Chart | What it shows | Why it matters |
|---|---|---|
| Fraud vs. legitimate count (bar chart) | The raw class imbalance | Motivates the entire imbalance-handling section — visually shows fraud is a needle in a haystack |
| Transaction amount distribution (all transactions) | Most transactions are small-value | Helps decide whether `Amount` needs scaling (it does — it's on a very different scale from the PCA components) |
| Amount distribution — fraud only | Fraudulent amounts tend to cluster at lower values with occasional outliers | Suggests `Amount` alone is a weak but non-zero signal |
| Amount distribution — legitimate only | Baseline comparison | Lets us contrast fraud vs. legit amount behavior directly |
| Correlation heatmap | Which V-features correlate with `Class` | Because V-features are PCA outputs, correlations are already decorrelated from each other, but a few (e.g. V17, V14, V12 in the original dataset) show noticeably stronger correlation with fraud |
| Transaction time distribution | Two-day cyclical pattern with dips (likely night hours) | Confirms `Time` reflects a real-world daily cycle rather than random noise, though it turns out to be a weak predictor on its own |
| Boxplot of transaction amounts (fraud vs legit) | Spread and outliers side-by-side | Makes outlier-heavy fraud amounts easy to see at a glance, complements the histograms |

Each of these is generated in `notebooks/fraud_detection.ipynb` with
matplotlib/seaborn, and is intentionally chosen because it either explains
the imbalance problem, justifies a preprocessing decision, or hints at
feature importance — no chart is included just to pad the notebook.

## 4. Preprocessing

- **Duplicates**: the raw dataset has ~1,000 exact duplicate rows; these
  are dropped.
- **Missing values**: none present, but the pipeline checks defensively.
- **Scaling**: only `Time` and `Amount` are scaled with `StandardScaler`.
  The `V1`–`V28` columns are already outputs of PCA (which itself involves
  standardization), so scaling them again is unnecessary.
- **Avoiding data leakage**: the train/test split happens *first*. The
  scaler is `fit()` only on the training set and then only `transform()`-ed
  onto the test set. If we fit the scaler on the full dataset before
  splitting, statistics from the test set (its mean/std) would leak into
  the training process, giving an overly optimistic — and dishonest —
  evaluation.

## 5. Why accuracy is misleading here

With fraud at ~0.17% of transactions, a trivial model that predicts
**"legitimate" for every single transaction** would score:

```
Accuracy = 284,315 correct / 284,807 total ≈ 99.83%
```

That looks excellent on paper but is completely useless — it catches
**zero** fraud, which is the entire point of the project. This is why the
project reports Precision, Recall, F1, and ROC-AUC instead of leading with
accuracy.

## 6. Handling class imbalance

Two approaches are implemented and compared:

**A. Class weights** — no synthetic data is created. Instead, the loss
function is adjusted so misclassifying a fraud case is penalized far more
heavily than misclassifying a legitimate one. Simple, fast, no risk of
generating unrealistic synthetic points, but sometimes less effective than
resampling for tree-based models.

**B. SMOTE (Synthetic Minority Over-sampling Technique)** — generates
synthetic fraud examples by interpolating between existing fraud examples'
feature vectors, until the training set is balanced.

**SMOTE is applied only to the training set, strictly after the
train/test split.** If SMOTE were applied before splitting, some synthetic
training points could be near-duplicates of real points that end up in the
test set, letting the model "see" test-like data during training — an
information leak that would make test performance look artificially
better than it would be on genuinely unseen transactions.

## 7. Models trained

1. **Logistic Regression** — simple, interpretable, fast baseline.
2. **Decision Tree** — captures non-linear splits, easy to explain in a
   viva ("it asks yes/no questions about feature values").
3. **Random Forest** — an ensemble of many decision trees; generally the
   strongest of the three on this kind of tabular data, and the model
   ultimately saved and deployed in the Streamlit app.

Each model is trained under both the class-weight and SMOTE strategies,
giving six model/strategy combinations to compare (see `src/train.py`'s
console output for the actual numbers on your machine, since results
depend on the exact data split).

## 8. Evaluation metrics explained simply

- **Precision** — *"Of all the transactions the model called fraud, how
  many actually were fraud?"* Low precision means too many false alarms
  (legitimate customers getting flagged).
- **Recall** — *"Of all the actual fraud cases, how many did the model
  catch?"* Low recall means real fraud is slipping through undetected.
- **F1-score** — the harmonic mean of precision and recall; a single
  number that balances both, useful when you can't optimize for one
  metric alone.
- **ROC-AUC** — measures how well the model separates the two classes
  across every possible decision threshold, independent of a single
  cutoff choice. 0.5 = random guessing, 1.0 = perfect separation.
- **Confusion Matrix** — the raw counts behind all of the above: true
  negatives (correctly caught legitimate), false positives (legitimate
  flagged as fraud), false negatives (fraud missed), true positives
  (fraud correctly caught).

In fraud detection, **recall is usually prioritized over precision** — a
missed fraud (false negative) typically costs the bank/customer far more
than a false alarm (false positive) that a human reviewer quickly clears.
This project reports all metrics so that trade-off is visible, rather than
optimizing blindly for one number.

## 9. Hyperparameter tuning

A `RandomizedSearchCV` (3-fold cross-validation, F1-scoring) is run over
the Random Forest's `n_estimators`, `max_depth`, `min_samples_split`, and
`min_samples_leaf`. Randomized search (rather than an exhaustive grid
search) is used to keep runtime reasonable — with only 492 fraud examples
in the whole dataset, an exhaustive grid search would take far longer for
a very small expected gain in this project's scope.

## 10. Deployment

- The final chosen model, its fitted scaler, and the expected feature
  column order are saved with `joblib` to `models/`.
- `src/predict.py` provides a reusable function to score new transactions
  or a whole batch.
- `app.py` wraps this in a **Streamlit** web app with two modes: single
  manual-entry prediction, and CSV batch upload with a downloadable
  results file — designed to be demoed live in a viva or presentation.

## 11. Presentation explanation (for slides)

1. **Problem** — detect the ~0.17% of transactions that are fraudulent.
2. **Challenge** — severe class imbalance makes plain accuracy useless.
3. **Approach** — clean data → split → scale (train-only) → compare
   class-weighting vs. SMOTE → train 3 models → evaluate with
   precision/recall/F1/ROC-AUC → tune the best model → deploy it.
4. **Result** — Random Forest + SMOTE typically gives the strongest
   recall/F1 trade-off on this dataset (state your actual run's numbers
   here once `train.py` has been executed on your machine).
5. **Demo** — live Streamlit app, single prediction and batch CSV upload.

## 12. Viva questions and answers

**Q: Why not just use accuracy?**
A: Because the dataset is ~99.8% legitimate transactions — a model that
never predicts fraud still gets ~99.8% accuracy while being useless.
Precision, recall, F1, and ROC-AUC give an honest picture instead.

**Q: What's the difference between precision and recall in this context?**
A: Precision asks "of the transactions we flagged, how many were really
fraud?" Recall asks "of the real fraud cases, how many did we catch?"
There's usually a trade-off between them.

**Q: Why is recall more important than precision for fraud detection?**
A: A missed fraud (false negative) can mean real financial loss, while a
false alarm (false positive) usually just means a transaction gets a
manual review — a much smaller cost.

**Q: What is SMOTE and why apply it only to the training set?**
A: SMOTE generates synthetic minority-class (fraud) samples by
interpolating between real ones. Applying it before the train/test split
risks synthetic training points being near-duplicates of real points that
land in the test set, which would leak information and inflate test
performance dishonestly.

**Q: What is data leakage, and where could it happen in this project?**
A: Data leakage is when information from outside the training data
(especially from the test set) improperly influences model training,
making evaluation look better than it would be in the real world. In this
project, it could happen if the scaler were fit on the full dataset
(instead of train-only) or if SMOTE were applied before splitting.

**Q: Why scale only Time and Amount, not V1–V28?**
A: The V-columns are already the output of a PCA transformation done by
the dataset's creators, which inherently involves standardization. Time
and Amount are the two raw, un-transformed features, so they're on very
different numeric scales and benefit from `StandardScaler`.

**Q: Why use Random Forest as the final model instead of Logistic
Regression?**
A: Random Forest can capture non-linear relationships between features
and typically achieves a stronger recall/F1 balance on this kind of
tabular fraud data than a linear model, at some cost of interpretability.

**Q: How does class weighting work, as an alternative to SMOTE?**
A: It adjusts the model's loss function so that misclassifying the
minority (fraud) class is penalized more heavily than misclassifying the
majority class, without creating any synthetic data points.

**Q: What does ROC-AUC actually measure?**
A: How well the model ranks positive (fraud) instances above negative
(legitimate) ones across all possible classification thresholds — not
just the one threshold (usually 0.5) used to generate the confusion
matrix.

**Q: What would you improve if you had more time/data?**
A: Try gradient-boosted models (XGBoost/LightGBM), engineer time-based
features (e.g. transaction frequency per card in a rolling window), and
validate on more recent/larger transaction data since this dataset is
from 2013 and fraud patterns evolve over time.

## 13. Limitations

- Dataset is from 2013 and may not reflect current fraud patterns.
- V1–V28 being anonymized PCA components limits interpretability — we
  can't explain fraud in terms of real-world features like "merchant
  category" or "location".
- SMOTE synthetic samples are not real fraud cases; they approximate the
  minority class's distribution but could introduce artifacts in some
  situations.
