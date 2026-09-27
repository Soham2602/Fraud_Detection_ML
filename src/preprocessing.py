"""
preprocessing.py
-----------------
Handles data loading, cleaning, feature scaling, train/test splitting,
and class-imbalance correction for the Fraud Detection project.

IMPORTANT DESIGN PRINCIPLE (avoiding data leakage):
Any transformation that "learns" something from the data
(StandardScaler statistics, SMOTE's synthetic sample generation, etc.)
must be fit ONLY on the training set, never on the full dataset and
never on the test set. This file is structured so that the split
always happens BEFORE scaling and BEFORE resampling.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE


def load_data(path="data/creditcard.csv"):
    """Load the raw dataset from disk."""
    df = pd.read_csv(path)
    return df


def inspect_data(df):
    """Print a quick data-quality report. Used during EDA / sanity checks."""
    report = {
        "shape": df.shape,
        "dtypes": df.dtypes.to_dict(),
        "missing_values": df.isnull().sum().sum(),
        "duplicate_rows": df.duplicated().sum(),
        "class_distribution": df["Class"].value_counts().to_dict(),
        "class_distribution_pct": (df["Class"].value_counts(normalize=True) * 100).to_dict(),
    }
    return report


def clean_data(df):
    """
    Basic cleaning:
    - Drop exact duplicate rows (the Kaggle dataset has ~1000 of these).
    - Confirm there are no missing values (there normally aren't, but we
      check defensively since real-world data pipelines should never assume).
    """
    df = df.copy()

    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    print(f"Removed {before - after} duplicate rows.")

    missing = df.isnull().sum().sum()
    if missing > 0:
        print(f"Found {missing} missing values — dropping affected rows.")
        df = df.dropna()
    else:
        print("No missing values found.")

    return df.reset_index(drop=True)


def split_data(df, target_col="Class", test_size=0.2, random_state=42):
    """
    Split BEFORE any scaling or resampling.
    stratify=y ensures the tiny fraud class is represented proportionally
    in both the train and test sets — critical for imbalanced data.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test


def scale_features(X_train, X_test, columns_to_scale=("Time", "Amount")):
    """
    Scale ONLY 'Time' and 'Amount'.
    The V1..V28 columns are already outputs of a PCA transformation
    (done by the dataset creators) and are already roughly standardized,
    so we leave them untouched.

    The scaler is FIT on X_train only, then applied (transform, not fit)
    to X_test. This is the correct way to avoid data leakage — if we fit
    the scaler on the combined train+test data, information about the
    test set's distribution would "leak" into training.
    """
    X_train = X_train.copy()
    X_test = X_test.copy()

    scaler = StandardScaler()
    cols = [c for c in columns_to_scale if c in X_train.columns]

    X_train[cols] = scaler.fit_transform(X_train[cols])
    X_test[cols] = scaler.transform(X_test[cols])

    return X_train, X_test, scaler


def balance_with_smote(X_train, y_train, random_state=42):
    """
    Apply SMOTE (Synthetic Minority Over-sampling Technique) to the
    TRAINING SET ONLY.

    Why only the training set?
    SMOTE creates synthetic fraud examples by interpolating between real
    fraud examples. If we applied SMOTE before the train/test split, some
    synthetic points in the training set could be near-duplicates of real
    points that ended up in the test set. The model would then be
    evaluated on data suspiciously similar to what it trained on,
    producing an inflated, unrealistic performance score. Applying SMOTE
    strictly after the split, and only to the training fold, avoids this.
    """
    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    return X_resampled, y_resampled


def get_class_weights(y_train):
    """
    Alternative to SMOTE: compute class weights so that the minority
    (fraud) class contributes more heavily to the loss function, without
    creating any synthetic data. Returned as a dict usable directly by
    sklearn's `class_weight` parameter.
    """
    classes = y_train.value_counts()
    total = len(y_train)
    weight_for_0 = total / (2.0 * classes[0])
    weight_for_1 = total / (2.0 * classes[1])
    return {0: weight_for_0, 1: weight_for_1}


if __name__ == "__main__":
    df = load_data()
    print(inspect_data(df))
    df = clean_data(df)
    X_train, X_test, y_train, y_test = split_data(df)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    print("Class weights option:", get_class_weights(y_train))
    X_res, y_res = balance_with_smote(X_train_scaled, y_train)
    print("Before SMOTE:", y_train.value_counts().to_dict())
    print("After SMOTE:", y_res.value_counts().to_dict())
