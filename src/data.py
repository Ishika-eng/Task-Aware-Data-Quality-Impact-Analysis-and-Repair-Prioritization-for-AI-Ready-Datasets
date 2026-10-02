"""
Dataset loading for the TaskClean project.

Why UCI Adult: it has both categorical and numeric features (so detectors
like inconsistency-detection and encoders actually get exercised, unlike
the earlier all-numeric breast-cancer dataset), a real binary classification
target, and a meaningful class imbalance -- all of which make the impact
analysis more realistic.

Why we drop natively-missing rows here: this project's core experiment
requires a dataset with ZERO known defects as the starting point, so that
when we later inject (say) 10% missing values on purpose, we know the
resulting missingness is exactly 10% and exactly where we put it -- not
confounded with missingness that was already there. Real-world missingness
in Adult (~7%, concentrated in workclass/occupation/native-country) is
removed for this reason, not because it's uninteresting.
"""

import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

TARGET_COL = "class"


def load_adult_clean() -> tuple[pd.DataFrame, pd.Series]:
    """Load UCI Adult, drop natively-missing rows, return (X, y)."""
    data = fetch_openml("adult", version=2, as_frame=True)
    df = data.frame.dropna().reset_index(drop=True)
    y = df[TARGET_COL].astype(str)
    X = df.drop(columns=[TARGET_COL])
    return X, y


def three_way_split(X: pd.DataFrame, y: pd.Series, val_size: float = 0.15,
                     test_size: float = 0.15, seed: int = 42):
    """70/15/15 train/val/test, stratified on the label so the class balance
    (~76/24 for Adult) is preserved in every split."""
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(val_size + test_size), random_state=seed, stratify=y
    )
    relative_test_size = test_size / (val_size + test_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=relative_test_size, random_state=seed, stratify=y_temp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def column_types(X: pd.DataFrame):
    categorical_cols = list(X.select_dtypes(include=["category", "object"]).columns)
    numeric_cols = list(X.select_dtypes(include=["number"]).columns)
    return categorical_cols, numeric_cols
