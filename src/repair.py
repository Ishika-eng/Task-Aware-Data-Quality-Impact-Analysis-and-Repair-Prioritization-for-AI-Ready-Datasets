"""
Repair strategies, one per error type.

Important honesty check: duplicates, outliers, and missing values can be
repaired algorithmically from the dirty data alone -- that's what real data
cleaning tools do. Label noise and feature corruption generally CANNOT be
repaired that way: fixing a mislabeled row needs a human (or a much more
sophisticated confident-learning system) to say what the *correct* label
is, and fixing a corrupted feature needs someone to trace and rebuild the
broken pipeline. Since this project injects errors into data we already
had clean, we simulate an idealized ("oracle") repair for those two by
referencing the clean ground truth. This models the realistic *best case*
outcome of paying the high effort cost (a domain expert re-annotates, or
an engineer fixes the pipeline) -- it is NOT a claim that these errors are
algorithmically fixable from the dirty data alone.
"""

import pandas as pd


def repair_duplicates(X: pd.DataFrame, y: pd.Series, **_):
    combined = X.copy()
    combined["__label__"] = y.values
    combined = combined.drop_duplicates()
    return combined.drop(columns="__label__").reset_index(drop=True), \
        combined["__label__"].reset_index(drop=True)


def repair_outliers(X: pd.DataFrame, y: pd.Series, lower_q: float = 0.01, upper_q: float = 0.99, **_):
    X_fixed = X.copy()
    for col in X_fixed.select_dtypes("number").columns:
        lo, hi = X_fixed[col].quantile([lower_q, upper_q])
        X_fixed[col] = X_fixed[col].clip(lo, hi)
    return X_fixed, y.copy()


def repair_missing_values(X: pd.DataFrame, y: pd.Series, **_):
    X_fixed = X.copy()
    for col in X_fixed.columns:
        X_fixed[col] = X_fixed[col].fillna(X_fixed[col].median())
    return X_fixed, y.copy()


def repair_label_noise(X: pd.DataFrame, y: pd.Series, clean_y: pd.Series = None, **_):
    """Oracle repair: restore the known-correct labels (see module docstring)."""
    if clean_y is None:
        raise ValueError("repair_label_noise needs clean_y (the ground-truth labels)")
    return X.copy(), clean_y.copy()


def repair_feature_corruption(X: pd.DataFrame, y: pd.Series, clean_X: pd.DataFrame = None, **_):
    """Oracle repair: restore the known-correct feature values (see module docstring)."""
    if clean_X is None:
        raise ValueError("repair_feature_corruption needs clean_X (the ground-truth features)")
    return clean_X.copy(), y.copy()


REPAIR_FUNCTIONS = {
    "duplicates": repair_duplicates,
    "outliers": repair_outliers,
    "missing_values": repair_missing_values,
    "label_noise": repair_label_noise,
    "feature_corruption": repair_feature_corruption,
}
