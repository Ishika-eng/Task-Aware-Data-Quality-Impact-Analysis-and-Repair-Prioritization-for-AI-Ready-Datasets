"""
Controlled data-quality error injection.

Why inject errors ourselves instead of just using a "naturally dirty" dataset?
Because to measure the *impact* of, say, label noise, we need a clean baseline
and a version that differs ONLY in label noise. Injecting errors with a known
rate (e.g. "5% of labels flipped") is what makes the impact measurement a
controlled experiment instead of a guess.

Every function here takes (X, y) as pandas/numpy objects and a `rate` in
[0, 1] controlling how much damage to do, and returns a corrupted copy.
Nothing is mutated in place, so the caller always keeps the clean original.
"""

import numpy as np
import pandas as pd


def inject_duplicates(X: pd.DataFrame, y: pd.Series, rate: float, seed: int = 0):
    """Duplicate a fraction of rows (appended, not replacing anything).

    Effect on an ML model: usually small. Duplicates bias the training
    distribution slightly toward whatever rows got copied, but they don't
    introduce wrong information.
    """
    rng = np.random.default_rng(seed)
    n_dupes = int(len(X) * rate)
    idx = rng.choice(X.index, size=n_dupes, replace=True)
    X_dirty = pd.concat([X, X.loc[idx]], ignore_index=True)
    y_dirty = pd.concat([y, y.loc[idx]], ignore_index=True)
    return X_dirty, y_dirty


def inject_outliers(X: pd.DataFrame, y: pd.Series, rate: float, seed: int = 0,
                     magnitude: float = 6.0):
    """Push a fraction of rows' feature values far outside their normal range.

    `magnitude` is in standard deviations. We pick random rows and random
    numeric columns within those rows and shift the value by
    magnitude * std, which is what you'd see from e.g. a sensor glitch or a
    unit-conversion bug.
    """
    rng = np.random.default_rng(seed)
    X_dirty = X.copy()
    n_rows = int(len(X) * rate)
    rows = rng.choice(X.index, size=n_rows, replace=False)
    stds = X.std(numeric_only=True)
    for r in rows:
        col = rng.choice(X.columns)
        sign = rng.choice([-1, 1])
        X_dirty.loc[r, col] = X_dirty.loc[r, col] + sign * magnitude * stds[col]
    return X_dirty, y.copy()


def inject_missing_values(X: pd.DataFrame, y: pd.Series, rate: float, seed: int = 0):
    """Blank out a fraction of individual cells (MCAR: missing completely at random).

    We leave NaNs in place rather than imputing here — imputation is a
    *repair* strategy, which belongs in the cleaning stage, not the error
    stage.
    """
    rng = np.random.default_rng(seed)
    X_dirty = X.copy()
    mask = rng.random(X_dirty.shape) < rate
    X_dirty = X_dirty.mask(mask)
    return X_dirty, y.copy()


def inject_label_noise(X: pd.DataFrame, y: pd.Series, rate: float, seed: int = 0):
    """Flip a fraction of labels to a different, uniformly-chosen class.

    This simulates mislabeled data (annotation errors), typically the most
    damaging error type because the model is directly taught the wrong
    answer for that example, not just given noisy inputs.
    """
    rng = np.random.default_rng(seed)
    y_dirty = y.copy()
    classes = y.unique()
    n_flip = int(len(y) * rate)
    idx = rng.choice(y.index, size=n_flip, replace=False)
    for i in idx:
        other_classes = [c for c in classes if c != y_dirty.loc[i]]
        y_dirty.loc[i] = rng.choice(other_classes)
    return X.copy(), y_dirty


def inject_feature_corruption(X: pd.DataFrame, y: pd.Series, rate: float, seed: int = 0):
    """Shuffle the values of a fraction of columns across rows.

    This simulates a corrupted feature pipeline (e.g. a join misalignment
    or a broken ETL step) where an entire feature's values no longer
    correspond to the right rows, destroying its relationship with the
    label. This is typically the most damaging error because it can wreck
    a feature the model relies on heavily, across every row that feature
    touches.
    """
    rng = np.random.default_rng(seed)
    X_dirty = X.copy()
    n_cols = max(1, int(len(X.columns) * rate))
    cols = rng.choice(X.columns, size=n_cols, replace=False)
    for col in cols:
        X_dirty[col] = X_dirty[col].sample(frac=1, random_state=seed).values
    return X_dirty, y.copy()


# Registry so the pipeline can loop over all error types generically.
ERROR_INJECTORS = {
    "duplicates": inject_duplicates,
    "outliers": inject_outliers,
    "missing_values": inject_missing_values,
    "label_noise": inject_label_noise,
    "feature_corruption": inject_feature_corruption,
}
