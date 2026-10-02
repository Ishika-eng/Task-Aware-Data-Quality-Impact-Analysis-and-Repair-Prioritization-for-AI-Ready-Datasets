"""
Outlier corruption.

Picks `rate` fraction of rows, and for each one pushes a randomly chosen
NUMERIC column's value far outside its normal range (by `magnitude`
standard deviations, in a random direction). Simulates things like sensor
glitches or unit-conversion bugs (e.g. age entered as 280 instead of 28).
"""

import numpy as np
import pandas as pd


def corrupt(X: pd.DataFrame, y: pd.Series, rate: float, seed: int = 0, magnitude: float = 6.0):
    rng = np.random.default_rng(seed)
    X_dirty = X.copy()
    numeric_cols = list(X_dirty.select_dtypes(include=["number"]).columns)
    X_dirty[numeric_cols] = X_dirty[numeric_cols].astype(float)  # int cols can't hold float outlier values
    stds = X_dirty[numeric_cols].std()

    n_rows = int(len(X_dirty) * rate)
    rows = rng.choice(X_dirty.index, size=n_rows, replace=False)

    log_rows = []
    for row_id in rows:
        col = rng.choice(numeric_cols)
        sign = rng.choice([-1, 1])
        original = X_dirty.at[row_id, col]
        corrupted = original + sign * magnitude * stds[col]
        X_dirty.at[row_id, col] = corrupted
        log_rows.append({
            "row_id": row_id, "error_type": "outliers", "column": col,
            "original_value": original, "corrupted_value": corrupted,
        })

    log_df = pd.DataFrame(log_rows)
    return X_dirty, y.copy(), log_df
