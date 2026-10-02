"""
Missing-value corruption.

We blank out `rate` fraction of individual CELLS (not rows), chosen
uniformly at random across every column -- numeric and categorical alike.
This is "missing completely at random" (MCAR), the simplest and most
common assumption for synthetic missingness experiments.
"""

import numpy as np
import pandas as pd


def corrupt(X: pd.DataFrame, y: pd.Series, rate: float, seed: int = 0):
    rng = np.random.default_rng(seed)
    X_dirty = X.copy()
    numeric_cols = list(X_dirty.select_dtypes(include=["number"]).columns)
    X_dirty[numeric_cols] = X_dirty[numeric_cols].astype(float)  # int cols can't hold NaN
    log_rows = []

    n_rows, n_cols = X_dirty.shape
    n_cells = int(n_rows * n_cols * rate)
    flat_positions = rng.choice(n_rows * n_cols, size=n_cells, replace=False)

    for pos in flat_positions:
        row_pos, col_pos = divmod(int(pos), n_cols)
        row_id = X_dirty.index[row_pos]
        col = X_dirty.columns[col_pos]
        original = X_dirty.at[row_id, col]
        X_dirty.at[row_id, col] = np.nan
        log_rows.append({
            "row_id": row_id, "error_type": "missing_values", "column": col,
            "original_value": original, "corrupted_value": np.nan,
        })

    log_df = pd.DataFrame(log_rows)
    return X_dirty, y.copy(), log_df
