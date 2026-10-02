"""
Feature corruption.

Corrupts `rate` fraction of individual CELLS (not whole columns -- see
below for why), by swapping each selected cell's value with another row's
value from the SAME column. Each individual value stays a plausible value
for that column (e.g. a real occupation that exists in the dataset), but
it no longer belongs to the right row, breaking that row's internal
consistency. Simulates a broken ETL join or a misaligned pipeline step.

Why cell-level, not column-level: Adult only has 14 columns, so a
"shuffle rate% of columns" design has no resolution at low rates --
int(14 * 0.05) and int(14 * 0.10) both round to "shuffle 1 column",
making 5% and 10% corruption indistinguishable. Operating at the cell
level (like missing_values.py) keeps `rate` meaningfully continuous and
comparable across error types.

This is deliberately the hardest error to detect from the dirty data
alone, since marginal (per-column) statistics look completely normal --
only the joint relationship across columns, for the affected rows, is
broken. That's exactly why Phase 5's detector for this needs a
multivariate (PCA-based) approach rather than a per-column check.
"""

import numpy as np
import pandas as pd


def corrupt(X: pd.DataFrame, y: pd.Series, rate: float, seed: int = 0):
    rng = np.random.default_rng(seed)
    X_dirty = X.copy()
    n_rows, n_cols = X_dirty.shape

    n_cells = int(n_rows * n_cols * rate)
    flat_positions = rng.choice(n_rows * n_cols, size=n_cells, replace=False)
    row_positions, col_positions = np.divmod(flat_positions, n_cols)

    log_frames = []
    for col_pos in np.unique(col_positions):
        col = X_dirty.columns[col_pos]
        selected_row_positions = row_positions[col_positions == col_pos]
        if len(selected_row_positions) < 2:
            continue  # need >=2 cells in this column to swap among themselves

        row_ids = X_dirty.index[selected_row_positions]
        original_values = X_dirty.loc[row_ids, col].copy()
        shuffled_values = original_values.sample(frac=1, random_state=seed)
        shuffled_values.index = row_ids  # re-align after the shuffle
        X_dirty.loc[row_ids, col] = shuffled_values

        changed = original_values != shuffled_values
        log_frames.append(pd.DataFrame({
            "row_id": row_ids[changed.values],
            "error_type": "feature_corruption",
            "column": col,
            "original_value": original_values[changed].values,
            "corrupted_value": shuffled_values[changed].values,
        }))

    log_df = pd.concat(log_frames, ignore_index=True) if log_frames else pd.DataFrame(
        columns=["row_id", "error_type", "column", "original_value", "corrupted_value"])
    return X_dirty, y.copy(), log_df
