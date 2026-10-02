"""
Label-error corruption.

Adult's target is binary (<=50K / >50K), so "corrupting a label" just means
flipping it to the other class -- no need to pick among multiple classes
like a multi-class dataset would require.
"""

import numpy as np
import pandas as pd


def corrupt(X: pd.DataFrame, y: pd.Series, rate: float, seed: int = 0):
    rng = np.random.default_rng(seed)
    y_dirty = y.copy()
    classes = sorted(y.unique())
    if len(classes) != 2:
        raise ValueError("label_errors.corrupt assumes a binary target")

    n_flip = int(len(y) * rate)
    idx = rng.choice(y.index, size=n_flip, replace=False)

    log_rows = []
    for row_id in idx:
        original = y_dirty.loc[row_id]
        flipped = classes[0] if original == classes[1] else classes[1]
        y_dirty.loc[row_id] = flipped
        log_rows.append({
            "row_id": row_id, "error_type": "label_errors", "column": "class",
            "original_value": original, "corrupted_value": flipped,
        })

    log_df = pd.DataFrame(log_rows)
    return X.copy(), y_dirty, log_df
