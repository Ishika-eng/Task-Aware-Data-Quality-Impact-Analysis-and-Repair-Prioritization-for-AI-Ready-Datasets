"""
Label-error corruption.

Adult's target is binary (<=50K / >50K), so "corrupting a label" just means
flipping it to the other class. For multi-class targets (used when TaskClean
calibrates itself on an uploaded dataset) a flipped label becomes a uniformly
random OTHER class. The binary path draws nothing extra from the RNG, so the
Adult results are unchanged.
"""

import numpy as np
import pandas as pd


def corrupt(X: pd.DataFrame, y: pd.Series, rate: float, seed: int = 0):
    rng = np.random.default_rng(seed)
    y_dirty = y.copy()
    classes = sorted(y.unique())
    if len(classes) < 2:
        raise ValueError("label_errors.corrupt needs at least two classes")

    n_flip = int(len(y) * rate)
    idx = rng.choice(y.index, size=n_flip, replace=False)

    log_rows = []
    for row_id in idx:
        original = y_dirty.loc[row_id]
        if len(classes) == 2:
            flipped = classes[0] if original == classes[1] else classes[1]
        else:
            others = [c for c in classes if c != original]
            flipped = others[int(rng.integers(len(others)))]
        y_dirty.loc[row_id] = flipped
        log_rows.append({
            "row_id": row_id, "error_type": "label_errors", "column": "class",
            "original_value": original, "corrupted_value": flipped,
        })

    log_df = pd.DataFrame(log_rows)
    return X.copy(), y_dirty, log_df
