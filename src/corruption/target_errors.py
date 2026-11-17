"""
Target-error corruption for REGRESSION (Phase 21, docs/PHASE21_SPEC.md).

The regression counterpart of label_errors: a numeric target that is wrong for some rows (a unit slip, a keying error, a
sensor fault). For the chosen rows the target is shifted by s * sigma_y with s ~ U(1, 3) and a random sign, where sigma_y is
the std of the (clean) target. Same return shape as every other corruptor, so it plugs into the generic loops.
"""

import numpy as np
import pandas as pd

SHIFT_RANGE = (1.0, 3.0)         # fixed in the spec, in units of the target's standard deviation


def corrupt(X: pd.DataFrame, y: pd.Series, rate: float, seed: int = 0):
    rng = np.random.default_rng(seed)
    yv = pd.to_numeric(y, errors="coerce").astype(float)
    sigma = float(yv.std(ddof=0))
    if not np.isfinite(sigma) or sigma == 0:
        raise ValueError("target_errors.corrupt needs a numeric target that varies")
    y_dirty = yv.copy()
    n = int(len(y) * rate)
    idx = rng.choice(y.index, size=n, replace=False)
    shifts = rng.uniform(*SHIFT_RANGE, size=n) * rng.choice([-1.0, 1.0], size=n) * sigma
    y_dirty.loc[idx] = yv.loc[idx].values + shifts
    log_df = pd.DataFrame({"row_id": idx, "error_type": "label_errors", "column": y.name if y.name is not None else "target",
                           "original_value": yv.loc[idx].values, "corrupted_value": y_dirty.loc[idx].values})
    return X.copy(), y_dirty, log_df
