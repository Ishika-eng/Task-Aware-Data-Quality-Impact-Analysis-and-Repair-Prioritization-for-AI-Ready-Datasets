"""
Registry of corruption functions, so the rest of the pipeline can loop over
all error types generically: CORRUPTORS[name](X, y, rate, seed).
"""

import itertools

from . import missing_values, label_errors, duplicates, outliers, feature_corruption

CORRUPTORS = {
    "missing_values": missing_values.corrupt,
    "label_errors": label_errors.corrupt,
    "duplicates": duplicates.corrupt,
    "outliers": outliers.corrupt,
    "feature_corruption": feature_corruption.corrupt,
}

RATES = [0.05, 0.10, 0.20]

_id_counter = itertools.count(1)


def corrupt_with_id(error_type: str, X, y, rate: float, seed: int = 0):
    """Same as CORRUPTORS[error_type](X, y, rate, seed), but also stamps a
    unique corruption_id onto every row of the returned log.

    Why this exists as a separate wrapper rather than building IDs into
    each corruptor module: phase3_4_test.py originally only added IDs when
    concatenating several runs' logs together, so a single corrupt_fn()
    call used directly (e.g. for oracle repair in Phase 8, which needs to
    invert exactly the cells one specific run corrupted) had no IDs at all.
    Centralizing ID assignment here means every call site gets them
    without duplicating counter logic across 5 corruptor files.
    """
    X_dirty, y_dirty, log_df = CORRUPTORS[error_type](X, y, rate=rate, seed=seed)
    log_df = log_df.copy()
    ids = [f"C{next(_id_counter):06d}" for _ in range(len(log_df))]
    log_df.insert(0, "corruption_id", ids)
    log_df["rate"] = rate
    return X_dirty, y_dirty, log_df
