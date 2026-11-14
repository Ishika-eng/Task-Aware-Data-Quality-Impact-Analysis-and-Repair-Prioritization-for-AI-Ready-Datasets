"""
Ground-truth injectors for the Phase 15 validation (docs/PHASE15_SPEC.md, section C and Amendment 1).

Deliberately NOT registered in corruption.CORRUPTORS (that registry drives the frozen Phase 7/9 experiments and the
self-benchmark). Contract: take (X, y, rate, seed), never mutate the input, return (X_dirty, y, log). The log has one
row per CHANGED CELL.

Three kinds, an equal mix (an ASSUMPTION, not a measurement of real data):
  swap_one    one numeric cell takes the value of the same column in another random row: typical for the column,
              wrong for this row.
  swap_block  two or more numeric cells are replaced together by those of ONE other row: each value is plausible alone and
              the block is internally consistent, but it does not fit the row's untouched columns.
  inflate     one cell pushed 4 standard deviations beyond the column median, in the direction it already lies. A CONTROL:
              visible to the univariate check.

`invisible` (per row) records whether EVERY changed value in the row lies inside the univariate IQR fences (k = 1.5) of
its clean column, i.e. whether the univariate detector could not have seen it from the values alone.
"""

import numpy as np
import pandas as pd

from joint_outliers import UNIVARIATE_K, eligible_columns

KINDS = ["swap_one", "swap_block", "inflate"]
INFLATE_SD = 4.0
LOG_COLUMNS = ["row_id", "error_type", "column", "original_value", "corrupted_value", "kind", "invisible"]


def inject_joint(X: pd.DataFrame, y: pd.Series, rate: float, seed: int = 0, columns: list[str] | None = None):
    """Corrupt `rate` of the ROWS. `columns` defaults to the eligible numeric columns of the clean X."""
    rng = np.random.default_rng(seed)
    cols = list(columns) if columns is not None else eligible_columns(X)[0]
    X_dirty = X.copy()
    for c in cols:
        X_dirty[c] = X_dirty[c].astype(float)
    if not cols or len(X) < 3:
        return X_dirty, y.copy(), pd.DataFrame(columns=LOG_COLUMNS)

    clean = X[cols].astype(float)
    q1, q3 = clean.quantile(0.25), clean.quantile(0.75)
    lo, hi = q1 - UNIVARIATE_K * (q3 - q1), q3 + UNIVARIATE_K * (q3 - q1)
    med, sd = clean.median(), clean.std()

    n = len(X)
    n_rows = int(n * rate)
    rows = rng.choice(n, size=n_rows, replace=False)
    kinds = np.resize(np.array(KINDS, dtype=object), n_rows)
    rng.shuffle(kinds)

    log = []
    for pos, kind in zip(rows, kinds):
        row_id = X.index[pos]
        if kind == "swap_block" and len(cols) < 2:
            kind = "swap_one"
        if kind == "inflate":
            changes = {}
            c = cols[int(rng.integers(len(cols)))]
            x = clean.iat[pos, clean.columns.get_loc(c)]
            if not np.isfinite(x) or not np.isfinite(sd[c]) or sd[c] == 0:
                continue
            changes[c] = x + INFLATE_SD * sd[c] * (1.0 if x >= med[c] else -1.0)
        else:
            k = 1 if kind == "swap_one" else int(rng.integers(2, min(4, len(cols)) + 1))
            picked = [cols[int(j)] for j in rng.choice(len(cols), size=k, replace=False)]
            donor = int(rng.integers(n))
            tries = 0
            while donor == pos and tries < 5:
                donor, tries = int(rng.integers(n)), tries + 1
            if donor == pos:
                continue
            changes = {c: clean.iat[donor, clean.columns.get_loc(c)] for c in picked}
        # keep only cells that actually change and are finite
        changes = {c: v for c, v in changes.items()
                   if np.isfinite(v) and np.isfinite(clean.iat[pos, clean.columns.get_loc(c)])
                   and v != clean.iat[pos, clean.columns.get_loc(c)]}
        if not changes:
            continue
        invisible = all(lo[c] <= v <= hi[c] for c, v in changes.items())
        for c, v in changes.items():
            orig = clean.iat[pos, clean.columns.get_loc(c)]
            X_dirty.at[row_id, c] = v
            log.append({"row_id": row_id, "error_type": "joint_outliers", "column": c, "original_value": orig,
                        "corrupted_value": v, "kind": kind, "invisible": bool(invisible)})
    return X_dirty, y.copy(), pd.DataFrame(log, columns=LOG_COLUMNS)
