"""
Ground-truth injectors for the Phase 14 validation (docs/PHASE14_SPEC.md, section C).

Deliberately NOT registered in corruption.CORRUPTORS: that registry drives the frozen Phase 7/9 experiments and the
self-benchmark, and adding to it would change what they run. Same contract as the corruption/ modules: take
(X, y, rate, seed), never mutate the input, return (X_dirty, y, log) where the log records every changed cell.
"""

import numpy as np
import pandas as pd

KINDS = ["case", "whitespace", "punctuation", "abbreviation", "typo"]   # equal mix: an ASSUMPTION, see the spec

_LETTERS = np.array(list("abcdefghijklmnopqrstuvwxyz"))


def _vary(v: str, kind: str, rng: np.random.Generator):
    """(new value, kind actually used). Falls back to another kind when `kind` cannot change this value."""
    if kind == "abbreviation" and len(v) > 3:
        return v[:3], "abbreviation"
    if kind == "typo" and len(v) >= 5:
        i = int(rng.integers(1, len(v) - 1))
        op = int(rng.integers(3))
        if op == 0:
            out = v[:i] + v[i + 1:]                                            # delete a character
        elif op == 1:
            out = v[:i] + v[i + 1] + v[i] + v[i + 2:] if i + 2 <= len(v) else v[:i - 1] + v[i] + v[i - 1] + v[i + 1:]
        else:
            c = str(rng.choice(_LETTERS))
            out = v[:i] + (c if c != v[i].lower() else "x") + v[i + 1:]       # substitute a character
        if out != v and out.lower() != v.lower():
            return out, "typo"
    if kind == "punctuation":
        if "-" in v:
            return v.replace("-", " " if rng.random() < 0.5 else "_"), "punctuation"
        return v + ".", "punctuation"
    if kind == "whitespace":
        return (" " + v) if rng.random() < 0.5 else (v + " "), "whitespace"
    up, low = v.upper(), v.lower()
    if up != v:
        return up, "case"
    if low != v:
        return low, "case"
    return v + " ", "whitespace"                                              # digits-only value: no case to change


def inject_inconsistency(X: pd.DataFrame, y: pd.Series, rate: float, seed: int = 0):
    """Rewrite `rate` of the CATEGORICAL cells as an inconsistent variant of their own value.

    rate is a fraction of categorical cells (comparable to how missing_values and feature_corruption count cells).
    The requested kind cycles through KINDS so the mix is equal; the log records the kind actually used."""
    rng = np.random.default_rng(seed)
    X_dirty = X.copy()
    cat_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    for c in cat_cols:
        X_dirty[c] = X_dirty[c].astype(object)
    n_rows = len(X_dirty)
    n_cells = int(n_rows * len(cat_cols) * rate)
    flat = rng.choice(n_rows * len(cat_cols), size=n_cells, replace=False)
    kinds = np.resize(np.array(KINDS, dtype=object), n_cells)
    rng.shuffle(kinds)
    log = []
    for pos, kind in zip(flat, kinds):
        r, cj = divmod(int(pos), len(cat_cols))
        col, row_id = cat_cols[cj], X_dirty.index[r]
        v = X_dirty.at[row_id, col]
        if not isinstance(v, str):
            continue
        new, used = _vary(v, kind, rng)
        X_dirty.at[row_id, col] = new
        log.append({"row_id": row_id, "error_type": "inconsistency", "column": col, "original_value": v,
                    "corrupted_value": new, "kind": used})
    return X_dirty, y.copy(), pd.DataFrame(log, columns=["row_id", "error_type", "column", "original_value",
                                                          "corrupted_value", "kind"])


IRRELEVANT_KINDS = ["gaussian", "uniform_int", "random_cat3", "random_cat30", "constant", "exact_copy",
                    "linear_copy", "sequential_id", "shuffled_id", "string_id"]


def add_irrelevant_columns(X: pd.DataFrame, seed: int, copy_of: str = "age", offset: int = 0):
    """Append columns that are irrelevant BY CONSTRUCTION. Returns (X_aug, {column: kind}).

    `offset` shifts the identifier columns so a test split gets ids that never occur in training, as on new data;
    random columns are drawn independently each call (use a different seed for the test split)."""
    rng = np.random.default_rng([seed, 14])
    n = len(X)
    out = X.copy()
    truth = {}

    def put(name, values, kind):
        out[name] = values
        truth[name] = kind

    put("x_gaussian", rng.normal(size=n), "gaussian")
    put("x_uniform_int", rng.integers(0, 1000, n).astype(float), "uniform_int")
    put("x_cat3", pd.Series(rng.choice(["p", "q", "r"], n), index=X.index).astype(object), "random_cat3")
    put("x_cat30", pd.Series(rng.choice([f"c{i}" for i in range(30)], n), index=X.index).astype(object), "random_cat30")
    put("x_constant", pd.Series(["same"] * n, index=X.index).astype(object), "constant")
    put("x_copy_of_" + copy_of, X[copy_of].values, "exact_copy")
    put("x_linear_of_" + copy_of, X[copy_of].astype(float).values * 0.5 + 3.0, "linear_copy")
    put("x_row_id", np.arange(offset, offset + n).astype(float), "sequential_id")
    put("x_random_id", (rng.permutation(n) + offset).astype(float), "shuffled_id")
    put("x_string_id", pd.Series([f"user_{offset + i:07d}" for i in range(n)], index=X.index).astype(object), "string_id")
    return out, truth
