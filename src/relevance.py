"""
Irrelevant-feature detection (Phase 14). Method fixed in docs/PHASE14_SPEC.md.

REPORT-ONLY: this never changes the cleaned file and never drops a column. Whether a feature is useful depends on
the model and the task, and removing one is a modelling decision, so the output is a recommendation with its
evidence. Checks, first match wins:

  constant   one distinct value, or the top value covers >= 99.5% of the non-null rows
  id_like    an identifier / free text: nearly every row distinct
  redundant  a numeric pair with |Pearson| >= 0.995, or a one-to-one value mapping between two columns
             (e.g. Adult's `education` and `education-num`); the LATER column is flagged and names the other
  no_detectable_relevance
             random-probe test (needs a target). Twelve random probe columns are added; a random forest's permutation
             importance on a held-out split, AND the mutual information with the target, are measured for every
             column. A column is flagged only if BOTH are no larger than the largest probe's. Requiring both stops
             a feature being flagged just because a correlated twin absorbed its importance.

"No detectable relevance" is not proof a feature is useless: it can matter only in combination with others, and
the test is random-forest based. It is a prompt for a human, nothing more.
"""

import re

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# --- fixed in docs/PHASE14_SPEC.md before any validation run ------------------------------------------------
NEAR_CONSTANT_SHARE = 0.995
ID_UNIQUE_RATIO_CAT = 0.95          # categorical / text: unique ratio at or above this (and >= ID_MIN_ROWS rows)
ID_UNIQUE_RATIO_NUM = 0.98          # integer-like numeric: unique ratio at or above this AND strictly monotonic
ID_NAME_UNIQUE_RATIO = 0.5          # a column NAMED like an id needs only this much uniqueness
ID_MIN_ROWS = 50
_ID_NAME = re.compile(r"(^|[^a-z])(id|uuid|guid|index|idx)([^a-z]|$)", re.IGNORECASE)
REDUNDANT_CORR = 0.995
REDUNDANT_MIN_PAIRS = 30            # rows where both columns are present
REDUNDANT_MAX_LEVELS = 200          # one-to-one mapping check only for columns with at most this many values
PROBES_PER_KIND = 4                 # 3 kinds x 4 = 12 probes. A truly irrelevant column is exchangeable with the probes,
                                    # so it beats all K of them with probability 1/(K+1): K=3 would keep ~25% of noise
                                    # columns on each criterion; K=12 keeps ~8% (see docs/PHASE14_SPEC.md, Amendment 1)
PROBE_SEED_SALT = 14014             # probes are drawn from default_rng([seed, salt]), NOT default_rng(seed), so they can
                                    # never coincide with data a caller generated from the same integer seed
PROBE_TREES = 150
PROBE_REPEATS = 5
PROBE_TEST_SIZE = 0.3
PROBE_MAX_ROWS = 6000
PROBE_MAX_COLUMNS = 100
PROBE_MAX_LEVELS = 50               # categorical columns with more levels are not modelled (matches taskclean)
PROBE_MIN_ROWS = 200
PROBE_MIN_CLASS_ROWS = 10

RECOMMEND = {
    "constant": "drop candidate",
    "id_like": "drop candidate (identifier)",
    "redundant": "drop candidate",
    "no_detectable_relevance": "review (weak evidence)",
    "relevant": "keep",
    "not_assessed": "-",
}


def _structural_flags(X: pd.DataFrame, integer_like: dict[str, bool]) -> dict[str, dict]:
    """constant / id_like / redundant, from the data alone (no target needed)."""
    out: dict[str, dict] = {}
    n_total = len(X)
    for c in X.columns:
        s = X[c].dropna()
        n = len(s)
        if n == 0:
            continue
        nun = s.nunique()
        top = s.value_counts().iloc[0] / n
        if nun <= 1:
            out[c] = {"verdict": "constant", "reason": "a single distinct value: carries no information"}
            continue
        if top >= NEAR_CONSTANT_SHARE:
            out[c] = {"verdict": "constant", "reason": f"{top:.1%} of rows share one value: almost no information"}
            continue
        ratio = nun / n
        numeric = pd.api.types.is_numeric_dtype(X[c])
        named = bool(_ID_NAME.search(str(c))) and ratio >= ID_NAME_UNIQUE_RATIO
        cat_id = (not numeric) and n >= ID_MIN_ROWS and ratio >= ID_UNIQUE_RATIO_CAT
        num_id = numeric and integer_like.get(c, False) and n >= ID_MIN_ROWS and ratio >= ID_UNIQUE_RATIO_NUM \
            and (s.is_monotonic_increasing or s.is_monotonic_decreasing)
        if named or cat_id or num_id:
            out[c] = {"verdict": "id_like",
                      "reason": f"{ratio:.0%} of values are distinct: looks like an identifier or free text"}
    # redundancy: only among columns not already flagged, earlier column is kept
    cols = [c for c in X.columns if c not in out]
    nun = {c: X[c].nunique() for c in cols}
    num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(X[c])]
    corr = X[num_cols].corr(min_periods=REDUNDANT_MIN_PAIRS).abs() if len(num_cols) >= 2 else None
    codes = {}
    for j, cj in enumerate(cols):
        if cj in out:
            continue
        for ci in cols[:j]:
            if ci in out and out[ci]["verdict"] == "redundant":
                continue                                           # compare against kept columns only
            if corr is not None and ci in corr.index and cj in corr.index:
                r = corr.loc[ci, cj]
                if pd.notna(r) and r >= REDUNDANT_CORR:
                    out[cj] = {"verdict": "redundant", "redundant_with": ci,
                               "reason": f"numeric correlation {r:.3f} with '{ci}': says the same thing"}
                    break
            if nun[ci] == nun[cj] and 2 <= nun[cj] <= REDUNDANT_MAX_LEVELS:
                for c in (ci, cj):
                    if c not in codes:
                        codes[c] = pd.Series(pd.factorize(X[c])[0], index=X.index).where(X[c].notna())
                pair = pd.DataFrame({"a": codes[ci], "b": codes[cj]}).dropna()
                if len(pair) >= REDUNDANT_MIN_PAIRS and pair.groupby("a")["b"].nunique().max() == 1 \
                        and pair.groupby("b")["a"].nunique().max() == 1:
                    out[cj] = {"verdict": "redundant", "redundant_with": ci,
                               "reason": f"one-to-one with '{ci}': every value of one maps to exactly one value of the other"}
                    break
    return out


def _probe_columns(n: int, rng: np.random.Generator) -> pd.DataFrame:
    cols = {}
    for i in range(PROBES_PER_KIND):
        cols[f"__probe_gaussian_{i}"] = rng.normal(size=n)
        cols[f"__probe_integer_{i}"] = rng.integers(0, 1000, n).astype(float)
        cols[f"__probe_cat20_{i}"] = pd.Series(rng.choice([f"p{j}" for j in range(20)], n), dtype="category")
    return pd.DataFrame(cols)


def _relevance_scores(Xc: pd.DataFrame, y: pd.Series, seed: int) -> pd.DataFrame | str:
    """Permutation importance and mutual information for every column of Xc plus 12 random probes.
    Returns a DataFrame indexed by column, or a string explaining why the test could not run."""
    rng = np.random.default_rng([seed, PROBE_SEED_SALT])
    labeled = y.notna()
    Xc, y = Xc.loc[labeled], y.loc[labeled].astype(str)
    if len(Xc) < PROBE_MIN_ROWS:
        return f"fewer than {PROBE_MIN_ROWS} labeled rows"
    vc = y.value_counts()
    keep_classes = vc[vc >= PROBE_MIN_CLASS_ROWS].index
    if len(keep_classes) < 2:
        return f"fewer than two classes with at least {PROBE_MIN_CLASS_ROWS} rows"
    sel = y.isin(keep_classes)
    Xc, y = Xc.loc[sel], y.loc[sel]
    if len(Xc) > PROBE_MAX_ROWS:
        pick = rng.choice(len(Xc), PROBE_MAX_ROWS, replace=False)
        Xc, y = Xc.iloc[np.sort(pick)], y.iloc[np.sort(pick)]
    D = pd.concat([Xc.reset_index(drop=True), _probe_columns(len(Xc), rng)], axis=1)
    y = y.reset_index(drop=True)
    cat_cols = [c for c in D.columns if not pd.api.types.is_numeric_dtype(D[c])]
    for c in cat_cols:
        D[c] = D[c].astype(object).where(D[c].notna(), np.nan).astype(str).replace("nan", np.nan)
    num_cols = [c for c in D.columns if c not in cat_cols]
    parts = []
    if cat_cols:
        parts.append(("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                                       ("enc", OneHotEncoder(handle_unknown="ignore"))]), cat_cols))
    if num_cols:
        parts.append(("num", SimpleImputer(strategy="median"), num_cols))
    pipe = Pipeline([("prep", ColumnTransformer(parts)),
                     ("rf", RandomForestClassifier(n_estimators=PROBE_TREES, min_samples_leaf=2,
                                                   random_state=seed, n_jobs=-1))])
    stratify = y if y.value_counts().min() >= 2 else None
    D_tr, D_te, y_tr, y_te = train_test_split(D, y, test_size=PROBE_TEST_SIZE, random_state=seed, stratify=stratify)
    pipe.fit(D_tr, y_tr)
    pi = permutation_importance(pipe, D_te, y_te, n_repeats=PROBE_REPEATS, random_state=seed,
                                scoring="balanced_accuracy", n_jobs=1)
    M = D.copy()
    for c in cat_cols:
        M[c] = pd.factorize(M[c])[0].astype(float)
    for c in num_cols:
        M[c] = M[c].fillna(M[c].median())
    mi = mutual_info_classif(M.values, y.values, discrete_features=np.array([c in cat_cols for c in M.columns]),
                             random_state=seed)
    return pd.DataFrame({"perm_importance": pi.importances_mean, "perm_importance_std": pi.importances_std,
                         "mutual_info": mi}, index=D.columns)


def assess_features(X: pd.DataFrame, y: pd.Series | None, seed: int = 42,
                    integer_like: dict[str, bool] | None = None) -> pd.DataFrame:
    """One row per column of X: verdict, reason, evidence and a recommendation."""
    integer_like = integer_like or {}
    flags = _structural_flags(X, integer_like)
    rows = {c: {"column": c, "kind": "numeric" if pd.api.types.is_numeric_dtype(X[c]) else "categorical",
                "verdict": None, "reason": "", "redundant_with": None, "perm_importance": np.nan,
                "perm_importance_std": np.nan, "mutual_info": np.nan, "probe_importance_threshold": np.nan,
                "probe_mi_threshold": np.nan} for c in X.columns}
    for c, f in flags.items():
        rows[c].update(f)

    remaining = [c for c in X.columns if c not in flags]
    why_not = None
    wide = [c for c in remaining if not pd.api.types.is_numeric_dtype(X[c]) and X[c].nunique() > PROBE_MAX_LEVELS]
    for c in wide:
        rows[c].update(verdict="not_assessed", reason=f"more than {PROBE_MAX_LEVELS} distinct values: not modelled")
    modelled = [c for c in remaining if c not in wide]
    if y is None:
        why_not = "no target column: relevance to the target cannot be tested"
    elif len(modelled) > PROBE_MAX_COLUMNS:
        why_not = f"more than {PROBE_MAX_COLUMNS} candidate columns: relevance test skipped"
    elif not modelled:
        why_not = "no columns left to test"
    scores = _relevance_scores(X[modelled], y, seed) if why_not is None else why_not
    if isinstance(scores, str):
        for c in modelled:
            rows[c].update(verdict="not_assessed", reason=scores)
    else:
        probes = scores.loc[[i for i in scores.index if str(i).startswith("__probe")]]
        thr_imp = max(float(probes.perm_importance.max()), 0.0)
        thr_mi = float(probes.mutual_info.max())
        for c in modelled:
            s = scores.loc[c]
            weak = s.perm_importance <= thr_imp and s.mutual_info <= thr_mi
            rows[c].update(perm_importance=float(s.perm_importance), perm_importance_std=float(s.perm_importance_std),
                           mutual_info=float(s.mutual_info), probe_importance_threshold=thr_imp,
                           probe_mi_threshold=thr_mi,
                           verdict="no_detectable_relevance" if weak else "relevant",
                           reason=("not distinguishable from random probe columns on either permutation importance "
                                   f"({s.perm_importance:.4f} vs {thr_imp:.4f}) or mutual information "
                                   f"({s.mutual_info:.4f} vs {thr_mi:.4f})") if weak else
                                  f"clearly above the random probes (importance {s.perm_importance:.4f} vs "
                                  f"{thr_imp:.4f}, mutual information {s.mutual_info:.4f} vs {thr_mi:.4f})")
    out = pd.DataFrame(list(rows.values()))
    out["recommendation"] = out.verdict.map(RECOMMEND)
    return out
