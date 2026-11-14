"""
Bivariate and multivariate outlier detection (Phase 15). Method and failure criteria: docs/PHASE15_SPEC.md, written
before this module existed. FLAG-ONLY: nothing here proposes a replacement value.

One method at two scopes. Numeric columns are rank-transformed to normal scores (a Gaussian copula), a Minimum
Covariance Determinant fit gives a robust squared Mahalanobis distance, and a row is flagged when that distance exceeds
a chi-square cut-off with a stated null rate:

  multivariate   all eligible numeric columns together (>= 3 columns), alpha = 0.001
  bivariate      each eligible pair with 0.5 <= |Spearman rho| < 0.98 (at most the 20 strongest), alpha = 0.001 / pairs

The point of the bivariate scope is dilution: one disturbed pair barely moves a many-column distance, which is why the
PCA feature detector found ~25x fewer errors than the targeted cross-feature predictor.

The scores catch a value that is typical for its column but wrong for its row. They do NOT catch a categorical
combination (that is detect_feature_corruption_crossfeature) or a non-monotone relationship (see the spec, section E).

Pure functions, no dependency on taskclean.py (no import cycle).
"""

import warnings

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm, rankdata
from sklearn.covariance import MinCovDet

from detect import DetectionResult, detect_outliers_iqr
from relevance import _ID_NAME, ID_NAME_UNIQUE_RATIO, ID_UNIQUE_RATIO_NUM

# --- fixed in docs/PHASE15_SPEC.md before any code or validation run -----------------------------------------
MIN_ROWS = 200                      # usable (complete) rows needed for a scope, otherwise it declines
MIN_LEVELS = 10                     # a column needs MORE than this many distinct values
MAX_MODAL_SHARE = 0.5               # one value covering this share or more of the rows: degenerate (capital-gain)
MULTI_MIN_COLUMNS = 3
ALPHA = 0.001                       # multivariate; the bivariate family-wise alpha is ALPHA / pairs tested
PAIR_MIN_RHO = 0.5                  # below this the pair has no relationship worth testing
PAIR_MAX_RHO = 0.98                 # at or above this it is a near-duplicate (the redundant-column check's job)
MAX_PAIRS = 20
FIT_MAX_ROWS = 10_000               # the covariance is fitted on at most this many rows; every usable row is scored
UNIVARIATE_K = 1.5                  # `also_univariate` uses detect_outliers_iqr at this k
SAMPLE_SEED_SALT = 15015            # the fit sample is drawn from default_rng([seed, salt])
TIE_SEED_SALT = 15016               # random tie-breaking draws come from default_rng([seed, salt]) (Amendment 2)
TIES = "random"                     # the design of Amendment 2, used by the product. "average" is the PRE-REGISTERED
                                    # original, kept so the validation can run and judge both side by side.

RECORD_COLUMNS = ["row_id", "scope", "column", "direction", "distance", "cutoff", "also_univariate", "detail"]


def _id_like_numeric(s: pd.Series, name) -> bool:
    nn = s.dropna()
    if len(nn) == 0:
        return False
    ratio = nn.nunique() / len(nn)
    if _ID_NAME.search(str(name)) and ratio >= ID_NAME_UNIQUE_RATIO:
        return True
    if ratio >= ID_UNIQUE_RATIO_NUM and (nn == nn.round()).all():
        d = np.diff(nn.to_numpy(dtype=float))
        return bool((d > 0).all() or (d < 0).all())
    return False


def eligible_columns(X: pd.DataFrame) -> tuple[list[str], dict[str, str]]:
    """(eligible numeric columns, {skipped column: reason})."""
    ok, skipped = [], {}
    for c in X.columns:
        s = X[c]
        if not pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
            continue
        nn = s.dropna()
        if nn.nunique() <= MIN_LEVELS:
            skipped[c] = f"{nn.nunique()} distinct values: too few for a joint distance"
        elif (nn.quantile(0.75) - nn.quantile(0.25)) == 0 or nn.value_counts(normalize=True).iloc[0] >= MAX_MODAL_SHARE:
            skipped[c] = "zero-inflated or near-constant: its spread is not defined"
        elif _id_like_numeric(s, c):
            skipped[c] = "looks like an identifier"
        else:
            ok.append(c)
    return ok, skipped


def copula(values: np.ndarray, ties: str = "average", seed: int = 0) -> np.ndarray:
    """Rank -> normal scores, column by column.

    ties="average": tied values share their average rank (the pre-registered design). A value that covers a large share
        of the rows becomes a POINT MASS in normal-score space, which a Gaussian cut-off cannot describe.
    ties="random": tied values are ordered at random (seeded), so the marginal is exactly uniform and then normal
        (a randomised probability-integral transform; Amendment 2). Rows that share a value get different scores, which
        is noise by construction and lowers power on columns with many ties, but keeps the null calibrated."""
    n = values.shape[0]
    out = np.empty(values.shape, dtype=float)
    rng = np.random.default_rng([seed, TIE_SEED_SALT])
    for j in range(values.shape[1]):
        if ties == "random":
            order = np.lexsort((rng.random(n), values[:, j]))
            ranks = np.empty(n, dtype=float)
            ranks[order] = np.arange(1, n + 1)
        else:
            ranks = rankdata(values[:, j], method="average")
        out[:, j] = norm.ppf((ranks - 0.5) / n)
    return out


def _fit_scope(Z: np.ndarray, seed: int):
    """(squared robust distance for every row, signed standardised residual per row and column, None) or
    (None, None, reason) when the fit is not usable. No silent fallback to a non-robust covariance."""
    n = len(Z)
    if n > FIT_MAX_ROWS:
        rng = np.random.default_rng([seed, SAMPLE_SEED_SALT])
        fit_rows = np.sort(rng.choice(n, FIT_MAX_ROWS, replace=False))
    else:
        fit_rows = np.arange(n)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            mcd = MinCovDet(random_state=seed).fit(Z[fit_rows])
        except (ValueError, np.linalg.LinAlgError) as e:
            return None, None, f"covariance fit failed ({type(e).__name__})"
    if any("full rank" in str(w.message) or "singular" in str(w.message).lower() for w in caught):
        return None, None, "the covariance matrix is singular (columns are linearly dependent)"
    P = mcd.precision_
    if not np.all(np.isfinite(P)) or np.any(np.diag(P) <= 0):
        return None, None, "the precision matrix is not usable"
    d = Z - mcd.location_
    dist = np.einsum("ij,jk,ik->i", d, P, d)
    resid = (d @ P) / np.sqrt(np.diag(P))              # standardised conditional residual of each column given the rest
    return dist, resid, None


def _scope_records(rows: pd.Index, cols: list[str], dist, resid, cutoff, scope, X, univariate_mask):
    flag = dist > cutoff
    if not flag.any():
        return []
    out = []
    for i in np.flatnonzero(flag):
        j = int(np.argmax(np.abs(resid[i])))
        row_id = rows[i]
        col = cols[j]
        direction = "higher" if resid[i, j] > 0 else "lower"
        out.append({"row_id": row_id, "scope": scope, "column": col, "direction": direction,
                    "distance": float(dist[i]), "cutoff": float(cutoff),
                    "also_univariate": bool(univariate_mask.get(row_id, False)),
                    "detail": f"'{col}' is {direction} than the other column(s) in this row imply "
                              f"(robust squared distance {dist[i]:.1f} > {cutoff:.1f}; scope: {scope}). Flagged, not repaired"})
    return out


def detect_joint_outliers(X: pd.DataFrame, seed: int = 42, ties: str | None = None) -> DetectionResult:
    """Rows with an unusual combination of numeric values. `row_mask` / `rate` count JOINT-ONLY rows (flagged here
    and not by the univariate IQR check), so the two issue types never double-count a row. `extra["records"]` has one
    row per (flagged row, scope); `extra["scopes"]` says what ran, what declined and why."""
    ties = TIES if ties is None else ties
    cols, skipped = eligible_columns(X)
    univ = detect_outliers_iqr(X, k=UNIVARIATE_K).row_mask
    univariate_mask = univ.to_dict() if univ is not None else {}
    records: list[dict] = []
    scopes: list[dict] = []
    declined: dict[str, str] = {}

    def run(scope, use_cols, alpha):
        sub = X[use_cols].dropna()
        if len(sub) < MIN_ROWS:
            declined[scope] = f"only {len(sub)} rows have every value present (needs {MIN_ROWS})"
            return
        Z = copula(sub.to_numpy(dtype=float), ties, seed)
        dist, resid, why = _fit_scope(Z, seed)
        if why:
            declined[scope] = why
            return
        cutoff = float(chi2.ppf(1 - alpha, len(use_cols)))
        rec = _scope_records(sub.index, use_cols, dist, resid, cutoff, scope, X, univariate_mask)
        records.extend(rec)
        scopes.append({"scope": scope, "columns": list(use_cols), "rows_used": len(sub), "alpha": alpha,
                       "cutoff": cutoff, "n_flagged": len(rec)})

    # multivariate
    if len(cols) >= MULTI_MIN_COLUMNS:
        run("multivariate", cols, ALPHA)
    elif cols:
        declined["multivariate"] = f"only {len(cols)} eligible numeric column(s) (needs {MULTI_MIN_COLUMNS})"
    else:
        declined["multivariate"] = "no eligible numeric columns"

    # bivariate
    pairs = []
    if len(cols) >= 2:
        rho = X[cols].corr(method="spearman").abs()
        for i, a in enumerate(cols):
            for b in cols[i + 1:]:
                r = rho.at[a, b]
                if np.isfinite(r) and PAIR_MIN_RHO <= r < PAIR_MAX_RHO:
                    pairs.append((float(r), a, b))
    pairs = sorted(pairs, key=lambda t: (-t[0], t[1], t[2]))[:MAX_PAIRS]
    if pairs:
        for r, a, b in pairs:
            run(f"{a} x {b}", [a, b], ALPHA / len(pairs))
    else:
        declined["bivariate"] = (f"no pair of eligible columns has {PAIR_MIN_RHO} <= |Spearman rho| < {PAIR_MAX_RHO}"
                                 if len(cols) >= 2 else "fewer than 2 eligible numeric columns")

    rec_df = pd.DataFrame(records, columns=RECORD_COLUMNS)
    flagged_any = pd.Series(False, index=X.index)
    joint_only = pd.Series(False, index=X.index)
    if len(rec_df):
        flagged_any.loc[rec_df.row_id.unique()] = True
        joint_only = flagged_any & ~univ.reindex(X.index, fill_value=False)
    return DetectionResult("joint_outliers", float(joint_only.mean()) if len(X) else 0.0, joint_only,
                           {"records": rec_df, "scopes": scopes, "declined": declined, "eligible_columns": cols,
                            "skipped_columns": skipped, "pairs_tested": [(a, b, r) for r, a, b in pairs],
                            "row_mask_any": flagged_any, "ties": ties, "n_flagged_any": int(flagged_any.sum()),
                            "n_joint_only": int(joint_only.sum())})
