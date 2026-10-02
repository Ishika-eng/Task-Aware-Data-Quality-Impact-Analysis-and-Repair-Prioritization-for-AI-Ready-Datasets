"""
Data Quality Detection Engine (Phase 5).

Unlike corruption.py (which injects errors and therefore knows the ground
truth), everything here works the way a real deployed system would: it
only sees the dirty data and has to guess what's wrong. This is the
"Quality Audit" + "Error Characterization" step in the pipeline diagram.

Five detectors as specified, plus one extra (feature_corruption) since it's
one of our five injected error types and the later Quality Report /
Repair Prioritizer need an estimate for it too:

- detect_missing / detect_duplicates / detect_outliers_iqr / detect_inconsistency:
  straightforward checks on the raw data.

- detect_label_noise: a lightweight version of "confident learning" (the
  idea behind the cleanlab library). We cross-validate OUR model (Random
  Forest) so every row gets a prediction from a fold that never trained on
  it, then flag rows where the model is CONFIDENT about a class that
  disagrees with the given label. These are "suspected" errors, not
  confirmed ones -- the confidence threshold controls how sure we need the
  model to be before accusing a label of being wrong.

- detect_feature_corruption_pca: there's no per-column signal for
  corruption that shuffles values ACROSS rows -- each individual value
  stays plausible, only its combination with the rest of that row stops
  making sense. A univariate check can't see this. We fit PCA (a linear
  model of "normal" multivariate structure) on a fully encoded/imputed
  version of the data and flag rows that reconstruct poorly.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_predict, KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data import column_types


@dataclass
class DetectionResult:
    name: str
    rate: float                 # fraction of rows (or cells, for missing) flagged
    row_mask: pd.Series = None  # boolean Series aligned to X.index, True = flagged
    extra: dict = field(default_factory=dict)


def detect_missing(X: pd.DataFrame) -> DetectionResult:
    cell_missing = X.isna()
    rate = cell_missing.values.mean()
    row_mask = cell_missing.any(axis=1)
    per_column = cell_missing.mean().to_dict()
    return DetectionResult("missing_values", rate, row_mask,
                            {"cell_mask": cell_missing, "per_column_rate": per_column})


def detect_duplicates(X: pd.DataFrame) -> DetectionResult:
    dup_mask = X.duplicated(keep="first")
    rate = dup_mask.mean()
    return DetectionResult("duplicates", rate, dup_mask)


def detect_outliers_iqr(X: pd.DataFrame, k: float = 1.5,
                         fallback_lower_q: float = 0.005, fallback_upper_q: float = 0.995) -> DetectionResult:
    """Univariate: flag a row if ANY numeric column falls outside
    [Q1 - k*IQR, Q3 + k*IQR]. NaN cells never count as out-of-range (they're
    the missing-value detector's job).

    Zero-inflated / highly skewed columns (e.g. Adult's capital-gain,
    ~92% zeros) make Q1 == Q3 == 0, so IQR collapses to 0 and classic Tukey
    bounds become [0, 0] -- flagging every nonzero value as an "outlier"
    even with no corruption at all. For any column where that happens, we
    fall back to a percentile rule (flag the extreme fallback_lower_q /
    fallback_upper_q tails) instead, which adapts to the column's actual
    shape rather than collapsing to a degenerate rule.

    KNOWN LIMITATION (worth reporting, not a bug to chase further):
    capital-gain is additionally TOP-CODED in the real Census data -- about
    6% of people with any capital gains are capped at exactly 99999, a
    legitimate real value, not an error. That means even the 95th
    percentile of its nonzero values is already 99999, so there is no
    percentile threshold that can separate an injected extreme value from
    a genuinely large (but real) one on this specific column. This is a
    property of the data, not something a smarter generic statistical rule
    fixes -- it would need a domain-specific rule for known-capped columns.
    As a result, recall on capital-gain/capital-loss stays well below the
    other numeric columns (~30% vs ~99%+) no matter how this fallback is
    tuned.
    """
    numeric = X.select_dtypes("number")
    q1, q3 = numeric.quantile(0.25), numeric.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - k * iqr, q3 + k * iqr

    degenerate_cols = iqr[iqr == 0].index
    if len(degenerate_cols) > 0:
        fallback_lower = numeric[degenerate_cols].quantile(fallback_lower_q)
        fallback_upper = numeric[degenerate_cols].quantile(fallback_upper_q)
        lower[degenerate_cols] = fallback_lower
        upper[degenerate_cols] = fallback_upper

    out_of_range = (numeric < lower) | (numeric > upper)
    row_mask = out_of_range.any(axis=1).reindex(X.index, fill_value=False)
    rate = row_mask.mean()
    per_column_flags = {col: out_of_range[col] for col in out_of_range.columns}
    return DetectionResult("outliers", rate, row_mask,
                            {"degenerate_columns": list(degenerate_cols), "per_column_flags": per_column_flags,
                             "bounds": (lower, upper)})


def detect_inconsistency(X: pd.DataFrame, categorical_cols: list[str] | None = None) -> DetectionResult:
    """Flag categorical values that normalize to the same thing but are
    written differently (e.g. 'Male' / 'male' / 'M'). NaN cells are
    excluded from the comparison (that's missing-value territory)."""
    categorical_cols = categorical_cols or list(X.select_dtypes(include=["object", "category"]).columns)
    row_mask = pd.Series(False, index=X.index)
    if not categorical_cols:
        return DetectionResult("inconsistency", 0.0, row_mask, {"columns": []})

    for col in categorical_cols:
        col_values = X[col].astype(str)
        not_null = X[col].notna()
        normalized = col_values.where(not_null).str.strip().str.lower()
        variants_per_norm = col_values.where(not_null).groupby(normalized).transform("nunique")
        row_mask |= (variants_per_norm > 1).fillna(False)
    rate = row_mask.mean()
    return DetectionResult("inconsistency", rate, row_mask, {"columns": categorical_cols})


def _build_encoding_pipeline(X: pd.DataFrame) -> ColumnTransformer:
    """Fully imputed + encoded numeric representation, used only internally
    by the model-based detectors (label noise, feature corruption) -- NOT
    used by detect_missing/detect_outliers_iqr, which need to see the raw
    NaNs/values to do their job."""
    categorical_cols, numeric_cols = column_types(X)
    return ColumnTransformer([
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical_cols),
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), numeric_cols),
    ])


def detect_label_noise(X: pd.DataFrame, y: pd.Series, cv: int = 5,
                        confidence_threshold: float = 0.9, seed: int = 0) -> DetectionResult:
    pipeline = Pipeline([
        ("prep", _build_encoding_pipeline(X)),
        ("clf", RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)),
    ])
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
    oof_proba = cross_val_predict(pipeline, X, y, cv=skf, method="predict_proba", n_jobs=-1)
    classes = pipeline.fit(X, y).named_steps["clf"].classes_
    oof_pred_idx = oof_proba.argmax(axis=1)
    oof_pred = classes[oof_pred_idx]
    oof_confidence = oof_proba.max(axis=1)

    disagrees = oof_pred != y.values
    confident = oof_confidence >= confidence_threshold
    row_mask = pd.Series(disagrees & confident, index=X.index)
    rate = row_mask.mean()
    return DetectionResult("label_noise", rate, row_mask,
                            {"oof_pred": pd.Series(oof_pred, index=X.index),
                             "oof_confidence": pd.Series(oof_confidence, index=X.index)})


def detect_feature_corruption_pca(X: pd.DataFrame, variance_to_keep: float = 0.95,
                                   k: float = 3.0, seed: int = 0) -> DetectionResult:
    encoder = _build_encoding_pipeline(X)
    encoded = encoder.fit_transform(X)
    if hasattr(encoded, "toarray"):
        encoded = encoded.toarray()

    pca = PCA(n_components=variance_to_keep, random_state=seed)
    projected = pca.fit_transform(encoded)
    reconstructed = pca.inverse_transform(projected)
    recon_error = np.mean((encoded - reconstructed) ** 2, axis=1)

    threshold = recon_error.mean() + k * recon_error.std()
    row_mask = pd.Series(recon_error > threshold, index=X.index)
    rate = row_mask.mean()
    return DetectionResult("feature_corruption", rate, row_mask,
                            {"reconstruction_error": pd.Series(recon_error, index=X.index)})


def detect_feature_corruption_crossfeature(X: pd.DataFrame, cat_confidence_threshold: float = 0.05,
                                            num_z_threshold: float = 4.0, cv: int = 5,
                                            n_estimators: int = 150, seed: int = 0) -> DetectionResult:
    """Per-column cross-feature prediction: for each column, predict it from
    ALL the other columns (cross-validated, so every row's prediction comes
    from a model that never saw it), then flag a cell as suspect if the
    ACTUAL value disagrees badly with what the other columns predict.

    This targets the corruption mechanism directly -- a swapped cell no
    longer fits its row's other values -- which is a much more targeted
    signal than a single global reconstruction-error number (PCA) or a
    generic anomaly score (Isolation Forest), both of which average the
    one-column disruption across the whole feature space and lose it in
    the noise. Empirically (see phase5_test.py) this roughly 25x's recall
    over the PCA approach, at a tunable precision cost via the threshold.

    Categorical columns: flag if the model's predicted probability for the
    OBSERVED category is below cat_confidence_threshold (i.e. the model is
    confident the row should look different there).
    Numeric columns: flag if the observed value's robust z-score against
    the model's prediction exceeds num_z_threshold.

    Beyond the boolean flag, `extra["per_column"][col]` also carries the
    raw continuous score and the cross-validated model's own out-of-fold
    predicted value for that cell. Two reasons: (1) Phase 9's conservative
    repair policy needs a STRICTER secondary threshold for "auto-repair"
    vs. this function's (looser) "flag for review" threshold, and
    recomputing the cross-validated model just to apply a different cutoff
    would double an already expensive step for nothing; (2) that same
    out-of-fold prediction is a better repair VALUE than generic
    mean/median imputation -- it's row-specific, reusing the same model
    that already looked at this exact row's other columns.
    """
    categorical_cols, numeric_cols = column_types(X)
    row_mask = pd.Series(False, index=X.index)
    per_column_flags = {}
    per_column = {}

    for col in categorical_cols:
        other_cols = [c for c in X.columns if c != col]
        prep = _build_encoding_pipeline(X[other_cols])
        pipe = Pipeline([("prep", prep),
                          ("clf", RandomForestClassifier(n_estimators=n_estimators, random_state=seed, n_jobs=-1))])
        # Missing cells in the TARGET column itself must be imputed before
        # cross-validating on it -- StratifiedKFold rejects NaN in y, and
        # this column can legitimately have missing values (e.g. when this
        # detector runs on a dataset with missing_values ALSO injected).
        mode = X[col].mode(dropna=True)
        fill_value = mode.iloc[0] if len(mode) else "missing"
        target = X[col].fillna(fill_value).astype(str)
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
        proba = cross_val_predict(pipe, X[other_cols], target, cv=skf, method="predict_proba", n_jobs=-1)
        classes = pipe.fit(X[other_cols], target).named_steps["clf"].classes_
        class_idx = {c: i for i, c in enumerate(classes)}
        actual_proba = np.array([proba[i, class_idx[v]] for i, v in enumerate(target.values)])
        predicted_value = classes[proba.argmax(axis=1)]
        flagged = pd.Series(actual_proba < cat_confidence_threshold, index=X.index)
        per_column_flags[col] = flagged
        per_column[col] = {
            "score": pd.Series(actual_proba, index=X.index),  # LOWER = more suspicious
            "predicted_value": pd.Series(predicted_value, index=X.index),
            "kind": "categorical",
        }
        row_mask |= flagged

    for col in numeric_cols:
        other_cols = [c for c in X.columns if c != col]
        prep = _build_encoding_pipeline(X[other_cols])
        pipe = Pipeline([("prep", prep),
                          ("reg", RandomForestRegressor(n_estimators=n_estimators, random_state=seed, n_jobs=-1))])
        target = X[col].astype(float)
        imputed_target = target.fillna(target.median())
        kf = KFold(n_splits=cv, shuffle=True, random_state=seed)
        oof_pred = cross_val_predict(pipe, X[other_cols], imputed_target, cv=kf, n_jobs=-1)
        residual = imputed_target.values - oof_pred
        z = (residual - residual.mean()) / (residual.std() + 1e-9)
        flagged = pd.Series(np.abs(z) > num_z_threshold, index=X.index)
        per_column_flags[col] = flagged
        per_column[col] = {
            "score": pd.Series(np.abs(z), index=X.index),  # HIGHER = more suspicious
            "predicted_value": pd.Series(oof_pred, index=X.index),
            "kind": "numeric",
        }
        row_mask |= flagged

    rate = row_mask.mean()
    return DetectionResult("feature_corruption", rate, row_mask,
                            {"per_column_flags": per_column_flags, "per_column": per_column})


def run_full_audit(X: pd.DataFrame, y: pd.Series, seed: int = 0) -> dict[str, DetectionResult]:
    return {
        "missing_values": detect_missing(X),
        "duplicates": detect_duplicates(X),
        "outliers": detect_outliers_iqr(X),
        "inconsistency": detect_inconsistency(X),
        "label_noise": detect_label_noise(X, y, seed=seed),
        "feature_corruption": detect_feature_corruption_crossfeature(X, seed=seed),
    }
