"""
Regression support (Phase 21; docs/PHASE21_SPEC.md): numeric-target preparation, the target-error detector and the
target-stratification helper. Classification code paths do not import anything from here that changes their behaviour.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline

from detect import DetectionResult, _build_encoding_pipeline

TASKS = ("classification", "regression")
TARGET_Z = 4.0               # robust z of the out-of-fold residual (scaled within bands of the predicted value) above which a target value is flagged (fixed in the spec)
SCALE_BANDS = 5              # Amendment A1
MIN_DISTINCT_TARGET = 2
CLASS_LIKE_MAX = 20          # a numeric target with <= this many distinct values is warned about ("looks like classes")


def suggest_task(target: pd.Series, max_classes: int = CLASS_LIKE_MAX) -> dict:
    """Automatic suggestion from the target's values (the user can always override)."""
    nn = target.dropna()
    numeric = pd.to_numeric(nn, errors="coerce")
    share_num = float(numeric.notna().mean()) if len(nn) else 0.0
    n_distinct = int(nn.nunique())
    if share_num >= 0.95 and n_distinct > max_classes:
        return {"task": "regression", "reason": f"numeric target with {n_distinct:,} distinct values"}
    return {"task": "classification",
            "reason": (f"{n_distinct} distinct value(s)" + ("" if share_num < 0.95 else ", few enough to be classes"))}


def prepare_target(df: pd.DataFrame, target: str) -> tuple[pd.Series, list[str]]:
    """Numeric target (float, NaN where missing or non-numeric) plus notes. Raises ValueError with a clear message."""
    raw = df[target]
    num = pd.to_numeric(raw, errors="coerce").astype(float)
    num[~np.isfinite(num)] = np.nan
    bad = int((raw.notna() & num.isna()).sum())
    if num.notna().sum() == 0:
        raise ValueError(f"target '{target}' has no numeric values -- regression needs a numeric target")
    if num.nunique() < MIN_DISTINCT_TARGET:
        raise ValueError(f"target '{target}' is constant -- nothing to predict")
    notes = []
    if bad:
        notes.append(f"{bad:,} value(s) in the target '{target}' are not numbers: they are treated as a missing target")
    if num.nunique() <= CLASS_LIKE_MAX:
        notes.append(f"the target has only {num.nunique()} distinct values; if they are categories, choose classification")
    return num, notes


def strat_labels(y: pd.Series, bins: int = 5) -> pd.Series:
    """Quantile-bin labels used wherever classification stratifies by class (splits, sampling)."""
    try:
        return pd.qcut(y, q=bins, labels=False, duplicates="drop").astype(int).astype(str)
    except ValueError:
        return pd.Series("0", index=y.index)


def detect_target_errors(X: pd.DataFrame, y: pd.Series, cv: int = 5, z: float = TARGET_Z, seed: int = 0) -> DetectionResult:
    """Out-of-fold random-forest regression; flag targets whose residual is more than `z` robust sigmas (MAD, per band of predicted value) from the
    typical residual. Returns the same shape as detect_label_noise: oof_pred is the proposed replacement and oof_confidence
    carries the absolute robust z (how far out the flagged value is)."""
    pipe = Pipeline([("prep", _build_encoding_pipeline(X)),
                     ("rf", RandomForestRegressor(n_estimators=200, random_state=seed, n_jobs=-1))])
    oof = cross_val_predict(pipe, X, y.astype(float), cv=KFold(n_splits=cv, shuffle=True, random_state=seed), n_jobs=-1)
    resid = y.astype(float).to_numpy() - oof
    # Amendment A1: residuals grow with the predicted value, so a single global scale flags the high end of the target on clean
    # data (3.9% of clean California Housing). The scale is therefore estimated within bands of the predicted value (MAD).
    band = pd.qcut(pd.Series(oof), q=SCALE_BANDS, labels=False, duplicates="drop").to_numpy()
    zs = np.zeros(len(resid))
    for k in np.unique(band):
        m = band == k
        med = float(np.median(resid[m]))
        mad = float(np.median(np.abs(resid[m] - med)) * 1.4826)
        scale = mad if mad > 0 else float(np.std(resid[m]))
        zs[m] = np.abs(resid[m] - med) / scale if scale > 0 else 0.0
    mask = pd.Series(zs > z, index=X.index)
    return DetectionResult("label_noise", float(mask.mean()), mask,
                           {"oof_pred": pd.Series(oof, index=X.index), "oof_confidence": pd.Series(zs, index=X.index),
                            "banded_scale": True, "z_threshold": z})


def target_profile_regression(df: pd.DataFrame, target: str) -> dict:
    """Target block for the health report when the task is regression (the counterpart of health_snapshot.target_profile)."""
    if target not in df.columns:
        raise ValueError(f"target column '{target}' not found in the dataset")
    raw = df[target]
    num = pd.to_numeric(raw, errors="coerce").astype(float)
    num[~np.isfinite(num)] = np.nan
    n = len(raw)
    labelled = num.dropna()
    bad = int((raw.notna() & num.isna()).sum())
    out = {"target": str(target), "task": "regression", "rows": int(n), "rows_without_target": int(n - len(labelled)),
           "rows_without_target_pct": float((n - len(labelled)) / n) if n else 0.0, "non_numeric_values": bad,
           "n_distinct": int(labelled.nunique()), "usable": True, "warnings": []}
    if len(labelled) == 0:
        out["usable"] = False
        out["warnings"].append("no numeric values: regression needs a numeric target")
        return out
    q = labelled.quantile([0.0, 0.25, 0.5, 0.75, 1.0]).to_list()
    out.update(mean=float(labelled.mean()), std=float(labelled.std(ddof=0)), min=q[0], q25=q[1], median=q[2], q75=q[3], max=q[4],
               skew=float(labelled.skew()) if len(labelled) > 2 else 0.0)
    if labelled.nunique() < MIN_DISTINCT_TARGET:
        out["usable"] = False
        out["warnings"].append("the target is constant: nothing to predict")
    elif labelled.nunique() <= CLASS_LIKE_MAX:
        out["warnings"].append(f"only {labelled.nunique()} distinct values: if they are categories, choose classification")
    if bad:
        out["warnings"].append(f"{bad:,} value(s) are not numbers and will be treated as a missing target")
    if abs(out["skew"]) > 2:
        out["warnings"].append(f"the target is strongly skewed (skewness {out['skew']:.1f}); a few very large values can dominate R-squared")
    return out
