"""
Self-benchmark for REGRESSION (Phase 21; docs/PHASE21_SPEC.md). The same idea as selfbench.py on a numeric target:

  clean-ish reference from the user's data -> split (stratified by target quintile) -> for each error type x rate x seed inject
  ONLY that error into the training rows, train a Random Forest regressor, measure the R-squared damage against the clean-trained
  model (paired), then run the REAL TaskClean path on the corrupted rows and measure recovery, repair precision/recall and harm.

The evidence tables, the noise-aware harm rule and the safe-auto policy are selfbench.py's own functions (they only read the
'damage'/'recovery' columns), so a regression calibration is judged by exactly the rule classification uses. Metric: R-squared
(primary) with RMSE reported. Declines, with a reason, when the target cannot be predicted better than the mean.
"""
from __future__ import annotations

import time
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from baseline import build_preprocessor
from corruption import corruptors_for
from regression import strat_labels
from selfbench import (BENCH_ISSUES, DETECTOR_OF, LEARNABLE_MARGIN, MAX_MISSING_COLUMN, MAX_REFERENCE_ROWS, MIN_REFERENCE_ROWS,
                       NOISE_FLOOR_MIN, NULL_DRAWS, NULL_DROP_FRACTION, RATES, SEED0, TEST_FRACTION, UNIT_OF, SelfBench,
                       _combine_evidence, _detected_set, _evidence_table, _prf, _repaired_set, _same_data, _true_set,
                       _tracking_one)
from taskclean import AuditState, _apply, _full_record, _stratified_sample_index, audit_dataset

Progress = Callable[[float, str], None]
METRIC = "R-squared"


def _build_reference(state: AuditState, max_rows: int, seed: int):
    """(X_ref, y_ref, info) or (None, None, {"reason"}). Complete rows, no duplicates, wild target values removed."""
    if state.y is None:
        return None, None, {"reason": "no target column: the regression calibration needs a numeric target"}
    X, y = state.X, state.y
    labeled = y.notna()
    keep, dropped = [], {}
    for c in X.columns:
        if c in state.unmodeled_columns:
            dropped[c] = "high-cardinality text"
        elif X.loc[labeled, c].isna().mean() > MAX_MISSING_COLUMN:
            dropped[c] = f"{X.loc[labeled, c].isna().mean():.0%} missing"
        else:
            keep.append(c)
    if not keep:
        return None, None, {"reason": "no feature columns are complete enough to build a reference"}
    Xr, yr = X.loc[labeled, keep], y.loc[labeled]
    n_labeled = len(Xr)
    complete = ~Xr.isna().any(axis=1)
    Xr, yr = Xr[complete], yr[complete]
    n_complete = len(Xr)
    dup = _full_record(Xr, yr).duplicated(keep="first")
    Xr, yr = Xr[~dup], yr[~dup]
    n_dedup = len(Xr)
    prop = state.proposals
    n_removed = 0
    if not prop.empty:
        bad = Xr.index.intersection(pd.Index(prop.loc[(prop.issue_key == "label_errors") & (prop.tier == "strict"), "row_id"]))
        n_removed = len(bad)
        Xr, yr = Xr.drop(index=bad), yr.drop(index=bad)
    info = {"rows_labeled": n_labeled, "rows_complete": n_complete, "rows_after_dedup": n_dedup,
            "rows_removed_label_problems": n_removed, "rare_classes_dropped": [], "columns_left_out": dropped,
            "columns_used": keep}
    if len(Xr) < MIN_REFERENCE_ROWS:
        info["reason"] = (f"only {len(Xr):,} clean-ish rows remain after removing incomplete rows, duplicates and wild target "
                          f"values (need at least {MIN_REFERENCE_ROWS:,})")
        return None, None, info
    if len(Xr) > max_rows:
        idx = _stratified_sample_index(strat_labels(yr), Xr.index, max_rows, seed)
        Xr, yr = Xr.loc[idx], yr.loc[idx]
        info["capped_to_rows"] = len(Xr)
    info["rows_used"] = len(Xr)
    return Xr, yr, info


def _fit_scores(X_tr, y_tr, X_te, y_te, seed: int, n_estimators: int) -> dict:
    pipe = Pipeline([("prep", build_preprocessor(X_tr)),
                     ("rf", RandomForestRegressor(n_estimators=n_estimators, random_state=seed, n_jobs=-1))])
    pipe.fit(X_tr, y_tr)
    pred = pipe.predict(X_te)
    return {"r2": float(r2_score(y_te, pred)), "rmse": float(np.sqrt(mean_squared_error(y_te, pred)))}


def run_self_benchmark_regression(state: AuditState, n_seeds: int = 3, rates=RATES, max_rows: int = MAX_REFERENCE_ROWS,
                                  n_estimators: int = 100, progress: Progress | None = None) -> SelfBench:
    if n_seeds < 2:
        raise ValueError("n_seeds must be at least 2 (sign agreement and noise are meaningless with one run)")
    say = progress or (lambda f, m: None)
    t0 = time.monotonic()
    rates = tuple(rates)
    corruptors = corruptors_for("regression")

    def fail(reason, info=None, **kw):
        return SelfBench(ok=False, reason=reason, reference=info or {}, metric=METRIC, n_estimators=n_estimators,
                         n_seeds=n_seeds, task="regression", seconds=time.monotonic() - t0, **kw)

    say(0.0, "Building a clean-ish reference from your data")
    Xr, yr, info = _build_reference(state, max_rows, SEED0)
    if Xr is None:
        return fail(info.pop("reason"), info)
    X_tr, X_te, y_tr, y_te = train_test_split(Xr, yr, test_size=TEST_FRACTION, stratify=strat_labels(yr), random_state=SEED0)
    info.update(rows_train=len(X_tr), rows_test=len(X_te), minority_class_share=float("nan"))
    seeds = [SEED0 + i for i in range(n_seeds)]

    say(0.02, "Training clean reference models")
    base = {s: _fit_scores(X_tr, y_tr, X_te, y_te, s, n_estimators) for s in seeds}
    baseline = {s: v["r2"] for s, v in base.items()}
    say(0.03, "Checking the target is predictable at all")
    perm_rng = np.random.default_rng(SEED0 + 1)
    shuffled = [_fit_scores(X_tr, pd.Series(perm_rng.permutation(y_tr.values), index=y_tr.index), X_te, y_te, s, n_estimators)["r2"]
                for s in seeds]
    chance = max(0.0, float(np.mean(shuffled)))            # the mean predictor scores R-squared = 0
    if np.mean(list(baseline.values())) - chance < LEARNABLE_MARGIN:
        return fail(f"the target is barely predictable from the features (clean-trained R-squared "
                    f"{np.mean(list(baseline.values())):.3f} vs {chance:.3f} for the same model trained on shuffled targets), "
                    "so data-quality damage cannot be measured above the noise", info, baseline_f1=baseline, chance_f1=chance)

    say(0.04, "Measuring the noise floor (null experiment)")
    rng = np.random.default_rng(SEED0)
    null = []
    for seed in seeds:
        for _ in range(NULL_DRAWS):
            keep = rng.random(len(X_tr)) >= NULL_DROP_FRACTION
            null.append(_fit_scores(X_tr[keep], y_tr[keep], X_te, y_te, seed, n_estimators)["r2"] - baseline[seed])
    null_noise = float(np.sqrt(np.mean(np.square(null))))
    noise_floor = max(NOISE_FLOOR_MIN, null_noise)

    target = state.target
    has_numeric = any(pd.api.types.is_numeric_dtype(X_tr[c]) for c in X_tr.columns)
    issues, skipped = [], {}
    for issue in BENCH_ISSUES:
        if issue == "outliers" and not has_numeric:
            skipped[issue] = "no numeric columns to corrupt"
        else:
            issues.append(issue)
    total = len(issues) * len(rates) * n_seeds + 1
    done, rows = 0, []
    for issue in issues:
        try:
            for rate in rates:
                for seed in seeds:
                    say(0.05 + 0.9 * done / total, f"{issue} at {rate:.0%} (seed {seed})")
                    X_d, y_d, log = corruptors[issue](X_tr, y_tr, rate=rate, seed=seed)
                    dirty = _fit_scores(X_d, y_d, X_te, y_te, seed, n_estimators)
                    d = X_d.copy()
                    d[target] = y_d.values
                    st = audit_dataset(d, target, detectors={DETECTOR_OF.get(issue, issue)}, seed=seed, task="regression")
                    cleaned, rlog = _apply(st, {issue}, [])
                    X_rep, y_rep = cleaned.drop(columns=[target]), pd.to_numeric(cleaned[target], errors="coerce")
                    repaired = _fit_scores(X_rep, y_rep, X_te, y_te, seed, n_estimators)
                    restored = _same_data(X_rep, y_rep, X_tr, y_tr)
                    orig_index = X_d.index
                    truth = _true_set(issue, log)
                    det_p, det_r = _prf(truth, _detected_set(issue, st, orig_index))
                    rep_p, rep_r = _prf(truth, _repaired_set(issue, rlog, orig_index))
                    det = st.detections[DETECTOR_OF.get(issue, issue)]
                    rows.append({"issue": issue, "rate": rate, "seed": seed, "baseline_f1": baseline[seed],
                                 "dirty_f1": dirty["r2"], "repaired_f1": repaired["r2"],
                                 "damage": baseline[seed] - dirty["r2"], "recovery": repaired["r2"] - dirty["r2"],
                                 "baseline_rmse": base[seed]["rmse"], "dirty_rmse": dirty["rmse"], "repaired_rmse": repaired["rmse"],
                                 "detected_rate": det.rate, "detector_precision": det_p, "detector_recall": det_r,
                                 "repair_precision": rep_p, "repair_recall": rep_r, "restored_exactly": restored})
                    done += 1
        except Exception as e:                      # one failing issue must not sink the whole calibration
            skipped[issue] = f"failed during calibration: {type(e).__name__}: {e}"
            rows = [r for r in rows if r["issue"] != issue]
    if not rows:
        return fail("no error type could be benchmarked on this dataset", info, baseline_f1=baseline, skipped_issues=skipped)
    runs = pd.DataFrame(rows)

    say(0.96, "Measuring detector false-positive floors on the reference")
    ref_df = Xr.copy()
    ref_df[target] = yr.values
    fs = audit_dataset(ref_df, target, detectors={"missing_values", "duplicates", "outliers", "label_noise"}, seed=SEED0,
                       task="regression")
    floors = {("label_errors" if k == "label_noise" else k): float(v.rate) for k, v in fs.detections.items()}

    evidence = _combine_evidence(_evidence_table(runs, noise_floor), None, noise_floor, None)
    impact_curve = (runs.groupby(["issue", "rate"]).agg(mean_damage=("damage", "mean"), damage_std=("damage", "std"))
                    .reset_index().rename(columns={"issue": "error_type"}))
    reliability = (runs.groupby("issue")[["detector_precision", "detector_recall"]].mean().reset_index()
                   .rename(columns={"issue": "error_type", "detector_precision": "precision", "detector_recall": "recall"}))
    reliability.insert(1, "unit", reliability["error_type"].map(UNIT_OF))
    tr = _tracking_one(runs, "damage", noise_floor, METRIC)
    tracking = pd.DataFrame([{"error_type": k, **v, "pearson_r_macrof1": v["pearson_r"], "pearson_r_auc": float("nan")}
                             for k, v in tr.items()])
    say(1.0, "Self-benchmark complete")
    return SelfBench(ok=True, metric=METRIC, n_estimators=n_estimators, n_seeds=n_seeds, rates=rates, reference=info,
                     noise_floor=noise_floor, null_noise=null_noise, has_auc=False, baseline_f1=baseline,
                     chance_f1=chance, runs=runs, evidence=evidence, impact_curve=impact_curve,
                     detector_reliability=reliability, detector_tracking=tracking,
                     metric_disagreement=pd.DataFrame(columns=["error_type", "disagree", "note"]), baseline_floors=floors,
                     skipped_issues=skipped, seconds=time.monotonic() - t0, task="regression")
