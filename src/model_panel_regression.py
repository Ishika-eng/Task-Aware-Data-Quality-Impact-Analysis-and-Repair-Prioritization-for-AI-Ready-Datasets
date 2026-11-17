"""
Model panel for REGRESSION (Phase 21; docs/PHASE21_SPEC.md). The same fair-comparison rules as model_panel.py (split first,
audit and repairs on the training rows only, arms before / default / aggressive / each repair alone, null-experiment noise floor,
>= 80% seed agreement, leakage count, no percentage for a within-noise delta) with four regressors and R-squared:

  Ridge (linear), Random forest, Gradient boosting (HistGradientBoostingRegressor), Neural net (MLPRegressor on a standardised
  target). Report-only: it never changes which repairs are applied. The test rows are as dirty as the user's file, so a gain
  means "training on the repaired rows predicted the real rows better", not that a repair was correct.
"""
from __future__ import annotations

import time
import warnings
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from baseline import build_preprocessor
from model_panel import (AGGRESSIVE, AUDIT_MODEL_ROWS, DEFAULT_SEEDS, GB_ITER, LEARNABLE_MARGIN, MAX_PANEL_ROWS,
                         MIN_MLP_TRAIN_ROWS, MIN_PANEL_ROWS, NULL_DRAWS, NULL_DROP_FRACTION, PANEL_DETECTORS, RF_TREES,
                         SINGLE_ARMS, TEST_SIZE, _align, _dense, _frame_key, _record_hash)
from regression import strat_labels
from selfbench import NOISE_FLOOR_MIN, SIGN_AGREEMENT, classify_repairability
from taskclean import AuditState, _apply, _prepare, _stratified_sample_index, audit_dataset, default_apply_issues

MODELS = ("ridge", "random_forest", "gradient_boosting", "neural_net")
MODEL_LABELS = {"ridge": "Ridge regression", "random_forest": "Random forest", "gradient_boosting": "Gradient boosting",
                "neural_net": "Neural net (MLP)"}
MODEL_ROLE = {"ridge": "linear: the most sensitive to outliers and wild targets",
              "random_forest": "bagged trees: steady, resists a few wild targets",
              "gradient_boosting": "boosted trees: strong on tabular data, can chase noisy targets",
              "neural_net": "small neural network: sensitive to skew and scale"}
ARM_LABEL = {"before": "Before (raw)", "default": "Default file", "aggressive": "Aggressive file",
             "duplicates": "Duplicates only", "missing_values": "Missing values only", "outliers": "Outliers only",
             "inconsistency": "Spellings only", "label_errors": "Target fixes only"}
ARM_TEXT = {"before": "Raw training rows. Blanks are filled and categories encoded, nothing is cleaned.",
            "default": "The repairs the evidence says are safe to apply without review.",
            "aggressive": "Every proposed repair except feature-anomaly repairs.",
            "duplicates": "Exact duplicate rows removed from the training rows.",
            "missing_values": "Missing cells filled (KNN or median) using the training rows only.",
            "outliers": "Extreme values replaced by the fence value, fences from the training rows.",
            "inconsistency": "Spelling variants merged into one canonical spelling.",
            "label_errors": "Suspected wrong target values replaced by the cross-validated model prediction."}


def make_model(name: str, X_tr: pd.DataFrame, seed: int) -> Pipeline:
    prep = build_preprocessor(X_tr)
    if name == "ridge":
        return Pipeline([("prep", prep), ("reg", Ridge(alpha=1.0))])
    if name == "random_forest":
        return Pipeline([("prep", prep), ("reg", RandomForestRegressor(n_estimators=RF_TREES, random_state=seed, n_jobs=-1))])
    if name == "gradient_boosting":
        return Pipeline([("prep", prep), ("dense", FunctionTransformer(_dense, accept_sparse=True)),
                         ("reg", HistGradientBoostingRegressor(max_iter=GB_ITER, learning_rate=0.1, random_state=seed))])
    if name == "neural_net":
        mlp = MLPRegressor(hidden_layer_sizes=(64,), early_stopping=True, max_iter=200, random_state=seed)
        return Pipeline([("prep", prep), ("reg", TransformedTargetRegressor(regressor=mlp, transformer=StandardScaler()))])
    raise ValueError(name)


def _typed(frame: pd.DataFrame, target: str, drop=()) -> tuple[pd.DataFrame, pd.Series]:
    """The audit's own feature preparation with a numeric target, rows with a target only."""
    X, y, *_ = _prepare(frame.reset_index(drop=True), target, "regression")
    keep = y.notna()
    X = X.drop(columns=[c for c in drop if c in X.columns])
    if X.shape[1] == 0:
        raise ValueError("no usable feature columns after leaving out identifier-like columns")
    return X.loc[keep].reset_index(drop=True), y.loc[keep].reset_index(drop=True)


def _fit_score(name, X_tr, y_tr, X_te, y_te, seed, noleak) -> dict:
    Xa, Xb = _align(X_tr, X_te)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pipe = make_model(name, Xa, seed).fit(Xa, y_tr)
    pred = pipe.predict(Xb)
    out = {"score": float(r2_score(y_te, pred)), "rmse": float(np.sqrt(mean_squared_error(y_te, pred))),
           "converged": not any(issubclass(w.category, ConvergenceWarning) for w in caught)}
    out["score_noleak"] = float(r2_score(y_te[noleak], pred[noleak])) if noleak.sum() >= 30 else float("nan")
    return out


def run_model_panel_regression(state: AuditState, seeds=DEFAULT_SEEDS, max_rows: int = MAX_PANEL_ROWS,
                               models: tuple[str, ...] = MODELS, progress: Callable[[float, str], None] | None = None) -> dict:
    say = progress or (lambda f, m: None)
    t0 = time.monotonic()
    target = state.target
    if target is None:
        return {"status": "declined", "reason": "the model panel needs a target column (no target was chosen)"}
    df = state.df.reset_index(drop=True)
    y_all = state.y.reset_index(drop=True)
    ok = y_all.notna()
    if ok.sum() < MIN_PANEL_ROWS:
        return {"status": "declined", "reason": f"the panel needs at least {MIN_PANEL_ROWS} rows with a numeric target; this file "
                f"has {int(ok.sum()):,}"}
    notes: list[str] = []
    idx = df.index[ok]
    if len(idx) > max_rows:
        idx = _stratified_sample_index(strat_labels(y_all[ok]), idx, max_rows, int(seeds[0]))
        notes.append(f"the panel's models were trained on a random sample of {len(idx):,} of {int(ok.sum()):,} rows, to keep the "
                     "run to minutes (the audit itself used every row)")
    df_p, y_p = df.loc[idx].reset_index(drop=True), y_all.loc[idx].reset_index(drop=True)
    strata = strat_labels(y_p)
    drop_cols = list(state.unmodeled_columns)
    if drop_cols:
        notes.append(f"{len(drop_cols)} identifier-like or high-cardinality text column(s) were left out of every model: "
                     + ", ".join(drop_cols[:5]) + (", ..." if len(drop_cols) > 5 else ""))
    default_issues = default_apply_issues(state) & set(SINGLE_ARMS)
    arm_issues = {"default": default_issues, "aggressive": set(AGGRESSIVE), **{a: {a} for a in SINGLE_ARMS}}
    n_models = len(models)
    unit_total = len(seeds) * (3 + 6 * n_models + NULL_DRAWS * n_models)
    done = [0.0]

    def tick(units: float, msg: str):
        done[0] += units
        say(min(0.99, done[0] / unit_total), msg)

    runs, nulls, arm_meta, leak_shares, failures, skipped = [], [], {}, [], [], {}
    for seed in seeds:
        seed = int(seed)
        try:
            tr_i, te_i = train_test_split(df_p.index, test_size=TEST_SIZE, random_state=seed, stratify=strata)
        except ValueError:
            tr_i, te_i = train_test_split(df_p.index, test_size=TEST_SIZE, random_state=seed)
        tr_raw, te_raw = df_p.loc[tr_i].reset_index(drop=True), df_p.loc[te_i].reset_index(drop=True)
        leak = _record_hash(te_raw).isin(set(_record_hash(tr_raw)))
        leak_shares.append(float(leak.mean()))
        noleak = (~leak).to_numpy()
        say(min(0.99, done[0] / unit_total), f"Seed {seed}: finding repairs on the training rows")
        st = audit_dataset(tr_raw, target, detectors=PANEL_DETECTORS, seed=seed, max_model_rows=AUDIT_MODEL_ROWS, task="regression")
        st.selfbench = state.selfbench           # Phase 23 A1: the default arm respects the calibration's column scope
        tick(3, f"Seed {seed}: repairs proposed")
        try:
            X_te, y_te = _typed(te_raw, target, drop_cols)
        except ValueError as e:
            return {"status": "declined", "reason": f"the test rows could not be prepared: {e}"}
        frames = {"before": tr_raw}
        for arm, issues in arm_issues.items():
            if not issues:
                continue
            cleaned, log = _apply(st, issues, [], respect_scope=(arm == "default"))
            applied = log[log.action == "applied"]
            m = arm_meta.setdefault(arm, {"rows_removed": [], "cells_changed": []})
            m["rows_removed"].append(len(tr_raw) - len(cleaned))
            m["cells_changed"].append(int((applied.applied_value != "row dropped").sum()))
            frames[arm] = cleaned
        arm_meta.setdefault("before", {"rows_removed": [0], "cells_changed": [0]})
        before_key = _frame_key(tr_raw)
        cache: dict[tuple, dict] = {}
        for name in models:
            if name == "neural_net" and len(tr_raw) < MIN_MLP_TRAIN_ROWS:
                skipped[name] = f"fewer than {MIN_MLP_TRAIN_ROWS} training rows"
                continue
            for arm, frame in frames.items():
                key = (name, _frame_key(frame))
                try:
                    if key not in cache:
                        X_tr, y_tr = _typed(frame, target, drop_cols)
                        cache[key] = _fit_score(name, X_tr, y_tr, X_te, y_te, seed, noleak)
                        tick(1, f"Seed {seed}: {MODEL_LABELS[name]} on {ARM_LABEL[arm].lower()}")
                    sc = cache[key]
                except Exception as e:      # noqa: BLE001  one failing model must not sink the panel
                    failures.append(f"{MODEL_LABELS[name]} / {ARM_LABEL[arm]} / seed {seed}: {type(e).__name__}: {e}")
                    continue
                runs.append({"seed": seed, "arm": arm, "model": name, "same_as_before": _frame_key(frame) == before_key, **sc})
        rng = np.random.default_rng(seed)
        for name in models:
            if name in skipped:
                continue
            base = next((r for r in runs if r["seed"] == seed and r["arm"] == "before" and r["model"] == name), None)
            if base is None:
                continue
            for d in range(NULL_DRAWS):
                try:
                    keep = rng.random(len(tr_raw)) >= NULL_DROP_FRACTION
                    X_tr, y_tr = _typed(tr_raw.loc[keep], target, drop_cols)
                    sc = _fit_score(name, X_tr, y_tr, X_te, y_te, seed, noleak)
                    nulls.append({"seed": seed, "model": name, "d": sc["score"] - base["score"]})
                except Exception as e:      # noqa: BLE001
                    failures.append(f"null experiment {MODEL_LABELS[name]} seed {seed}: {type(e).__name__}: {e}")
                tick(1, f"Seed {seed}: measuring the noise floor ({MODEL_LABELS[name]})")
    if not runs:
        return {"status": "declined", "reason": "no model could be trained on this data: " + "; ".join(failures[:3])}
    R, N = pd.DataFrame(runs), pd.DataFrame(nulls)
    floors = {}
    for name in R.model.unique():
        n = N[N.model == name] if len(N) else N
        floors[name] = max(NOISE_FLOOR_MIN, float(np.sqrt(np.mean(np.square(n.d))))) if len(n) else NOISE_FLOOR_MIN

    results, notes_conv = [], []
    for name in [m for m in models if m in set(R.model)]:
        before = R[(R.model == name) & (R.arm == "before")].set_index("seed")
        learnable = bool(before.score.mean() > LEARNABLE_MARGIN)          # the mean predictor scores R-squared = 0
        for arm in ["before"] + [a for a in ("default", "aggressive") + SINGLE_ARMS if a in set(R.arm)]:
            g = R[(R.model == name) & (R.arm == arm)].set_index("seed")
            common = g.index.intersection(before.index)
            if len(common) == 0:
                continue
            row = {"model": name, "arm": arm, "n_seeds": int(len(common)),
                   "score_mean": float(g.loc[common, "score"].mean()), "score_std": float(g.loc[common, "score"].std(ddof=0)),
                   "rmse_mean": float(g.loc[common, "rmse"].mean()),
                   "score_noleak_mean": float(g.loc[common, "score_noleak"].mean()) if g.loc[common, "score_noleak"].notna().any() else None,
                   "learnable": learnable, "noise_floor": floors[name]}
            if arm == "before":
                row.update(delta_score=0.0, delta_rmse=0.0, verdict="baseline", seeds_agree=1.0)
            else:
                d = (g.loc[common, "score"] - before.loc[common, "score"]).to_numpy()
                dr = (g.loc[common, "rmse"] - before.loc[common, "rmse"]).to_numpy()
                same = bool(g.loc[common, "same_as_before"].all())
                if same:
                    verdict, agree = "no change", 1.0
                elif not np.any(d):
                    verdict, agree = "no effect", 1.0
                elif len(common) < 2:
                    verdict, agree = "not enough runs", 1.0
                else:
                    verdict = {"positive": "improved", "negative": "worse", "weak": "within noise"}[classify_repairability(d, floors[name])]
                    agree = float(max((d > 0).sum(), (d < 0).sum()) / len(d))
                row.update(delta_score=float(d.mean()), delta_rmse=float(dr.mean()), verdict=verdict, seeds_agree=agree)
            if not bool(g.loc[common, "converged"].all()):
                notes_conv.append(f"{MODEL_LABELS[name]} did not fully converge on some runs ({ARM_LABEL[arm]})")
            results.append(row)
    arms = [{"key": a, "label": ARM_LABEL[a], "text": ARM_TEXT[a], "rows_removed": float(np.mean(arm_meta[a]["rows_removed"])),
             "cells_changed": float(np.mean(arm_meta[a]["cells_changed"]))}
            for a in ["before", "default", "aggressive", *SINGLE_ARMS] if a in arm_meta and a in set(R.arm)]
    interaction = []
    for name in sorted(set(R.model)):
        res = {r["arm"]: r for r in results if r["model"] == name}
        singles = [res[a]["delta_score"] for a in SINGLE_ARMS if a in res]
        if "aggressive" in res and singles:
            interaction.append({"model": name, "aggressive": res["aggressive"]["delta_score"], "sum_of_singles": float(sum(singles))})
    notes.extend(f"{MODEL_LABELS[k]} was skipped: {v}" for k, v in skipped.items())
    if failures:
        notes.append(f"{len(failures)} model run(s) failed and were left out (first: {failures[0]})")
    notes.extend(sorted(set(notes_conv)))
    unlearn = [MODEL_LABELS[r["model"]] for r in results if r["arm"] == "before" and not r["learnable"]]
    if unlearn:
        notes.append("the target is barely predictable from the features for " + ", ".join(unlearn)
                     + " (R-squared within 0.05 of always predicting the mean): differences are not meaningful")
    return {
        "status": "ok", "task": "regression", "metric": "R-squared on held-out rows", "seeds": [int(s) for s in seeds],
        "n_seeds": len(seeds), "rows_used": int(len(df_p)), "train_rows": int(len(tr_raw)), "test_rows": int(len(te_raw)),
        "models": [{"key": m, "label": MODEL_LABELS[m], "role": MODEL_ROLE[m], "noise_floor": floors.get(m)}
                   for m in models if m in set(R.model)],
        "arms": arms, "results": results, "interaction": interaction,
        "leakage": {"test_rows_with_exact_copy_in_train_pct": float(np.mean(leak_shares)) * 100,
                    "note": "Test rows that also appear (all columns identical) in the training rows. They inflate the "
                            "'before' score; 'score_noleak_mean' scores only the rows with no copy."},
        "default_issues": sorted(default_issues), "notes": notes, "seconds": round(time.monotonic() - t0, 1),
        "settings": {"test_size": TEST_SIZE, "rf_trees": RF_TREES, "gb_iterations": GB_ITER, "mlp_hidden": 64,
                     "null_draws_per_seed": NULL_DRAWS, "null_drop_fraction": NULL_DROP_FRACTION,
                     "sign_agreement": SIGN_AGREEMENT, "noise_floor_min": NOISE_FLOOR_MIN},
        "caveat": ("This compares models trained on your raw rows with models trained on the repaired rows, scored on the same "
                   "untouched held-out rows, which are as dirty as your file. A gain means the repaired training rows predicted "
                   "the real rows better; it does not prove each repair was correct, and a loss can mean the held-out targets "
                   "are noisy. Changes smaller than the noise floor are labelled 'within noise'."),
    }


def panel_rows_regression(panel: dict) -> pd.DataFrame:
    if panel.get("status") != "ok":
        return pd.DataFrame([{"status": panel.get("status"), "reason": panel.get("reason")}])
    cols = ["model", "arm", "n_seeds", "score_mean", "score_std", "delta_score", "rmse_mean", "delta_rmse", "verdict",
            "seeds_agree", "noise_floor", "score_noleak_mean", "learnable"]
    out = pd.DataFrame(panel["results"])
    for c in cols:
        if c not in out:
            out[c] = np.nan
    return out[cols]
