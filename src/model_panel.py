"""
Model panel: does cleaning this dataset change how well models learn from it? (Phase 19, Stage U; docs/PHASE19_SPEC.md)

Four model families (logistic regression, random forest, gradient boosting, a small neural net) are trained on the user's
own file twice with the same split and seed: once on the raw training rows ("before", only the survival preprocessing every
phase uses: fill blanks, one-hot, scale) and once on the training rows after repair ("after"). The rules that keep the
comparison fair:

  * the split is made FIRST, on the raw rows; test rows are never edited, dropped or read by any cleaner;
  * the audit that proposes the repairs runs on the TRAINING rows only (so imputed values, outlier fences, duplicate
    removal and spelling merges are all learned from training data);
  * arms: before; the default file (what the evidence-based policy applies without review); the aggressive file (every
    proposed repair except feature-anomaly repairs); and each repair type on its own;
  * every delta is read against a noise floor measured per model by a null experiment (retrain on a random 90% of the
    training rows). A delta is "improved" / "worse" only if it clears the floor AND >= 80% of seeds agree on its sign
    (Phase 10's rule); anything else is "within noise" and no percentage is shown for it;
  * the test rows that have an exact copy in the training rows are counted, and scores are also given without them
    (copies across the split inflate the "before" score and make de-duplication look like a loss).

What it does NOT show: that a repair was correct. The test set is as dirty as the user's file, so a gain means "training on
the cleaned rows predicted the real, dirty rows better". It is information only: it never changes which repairs are applied.
"""
from __future__ import annotations

import time
import warnings
from typing import Callable

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from baseline import build_preprocessor
from selfbench import MIN_CLASS_ROWS, NOISE_FLOOR_MIN, SIGN_AGREEMENT, _auc, classify_repairability
from taskclean import (AuditState, _apply, _prepare, _stratified_sample_index, audit_dataset, default_apply_issues)

MODELS = ("logistic_regression", "random_forest", "gradient_boosting", "neural_net")
MODEL_LABELS = {"logistic_regression": "Logistic regression", "random_forest": "Random forest",
                "gradient_boosting": "Gradient boosting", "neural_net": "Neural net (MLP)"}
MODEL_ROLE = {"logistic_regression": "linear: the most sensitive to outliers and scale",
              "random_forest": "bagged trees: steady, resists noise",
              "gradient_boosting": "boosted trees: strong on tabular data, can chase noisy labels",
              "neural_net": "small neural network: sensitive to skew and scale"}

# --- fixed parameters (docs/PHASE19_SPEC.md) -----------------------------------------------------------------------
DEFAULT_SEEDS = (42, 43, 44)           # 3 seeds is the quick default; 5 (42-46) is the thorough run
TEST_SIZE = 0.20
MAX_PANEL_ROWS = 8000                  # stratified cap for the panel's fits (the audit itself still sees every row)
MIN_PANEL_ROWS = 600
MIN_MLP_TRAIN_ROWS = 500
RF_TREES = 100                         # the research harness uses 300; 100 keeps the panel to minutes (stated on the page)
GB_ITER = 150
NULL_DRAWS = 3                         # null-experiment draws per seed (the research harness uses 4)
NULL_DROP_FRACTION = 0.10
LEARNABLE_MARGIN = 0.05                # before-score must beat the majority-class level by this, or deltas are meaningless
AUDIT_MODEL_ROWS = 4000                # label-noise detector cap inside each training-split audit
PANEL_DETECTORS = {"missing_values", "duplicates", "outliers", "inconsistency", "label_noise"}
SINGLE_ARMS = ("duplicates", "missing_values", "outliers", "inconsistency", "label_errors")
AGGRESSIVE = frozenset(SINGLE_ARMS)

ARM_LABEL = {"before": "Before (raw)", "default": "Default file", "aggressive": "Aggressive file",
             "duplicates": "Duplicates only", "missing_values": "Missing values only", "outliers": "Outliers only",
             "inconsistency": "Spellings only", "label_errors": "Label fixes only"}
ARM_TEXT = {"before": "Raw training rows. Blanks are filled and categories encoded, nothing is cleaned.",
            "default": "The repairs the evidence says are safe to apply without review.",
            "aggressive": "Every proposed repair except feature-anomaly repairs.",
            "duplicates": "Exact duplicate rows removed from the training rows.",
            "missing_values": "Missing cells filled (KNN or median) using the training rows only.",
            "outliers": "Extreme values replaced by the fence value, fences from the training rows.",
            "inconsistency": "Spelling variants merged into one canonical spelling.",
            "label_errors": "Suspected wrong labels corrected by the label detector."}


# ---------------------------------------------------------------------------------------------------------------
# data preparation
# ---------------------------------------------------------------------------------------------------------------
def _dense(X):
    return X.toarray() if sparse.issparse(X) else np.asarray(X)


def make_model(name: str, X_tr: pd.DataFrame, seed: int) -> Pipeline:
    prep = build_preprocessor(X_tr)
    if name == "logistic_regression":
        return Pipeline([("prep", prep), ("clf", LogisticRegression(max_iter=1000, random_state=seed))])
    if name == "random_forest":
        return Pipeline([("prep", prep), ("clf", RandomForestClassifier(n_estimators=RF_TREES, random_state=seed, n_jobs=-1))])
    if name == "gradient_boosting":
        return Pipeline([("prep", prep), ("dense", FunctionTransformer(_dense, accept_sparse=True)),
                         ("clf", HistGradientBoostingClassifier(max_iter=GB_ITER, learning_rate=0.1, random_state=seed))])
    if name == "neural_net":
        return Pipeline([("prep", prep), ("clf", MLPClassifier(hidden_layer_sizes=(64,), early_stopping=True,
                                                               max_iter=200, random_state=seed))])
    raise ValueError(name)


def _typed(frame: pd.DataFrame, target: str, drop=()) -> tuple[pd.DataFrame, pd.Series]:
    """The audit's own feature preparation (placeholders -> missing, numbers stored as text -> numbers), labeled rows only.
    `drop` lists identifier-like / very high-cardinality text columns, which every model leaves out (a one-hot of an ID
    column only lets a model memorise rows)."""
    X, y, *_ = _prepare(frame.reset_index(drop=True), target)
    keep = y.notna()
    X = X.drop(columns=[c for c in drop if c in X.columns])
    if X.shape[1] == 0:
        raise ValueError("no usable feature columns after leaving out identifier-like columns")
    return X.loc[keep].reset_index(drop=True), y.loc[keep].reset_index(drop=True)


def _is_num(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)


def _align(Xa: pd.DataFrame, Xb: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train and test are prepared separately; keep the columns both have and give both the same kind (numeric or text)."""
    common = [c for c in Xa.columns if c in Xb.columns]
    Xa, Xb = Xa[common].copy(), Xb[common].copy()
    for c in common:
        if _is_num(Xa[c]) != _is_num(Xb[c]):
            Xa[c] = Xa[c].astype(object).where(Xa[c].notna(), np.nan).map(lambda v: v if pd.isna(v) else str(v)).astype("category")
            Xb[c] = Xb[c].astype(object).where(Xb[c].notna(), np.nan).map(lambda v: v if pd.isna(v) else str(v)).astype("category")
    return Xa, Xb


def _record_hash(df: pd.DataFrame) -> pd.Series:
    return pd.util.hash_pandas_object(df.astype(str), index=False)


def _frame_key(df: pd.DataFrame) -> tuple:
    return (df.shape, int(pd.util.hash_pandas_object(df.astype(str), index=False).sum() % (1 << 61)))


def _chance_macro_f1(y: pd.Series) -> float:
    p = float(y.value_counts(normalize=True).iloc[0])
    return 2 * p / (1 + p) / y.nunique()


# ---------------------------------------------------------------------------------------------------------------
# one fit
# ---------------------------------------------------------------------------------------------------------------
def _fit_score(name, X_tr, y_tr, X_te, y_te, seed, noleak) -> dict:
    Xa, Xb = _align(X_tr, X_te)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pipe = make_model(name, Xa, seed).fit(Xa, y_tr)
    pred = pipe.predict(Xb)
    out = {"f1": float(f1_score(y_te, pred, average="macro", zero_division=0)),
           "acc": float(accuracy_score(y_te, pred)), "auc": _auc(pipe, Xb, y_te),
           "converged": not any(issubclass(w.category, ConvergenceWarning) for w in caught)}
    out["f1_noleak"] = (float(f1_score(y_te[noleak], pred[noleak], average="macro", zero_division=0))
                        if noleak.sum() >= 30 else float("nan"))
    return out


# ---------------------------------------------------------------------------------------------------------------
# the panel
# ---------------------------------------------------------------------------------------------------------------
def run_model_panel(state: AuditState, seeds=DEFAULT_SEEDS, max_rows: int = MAX_PANEL_ROWS,
                    models: tuple[str, ...] = MODELS, progress: Callable[[float, str], None] | None = None) -> dict:
    say = progress or (lambda f, m: None)
    t0 = time.monotonic()
    target = state.target
    if target is None:
        return {"status": "declined", "reason": "the model panel needs a target column (no target was chosen)"}
    df = state.df.reset_index(drop=True)
    y_all = state.y.reset_index(drop=True)
    labeled = y_all.notna()
    counts = y_all[labeled].value_counts()
    rare = counts[counts < MIN_CLASS_ROWS].index
    notes: list[str] = []
    if len(rare):
        notes.append(f"{len(rare)} class(es) with fewer than {MIN_CLASS_ROWS} rows were left out of the panel: "
                     + ", ".join(map(str, list(rare)[:5])))
    ok = labeled & ~y_all.isin(rare)
    if ok.sum() < MIN_PANEL_ROWS or y_all[ok].nunique() < 2:
        return {"status": "declined", "reason": f"the panel needs at least {MIN_PANEL_ROWS} labelled rows in two or more "
                f"classes; this file has {int(ok.sum()):,} usable rows"}
    idx = df.index[ok]
    if len(idx) > max_rows:
        idx = _stratified_sample_index(y_all[ok], idx, max_rows, int(seeds[0]))
        notes.append(f"the panel's models were trained on a stratified sample of {len(idx):,} of {int(ok.sum()):,} labelled "
                     "rows, to keep the run to minutes (the audit itself used every row)")
    df_p, y_p = df.loc[idx].reset_index(drop=True), y_all.loc[idx].reset_index(drop=True)
    chance = _chance_macro_f1(y_p)
    drop_cols = list(state.unmodeled_columns)
    if drop_cols:
        notes.append(f"{len(drop_cols)} identifier-like or high-cardinality text column(s) were left out of every model: "
                     + ", ".join(drop_cols[:5]) + (", ..." if len(drop_cols) > 5 else ""))

    default_issues = (default_apply_issues(state) & set(SINGLE_ARMS))
    arm_issues = {"default": default_issues, "aggressive": set(AGGRESSIVE)}
    for a in SINGLE_ARMS:
        arm_issues[a] = {a}
    n_models = len(models)
    # work estimate for the progress bar: per seed one audit (weighted 3), <= 8 arms x models, null draws x models
    unit_total = len(seeds) * (3 + 6 * n_models + NULL_DRAWS * n_models)
    done = [0.0]

    def tick(units: float, msg: str):
        done[0] += units
        say(min(0.99, done[0] / unit_total), msg)

    runs: list[dict] = []                    # one row per (seed, arm, model)
    nulls: list[dict] = []                   # one row per (seed, model, draw)
    arm_meta: dict[str, dict] = {}
    leak_shares, failures, skipped = [], [], {}

    for seed in seeds:
        seed = int(seed)
        try:
            tr_i, te_i = train_test_split(df_p.index, test_size=TEST_SIZE, random_state=seed, stratify=y_p)
        except ValueError:
            tr_i, te_i = train_test_split(df_p.index, test_size=TEST_SIZE, random_state=seed)
        tr_raw, te_raw = df_p.loc[tr_i].reset_index(drop=True), df_p.loc[te_i].reset_index(drop=True)
        leak = _record_hash(te_raw).isin(set(_record_hash(tr_raw)))
        leak_shares.append(float(leak.mean()))
        noleak = (~leak).to_numpy()

        say(min(0.99, done[0] / unit_total), f"Seed {seed}: finding repairs on the training rows")
        st = audit_dataset(tr_raw, target, detectors=PANEL_DETECTORS, seed=seed, max_model_rows=AUDIT_MODEL_ROWS)
        tick(3, f"Seed {seed}: repairs proposed")
        try:
            X_te, y_te = _typed(te_raw, target, drop_cols)
        except ValueError as e:
            return {"status": "declined", "reason": f"the test rows could not be prepared: {e}"}
        assert len(y_te) == len(noleak), "every test row is labelled, so the test frame keeps all its rows"

        frames = {"before": tr_raw}
        for arm, issues in arm_issues.items():
            if not issues:
                continue
            cleaned, log = _apply(st, issues, [])
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

        # noise floor: retrain on a random 90% of the training rows (a perturbation as large as a repair's)
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
                    nulls.append({"seed": seed, "model": name, "d_f1": sc["f1"] - base["f1"],
                                  "d_auc": sc["auc"] - base["auc"]})
                except Exception as e:      # noqa: BLE001
                    failures.append(f"null experiment {MODEL_LABELS[name]} seed {seed}: {type(e).__name__}: {e}")
                tick(1, f"Seed {seed}: measuring the noise floor ({MODEL_LABELS[name]})")

    if not runs:
        return {"status": "declined", "reason": "no model could be trained on this data: " + "; ".join(failures[:3])}
    R, N = pd.DataFrame(runs), pd.DataFrame(nulls)
    floors, floors_auc = {}, {}
    for name in R.model.unique():
        n = N[N.model == name]
        floors[name] = max(NOISE_FLOOR_MIN, float(np.sqrt(np.mean(np.square(n.d_f1))))) if len(n) else NOISE_FLOOR_MIN
        a = n.d_auc.dropna()
        floors_auc[name] = max(NOISE_FLOOR_MIN, float(np.sqrt(np.mean(np.square(a))))) if len(a) else NOISE_FLOOR_MIN

    results, notes_conv = [], []
    for name in [m for m in models if m in set(R.model)]:
        before = R[(R.model == name) & (R.arm == "before")].set_index("seed")
        learnable = bool(before.f1.mean() > chance + LEARNABLE_MARGIN)
        for arm in ["before"] + [a for a in ("default", "aggressive") + SINGLE_ARMS if a in set(R.arm)]:
            g = R[(R.model == name) & (R.arm == arm)].set_index("seed")
            common = g.index.intersection(before.index)
            if len(common) == 0:
                continue
            row = {"model": name, "arm": arm, "n_seeds": int(len(common)),
                   "f1_mean": float(g.loc[common, "f1"].mean()), "f1_std": float(g.loc[common, "f1"].std(ddof=0)),
                   "auc_mean": float(g.loc[common, "auc"].mean()), "acc_mean": float(g.loc[common, "acc"].mean()),
                   "f1_noleak_mean": float(g.loc[common, "f1_noleak"].mean()) if g.loc[common, "f1_noleak"].notna().any() else None,
                   "learnable": learnable, "noise_floor": floors[name]}
            if arm == "before":
                row.update(delta_f1=0.0, delta_auc=0.0, verdict="baseline", delta_pct=None, seeds_agree=1.0)
            else:
                d = (g.loc[common, "f1"] - before.loc[common, "f1"]).to_numpy()
                da = (g.loc[common, "auc"] - before.loc[common, "auc"]).to_numpy()
                same = bool(g.loc[common, "same_as_before"].all())
                if same:
                    verdict, agree = "no change", 1.0                 # nothing was repaired: the training rows are the raw rows
                elif not np.any(d):
                    verdict, agree = "no effect", 1.0                 # cells changed but the trained model came out identical
                elif len(common) < 2:
                    verdict, agree = "not enough runs", 1.0
                else:
                    status = classify_repairability(d, floors[name])
                    verdict = {"positive": "improved", "negative": "worse", "weak": "within noise"}[status]
                    agree = float(max((d > 0).sum(), (d < 0).sum()) / len(d))
                a_status = classify_repairability(da, floors_auc[name]) if np.isfinite(da).all() and len(da) >= 2 and np.any(da) else "weak"
                row.update(delta_f1=float(d.mean()), delta_auc=float(da.mean()) if np.isfinite(da).all() else None,
                           verdict=verdict, seeds_agree=agree, delta_pct=None,
                           auc_verdict={"positive": "improved", "negative": "worse", "weak": "within noise"}[a_status])
                if verdict in ("improved", "worse"):
                    row["delta_pct"] = 100 * float(d.mean()) / float(before.loc[common, "f1"].mean())
            if not bool(g.loc[common, "converged"].all()):
                notes_conv.append(f"{MODEL_LABELS[name]} did not fully converge on some runs ({ARM_LABEL[arm]})")
            results.append(row)

    arms = []
    for arm in ["before", "default", "aggressive", *SINGLE_ARMS]:
        if arm in arm_meta and arm in set(R.arm):
            m = arm_meta[arm]
            arms.append({"key": arm, "label": ARM_LABEL[arm], "text": ARM_TEXT[arm],
                         "rows_removed": float(np.mean(m["rows_removed"])), "cells_changed": float(np.mean(m["cells_changed"]))})

    # does the all-repairs result equal the sum of the single repairs? (it usually does not)
    interaction = []
    for name in sorted(set(R.model)):
        res = {r["arm"]: r for r in results if r["model"] == name}
        singles = [res[a]["delta_f1"] for a in SINGLE_ARMS if a in res]
        if "aggressive" in res and singles:
            interaction.append({"model": name, "aggressive": res["aggressive"]["delta_f1"], "sum_of_singles": float(sum(singles))})

    if skipped:
        notes.extend(f"{MODEL_LABELS[k]} was skipped: {v}" for k, v in skipped.items())
    if failures:
        notes.append(f"{len(failures)} model run(s) failed and were left out (first: {failures[0]})")
    notes.extend(sorted(set(notes_conv)))
    unlearn = [MODEL_LABELS[r["model"]] for r in results if r["arm"] == "before" and not r["learnable"]]
    if unlearn:
        notes.append("the target is barely predictable from the features for " + ", ".join(unlearn)
                     + " (macro-F1 within 0.05 of always guessing the biggest class): differences are not meaningful")
    ns = len(seeds)
    return {
        "status": "ok", "metric": "macro-F1 on held-out rows", "seeds": [int(s) for s in seeds], "n_seeds": ns,
        "rows_used": int(len(df_p)), "train_rows": int(len(tr_raw)), "test_rows": int(len(te_raw)),
        "n_classes": int(y_p.nunique()), "chance_f1": chance,
        "models": [{"key": m, "label": MODEL_LABELS[m], "role": MODEL_ROLE[m], "noise_floor": floors.get(m),
                    "noise_floor_auc": floors_auc.get(m)} for m in models if m in set(R.model)],
        "arms": arms, "results": results, "interaction": interaction,
        "leakage": {"test_rows_with_exact_copy_in_train_pct": float(np.mean(leak_shares)) * 100,
                    "note": "Test rows that also appear (all columns identical) in the training rows. They inflate the "
                            "'before' score; 'f1_noleak_mean' scores only the rows with no copy."},
        "default_issues": sorted(default_issues), "notes": notes, "seconds": round(time.monotonic() - t0, 1),
        "settings": {"test_size": TEST_SIZE, "rf_trees": RF_TREES, "gb_iterations": GB_ITER, "mlp_hidden": 64,
                     "null_draws_per_seed": NULL_DRAWS, "null_drop_fraction": NULL_DROP_FRACTION,
                     "sign_agreement": SIGN_AGREEMENT, "noise_floor_min": NOISE_FLOOR_MIN},
        "caveat": ("This compares models trained on your raw rows with models trained on the repaired rows, scored on the same "
                   "untouched held-out rows, which are as dirty as your file. A gain means the repaired training rows predicted "
                   "the real rows better; it does not prove each repair was correct, and a loss can mean the held-out labels "
                   "are noisy. Changes smaller than the noise floor are labelled 'within noise' and carry no percentage."),
    }


def panel_rows(panel: dict) -> pd.DataFrame:
    """Flat table for download: one row per model and arm."""
    if panel.get("status") != "ok":
        return pd.DataFrame([{"status": panel.get("status"), "reason": panel.get("reason")}])
    cols = ["model", "arm", "n_seeds", "f1_mean", "f1_std", "delta_f1", "delta_pct", "verdict", "seeds_agree", "noise_floor",
            "auc_mean", "delta_auc", "auc_verdict", "acc_mean", "f1_noleak_mean", "learnable"]
    out = pd.DataFrame(panel["results"])
    for c in cols:
        if c not in out:
            out[c] = np.nan
    return out[cols]
