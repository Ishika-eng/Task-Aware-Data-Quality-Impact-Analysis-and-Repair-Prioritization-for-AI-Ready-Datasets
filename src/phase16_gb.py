"""
Phase 16 harness: does the project's damage / repair evidence depend on the downstream model?
Method, criteria and consequences: docs/PHASE16_SPEC.md (written before this module existed).

One variable: the classifier. Everything else (Adult clean frame, 70/15/15 split with seed 42, corruptors, RATES,
preprocessing, seeds, detect-and-repair) is the frozen Phase 7 / Phase 9 code path, imported and called unchanged. The
loops are re-implemented here with the model injected as a factory (the frozen modules hard-code the Random Forest).
Stage 0 proves the re-implementation is faithful by reproducing the frozen RF numbers before any GB number is believed.

Writes only results/phase16_*. It never touches results/training_log.csv or any frozen file.

    .venv/bin/python src/phase16_gb.py --stage 0     harness check (G0)
    .venv/bin/python src/phase16_gb.py --stage 1     hyperparameter selection on clean data + clean scores (G1)
    .venv/bin/python src/phase16_gb.py --stage 2     damage per issue/rate/seed + null-experiment noise floor (G2, G3)
    .venv/bin/python src/phase16_gb.py --stage 3     capacity arms, damage only (G8)
    .venv/bin/python src/phase16_gb.py --stage 4     repair: dirty / repaired / oracle, status, risk, evidence table (G4-G6)
    .venv/bin/python src/phase16_gb.py --stage 5     priority ranking on the GB evidence table (G7); seconds, no model fits
    .venv/bin/python src/phase16_gb.py --stage 6     Bank Marketing replication of the Phase 13 ground-truth protocol (G9)
Stage 7 is not built yet. Every stage after 1 refuses to run without results/phase16_gb_config.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from baseline import POS_LABEL, build_models, build_preprocessor, evaluate
from corruption import CORRUPTORS, corrupt_with_id
from data import load_adult_clean, three_way_split
from phase7_impact import SPLIT_SEED, build_rf_pipeline
from phase8_oracle_repair import oracle_repair
from phase9_repair_engine import detect_and_repair, grade_repair

warnings.filterwarnings("ignore")

ROOT = os.path.join(os.path.dirname(__file__), "..")
RESULTS_DIR = os.path.join(ROOT, "results")
CONFIG_PATH = os.path.join(RESULTS_DIR, "phase16_gb_config.json")

SEEDS = [42, 43, 44, 45, 46]

# --- fixed in docs/PHASE16_SPEC.md section B before any dirty run ---------------------------------------------------
GB_FIXED = {"early_stopping": False, "min_samples_leaf": 20, "l2_regularization": 0.0, "max_leaf_nodes": 31}
GRID_MAX_ITER = [100, 200, 400]
GRID_LEARNING_RATE = [0.05, 0.1]
CAPACITY_ARMS = {                                   # Stage 3; independent of what Stage 1 selects
    "shallow": {"max_leaf_nodes": 8, "max_iter": 100, "learning_rate": 0.1},
    "deep": {"max_leaf_nodes": 63, "max_iter": 400, "learning_rate": 0.1},
}
G0_TOLERANCE = 1e-9
G1_MARGIN = 0.01                                    # GB clean test positive-F1 must be >= RF clean minus this
RF_CLEAN_F1_RECORDED = 0.6804                       # results/phase2_clean_baseline.csv (random_forest row)


# ---------------------------------------------------------------------------------------------------------------
# Models: one factory per model, all with the same signature (X_train, seed) -> unfitted Pipeline
# ---------------------------------------------------------------------------------------------------------------
def _dense(X):
    return X.toarray() if sparse.issparse(X) else np.asarray(X)


def build_gb_pipeline(X: pd.DataFrame, seed: int, **params) -> Pipeline:
    """Same preprocessing as every earlier phase; a densifying step (HistGradientBoostingClassifier rejects sparse
    input; no value changes); then the booster. `params` override GB_FIXED (max_iter, learning_rate, ...)."""
    cfg = {**GB_FIXED, **params}
    return Pipeline([
        ("prep", build_preprocessor(X)),
        ("dense", FunctionTransformer(_dense, accept_sparse=True)),
        ("clf", HistGradientBoostingClassifier(random_state=seed, **cfg)),
    ])


def gb_factory(params: dict):
    return lambda X, seed: build_gb_pipeline(X, seed, **params)


def rf_factory(X, seed):
    return build_rf_pipeline(X, seed=seed)


def scores(pipe: Pipeline, X_te: pd.DataFrame, y_te: pd.Series, pos_label=POS_LABEL) -> dict:
    """Positive-class F1 (the research metric; identical to baseline.evaluate's `f1`), macro-F1, ROC-AUC.
    `pos_label` defaults to Adult's '>50K'; Stage 6 passes Bank Marketing's minority class."""
    pred = pipe.predict(X_te)
    proba = pipe.predict_proba(X_te)[:, list(pipe.classes_).index(pos_label)]
    return {"posf1": float(f1_score(y_te, pred, pos_label=pos_label)),
            "macrof1": float(f1_score(y_te, pred, average="macro")),
            "auc": float(roc_auc_score((y_te == pos_label).astype(int), proba))}


def load_split():
    X, y = load_adult_clean()
    return three_way_split(X, y, seed=SPLIT_SEED)          # X_train, X_val, X_test, y_train, y_val, y_test


# ---------------------------------------------------------------------------------------------------------------
# The two experiment cells, model injected (Phase 7 and Phase 9 analogues)
# ---------------------------------------------------------------------------------------------------------------
def damage_cell(factory, X_train, y_train, X_test, y_test, error_type: str, rate: float, seed: int,
                dirty=None) -> dict:
    """Phase 7: inject ONLY this error into the clean training data, train, score on the untouched test set.
    Same call as phase7_impact.run_experiment (CORRUPTORS, not corrupt_with_id). `dirty` = an already-built
    (X_dirty, y_dirty), so several arms can share one corruption draw."""
    X_dirty, y_dirty = dirty if dirty is not None else CORRUPTORS[error_type](X_train, y_train, rate=rate, seed=seed)[:2]
    t0 = time.monotonic()
    pipe = factory(X_dirty, seed)
    pipe.fit(X_dirty, y_dirty)
    s = scores(pipe, X_test, y_test)
    return {"error_type": error_type, "rate": rate, "seed": seed, "fit_seconds": round(time.monotonic() - t0, 2),
            **{f"dirty_{k}": v for k, v in s.items()}, "dirty_f1": s["posf1"]}


def make_repair_bundle(error_type: str, X_train, y_train, rate: float, seed: int) -> dict:
    """Everything in a Phase 9 cell that does NOT depend on the downstream model, so it is computed once and reused
    by every arm: the dirty frame, the repaired frame (detectors included), the oracle-restored frame, the grading."""
    X_dirty, y_dirty, log_df = corrupt_with_id(error_type, X_train, y_train, rate=rate, seed=seed)
    X_rep, y_rep, repaired_set, n_flagged_only = detect_and_repair(error_type, X_dirty, y_dirty, seed=seed)
    X_orc, y_orc = oracle_repair(error_type, X_dirty, y_dirty, log_df)
    correctness = grade_repair(error_type, log_df, repaired_set, X_dirty, X_rep)
    if error_type == "label_errors" and correctness["n_checkable"] == 0 and correctness["tp"] > 0:
        tp_rows = set(log_df["row_id"]) & repaired_set
        lookup = log_df.set_index("row_id")["original_value"]
        correctness["restored_correctly"] = sum(1 for r in tp_rows if str(lookup.loc[r]) == str(y_rep.loc[r]))
        correctness["n_checkable"] = len(tp_rows)
    return {"error_type": error_type, "rate": rate, "seed": seed, "dirty": (X_dirty, y_dirty),
            "repaired": (X_rep, y_rep), "oracle": (X_orc, y_orc), "correctness": correctness,
            "n_flagged_not_repaired": n_flagged_only}


def repair_cell(factory, bundle: dict, X_test, y_test, clean_scores: dict) -> dict:
    """Phase 9: score dirty / repaired / oracle with the injected model. `clean_scores` is this seed's clean baseline."""
    seed = bundle["seed"]
    out = {"error_type": bundle["error_type"], "rate": bundle["rate"], "seed": seed}
    for arm in ("dirty", "repaired", "oracle"):
        Xa, ya = bundle[arm]
        pipe = factory(Xa, seed)
        pipe.fit(Xa, ya)
        for k, v in scores(pipe, X_test, y_test).items():
            out[f"{arm}_{k}"] = v
    for k, v in clean_scores.items():
        out[f"clean_{k}"] = v
    for m in ("posf1", "macrof1", "auc"):
        out[f"damage_{m}"] = out[f"clean_{m}"] - out[f"dirty_{m}"]
        out[f"recovery_{m}"] = out[f"repaired_{m}"] - out[f"dirty_{m}"]
    # the frozen Phase 9 column names for the primary metric, so the two result files line up
    out.update(clean_f1=out["clean_posf1"], dirty_f1=out["dirty_posf1"], repaired_f1=out["repaired_posf1"],
               oracle_f1=out["oracle_posf1"])
    c = bundle["correctness"]
    out.update(repair_precision=c["precision"], repair_recall=c["recall"], tp=c["tp"], fp=c["fp"], fn=c["fn"],
               restored_correctly=c["restored_correctly"], n_checkable=c["n_checkable"],
               n_flagged_not_repaired=bundle["n_flagged_not_repaired"])
    return out


# ---------------------------------------------------------------------------------------------------------------
# Hyperparameter selection (spec section B): clean data only, validation split only, rule fixed in advance
# ---------------------------------------------------------------------------------------------------------------
def select_config(candidates: list[dict]) -> dict:
    """Highest clean-validation positive-F1; ties: smaller max_iter, then smaller learning_rate."""
    return min(candidates, key=lambda c: (-round(c["val_posf1"], 12), c["max_iter"], c["learning_rate"]))


def require_config() -> dict:
    """Stages after Stage 1 refuse to run until the configuration was chosen and recorded."""
    if not os.path.exists(CONFIG_PATH):
        raise SystemExit("results/phase16_gb_config.json is missing: run --stage 1 first. The configuration must be "
                         "chosen on clean data before any dirty run (docs/PHASE16_SPEC.md, section B).")
    with open(CONFIG_PATH) as f:
        return json.load(f)


def selected_params(config: dict) -> dict:
    s = config["selected"]
    return {"max_iter": s["max_iter"], "learning_rate": s["learning_rate"]}


# ---------------------------------------------------------------------------------------------------------------
# Stage 0: harness check (G0)
# ---------------------------------------------------------------------------------------------------------------
G0_DAMAGE_CELLS = [("label_errors", 0.10), ("duplicates", 0.05), ("outliers", 0.20)]
G0_REPAIR_CELL = ("feature_corruption", 0.10)
G0_SEED = 42


def stage0() -> pd.DataFrame:
    X_tr, X_val, X_te, y_tr, y_val, y_te = load_split()
    f7 = pd.read_csv(os.path.join(RESULTS_DIR, "phase7_impact_raw_full.csv"))
    f9 = pd.read_csv(os.path.join(RESULTS_DIR, "phase9_repair_raw_full.csv"))
    rows = []

    def record(check, cell, frozen, fresh):
        diff = abs(frozen - fresh)
        rows.append({"check": check, "cell": cell, "frozen": frozen, "harness": fresh, "abs_diff": diff,
                     "tolerance": G0_TOLERANCE, "pass": bool(diff <= G0_TOLERANCE)})
        print(f"  {check:22s} {cell:28s} frozen={frozen:.12f} harness={fresh:.12f} diff={diff:.2e} "
              f"{'PASS' if diff <= G0_TOLERANCE else 'FAIL'}")

    print("Stage 0a: Random Forest through the new harness vs the frozen Phase 7 CSV (seed 42)")
    for et, rate in G0_DAMAGE_CELLS:
        frozen = float(f7[(f7.error_type == et) & (f7.rate == rate) & (f7.seed == G0_SEED)].dirty_f1.iloc[0])
        got = damage_cell(rf_factory, X_tr, y_tr, X_te, y_te, et, rate, G0_SEED)["dirty_f1"]
        record("phase7 dirty_f1", f"{et} {rate:.0%}", frozen, got)

    print("Stage 0b: Random Forest through the new harness vs the frozen Phase 9 CSV (seed 42)")
    et, rate = G0_REPAIR_CELL
    clean_pipe = rf_factory(X_tr, G0_SEED).fit(X_tr, y_tr)
    clean = scores(clean_pipe, X_te, y_te)
    bundle = make_repair_bundle(et, X_tr, y_tr, rate, G0_SEED)
    got = repair_cell(rf_factory, bundle, X_te, y_te, clean)
    ref = f9[(f9.error_type == et) & (f9.rate == rate) & (f9.seed == G0_SEED)].iloc[0]
    record("phase9 repaired_f1", f"{et} {rate:.0%}", float(ref.repaired_f1), got["repaired_f1"])
    for extra in ("clean_f1", "dirty_f1", "oracle_f1"):          # informative: the spec gates only repaired_f1
        diff = abs(float(ref[extra]) - got[extra])
        print(f"    (informative) {extra:10s} frozen={float(ref[extra]):.12f} harness={got[extra]:.12f} diff={diff:.2e}")
    rows[-1]["informative_diffs"] = json.dumps({e: abs(float(ref[e]) - got[e]) for e in ("clean_f1", "dirty_f1", "oracle_f1")})

    print("Stage 0c: gradient boosting is deterministic in random_state on clean data (two seeds, one fixed config)")
    p = {"max_iter": GRID_MAX_ITER[0], "learning_rate": GRID_LEARNING_RATE[-1]}
    a = build_gb_pipeline(X_tr, 1, **p).fit(X_tr, y_tr)
    b = build_gb_pipeline(X_tr, 2, **p).fit(X_tr, y_tr)
    identical = bool((a.predict(X_te) == b.predict(X_te)).all())
    pa, pb = a.predict_proba(X_te), b.predict_proba(X_te)
    maxdiff = float(np.abs(pa - pb).max())
    rows.append({"check": "gb determinism", "cell": f"max_iter={p['max_iter']} lr={p['learning_rate']}",
                 "frozen": 0.0, "harness": maxdiff, "abs_diff": maxdiff, "tolerance": 0.0, "pass": identical})
    print(f"  identical predictions across random_state 1 vs 2: {identical}  (max |proba diff| = {maxdiff:.2e})")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, "phase16_gb_harness_check.csv"), index=False)
    gate = bool(df["pass"].all())
    print(f"\nG0 {'PASS' if gate else 'FAIL'}: {int(df['pass'].sum())}/{len(df)} checks pass. "
          f"Saved results/phase16_gb_harness_check.csv")
    return df


# ---------------------------------------------------------------------------------------------------------------
# Stage 1: selection on clean data, then clean scores for context (G1)
# ---------------------------------------------------------------------------------------------------------------
def stage1() -> dict:
    X_tr, X_val, X_te, y_tr, y_val, y_te = load_split()
    print("Stage 1a: six candidates, fitted on CLEAN training data, scored on the CLEAN validation split only")
    candidates = []
    for max_iter in GRID_MAX_ITER:
        for lr in GRID_LEARNING_RATE:
            t0 = time.monotonic()
            pipe = build_gb_pipeline(X_tr, SPLIT_SEED, max_iter=max_iter, learning_rate=lr).fit(X_tr, y_tr)
            val = scores(pipe, X_val, y_val)
            candidates.append({"max_iter": max_iter, "learning_rate": lr, "val_posf1": val["posf1"],
                               "val_macrof1": val["macrof1"], "val_auc": val["auc"],
                               "fit_seconds": round(time.monotonic() - t0, 2)})
            print(f"  max_iter={max_iter:3d} lr={lr:<5} val posF1={val['posf1']:.4f}  macroF1={val['macrof1']:.4f}  "
                  f"AUC={val['auc']:.4f}  ({candidates[-1]['fit_seconds']}s)")
    chosen = select_config(candidates)
    config = {"selected": {k: chosen[k] for k in ("max_iter", "learning_rate")},
              "fixed": GB_FIXED, "selection_rule": "highest clean-validation positive-class F1; ties: smaller max_iter, "
              "then smaller learning_rate", "candidates": candidates, "split_seed": SPLIT_SEED,
              "written_before_any_dirty_run": True}
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  selected: max_iter={chosen['max_iter']}, learning_rate={chosen['learning_rate']}  "
          f"-> results/phase16_gb_config.json (test split not consulted)")

    print("\nStage 1b: clean TEST scores (context for G1; the test split is used only from here on)")
    p = selected_params(config)
    rows = []
    for seed in SEEDS:
        for name, factory in (("gradient_boosting", gb_factory(p)), ("random_forest", rf_factory)):
            t0 = time.monotonic()
            pipe = factory(X_tr, seed).fit(X_tr, y_tr)
            s = scores(pipe, X_te, y_te)
            rows.append({"model": name, "seed": seed, **s, "fit_seconds": round(time.monotonic() - t0, 2)})
            print(f"  {name:18s} seed={seed}  posF1={s['posf1']:.4f}  macroF1={s['macrof1']:.4f}  AUC={s['auc']:.4f}")
    lr_pipe = build_models(build_preprocessor(X_tr), seed=SPLIT_SEED)["logistic_regression"].fit(X_tr, y_tr)
    s = scores(lr_pipe, X_te, y_te)
    rows.append({"model": "logistic_regression", "seed": SPLIT_SEED, **s, "fit_seconds": float("nan")})
    print(f"  {'logistic_regression':18s} seed={SPLIT_SEED}  posF1={s['posf1']:.4f}  macroF1={s['macrof1']:.4f}  AUC={s['auc']:.4f}")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, "phase16_gb_clean.csv"), index=False)

    gb = df[df.model == "gradient_boosting"]
    rf42 = float(df[(df.model == "random_forest") & (df.seed == SPLIT_SEED)].posf1.iloc[0])
    gb_mean = float(gb.posf1.mean())
    threshold = RF_CLEAN_F1_RECORDED - G1_MARGIN
    g1 = gb_mean >= threshold
    print(f"\nGB clean test positive-F1 (mean of {len(gb)} seeds) = {gb_mean:.4f}, std across seeds = {gb.posf1.std():.2e}")
    print(f"RF clean test positive-F1, seed 42 = {rf42:.4f} (recorded in Phase 2: {RF_CLEAN_F1_RECORDED})")
    print(f"G1 {'PASS' if g1 else 'FAIL'}: GB {gb_mean:.4f} vs threshold {threshold:.4f}")
    config["g1"] = {"gb_clean_posf1_mean": gb_mean, "threshold": threshold, "pass": bool(g1),
                    "gb_seed_std": float(gb.posf1.std()), "rf_seed42": rf42}
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)

    # the determinism check again, now with the SELECTED configuration (Stage 0c used a fixed candidate)
    a = gb_factory(p)(X_tr, 1).fit(X_tr, y_tr)
    b = gb_factory(p)(X_tr, 2).fit(X_tr, y_tr)
    same = bool((a.predict(X_te) == b.predict(X_te)).all())
    print(f"Determinism with the selected configuration (random_state 1 vs 2): identical predictions = {same}")
    config["determinism_selected_config"] = same
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)
    return config


# ---------------------------------------------------------------------------------------------------------------
# Stage 2 (damage + noise floor) and Stage 3 (capacity arms)
# ---------------------------------------------------------------------------------------------------------------
ERROR_TYPES = ["missing_values", "label_errors", "duplicates", "outliers", "feature_corruption"]
RATES = [0.05, 0.10, 0.20]
METRICS = ("posf1", "macrof1", "auc")
INFORMATIVE_RF_DAMAGE = 0.005                       # spec section E: informative cell = RF mean damage >= this
PATHS = {k: os.path.join(RESULTS_DIR, f"phase16_gb_{k}") for k in
         ("damage_raw.csv", "damage_summary.csv", "noise.json", "capacity.csv", "criteria.json",
          "repair_raw.csv", "repair_summary.csv", "evidence_table.csv",
          "priority_headline.csv", "priority_combos.csv", "priority_sensitivity.csv",
          "bank_runs.csv", "bank_meta.json", "bank_harness_check.csv", "bank_evidence_posf1.csv",
          "bank_evidence_macrof1.csv", "bank_evidence_auc.csv")}
SAFE_MIN_PRECISION = 0.95                           # taskclean.SAFE_REPAIR_MIN_PRECISION (the product's gate)
SAFE_MAX_RISK = 1e-9


def run_damage(factories: dict, X_tr, y_tr, X_te, y_te, error_types, rates, seeds, log=print) -> pd.DataFrame:
    """Phase 7 analogue for every arm in `factories` ({arm: factory}). The corruption draw for a (type, rate, seed)
    is built once and shared by all arms; each arm's clean baseline is fitted once per seed. damage = clean - dirty."""
    baselines = {}
    for arm, f in factories.items():
        for seed in seeds:
            baselines[(arm, seed)] = scores(f(X_tr, seed).fit(X_tr, y_tr), X_te, y_te)
    rows = []
    for et in error_types:
        for rate in rates:
            for seed in seeds:
                dirty = CORRUPTORS[et](X_tr, y_tr, rate=rate, seed=seed)[:2]
                for arm, f in factories.items():
                    r = damage_cell(f, X_tr, y_tr, X_te, y_te, et, rate, seed, dirty=dirty)
                    base = baselines[(arm, seed)]
                    row = {"arm": arm, "error_type": et, "rate": rate, "seed": seed,
                           "fit_seconds": r["fit_seconds"]}
                    for m in METRICS:
                        row[f"baseline_{m}"] = base[m]
                        row[f"dirty_{m}"] = r[f"dirty_{m}"]
                        row[f"damage_{m}"] = base[m] - r[f"dirty_{m}"]
                    rows.append(row)
                    log(f"  {arm:9s} {et:19s} rate={rate:4.0%} seed={seed}  posF1={row['dirty_posf1']:.4f}  "
                        f"damage={row['damage_posf1']:+.4f}")
    return pd.DataFrame(rows)


def rms(values) -> float:
    return float(np.sqrt(np.mean(np.square(np.asarray(values, dtype=float)))))


def null_experiment(factory, X_tr, y_tr, X_te, y_te, seeds, baselines: dict | None = None, pos_label=POS_LABEL) -> dict:
    """selfbench's null experiment, for one model: retrain on a random 90% of the training rows (NULL_DRAWS draws per
    seed, one random stream seeded SEED0) and measure how far each metric moves from that seed's clean baseline.
    null_noise = root-mean-square of those moves; noise_floor = max(NOISE_FLOOR_MIN, null_noise)."""
    from selfbench import NOISE_FLOOR_MIN, NULL_DRAWS, NULL_DROP_FRACTION, SEED0
    rng = np.random.default_rng(SEED0)
    baselines = baselines or {s: scores(factory(X_tr, s).fit(X_tr, y_tr), X_te, y_te, pos_label) for s in seeds}
    moves = {m: [] for m in METRICS}
    for seed in seeds:
        for _ in range(NULL_DRAWS):
            keep = rng.random(len(X_tr)) >= NULL_DROP_FRACTION
            sc = scores(factory(X_tr[keep], seed).fit(X_tr[keep], y_tr[keep]), X_te, y_te, pos_label)
            for m in METRICS:
                moves[m].append(sc[m] - baselines[seed][m])
    out = {"n_draws": len(moves["posf1"]), "drop_fraction": NULL_DROP_FRACTION, "noise_floor_min": NOISE_FLOOR_MIN}
    for m in METRICS:
        out[m] = {"null_noise": rms(moves[m]), "noise_floor": max(NOISE_FLOOR_MIN, rms(moves[m])),
                  "moves": [float(v) for v in moves[m]]}
    return out


def summarise_damage(raw: pd.DataFrame, rf_summary: pd.DataFrame | None = None) -> pd.DataFrame:
    g = raw.groupby(["error_type", "rate"])
    out = g.agg(**{f"mean_damage_{m}": (f"damage_{m}", "mean") for m in METRICS},
                **{f"std_damage_{m}": (f"damage_{m}", "std") for m in METRICS},
                mean_dirty_posf1=("dirty_posf1", "mean"), n_seeds=("seed", "nunique")).reset_index()
    if rf_summary is not None:
        rf = rf_summary.rename(columns={"mean_damage": "rf_mean_damage", "std_damage": "rf_std_damage"})
        out = out.merge(rf[["error_type", "rate", "rf_mean_damage", "rf_std_damage"]], on=["error_type", "rate"], how="left")
        out["informative"] = out.rf_mean_damage >= INFORMATIVE_RF_DAMAGE
        out["gb_over_rf"] = np.where(out.informative, out.mean_damage_posf1 / out.rf_mean_damage, np.nan)
    return out.sort_values(["error_type", "rate"]).reset_index(drop=True)


# --- criteria (pure functions over {(issue, rate): mean damage}; spec section E) -------------------------------------
def _top_issue(damage: dict, rate: float) -> str:
    cells = {et: d for (et, r), d in damage.items() if abs(r - rate) < 1e-9}
    return max(cells, key=cells.get)


def criterion_g2(gb: dict, rf: dict) -> dict:
    """Spearman of GB vs RF mean damage over all cells (>= 0.8) AND label errors largest at 10% and 20%."""
    from scipy.stats import spearmanr
    keys = sorted(set(gb) & set(rf))
    rho = float(spearmanr([gb[k] for k in keys], [rf[k] for k in keys]).statistic) if len(keys) > 2 else float("nan")
    top10, top20 = _top_issue(gb, 0.10), _top_issue(gb, 0.20)
    ok = bool(np.isfinite(rho) and rho >= 0.8 and top10 == "label_errors" and top20 == "label_errors")
    return {"n_cells": len(keys), "spearman": rho, "top_at_10pct": top10, "top_at_20pct": top20, "pass": ok}


def criterion_g3(gb: dict, rf: dict) -> dict:
    """Median GB/RF damage over the informative cells (RF damage >= 0.005) in [0.5, 2.0]."""
    cells = sorted(k for k in rf if rf[k] >= INFORMATIVE_RF_DAMAGE and k in gb)
    ratios = {f"{et}@{r:.0%}": gb[(et, r)] / rf[(et, r)] for et, r in cells}
    med = float(np.median(list(ratios.values()))) if ratios else float("nan")
    return {"n_informative": len(cells), "ratios": ratios, "median_ratio": med,
            "pass": bool(np.isfinite(med) and 0.5 <= med <= 2.0)}


def criterion_g8(arms: dict) -> dict:
    """Label errors have the largest damage at 10% and at 20% in every capacity arm (all six cases)."""
    cases = {f"{arm}@{rate:.0%}": _top_issue(d, rate) for arm, d in arms.items() for rate in (0.10, 0.20)}
    return {"cases": cases, "pass": bool(len(cases) == 2 * len(arms) and len(arms) >= 3 and
                                         all(v == "label_errors" for v in cases.values()))}


def _damage_dict(df: pd.DataFrame, col="mean_damage_posf1") -> dict:
    return {(r.error_type, float(r.rate)): float(getattr(r, col)) for r in df.itertuples()}


def write_criteria() -> dict:
    """Recompute every criterion that the files on disk allow, mechanically. Nothing is judged by eye."""
    out = {}
    hc = os.path.join(RESULTS_DIR, "phase16_gb_harness_check.csv")
    if os.path.exists(hc):
        h = pd.read_csv(hc)
        out["G0"] = {"checks": len(h), "passed": int(h["pass"].sum()), "pass": bool(h["pass"].all())}
    if os.path.exists(CONFIG_PATH):
        cfg = json.load(open(CONFIG_PATH))
        if "g1" in cfg:
            out["G1"] = cfg["g1"]
    rf_path = os.path.join(RESULTS_DIR, "phase7_impact_summary_full.csv")
    if os.path.exists(PATHS["damage_summary.csv"]) and os.path.exists(rf_path):
        s = pd.read_csv(PATHS["damage_summary.csv"])
        rf = _damage_dict(pd.read_csv(rf_path).rename(columns={"mean_damage": "mean_damage_posf1"}))
        gb = _damage_dict(s)
        out["G2"], out["G3"] = criterion_g2(gb, rf), criterion_g3(gb, rf)
    if os.path.exists(PATHS["capacity.csv"]):
        cap = pd.read_csv(PATHS["capacity.csv"])
        arms = {a: _damage_dict(summarise_damage(d)) for a, d in cap.groupby("arm")}
        out["G8"] = criterion_g8(arms)
    if os.path.exists(PATHS["evidence_table.csv"]):
        ev = pd.read_csv(PATHS["evidence_table.csv"])
        out["G4"], out["G5"], out["G6"] = criterion_g4(ev), criterion_g5(ev), criterion_g6(ev)
        out["G7"] = criterion_g7(pd.read_csv(os.path.join(RESULTS_DIR, "phase10_evidence_table.csv")), ev)
    bank_ok = (os.path.exists(PATHS["bank_meta.json"]) and os.path.exists(PATHS["bank_evidence_macrof1.csv"]))
    if bank_ok:
        meta = json.load(open(PATHS["bank_meta.json"]))
        if not meta.get("harness_check_pass"):
            out["G9"] = {"pass": None, "reason": "Stage 6 harness check against the frozen Phase 13 RF numbers failed; not reported"}
        else:
            from phase13_transfer import safe_decisions
            gbm = pd.read_csv(PATHS["bank_evidence_macrof1.csv"])
            out["G9"] = criterion_g9(gbm, pd.read_csv(_bank_tag("macrof1")), safe_decisions(gbm))
            out["G9"]["bank_meta"] = {k: meta[k] for k in ("gb_clean_seed42", "rf_clean_seed42", "gb_noise_floor", "rf_noise_floor")}
    with open(PATHS["criteria.json"], "w") as f:
        json.dump(out, f, indent=2, default=lambda o: bool(o) if isinstance(o, np.bool_) else float(o))
    return out


def stage2() -> pd.DataFrame:
    config = require_config()
    p = selected_params(config)
    X_tr, _, X_te, y_tr, _, y_te = load_split()
    factory = gb_factory(p)
    print(f"Stage 2: GB {p} | {len(ERROR_TYPES)} issues x {len(RATES)} rates x {len(SEEDS)} seeds = "
          f"{len(ERROR_TYPES) * len(RATES) * len(SEEDS)} damage fits")
    t0 = time.monotonic()
    raw = run_damage({"selected": factory}, X_tr, y_tr, X_te, y_te, ERROR_TYPES, RATES, SEEDS)
    raw.to_csv(PATHS["damage_raw.csv"], index=False)

    print("\nNull experiment (selfbench's definition: retrain on a random 90% of the training rows)")
    noise = null_experiment(factory, X_tr, y_tr, X_te, y_te, SEEDS)
    with open(PATHS["noise.json"], "w") as f:
        json.dump(noise, f, indent=2)
    for m in METRICS:
        print(f"  {m:8s} null_noise={noise[m]['null_noise']:.5f}  noise_floor=max(0.003, .)={noise[m]['noise_floor']:.5f}")

    rf = pd.read_csv(os.path.join(RESULTS_DIR, "phase7_impact_summary_full.csv"))
    summary = summarise_damage(raw.drop(columns="arm"), rf)
    summary.to_csv(PATHS["damage_summary.csv"], index=False)
    show = summary[["error_type", "rate", "mean_damage_posf1", "std_damage_posf1", "rf_mean_damage", "gb_over_rf",
                    "mean_damage_macrof1", "mean_damage_auc"]]
    print("\nStage 2 summary (damage = clean - dirty; positive-class F1 unless stated)")
    print(show.round(4).to_string(index=False))
    crit = write_criteria()
    print(f"\nG2: {json.dumps(crit.get('G2'), default=float)}\nG3: {json.dumps(crit.get('G3'), default=float)}")
    print(f"({time.monotonic() - t0:.0f}s) Saved phase16_gb_damage_raw.csv, _damage_summary.csv, _noise.json, _criteria.json")
    return summary


def stage3() -> pd.DataFrame:
    require_config()
    if not os.path.exists(PATHS["damage_raw.csv"]):
        raise SystemExit("run --stage 2 first: the selected configuration is the third capacity arm and is read from it")
    X_tr, _, X_te, y_tr, _, y_te = load_split()
    print("Stage 3: capacity arms 'shallow' and 'deep' (the selected arm is Stage 2's result), damage only")
    t0 = time.monotonic()
    extra = run_damage({a: gb_factory(c) for a, c in CAPACITY_ARMS.items()}, X_tr, y_tr, X_te, y_te,
                       ERROR_TYPES, RATES, SEEDS)
    selected = pd.read_csv(PATHS["damage_raw.csv"])
    cap = pd.concat([selected, extra], ignore_index=True)
    cap.to_csv(PATHS["capacity.csv"], index=False)
    print("\nStage 3: mean damage (positive-class F1) by arm")
    wide = cap.groupby(["error_type", "rate", "arm"]).damage_posf1.mean().unstack("arm")
    print(wide[["shallow", "selected", "deep"]].round(4).to_string())
    crit = write_criteria()
    print(f"\nG8: {json.dumps(crit.get('G8'), default=float)}")
    print(f"({time.monotonic() - t0:.0f}s) Saved phase16_gb_capacity.csv, _criteria.json")
    return cap


# ---------------------------------------------------------------------------------------------------------------
# Stage 4: repair evidence under GB, status/risk by the frozen rules, compared with the frozen RF evidence table
# ---------------------------------------------------------------------------------------------------------------
from selfbench import classify_repairability, harm_risk, material_harm_risk, SIGN_AGREEMENT    # noqa: E402

STATUS_ORDER = ("negative", "weak", "positive")


def stricter_status(a: str, b: str) -> str:
    """Spec F: 'negative' if either is negative, 'positive' only if both are, otherwise 'weak'."""
    if "negative" in (a, b):
        return "negative"
    return "positive" if a == b == "positive" else "weak"


def is_safe_auto(risk: float, precision: float, status: str) -> bool:
    """The product's gate (taskclean: risk <= 1e-9, precision >= 0.95, status != negative). The never-auto list and the
    power guard are not part of this comparison: they do not depend on the model."""
    return bool(risk <= SAFE_MAX_RISK and precision >= SAFE_MIN_PRECISION and status != "negative")


def evidence_from_raw(raw: pd.DataFrame, floor: float, rf_evidence: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per (issue, rate) evidence table from per-seed repair rows, with the frozen Phase 10 definitions of status
    (classify_repairability), p_harm / magnitude / risk (harm_risk). `floor` is the measured GB noise floor; status at
    the fixed 0.003 is reported beside it. Columns follow results/phase10_evidence_table.csv, plus rf_* and gating."""
    rows = []
    for (et, rate), g in raw.groupby(["error_type", "rate"]):
        rec = g["recovery_posf1"].to_numpy(dtype=float)
        status = classify_repairability(rec, floor)
        p_harm, magnitude, risk = harm_risk(rec)
        p_mat, mat_risk = material_harm_risk(rec, floor)
        n_pos, n_neg = int((rec > 0).sum()), int((rec < 0).sum())
        rows.append({
            "error_type": et, "rate": float(rate),
            "mean_damage": g["damage_posf1"].mean(), "damage_std": g["damage_posf1"].std(),
            "repair_precision": g["repair_precision"].mean(), "repair_recall": g["repair_recall"].mean(),
            "mean_recovery": rec.mean(), "sign_agreement": max(n_pos, n_neg) / len(rec),
            "n_pos_seeds": n_pos, "n_neg_seeds": n_neg, "n_seeds": len(rec),
            "repairability_status": status, "status_at_fixed_0.003": classify_repairability(rec, 0.003),
            "evidence_status": "partial" if status == "weak" else "complete",
            "p_harm": p_harm, "harm_magnitude": magnitude, "risk": risk, "material_risk": mat_risk,
            "noise_floor": floor,
            "mean_recovery_macrof1": g["recovery_macrof1"].mean(), "mean_recovery_auc": g["recovery_auc"].mean(),
            "mean_oracle_recovery_posf1": (g["oracle_posf1"] - g["dirty_posf1"]).mean(),
        })
    out = pd.DataFrame(rows).sort_values(["rate", "error_type"]).reset_index(drop=True)
    out["gb_safe_auto"] = [is_safe_auto(r.risk, r.repair_precision, r.repairability_status) for r in out.itertuples()]
    if rf_evidence is not None:
        rf = rf_evidence.rename(columns={"repairability_status": "rf_status", "mean_recovery": "rf_mean_recovery",
                                         "risk": "rf_risk", "repair_precision": "rf_precision"})
        out = out.merge(rf[["error_type", "rate", "rf_status", "rf_mean_recovery", "rf_risk", "rf_precision"]],
                        on=["error_type", "rate"], how="left")
        out["status_agrees"] = out.repairability_status == out.rf_status
        out["effective_status"] = [stricter_status(a, b) for a, b in zip(out.repairability_status, out.rf_status)]
        out["rf_safe_auto"] = [is_safe_auto(r.rf_risk, r.rf_precision, r.rf_status) for r in out.itertuples()]
        out["safe_under_both"] = out.gb_safe_auto & out.rf_safe_auto
        out["gb_only_harm"] = out.rf_status.isin(["positive", "weak"]) & (out.repairability_status == "negative")
    return out


def criterion_g4(evidence: pd.DataFrame) -> dict:
    """Status agreement over all cells (>= 11 of 15), with the 'everything weak' chance baseline and Cohen's kappa."""
    from sklearn.metrics import cohen_kappa_score
    n = len(evidence)
    agree = int((evidence.repairability_status == evidence.rf_status).sum())
    all_weak = int((evidence.rf_status == "weak").sum())               # agreement of the trivial 'always weak' guess
    kappa = float(cohen_kappa_score(evidence.rf_status, evidence.repairability_status)) \
        if evidence.rf_status.nunique() > 1 or evidence.repairability_status.nunique() > 1 else float("nan")
    disagree = [f"{r.error_type}@{r.rate:.0%}: RF {r.rf_status} / GB {r.repairability_status}"
                for r in evidence.itertuples() if r.rf_status != r.repairability_status]
    return {"n_cells": n, "agree": agree, "needed": 11, "chance_baseline_all_weak": all_weak, "cohens_kappa": kappa,
            "disagreements": disagree, "pass": bool(n == 15 and agree >= 11)}


def criterion_g5(evidence: pd.DataFrame) -> dict:
    """Duplicate removal passes the safe-auto gate under GB at every rate."""
    d = evidence[evidence.error_type == "duplicates"].sort_values("rate")
    cells = {f"{r.rate:.0%}": {"status": r.repairability_status, "risk": float(r.risk), "precision": float(r.repair_precision),
                              "material_risk": float(r.material_risk), "safe": bool(r.gb_safe_auto)} for r in d.itertuples()}
    return {"cells": cells, "pass": bool(len(d) == 3 and all(v["safe"] for v in cells.values()))}


def criterion_g6(evidence: pd.DataFrame) -> dict:
    """GB-only harm: RF positive or weak, GB negative. None for duplicates and at most 1 across all cells."""
    h = evidence[(evidence.rf_status.isin(["positive", "weak"])) & (evidence.repairability_status == "negative")]
    cells = [f"{r.error_type}@{r.rate:.0%}: RF {r.rf_status} -> GB negative" for r in h.itertuples()]
    dup = int((h.error_type == "duplicates").sum())
    return {"n_gb_only_harm": len(h), "duplicates_harmed": dup, "cells": cells,
            "pass": bool(len(evidence) == 15 and dup == 0 and len(h) <= 1)}


def stage4() -> pd.DataFrame:
    config = require_config()
    p = selected_params(config)
    noise = json.load(open(PATHS["noise.json"])) if os.path.exists(PATHS["noise.json"]) else None
    if noise is None:
        raise SystemExit("run --stage 2 first: the GB noise floor comes from its null experiment")
    floor = float(noise["posf1"]["noise_floor"])
    X_tr, _, X_te, y_tr, _, y_te = load_split()
    factory = gb_factory(p)
    clean = {s: scores(factory(X_tr, s).fit(X_tr, y_tr), X_te, y_te) for s in SEEDS}
    cells = [(et, rate, seed) for et in ERROR_TYPES for rate in RATES for seed in SEEDS]
    print(f"Stage 4: GB {p} | {len(cells)} repair cells (detect + repair once, three GB fits each) | noise floor {floor}")
    partial = PATHS["repair_raw.csv"] + ".partial"                     # per-cell checkpoint; deleted on completion
    rows = pd.read_csv(partial).to_dict("records") if os.path.exists(partial) else []
    done = {(r["error_type"], round(float(r["rate"]), 4), int(r["seed"])) for r in rows}
    if rows:
        print(f"resuming: {len(rows)} cells already in {os.path.basename(partial)}")
    t0 = time.monotonic()
    for i, (et, rate, seed) in enumerate(cells, 1):
        if (et, round(rate, 4), seed) in done:
            continue
        tc = time.monotonic()
        bundle = make_repair_bundle(et, X_tr, y_tr, rate, seed)
        t_detect = time.monotonic() - tc
        row = repair_cell(factory, bundle, X_te, y_te, clean[seed])
        row["detect_seconds"] = round(t_detect, 1)
        rows.append(row)
        pd.DataFrame(rows).to_csv(partial, index=False)
        print(f"[{i:2d}/{len(cells)}] {et:19s} {rate:.0%} seed {seed}: detect+repair {t_detect:5.1f}s | total "
              f"{time.monotonic() - tc:5.1f}s | damage {row['damage_posf1']:+.4f} recovery {row['recovery_posf1']:+.4f} "
              f"precision {row['repair_precision']:.3f}", flush=True)
    raw = pd.DataFrame(rows)
    raw.to_csv(PATHS["repair_raw.csv"], index=False)
    if os.path.exists(partial):
        os.remove(partial)

    rf_ev = pd.read_csv(os.path.join(RESULTS_DIR, "phase10_evidence_table.csv"))
    ev = evidence_from_raw(raw, floor, rf_ev)
    ev.to_csv(PATHS["evidence_table.csv"], index=False)
    summ = raw.groupby(["error_type", "rate"]).agg(
        **{f"mean_{c}": (c, "mean") for c in ("clean_posf1", "dirty_posf1", "repaired_posf1", "oracle_posf1", "damage_posf1",
                                              "recovery_posf1", "recovery_macrof1", "recovery_auc",
                                              "repair_precision", "repair_recall")},
        std_recovery_posf1=("recovery_posf1", "std"), n_seeds=("seed", "nunique")).reset_index()
    summ.to_csv(PATHS["repair_summary.csv"], index=False)

    show = ev[["error_type", "rate", "mean_recovery", "sign_agreement", "repairability_status", "rf_status", "rf_mean_recovery",
               "risk", "repair_precision", "gb_safe_auto", "rf_safe_auto"]]
    print("\nStage 4: repair recovery (repaired - dirty, positive-class F1), status under GB vs the frozen RF table")
    print(show.round(4).to_string(index=False))
    crit = write_criteria()
    for k in ("G4", "G5", "G6"):
        print(f"\n{k}: {json.dumps(crit.get(k), indent=1, default=float)}")
    print(f"({time.monotonic() - t0:.0f}s) Saved phase16_gb_repair_raw.csv, _repair_summary.csv, _evidence_table.csv, _criteria.json")
    return ev


# ---------------------------------------------------------------------------------------------------------------
# Stage 5: Phase 10's priority machinery (imported unchanged) on the GB evidence table
# ---------------------------------------------------------------------------------------------------------------
import itertools                                                                                # noqa: E402
from phase10_prioritization import compute_priority, run_sensitivity_grid                       # noqa: E402

WEIGHT_VALUES = (0.2, 0.3, 0.4, 0.5, 0.6)           # Phase 10's grid: every triple summing to 1 -> 15 combinations
HEADLINE_RATE = 0.10
EQUAL_W = (1 / 3, 1 / 3, 1 / 3)


def headline_ranking(evidence: pd.DataFrame, rate: float = HEADLINE_RATE, w=EQUAL_W) -> list[str]:
    """The five issues, highest priority first, at one rate and one weight triple (Phase 10's compute_priority)."""
    return compute_priority(evidence, rate, *w)["error_type"].tolist()


def weight_combos(values=WEIGHT_VALUES) -> list[tuple]:
    return [w for w in itertools.product(values, repeat=3) if abs(sum(w) - 1.0) < 1e-9]


def combo_rankings(evidence: pd.DataFrame, rate: float = HEADLINE_RATE) -> dict:
    """{(w_impact, w_risk, w_repair): ranking} over the whole weight grid."""
    return {w: compute_priority(evidence, rate, *w)["error_type"].tolist() for w in weight_combos()}


def label_first_feature_last_share(rankings: dict) -> float:
    """Share of weight combinations in which label errors rank first AND feature corruption ranks last (one joint
    event per combination: G7's second condition)."""
    hits = [r[0] == "label_errors" and r[-1] == "feature_corruption" for r in rankings.values()]
    return float(np.mean(hits)) if hits else float("nan")


def criterion_g7(rf_evidence: pd.DataFrame, gb_evidence: pd.DataFrame, rate: float = HEADLINE_RATE) -> dict:
    """Kendall tau between the RF and GB headline rankings (equal weights) >= 0.6 AND the label-first /
    feature-corruption-last share over the 15 weight combinations >= 0.8 (RF reference: 1.0)."""
    from scipy.stats import kendalltau
    rf_rank, gb_rank = headline_ranking(rf_evidence, rate), headline_ranking(gb_evidence, rate)
    issues = sorted(rf_rank)
    tau = float(kendalltau([rf_rank.index(i) for i in issues], [gb_rank.index(i) for i in issues]).statistic)
    gb_share = label_first_feature_last_share(combo_rankings(gb_evidence, rate))
    rf_share = label_first_feature_last_share(combo_rankings(rf_evidence, rate))
    return {"rate": rate, "rf_ranking": rf_rank, "gb_ranking": gb_rank, "kendall_tau": tau, "tau_needed": 0.6,
            "gb_share_label_first_feature_last": gb_share, "rf_share_label_first_feature_last": rf_share,
            "share_needed": 0.8, "n_combos": len(weight_combos()),
            "pass": bool(np.isfinite(tau) and tau >= 0.6 and np.isfinite(gb_share) and gb_share >= 0.8)}


def stage5() -> pd.DataFrame:
    require_config()
    if not os.path.exists(PATHS["evidence_table.csv"]):
        raise SystemExit("run --stage 4 first: the priority ranking is built from the GB evidence table")
    gb = pd.read_csv(PATHS["evidence_table.csv"])
    rf = pd.read_csv(os.path.join(RESULTS_DIR, "phase10_evidence_table.csv"))

    heads = []
    for model, ev in (("RF", rf), ("GB", gb)):
        for rate in RATES:
            r = compute_priority(ev, rate, *EQUAL_W)
            r.insert(0, "rank", range(1, len(r) + 1))
            r.insert(0, "model", model)
            heads.append(r[["model", "rate", "rank", "error_type", "priority_score", "impact_norm", "risk_norm",
                            "repairability_norm", "repairability_status", "mean_damage", "mean_recovery", "risk"]])
    heads = pd.concat(heads, ignore_index=True)
    heads.to_csv(PATHS["priority_headline.csv"], index=False)

    combos = []
    for model, ev in (("RF", rf), ("GB", gb)):
        for w, ranking in combo_rankings(ev).items():
            combos.append({"model": model, "w_impact": w[0], "w_risk": w[1], "w_repair": w[2],
                           "ranking": " > ".join(ranking), "first": ranking[0], "last": ranking[-1],
                           "label_first_feature_last": ranking[0] == "label_errors" and ranking[-1] == "feature_corruption"})
    pd.DataFrame(combos).to_csv(PATHS["priority_combos.csv"], index=False)
    sens = pd.concat([run_sensitivity_grid(ev, HEADLINE_RATE).assign(model=m) for m, ev in (("RF", rf), ("GB", gb))])
    sens.to_csv(PATHS["priority_sensitivity.csv"], index=False)

    print("Stage 5: headline ranking at equal weights (highest priority first)")
    for rate in RATES:
        for model in ("RF", "GB"):
            h = heads[(heads.model == model) & (heads.rate == rate)]
            print(f"  {rate:>4.0%} {model}: " + " > ".join(h.error_type))
    print("\nWeight grid at 10% (rank position over the 15 combinations)")
    print(sens[["model", "error_type", "mean_rank", "min_rank", "max_rank", "rank_1_pct"]].round(3).to_string(index=False))
    crit = write_criteria()
    print(f"\nG7: {json.dumps(crit.get('G7'), indent=1, default=float)}")
    return heads


# ---------------------------------------------------------------------------------------------------------------
# Stage 6: Bank Marketing, the Phase 13 ground-truth protocol with GB as the downstream model (G9)
# ---------------------------------------------------------------------------------------------------------------
BANK_SLUG = "bank_marketing"


def _bank_tag(metric: str) -> str:
    return os.path.join(RESULTS_DIR, f"phase13_{BANK_SLUG}_groundtruth_evidence_{metric}.csv")


def criterion_g9(gb_macro: pd.DataFrame, rf_macro: pd.DataFrame, gb_decisions: pd.Series) -> dict:
    """Spec E + G-bis: on Bank Marketing, macro-F1 (the Phase 13 primary):
    (a) Spearman of GB vs RF mean damage over the 15 cells >= 0.8;
    (b) status agreement >= 11 of 15 (criterion_g4);
    (c) duplicate removal passes the product's gate (`safe_decisions`) under GB at all three rates."""
    from scipy.stats import spearmanr
    gb_d, rf_d = _damage_dict(gb_macro, "mean_damage"), _damage_dict(rf_macro, "mean_damage")
    keys = sorted(set(gb_d) & set(rf_d))
    rho = float(spearmanr([gb_d[k] for k in keys], [rf_d[k] for k in keys]).statistic) if len(keys) > 2 else float("nan")
    merged = gb_macro[["error_type", "rate", "repairability_status"]].merge(
        rf_macro[["error_type", "rate", "repairability_status"]].rename(columns={"repairability_status": "rf_status"}),
        on=["error_type", "rate"])
    g4 = criterion_g4(merged)
    dup = gb_macro[gb_macro.error_type == "duplicates"].sort_values("rate")
    cells = {f"{r.rate:.0%}": {"status": r.repairability_status, "material_risk": float(r.material_risk),
                              "strict_risk": float(r.risk), "precision": float(r.repair_precision),
                              "safe": bool(gb_decisions.loc[(r.error_type, r.rate)])} for r in dup.itertuples()}
    a, b = bool(np.isfinite(rho) and rho >= 0.8), bool(g4["pass"])
    c = bool(len(cells) == 3 and all(v["safe"] for v in cells.values()))
    return {"n_cells": len(keys), "spearman": rho, "a_spearman_pass": a,
            "top_at_10pct": {"gb": _top_issue(gb_d, 0.10), "rf": _top_issue(rf_d, 0.10)},
            "top_at_20pct": {"gb": _top_issue(gb_d, 0.20), "rf": _top_issue(rf_d, 0.20)},
            "b_agree": g4["agree"], "b_needed": 11, "b_chance_baseline_all_weak": g4["chance_baseline_all_weak"],
            "b_kappa": g4["cohens_kappa"], "b_disagreements": g4["disagreements"], "b_pass": b,
            "c_duplicates": cells, "c_pass": c, "pass": bool(a and b and c)}


def stage6() -> pd.DataFrame:
    config = require_config()
    p = selected_params(config)
    import phase13_transfer as p13
    from selfbench import _evidence_table, _same_data, NOISE_FLOOR_MIN, NULL_DRAWS, NULL_DROP_FRACTION
    t0 = time.monotonic()
    df, target = p13.load_raw(BANK_SLUG)
    X, y = p13.clean_dataset(df, target, p13.CAP_ROWS)
    pos = y.value_counts().idxmin()
    X_tr, _, X_te, y_tr, _, y_te = three_way_split(X, y, seed=42)
    seeds = p13.SEEDS
    factory = gb_factory(p)
    print(f"Stage 6: Bank Marketing, {len(X):,} clean rows (train {len(X_tr):,} / test {len(X_te):,}), positive class {pos!r}, GB {p}")

    base = {sd: scores(factory(X_tr, sd).fit(X_tr, y_tr), X_te, y_te, pos) for sd in seeds}
    rf_base = {sd: p13._scores(X_tr, y_tr, X_te, y_te, sd, pos) for sd in seeds[:1]}
    print("  clean baseline GB:", {k: round(v, 4) for k, v in base[seeds[0]].items()},
          "| RF:", {k: round(v, 4) for k, v in rf_base[seeds[0]].items()})

    noise = null_experiment(factory, X_tr, y_tr, X_te, y_te, seeds, baselines=base, pos_label=pos)
    floor = {m: noise[m]["noise_floor"] for m in METRICS}
    rf_meta = json.load(open(os.path.join(RESULTS_DIR, f"phase13_{BANK_SLUG}_groundtruth_meta.json")))
    print("  GB noise floors:", {m: round(v, 4) for m, v in floor.items()}, "| RF:", {m: round(v, 4) for m, v in rf_meta["noise_floor"].items()})

    frozen = pd.read_csv(os.path.join(RESULTS_DIR, f"phase13_{BANK_SLUG}_groundtruth_runs.csv"))
    rows, check = [], []
    total = len(p13.ISSUES) * len(p13.RATES) * len(seeds)
    for issue in p13.ISSUES:
        for rate in p13.RATES:
            for seed in seeds:
                tc = time.monotonic()
                X_d, y_d, log = p13.CORRUPTORS[issue](X_tr, y_tr, rate=rate, seed=seed)       # Phase 13's calls, unchanged
                X_r, y_r, repaired_set, _ = p13.detect_and_repair(issue, X_d, y_d, seed)
                g = p13.grade_repair(issue, log, repaired_set, X_d, X_r)
                dirty = scores(factory(X_d, seed).fit(X_d, y_d), X_te, y_te, pos)
                rep = scores(factory(X_r, seed).fit(X_r, y_r), X_te, y_te, pos)
                row = {"issue": issue, "rate": rate, "seed": seed, "repair_precision": g["precision"],
                       "repair_recall": g["recall"], "restored_exactly": _same_data(X_r, y_r, X_tr, y_tr)}
                for m in METRICS:
                    row.update({f"base_{m}": base[seed][m], f"dirty_{m}": dirty[m], f"repaired_{m}": rep[m],
                                f"damage_{m}": base[seed][m] - dirty[m], f"recovery_{m}": rep[m] - dirty[m]})
                rows.append(row)
                if seed == seeds[0]:                          # harness check: the frozen RF scorer on the same frames
                    fr = frozen[(frozen.issue == issue) & (frozen.rate == rate) & (frozen.seed == seed)].iloc[0]
                    rf_d, rf_r = p13._scores(X_d, y_d, X_te, y_te, seed, pos), p13._scores(X_r, y_r, X_te, y_te, seed, pos)
                    for m in METRICS:
                        for arm, val in (("dirty", rf_d[m]), ("repaired", rf_r[m]), ("base", rf_base[seed][m])):
                            check.append({"issue": issue, "rate": rate, "seed": seed, "arm": arm, "metric": m,
                                          "reproduced": val, "frozen": float(fr[f"{arm}_{m}"]),
                                          "abs_diff": abs(val - float(fr[f"{arm}_{m}"]))})
                print(f"  [{len(rows):2d}/{total}] {issue:18s} {rate:4.0%} seed {seed}: {time.monotonic() - tc:5.1f}s | damage macroF1 "
                      f"{row['damage_macrof1']:+.4f} | recovery macroF1 {row['recovery_macrof1']:+.4f} | P/R {g['precision']:.2f}/{g['recall']:.2f}", flush=True)
    runs = pd.DataFrame(rows)
    runs.to_csv(PATHS["bank_runs.csv"], index=False)
    chk = pd.DataFrame(check)
    chk["pass"] = chk.abs_diff <= G0_TOLERANCE
    chk.to_csv(PATHS["bank_harness_check.csv"], index=False)
    harness_ok = bool(chk["pass"].all())
    print(f"\nHarness check (frozen RF scorer on this stage's frames vs frozen Phase 13 runs, seed {seeds[0]}): "
          f"{int(chk['pass'].sum())}/{len(chk)} within {G0_TOLERANCE:g}, max diff {chk.abs_diff.max():.2e} -> {'PASS' if harness_ok else 'FAIL'}")

    for m in METRICS:
        view = runs.assign(damage=runs[f"damage_{m}"], recovery=runs[f"recovery_{m}"])
        _evidence_table(view, floor[m]).to_csv(PATHS[f"bank_evidence_{m}.csv"], index=False)
    meta = {"dataset": "bank-marketing", "positive_class": str(pos), "clean_rows": len(X), "train_rows": len(X_tr),
            "test_rows": len(X_te), "gb_params": p, "gb_clean_seed42": base[seeds[0]], "rf_clean_seed42": rf_base[seeds[0]],
            "gb_clean_identical_across_seeds": bool(len({tuple(base[s].values()) for s in seeds}) == 1),
            "gb_noise_floor": floor, "gb_null_noise": {m: noise[m]["null_noise"] for m in METRICS},
            "rf_noise_floor": rf_meta["noise_floor"], "harness_check_pass": harness_ok,
            "harness_check_max_abs_diff": float(chk.abs_diff.max()), "seconds": time.monotonic() - t0}
    json.dump(meta, open(PATHS["bank_meta.json"], "w"), indent=2)

    ev = pd.read_csv(PATHS["bank_evidence_macrof1.csv"])
    print("\nBank Marketing, GB evidence (macro-F1, the Phase 13 primary) vs frozen RF")
    rf_ev = pd.read_csv(_bank_tag("macrof1"))
    show = ev.merge(rf_ev[["error_type", "rate", "mean_damage", "mean_recovery", "repairability_status"]],
                    on=["error_type", "rate"], suffixes=("_gb", "_rf"))
    print(show[["error_type", "rate", "mean_damage_gb", "mean_damage_rf", "mean_recovery_gb", "mean_recovery_rf",
                "repairability_status_gb", "repairability_status_rf", "material_risk", "repair_precision"]].round(4).to_string(index=False))
    crit = write_criteria()
    print(f"\nG9: {json.dumps(crit.get('G9'), indent=1, default=float)}")
    print(f"({time.monotonic() - t0:.0f}s) Saved phase16_gb_bank_*.csv/json, _criteria.json")
    return runs


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, required=True)
    args = ap.parse_args(argv)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    if args.stage == 0:
        stage0()
    elif args.stage == 1:
        stage1()
    elif args.stage in (2, 3, 4, 5, 6):
        {2: stage2, 3: stage3, 4: stage4, 5: stage5, 6: stage6}[args.stage]()
    elif args.stage == 7:
        require_config()
        raise SystemExit(f"stage {args.stage} is not built yet")
    else:
        raise SystemExit("stage must be 0-7")


if __name__ == "__main__":
    main()
