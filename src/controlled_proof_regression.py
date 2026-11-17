"""
Controlled proof for REGRESSION (Phase 21, criterion R9; method and criteria fixed in docs/PHASE21_SPEC.md before the run).

California Housing (a fixed 8,000-row sample): hold out CLEAN test rows, inject all five error types (5% each, one of them wrong
target values) into the TRAINING rows only, repair with the real TaskClean path (task='regression'), and score four regressors on
the clean test rows. Arms and tuned cleaning exactly as Phase 20 (controlled_proof.py), with R-squared instead of macro-F1; the
summary logic (noise floors, verdicts, share of damage recovered, criteria C0-C3) is controlled_proof.summarise.

Usage (project root):
    .venv/bin/python src/controlled_proof_regression.py            # 5 seeds, writes results/phase21_proof_*
    .venv/bin/python src/controlled_proof_regression.py --seeds 3
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import controlled_proof as cp
from corruption import corruptors_for
from model_panel import AGGRESSIVE, AUDIT_MODEL_ROWS, NULL_DRAWS, NULL_DROP_FRACTION, PANEL_DETECTORS, SINGLE_ARMS, _frame_key
from model_panel_regression import ARM_LABEL as PANEL_ARM_LABEL, MODEL_LABELS, MODELS, _fit_score, _typed
from regression import strat_labels
from taskclean import _apply, audit_dataset, default_apply_issues

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
DATA = os.path.join(os.path.dirname(__file__), "..", "data", "california_housing.csv")
TARGET = "MedHouseVal"
N_ROWS, SAMPLE_SEED, RATE = 8000, 7, 0.05
APPLY_ORDER = ["missing_values", "outliers", "label_errors", "feature_corruption", "duplicates"]   # as Phase 3: duplicates last
SEEDS = cp.SEEDS


def clean_frame() -> pd.DataFrame:
    df = pd.read_csv(DATA)
    return df.sample(N_ROWS, random_state=SAMPLE_SEED).reset_index(drop=True)


def inject(train_df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """All five error types into the training rows (the demo recipe), with wrong TARGET values for 'label_errors'."""
    X, y = train_df.drop(columns=TARGET), train_df[TARGET]
    for et in APPLY_ORDER:
        X, y, _ = corruptors_for("regression")[et](X, y, rate=RATE, seed=seed)
    out = X.copy()
    out[TARGET] = y.values
    return out.reset_index(drop=True)


def repair_frames(dirty: pd.DataFrame, seed: int):
    st = audit_dataset(dirty, TARGET, detectors=PANEL_DETECTORS, seed=seed, max_model_rows=AUDIT_MODEL_ROWS, task="regression")
    default_issues = default_apply_issues(st) & set(SINGLE_ARMS)
    arms = {"aggressive": set(AGGRESSIVE), **{a: {a} for a in SINGLE_ARMS}}
    if default_issues:
        arms["default"] = default_issues
    frames = {"dirty": dirty}
    for arm, issues in arms.items():
        frames[arm] = _apply(st, issues, [])[0]
    frames.setdefault("default", dirty)
    return frames, st, sorted(default_issues)


def select_repair(dirty: pd.DataFrame, models, seed: int, drop_cols, leak_free_val: bool = True) -> dict:
    """Tuned cleaning's choice per model. Its only input is the dirty training frame (75/25 split, repairs fitted on the 75%,
    candidates scored on the dirty 25%, kept only if they beat raw by more than SELECT_MARGIN; Amendment A1 of Phase 20: validation
    rows that are copies of training rows are dropped)."""
    strata = strat_labels(dirty[TARGET])
    tr_i, va_i = train_test_split(dirty.index, test_size=cp.VAL_FRACTION, random_state=seed, stratify=strata)
    tr, va = dirty.loc[tr_i].reset_index(drop=True), dirty.loc[va_i].reset_index(drop=True)
    if leak_free_val:
        h = lambda f: pd.util.hash_pandas_object(f.astype(str), index=False)          # noqa: E731
        va = va[~h(va).isin(set(h(tr)))].reset_index(drop=True)
    frames, _, _ = repair_frames(tr, seed)
    X_va, y_va = _typed(va, TARGET, drop_cols)
    ones = np.ones(len(y_va), dtype=bool)
    out, cache = {}, {}
    for name in models:
        scores = {}
        for arm, frame in frames.items():
            if arm == "default":
                continue
            key = (name, _frame_key(frame))
            if key not in cache:
                X_tr, y_tr = _typed(frame, TARGET, drop_cols)
                cache[key] = _fit_score(name, X_tr, y_tr, X_va, y_va, seed, ones)["score"]
            scores[arm] = cache[key]
        cands = {a: s for a, s in scores.items() if a != "dirty"}
        best = max(cands, key=cands.get) if cands else None
        gain = (cands[best] - scores["dirty"]) if best else 0.0
        out[name] = {"arm": best if best and gain > cp.SELECT_MARGIN else "dirty", "best_candidate": best,
                     "best_gain": float(gain), "val": {k: float(v) for k, v in scores.items()}}
    return out


def run_controlled_proof_regression(seeds=SEEDS, models=MODELS, progress: Callable[[float, str], None] | None = None) -> dict:
    say = progress or (lambda f, m: None)
    t0 = time.monotonic()
    clean = clean_frame()
    rows, nulls, choices = [], [], []
    per_seed = 3 + 2 * len(models) * 14
    done = [0]

    def tick(msg, n=1):
        done[0] += n
        say(min(0.99, done[0] / (per_seed * len(seeds))), msg)

    default_issues_seen = set()
    for seed in seeds:
        seed = int(seed)
        tr_i, te_i = train_test_split(clean.index, test_size=0.2, random_state=seed, stratify=strat_labels(clean[TARGET]))
        clean_tr, clean_te = clean.loc[tr_i].reset_index(drop=True), clean.loc[te_i].reset_index(drop=True)
        dirty = inject(clean_tr, seed)
        tick(f"Seed {seed}: injecting the five error types into the training rows", 1)
        frames, st, default_issues = repair_frames(dirty, seed)
        default_issues_seen |= set(default_issues)
        frames["clean"] = clean_tr
        drop_cols = list(st.unmodeled_columns)
        X_te, y_te = _typed(clean_te, TARGET, drop_cols)
        ones = np.ones(len(y_te), dtype=bool)
        tick(f"Seed {seed}: repairs proposed", 2)
        selected_by = {"tuned": select_repair(dirty, models, seed, drop_cols, True),
                       "tuned_v0": select_repair(dirty, models, seed, drop_cols, False)}
        cache = {}
        for name in models:
            for arm in ["clean", "dirty", "default", "aggressive", *SINGLE_ARMS]:
                key = (name, _frame_key(frames[arm]))
                if key not in cache:
                    X_tr, y_tr = _typed(frames[arm], TARGET, drop_cols)
                    sc = _fit_score(name, X_tr, y_tr, X_te, y_te, seed, ones)
                    cache[key] = {"f1": sc["score"], "auc": sc["rmse"], "acc": float("nan")}     # f1 holds R-squared, auc holds RMSE
                    tick(f"Seed {seed}: {MODEL_LABELS[name]} on {cp.ARM_LABEL[arm].lower()}", 1)
                rows.append({"seed": seed, "model": name, "arm": arm, "same_as_dirty": _frame_key(frames[arm]) == _frame_key(dirty),
                             **cache[key]})
            for variant, selected in selected_by.items():
                pick = selected[name]["arm"]
                rows.append({"seed": seed, "model": name, "arm": variant, "same_as_dirty": pick == "dirty",
                             **cache[(name, _frame_key(frames[pick]))]})
                choices.append({"seed": seed, "model": name, "variant": variant, "chosen": pick,
                                "best_candidate": selected[name]["best_candidate"], "best_val_gain": selected[name]["best_gain"]})
            base = cache[(name, _frame_key(frames["clean"]))]["f1"]
            rng = np.random.default_rng(seed)
            for d in range(NULL_DRAWS):
                keep = rng.random(len(clean_tr)) >= NULL_DROP_FRACTION
                X_tr, y_tr = _typed(clean_tr.loc[keep], TARGET, drop_cols)
                nulls.append({"seed": seed, "model": name, "d_f1": _fit_score(name, X_tr, y_tr, X_te, y_te, seed, ones)["score"] - base})
                tick(f"Seed {seed}: measuring the noise floor ({MODEL_LABELS[name]})", 1)
    out = cp.summarise(pd.DataFrame(rows), pd.DataFrame(nulls), pd.DataFrame(choices), list(seeds), models, default_issues_seen,
                       time.monotonic() - t0, labels=MODEL_LABELS, metric="R-squared on CLEAN held-out rows",
                       extra={"task": "regression", "dataset": "California Housing (8,000-row sample)",
                              "caveat": ("Synthetic, random errors injected into one dataset (California Housing) at 5% each, one of them "
                                         "wrong target values; test rows are clean. This shows the repairs move training data toward "
                                         "the truth on this setup. It does not show the same gain on another dataset, on real-world "
                                         "errors, or on an upload (which has no clean truth). Feature corruption is injected but not "
                                         "repaired by TaskClean, so the clean ceiling is not expected to be reached.")})
    out["arms"] = [{**a, "label": a["label"].replace("Label fixes only", "Target fixes only")} for a in out["arms"]]
    return out


def main():
    n = int(sys.argv[sys.argv.index("--seeds") + 1]) if "--seeds" in sys.argv else len(SEEDS)
    last = [0.0]

    def prog(f, m):
        if time.time() - last[0] > 20:
            print(f"  {f:.0%} {m}", flush=True)
            last[0] = time.time()

    out = run_controlled_proof_regression(SEEDS[:n], progress=prog)
    os.makedirs(RESULTS, exist_ok=True)
    pd.DataFrame(out["results"]).to_csv(os.path.join(RESULTS, "phase21_proof_summary.csv"), index=False)
    json.dump(out, open(os.path.join(RESULTS, "phase21_proof.json"), "w"), indent=2)
    json.dump(out["criteria"], open(os.path.join(RESULTS, "phase21_proof_criteria.json"), "w"), indent=2)
    for p in out["models"]:
        print(f"{p['label']:20s} clean {p['clean_f1']:.4f}  dirty {p['dirty_f1']:.4f}  damage {p['damage']:+.4f}  "
              f"(beyond noise: {p['damage_beyond_noise']}, floor {p['noise_floor']:.4f})")
    for c in out["results"]:
        if c["arm"] not in ("clean", "dirty"):
            sr = "" if c["share_recovered"] is None else f"  recovered {c['share_recovered']:.0%}"
            print(f"  {c['model']:20s} {c['arm']:15s} {c['delta_f1']:+.4f} {c['verdict']}{sr}")
    print(json.dumps(out["criteria"], indent=1))
    print("tuned choices:", out["tuned_choices"])


if __name__ == "__main__":
    main()
