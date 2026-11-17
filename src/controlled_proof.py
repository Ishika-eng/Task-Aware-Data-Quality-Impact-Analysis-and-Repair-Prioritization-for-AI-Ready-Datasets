"""
Controlled proof (Phase 20; method and criteria fixed in docs/PHASE20_SPEC.md before any run).

Start from clean Adult rows, hold out a CLEAN test set, inject all five error types into the training rows only, repair them with
the real TaskClean path, and score every model on the clean test rows. "Repaired beats dirty" then means the repair moved the
training data toward the truth, which the Model panel (whose held-out rows are as dirty as the user's file) cannot show.

Arms: clean ceiling, dirty control, default file, aggressive file, each repair alone, and TUNED cleaning: per model, candidate
repairs are scored on a validation slice of the DIRTY training rows (no clean data, no test data) and the best one is kept only if it
beats raw by more than SELECT_MARGIN; otherwise the raw data is kept.

Usage (project root):
    .venv/bin/python src/controlled_proof.py            # 5 seeds, writes results/phase20_proof_*
    .venv/bin/python src/controlled_proof.py --seeds 3  # quicker
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

from data import load_adult_clean
from model_panel import (AUDIT_MODEL_ROWS, MODEL_LABELS, MODELS, NULL_DRAWS, NULL_DROP_FRACTION, PANEL_DETECTORS, SINGLE_ARMS,
                         AGGRESSIVE, _fit_score, _frame_key, _typed)
from quality_report import build_combined_dirty
from selfbench import NOISE_FLOOR_MIN, SIGN_AGREEMENT, classify_repairability
from taskclean import _apply, audit_dataset, default_apply_issues

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
TARGET = "class"
N_ROWS = 8000
SAMPLE_SEED = 7
RATE = 0.05
SELECT_MARGIN = 0.005          # fixed in the spec: a candidate must beat raw on validation by more than this
VAL_FRACTION = 0.25
SEEDS = (42, 43, 44, 45, 46)
ARM_ORDER = ["clean", "dirty", "default", "aggressive", *SINGLE_ARMS, "tuned", "tuned_v0"]
ARM_LABEL = {"clean": "Clean ceiling", "dirty": "Dirty (control)", "default": "Default file", "aggressive": "Aggressive file",
             "duplicates": "Duplicates only", "missing_values": "Missing values only", "outliers": "Outliers only",
             "inconsistency": "Spellings only", "label_errors": "Label fixes only", "tuned": "Tuned cleaning",
             "tuned_v0": "Tuned (raw validation)"}
ARM_TEXT = {"clean": "Trained on the clean training rows: the most perfect cleaning could reach.",
            "dirty": "Trained on the training rows after all five error types were injected. No cleaning.",
            "default": "The repairs the evidence-based policy applies without review.",
            "aggressive": "Every proposed repair except feature-anomaly repairs.",
            "duplicates": "Exact duplicate rows removed.", "missing_values": "Missing cells filled.",
            "outliers": "Extreme values replaced by the fence value.", "inconsistency": "Spelling variants merged.",
            "label_errors": "Suspected wrong labels corrected.",
            "tuned_v0": "As tuned, but the validation slice is used as is, including rows that are copies of training rows (the "
                        "original design, kept so the effect of the leak-free change is visible).",
            "tuned": "Per model: the best repair on a validation slice of the dirty rows, kept only if it beats raw by more than "
                     f"{SELECT_MARGIN}; otherwise raw. Chosen without clean data or test data."}


def clean_frame() -> pd.DataFrame:
    X, y = load_adult_clean()
    idx = X.sample(N_ROWS, random_state=SAMPLE_SEED).index
    df = X.loc[idx].copy()
    df[TARGET] = y.loc[idx].values
    return df.reset_index(drop=True)


def inject(train_df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """All five error types into the training rows, the demo file's recipe (Phase 3 corruptors)."""
    X = train_df.drop(columns=TARGET)
    y = train_df[TARGET]
    Xd, yd, _ = build_combined_dirty(X, y, rate=RATE, seed=seed)
    out = Xd.copy()
    out[TARGET] = yd.values
    return out.reset_index(drop=True)


def repair_frames(dirty: pd.DataFrame, seed: int, default_from=None):
    """Apply the real TaskClean repairs to `dirty`. Returns ({arm: frame}, audit state). Reads only `dirty`."""
    st = audit_dataset(dirty, TARGET, detectors=PANEL_DETECTORS, seed=seed, max_model_rows=AUDIT_MODEL_ROWS)
    default_issues = default_apply_issues(st) & set(SINGLE_ARMS)
    arms = {"aggressive": set(AGGRESSIVE), **{a: {a} for a in SINGLE_ARMS}}
    if default_issues:
        arms["default"] = default_issues
    frames = {"dirty": dirty}
    for arm, issues in arms.items():
        frames[arm] = _apply(st, issues, [])[0]
    frames.setdefault("default", dirty)                 # nothing auto-approved: the default file equals the dirty one
    return frames, st, sorted(default_issues)


def select_repair(dirty: pd.DataFrame, models, seed: int, drop_cols, leak_free_val: bool = True) -> dict:
    """TUNED cleaning's choice for each model. Its only input is the dirty training frame: it splits it 75/25, repairs the 75% (the
    repairs are fitted on those rows only), and scores each candidate on the dirty 25%. Returns {model: {"arm", "val": {...}}}."""
    y = dirty[TARGET]
    try:
        tr_i, va_i = train_test_split(dirty.index, test_size=VAL_FRACTION, random_state=seed, stratify=y)
    except ValueError:
        tr_i, va_i = train_test_split(dirty.index, test_size=VAL_FRACTION, random_state=seed)
    tr, va = dirty.loc[tr_i].reset_index(drop=True), dirty.loc[va_i].reset_index(drop=True)
    if leak_free_val:        # Amendment A1: validation rows that also appear in the training slice would reward keeping duplicates
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
                cache[key] = _fit_score(name, X_tr, y_tr, X_va, y_va, seed, ones)["f1"]
            scores[arm] = cache[key]
        cands = {a: s for a, s in scores.items() if a != "dirty"}
        best = max(cands, key=cands.get) if cands else None
        gain = (cands[best] - scores["dirty"]) if best else 0.0
        out[name] = {"arm": best if best and gain > SELECT_MARGIN else "dirty",
                     "best_candidate": best, "best_gain": float(gain), "val": {k: float(v) for k, v in scores.items()}}
    return out


def run_controlled_proof(seeds=SEEDS, models=MODELS, progress: Callable[[float, str], None] | None = None) -> dict:
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
        tr_i, te_i = train_test_split(clean.index, test_size=0.2, random_state=seed, stratify=clean[TARGET])
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
                    cache[key] = _fit_score(name, X_tr, y_tr, X_te, y_te, seed, ones)
                    tick(f"Seed {seed}: {MODEL_LABELS[name]} on {ARM_LABEL[arm].lower()}", 1)
                rows.append({"seed": seed, "model": name, "arm": arm, "same_as_dirty": _frame_key(frames[arm]) == _frame_key(dirty),
                             **{k: cache[key][k] for k in ("f1", "auc", "acc")}})
            for variant, selected in selected_by.items():
                pick = selected[name]["arm"]
                sc = cache[(name, _frame_key(frames[pick]))]
                rows.append({"seed": seed, "model": name, "arm": variant, "same_as_dirty": pick == "dirty",
                             **{k: sc[k] for k in ("f1", "auc", "acc")}})
                choices.append({"seed": seed, "model": name, "variant": variant, "chosen": pick,
                                "best_candidate": selected[name]["best_candidate"], "best_val_gain": selected[name]["best_gain"]})
            base = cache[(name, _frame_key(frames["clean"]))]["f1"]
            rng = np.random.default_rng(seed)
            for d in range(NULL_DRAWS):
                keep = rng.random(len(clean_tr)) >= NULL_DROP_FRACTION
                X_tr, y_tr = _typed(clean_tr.loc[keep], TARGET, drop_cols)
                nulls.append({"seed": seed, "model": name, "d_f1": _fit_score(name, X_tr, y_tr, X_te, y_te, seed, ones)["f1"] - base})
                tick(f"Seed {seed}: measuring the noise floor ({MODEL_LABELS[name]})", 1)
    return summarise(pd.DataFrame(rows), pd.DataFrame(nulls), pd.DataFrame(choices), list(seeds), models, default_issues_seen,
                     time.monotonic() - t0)


def summarise(R: pd.DataFrame, N: pd.DataFrame, C: pd.DataFrame, seeds, models, default_issues, seconds,
              labels=None, metric: str | None = None, extra: dict | None = None) -> dict:
    """`labels`, `metric` and `extra` let the regression proof (controlled_proof_regression.py) reuse this logic; R's 'f1' column
    then holds R-squared. Defaults reproduce the classification result exactly."""
    labels = labels or MODEL_LABELS
    floors = {m: max(NOISE_FLOOR_MIN, float(np.sqrt(np.mean(np.square(N[N.model == m].d_f1))))) for m in models}
    cells, per_model = [], []
    for m in models:
        g = lambda arm: R[(R.model == m) & (R.arm == arm)].set_index("seed")          # noqa: E731
        clean, dirty = g("clean"), g("dirty")
        dmg = (clean.f1 - dirty.f1)
        dstat = classify_repairability(-dmg.to_numpy(), floors[m])                    # "negative" = dirty is worse than clean
        damage_real = dstat == "negative"
        per_model.append({"model": m, "label": labels[m], "noise_floor": floors[m], "clean_f1": float(clean.f1.mean()),
                          "dirty_f1": float(dirty.f1.mean()), "damage": float(dmg.mean()), "damage_beyond_noise": bool(damage_real)})
        for arm in ARM_ORDER:
            a = g(arm)
            row = {"model": m, "arm": arm, "f1_mean": float(a.f1.mean()), "f1_std": float(a.f1.std(ddof=0)),
                   "auc_mean": float(a.auc.mean()), "n_seeds": int(len(a))}
            if arm in ("clean", "dirty"):
                row.update(delta_f1=None if arm == "clean" else 0.0, verdict="ceiling" if arm == "clean" else "baseline",
                           share_recovered=None, seeds_agree=None)
            else:
                d = (a.f1 - dirty.f1).to_numpy()
                same = bool(a.same_as_dirty.all())
                if same:
                    verdict, agree = "no change", 1.0
                elif not np.any(d):
                    verdict, agree = "no effect", 1.0
                else:
                    verdict = {"positive": "improved", "negative": "worse", "weak": "within noise"}[classify_repairability(d, floors[m])]
                    agree = float(max((d > 0).sum(), (d < 0).sum()) / len(d))
                row.update(delta_f1=float(d.mean()), verdict=verdict, seeds_agree=agree,
                           share_recovered=(float(d.mean() / dmg.mean()) if damage_real and verdict in ("improved", "worse") else None))
            cells.append(row)
    cell = lambda m, a: next(c for c in cells if c["model"] == m and c["arm"] == a)     # noqa: E731
    n_damaged = sum(p["damage_beyond_noise"] for p in per_model)
    tuned = [cell(m, "tuned")["verdict"] for m in models]
    tuned0 = [cell(m, "tuned_v0")["verdict"] for m in models]
    default = [cell(m, "default")["verdict"] for m in models]
    criteria = {
        "C0_evaluable": {"pass": bool(n_damaged >= min(3, len(models))), "models_damaged_beyond_noise": int(n_damaged)},
        "C1_complete": {"pass": bool(len(cells) == len(models) * len(ARM_ORDER)), "cells": len(cells)},
        "C2_tuned_positive": {"pass": bool(tuned.count("improved") >= 2 and "worse" not in tuned), "verdicts": dict(zip(models, tuned))},
        "C2_tuned_v0_positive": {"pass": bool(tuned0.count("improved") >= 2 and "worse" not in tuned0), "verdicts": dict(zip(models, tuned0))},
        "C3_default_not_worse": {"pass": bool("worse" not in default), "verdicts": dict(zip(models, default))},
    }
    choice_counts = {f"{v}/{m}": C[(C.model == m) & (C.variant == v)].chosen.value_counts().to_dict() for m in models for v in ("tuned", "tuned_v0")}
    return {"status": "ok", "seeds": [int(s) for s in seeds], "n_seeds": len(seeds), "rows": N_ROWS, "rate_per_error_type": RATE,
            "metric": metric or "macro-F1 on CLEAN held-out rows", "models": per_model, "arms": [{"key": a, "label": ARM_LABEL[a], "text": ARM_TEXT[a]} for a in ARM_ORDER],
            "results": cells, "criteria": criteria, "tuned_choices": choice_counts, "default_issues": sorted(default_issues),
            "select_margin": SELECT_MARGIN, "seconds": round(seconds, 1),
            "caveat": ("Synthetic, random errors injected into one dataset (UCI Adult) at 5% each; test rows are clean. This shows the repairs "
                       "move training data toward the truth on this setup. It does not show the same gain on another dataset, on "
                       "real-world errors, or on an upload (which has no clean truth). Feature corruption is injected but not repaired by "
                       "TaskClean, so the clean ceiling is not expected to be reached."), **(extra or {})}


def main():
    n = int(sys.argv[sys.argv.index("--seeds") + 1]) if "--seeds" in sys.argv else len(SEEDS)
    seeds = SEEDS[:n]
    last = [0.0]

    def prog(f, m):
        if time.time() - last[0] > 20:
            print(f"  {f:.0%} {m}", flush=True)
            last[0] = time.time()

    out = run_controlled_proof(seeds, progress=prog)
    os.makedirs(RESULTS, exist_ok=True)
    pd.DataFrame(out["results"]).to_csv(os.path.join(RESULTS, "phase20_proof_summary.csv"), index=False)
    json.dump(out, open(os.path.join(RESULTS, "phase20_proof.json"), "w"), indent=2)
    json.dump(out["criteria"], open(os.path.join(RESULTS, "phase20_proof_criteria.json"), "w"), indent=2)
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
