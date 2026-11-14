"""
Phase 14 validation (method and success/failure criteria: docs/PHASE14_SPEC.md, written before this ran).

  inconsistency  inject variants into Adult's categorical cells (5/10/20%), then run the REAL product path
                 (audit_dataset -> _apply) and score detection, repair correctness and model impact. Same discipline
                 as Phases 7 and 9: seeds 42-46, paired difference against a clean baseline per seed, damage measured
                 on the untouched clean test set, corruption on the training split only.
  relevance      add columns that are irrelevant by construction and measure whether they are caught, whether real
                 Adult columns are wrongly flagged, and what dropping the flagged columns does to F1.
  realdata       run the inconsistency detector on every real dataset in data/ and save what it flags, for a human
                 to read (an independent check on false positives that does not depend on our own injector).

Usage (from the project root):
    .venv/bin/python src/phase14_validation.py                 # one seed: fast smoke test
    .venv/bin/python src/phase14_validation.py --full          # seeds 42-46 (the real run)
    .venv/bin/python src/phase14_validation.py --only realdata # one section
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd

from baseline import evaluate
from data import load_adult_clean, three_way_split
from phase7_impact import SPLIT_SEED, build_rf_pipeline
from phase10_prioritization import MAJORITY_THRESHOLD, NOISE_FLOOR
from phase14_injectors import IRRELEVANT_KINDS, KINDS, add_irrelevant_columns, inject_inconsistency
from relevance import assess_features
from taskclean import _apply, audit_dataset

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RATES = [0.05, 0.10, 0.20]
TARGET = "class"
FLAGGED_VERDICTS = {"constant", "id_like", "redundant", "no_detectable_relevance"}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def obj_frame(X: pd.DataFrame) -> pd.DataFrame:
    """Non-numeric columns as plain object (what the project's pipeline treats as categorical on every pandas)."""
    out = X.copy()
    for c in out.columns:
        if not pd.api.types.is_numeric_dtype(out[c]):
            out[c] = out[c].astype(object)
    return out


def fit_scores(X_tr, y_tr, X_te, y_te, seed) -> dict:
    pipe = build_rf_pipeline(X_tr, seed=seed)
    pipe.fit(X_tr, y_tr)
    return evaluate(pipe, X_te, y_te)


def save(df: pd.DataFrame, name: str):
    df.to_csv(os.path.join(RESULTS_DIR, name), index=False)
    log(f"saved {name} ({len(df)} rows)")


def load_split():
    X, y = load_adult_clean()
    X_tr, X_val, X_te, y_tr, y_val, y_te = three_way_split(X, y, seed=SPLIT_SEED)
    return obj_frame(X_tr), obj_frame(X_te), y_tr, y_te


# ------------------------------------------------------------------------------------------------ inconsistency
def run_inconsistency(seeds):
    X_tr, X_te, y_tr, y_te = load_split()
    log(f"clean baselines for {len(seeds)} seed(s)")
    base = {s: fit_scores(X_tr, y_tr, X_te, y_te, s) for s in seeds}
    if SPLIT_SEED in base:
        fresh = base[SPLIT_SEED]["f1"]
        log(f"sanity vs Phase 2 recorded baseline 0.6804: {'MATCHES' if abs(fresh - 0.6804) < 1e-3 else 'DOES NOT MATCH'} "
            f"(fresh={fresh:.4f})")

    runs, by_kind = [], []
    for rate in RATES:
        for seed in seeds:
            X_d, y_d, inj = inject_inconsistency(X_tr, y_tr, rate=rate, seed=seed)
            pos = X_d.index.get_indexer(inj.row_id)
            injected = {(int(p), c) for p, c in zip(pos, inj.column)}
            kind_of = {(int(p), c): k for p, c, k in zip(pos, inj.column, inj.kind)}
            dirty = fit_scores(X_d, y_d, X_te, y_te, seed)

            d = X_d.reset_index(drop=True)
            d[TARGET] = y_d.values
            st = audit_dataset(d, TARGET, detectors={"inconsistency"}, seed=seed)
            cells = st.detections["inconsistency"].extra["cells"]
            flagged = {(int(r), c) for r, c in zip(cells.row_id, cells["column"])}
            cleaned, rlog = _apply(st, {"inconsistency"}, [])
            applied = rlog[rlog.action == "applied"]
            applied_set = {(int(r), c) for r, c in zip(applied.row_id, applied["column"])}
            clean_orig = {(p, c): X_tr.iat[p, X_tr.columns.get_loc(c)] for p, c in applied_set | injected}
            correct = {(p, c) for p, c in applied_set if cleaned.at[p, c] == clean_orig[(p, c)]}
            X_rep = obj_frame(cleaned.drop(columns=TARGET))
            X_rep.index = X_d.index
            repaired = fit_scores(X_rep, y_d, X_te, y_te, seed)

            tp = len(flagged & injected)
            runs.append({
                "error_type": "inconsistency", "rate": rate, "seed": seed,
                "n_injected": len(injected), "n_flagged": len(flagged), "n_flagged_injected": tp,
                "detect_precision": tp / len(flagged) if flagged else np.nan,
                "detect_recall": tp / len(injected) if injected else np.nan,
                "n_applied": len(applied_set), "n_applied_correct": len(correct),
                "n_applied_not_injected": len(applied_set - injected),
                "repair_precision": len(correct) / len(applied_set) if applied_set else 1.0,
                "repair_recall": len(correct & injected) / len(injected) if injected else np.nan,
                "baseline_f1": base[seed]["f1"], "dirty_f1": dirty["f1"], "repaired_f1": repaired["f1"],
                "f1_damage": base[seed]["f1"] - dirty["f1"], "recovery": repaired["f1"] - dirty["f1"],
                "baseline_auc": base[seed]["roc_auc"], "dirty_auc": dirty["roc_auc"], "repaired_auc": repaired["roc_auc"],
                "auc_damage": base[seed]["roc_auc"] - dirty["roc_auc"], "auc_recovery": repaired["roc_auc"] - dirty["roc_auc"],
            })
            for kind in KINDS:
                ks = {k for k, v in kind_of.items() if v == kind}
                by_kind.append({"rate": rate, "seed": seed, "kind": kind, "n_injected": len(ks),
                                "n_flagged": len(ks & flagged), "n_repaired_correct": len(ks & correct)})
            r = runs[-1]
            log(f"rate={rate:.0%} seed={seed}: injected={r['n_injected']} detP={r['detect_precision']:.3f} "
                f"detR={r['detect_recall']:.3f} repP={r['repair_precision']:.3f} repR={r['repair_recall']:.3f} "
                f"damage={r['f1_damage']:+.4f} recovery={r['recovery']:+.4f}")
    return pd.DataFrame(runs), pd.DataFrame(by_kind)


def summarise_inconsistency(runs: pd.DataFrame, by_kind: pd.DataFrame):
    impact = runs.groupby(["error_type", "rate"]).agg(
        mean_f1=("dirty_f1", "mean"), std_f1=("dirty_f1", "std"), mean_damage=("f1_damage", "mean"),
        std_damage=("f1_damage", "std"), mean_auc_damage=("auc_damage", "mean"), n_seeds=("seed", "nunique")).reset_index()

    rows = []                       # the Phase 10 evidence-table rules, applied unchanged
    for (et, rate), g in runs.groupby(["error_type", "rate"]):
        n, rec = len(g), g["recovery"]
        n_pos, n_neg = int((rec > 0).sum()), int((rec < 0).sum())
        agree = max(n_pos, n_neg) / n
        if agree >= MAJORITY_THRESHOLD and rec.mean() >= NOISE_FLOOR:
            status = "positive"
        elif agree >= MAJORITY_THRESHOLD and rec.mean() <= -NOISE_FLOOR:
            status = "negative"
        else:
            status = "weak"
        harmful = rec[rec < 0]
        p_harm, mag = n_neg / n, (abs(harmful.mean()) if len(harmful) else 0.0)
        dmg = impact[(impact.error_type == et) & (impact.rate == rate)].iloc[0]
        rows.append({"error_type": et, "rate": rate, "mean_damage": dmg.mean_damage, "damage_std": dmg.std_damage,
                     "repair_precision": g.repair_precision.mean(), "repair_recall": g.repair_recall.mean(),
                     "mean_recovery": rec.mean(), "sign_agreement": agree, "n_pos_seeds": n_pos, "n_neg_seeds": n_neg,
                     "repairability_status": status,
                     "evidence_status": {"positive": "complete", "negative": "complete", "weak": "partial"}[status],
                     "p_harm": p_harm, "harm_magnitude": mag, "risk": p_harm * mag})
    evidence = pd.DataFrame(rows)

    tp, fl, inj = runs.n_flagged_injected.sum(), runs.n_flagged.sum(), runs.n_injected.sum()
    prec, recall = tp / fl, tp / inj
    detection = pd.DataFrame([{"error_type": "inconsistency", "unit": "cells", "true_count": int(inj),
                               "detected_count": int(fl), "precision": prec, "recall": recall,
                               "f1": 2 * prec * recall / (prec + recall), "tp": int(tp), "fp": int(fl - tp),
                               "fn": int(inj - tp)}])
    kind = by_kind.groupby("kind").agg(injected=("n_injected", "sum"), flagged=("n_flagged", "sum"),
                                       repaired_correct=("n_repaired_correct", "sum")).reset_index()
    kind["flag_recall"] = kind.flagged / kind.injected
    kind["repair_recall"] = kind.repaired_correct / kind.injected
    return impact, evidence, detection, kind


# ---------------------------------------------------------------------------------------------- false-positive floor
def false_positive_floor():
    X_tr, X_te, y_tr, _ = load_split()
    out = {}
    for name, X, y in (("adult_train_clean", X_tr, y_tr),):
        d = X.reset_index(drop=True)
        d[TARGET] = y.values
        st = audit_dataset(d, TARGET, detectors={"inconsistency"})
        det = st.detections["inconsistency"]
        out[name] = {"rows": len(d), "rate": float(det.rate), "flagged_cells": int(len(det.extra["cells"]))}
        log(f"false-positive floor on {name}: rate={det.rate:.5f}, flagged cells={len(det.extra['cells'])}")
    return out


def realdata_check():
    rows = []
    for fn in sorted(os.listdir(DATA_DIR)):
        if not fn.endswith("_original.csv"):
            continue
        df = pd.read_csv(os.path.join(DATA_DIR, fn), low_memory=False)
        st = audit_dataset(df, None, detectors={"inconsistency"})
        det = st.detections["inconsistency"]
        cells = det.extra["cells"]
        log(f"{fn}: {len(df)} rows, {df.shape[1]} cols, inconsistency rate={det.rate:.4f}, flagged cells={len(cells)}")
        if len(cells):
            top = cells.groupby(["column", "original_value", "proposed_value", "rule", "tier"], dropna=False).size() \
                       .reset_index(name="n_cells")
            top.insert(0, "dataset", fn.replace("_original.csv", ""))
            rows.append(top)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["dataset", "column", "original_value", "proposed_value", "rule", "tier", "n_cells"])


# ------------------------------------------------------------------------------------------------------ relevance
def integer_like_of(X):
    return {c: bool(pd.api.types.is_numeric_dtype(X[c]) and (X[c].dropna() % 1 == 0).all()) for c in X.columns}


def run_relevance(seeds):
    X_tr, X_te, y_tr, y_te = load_split()
    rows, utility = [], []
    for seed in seeds:
        log(f"relevance seed {seed}")
        res_clean = assess_features(X_tr, y_tr, seed=seed, integer_like=integer_like_of(X_tr)).set_index("column")
        X_aug, truth = add_irrelevant_columns(X_tr, seed)
        X_aug_te, _ = add_irrelevant_columns(X_te, seed + 1000, offset=10_000_000)
        res_aug = assess_features(X_aug, y_tr, seed=seed, integer_like=integer_like_of(X_aug)).set_index("column")
        for setting, res in (("clean", res_clean), ("augmented", res_aug)):
            for col, r in res.iterrows():
                rows.append({"seed": seed, "setting": setting, "column": col, "truth": truth.get(col, "real"),
                             "verdict": r.verdict, "redundant_with": r.redundant_with,
                             "perm_importance": r.perm_importance, "mutual_info": r.mutual_info})
        real = list(X_tr.columns)
        drop_clean = [c for c in real if res_clean.verdict[c] in FLAGGED_VERDICTS]
        drop_aug = [c for c in X_aug.columns if res_aug.verdict[c] in FLAGGED_VERDICTS]
        f_orig = fit_scores(X_tr, y_tr, X_te, y_te, seed)["f1"]
        f_aug = fit_scores(X_aug, y_tr, X_aug_te, y_te, seed)["f1"]
        f_aug_drop = fit_scores(X_aug.drop(columns=drop_aug), y_tr, X_aug_te.drop(columns=drop_aug), y_te, seed)["f1"]
        f_orig_drop = (fit_scores(X_tr.drop(columns=drop_clean), y_tr, X_te.drop(columns=drop_clean), y_te, seed)["f1"]
                       if drop_clean else f_orig)
        utility.append({"seed": seed, "f1_original_14_columns": f_orig, "f1_with_irrelevant_added": f_aug,
                        "f1_after_dropping_flagged_from_added": f_aug_drop,
                        "f1_after_dropping_flagged_from_original": f_orig_drop,
                        "n_flagged_real_columns": len(drop_clean), "flagged_real_columns": ";".join(drop_clean),
                        "n_added": len(truth), "n_added_flagged": sum(c in truth for c in drop_aug)})
        log(f"  flagged real columns (clean): {drop_clean}; added flagged {utility[-1]['n_added_flagged']}/{len(truth)}; "
            f"F1 orig={f_orig:.4f} +noise={f_aug:.4f} dropped={f_aug_drop:.4f} real-dropped={f_orig_drop:.4f}")
    return pd.DataFrame(rows), pd.DataFrame(utility)


def summarise_relevance(rows: pd.DataFrame, utility: pd.DataFrame) -> dict:
    aug = rows[rows.setting == "augmented"]
    added = aug[aug.truth != "real"]
    recall_by_kind = {k: float(g.verdict.isin(FLAGGED_VERDICTS).mean()) for k, g in added.groupby("truth")}
    verdicts_by_kind = {k: g.verdict.value_counts().to_dict() for k, g in added.groupby("truth")}
    real = rows[(rows.setting == "clean") & (rows.truth == "real")]
    flagged_freq = {c: float(g.verdict.isin(FLAGGED_VERDICTS).mean()) for c, g in real.groupby("column")}
    flagged_freq = {c: f for c, f in flagged_freq.items() if f > 0}
    verdict_of_flagged = {c: real[(real.column == c) & real.verdict.isin(FLAGGED_VERDICTS)].verdict.value_counts().to_dict()
                          for c in flagged_freq}
    n_seeds = int(rows.seed.nunique())
    u = utility
    return {
        "n_seeds": n_seeds,
        "recall_by_injected_kind": recall_by_kind, "verdicts_by_injected_kind": verdicts_by_kind,
        "overall_recall_on_injected": float(added.verdict.isin(FLAGGED_VERDICTS).mean()),
        "real_columns_flagged_on_clean_adult": {"fraction_of_seeds_flagged": flagged_freq, "verdicts": verdict_of_flagged},
        "mean_real_columns_flagged_per_seed": float(u.n_flagged_real_columns.mean()),
        "f1": {c: {"mean": float(u[c].mean()), "std": float(u[c].std()) if len(u) > 1 else None}
               for c in ["f1_original_14_columns", "f1_with_irrelevant_added", "f1_after_dropping_flagged_from_added",
                         "f1_after_dropping_flagged_from_original"]},
        "paired_effect_of_dropping_flagged_real_columns": {
            "mean_f1_change": float((u.f1_after_dropping_flagged_from_original - u.f1_original_14_columns).mean())},
        "paired_effect_of_adding_irrelevant_columns": {
            "mean_f1_change": float((u.f1_with_irrelevant_added - u.f1_original_14_columns).mean())},
        "paired_recovery_from_dropping_flagged_added": {
            "mean_f1_change": float((u.f1_after_dropping_flagged_from_added - u.f1_with_irrelevant_added).mean())},
    }


# ------------------------------------------------------------------------------------------------------- main
if __name__ == "__main__":
    full = "--full" in sys.argv
    only = sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else "all"
    seeds = [42, 43, 44, 45, 46] if full else [42]
    suffix = "" if full else "_single_seed"
    os.makedirs(RESULTS_DIR, exist_ok=True)
    log(f"phase 14 validation: {'FULL' if full else 'single-seed smoke test'}, section={only}")

    if only in ("all", "realdata"):
        rd = realdata_check()
        save(rd, f"phase14_inconsistency_realdata{suffix}.csv")
    if only in ("all", "inconsistency"):
        floor = false_positive_floor()
        runs, by_kind = run_inconsistency(seeds)
        impact, evidence, detection, kind = summarise_inconsistency(runs, by_kind)
        save(runs, f"phase14_inconsistency_runs{suffix}.csv")
        save(kind, f"phase14_inconsistency_by_kind{suffix}.csv")
        save(impact, f"phase14_impact_summary{suffix}.csv")
        save(evidence, f"phase14_evidence_table{suffix}.csv")
        save(detection, f"phase14_detection_scores{suffix}.csv")
        with open(os.path.join(RESULTS_DIR, f"phase14_baseline_rates{suffix}.json"), "w") as f:
            json.dump({"detector_floors": {"inconsistency": floor["adult_train_clean"]["rate"]}, "detail": floor,
                       "note": "false-positive floor of the Phase 14 inconsistency detector on clean UCI Adult"}, f, indent=2)
        print("\n" + evidence.round(4).to_string(index=False))
        print("\n" + kind.round(3).to_string(index=False))
    if only in ("all", "relevance"):
        rows, utility = run_relevance(seeds)
        summary = summarise_relevance(rows, utility)
        save(rows, f"phase14_relevance_validation{suffix}.csv")
        save(utility, f"phase14_relevance_utility{suffix}.csv")
        with open(os.path.join(RESULTS_DIR, f"phase14_relevance_summary{suffix}.json"), "w") as f:
            json.dump(summary, f, indent=2)
        log("saved phase14_relevance_summary" + suffix + ".json")
        print(json.dumps(summary, indent=2))
