"""
Phase 13: transfer to a second dataset, and validation of the calibration.

Questions, fixed BEFORE running (answers are reported whatever they are):

  Q1 Robustness.   Does the product run end to end on a different real dataset, with honest repair logs?
  Q2 Transfer.     Do the Adult findings hold on dataset 2 (which error hurts most, which repairs are safe,
                   how repairs are prioritized)?
  Q3 Calibration.  Does the quick self-benchmark (what a user gets from "Calibrate on this dataset") agree with
                   an independent, larger controlled experiment on the same dataset?
  Q4 Value.        Does calibrated evidence agree with that ground truth BETTER than the Adult evidence
                   transferred to this dataset? The dangerous failure is a FALSE APPROVAL: a repair type judged
                   safe to apply automatically that the ground truth shows is not.

Ground truth = the research protocol (Phases 7-10) run on the CLEANED dataset 2: complete-case, de-duplicated,
70/15/15 split, 5 seeds, 300-tree Random Forest, repairs through the Phase 9 engine (a code path independent of
the product layer), harm judged against a noise floor from a null experiment.

Metrics. Pre-specified: positive-class F1, as in Phases 2-9. Added AFTER seeing a first partial run (disclosed in
the report): macro-F1 and ROC-AUC. On a heavily imbalanced target (bank-marketing: 11.7% positive) label noise
pushes the model toward the minority class and RAISES positive-class F1, so that metric cannot measure damage
there; and the calibration under test measures macro-F1, so Q3/Q4 must compare macro-F1 with macro-F1. All three
are recorded for every run. The calibration under
test instead starts from the dataset as an uploader would provide it (raw, or with errors injected) and derives
its own clean-ish reference.

Stages (long ones run in the background):  data | groundtruth | calibration | regimes | report
"""

from __future__ import annotations

import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline

from baseline import build_preprocessor
from corruption import CORRUPTORS, RATES
from data import three_way_split
from detect import detect_feature_corruption_crossfeature
from phase9_repair_engine import detect_and_repair, grade_repair
from phase10_prioritization import compute_priority, run_sensitivity_grid
from quality_report import build_combined_dirty
from selfbench import (DETECTOR_OF, NOISE_FLOOR_MIN, NULL_DRAWS, NULL_DROP_FRACTION, _evidence_table, _prf,
                       _same_data, _stratified_sample_index, _true_set, run_self_benchmark)
from taskclean import (NEVER_AUTO_APPLY, SAFE_REPAIR_MIN_PRECISION, _full_record, _prepare, apply_repairs,
                       attach_self_benchmark, audit_dataset, default_apply_issues)

warnings.filterwarnings("ignore")

ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR = os.path.join(ROOT, "data")
RESULTS_DIR = os.path.join(ROOT, "results")

DATASETS = {"bank_marketing": "bank-marketing", "heart_c": "heart-c", "credit_g": "credit-g"}
PRIMARY = "bank_marketing"
SEEDS = [42, 43, 44, 45, 46]
CAP_ROWS = 12_000                 # cleaned rows used by the ground-truth protocol (keeps the slow detectors tractable)
N_ESTIMATORS = 300                # as in Phases 2-9
ISSUES = ["missing_values", "duplicates", "outliers", "label_errors", "feature_corruption"]
BENCH_ISSUES = ["missing_values", "duplicates", "outliers", "label_errors"]    # what the self-benchmark measures


# ---------------------------------------------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------------------------------------------
def fetch_raw(slug: str) -> tuple[pd.DataFrame, str]:
    from sklearn.datasets import fetch_openml
    bunch = fetch_openml(DATASETS[slug], version=1, as_frame=True)
    df = bunch.frame
    df.to_csv(os.path.join(DATA_DIR, f"{slug}_original.csv"), index=False)
    return df, bunch.target_names[0]


def load_raw(slug: str) -> tuple[pd.DataFrame, str]:
    path = os.path.join(DATA_DIR, f"{slug}_original.csv")
    if not os.path.exists(path):
        return fetch_raw(slug)
    target = {"bank_marketing": "Class", "heart_c": "num", "credit_g": "class"}[slug]
    return pd.read_csv(path), target


def clean_dataset(df: pd.DataFrame, target: str, cap_rows: int, seed: int = 42):
    """The ground-truth reference: labeled, complete, de-duplicated rows (stratified cap)."""
    X, y, *_ = _prepare(df.reset_index(drop=True), target)
    ok = y.notna() & ~X.isna().any(axis=1)
    X, y = X[ok], y[ok]
    dup = _full_record(X, y).duplicated(keep="first")
    X, y = X[~dup], y[~dup]
    if len(X) > cap_rows:
        idx = _stratified_sample_index(y, X.index, cap_rows, seed)
        X, y = X.loc[idx], y.loc[idx]
    return X, y


METRICS = ("posf1", "macrof1", "auc")


def _scores(X_tr, y_tr, X_te, y_te, seed: int, pos_label: str) -> dict:
    """One fit, three metrics: positive-class F1 (the research metric), macro-F1, ROC-AUC (threshold-free)."""
    from sklearn.metrics import roc_auc_score
    pipe = Pipeline([("prep", build_preprocessor(X_tr)),
                     ("clf", RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=seed, n_jobs=-1))])
    pipe.fit(X_tr, y_tr)
    pred = pipe.predict(X_te)
    proba = pipe.predict_proba(X_te)[:, list(pipe.classes_).index(pos_label)]
    return {"posf1": float(f1_score(y_te, pred, pos_label=pos_label, average="binary", zero_division=0)),
            "macrof1": float(f1_score(y_te, pred, average="macro", zero_division=0)),
            "auc": float(roc_auc_score((y_te == pos_label).astype(int), proba))}


# ---------------------------------------------------------------------------------------------------------------
# Ground truth: the Phase 7-10 protocol on the cleaned dataset 2
# ---------------------------------------------------------------------------------------------------------------
def run_groundtruth(slug: str = PRIMARY):
    t0 = time.monotonic()
    df, target = load_raw(slug)
    X, y = clean_dataset(df, target, CAP_ROWS)
    assert y.nunique() == 2, "phase 13 ground truth assumes a binary target"
    pos = y.value_counts().idxmin()                                   # minority class, as '>50K' was for Adult
    X_tr, X_va, X_te, y_tr, y_va, y_te = three_way_split(X, y, seed=42)
    print(f"[{slug}] clean rows {len(X):,} (train {len(X_tr):,} / test {len(X_te):,}); positive class {pos!r} "
          f"({(y == pos).mean():.1%})")

    base = {sd: _scores(X_tr, y_tr, X_te, y_te, sd, pos) for sd in SEEDS}
    print("  clean baseline:", {sd: {k: round(v, 3) for k, v in m.items()} for sd, m in list(base.items())[:2]}, "...")

    rng = np.random.default_rng(42)                                   # null experiment -> noise floor per metric
    diffs = {m: [] for m in METRICS}
    for sd in SEEDS:
        for _ in range(NULL_DRAWS):
            keep = rng.random(len(X_tr)) >= NULL_DROP_FRACTION
            sc = _scores(X_tr[keep], y_tr[keep], X_te, y_te, sd, pos)
            for m in METRICS:
                diffs[m].append(sc[m] - base[sd][m])
    null_noise = {m: float(np.sqrt(np.mean(np.square(diffs[m])))) for m in METRICS}
    noise_floor = {m: max(NOISE_FLOOR_MIN, null_noise[m]) for m in METRICS}
    print("  noise floors:", {m: round(v, 4) for m, v in noise_floor.items()})

    rows = []
    total = len(ISSUES) * len(RATES) * len(SEEDS)
    for issue in ISSUES:
        for rate in RATES:
            for seed in SEEDS:
                X_d, y_d, log = CORRUPTORS[issue](X_tr, y_tr, rate=rate, seed=seed)
                dirty = _scores(X_d, y_d, X_te, y_te, seed, pos)
                X_r, y_r, repaired_set, _ = detect_and_repair(issue, X_d, y_d, seed)      # Phase 9 engine
                rep = _scores(X_r, y_r, X_te, y_te, seed, pos)
                g = grade_repair(issue, log, repaired_set, X_d, X_r)
                row = {"issue": issue, "rate": rate, "seed": seed, "repair_precision": g["precision"],
                       "repair_recall": g["recall"], "restored_exactly": _same_data(X_r, y_r, X_tr, y_tr)}
                for m in METRICS:
                    row.update({f"base_{m}": base[seed][m], f"dirty_{m}": dirty[m], f"repaired_{m}": rep[m],
                                f"damage_{m}": base[seed][m] - dirty[m], f"recovery_{m}": rep[m] - dirty[m]})
                rows.append(row)
                print(f"  [{len(rows):2d}/{total}] {issue:18s} {rate:4.0%} seed {seed}: damage posF1 "
                      f"{row['damage_posf1']:+.4f} macroF1 {row['damage_macrof1']:+.4f} AUC {row['damage_auc']:+.4f} | "
                      f"recovery macroF1 {row['recovery_macrof1']:+.4f} | repair P/R {g['precision']:.2f}/{g['recall']:.2f}",
                      flush=True)
    runs = pd.DataFrame(rows)
    evidence = {}
    for m in METRICS:                                                # same rules, one evidence table per metric
        view = runs.assign(damage=runs[f"damage_{m}"], recovery=runs[f"recovery_{m}"])
        evidence[m] = _evidence_table(view, noise_floor[m])

    # detector precision / recall at 10% (Phase 5 style), blind detectors on the corrupted training data
    det_rows = []
    for issue in ISSUES:
        for seed in SEEDS[:3]:
            X_d, y_d, log = CORRUPTORS[issue](X_tr, y_tr, rate=0.10, seed=seed)
            orig = X_d.index
            d = X_d.copy()
            d[target] = y_d.values
            if issue == "feature_corruption":
                res = detect_feature_corruption_crossfeature(X_d, seed=seed)
                got = {(orig[i], c) for c, f in res.extra["per_column_flags"].items()
                       for i in np.flatnonzero(f.reindex(X_d.index, fill_value=False).values)}
                truth = set(zip(log.row_id, log.column))
            else:
                from selfbench import _detected_set
                st = audit_dataset(d, target, detectors={DETECTOR_OF.get(issue, issue)}, seed=seed)
                got, truth = _detected_set(issue, st, orig), _true_set(issue, log)
            p, r = _prf(truth, got)
            det_rows.append({"issue": issue, "seed": seed, "precision": p, "recall": r})
    detection = pd.DataFrame(det_rows).groupby("issue")[["precision", "recall"]].mean().reset_index()

    tag = os.path.join(RESULTS_DIR, f"phase13_{slug}")
    runs.to_csv(f"{tag}_groundtruth_runs.csv", index=False)
    for m in METRICS:
        evidence[m].to_csv(f"{tag}_groundtruth_evidence_{m}.csv", index=False)
    detection.to_csv(f"{tag}_groundtruth_detection.csv", index=False)
    meta = {"dataset": DATASETS[slug], "clean_rows": len(X), "train_rows": len(X_tr), "test_rows": len(X_te),
            "positive_class": str(pos), "positive_share": float((y == pos).mean()),
            "baseline": {str(k): v for k, v in base.items()}, "noise_floor": noise_floor,
            "null_noise_rms": null_noise, "seconds": time.monotonic() - t0}
    json.dump(meta, open(f"{tag}_groundtruth_meta.json", "w"), indent=2)
    print(f"ground truth done in {meta['seconds'] / 60:.1f} min")


# ---------------------------------------------------------------------------------------------------------------
# Calibration under test: what an uploader would get
# ---------------------------------------------------------------------------------------------------------------
def run_calibration(slug: str = PRIMARY, n_seeds: int = 3):
    df, target = load_raw(slug)
    out = {}
    # (a) the file exactly as published
    print(f"[{slug}] calibrating on the RAW file ({len(df):,} rows)")
    out["raw"] = (df, target)
    # (b) a realistically messy upload: a 12k-row sample with all five error types injected at 5%
    X, y = clean_dataset(df, target, CAP_ROWS)
    Xd, yd, _ = build_combined_dirty(X, y, rate=0.05, seed=42)
    dirty = Xd.copy()
    dirty[target] = yd.values
    out["dirty"] = (dirty.reset_index(drop=True), target)
    print(f"[{slug}] calibrating on a DIRTY upload ({len(dirty):,} rows, 5% of each error injected)")
    for name, (frame, tgt) in out.items():
        t0 = time.monotonic()
        state = audit_dataset(frame, tgt)
        sb = run_self_benchmark(state, n_seeds=n_seeds)
        tag = os.path.join(RESULTS_DIR, f"phase13_{slug}_calibration_{name}")
        if sb.ok:
            sb.evidence.to_csv(f"{tag}_evidence.csv", index=False)
            sb.runs.to_csv(f"{tag}_runs.csv", index=False)
            sb.detector_reliability.to_csv(f"{tag}_detection.csv", index=False)
            json.dump({"ok": True, "noise_floor": sb.noise_floor, "reference_rows": sb.reference["rows_used"],
                       "baseline_macro_f1": float(np.mean(list(sb.baseline_f1.values()))),
                       "chance_f1": sb.chance_f1, "seconds": time.monotonic() - t0},
                      open(f"{tag}_meta.json", "w"), indent=2)
            print(f"  {name}: calibrated in {time.monotonic() - t0:.0f}s (noise floor {sb.noise_floor:.4f}, "
                  f"reference {sb.reference['rows_used']:,} rows)")
        else:
            json.dump({"ok": False, "reason": sb.reason}, open(f"{tag}_meta.json", "w"), indent=2)
            print(f"  {name}: DECLINED -- {sb.reason}")


# ---------------------------------------------------------------------------------------------------------------
# Regime checks: small real datasets through the whole product
# ---------------------------------------------------------------------------------------------------------------
def run_regimes():
    from test_taskclean import assert_log_honest
    rows = []
    for slug in ("heart_c", "credit_g"):
        df, target = load_raw(slug)
        df = df.reset_index(drop=True)
        state = audit_dataset(df, target)
        sb = run_self_benchmark(state, n_seeds=3)
        attach_self_benchmark(state, sb)
        res = apply_repairs(state)
        assert_log_honest(df, res.cleaned, res.repair_log, f"{slug}/evidence-based")
        assert_log_honest(df, res.cleaned_aggressive, res.repair_log_aggressive, f"{slug}/aggressive")
        pol = state.policy.set_index("issue_key")
        rows.append({"dataset": DATASETS[slug], "rows": len(df), "columns": df.shape[1],
                     "calibration": "ran" if sb.ok else "declined",
                     "reason_or_noise_floor": f"{sb.noise_floor:.4f}" if sb.ok else sb.reason,
                     "auto_repaired": ", ".join(sorted(default_apply_issues(state))) or "none",
                     "evidence_source_duplicates": pol.loc["duplicates", "evidence_source"],
                     "rows_dropped_evidence_based": len(df) - len(res.cleaned),
                     "cells_changed_aggressive": res.summary_aggressive["cells_modified"],
                     "logs_honest": True})
        print(f"[{slug}] {rows[-1]}")
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS_DIR, "phase13_regime_checks.csv"), index=False)


# ---------------------------------------------------------------------------------------------------------------
# Comparison report
# ---------------------------------------------------------------------------------------------------------------
def safe_decisions(evidence: pd.DataFrame) -> pd.Series:
    """The product's safe-auto rule, applied row by row. Uses noise-aware material risk when the evidence has it
    (calibration and ground truth), else Phase 10's risk (the Adult table)."""
    risk = evidence["material_risk"] if "material_risk" in evidence else evidence["risk"]
    safe = ((risk <= 1e-9) & (evidence["repair_precision"] >= SAFE_REPAIR_MIN_PRECISION)
            & (evidence["repairability_status"] != "negative") & ~evidence["error_type"].isin(NEVER_AUTO_APPLY))
    return pd.Series(safe.values, index=pd.MultiIndex.from_frame(evidence[["error_type", "rate"]]))


def _damage_grid(ev: pd.DataFrame) -> pd.DataFrame:
    return ev.pivot(index="error_type", columns="rate", values="mean_damage")


def run_report(slug: str = PRIMARY):
    tag = os.path.join(RESULTS_DIR, f"phase13_{slug}")
    meta = json.load(open(f"{tag}_groundtruth_meta.json"))
    gt = {m: pd.read_csv(f"{tag}_groundtruth_evidence_{m}.csv") for m in METRICS}
    adult = pd.read_csv(os.path.join(RESULTS_DIR, "phase10_evidence_table.csv"))
    name = meta["dataset"]

    print("=" * 100)
    print(f"{name}: {meta['clean_rows']:,} clean rows (train {meta['train_rows']:,} / test {meta['test_rows']:,}), "
          f"positive class {meta['positive_class']!r} = {meta['positive_share']:.1%}")
    base = pd.DataFrame(meta["baseline"]).T.mean()
    print("clean baseline (mean of 5 seeds):", {m: round(v, 3) for m, v in base.items()},
          "| noise floors:", {m: round(v, 4) for m, v in meta["noise_floor"].items()})
    print("=" * 100)

    print("\nQ2 TRANSFER -- mean damage by error type and rate")
    print("\nAdult, positive-class F1 (Phase 7):")
    print(_damage_grid(adult).round(4).to_string())
    for m, label in (("posf1", "positive-class F1 (the research metric)"), ("macrof1", "macro-F1"),
                     ("auc", "ROC-AUC (threshold-free)")):
        print(f"\n{name}, {label}:")
        print(_damage_grid(gt[m]).round(4).to_string())
    a_flat = _damage_grid(adult).stack()
    rank20 = lambda ev: _damage_grid(ev)[0.20].sort_values(ascending=False).index.tolist()   # noqa: E731
    print("\norder of damage at 20% (largest first):")
    print("  Adult             :", rank20(adult))
    for m in METRICS:
        g_flat = _damage_grid(gt[m]).stack().reindex(a_flat.index)
        print(f"  {name} {m:8s}: {rank20(gt[m])}   (rank corr with Adult over 15 points: "
              f"{spearmanr(a_flat, g_flat)[0]:+.2f})")

    print("\nrepairability at 10% (status, observed harm, repair precision/recall):")
    for label, ev in (("Adult", adult), (f"{name} macro-F1", gt["macrof1"]), (f"{name} posF1", gt["posf1"])):
        sub = ev[ev.rate == 0.10].set_index("error_type")
        print(f"  {label}:")
        for issue in ISSUES:
            r = sub.loc[issue]
            print(f"    {issue:18s} {r.repairability_status:9s} P(harm) {r.p_harm:.2f}  precision {r.repair_precision:.2f}"
                  f"  recall {r.repair_recall:.2f}  mean recovery {r.mean_recovery:+.4f}")

    print("\nPrioritization (Phase 10 rules), 10%, equal weights:")
    for label, ev in (("Adult", adult), (f"{name} posF1", gt["posf1"]), (f"{name} macro-F1", gt["macrof1"])):
        pr = compute_priority(ev, 0.10, 1 / 3, 1 / 3, 1 / 3)
        sens = run_sensitivity_grid(ev, 0.10)
        print(f"  {label}: " + " > ".join(f"{r.error_type}({r.priority_score:+.2f})" for r in pr.itertuples()))
        print(f"      rank 1 in {{{', '.join(f'{r.error_type}: {r.rank_1_pct:.0%}' for r in sens.itertuples() if r.rank_1_pct > 0)}}}"
              f" of 15 weight combos; always last: {[r.error_type for r in sens.itertuples() if r.min_rank == r.max_rank == 5]}")

    # ---- Q3 / Q4: calibration vs ground truth, macro-F1 against macro-F1 -------------------------------------
    sources = {"Adult evidence (transferred)": adult}
    for kind in ("raw", "dirty"):
        pth = f"{tag}_calibration_{kind}_evidence.csv"
        if os.path.exists(pth):
            sources[f"calibration, {kind} upload"] = pd.read_csv(pth)
    truth = safe_decisions(gt["macrof1"])
    truth_pos = safe_decisions(gt["posf1"])
    decidable = [(i, r) for i in ("missing_values", "duplicates", "outliers") for r in RATES]

    print("\n" + "=" * 100)
    print("Q3/Q4 CALIBRATION vs GROUND TRUTH -- safe-auto decision per (repair type, corruption rate)")
    print("(label and feature-anomaly repairs are never auto-applied by design, so only the 9 decidable cells count)")
    print("=" * 100)
    table = pd.DataFrame({"ground truth (macro-F1)": truth.reindex(decidable),
                          "ground truth (posF1)": truth_pos.reindex(decidable)})
    for k, ev in sources.items():
        table[k] = safe_decisions(ev).reindex(decidable)
    print(table.map(lambda v: "SAFE" if v else "review").to_string())
    rows = []
    for k in sources:
        for ref, ref_name in ((truth, "macro-F1 ground truth"), (truth_pos, "posF1 ground truth")):
            d = table[k].astype(bool)
            t = ref.reindex(decidable).astype(bool)
            rows.append({"source": k, "against": ref_name, "agree": f"{int((d == t).sum())}/{len(d)}",
                         "false_approvals": int((d & ~t).sum()), "missed_safe_repairs": int((~d & t).sum())})
    summary = pd.DataFrame(rows)
    print("\n" + summary.to_string(index=False))

    corr_rows = []
    g_macro = _damage_grid(gt["macrof1"]).stack()
    for k, ev in sources.items():
        e = _damage_grid(ev).stack()
        common = e.index.intersection(g_macro.index)
        corr_rows.append({"source": k, "points": len(common),
                          "damage_rank_corr_vs_macro_ground_truth": spearmanr(e.loc[common], g_macro.loc[common])[0]})
    corr = pd.DataFrame(corr_rows)
    print("\ndamage ordering vs macro-F1 ground truth (Spearman):")
    print(corr.round(2).to_string(index=False))

    print("\ndetector precision/recall at 10% -- ground truth (blind detectors, cleaned data):")
    det = pd.read_csv(f"{tag}_groundtruth_detection.csv").set_index("issue")
    print(det.round(2).to_string())
    for kind in ("raw", "dirty"):
        pth = f"{tag}_calibration_{kind}_detection.csv"
        if os.path.exists(pth):
            print(f"calibration ({kind} upload), measured on its own reference:")
            print(pd.read_csv(pth).set_index("error_type")[["precision", "recall"]].round(2).to_string())

    table.to_csv(f"{tag}_comparison_decisions.csv")
    summary.to_csv(f"{tag}_comparison_summary.csv", index=False)
    corr.to_csv(f"{tag}_comparison_damage_corr.csv", index=False)


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    if stage in ("data", "all"):
        for s in DATASETS:
            frame, tgt = fetch_raw(s)
            print(f"saved data/{s}_original.csv {frame.shape} target={tgt!r}")
    if stage in ("groundtruth", "all"):
        run_groundtruth()
    if stage in ("calibration", "all"):
        run_calibration()
    if stage in ("regimes", "all"):
        run_regimes()
    if stage in ("report", "all"):
        run_report()
