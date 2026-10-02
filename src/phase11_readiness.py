"""
AI-Readiness definition and validation (Phase 11A/11B).

Phase 11A: define measurable dimensions, each grounded in machinery this
project already built and already graded -- not a new invented metric.
compute_readiness_dimensions() returns the FULL multidimensional report for
a given dataset; it does not collapse anything into one score. Whether a
single score is even justified is an empirical question answered in 11B
(run_validation below), not assumed up front.

Phase 11B: validate whether the DETECTED (not oracle) signal for each issue
type actually tracks measured ML damage (from Phase 7). This matters
because the readiness report, when run on a real unlabeled dataset, only
ever has detector output to work with -- never the ground truth. If
detected rate doesn't track real damage, a readiness score built on it
would be misleading no matter how good it looks.
"""

import os

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from corruption import CORRUPTORS
from data import load_adult_clean, three_way_split
from detect import (detect_missing, detect_duplicates, detect_outliers_iqr, detect_inconsistency,
                     detect_label_noise, detect_feature_corruption_crossfeature)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
SPLIT_SEED = 42
VALIDATION_RATES = [0.05, 0.10, 0.20]

# Phase 7's measured Impact_e(rate) curve, read from the saved summary --
# used to translate a DETECTED rate into an ESTIMATED F1 cost, rather than
# reporting a raw percentage with no task-relevance attached.
_impact_curve_cache = None


def _impact_curve() -> pd.DataFrame:
    global _impact_curve_cache
    if _impact_curve_cache is None:
        _impact_curve_cache = pd.read_csv(os.path.join(RESULTS_DIR, "phase7_impact_summary_full.csv"))
    return _impact_curve_cache


def _estimate_impact(error_type: str, detected_rate: float) -> tuple[float, bool]:
    """Linearly interpolate Phase 7's measured damage curve (fit at 5/10/20%)
    to estimate the F1 cost of a detected rate that doesn't land exactly on
    one of those three points. Returns (estimate, is_extrapolated).

    A (0, 0) anchor is added -- zero corruption should mean ~zero added
    damage -- so a detected rate BELOW 5% (common on real data, where
    tested-range extrapolation would otherwise silently clamp to the 5%
    curve value and overstate a small problem) interpolates down toward
    zero instead. np.interp still clamps for rates ABOVE 20% (our highest
    tested point), which is flagged as `is_extrapolated=True` rather than
    presented as a precise figure -- it understates impact for severely
    out-of-range rates and should be read as "at least this much."
    """
    curve = _impact_curve()[_impact_curve().error_type == error_type].sort_values("rate")
    rates = np.concatenate([[0.0], curve["rate"].values])
    damages = np.concatenate([[0.0], curve["mean_damage"].values])
    estimate = float(np.interp(detected_rate, rates, damages))
    is_extrapolated = detected_rate > rates.max()
    return estimate, is_extrapolated


def _repair_guidance(error_type: str, rate: float) -> dict:
    """Phase 10's repairability_status/risk for the closest evaluated rate."""
    path = os.path.join(RESULTS_DIR, "phase10_evidence_table.csv")
    if not os.path.exists(path):
        return {"repairability_status": "unknown", "risk": None}
    evidence = pd.read_csv(path)
    sub = evidence[evidence.error_type == error_type]
    if sub.empty:
        return {"repairability_status": "unknown", "risk": None}
    closest = sub.iloc[(sub["rate"] - rate).abs().argsort().iloc[0]]
    return {"repairability_status": closest["repairability_status"], "risk": float(closest["risk"])}


def compute_readiness_dimensions(X: pd.DataFrame, y: pd.Series, seed: int = SPLIT_SEED) -> dict:
    """Phase 11A: the full multidimensional readiness report for a dataset.
    Nothing is reduced to one number here.
    """
    missing = detect_missing(X)
    duplicates = detect_duplicates(X)
    outliers = detect_outliers_iqr(X)
    inconsistency = detect_inconsistency(X)
    label_noise = detect_label_noise(X, y, seed=seed)
    feature_corruption = detect_feature_corruption_crossfeature(X, seed=seed)

    detected = {
        "missing_values": missing.rate, "duplicates": duplicates.rate,
        "outliers": outliers.rate, "label_errors": label_noise.rate,
        "feature_corruption": feature_corruption.rate,
    }

    dimensions = {
        "completeness": 1 - missing.rate,
        "consistency": 1 - inconsistency.rate,
        "uniqueness": 1 - duplicates.rate,
        "anomaly_burden": outliers.rate,
        "label_reliability": 1 - label_noise.rate,
        "feature_reliability": 1 - feature_corruption.rate,
    }

    task_impact = {}
    repair_guidance = {}
    for error_type, rate in detected.items():
        estimate, is_extrapolated = _estimate_impact(error_type, rate)
        task_impact[error_type] = {"estimate": estimate, "is_extrapolated": is_extrapolated}
        repair_guidance[error_type] = _repair_guidance(error_type, rate)

    return {
        "n_rows": len(X), "n_columns": X.shape[1],
        "detected_rates": detected,
        "quality_dimensions": dimensions,
        "estimated_task_impact": task_impact,
        "repair_guidance": repair_guidance,
    }


def run_validation(rates=VALIDATION_RATES, seed: int = SPLIT_SEED) -> pd.DataFrame:
    """Phase 11B: for each error type x rate, inject ONLY that error, run
    its matching detector (blind, as a real system would), and record the
    DETECTED rate alongside Phase 7's MEASURED damage at that same
    error_type/rate. This is the evidence that answers "does a detected
    quality signal actually track real ML damage" -- not assumed, checked.
    """
    X, y = load_adult_clean()
    X_train, X_val, X_test, y_train, y_val, y_test = three_way_split(X, y, seed=SPLIT_SEED)
    impact_curve = _impact_curve()

    detector_fns = {
        "missing_values": lambda Xd, yd: detect_missing(Xd).rate,
        "duplicates": lambda Xd, yd: detect_duplicates(Xd).rate,
        "outliers": lambda Xd, yd: detect_outliers_iqr(Xd).rate,
        "label_errors": lambda Xd, yd: detect_label_noise(Xd, yd, seed=seed).rate,
        "feature_corruption": lambda Xd, yd: detect_feature_corruption_crossfeature(Xd, seed=seed).rate,
    }

    rows = []
    for error_type, corrupt_fn in CORRUPTORS.items():
        for rate in rates:
            X_dirty, y_dirty, _log = corrupt_fn(X_train, y_train, rate=rate, seed=seed)
            detected_rate = detector_fns[error_type](X_dirty, y_dirty)
            measured = impact_curve[(impact_curve.error_type == error_type) & (impact_curve.rate == rate)]
            measured_damage = float(measured["mean_damage"].iloc[0])
            rows.append({
                "error_type": error_type, "true_rate": rate,
                "detected_rate": detected_rate, "measured_damage": measured_damage,
            })
            print(f"  {error_type:20s} true_rate={rate:.0%}  detected_rate={detected_rate:.3f}  "
                  f"measured_damage={measured_damage:+.4f}")

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 11C: Multidimensional readiness report
# ---------------------------------------------------------------------------

# Locked in from the Phase 11B validation run (see phase11b_correlation.csv).
# NOT recomputed per report -- this is a one-time empirical finding about
# each detector's relationship to real measured damage, n=3 rates each
# (too small to claim statistical significance; status captures direction/
# reliability, not a p-value claim).
EVIDENCE_STATUS = {
    "missing_values": "weak",
    "duplicates": "weak",
    "outliers": "weak",
    "label_errors": "inverted",
    "feature_corruption": "weak",
    "inconsistency": "not_estimable",
}

EVIDENCE_NOTES = {
    "missing_values": "Weak/inconclusive relationship with measured impact across tested rates "
                       "(r=0.61, n=3 -- not statistically significant at this sample size).",
    "duplicates": "Direction consistent with impact rising at higher rates, but not statistically "
                  "significant at n=3 (r=0.78).",
    "outliers": "Strongest observed trend among all dimensions (r=0.99, n=3) but still not "
                "statistically significant at this sample size; detector also has a known high "
                "false-positive baseline even on clean data (Phase 5/6).",
    "label_errors": "INVERTED relationship: detected rate DECREASES as true corruption (and real "
                     "measured damage) increases, most likely because the cross-validated detector's "
                     "own model is degraded by the label noise it's trying to audit. Do not use the "
                     "detected rate as a proxy for label-error severity.",
    "feature_corruption": "Strong observed trend (r=0.95, n=3, not statistically significant) but the "
                           "detector has a known high false-positive baseline (~50%+ flagged even on "
                           "clean data -- see Phase 5/6); treat the raw flagged rate cautiously.",
    "inconsistency": "Not independently validated -- no corruption type exists in this project to test "
                      "this detector against a known ground truth.",
}

RECOMMENDATION_TEXT = {
    "positive": "Repair recommended; evidence from controlled testing supports a net ML benefit.",
    "negative": "Automatic repair NOT recommended under the evaluated repair strategy -- controlled "
                "testing demonstrated it is net-harmful. This does not mean the underlying error is "
                "inherently unrepairable, only that the strategy we evaluated was harmful here. "
                "Flag for manual review instead of auto-repairing.",
    "weak": "Repair evidence is inconclusive at this severity; review before repairing automatically.",
    "unknown": "No repair evidence available for this error type/rate.",
}

LABELS = {
    "missing_values": "Completeness", "duplicates": "Uniqueness", "outliers": "Anomaly burden",
    "label_errors": "Label reliability", "feature_corruption": "Feature reliability",
    "inconsistency": "Consistency",
}

BASELINE_RATES_PATH = os.path.join(RESULTS_DIR, "phase11_baseline_rates.json")


def _load_baseline_rates() -> dict:
    import json
    if not os.path.exists(BASELINE_RATES_PATH):
        return {}
    with open(BASELINE_RATES_PATH) as f:
        raw = json.load(f)
    # detect.py's run_full_audit names its label detector "label_noise";
    # corruption/ and everywhere else in this project use "label_errors"
    # (same mismatch already fixed once in quality_report.py -- see its
    # AUDIT_NAME_TO_BENCHMARK_NAME comment for why both names are kept).
    if "label_noise" in raw:
        raw["label_errors"] = raw.pop("label_noise")
    return raw


def _status(error_type: str, observed_rate: float, baseline_rate: float | None, evidence_status: str) -> str:
    """Status is NOT just "is the rate high" -- it's gated by how much we can
    trust the signal at all (evidence_status), THEN how far above this
    detector's own known false-positive floor (baseline_rate, measured on
    genuinely clean data with the same pipeline) the observed rate sits.
    """
    if evidence_status == "inverted":
        return "High uncertainty"
    if evidence_status == "not_estimable":
        return "Evidence-limited"
    if baseline_rate is None:
        return "Unknown"
    if observed_rate > baseline_rate * 1.5 + 0.02:
        return "Attention"
    return "Good"


def generate_readiness_report(X: pd.DataFrame, y: pd.Series, dataset_name: str, target_name: str,
                               seed: int = SPLIT_SEED) -> dict:
    dims = compute_readiness_dimensions(X, y, seed=seed)
    baseline_rates = _load_baseline_rates()

    dimension_rows = []
    for error_type, observed_rate in dims["detected_rates"].items():
        evidence_status = EVIDENCE_STATUS.get(error_type, "not_estimable")
        baseline_rate = baseline_rates.get(error_type)
        status = _status(error_type, observed_rate, baseline_rate, evidence_status)
        dimension_rows.append({
            "dimension": LABELS.get(error_type, error_type),
            "error_type": error_type,
            "observed_rate": observed_rate,
            "baseline_rate": baseline_rate,
            "evidence_status": evidence_status,
            "evidence_note": EVIDENCE_NOTES.get(error_type, ""),
            "status": status,
        })
    # inconsistency has its own detector but isn't in CORRUPTORS/detected_rates
    # the same way (no corruption to compare against) -- include separately
    inconsistency_rate = 1 - dims["quality_dimensions"]["consistency"]
    dimension_rows.append({
        "dimension": "Consistency", "error_type": "inconsistency",
        "observed_rate": inconsistency_rate, "baseline_rate": baseline_rates.get("inconsistency"),
        "evidence_status": EVIDENCE_STATUS["inconsistency"],
        "evidence_note": EVIDENCE_NOTES["inconsistency"],
        "status": _status("inconsistency", inconsistency_rate, baseline_rates.get("inconsistency"),
                           EVIDENCE_STATUS["inconsistency"]),
    })

    task_impact_rows = []
    for error_type, est_impact in dims["estimated_task_impact"].items():
        guidance = dims["repair_guidance"][error_type]
        task_impact_rows.append({
            "error_type": error_type, "detected_rate": dims["detected_rates"][error_type],
            "benchmark_impact_estimate": est_impact["estimate"], "is_boundary_estimate": est_impact["is_extrapolated"],
            "repairability_status": guidance["repairability_status"], "risk": guidance["risk"],
            "recommendation": RECOMMENDATION_TEXT.get(guidance["repairability_status"], RECOMMENDATION_TEXT["unknown"]),
        })

    return {
        "dataset_name": dataset_name, "target_name": target_name,
        "n_rows": dims["n_rows"], "n_columns": dims["n_columns"],
        "model_used": "RandomForestClassifier (n_estimators=300) -- the model this project validated against",
        "dimensions": dimension_rows,
        "task_impact": task_impact_rows,
    }


def print_readiness_report(report: dict):
    print("=" * 100)
    print(f"TASKCLEAN AI-READINESS REPORT")
    print("=" * 100)
    print("1. DATASET OVERVIEW")
    print("-" * 100)
    print(f"  Dataset:     {report['dataset_name']}")
    print(f"  Rows:        {report['n_rows']:,}")
    print(f"  Columns:     {report['n_columns']}")
    print(f"  Target:      {report['target_name']}")
    print(f"  Task type:   Binary classification")
    print(f"  Model used for validation: {report['model_used']}")

    print(f"\n2. READINESS DIMENSIONS")
    print("-" * 100)
    print(f"  {'Dimension':22s} {'Observed':>10s}  {'Baseline':>10s}  {'Evidence':14s} {'Status':18s}")
    for d in report["dimensions"]:
        baseline_str = f"{d['baseline_rate']*100:.1f}%" if d["baseline_rate"] is not None else "n/a"
        print(f"  {d['dimension']:22s} {d['observed_rate']*100:9.1f}%  {baseline_str:>10s}  "
              f"{d['evidence_status']:14s} {d['status']:18s}")
    print("\n  Evidence notes (do NOT read 'observed rate' as a validity claim -- see Phase 6/11B):")
    for d in report["dimensions"]:
        print(f"    - {d['dimension']}: {d['evidence_note']}")

    print(f"\n3. TASK-SPECIFIC ESTIMATED IMPACT")
    print("   (Benchmark-based task-impact estimate: the detected rate mapped onto the controlled")
    print("    Phase 7 damage curve -- Adult/RandomForest. NOT a measured or validated prediction")
    print("    of F1 loss for this specific dataset instance)")
    print("-" * 100)
    for t in report["task_impact"]:
        flag = ("  [boundary estimate: observed rate exceeds the experimentally validated 0-20% range; "
                "value is the 20% benchmark, not an extrapolated prediction]" if t["is_boundary_estimate"] else "")
        print(f"  {LABELS.get(t['error_type'], t['error_type']):22s} "
              f"detected={t['detected_rate']*100:5.1f}%  benchmark impact estimate ~= "
              f"{t['benchmark_impact_estimate']:+.4f}{flag}")

    print(f"\n4. REPAIR GUIDANCE (from Phase 10)")
    print("-" * 100)
    print(f"  {'Issue':22s} {'Repair status':14s} Recommendation")
    for t in report["task_impact"]:
        print(f"  {LABELS.get(t['error_type'], t['error_type']):22s} {t['repairability_status']:14s} "
              f"{t['recommendation']}")

    print(f"\n5. LIMITATIONS")
    print("-" * 100)
    print("  - This report is ESTIMATE-BASED, not a measured ground-truth audit: on a real dataset,")
    print("    true error locations/rates are unknown, so every number above comes from a detector")
    print("    with its own known precision/recall limits (Phase 5) and its own relationship (or lack")
    print("    thereof) to real ML impact (Phase 11B, n=3 rates per error type -- not enough for")
    print("    statistical significance, directional evidence only).")
    print("  - Task-impact estimates are specific to the Adult dataset + RandomForest combination this")
    print("    project validated against; they do not transfer to a different dataset or model.")
    print("  - label_errors' detected rate is actively MISLEADING (inverted relationship) -- treat any")
    print("    report showing a low label-error rate with suspicion rather than reassurance.")
    print("  - No single composite 'readiness score' is reported: Phase 11B's pooled correlation across")
    print("    all detected rates vs. measured damage was r=-0.24 (not significant), which does not")
    print("    provide sufficient evidence that an aggregate score built from raw detector rates would")
    print("    be meaningful. This does not prove no single score could ever work -- only that the")
    print("    naive aggregate approach tested here isn't justified by the evidence collected.")


if __name__ == "__main__":
    print("=" * 90)
    print("PHASE 11B -- Does detected quality signal track measured ML damage?")
    print("=" * 90)
    validation = run_validation()
    validation.to_csv(os.path.join(RESULTS_DIR, "phase11b_validation.csv"), index=False)

    print("\n" + "=" * 90)
    print("Correlation: detected_rate vs. measured_damage, PER ERROR TYPE (n=3 rates each)")
    print("=" * 90)
    corr_rows = []
    for error_type, g in validation.groupby("error_type"):
        if g["detected_rate"].std() < 1e-9 or g["measured_damage"].std() < 1e-9:
            pearson_r, pearson_p, spearman_r = np.nan, np.nan, np.nan
        else:
            pearson_r, pearson_p = pearsonr(g["detected_rate"], g["measured_damage"])
            spearman_r, _ = spearmanr(g["detected_rate"], g["measured_damage"])
        corr_rows.append({"error_type": error_type, "pearson_r": pearson_r,
                           "pearson_p": pearson_p, "spearman_r": spearman_r})
    corr_df = pd.DataFrame(corr_rows)
    print(corr_df.round(3).to_string(index=False))

    print("\nOverall (pooled across all 15 points, all error types together):")
    overall_pearson_r, overall_pearson_p = pearsonr(validation["detected_rate"], validation["measured_damage"])
    print(f"  Pearson r = {overall_pearson_r:.3f} (p={overall_pearson_p:.3f})")

    corr_df.to_csv(os.path.join(RESULTS_DIR, "phase11b_correlation.csv"), index=False)
    print(f"\nSaved phase11b_validation.csv and phase11b_correlation.csv")
