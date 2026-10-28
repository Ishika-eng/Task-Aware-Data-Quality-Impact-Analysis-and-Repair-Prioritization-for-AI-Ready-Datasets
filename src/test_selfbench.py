"""
Tests for the self-benchmark (tier 2: calibrating TaskClean on the uploaded dataset).

Run:  python3 test_selfbench.py      (about a minute)
"""

import json
import os
import warnings

import numpy as np
import pandas as pd

from selfbench import (SelfBench, _combine_evidence, _metric_disagreement, classify_repairability, harm_risk,
                       material_harm_risk, run_self_benchmark)
from taskclean import (MAX_CERTIFIABLE_NOISE_FLOOR, apply_repairs, attach_self_benchmark, audit_dataset,
                       outputs_as_bytes)
from test_taskclean import assert_log_honest

warnings.filterwarnings("ignore")
RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")


def synthetic(n=3500, seed=1, classes=2):
    """A dataset whose target genuinely depends on the features, plus some natural mess."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "x1": rng.normal(0, 1, n), "x2": rng.normal(0, 1, n), "x3": rng.normal(0, 1, n),
        "x4": rng.uniform(0, 10, n),
        "grade": rng.choice(["a", "b", "c"], n), "region": rng.choice(["n", "s", "e", "w"], n),
    })
    score = (df.x1 + 0.8 * df.x2 - 0.5 * df.x3 + 0.15 * df.x4
             + df.grade.map({"a": 0.8, "b": 0.0, "c": -0.8}) + rng.normal(0, 0.6, n))
    if classes == 2:
        df["y"] = np.where(score > 0.3, "yes", "no")
    else:
        df["y"] = pd.cut(score, bins=[-99, -0.5, 0.8, 99], labels=["low", "mid", "high"]).astype(str)
    df.loc[rng.choice(n, 120, replace=False), "x2"] = np.nan
    df.loc[rng.choice(n, 90, replace=False), "region"] = np.nan
    df.loc[rng.choice(n, 15, replace=False), "x4"] *= 40
    return pd.concat([df, df.iloc[:70]], ignore_index=True)


def test_rules_match_adult_phase10():
    """The evidence rules in selfbench.py must reproduce the Phase 10 table computed from the Adult runs."""
    raw = pd.read_csv(os.path.join(RESULTS, "phase9_repair_raw_full.csv"))
    expected = pd.read_csv(os.path.join(RESULTS, "phase10_evidence_table.csv")).set_index(["error_type", "rate"])
    for (issue, rate), g in raw.groupby(["error_type", "rate"]):
        rec = g["recovery"].values
        exp = expected.loc[(issue, rate)]
        assert classify_repairability(rec, 0.003) == exp["repairability_status"], (issue, rate)
        assert np.isclose(harm_risk(rec)[2], exp["risk"]), (issue, rate)
    print("OK rules: status and risk reproduce all 15 rows of the Adult Phase 10 evidence table")

    rec = [-0.004, -0.002, 0.001]                      # small negatives: harm by Phase 10's definition ...
    assert harm_risk(rec)[2] > 0
    assert material_harm_risk(rec, noise_floor=0.01) == (0.0, 0.0), "... but not material when noise is larger"
    assert material_harm_risk([-0.05, 0.0, 0.01], 0.01)[0] > 0
    print("OK material harm only counts beyond the noise floor")


def test_exact_restoration_is_never_harm():
    """Exact restoration of the clean data cannot be harmful: a negative 'recovery' is the corruption helping by
    chance. Non-exact repairs keep the strict material-harm rule."""
    from selfbench import _evidence_table
    base = {"issue": "duplicates", "rate": 0.05, "repair_precision": 1.0, "repair_recall": 1.0,
            "baseline_f1": 0.8, "detected_rate": 0.05, "dirty_f1": 0.0, "repaired_f1": 0.0}
    lucky = pd.DataFrame([{**base, "seed": 1, "damage": -0.017, "recovery": -0.017, "restored_exactly": True},
                          {**base, "seed": 2, "damage": -0.004, "recovery": -0.004, "restored_exactly": True},
                          {**base, "seed": 3, "damage": 0.006, "recovery": 0.006, "restored_exactly": True}])
    row = _evidence_table(lucky, noise_floor=0.004).iloc[0]
    assert row.repairability_status == "weak" and row.material_risk == 0 and bool(row.restored_exactly)
    assert row.p_harm > 0, "Phase 10's raw risk is still reported, unchanged"
    real_gain = lucky.assign(damage=[0.02, 0.015, 0.03], recovery=[0.02, 0.015, 0.03])
    assert _evidence_table(real_gain, 0.004).iloc[0].repairability_status == "positive"

    inexact = lucky.assign(restored_exactly=False, repair_precision=0.99)       # e.g. imputation: harm is real harm
    row = _evidence_table(inexact, noise_floor=0.004).iloc[0]
    assert row.material_risk > 0, "a non-exact repair that loses more than the noise floor is harm"
    print("OK exact restoration never counts as harm; inexact repairs keep the strict rule")


def _ev_row(issue, rate=0.05, status="weak", material=0.0, exact=False, precision=1.0, damage=0.01, recovery=0.0):
    return {"error_type": issue, "rate": rate, "mean_damage": damage, "damage_std": 0.0, "repair_precision": precision,
            "repair_recall": 1.0, "mean_recovery": recovery, "sign_agreement": 1.0, "n_pos_seeds": 0, "n_neg_seeds": 0,
            "repairability_status": status, "evidence_status": "partial", "p_harm": 0.0, "harm_magnitude": 0.0,
            "risk": 0.0, "p_material_harm": float(material > 0), "material_risk": material, "restored_exactly": exact}


def test_two_metric_evidence_and_power_guard():
    """Fix 1: a repair is safe only if it is safe under BOTH macro-F1 and ROC-AUC. Fix 2: with a large noise floor a
    non-exact repair cannot be certified safe, while exact restoration (duplicate removal) still can."""
    f1 = pd.DataFrame([_ev_row("outliers", status="weak"), _ev_row("missing_values", status="weak")])
    auc = pd.DataFrame([_ev_row("outliers", status="negative", material=0.02, recovery=-0.03),
                        _ev_row("missing_values", status="weak")])
    both = _combine_evidence(f1, auc, floor_f1=0.004, floor_auc=0.003).set_index("error_type")
    assert both.loc["outliers", "repairability_status"] == "negative", "harm on EITHER metric makes it negative"
    assert both.loc["outliers", "material_risk"] == 0.02, "material risk is the worse of the two metrics"
    assert both.loc["missing_values", "repairability_status"] == "weak"
    assert not both.underpowered.any(), "small noise floors must not trigger the guard"

    noisy = _combine_evidence(pd.DataFrame([_ev_row("missing_values"), _ev_row("duplicates", exact=True)]),
                              pd.DataFrame([_ev_row("missing_values"), _ev_row("duplicates", exact=True)]),
                              floor_f1=0.016, floor_auc=0.003).set_index("error_type")
    assert bool(noisy.loc["missing_values", "underpowered"]) and not bool(noisy.loc["duplicates", "underpowered"]), \
        "the guard blocks inexact repairs but exempts exact restoration"
    just_under = _combine_evidence(pd.DataFrame([_ev_row("missing_values")]), None,
                                   floor_f1=MAX_CERTIFIABLE_NOISE_FLOOR - 1e-6, floor_auc=None)
    assert not just_under.underpowered.any(), "at or below the cap is certifiable"

    # through the policy: same evidence, noisy vs quiet
    state = audit_dataset(synthetic(), "y")
    for floor, expect_auto in ((0.016, False), (0.004, True)):
        rows = [_ev_row(i, r) for i in ("missing_values", "outliers", "label_errors") for r in (0.05, 0.10, 0.20)]
        rows += [_ev_row("duplicates", r, exact=True) for r in (0.05, 0.10, 0.20)]
        ev = _combine_evidence(pd.DataFrame(rows), pd.DataFrame(rows), floor_f1=floor, floor_auc=floor)
        attach_self_benchmark(state, SelfBench(ok=True, evidence=ev))
        pol = state.policy.set_index("issue_key")
        assert pol.loc["duplicates", "default_action"] == "auto-repair", "exact dedup is certifiable at any noise"
        assert (pol.loc["missing_values", "default_action"] == "auto-repair") is expect_auto
        if not expect_auto:
            assert "cannot be certified safe" in pol.loc["missing_values", "reason"]
    print("OK two-metric evidence (worst of F1/AUC) and the power guard (blocks noisy approvals, exempts exact repairs)")


def test_metric_disagreement_flag():
    imbalanced = pd.DataFrame({"error_type": ["label_errors"] * 3 + ["duplicates"] * 3 + ["outliers"] * 3,
                               "rate": [0.05, 0.10, 0.20] * 3,
                               "mean_damage": [-0.005, -0.011, -0.008, 0.001, 0.0, 0.001, 0.007, 0.012, 0.015],
                               "mean_damage_auc": [0.013, 0.026, 0.050, 0.0, 0.0, 0.0, 0.0, -0.001, 0.001]})
    flagged = _metric_disagreement(imbalanced, floor_f1=0.007, floor_auc=0.003, minority_share=0.117).set_index("error_type")
    assert "label_errors" in flagged.index and "ROC-AUC shows real damage" in flagged.loc["label_errors", "note"]
    assert "imbalanced" in flagged.loc["label_errors", "note"], "the imbalance cause is named when the minority class is rare"
    assert "outliers" in flagged.index and "does not" in flagged.loc["outliers", "note"], "the opposite pattern is flagged too"
    assert "duplicates" not in flagged.index, "agreement (no damage on either) is not flagged"
    balanced = _metric_disagreement(imbalanced, 0.007, 0.003, minority_share=0.45).set_index("error_type")
    assert "imbalanced" not in balanced.loc["label_errors", "note"]
    print("OK F1/AUC disagreement is flagged in both directions, with the imbalance cause only when it applies")


def test_calibrates_and_drives_policy():
    df = synthetic()
    state = audit_dataset(df, "y")
    sb = run_self_benchmark(state, n_seeds=3)
    assert sb.ok, sb.reason
    assert sb.noise_floor >= 0.003 and not np.isnan(sb.null_noise)
    ev = sb.evidence
    needed = {"error_type", "rate", "mean_damage", "repair_precision", "repair_recall", "mean_recovery",
              "repairability_status", "risk", "material_risk", "p_harm"}
    assert needed <= set(ev.columns), "evidence must follow the Phase 10 schema (plus material risk)"
    assert sb.has_auc and sb.noise_floor_auc >= 0.003, "ROC-AUC must be measured alongside macro-F1"
    assert {"status_macrof1", "status_auc", "mean_damage_auc", "material_risk_auc", "underpowered",
            "noise_floor", "noise_floor_auc"} <= set(ev.columns)
    assert np.isfinite(ev.mean_damage_auc).all() and "mean_damage_auc" in sb.impact_curve
    assert sb.metric_disagreement is not None and "minority_class_share" in sb.reference
    assert set(ev.error_type) == {"missing_values", "duplicates", "outliers", "label_errors"}
    assert set(ev.repairability_status) <= {"positive", "weak", "negative"}
    dmg = sb.impact_curve.set_index(["error_type", "rate"])["mean_damage"]
    assert dmg[("label_errors", 0.20)] > dmg[("duplicates", 0.20)] + 0.02, \
        "20% label noise must hurt far more than 20% duplicates on this dataset"
    assert dmg[("label_errors", 0.20)] > dmg[("label_errors", 0.05)], "damage must grow with the corruption rate"
    dup = ev[ev.error_type == "duplicates"]
    assert (dup.repair_precision > 0.99).all() and (dup.repair_recall > 0.99).all(), "exact dedup is exact"
    assert dup.restored_exactly.all() and (dup.material_risk == 0).all() and (dup.repairability_status != "negative").all()
    assert set(sb.baseline_floors) == {"missing_values", "duplicates", "outliers", "label_errors"}
    assert sb.baseline_floors["missing_values"] == 0 and sb.baseline_floors["duplicates"] == 0

    attach_self_benchmark(state, sb)
    pol = state.policy.set_index("issue_key")
    assert (pol.loc[["missing_values", "duplicates", "outliers", "label_errors"], "evidence_source"]
            == "this dataset (self-benchmark)").all()
    assert pol.loc["duplicates", "default_action"] == "auto-repair"
    assert pol.loc["label_errors", "default_action"] != "auto-repair", "label repair is never automatic"
    assert "never applied automatically" in pol.loc["label_errors", "reason"]
    print(f"OK calibrated in {sb.seconds:.0f}s: noise floor {sb.noise_floor:.4f}, evidence schema, damage ordering, "
          "policy driven by this dataset's evidence")

    # reports use the dataset's own numbers and both cleaned files stay honest
    res = apply_repairs(state)
    q = res.quality_report.set_index("issue_key")
    assert q.loc["label_errors", "reliability_source"] == "this dataset (self-benchmark)"
    imp = res.impact_report.set_index("issue_key")
    assert imp.loc["label_errors", "impact_source"] == "this dataset (self-benchmark)"
    assert imp.loc["label_errors", "impact_unit"] == "macro-F1"
    assert imp.loc["feature_corruption", "impact_source"] in ("-", "UCI Adult benchmark")
    files = outputs_as_bytes(res)
    assert "selfbenchmark_evidence.csv" in files and "selfbenchmark_runs.csv" in files
    report = json.loads(files["readiness_report.json"])
    assert report["self_benchmark"]["ran"] is True and report["self_benchmark"]["evidence"]
    assert "noise_floor_auc" in report["self_benchmark"] and "metric_disagreement" in report["self_benchmark"]
    assert "impact_estimate_auc" in report["benchmark_task_impact"][0]
    assert any("CLEAN-ISH" in t for t in report["limitations"])
    assert_log_honest(df, res.cleaned, res.repair_log, "calibrated/evidence-based")
    assert_log_honest(df, res.cleaned_aggressive, res.repair_log_aggressive, "calibrated/aggressive")
    print("OK reports, output files, readiness JSON and both repair logs reflect the calibration")


def test_declines_when_evidence_would_be_meaningless():
    # too little clean-ish data
    small = synthetic(n=400)
    state = audit_dataset(small, "y")
    sb = run_self_benchmark(state, n_seeds=2)
    assert not sb.ok and "clean-ish rows" in sb.reason, sb.reason
    attach_self_benchmark(state, sb)
    pol = state.policy.set_index("issue_key")
    assert (pol.loc["duplicates", "evidence_source"] == "UCI Adult benchmark"), "must fall back to Adult evidence"
    res = apply_repairs(state)
    rep = json.loads(outputs_as_bytes(res)["readiness_report.json"])
    assert rep["self_benchmark"]["ran"] is False and "clean-ish rows" in rep["self_benchmark"]["reason"]
    assert any("Self-benchmark did not run" in t for t in rep["limitations"])
    assert "selfbenchmark_evidence.csv" not in outputs_as_bytes(res)

    # a target that is pure noise: damage cannot be measured, so "no harm observed" would be vacuous
    rng = np.random.default_rng(3)
    noise = synthetic()
    noise["y"] = rng.choice(["yes", "no"], len(noise))
    sb = run_self_benchmark(audit_dataset(noise, "y"), n_seeds=2)
    assert not sb.ok and "barely predictable" in sb.reason, sb.reason

    # no target
    sb = run_self_benchmark(audit_dataset(synthetic(), None), n_seeds=2)
    assert not sb.ok and "no target" in sb.reason
    print("OK declines (and falls back to Adult evidence) for tiny data, an unpredictable target, and no target")


def test_multiclass():
    df = synthetic(classes=3)
    sb = run_self_benchmark(audit_dataset(df, "y"), n_seeds=2)
    assert sb.ok, sb.reason
    assert (sb.evidence[sb.evidence.error_type == "label_errors"].mean_damage.max() > 0.01)
    print("OK multi-class target calibrates (label corruption flips to a random other class)")


if __name__ == "__main__":
    test_rules_match_adult_phase10()
    test_exact_restoration_is_never_harm()
    test_two_metric_evidence_and_power_guard()
    test_metric_disagreement_flag()
    test_calibrates_and_drives_policy()
    test_declines_when_evidence_would_be_meaningless()
    test_multiclass()
    print("\nALL SELF-BENCHMARK TESTS PASS")
