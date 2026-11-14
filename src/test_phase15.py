"""
Invariants for the Phase 15 detector (bivariate and multivariate outliers, flag-only) and its injectors.
Run:  .venv/bin/python src/test_phase15.py

Every check asserts something that can fail: planted ground truth, a null that must not be over-flagged, and the
cases the guards exist for.
"""

import numpy as np
import pandas as pd

import joint_outliers as jo
from data import load_adult_clean
from joint_outliers import RECORD_COLUMNS, detect_joint_outliers, eligible_columns
from phase15_injectors import KINDS, inject_joint


def correlated(n=3000, rho=0.9, seed=0, independent=2):
    """a, b share correlation rho (a pair); the rest are independent."""
    rng = np.random.default_rng(seed)
    a = rng.normal(size=n)
    b = rho * a + np.sqrt(1 - rho ** 2) * rng.normal(size=n)
    d = {"a": a, "b": b}
    for i in range(independent):
        d[f"i{i}"] = rng.normal(size=n)
    return pd.DataFrame(d)


def plant_swaps(X, col, n_rows, seed):
    rng = np.random.default_rng(seed)
    rows = rng.choice(len(X), n_rows, replace=False)
    out = X.copy()
    donors = rng.choice(len(X), n_rows, replace=False)
    out.loc[X.index[rows], col] = X[col].to_numpy()[donors]
    return out, set(X.index[rows])


def expected_swap_recall(rho, alpha=0.001, n=400_000, seed=0):
    """Power of a one-pair chi-square test against a swapped value, from the TRUE covariance. A swap makes a independent
    of its row's b, so the squared distance is that of (a', b) with a', b independent standard normals."""
    from scipy.stats import chi2
    z = np.random.default_rng(seed).normal(size=(n, 2))
    P = np.linalg.inv(np.array([[1.0, rho], [rho, 1.0]]))
    return float((np.einsum("ij,jk,ik->i", z, P, z) > chi2.ppf(1 - alpha, 2)).mean())


def test_planted_found_null_not_overflagged():
    X = correlated(3000, rho=0.9, seed=1)

    # (1) a jointly implausible, marginally typical value must be found: a = -2.5 * sign(b) where |b| > 1, which is
    # inside the univariate fences (about +-2.7) but nowhere near what b implies (0.9 * b)
    rng = np.random.default_rng(2)
    cand = X.index[np.abs(X.b) > 1.0]
    planted = set(rng.choice(cand, 80, replace=False))
    Xd = X.copy()
    Xd.loc[list(planted), "a"] = -2.5 * np.sign(X.loc[list(planted), "b"])
    r = detect_joint_outliers(Xd, seed=0)
    flagged = set(r.extra["row_mask_any"][r.extra["row_mask_any"]].index)
    tp = len(flagged & planted)
    assert tp / len(planted) >= 0.9, f"recall {tp / len(planted):.2f} on strongly implausible rows"
    assert tp / len(flagged) >= 0.8, f"precision {tp / len(flagged):.2f}"
    assert "a x b" in {s["scope"] for s in r.extra["scopes"]}
    assert (r.extra["records"].scope == "a x b").any()
    univ_visible = int(r.extra["row_mask_any"][list(planted)].sum() - r.row_mask[list(planted)].sum())
    assert univ_visible <= 0.1 * len(planted), "these values are inside the IQR fences: they must be joint-only"

    # (2) a RANDOM swap is a much weaker anomaly, and the measured recall must agree with the theoretical power of the
    # cut-off (this ties the implementation to the maths instead of to a number I chose)
    Xs, sw = plant_swaps(X, "a", 600, seed=3)
    rs = detect_joint_outliers(Xs, seed=0)
    fl = set(rs.extra["row_mask_any"][rs.extra["row_mask_any"]].index)
    measured = len(fl & sw) / len(sw)
    theory = expected_swap_recall(0.9)
    assert abs(measured - theory) < 0.07, f"measured swap recall {measured:.3f} vs theoretical power {theory:.3f}"

    # (3) null: independent columns, nothing planted -> at most 1% of rows flagged, and no pair scope exists
    rn = detect_joint_outliers(pd.DataFrame(np.random.default_rng(3).normal(size=(3000, 4)), columns=list("wxyz")), seed=0)
    assert rn.extra["n_flagged_any"] <= 0.01 * 3000, rn.extra["n_flagged_any"]
    assert rn.extra["pairs_tested"] == [] and "bivariate" in rn.extra["declined"]
    print(f"OK strongly implausible rows: recall {tp / len(planted):.2f}, precision {tp / len(flagged):.2f}; "
          f"random-swap recall {measured:.3f} matches theory {theory:.3f}; independent null flags {rn.extra['n_flagged_any']}/3000")


def test_attribution():
    """Attribution is the column with the largest standardised conditional residual. That identifies the disturbed
    column when the others pin it down, and it CANNOT in a two-column scope: there z_a^2 - z_b^2 is proportional to
    a^2 - b^2, so the 'top contributor' is just the more extreme of the two. Both facts are asserted."""
    # (1) three columns, c depends on a and b: swapping c leaves a and b individually well explained
    rng = np.random.default_rng(4)
    n = 20_000
    a, b = rng.normal(size=n), rng.normal(size=n)
    X = pd.DataFrame({"a": a, "b": b, "c": 0.6 * a + 0.6 * b + 0.5 * rng.normal(size=n)})
    swap_rows = rng.choice(n, 2000, replace=False)
    Xd = X.copy()
    Xd.loc[Xd.index[swap_rows], "c"] = X["c"].to_numpy()[rng.permutation(n)[:2000]]
    rec = detect_joint_outliers(Xd, seed=0).extra["records"]
    rec = rec[rec.row_id.isin(set(swap_rows)) & (rec.scope == "multivariate")]
    assert len(rec) >= 150, len(rec)
    share = (rec["column"] == "c").mean()
    # chance is 1/3. Measured about 0.62 across seeds even in this favourable structure: better than chance, NOT reliable
    assert share >= 0.5, f"the disturbed column was named in only {share:.2f} of {len(rec)} flagged rows"
    assert set(rec.direction) <= {"higher", "lower"}

    # (2) a correlated PAIR: the named column is the one with the larger absolute normal score, whichever was disturbed
    P = correlated(4000, rho=0.9, seed=5, independent=0)
    Pd, planted = plant_swaps(P, "a", 400, seed=6)
    r = detect_joint_outliers(Pd, seed=0)
    pr = r.extra["records"]
    pr = pr[pr.scope == "a x b"]
    Z = jo.copula(Pd[["a", "b"]].to_numpy(dtype=float))
    bigger = np.where(np.abs(Z[:, 0]) >= np.abs(Z[:, 1]), "a", "b")
    agree = float((pr["column"].to_numpy() == bigger[Pd.index.get_indexer(pr.row_id)]).mean())
    assert len(pr) >= 30 and agree >= 0.95, f"pair attribution follows the larger |score| in {agree:.2f} of {len(pr)} records"
    truly_a = float((pr[pr.row_id.isin(planted)]["column"] == "a").mean())
    print(f"OK attribution: 3-column scope names the disturbed column {share:.0%} of the time; "
          f"in a pair it follows the larger |score| ({agree:.0%}), so it named the disturbed column only {truly_a:.0%}")


def test_eligibility_and_declines():
    rng = np.random.default_rng(0)
    n = 1000
    X = pd.DataFrame({"x": rng.normal(size=n), "y": rng.normal(size=n), "z": rng.normal(size=n),
                      "const": 5.0, "zero_inflated": np.where(rng.random(n) < 0.8, 0.0, rng.normal(size=n)),
                      "binary": rng.integers(0, 2, n), "row_id": np.arange(n, dtype=float), "cat": ["a"] * n})
    cols, skipped = eligible_columns(X)
    assert cols == ["x", "y", "z"], cols
    assert set(skipped) == {"const", "zero_inflated", "binary", "row_id"}, skipped
    assert "cat" not in cols and "cat" not in skipped          # non-numeric columns are simply out of scope

    # fewer than 200 usable rows: declines, flags nothing, does not raise
    small = detect_joint_outliers(X.iloc[:150][["x", "y", "z"]], seed=0)
    assert "multivariate" in small.extra["declined"] and small.rate == 0.0 and small.extra["records"].empty
    # too few columns
    two = detect_joint_outliers(X[["x", "y"]], seed=0)
    assert "needs 3" in two.extra["declined"]["multivariate"]
    # rows with a missing cell are skipped for that scope, and a heavily missing frame declines
    holes = X[["x", "y", "z"]].copy()
    holes.loc[holes.index[:850], "x"] = np.nan
    assert "only 150 rows" in detect_joint_outliers(holes, seed=0).extra["declined"]["multivariate"]
    print("OK eligibility: constant / zero-inflated / binary / identifier columns skipped; small and gappy data decline")


def test_near_duplicate_pair_skipped_and_singular_fit_declines():
    rng = np.random.default_rng(1)
    n = 1500
    a = rng.normal(size=n)
    X = pd.DataFrame({"a": a, "a2": 2 * a + 3, "c": rng.normal(size=n), "d": rng.normal(size=n)})
    r = detect_joint_outliers(X, seed=0)
    assert ("a", "a2") not in {(p[0], p[1]) for p in r.extra["pairs_tested"]}, "rho = 1 pair is a duplicate, not a pair"
    reason = r.extra["declined"]["multivariate"]
    assert "singular" in reason, reason                     # exact linear dependence: declines, never falls back
    assert r.extra["n_flagged_any"] == 0
    print("OK a rho=1 pair is skipped and the singular multivariate fit declines with a reason")


def test_joint_only_rate_and_univariate_flag():
    X = correlated(3000, rho=0.9, seed=7)
    Xd = X.copy()
    # row A: a is far outside its own fences AND opposite to what b implies -> visible to both detectors
    both = X.index[(X.b > 1.5)][:5]
    Xd.loc[both, "a"] = -4.0
    # row B: a lone huge value in an independent column. The copula caps it at ~ +-3.6 normal-score units, which a
    # chi-square cut-off on 4 columns does not flag: the univariate detector is the one that owns this row
    lone = Xd.index[10]
    Xd.loc[lone, "i0"] = 40.0
    r = detect_joint_outliers(Xd, seed=0)
    any_, joint_only = r.extra["row_mask_any"], r.row_mask
    assert any_.loc[both].all(), "an implausible pair value must be flagged jointly"
    assert not joint_only.loc[both].any(), "...but it is univariate-visible too, so it is not counted as joint-only"
    rec = r.extra["records"]
    assert rec[rec.row_id.isin(both)].also_univariate.all()
    assert not bool(any_.loc[lone]), "a lone extreme in an independent column is the univariate detector's, not this one's"
    assert (joint_only <= any_).all() and r.extra["n_joint_only"] == int(joint_only.sum())
    assert abs(r.rate - joint_only.mean()) < 1e-12
    print("OK joint flags on univariate-visible rows are excluded from the joint-only headline rate; "
          "a lone extreme is left to the univariate detector (rank transform bounds its influence)")


def test_flag_only_deterministic_and_pure():
    X = correlated(3000, rho=0.8, seed=8)
    Xd, _ = plant_swaps(X, "a", 60, seed=9)
    snap = Xd.copy()
    r1, r2 = detect_joint_outliers(Xd, seed=3), detect_joint_outliers(Xd, seed=3)
    pd.testing.assert_frame_equal(Xd, snap)                 # input never mutated
    pd.testing.assert_frame_equal(r1.extra["records"], r2.extra["records"])
    assert list(r1.extra["records"].columns) == RECORD_COLUMNS
    assert not any("propos" in c or "repair" in c or "replacement" in c for c in r1.extra["records"].columns), \
        "flag-only: a record must never carry a replacement value"
    print("OK deterministic, input untouched, records carry no proposed value")


def test_large_input_fit_is_sampled_but_all_rows_scored():
    rng = np.random.default_rng(10)
    n = 12_000
    a = rng.normal(size=n)
    X = pd.DataFrame({"a": a, "b": 0.9 * a + 0.43 * rng.normal(size=n), "c": rng.normal(size=n)})
    r = detect_joint_outliers(X, seed=0)
    assert r.extra["scopes"][0]["rows_used"] == n, "every usable row is scored even when the fit is sampled"
    assert r.extra["n_flagged_any"] <= 0.01 * n
    print("OK 12,000 rows: fitted on a sample, all rows scored, null flag rate within 1%")


def test_gating_on_real_data():
    from sklearn.datasets import load_wine
    wine = load_wine(as_frame=True).data
    rw = detect_joint_outliers(wine, seed=0)
    assert len(wine) < jo.MIN_ROWS and rw.rate == 0.0 and not rw.extra["scopes"], "Wine (178 rows) must decline"
    assert "multivariate" in rw.extra["declined"]
    X, _ = load_adult_clean()
    ra = detect_joint_outliers(X, seed=0)
    assert "bivariate" in ra.extra["declined"], "Adult has no pair with |rho| >= 0.5: the bivariate scope must decline"
    print(f"OK gating: Wine declines on rows; Adult bivariate declines (Adult's clean multivariate flag rate is a "
          f"VALIDATION result, criterion F5, not an invariant: {ra.extra['n_flagged_any'] / len(X):.2%} with ties='{ra.extra['ties']}')")


def test_injectors():
    X = correlated(2000, rho=0.9, seed=11, independent=3)
    snap = X.copy()
    y = pd.Series(np.zeros(len(X)))
    Xd, yd, log = inject_joint(X, y, rate=0.10, seed=5)
    pd.testing.assert_frame_equal(X, snap)                  # clean input never mutated
    rows = log.row_id.nunique()
    assert abs(rows - 200) <= 6, rows                       # ~10% of rows (a few ties may be skipped)
    assert set(log.kind) == set(KINDS)
    for _, g in log.groupby("row_id"):
        assert g.kind.nunique() == 1
    # the log is faithful: every logged cell changed to the logged value, every other cell is untouched
    changed = {(r, c) for r, c in zip(log.row_id, log["column"])}
    diff = {(r, c) for c in X.columns for r in X.index[(Xd[c] != X[c]).to_numpy()]}
    assert changed == diff, "log does not match the changed cells"
    for r in log.itertuples():
        assert Xd.at[r.row_id, r.column] == r.corrupted_value and X.at[r.row_id, r.column] == r.original_value
    # swap_block changes at least 2 cells in a row; the other kinds change exactly 1
    per_row = log.groupby("row_id").agg(kind=("kind", "first"), n=("column", "size"))
    assert (per_row[per_row.kind == "swap_block"].n >= 2).all() and (per_row[per_row.kind != "swap_block"].n == 1).all()
    # inflate is the visible control; swaps of a column's own values are mostly invisible to the univariate check
    inflated = log[log.kind == "inflate"]
    assert not inflated.invisible.any(), "an inflated value (4 SD beyond the median) must lie outside the IQR fences"
    assert log[log.kind == "swap_one"].invisible.mean() > 0.9
    # a swapped value is a value that exists in the column: it never leaves the column's own range
    for r in log[log.kind != "inflate"].itertuples():
        assert X[r.column].min() <= r.corrupted_value <= X[r.column].max()
    # deterministic
    _, _, log2 = inject_joint(X, y, rate=0.10, seed=5)
    pd.testing.assert_frame_equal(log, log2)
    _, _, log3 = inject_joint(X, y, rate=0.10, seed=6)
    assert not log.equals(log3)
    print(f"OK injectors: {rows} rows, log faithful, inflate always visible, swaps {log[log.kind == 'swap_one'].invisible.mean():.0%} invisible")


def product_frame(n=3000, seed=21, plant=80):
    rng = np.random.default_rng(seed)
    X = correlated(n, rho=0.9, seed=seed, independent=2)
    X["group"] = rng.choice(["p", "q"], n)
    cand = X.index[np.abs(X.b) > 1.0]
    rows = rng.choice(cand, plant, replace=False)
    X.loc[rows, "a"] = -2.5 * np.sign(X.loc[rows, "b"])
    X["y"] = rng.choice(["no", "yes"], n)
    return X, set(rows)


def test_product_off_by_default_and_flag_only():
    from taskclean import FLAG_ONLY_ISSUES, ISSUES, NEVER_AUTO_APPLY, apply_repairs, audit_dataset
    assert "joint_outliers" in ISSUES and "joint_outliers" in NEVER_AUTO_APPLY and FLAG_ONLY_ISSUES == {"joint_outliers"}
    df, planted = product_frame()

    # off by default: the detector does not run, nothing is proposed, the policy says why
    off = audit_dataset(df, "y", detectors={"joint_outliers", "outliers"}, seed=42)
    assert "joint_outliers" not in off.detections and "off by default" in off.skipped["joint_outliers"]
    assert (off.proposals.issue_key != "joint_outliers").all()
    pr = off.policy.set_index("issue_key").loc["joint_outliers"]
    assert not pr.detector_ran and pr.default_action == "not assessed"

    # opt-in: flags only, never a repair, never applied, even when a human forces it
    on = audit_dataset(df, "y", detectors={"joint_outliers", "outliers"}, seed=42, include_joint_outliers=True)
    prop = on.proposals[on.proposals.issue_key == "joint_outliers"]
    assert len(prop) > 0 and (prop.tier == "loose").all() and prop.proposed_value.isna().all()
    assert set(prop.problem) == {"Unusual combination of values"}
    assert len(set(prop.row_id) & planted) >= 0.9 * len(planted), "the planted jointly-implausible rows must be flagged"
    pol = on.policy.set_index("issue_key").loc["joint_outliers"]
    assert pol.detector_ran and pol.default_action == "flag for human review" and not pol.safe_auto
    assert pol.n_repair_candidates == 0 and pol.n_flag_only == len(prop) and pol.repairability_status == "flag_only"
    assert "calibrated on synthetic" in pol.reason and "NOT a severity" in pol.reason, "the policy must state the validation"
    base = apply_repairs(on, set())
    forced = apply_repairs(on, {"joint_outliers"})
    pd.testing.assert_frame_equal(base.cleaned, forced.cleaned)          # a human override still changes nothing
    jl = forced.repair_log[forced.repair_log.problem == "Unusual combination of values"]
    assert len(jl) == len(prop) and (jl.action == "flagged_only").all()
    assert "joint_outliers" not in forced.readiness["aggressive_variant"]["issues_applied"]
    imp = forced.impact_report.set_index("issue_key").loc["joint_outliers"]
    assert imp.policy_decision == "flagged for human review (not applied)" and np.isnan(imp.benchmark_impact_estimate)
    q = forced.quality_report.set_index("issue_key").loc["joint_outliers"]
    assert q.detector_type == "statistical estimate" and "Flag-only" in q.note
    dim = {d["issue_key"]: d for d in forced.readiness["dimensions"]}["joint_outliers"]
    assert dim["status"] == "Evidence-limited", dim["status"]            # never "Good": there is no validated floor

    # Amendment 3: the product path flags exactly the rows the detector flags on the same frame and seed
    direct = detect_joint_outliers(on.X, seed=42)
    assert set(prop.row_id) == set(direct.row_mask[direct.row_mask].index)
    print(f"OK product: off by default; opt-in flags {prop.row_id.nunique()} rows, applies nothing even when forced, "
          "status Evidence-limited, product == detector")


def test_product_declines_with_a_reason():
    from taskclean import audit_dataset
    df, _ = product_frame(n=120, plant=5)
    st = audit_dataset(df, "y", detectors={"joint_outliers"}, seed=42, include_joint_outliers=True)
    assert "joint_outliers" not in st.detections and "no scope could run" in st.skipped["joint_outliers"]
    assert any("joint-outlier check" in n for n in st.meta["notes"]), "a decline must be visible in the audit notes"
    assert st.policy.set_index("issue_key").loc["joint_outliers", "default_action"] == "not assessed"
    print("OK product: too little data declines visibly (data note and policy reason), no crash")


if __name__ == "__main__":
    test_planted_found_null_not_overflagged()
    test_attribution()
    test_eligibility_and_declines()
    test_near_duplicate_pair_skipped_and_singular_fit_declines()
    test_joint_only_rate_and_univariate_flag()
    test_flag_only_deterministic_and_pure()
    test_large_input_fit_is_sampled_but_all_rows_scored()
    test_gating_on_real_data()
    test_injectors()
    test_product_off_by_default_and_flag_only()
    test_product_declines_with_a_reason()
    print("\nALL PHASE 15 DETECTOR INVARIANTS HOLD")
