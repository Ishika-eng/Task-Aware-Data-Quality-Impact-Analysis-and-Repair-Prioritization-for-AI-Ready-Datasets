"""
Invariants for the Phase 17 harness (why KNN imputation hurts gradient boosting).
Run:  .venv/bin/python src/test_phase17.py

Covers what the arms and criteria promise before any number is believed: the composed frames fill exactly the cells they
claim to, the KNN copy matches the frozen repair, the guard rule picks point-mass columns by rule, and the criteria
arithmetic gives the pre-registered verdicts on synthetic recoveries.
"""

import numpy as np
import pandas as pd

import phase17_knn as k
from corruption import corrupt_with_id
from phase9_repair_engine import detect_and_repair


def frame(n=400, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "age": rng.integers(18, 80, n),
        "hours": rng.integers(20, 60, n),
        "gain": np.where(rng.random(n) < 0.9, 0, rng.integers(1000, 20000, n)),   # point mass at 0
        "cat": pd.Categorical(rng.choice(["a", "b", "c"], n)),
    })
    y = pd.Series(np.where(X.age + X.gain / 500 + rng.normal(scale=5, size=n) > 60, ">50K", "<=50K"))
    return X, y


def test_guard_rule_selects_point_mass_by_rule_not_by_name():
    X, _ = frame()
    assert k.point_mass_columns(X) == ["gain"]
    X2 = X.rename(columns={"gain": "anything_else"})
    assert k.point_mass_columns(X2) == ["anything_else"], "the rule must not depend on column names"
    assert k.point_mass_columns(X, threshold=0.99) == [], "a stricter threshold selects nothing here"
    X3 = X.copy()
    X3.loc[X3.index[:100], "gain"] = np.nan                         # the share is over OBSERVED cells
    assert k.point_mass_columns(X3) == ["gain"]


def test_knn_values_match_the_frozen_repair_exactly():
    X, y = frame()
    Xd, yd, _ = corrupt_with_id("missing_values", X, y, rate=0.10, seed=3)
    frozen, _, repaired_set, _ = detect_and_repair("missing_values", Xd, yd, seed=3)
    frames, _ = k.build_arms(Xd, k.knn_values(Xd))
    numeric = ["age", "hours", "gain"]
    assert np.abs(frames["knn_all"][numeric].values - frozen[numeric].astype(float).values).max() <= 1e-9
    assert (frames["knn_all"]["cat"].astype(str).values == frozen["cat"].astype(str).values).all()
    assert len(repaired_set) > 0


def test_arms_fill_exactly_the_cells_they_claim():
    X, y = frame()
    Xd, yd, _ = corrupt_with_id("missing_values", X, y, rate=0.15, seed=1)
    imputed = k.knn_values(Xd)
    frames, meta = k.build_arms(Xd, imputed)
    blank = Xd["gain"].isna()
    # dirty keeps its NaN; every KNN-involving arm fills a column only if it is one of its columns
    assert frames["dirty"]["age"].isna().any() and frames["dirty"]["gain"].isna().any()
    assert frames["knn_age"]["age"].isna().sum() == 0 and frames["knn_age"]["gain"].isna().sum() == blank.sum()
    assert frames["knn_all"][["age", "hours", "gain"]].isna().sum().sum() == 0
    # the capital-only analogue here is "gain": filled by KNN there, untouched elsewhere
    assert np.allclose(frames["knn_all"].loc[blank, "gain"], imputed.loc[blank, "gain"])
    # observed cells are never changed by any arm
    for arm, f in frames.items():
        for c in ("age", "hours", "gain"):
            obs = Xd[c].notna()
            assert (f.loc[obs, c].astype(float).values == Xd.loc[obs, c].astype(float).values).all(), (arm, c)
    # the control writes the median explicitly: no NaN left and equal to what the pipeline's imputer would use
    me = frames["median_explicit"]
    assert me[["age", "hours", "gain"]].isna().sum().sum() == 0
    assert (me.loc[blank, "gain"] == Xd["gain"].median()).all()
    assert (me.loc[Xd["age"].isna(), "age"] == Xd["age"].median()).all()
    # categorical cells are mode-filled in every non-dirty arm, left blank in dirty
    assert frames["dirty"]["cat"].isna().any() and not frames["knn_all"]["cat"].isna().any()
    assert (frames["knn_all"].loc[Xd["cat"].isna(), "cat"] == Xd["cat"].mode().iloc[0]).all()


def test_rounded_arm_rounds_only_integer_columns():
    X, y = frame()
    X = X.assign(ratio=np.random.default_rng(1).random(len(X)))                # a non-integer column
    Xd, _, _ = corrupt_with_id("missing_values", X, y, rate=0.2, seed=2)
    imputed = k.knn_values(Xd)
    frames, meta = k.build_arms(Xd, imputed)
    assert "ratio" not in meta["int_cols"] and {"age", "hours", "gain"} <= set(meta["int_cols"])
    blank = Xd["age"].isna()
    assert (frames["knn_rounded"].loc[blank, "age"] % 1 == 0).all()
    assert not np.allclose(frames["knn_all"].loc[blank, "age"] % 1, 0), "KNN values should be fractional here"
    rb = Xd["ratio"].isna()
    assert np.allclose(frames["knn_rounded"].loc[rb, "ratio"], frames["knn_all"].loc[rb, "ratio"])


def test_mechanism_rows_measure_blanked_cells_against_truth():
    X, y = frame()
    Xd, _, _ = corrupt_with_id("missing_values", X, y, rate=0.2, seed=4)
    imputed = k.knn_values(Xd)
    rows = {r["column"]: r for r in k.mechanism_rows(X, Xd, imputed, 0.2, 4)}
    assert rows["gain"]["n_blanked"] == int(Xd["gain"].isna().sum())
    assert 0 <= rows["gain"]["knn_exact_share"] <= 1 and 0 <= rows["gain"]["knn_novel_share"] <= 1
    assert rows["gain"]["median"] == Xd["gain"].median() == 0
    # the median of a 90% point-mass column is the mode, so it is exactly right about as often as the mode is common
    assert abs(rows["gain"]["median_exact_share"] - float((X.loc[Xd["gain"].isna(), "gain"] == 0).mean())) < 1e-12


def rec(**kw):
    """Build a recovery dict {(arm, rate): value}. Keys like 'knn_all@0.1'."""
    return {(key.split("@")[0], float(key.split("@")[1])): v for key, v in kw.items()}


def synthetic(all_=-0.02, no_cap=0.0, cap_only=-0.02, rounded=-0.02, guarded=0.0, gain=-0.015, loss=-0.004):
    d = {}
    for r in (0.1, 0.2):
        d.update({("knn_all", r): all_, ("knn_no_capital", r): no_cap, ("knn_capital_only", r): cap_only,
                  ("knn_rounded", r): rounded, ("knn_guarded", r): guarded})
        for c in k.SINGLE:
            d[(f"knn_{c}", r)] = gain if c == "capital-gain" else (loss if c == "capital-loss" else 0.0)
    return d


def test_k1_pass_partly_and_rejected():
    assert k.criterion_k1(synthetic())["verdict"] == "pass"
    assert k.criterion_k1(synthetic(no_cap=-0.012))["verdict"] == "partly"        # removes only 40%, capital_only keeps 100%
    assert k.criterion_k1(synthetic(no_cap=-0.012, cap_only=-0.005))["verdict"] == "H1 rejected"
    assert k.criterion_k1(synthetic(no_cap=-0.005, cap_only=-0.02))["verdict"] == "pass"   # 75% removed exactly on the line
    assert k.criterion_k1(synthetic(no_cap=-0.0051, cap_only=-0.02))["verdict"] == "partly"  # just under 75% removed
    assert k.criterion_k1(synthetic(all_=0.001))["verdict"] == "not evaluable", "no harm to explain"
    # both rates must hold: fail one rate and the verdict drops
    d = synthetic()
    d[("knn_no_capital", 0.2)] = -0.015
    assert k.criterion_k1(d)["verdict"] == "partly"


def test_k2_requires_capital_gain_largest_at_both_rates():
    assert k.criterion_k2(synthetic())["pass"]
    assert not k.criterion_k2(synthetic(gain=-0.001))["pass"]
    d = synthetic()
    d[("knn_age", 0.2)] = -0.03
    out = k.criterion_k2(d)
    assert not out["pass"] and out["largest"][0.2] == "age" and out["largest"][0.1] == "capital-gain"


def test_k3_k4_k5():
    assert k.criterion_k3(synthetic())["H2_rejected"]
    assert not k.criterion_k3(synthetic(rounded=-0.005))["H2_rejected"]
    deep, shallow = synthetic(all_=-0.02), synthetic(all_=-0.008)
    assert k.criterion_k4(deep, shallow)["verdict"] == "pass"                      # 40% of the deep harm
    assert k.criterion_k4(deep, synthetic(all_=-0.012))["verdict"] == "fail"       # 60%
    assert k.criterion_k4(synthetic(all_=0.0), shallow)["verdict"] == "not evaluable"
    assert k.criterion_k5(synthetic())["verdict"] == "pass"
    assert k.criterion_k5(synthetic(guarded=-0.01))["verdict"] == "fail"           # removes only half


def test_k6_evaluable_pass_fail_and_not_evaluable():
    def bank(all10, g10, all20, g20):
        return {("knn_all", 0.1): all10, ("knn_guarded", 0.1): g10, ("knn_all", 0.2): all20, ("knn_guarded", 0.2): g20}
    floor = 0.01
    assert k.criterion_k6(bank(-0.03, 0.0, -0.04, -0.005), floor)["verdict"] == "pass"
    assert k.criterion_k6(bank(-0.03, -0.015, -0.04, 0.0), floor)["verdict"] == "fail", "50% at 10% fails the pass"
    assert k.criterion_k6(bank(-0.005, -0.005, -0.008, -0.008), floor)["verdict"] == "not evaluable", "harm below the floor"
    # only the evaluable rate counts: 10% is below the floor (ignored even though the guard does nothing there)
    out = k.criterion_k6(bank(-0.005, -0.005, -0.04, 0.0), floor)
    assert out["verdict"] == "pass" and out["evaluable_rates"] == [0.2]
    assert k.criterion_k6(bank(-0.005, -0.005, -0.04, -0.03), floor)["verdict"] == "fail"
    assert k.criterion_k6(bank(0.01, 0.01, 0.02, 0.02), floor)["verdict"] == "not evaluable", "recovery >= 0 is not harm"


def test_f1_and_f2_criteria():
    old = {0.1: -0.0145, 0.2: -0.0195}
    assert k.criterion_f1({0.1: -0.003, 0.2: 0.0}, old)["verdict"] == "pass"
    assert k.criterion_f1({0.1: -0.006, 0.2: 0.0}, old)["verdict"] == "fail", "recovery below -0.005 fails"
    assert k.criterion_f1({0.1: -0.004, 0.2: -0.0098}, old)["verdict"] == "fail", "removes only ~50% at 20%"
    assert k.criterion_f1({0.1: -0.004, 0.2: -0.004}, {0.1: 0.0, 0.2: -0.0195})["verdict"] == "not evaluable"
    assert k.criterion_f2({0.1: -0.012, 0.2: -0.001}, {0.1: -0.001, 0.2: -0.001}, floor=0.0101)["caution"]
    assert not k.criterion_f2({0.1: -0.008, 0.2: 0.0}, {0.1: -0.001, 0.2: -0.001}, floor=0.0101)["caution"]


def test_product_guard_imputes_point_mass_with_median_and_leaves_others_knn():
    import taskclean as tc
    X, y = frame(n=600, seed=5)
    Xd, _, _ = corrupt_with_id("missing_values", X, y, rate=0.15, seed=2)
    num = ["age", "hours", "gain"]
    integer_like = {c: True for c in num}
    frames = tc._propose_missing(Xd, integer_like, Xd.isna(), Xd)
    by_col = {f["column"].iloc[0]: f for f in frames if f["column"].iloc[0] in num}
    gain = by_col["gain"]
    assert Xd["gain"].median() == 0
    assert (gain["proposed_value"].astype(float) == Xd["gain"].median()).all(), "point-mass column takes the median"
    assert "median imputation" in gain["method"].iloc[0] and "covers at least 50%" in gain["method"].iloc[0]
    assert by_col["age"]["method"].iloc[0].startswith("KNN") and by_col["hours"]["method"].iloc[0].startswith("KNN")
    # guard off reproduces the old behaviour: KNN for the point-mass column too (fractional values rounded since integer-like)
    old = {f["column"].iloc[0]: f for f in tc._propose_missing(Xd, integer_like, Xd.isna(), Xd, point_mass_share=1.01)}
    assert old["gain"]["method"].iloc[0].startswith("KNN")
    assert (old["gain"]["proposed_value"].astype(float) != 0).any(), "KNN does write non-zero values into a 90%-zero column"
    # a column that is not dominated by one value is untouched by the guard (same proposals guard on and off)
    assert np.allclose(by_col["age"]["proposed_value"].astype(float), old["age"]["proposed_value"].astype(float))
    # which row_ids are filled does not change
    assert set(gain["row_id"]) == set(old["gain"]["row_id"]) == set(Xd.index[Xd["gain"].isna()])


def test_k0_verdict_requires_every_part():
    ok = {"max_abs_numeric_diff": 0.0, "categorical_equal": True, "guard_cols": k.CAPITAL, "guard_ok": True}
    diffs = {"a": 0.0, "b": 0.0, "c": 0.0}
    assert k0(ok, diffs)["pass"]
    assert not k0({**ok, "max_abs_numeric_diff": 1e-6}, diffs)["pass"]
    assert not k0({**ok, "categorical_equal": False}, diffs)["pass"]
    assert not k0({**ok, "guard_ok": False}, diffs)["pass"]
    assert not k0(ok, {**diffs, "b": 1e-6})["pass"]


def k0(cell, diffs):
    return k.k0_verdict([cell, cell], diffs)


def test_shares_are_defined_against_the_reference_harm():
    d = {("knn_all", 0.1): -0.02, ("x", 0.1): -0.005, ("y", 0.1): 0.005}
    assert abs(k.share_removed(d, "x", 0.1) - 0.75) < 1e-12
    assert abs(k.share_kept(d, "x", 0.1) - 0.25) < 1e-12
    assert k.share_removed(d, "y", 0.1) > 1.0, "an arm that overshoots into positive recovery removes more than all the harm"
    assert k.share_removed({("knn_all", 0.1): 0.0, ("x", 0.1): 0.0}, "x", 0.1) is None


if __name__ == "__main__":
    tests = [v for n, v in sorted(globals().items()) if n.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"\n{len(tests)} tests passed")
