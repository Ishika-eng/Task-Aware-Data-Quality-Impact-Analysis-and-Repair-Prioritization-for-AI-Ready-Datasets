"""
Invariants for the Phase 16 harness (gradient boosting as a second model family).
Run:  .venv/bin/python src/test_phase16.py

Covers what the harness promises before any gradient-boosting number is believed: the selection rule, the stage gate,
that the model really is injected, that the booster takes the project's sparse preprocessing, that it is deterministic,
and that the Random Forest run through the new loop reproduces the frozen Phase 7 number exactly.
"""

import os

import numpy as np
import pandas as pd

import phase16_gb as g
from baseline import evaluate


def frame(n=600, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n),
                      "c": rng.choice(["x", "y", "z"], n), "d": rng.choice(["p", "q"], n)})
    y = pd.Series(np.where(X.a + (X.c == "x") + rng.normal(scale=0.5, size=n) > 0.5, ">50K", "<=50K"))
    return X, y


def test_select_config_rule():
    c = [{"max_iter": 200, "learning_rate": 0.1, "val_posf1": 0.70},
         {"max_iter": 100, "learning_rate": 0.1, "val_posf1": 0.70},
         {"max_iter": 100, "learning_rate": 0.05, "val_posf1": 0.70},
         {"max_iter": 400, "learning_rate": 0.05, "val_posf1": 0.69}]
    s = g.select_config(c)
    assert (s["max_iter"], s["learning_rate"]) == (100, 0.05), s            # tie: smaller max_iter, then smaller lr
    c.append({"max_iter": 400, "learning_rate": 0.1, "val_posf1": 0.71})
    assert g.select_config(c)["max_iter"] == 400                            # a strictly better score always wins
    assert g.select_config([{"max_iter": 100, "learning_rate": 0.1, "val_posf1": 0.5}])["val_posf1"] == 0.5


def test_stage_gate_refuses_without_config():
    old = g.CONFIG_PATH
    g.CONFIG_PATH = os.path.join(g.RESULTS_DIR, "phase16_gb_config_DOES_NOT_EXIST.json")
    try:
        for stage in range(2, 8):
            try:
                g.main(["--stage", str(stage)])
            except SystemExit as e:
                assert "phase16_gb_config.json is missing" in str(e), e
            else:
                raise AssertionError(f"stage {stage} ran without a recorded configuration")
        # and stages 0 and 1 are not gated
        assert g.CONFIG_PATH != old
    finally:
        g.CONFIG_PATH = old


def test_gb_pipeline_takes_sparse_preprocessing_and_overrides():
    X, y = frame()
    pipe = g.build_gb_pipeline(X, 0, max_iter=37, learning_rate=0.05).fit(X, y)
    clf = pipe.named_steps["clf"]
    assert clf.max_iter == 37 and clf.learning_rate == 0.05
    assert clf.early_stopping is False and clf.min_samples_leaf == 20 and clf.max_leaf_nodes == 31
    assert clf.l2_regularization == 0.0
    proba = pipe.predict_proba(X)
    assert proba.shape == (len(X), 2) and np.allclose(proba.sum(axis=1), 1)
    assert (pipe.predict(X) == y).mean() > 0.8                              # it actually learns the planted signal
    # the densifying step changes no value
    from scipy import sparse
    s = sparse.csr_matrix(np.array([[0.0, 1.5], [2.0, 0.0]]))
    assert np.array_equal(g._dense(s), s.toarray()) and np.array_equal(g._dense(s.toarray()), s.toarray())


def test_gb_is_deterministic_in_random_state():
    X, y = frame()
    a = g.build_gb_pipeline(X, 1, max_iter=50).fit(X, y)
    b = g.build_gb_pipeline(X, 999, max_iter=50).fit(X, y)
    assert np.array_equal(a.predict_proba(X), b.predict_proba(X))


def test_scores_match_frozen_evaluate():
    X, y = frame()
    pipe = g.build_gb_pipeline(X, 0, max_iter=40).fit(X, y)
    s, e = g.scores(pipe, X, y), evaluate(pipe, X, y)
    assert abs(s["posf1"] - e["f1"]) < 1e-15 and abs(s["auc"] - e["roc_auc"]) < 1e-15
    assert 0 < s["macrof1"] <= 1


def test_the_injected_model_is_the_one_used():
    X, y = frame(1000)
    X_tr, X_te, y_tr, y_te = X.iloc[:700], X.iloc[700:], y.iloc[:700], y.iloc[700:]
    used = []

    def spy(Xa, seed):
        used.append((len(Xa), seed))
        return g.build_gb_pipeline(Xa, seed, max_iter=20)

    row = g.damage_cell(spy, X_tr, y_tr, X_te, y_te, "label_errors", 0.2, 7)
    assert used and used[0][1] == 7 and used[0][0] == 700, used
    assert row["dirty_f1"] == row["dirty_posf1"] and 0 <= row["dirty_f1"] <= 1
    # more corruption, same everything else: the corrupted training labels differ from the clean ones
    from corruption import CORRUPTORS
    _, y_dirty, log = CORRUPTORS["label_errors"](X_tr, y_tr, rate=0.2, seed=7)
    assert (y_dirty != y_tr.loc[y_dirty.index]).sum() == len(log) > 0


def test_repair_cell_structure_and_arithmetic():
    X, y = frame(1000)
    X_tr, X_te, y_tr, y_te = X.iloc[:700], X.iloc[700:], y.iloc[:700], y.iloc[700:]
    factory = g.gb_factory({"max_iter": 20, "learning_rate": 0.1})
    clean = g.scores(factory(X_tr, 3).fit(X_tr, y_tr), X_te, y_te)
    bundle = g.make_repair_bundle("duplicates", X_tr, y_tr, 0.1, 3)
    # the repaired frame has the injected duplicates removed again, so it is the clean frame
    assert len(bundle["repaired"][0]) == len(X_tr) and len(bundle["dirty"][0]) > len(X_tr)
    assert bundle["correctness"]["precision"] > 0.9 and bundle["correctness"]["recall"] == 1.0
    row = g.repair_cell(factory, bundle, X_te, y_te, clean)
    for m in ("posf1", "macrof1", "auc"):
        assert abs(row[f"damage_{m}"] - (clean[m] - row[f"dirty_{m}"])) < 1e-15
        assert abs(row[f"recovery_{m}"] - (row[f"repaired_{m}"] - row[f"dirty_{m}"])) < 1e-15
    # model-independent part: building the bundle twice for the same cell gives the same frames
    b2 = g.make_repair_bundle("duplicates", X_tr, y_tr, 0.1, 3)
    assert b2["repaired"][0].equals(bundle["repaired"][0]) and b2["dirty"][1].equals(bundle["dirty"][1])


def test_rf_through_the_harness_reproduces_the_frozen_phase7_number():
    X_tr, _, X_te, y_tr, _, y_te = g.load_split()
    frozen = pd.read_csv(os.path.join(g.RESULTS_DIR, "phase7_impact_raw_full.csv"))
    ref = float(frozen[(frozen.error_type == "duplicates") & (frozen.rate == 0.05) & (frozen.seed == 42)].dirty_f1.iloc[0])
    got = g.damage_cell(g.rf_factory, X_tr, y_tr, X_te, y_te, "duplicates", 0.05, 42)["dirty_f1"]
    assert abs(ref - got) <= g.G0_TOLERANCE, (ref, got)


def test_harness_does_not_write_to_frozen_logs():
    src = open(os.path.join(os.path.dirname(__file__), "phase16_gb.py")).read()
    assert "log_training" not in src and "training_log" not in src.replace("results/training_log.csv", "")


def test_spec_constants_are_as_preregistered():
    assert g.GB_FIXED == {"early_stopping": False, "min_samples_leaf": 20, "l2_regularization": 0.0, "max_leaf_nodes": 31}
    assert g.GRID_MAX_ITER == [100, 200, 400] and g.GRID_LEARNING_RATE == [0.05, 0.1]
    assert g.CAPACITY_ARMS["shallow"] == {"max_leaf_nodes": 8, "max_iter": 100, "learning_rate": 0.1}
    assert g.CAPACITY_ARMS["deep"] == {"max_leaf_nodes": 63, "max_iter": 400, "learning_rate": 0.1}
    assert g.SEEDS == [42, 43, 44, 45, 46] and g.G0_TOLERANCE == 1e-9 and g.G1_MARGIN == 0.01


def cells(label=(0.01, 0.025, 0.065), other=0.004):
    """A synthetic {(issue, rate): damage}: label errors are the biggest at every rate."""
    d = {}
    for et in g.ERROR_TYPES:
        for i, r in enumerate(g.RATES):
            d[(et, r)] = label[i] if et == "label_errors" else other * (i + 1)
    return d


def test_g2_ordering_criterion():
    rf = cells()
    ok = g.criterion_g2(cells(label=(0.012, 0.03, 0.07)), rf)
    assert ok["pass"] and ok["spearman"] >= 0.8 and ok["top_at_10pct"] == ok["top_at_20pct"] == "label_errors", ok
    # label errors not the largest at 20% -> fails even though the overall correlation stays high
    gb = cells(); gb[("duplicates", 0.20)] = 0.09
    bad = g.criterion_g2(gb, rf)
    assert bad["top_at_20pct"] == "duplicates" and not bad["pass"], bad
    # reversed ordering -> negative correlation -> fails
    rev = {k: 1 / (v + 1e-3) for k, v in rf.items()}
    assert not g.criterion_g2(rev, rf)["pass"] and g.criterion_g2(rev, rf)["spearman"] < 0
    # constant damage has no defined correlation -> not a pass
    assert not g.criterion_g2({k: 0.01 for k in rf}, rf)["pass"]


def test_g3_magnitude_criterion_uses_informative_cells_only():
    rf = cells()                                                       # informative: RF damage >= 0.005
    inf = sorted(k for k, v in rf.items() if v >= g.INFORMATIVE_RF_DAMAGE)
    assert ("label_errors", 0.05) in inf and ("duplicates", 0.05) not in inf    # 0.004 is below the cut
    same = g.criterion_g3({k: v * 1.2 for k, v in rf.items()}, rf)
    assert same["pass"] and abs(same["median_ratio"] - 1.2) < 1e-12 and same["n_informative"] == len(inf)
    assert not g.criterion_g3({k: v * 2.5 for k, v in rf.items()}, rf)["pass"]
    assert not g.criterion_g3({k: v * 0.3 for k, v in rf.items()}, rf)["pass"]
    # a huge ratio on a NON-informative cell cannot move the median
    gb = {k: v * 1.0 for k, v in rf.items()}; gb[("duplicates", 0.05)] = 100.0
    assert g.criterion_g3(gb, rf)["pass"]
    assert not g.criterion_g3({}, {})["pass"]                          # nothing informative -> not a pass


def test_g8_capacity_criterion():
    arms = {"shallow": cells(), "selected": cells(), "deep": cells()}
    assert g.criterion_g8(arms)["pass"] and len(g.criterion_g8(arms)["cases"]) == 6
    arms["deep"] = cells(label=(0.01, 0.025, 0.003))                  # label errors collapse at 20% in the deep arm
    out = g.criterion_g8(arms)
    assert not out["pass"] and out["cases"]["deep@20%"] != "label_errors"
    assert not g.criterion_g8({"shallow": cells(), "selected": cells()})["pass"]    # needs all three arms


def test_rms_and_summary_arithmetic():
    assert abs(g.rms([3, 4]) - np.sqrt(12.5)) < 1e-12 and g.rms([0, 0]) == 0
    raw = pd.DataFrame({"arm": "selected", "error_type": ["a"] * 4 + ["b"] * 4, "rate": 0.1,
                        "seed": [1, 2, 3, 4] * 2, "dirty_posf1": 0.6, "damage_posf1": [0.0, 0.01, 0.02, 0.03] + [0.1] * 4,
                        "damage_macrof1": 0.0, "damage_auc": 0.0})
    rf = pd.DataFrame({"error_type": ["a", "b"], "rate": 0.1, "mean_damage": [0.004, 0.05], "std_damage": 0.001})
    s = g.summarise_damage(raw.drop(columns="arm"), rf).set_index("error_type")
    assert abs(s.loc["a", "mean_damage_posf1"] - 0.015) < 1e-12 and s.loc["a", "n_seeds"] == 4
    assert abs(s.loc["a", "std_damage_posf1"] - raw.damage_posf1[:4].std()) < 1e-12
    assert not s.loc["a", "informative"] and np.isnan(s.loc["a", "gb_over_rf"])           # 0.004 < 0.005
    assert s.loc["b", "informative"] and abs(s.loc["b", "gb_over_rf"] - 2.0) < 1e-12


def test_run_damage_shares_corruption_and_reports_arithmetic():
    X, y = frame(900)
    X_tr, X_te, y_tr, y_te = X.iloc[:600], X.iloc[600:], y.iloc[:600], y.iloc[600:]
    arms = {"A": g.gb_factory({"max_iter": 15, "learning_rate": 0.1}), "B": g.gb_factory({"max_iter": 40, "learning_rate": 0.1})}
    df = g.run_damage(arms, X_tr, y_tr, X_te, y_te, ["label_errors", "duplicates"], [0.1, 0.2], [1, 2], log=lambda *_: None)
    assert len(df) == 2 * 2 * 2 * 2 and set(df.arm) == {"A", "B"}
    for m in g.METRICS:
        assert np.allclose(df[f"damage_{m}"], df[f"baseline_{m}"] - df[f"dirty_{m}"])
    # GB has no seed noise: an arm's clean baseline is the same number for every seed
    assert df.groupby("arm").baseline_posf1.nunique().max() == 1
    # different capacity arms are different models, so the damage columns are not simply copied
    assert not np.allclose(df[df.arm == "A"].dirty_posf1.values, df[df.arm == "B"].dirty_posf1.values)


def test_null_experiment_structure():
    X, y = frame(900)
    X_tr, X_te, y_tr, y_te = X.iloc[:600], X.iloc[600:], y.iloc[:600], y.iloc[600:]
    out = g.null_experiment(g.gb_factory({"max_iter": 20, "learning_rate": 0.1}), X_tr, y_tr, X_te, y_te, [1, 2, 3])
    assert out["n_draws"] == 3 * 4 and out["drop_fraction"] == 0.1 and out["noise_floor_min"] == 0.003
    for m in g.METRICS:
        v = out[m]
        assert len(v["moves"]) == 12 and np.isfinite(v["null_noise"]) and v["null_noise"] > 0     # dropping 10% moves F1
        assert abs(v["null_noise"] - g.rms(v["moves"])) < 1e-15
        assert v["noise_floor"] == max(0.003, v["null_noise"])
    # same data, same seeds -> same draws (one stream seeded SEED0)
    again = g.null_experiment(g.gb_factory({"max_iter": 20, "learning_rate": 0.1}), X_tr, y_tr, X_te, y_te, [1, 2, 3])
    assert again["posf1"]["moves"] == out["posf1"]["moves"]


def test_stage3_needs_stage2_output():
    old = g.PATHS["damage_raw.csv"]
    g.PATHS["damage_raw.csv"] = os.path.join(g.RESULTS_DIR, "phase16_gb_NOPE.csv")
    try:
        try:
            g.stage3()
        except SystemExit as e:
            assert "run --stage 2 first" in str(e)
        else:
            raise AssertionError("stage 3 ran without stage 2's output")
    finally:
        g.PATHS["damage_raw.csv"] = old


def _ev(rows):
    """rows: (issue, rate, rf_status, gb_status, risk, precision)"""
    return pd.DataFrame([{"error_type": a, "rate": b, "rf_status": c, "repairability_status": d, "risk": e,
                          "repair_precision": f, "material_risk": e} for a, b, c, d, e, f in rows]).assign(
        gb_safe_auto=lambda d: [g.is_safe_auto(r.risk, r.repair_precision, r.repairability_status) for r in d.itertuples()])


def _fifteen(rf, gb):
    ets = g.ERROR_TYPES
    return _ev([(et, r, rf[i * 3 + j], gb[i * 3 + j], 0.0, 1.0) for i, et in enumerate(ets) for j, r in enumerate(g.RATES)])


def test_stricter_status_and_gate():
    assert g.stricter_status("positive", "positive") == "positive"
    assert g.stricter_status("positive", "weak") == "weak" and g.stricter_status("weak", "positive") == "weak"
    assert g.stricter_status("positive", "negative") == "negative" and g.stricter_status("negative", "weak") == "negative"
    assert g.is_safe_auto(0.0, 0.95, "weak") and g.is_safe_auto(1e-9, 1.0, "positive")
    assert not g.is_safe_auto(0.0, 0.9499, "positive") and not g.is_safe_auto(2e-9, 1.0, "positive")
    assert not g.is_safe_auto(0.0, 1.0, "negative")


def test_evidence_table_reproduces_frozen_phase10_statuses():
    """The status / risk code path, fed the frozen Phase 9 per-seed RF rows and the 0.003 floor, must give the frozen
    Phase 10 evidence table. This validates the arithmetic that Stage 4 applies to the GB rows."""
    raw = pd.read_csv(os.path.join(g.RESULTS_DIR, "phase9_repair_raw_full.csv"))
    raw = raw.rename(columns={"damage": "damage_posf1", "recovery": "recovery_posf1", "dirty_f1": "dirty_posf1",
                              "oracle_f1": "oracle_posf1"})
    raw["recovery_macrof1"] = raw["recovery_auc"] = 0.0
    frozen = pd.read_csv(os.path.join(g.RESULTS_DIR, "phase10_evidence_table.csv"))
    ev = g.evidence_from_raw(raw, 0.003, frozen)
    assert len(ev) == 15
    m = ev.merge(frozen, on=["error_type", "rate"], suffixes=("", "_frozen"))
    assert (m.repairability_status == m.repairability_status_frozen).all()
    assert (m.rf_status == m.repairability_status_frozen).all() and m.status_agrees.all()
    for c in ("mean_recovery", "risk", "p_harm", "harm_magnitude", "repair_precision", "sign_agreement"):
        assert np.allclose(m[c], m[c + "_frozen"], atol=1e-12), c
    assert (ev.effective_status == ev.repairability_status).all() and not ev.gb_only_harm.any()


def test_g4_agreement_kappa_and_chance_baseline():
    rf = ["weak"] * 8 + ["negative"] * 4 + ["positive"] * 3
    r = g.criterion_g4(_fifteen(rf, rf))
    assert r["agree"] == 15 and r["pass"] and r["chance_baseline_all_weak"] == 8 and abs(r["cohens_kappa"] - 1.0) < 1e-12
    gb = list(rf); gb[0], gb[1], gb[2], gb[8] = "negative", "positive", "negative", "positive"      # 4 differences -> 11
    r = g.criterion_g4(_fifteen(rf, gb))
    assert r["agree"] == 11 and r["pass"] and len(r["disagreements"]) == 4
    gb[3] = "positive"                                                                                # 12 -> 10 agree
    r = g.criterion_g4(_fifteen(rf, gb))
    assert r["agree"] == 10 and not r["pass"]


def test_g5_duplicates_gate_per_rate():
    base = [("duplicates", r, "positive", "weak", 0.0, 0.99) for r in g.RATES]
    assert g.criterion_g5(_ev(base))["pass"]
    assert not g.criterion_g5(_ev(base[:2] + [("duplicates", 0.2, "positive", "weak", 0.0005, 0.99)]))["pass"]   # risk > 0
    assert not g.criterion_g5(_ev(base[:2] + [("duplicates", 0.2, "positive", "negative", 0.0, 0.99)]))["pass"]
    assert not g.criterion_g5(_ev(base[:2] + [("duplicates", 0.2, "positive", "weak", 0.0, 0.94)]))["pass"]      # precision
    assert not g.criterion_g5(_ev(base[:2]))["pass"]                                                              # missing rate


def test_g6_gb_only_harm_rules():
    rf = ["weak"] * 15
    assert g.criterion_g6(_fifteen(rf, rf))["pass"]
    gb = list(rf); gb[4] = "negative"                                  # one non-duplicate cell: allowed, but listed
    r = g.criterion_g6(_fifteen(rf, gb))
    assert r["pass"] and r["n_gb_only_harm"] == 1 and len(r["cells"]) == 1
    gb[7] = "negative"                                                 # two cells: fail
    assert not g.criterion_g6(_fifteen(rf, gb))["pass"]
    gb = list(rf); gb[6] = "negative"                                  # a single duplicates cell: fail
    r = g.criterion_g6(_fifteen(rf, gb))
    assert not r["pass"] and r["duplicates_harmed"] == 1
    gb = ["negative"] * 15; assert g.criterion_g6(_fifteen(["negative"] * 15, gb))["n_gb_only_harm"] == 0   # RF already negative


def test_stage4_needs_stage2_noise_floor():
    old = g.PATHS["noise.json"]
    g.PATHS["noise.json"] = os.path.join(g.RESULTS_DIR, "phase16_gb_NOPE.json")
    try:
        try:
            g.stage4()
        except SystemExit as e:
            assert "run --stage 2 first" in str(e)
        else:
            raise AssertionError("stage 4 ran without the noise floor")
    finally:
        g.PATHS["noise.json"] = old


def _ranking_evidence(order):
    """Synthetic 10% evidence whose headline order is `order` (impact falls, risk 0, recovery falls along the list)."""
    n = len(order)
    return pd.DataFrame([{"error_type": et, "rate": 0.10, "mean_damage": (n - i) * 0.01, "risk": 0.0,
                          "mean_recovery": (n - i) * 0.001, "repairability_status": "weak"} for i, et in enumerate(order)])


def test_priority_machinery_reproduces_frozen_phase10():
    rf = pd.read_csv(os.path.join(g.RESULTS_DIR, "phase10_evidence_table.csv"))
    frozen = pd.read_csv(os.path.join(g.RESULTS_DIR, "phase10_priority_headline.csv"))
    assert g.headline_ranking(rf) == frozen.error_type.tolist()
    assert len(g.weight_combos()) == 15
    assert g.label_first_feature_last_share(g.combo_rankings(rf)) == 1.0       # spec: RF reference is 1.0


def test_g7_tau_and_share():
    rf_order = ["label_errors", "duplicates", "outliers", "missing_values", "feature_corruption"]
    rf = _ranking_evidence(rf_order)
    same = g.criterion_g7(rf, rf)
    assert abs(same["kendall_tau"] - 1.0) < 1e-12 and same["gb_ranking"] == rf_order and same["n_combos"] == 15
    # feature corruption no longer last: tau is high but the joint label-first/feature-last share is 0 -> fail
    gb = _ranking_evidence(["label_errors", "duplicates", "outliers", "feature_corruption", "missing_values"])
    r = g.criterion_g7(rf, gb)
    assert r["kendall_tau"] > 0.6 and r["gb_share_label_first_feature_last"] == 0.0 and not r["pass"]
    # reversed ranking: tau -1, fail
    rev = _ranking_evidence(rf_order[::-1])
    r = g.criterion_g7(rf, rev)
    assert abs(r["kendall_tau"] + 1.0) < 1e-12 and not r["pass"]


def test_stage5_needs_stage4_output():
    old = g.PATHS["evidence_table.csv"]
    g.PATHS["evidence_table.csv"] = os.path.join(g.RESULTS_DIR, "phase16_gb_NOPE.csv")
    try:
        try:
            g.stage5()
        except SystemExit as e:
            assert "run --stage 4 first" in str(e)
        else:
            raise AssertionError("stage 5 ran without stage 4's output")
    finally:
        g.PATHS["evidence_table.csv"] = old


def _bank_ev(statuses, damages, dup_material=0.0, dup_prec=1.0):
    rows = []
    for i, et in enumerate(g.ERROR_TYPES):
        for j, r in enumerate(g.RATES):
            k = i * 3 + j
            dup = et == "duplicates"
            rows.append({"error_type": et, "rate": r, "repairability_status": statuses[k], "mean_damage": damages[k],
                         "material_risk": dup_material if dup else 0.0, "risk": dup_material if dup else 0.0,
                         "repair_precision": dup_prec if dup else 0.5})
    return pd.DataFrame(rows)


def _safe(ev):
    return pd.Series(((ev.material_risk <= 1e-9) & (ev.repair_precision >= 0.95) & (ev.repairability_status != "negative")).values,
                     index=pd.MultiIndex.from_frame(ev[["error_type", "rate"]]))


def test_g9_three_parts_each_able_to_fail():
    st = ["weak"] * 15
    dm = [float(v) for v in np.linspace(0.001, 0.05, 15)]
    rf = _bank_ev(st, dm)
    ok = g.criterion_g9(_bank_ev(st, dm), rf, _safe(_bank_ev(st, dm)))
    assert ok["pass"] and ok["spearman"] > 0.99 and ok["b_agree"] == 15 and ok["c_pass"]
    flip = _bank_ev(st, dm[::-1])                                            # ordering reversed: (a) fails
    r = g.criterion_g9(flip, rf, _safe(flip))
    assert not r["a_spearman_pass"] and not r["pass"]
    stx = list(st)
    for i in (0, 1, 2, 3, 4):
        stx[i] = "negative"                                                  # 5 disagreements -> 10 agree: (b) fails
    bad = _bank_ev(stx, dm)
    r = g.criterion_g9(bad, rf, _safe(bad))
    assert not r["b_pass"] and r["b_agree"] == 10 and not r["pass"]
    risky = _bank_ev(st, dm, dup_material=0.002)                              # duplicates carry material risk: (c) fails
    r = g.criterion_g9(risky, rf, _safe(risky))
    assert not r["c_pass"] and not r["pass"]
    lowp = _bank_ev(st, dm, dup_prec=0.9)
    assert not g.criterion_g9(lowp, rf, _safe(lowp))["c_pass"]


def test_scores_accept_a_positive_label():
    X, y = frame()
    y = y.map({">50K": "yes", "<=50K": "no"})
    pipe = g.build_gb_pipeline(X, 0, max_iter=20).fit(X, y)
    s = g.scores(pipe, X, y, pos_label="yes")
    assert 0 < s["posf1"] <= 1 and 0.5 < s["auc"] <= 1


def test_stage6_refuses_without_config():
    old = g.CONFIG_PATH
    g.CONFIG_PATH = os.path.join(g.RESULTS_DIR, "phase16_gb_config_DOES_NOT_EXIST.json")
    try:
        try:
            g.stage6()
        except SystemExit as e:
            assert "phase16_gb_config.json is missing" in str(e)
        else:
            raise AssertionError("stage 6 ran without a recorded configuration")
    finally:
        g.CONFIG_PATH = old


def test_inconsistency_summary_arithmetic_and_status():
    runs = pd.DataFrame([{"rate": 0.1, "seed": s, "damage_posf1": 0.01, "recovery_posf1": r, "recovery_auc": 0.001,
                          "recovery_macrof1": 0.0, "repair_precision": 1.0, "repair_recall": 0.6}
                         for s, r in zip(range(5), [-0.004, -0.005, -0.006, -0.004, -0.002])])
    out = g.inconsistency_summary(runs, 0.003).iloc[0]
    assert abs(out.mean_recovery_posf1 + 0.0042) < 1e-12 and out.sign_agreement == 1.0
    assert out.repairability_status == "negative" and out.n_neg_seeds == 5 and abs(out.risk - 0.0042) < 1e-12
    runs["recovery_posf1"] = [0.004, 0.005, 0.006, 0.004, 0.002]
    assert g.inconsistency_summary(runs, 0.003).iloc[0].repairability_status == "positive"
    runs["recovery_posf1"] = [0.004, -0.005, 0.006, -0.004, 0.002]
    assert g.inconsistency_summary(runs, 0.003).iloc[0].repairability_status == "weak"
    frozen = pd.read_csv(os.path.join(g.RESULTS_DIR, "phase14_evidence_table.csv"))
    merged = g.inconsistency_summary(runs.assign(rate=0.05), 0.003, frozen)
    assert merged.rf_status.iloc[0] == "negative" and "status_agrees" in merged


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"\n{len(tests)} tests passed")
