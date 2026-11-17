"""
Invariants for the controlled proof (Phase 20; docs/PHASE20_SPEC.md).
Run:  .venv/bin/python src/test_phase20.py

C5 tuned selection reads only the dirty training rows (no clean data, no test rows); A1 leak-free validation; injection rate and
determinism; the summary logic (verdicts, share recovered, criteria) on planted numbers; the backend endpoint. Every check can fail.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.abspath(ROOT))

import controlled_proof as cp
import model_panel as mp


def small_clean(n=1200):
    return cp.clean_frame().sample(n, random_state=3).reset_index(drop=True)


def test_injection_is_deterministic_and_dirties_only_training_rows():
    clean = small_clean()
    a, b = cp.inject(clean, 42), cp.inject(clean, 42)
    assert a.equals(b), "same seed must give the same dirty frame"
    assert not cp.inject(clean, 43).equals(a), "a different seed must give a different dirty frame"
    assert len(a) > len(clean), "duplicates are injected, so the dirty frame has more rows"
    miss = a.isna().sum().sum() - clean.isna().sum().sum()
    assert miss > 0, "missing values must be injected"
    assert a.duplicated().sum() > clean.duplicated().sum(), "exact duplicates must be injected"
    assert set(a.columns) == set(clean.columns)
    print(f"OK injection: deterministic, {len(a) - len(clean)} rows added, {int(miss)} cells blanked")


def test_tuned_selection_reads_only_the_dirty_training_rows():
    """C5: select_repair gets one frame. Spy on what it audits and scores; the clean data loader must never be called."""
    dirty = cp.inject(small_clean(), 42)
    audited, scored = [], []
    real_audit, real_fit, real_load = cp.audit_dataset, cp._fit_score, cp.load_adult_clean
    cp.audit_dataset = lambda df, *a, **k: (audited.append(len(df)), real_audit(df, *a, **k))[1]
    cp._fit_score = lambda name, X_tr, y_tr, X_te, y_te, *a, **k: (scored.append((len(X_tr), len(X_te))), real_fit(name, X_tr, y_tr, X_te, y_te, *a, **k))[1]
    def boom(*a, **k):
        raise AssertionError("selection must not load the clean data")
    cp.load_adult_clean = boom
    try:
        out = cp.select_repair(dirty, ["logistic_regression"], 42, [], leak_free_val=True)
    finally:
        cp.audit_dataset, cp._fit_score, cp.load_adult_clean = real_audit, real_fit, real_load
    assert audited and max(audited) <= int(len(dirty) * (1 - cp.VAL_FRACTION)) + 2, f"audit saw more than the 75% slice: {audited}"
    assert scored and all(te <= int(len(dirty) * cp.VAL_FRACTION) + 2 for _, te in scored), f"scored on more than the 25% slice: {scored}"
    assert all(tr <= len(dirty) * (1 - cp.VAL_FRACTION) + 2 for tr, _ in scored), "a candidate was trained on validation rows"
    r = out["logistic_regression"]
    assert r["arm"] in r["val"] and set(r["val"]) >= {"dirty", "aggressive"}, r
    print(f"OK C5: selection audited {audited[0]} rows, scored on <= {max(te for _, te in scored)} validation rows, never loaded clean data")


def test_leak_free_validation_has_no_copies_of_training_rows():
    dirty = cp.inject(small_clean(), 42)
    seen = {}
    real_fit = cp._fit_score

    def spy(name, X_tr, y_tr, X_te, y_te, *a, **k):
        seen.setdefault("train", X_tr)
        seen.setdefault("val", X_te)
        return real_fit(name, X_tr, y_tr, X_te, y_te, *a, **k)
    cp._fit_score = spy
    try:
        cp.select_repair(dirty, ["logistic_regression"], 42, [], leak_free_val=True)
        leak_free_n = len(seen["val"])
        seen.clear()
        cp.select_repair(dirty, ["logistic_regression"], 42, [], leak_free_val=False)
        raw_n = len(seen["val"])
    finally:
        cp._fit_score = real_fit
    assert leak_free_n < raw_n, f"leak-free validation should drop copied rows (kept {leak_free_n} of {raw_n}); the dirty frame has duplicates"
    print(f"OK A1: validation slice {raw_n} rows -> {leak_free_n} after removing copies of training rows")


def planted(models=("a", "b", "c", "d"), clean=0.80, dirty=0.70, tuned_gain=0.03, tuned_models=("a", "b")):
    """Rows for summarise(): 3 seeds with tiny jitter so the noise floor is the 0.003 minimum."""
    rows, nulls, choices = [], [], []
    for seed in (1, 2, 3):
        j = 0.0005 * seed
        for m in models:
            for arm in cp.ARM_ORDER:
                f1 = {"clean": clean, "dirty": dirty}.get(arm, dirty)
                same = arm in ("dirty", "default") or (arm in ("tuned", "tuned_v0") and m not in tuned_models)
                if arm in ("tuned", "tuned_v0") and m in tuned_models:
                    f1, same = dirty + tuned_gain, False
                if arm == "aggressive":
                    f1, same = dirty - 0.02, False
                rows.append({"seed": seed, "model": m, "arm": arm, "same_as_dirty": same, "f1": f1 + j, "auc": 0.9, "acc": 0.8})
            nulls.append({"seed": seed, "model": m, "d_f1": 0.0005})
            choices.append({"seed": seed, "model": m, "variant": "tuned", "chosen": "outliers", "best_candidate": "outliers", "best_val_gain": 0.01})
            choices.append({"seed": seed, "model": m, "variant": "tuned_v0", "chosen": "dirty", "best_candidate": "outliers", "best_val_gain": 0.0})
    return pd.DataFrame(rows), pd.DataFrame(nulls), pd.DataFrame(choices)


def test_summary_logic_on_planted_numbers():
    models = ["a", "b", "c", "d"]
    for m in models:
        mp.MODEL_LABELS.setdefault(m, m)
    R, N, C = planted()
    out = cp.summarise(R, N, C, [1, 2, 3], models, set(), 1.0)
    cell = lambda m, a: next(c for c in out["results"] if c["model"] == m and c["arm"] == a)
    assert cell("a", "tuned")["verdict"] == "improved" and abs(cell("a", "tuned")["delta_f1"] - 0.03) < 1e-9
    assert abs(cell("a", "tuned")["share_recovered"] - 0.3) < 1e-6, "0.03 of a 0.10 damage is 30%"
    assert cell("c", "tuned")["verdict"] == "no change", "tuned equal to dirty must read 'no change', not a gain"
    assert cell("a", "aggressive")["verdict"] == "worse", "a harmful arm must be labelled worse"
    assert cell("a", "default")["verdict"] == "no change" and cell("a", "default")["share_recovered"] is None
    assert out["criteria"]["C0_evaluable"]["pass"] and out["criteria"]["C1_complete"]["pass"]
    assert out["criteria"]["C2_tuned_positive"]["pass"], "2 improved and none worse must pass"
    assert out["criteria"]["C3_default_not_worse"]["pass"]
    assert len(out["results"]) == 4 * len(cp.ARM_ORDER), "every arm for every model must be reported (C1)"
    # negative control: only one model improves -> C2 must fail; a worse model -> C2 must fail
    R1, N1, C1 = planted(tuned_models=("a",))
    assert not cp.summarise(R1, N1, C1, [1, 2, 3], models, set(), 1.0)["criteria"]["C2_tuned_positive"]["pass"]
    R2, N2, C2 = planted()
    mask = (R2.model == "c") & (R2.arm == "tuned")
    R2.loc[mask, "f1"] = 0.70 - 0.03 + R2.loc[mask, "seed"] * 0.0005
    R2.loc[mask, "same_as_dirty"] = False
    s2 = cp.summarise(R2, N2, C2, [1, 2, 3], models, set(), 1.0)
    assert next(c for c in s2["results"] if c["model"] == "c" and c["arm"] == "tuned")["verdict"] == "worse"
    assert not s2["criteria"]["C2_tuned_positive"]["pass"], "one worse model must fail C2"
    # not evaluable: when dirty is as good as clean, C0 fails and no share is computed
    R3, N3, C3 = planted(dirty=0.80, tuned_gain=0.0)
    s3 = cp.summarise(R3, N3, C3, [1, 2, 3], models, set(), 1.0)
    assert not s3["criteria"]["C0_evaluable"]["pass"], "no damage means the run is not evaluable"
    print("OK summary: verdicts, share of damage recovered, C0-C3 pass and fail on planted data")


def test_backend_serves_the_saved_proof():
    import backend.main as bm
    saved = os.path.join(ROOT, "results", "phase20_proof.json")
    bm.PROOF["live"] = None
    out = bm.read_proof()
    if os.path.exists(saved):
        want = json.load(open(saved))
        assert out["status"] == "ok" and out["source"].startswith("saved") and out["results"] == want["results"]
    else:
        assert out["status"] == "none"
    bm.PROOF["live"] = {"status": "running", "progress": 0.4, "message": "x"}
    assert bm.read_proof()["status"] == "running"
    bm.PROOF["live"] = None
    print("OK backend: /api/proof serves the saved registered run, or a live job's progress")


if __name__ == "__main__":
    tests = [v for n, v in sorted(globals().items()) if n.startswith("test_") and callable(v)]
    for t in tests:
        t()
    print(f"\nALL PHASE 20 CONTROLLED-PROOF INVARIANTS HOLD ({len(tests)} tests)")
