"""
Invariants for Phase 25, numeric placeholder codes read as missing (docs/PHASE25_SPEC.md).
Run:  .venv/bin/python src/test_phase25.py        Every check can fail.

S2 planted truth (recall and precision 1.00 at 1/5/20%, three seeds, numeric and text) and the counter-cases that must NOT fire,
S3 policy (evidence-based file never fills a code cell, the aggressive file and a human override do), S4 regression target,
the Baseline snapshot, the backend summary, and S0 (a file without codes is untouched).
S0 across the real files and S1/S4/S6 on the DS-jobs file are measured by src/validate_sentinels.py, not here.
"""
import os
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(ROOT))

from health_snapshot import build_snapshot
from taskclean import _apply, _prepare, audit_dataset


def frame(n=600, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "age": rng.integers(18, 80, n).astype(float),            # non-negative numeric: the column that gets codes
        "income": rng.gamma(3.0, 2000.0, n),                      # non-negative numeric, left clean
        "temp": rng.normal(5, 10, n).round(1),                    # genuine negatives (a -1 here is a real value)
        "city": rng.choice(["Pune", "Delhi", "Goa", "Agra"], n),
        "y": rng.choice(["p", "q"], n)})


def codes_of(df, target="y", task="classification"):
    X, y, *_ = _prepare(df, target, task)
    return X, y, X.attrs["codes"], X.attrs["code_majority"]


def test_planted_numeric_codes_are_found_exactly():
    for code in (-1, -999):
        for rate in (0.01, 0.05, 0.20):
            for seed in (0, 1, 2):
                df = frame(seed=seed)
                rng = np.random.default_rng(100 + seed)
                rows = np.sort(rng.choice(len(df), int(rate * len(df)), replace=False))
                df.loc[rows, "age"] = float(code)
                X, _, codes, major = codes_of(df)
                assert set(codes) == {"age"}, (code, rate, seed, sorted(codes))        # precision: nothing else in any column
                assert codes["age"]["rows"] == rows.tolist(), (code, rate, seed)        # recall 1.00 and no extra cell
                assert X["age"].isna().sum() == len(rows) and not major
                assert df["age"].iloc[rows[0]] == code, "the uploaded frame is never modified by the analysis copy"
    print("OK S2 numeric: planted -1 and -999 at 1/5/20%, 3 seeds: recall 1.00, precision 1.00")


def test_counter_cases_must_not_fire():
    n = 600
    # genuine negatives that contain -1
    df = frame(); df.loc[[3, 9, 27, 41, 52], "temp"] = -1.0
    assert "temp" not in codes_of(df)[2], "a column with genuine negatives is never a code column"
    # only two cells equal -1
    df = frame(); df.loc[[3, 9], "age"] = -1.0
    assert not codes_of(df)[2], "fewer than 3 occurrences is not a code"
    # a column that is 60% -1: a category, reported but not flagged
    df = frame(); df.loc[np.random.default_rng(0).choice(n, 360, replace=False), "age"] = -1.0
    X, _, codes, major = codes_of(df)
    assert not codes and "age" in major and abs(major["age"]["share"] - 0.6) < 1e-9, (codes, major)
    assert X["age"].isna().sum() == 0, "a majority code is left as it is"
    # -1 occurs and the rest contains -5
    df = frame(); df.loc[[1, 2, 3, 4], "age"] = -1.0; df.loc[[10, 11], "age"] = -5.0
    assert not codes_of(df)[2], "another negative among the other values means it is not a code column"
    # a classification target with labels {-1, 1} is never touched
    df = frame(); df["y"] = np.where(np.random.default_rng(0).random(n) < 0.5, -1, 1)
    _, y, codes, _ = codes_of(df)
    assert not codes and y.notna().all() and set(y) == {"-1", "1"}, "class labels -1/1 are legitimate"
    # ZIP-like text codes: all-numeric remaining values never make a text code column
    df = frame(); df["zip"] = np.random.default_rng(0).choice(["01234", "56789", "40001"], n)
    df.loc[[5, 6, 7, 8], "zip"] = "-1"
    assert "zip" not in codes_of(df)[2]
    # a column made only of codes has nothing to compare with
    df = frame(); df["allneg"] = -1.0
    assert "allneg" not in codes_of(df)[2]
    print("OK S2 counter-cases: genuine negatives, 2 cells, 60%, another negative present, labels -1/1, ZIP-like text, all-code column")


def test_text_columns_and_numeric_text():
    n = 600
    df = frame(); rows = np.sort(np.random.default_rng(1).choice(n, 30, replace=False))
    df.loc[rows, "city"] = "-1"
    df.loc[[0, 1, 2, 3], "city"] = " -1.0 "                           # '-1.0' and padding match too
    rows = np.union1d(rows, [0, 1, 2, 3])
    X, _, codes, _ = codes_of(df)
    assert set(codes) == {"city"} and codes["city"]["rows"] == rows.tolist(), codes
    assert X["city"].isna().sum() == len(rows) and "-1" not in set(X["city"].dropna().astype(str))
    # numbers stored as text: the parsed numbers follow the numeric rule
    df = frame(); df["age"] = df["age"].astype(int).astype(str); df.loc[[4, 5, 6, 7, 8], "age"] = "-1"
    X, _, codes, _ = codes_of(df)
    assert codes["age"]["rows"] == [4, 5, 6, 7, 8] and pd.api.types.is_numeric_dtype(X["age"])
    # the existing placeholder tokens still work and are not double counted
    df = frame(); df.loc[[1, 2, 3], "city"] = "N/A"; df.loc[[10, 11, 12], "city"] = "-1"
    _, _, codes, _ = codes_of(df)
    assert codes["city"]["rows"] == [10, 11, 12]
    print("OK S2 text: '-1', '-1.0', padded, numeric text; ordinary tokens unaffected")


def test_no_codes_means_nothing_changes():
    df = frame()
    X, _, codes, major = codes_of(df)
    assert codes == {} and major == {}
    st = audit_dataset(df, "y", detectors={"missing_values", "duplicates"})
    assert st.meta["code_cells"] == {} and not any("code such as -1" in n for n in st.meta["notes"])
    log = _apply(st, {"missing_values"}, [], respect_scope=True)[1]
    assert not any("code" in str(m) for m in log["method"]) and not [n for n in st.meta["notes"] if n.startswith("Held back")]
    print("OK S0: a file without codes has no code notes, no held-back note, no method text")


def coded_frame(n=500, seed=3):
    df = frame(n=n, seed=seed)
    df.loc[np.random.default_rng(5).choice(n, 20, replace=False), "income"] = np.nan          # ordinary blanks, 20 of them
    rows = np.sort(np.random.default_rng(6).choice(np.setdiff1d(np.arange(n), df.index[df.income.isna()]), 30, replace=False))
    df.loc[rows, "age"] = -1.0                                                                  # 30 code cells
    return df, rows


def test_policy_evidence_file_never_fills_a_code_cell():
    df, rows = coded_frame()
    st = audit_dataset(df, "y", detectors={"missing_values"})
    assert sorted(st.meta["code_cells"]["age"]) == rows.tolist()
    log0 = st.proposals
    assert (log0[(log0.column == "age")].shape[0]) == 30 and (log0[log0.column == "income"].shape[0]) == 20
    cleaned, log = _apply(st, {"missing_values"}, [], respect_scope=True)
    assert (cleaned.loc[rows, "age"] == -1).all(), "the evidence-based file leaves every code cell byte-identical"
    assert cleaned["income"].isna().sum() == 0, "ordinary blanks are still filled when the type is applied"
    held = log[(log.column == "age")]
    assert (held.action == "flagged_for_review").all() and held.original_value.astype(float).eq(-1).all()
    assert held.method.str.contains("code such as -1").all(), "the log says why the cell was proposed"
    notes = [n for n in st.meta["notes"] if n.startswith("Held back for review (codes):")]
    assert len(notes) == 1 and "30 cell(s)" in notes[0] and "age (30)" in notes[0], notes
    for _ in range(3):
        _apply(st, {"missing_values"}, [], respect_scope=True)
    assert len([n for n in st.meta["notes"] if n.startswith("Held back for review (codes):")]) == 1, "one note, however often it is applied"
    # aggressive file (respect_scope off) and a human override both fill them
    agg, _ = _apply(st, {"missing_values"}, [])
    assert not (agg.loc[rows, "age"] == -1).any() and agg["age"].notna().all()
    ov, lov = _apply(st, {"missing_values"}, ["missing_values"], respect_scope=True)
    assert not (ov.loc[rows, "age"] == -1).any() and (lov[lov.column == "age"].action == "applied").all()
    assert not [n for n in st.meta["notes"] if n.startswith("Held back for review (codes):")], "no note when nothing is held back"
    print("OK S3: evidence file keeps 30 code cells, fills 20 ordinary blanks; aggressive and override fill all; one idempotent note")


def test_regression_target_codes_become_a_missing_target():
    n = 300
    rng = np.random.default_rng(2)
    df = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.normal(size=n), "grp": rng.choice(list("abc"), n)})
    df["y"] = (3 + 2 * df.x1 + rng.normal(0, 0.3, n)).clip(lower=0.5).round(2)
    bad = np.sort(rng.choice(n, 12, replace=False))
    df.loc[bad, "y"] = -1.0
    st = audit_dataset(df, "y", task="regression")
    assert st.meta["n_rows_missing_target"] == 12 and st.y.isna().sum() == 12 and st.meta["codes"]["y"]["is_target"]
    assert "y" not in st.meta["code_cells"], "a target code is a missing target, not a repairable feature cell"
    assert any("target 'y' are the code -1" in n for n in st.meta["notes"])
    p = st.proposals
    flagged_rows = set(p[p.issue_key == "label_errors"].row_id)
    assert not flagged_rows & set(bad.tolist()), "the wrong-target detector never sees a coded target row"
    cleaned, log = _apply(st, {"label_errors", "missing_values", "outliers"}, [])
    assert (cleaned.loc[bad, "y"] == -1).all(), "a coded target is never repaired, not even in the aggressive path"
    print("OK S4: 12 coded target rows are missing targets, invisible to the wrong-target detector, never repaired")


def test_baseline_snapshot_counts_codes():
    df, rows = coded_frame()
    s = build_snapshot(df)
    c = s["completeness"]
    assert c["code_cells"] == 30 and c["missing_cells"] == 20 + 30 and c["code_columns"]["age"]["count"] == 30
    assert abs(c["missing_pct"] - 50 / df.size) < 1e-12
    col = next(x for x in s["columns"] if x["name"] == "age")
    assert col["code_cells"] == 30 and col["missing"] == 30
    clean = build_snapshot(frame())["completeness"]
    assert clean["code_cells"] == 0 and clean["code_columns"] == {}
    print("OK baseline: codes counted as missing cells with their own line; clean data shows none")


def test_backend_held_back_summary_reports_both_reasons():
    import backend.main as bm
    df, rows = coded_frame()
    df["pool"] = pd.Series("Gd", index=df.index).mask(np.random.default_rng(1).random(len(df)) < 0.7)   # 70% empty: outside the calibration
    st = audit_dataset(df, "y", detectors={"missing_values"})
    st.selfbench = SimpleNamespace(ok=True, reference={"columns_left_out": {"pool": "70% missing"}})
    _, log = _apply(st, {"missing_values"}, [], respect_scope=True)
    old = bm.default_apply_issues
    bm.default_apply_issues = lambda state: {"missing_values"}
    try:
        out = bm.held_back_summary({"state": st, "result": SimpleNamespace(repair_log=log), "applied": {"missing_values"}})
    finally:
        bm.default_apply_issues = old
    mv = out["missing_values"]
    assert mv["codes"]["columns"] == {"age": 30} and mv["calibration"]["columns"].keys() == {"pool"}, mv
    assert mv["count"] == mv["codes"]["count"] + mv["calibration"]["count"], "the two reasons never overlap"
    st.selfbench = None
    _, log2 = _apply(st, {"missing_values"}, [], respect_scope=True)
    bm.default_apply_issues = lambda state: {"missing_values"}
    try:
        only = bm.held_back_summary({"state": st, "result": SimpleNamespace(repair_log=log2), "applied": {"missing_values"}})
    finally:
        bm.default_apply_issues = old
    assert only["missing_values"]["calibration"]["count"] == 0 and only["missing_values"]["codes"]["count"] == 30
    print("OK backend: held_back_summary separates calibration-scope holds from code holds, without overlap")


def test_baseline_target_block_matches_the_audit():
    from regression import target_profile_regression
    n = 300
    rng = np.random.default_rng(2)
    df = pd.DataFrame({"x": rng.normal(size=n), "y": rng.uniform(1, 5, n).round(1)})
    clean = target_profile_regression(df, "y")
    assert clean["rows_without_target"] == 0 and "code_values" not in clean
    df.loc[[3, 4, 5, 6, 7, 8, 9], "y"] = -1.0
    t = target_profile_regression(df, "y")
    assert t["rows_without_target"] == 7 and t["code_values"] == 7 and t["min"] >= 1.0, t
    assert any("treated as a missing target" in w for w in t["warnings"])
    st = audit_dataset(df, "y", task="regression", detectors={"missing_values"})
    assert st.meta["n_rows_missing_target"] == t["rows_without_target"], "the Baseline target block and the audit agree"
    print("OK baseline target block: coded target rows counted as missing, minimum no longer -1, agrees with the audit")


if __name__ == "__main__":
    tests = [v for n, v in sorted(globals().items()) if n.startswith("test_") and callable(v)]
    for t in tests:
        t()
    print(f"\nALL PHASE 25 CODE-AS-MISSING INVARIANTS HOLD ({len(tests)} tests)")
