"""
Invariants for Phase 26: word placeholders, placeholders the user declares, two diagnostics (docs/PHASE26_SPEC.md).
Run:  .venv/bin/python src/test_phase26.py        Every check can fail.

S2 planted truth for words (recall and precision 1.00 at 1/5/20%, 3 seeds) and the counter-cases that must NOT fire; declared values
(exact cells, nothing else moves, the uploaded frame is never modified); keep_values; S3 policy; S4 targets; the F3 diagnostics;
request validation. S0/S1/S5/S6 on the real files are measured by src/validate_sentinels.py, not here.
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(ROOT))

from taskclean import _apply, _clean_declarations, _prepare, audit_dataset


def frame(n=600, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "chol": rng.integers(120, 400, n).astype(float),         # the column where 0 will mean "missing"
        "oldpeak": np.where(rng.random(n) < 0.3, 0.0, rng.gamma(2.0, 1.0, n).round(1)),   # legitimate zeros: must never move
        "city": rng.choice(["Pune", "Delhi", "Goa", "Agra"], n),
        "work": rng.choice(["private", "gov", "self", "other", "none"], n),
        "y": rng.choice(["p", "q"], n)})


def prep(df, target="y", task="classification", declared=None, keep=None):
    X, y, _, _, notes, _ = _prepare(df, target, task, declared, keep)
    return X, y, notes


def test_planted_words_are_found_exactly():
    for word in ("Unknown", "unspecified", "MISSING", "Not Applicable", "Unknown / Non-Applicable"):
        for rate in (0.01, 0.05, 0.20):
            for seed in (1, 2, 3):
                df = frame(seed=seed)
                rows = np.sort(np.random.default_rng(seed).choice(len(df), int(len(df) * rate), replace=False))
                df.loc[rows, "city"] = word
                X, _, _ = prep(df)
                codes = X.attrs["codes"]
                assert set(codes) == {"city"}, (word, rate, seed, sorted(codes))                  # precision: no other column
                assert codes["city"]["rows"] == rows.tolist(), (word, rate, seed)                  # recall 1.00, no extra cell
                assert X["city"].isna().sum() == len(rows) and df["city"].iloc[rows[0]] == word      # the frame is never modified
    print("OK S2 words: 5 spellings x 1/5/20% x 3 seeds, recall 1.00, precision 1.00")


def test_word_counter_cases_do_not_fire():
    df = frame()
    df.loc[[1, 2], "city"] = "Unknown"
    assert not prep(df)[0].attrs["codes"], "two cells are not a placeholder"
    df = frame()
    df.loc[: int(len(df) * 0.6), "city"] = "Unknown"
    X, _, notes = prep(df)
    assert not X.attrs["codes"] and "city" in X.attrs["code_majority"] and X["city"].isna().sum() == 0, "60% is a category"
    assert any("kept as a category" in n for n in notes)
    for text in ("none", "other", "Other-service", "Unknown Soldier", "missing (wrongly coded?)", "Unknown, TX"):
        df = frame()
        df.loc[10:40, "city"] = text
        assert not prep(df)[0].attrs["codes"], f"'{text}' must not be read as a placeholder"
    df = frame()
    df["work"] = "none"
    df.loc[:5, "work"] = "other"
    assert not prep(df)[0].attrs["codes"], "'none' and 'other' are real categories"
    print("OK S2 counter-cases: 2 cells, 60%, none, other, extra words, 'Unknown, TX', missing (wrongly coded?)")


def test_words_and_codes_in_one_column_and_numeric_text():
    df = frame()
    df["size"] = pd.Series(np.random.default_rng(2).choice(["10", "20", "30"], len(df)))
    df.loc[[3, 4, 5, 6], "size"] = "Unknown"
    df.loc[[10, 11, 12], "size"] = "-1"
    X, _, notes = prep(df)
    c = X.attrs["codes"]["size"]
    assert c["rows"] == [3, 4, 5, 6, 10, 11, 12] and c["n_word"] == 4 and c["n_code"] == 3 and c["count"] == 7, c
    assert pd.api.types.is_numeric_dtype(X["size"]), "a number-with-words column parses as numeric once the words are missing"
    assert any("placeholder word" in n and "size (4)" in n for n in notes) and any("a code such as -1" in n and "size (3)" in n for n in notes)
    print("OK words and -1 in one column are merged and counted per kind; numeric text with 'Unknown' is numeric")


def test_declared_values_are_exact_and_nothing_else_moves():
    df = frame()
    rows = np.sort(np.random.default_rng(4).choice(len(df), 40, replace=False))
    df.loc[rows, "chol"] = 0.0
    before = df.copy()
    base, _, _ = prep(df)
    X, _, notes = prep(df, declared={"chol": ["0"]})
    d = X.attrs["declared"]["chol"]
    assert d["rows"] == rows.tolist() and d["count"] == 40 and X["chol"].isna().sum() == 40, d     # recall and precision 1.00
    assert (X["oldpeak"] == 0).sum() == (df["oldpeak"] == 0).sum() > 100, "legitimate zeros elsewhere are never touched"
    for c in ("oldpeak", "city", "work"):
        assert X[c].equals(base[c]) or X[c].astype(str).equals(base[c].astype(str)), c
    assert not X.attrs["codes"], "declared cells are not 'codes': they are known missing"
    pd.testing.assert_frame_equal(df, before)
    assert any("You declared" in n and "chol: 0 (40)" in n for n in notes), notes
    X2, _, _ = prep(df, declared={"chol": ["0.0"]})                       # '0.0' and 0 are the same number
    assert X2.attrs["declared"]["chol"]["count"] == 40
    print("OK F2: declared 0 matches exactly the 40 planted cells; 30% legitimate zeros in oldpeak untouched; frame unmodified")


def test_declared_text_and_case_and_unmatched():
    df = frame()
    df.loc[[5, 6, 7], "city"] = "N.D."
    df.loc[[8], "city"] = "n.d."
    X, _, notes = prep(df, declared={"city": ["N.D."], "work": ["zzz"]})
    assert X.attrs["declared"]["city"]["rows"] == [5, 6, 7, 8], "case-insensitive text equality"
    assert X.attrs["declared"]["work"]["count"] == 0 and any("match no cell" in n and "work: zzz" in n for n in notes)
    print("OK F2: text values match case-insensitively; a value matching nothing is reported, not an error")


def test_keep_values_switches_off_detection_only_there():
    df = frame()
    df.loc[[1, 2, 3, 4], "city"] = "Unknown"
    df["temp"] = np.random.default_rng(0).integers(0, 50, len(df)).astype(float)
    df.loc[[1, 2, 3, 4, 5], "temp"] = -1.0
    df.loc[[7, 8, 9], "work"] = "Unknown"
    X, _, notes = prep(df, keep={"city", "temp"})
    assert set(X.attrs["codes"]) == {"work"}, "only the columns not kept are detected"
    assert X["city"].isna().sum() == 0 and (X["temp"] == -1).sum() == 5, "kept columns keep their values"
    assert not any("temp" in n for n in notes if "code such as" in n or "kept as a category" in n)
    Xd, _, _ = prep(df, declared={"city": ["Unknown"]}, keep={"city"})
    assert Xd["city"].isna().sum() == 4, "declaring still applies to a kept column"
    full, _, _ = prep(df)
    assert set(full.attrs["codes"]) == {"city", "temp", "work"}
    print("OK keep_values: genuine -1 and 'Unknown' stay values in the kept columns; others unaffected; declared still applies")


def test_no_patterns_means_nothing_changes():
    df = frame()
    X, y, notes = prep(df)
    assert X.attrs["codes"] == {} and X.attrs["declared"] == {} and X.attrs["code_majority"] == {}
    assert not any(k in n for n in notes for k in ("placeholder word", "declared", "ranges", "second line"))
    st = audit_dataset(df, "y", detectors={"missing_values"})
    assert st.meta["declared"] == {} and st.meta["keep_values"] == []
    _, log = _apply(st, {"missing_values"}, [], respect_scope=True)
    assert not any("read as missing" in str(m) or "declared missing" in str(m) for m in log["method"])
    from taskclean import _summary
    s = _summary(st, log, df, {"missing_values"}, [])
    assert "declared_missing" not in s and "keep_values" not in s, "audits without declarations keep the old summary keys"
    print("OK S0: no words, no declarations, no diagnostics: no notes, no method text, summary unchanged")


def words_frame(n=500, seed=3):
    df = frame(n=n, seed=seed)
    df.loc[np.random.default_rng(5).choice(n, 20, replace=False), "oldpeak"] = np.nan                     # ordinary blanks
    rows = np.sort(np.random.default_rng(6).choice(np.setdiff1d(np.arange(n), df.index[df.oldpeak.isna()]), 30, replace=False))
    df.loc[rows, "city"] = "Unknown"
    return df, rows


def test_policy_words_held_back_declared_applied():
    df, rows = words_frame()
    drows = np.sort(np.random.default_rng(8).choice(np.setdiff1d(np.arange(len(df)), rows), 25, replace=False))
    df.loc[drows, "chol"] = 0.0
    st = audit_dataset(df, "y", detectors={"missing_values"}, declared_missing={"chol": ["0"]})
    assert sorted(st.meta["code_cells"]["city"]) == rows.tolist() and sorted(st.meta["declared_cells"]["chol"]) == drows.tolist()
    cleaned, log = _apply(st, {"missing_values"}, [], respect_scope=True)
    assert (cleaned.loc[rows, "city"] == "Unknown").all(), "the evidence-based file leaves every word cell as it was"
    held = log[log.column == "city"]
    assert (held.action == "flagged_for_review").all() and (held.original_value == "Unknown").all()
    assert held.method.str.contains("word such as 'Unknown'").all() and not held.method.str.contains("code such as -1").any()
    assert not (cleaned.loc[drows, "chol"] == 0).any(), "declared cells are imputed like other missing cells (not held back)"
    dl = log[log.column == "chol"]
    assert (dl.action == "applied").all() and dl.method.str.contains("declared missing by you").all() and (dl.original_value.astype(float) == 0).all()
    assert cleaned["oldpeak"].isna().sum() == 0, "ordinary blanks are still filled"
    notes = [n for n in st.meta["notes"] if n.startswith("Held back for review (codes):")]
    assert len(notes) == 1 and "city (30)" in notes[0] and "chol" not in notes[0], notes
    agg, _ = _apply(st, {"missing_values"}, [])
    assert not (agg.loc[rows, "city"] == "Unknown").any(), "the aggressive file fills the words"
    ov, lov = _apply(st, {"missing_values"}, ["missing_values"], respect_scope=True)
    assert not (ov.loc[rows, "city"] == "Unknown").any() and (lov[lov.column == "city"].action == "applied").all()
    untouched = np.setdiff1d(np.arange(len(df)), np.concatenate([rows, drows, df.index[df.oldpeak.isna()]]))
    assert (cleaned.loc[untouched, ["chol", "city"]].to_numpy() == df.loc[untouched, ["chol", "city"]].to_numpy()).all()
    print("OK S3: word cells kept in the evidence file, filled by aggressive/override; declared cells imputed, logged with original 0")


def test_declared_target_is_a_missing_target_never_repaired():
    n = 300
    rng = np.random.default_rng(2)
    reg = pd.DataFrame({"x": rng.normal(size=n), "z": rng.integers(0, 5, n), "y": rng.uniform(1, 9, n).round(1)})
    bad = np.arange(10, 25)
    reg.loc[bad, "y"] = 0.0
    st = audit_dataset(reg, "y", task="regression", detectors={"missing_values", "label_noise"}, declared_missing={"y": ["0"]})
    assert st.meta["n_rows_missing_target"] == 15 and st.y.isna().sum() == 15 and st.meta["declared"]["y"]["is_target"]
    assert "y" not in st.meta["declared_cells"]
    assert not (set(st.proposals[st.proposals.issue_key == "label_errors"].row_id) & set(bad.tolist()))
    cleaned, _ = _apply(st, {"missing_values", "label_errors"}, [])
    assert (cleaned.loc[bad, "y"] == 0).all(), "a declared target value is never imputed, even in the aggressive path"
    assert any("missing target" in n and "declared" in n.lower() for n in st.meta["notes"])
    cls = pd.DataFrame({"x": rng.normal(size=n), "y": rng.choice(["a", "b", "Unknown"], n, p=[0.45, 0.45, 0.1])})
    st2 = audit_dataset(cls, "y", detectors={"missing_values", "label_noise"}, declared_missing={"y": ["Unknown"]})
    nun = int((cls.y == "Unknown").sum())
    assert st2.meta["n_rows_missing_target"] == nun > 0 and set(st2.y.dropna()) == {"a", "b"}
    cl2, _ = _apply(st2, {"missing_values", "label_errors"}, [])
    assert (cl2.loc[cls.y == "Unknown", "y"] == "Unknown").all(), "a declared label is never relabelled"
    print("OK S4: declared target values (regression 15, classification) are missing targets: excluded, never repaired")


def test_diagnostics_ranges_and_glued_number():
    n = 200
    rng = np.random.default_rng(1)
    rating = rng.choice([3.0, 3.5, 4.0, 4.5], n)
    lo = rng.integers(40, 90, n)
    df = pd.DataFrame({
        "salary": [f"${a}K-${a + 30}K (Glassdoor est.)" for a in lo],
        "band": rng.choice(["1 to 50 employees", "51 to 200 employees"], n),
        "company": [f"Acme {i % 40}\n{r}" for i, r in enumerate(rating)],
        "orphan": [f"Name {i % 40}\n{v}" for i, v in enumerate(rng.integers(100, 999, n))],
        "partial": [f"Item {i % 40}\n{v}" for i, v in enumerate(np.where(np.arange(n) < n // 2, 700, rng.integers(100, 600, n)))],
        "rating": rating, "other": rng.normal(size=n)})
    df["half"] = 700                                          # equals the glued number in only ~50% of 'partial' rows: below the 95% bar
    df.loc[n // 2:, "half"] = 1
    X, _, notes = prep(df, "rating", "regression")
    rng_notes = [x for x in notes if "numeric ranges" in x]
    assert len(rng_notes) == 1 and "'salary'" in rng_notes[0], notes
    glued = [x for x in notes if "second line" in x]
    assert len(glued) == 1 and "'company'" in glued[0] and "'rating'" in glued[0] and "100.0%" in glued[0], glued
    assert "is the target" in glued[0] and "orphan" not in " ".join(glued) and "partial" not in " ".join(glued), \
        "no matching column, or a column matching only half the rows, gives no note"
    _, _, n2 = prep(df, "other", "regression")
    g2 = [x for x in n2 if "second line" in x]
    assert len(g2) == 1 and "is the target" not in g2[0], "the stronger wording is only for the target"
    assert not any("numeric ranges" in x and "band" in x for x in notes), "'1 to 50 employees' is not a numeric range"
    assert X["salary"].dtype != float, "nothing is converted"
    print("OK F3: ranges and a glued sibling-matching number are reported (stronger for the target); no match, no note; nothing converted")


def test_request_validation():
    df = frame()
    for bad_decl, bad_keep in (({"nope": ["0"]}, []), ({}, ["nope"]), ({"city": [str(i) for i in range(21)]}, []), ({"city": ["x" * 41]}, [])):
        try:
            _clean_declarations(df, bad_decl, bad_keep)
        except ValueError:
            continue
        raise AssertionError((bad_decl, bad_keep))
    d, k = _clean_declarations(df, {"city": [" 0 ", "0", ""]}, ["work"])
    assert d == {"city": ["0"]} and k == {"work"}
    import time
    from fastapi import HTTPException
    import backend.main as bm
    did = bm.store_dataset(df, {"format": "csv"}, "t.csv")
    for bad in ({"declared_missing": {"nope": ["0"]}}, {"keep_values": ["nope"]}):
        try:
            bm.start_audit(bm.AuditRequest(dataset_id=did, target="y", **bad))
        except HTTPException as e:
            assert e.status_code == 400 and "nope" in e.detail, e.detail
        else:
            raise AssertionError(bad)
    df2 = frame()
    df2.loc[:29, "chol"] = 0.0
    did2 = bm.store_dataset(df2, {"format": "csv"}, "t2.csv")
    job = bm.start_audit(bm.AuditRequest(dataset_id=did2, target="y", declared_missing={"chol": ["0"]}, keep_values=["city"],
                                         model_panel=False))["job_id"]
    for _ in range(240):
        j = bm.read_job(job)
        if j["status"] in ("done", "error"):
            break
        time.sleep(0.5)
    assert j["status"] == "done", j
    a = bm.AUDITS[j["audit_id"]]
    assert a["state"].meta["declared"]["chol"]["count"] == 30 and a["state"].meta["keep_values"] == ["city"]
    assert a["result"].summary["declared_missing"]["chol"]["count"] == 30 and a["result"].summary["keep_values"] == ["city"]
    print("OK request validation: unknown column, too many/long values are errors (400); a valid declaration is accepted")


if __name__ == "__main__":
    tests = [v for n, v in sorted(globals().items()) if n.startswith("test_") and callable(v)]
    for t in tests:
        t()
    print(f"\nALL PHASE 26 INVARIANTS HOLD ({len(tests)} tests)")
