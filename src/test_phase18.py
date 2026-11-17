"""
Invariants for the baseline health report (Phase 18, docs/PHASE18_SPEC.md). Run:  .venv/bin/python src/test_phase18.py

H1 agreement with the audit, H2 exactness on planted ground truth, H3 immutability and determinism, H4 robustness,
H5 speed (measured on a real 48,842-row file when it is present). Every check can fail.
"""

import json
import os
import time

import numpy as np
import pandas as pd

import health_snapshot as hs
from taskclean import audit_dataset, load_csv

ROOT = os.path.join(os.path.dirname(__file__), "..")


def planted_frame():
    """300 base rows with known planted problems, then 12 exact duplicates and 5 near-duplicates appended."""
    rng = np.random.default_rng(0)
    n = 300
    age = rng.integers(20, 60, n).astype(float)
    income = rng.integers(40, 60, n).astype(float)
    income[:6] = 1000                                     # 6 planted outliers: beyond any IQR fence and |Z| > 3
    spend = rng.lognormal(0, 1.5, n).round(3)             # heavily skewed
    city = rng.choice(["Paris", "Rome", "Oslo", "Lima"], n).astype(object)
    state = np.array(["Texas"] * n, dtype=object)
    state[10:15] = "texas"                                # 5 case variants
    state[15:18] = "Texas "                               # 3 trailing-space variants
    label = rng.choice(["a", "b"], n).astype(object)
    df = pd.DataFrame({"age": age, "income": income, "spend": spend, "city": city, "state": state, "label": label})
    df.loc[20:24, "age"] = np.nan                         # 5 blank numeric cells
    df.loc[30:39, "city"] = np.nan                        # 10 blank text cells
    df.loc[40:46, "city"] = "N/A"                         # 7 placeholders
    df.loc[47:49, "city"] = "?"                           # 3 placeholders
    dups = df.iloc[100:112].copy()                        # 12 exact copies
    near = df.iloc[200:205].copy()
    near["city"] = near["city"].str.upper() + " "         # 5 rows equal to originals apart from case and a trailing space
    return pd.concat([df, dups, near], ignore_index=True)


def test_h2_planted_counts_are_exact():
    df = planted_frame()
    s = hs.build_snapshot(df, {"encoding": "utf-8"}, "planted.csv", b"x" * 1234)
    c = s["completeness"]
    assert (c["blank_cells"], c["placeholder_cells"], c["missing_cells"]) == (5 + 10, 10, 25), c
    # the 12 duplicated rows and 5 near-duplicate rows are clean copies of rows 100-111 / 200-204: no extra missing cells
    assert c["rows_with_missing"] == 25 and c["columns_entirely_missing"] == []
    assert s["duplicates"]["exact_rows"] == 12 and s["duplicates"]["near_extra_rows"] == 5
    assert s["duplicates"]["near_examples"] == list(range(312, 316)), "examples are the first four of the last five rows"
    assert s["duplicates"]["exact_examples"] == [300, 301, 302, 303], "the exact copies sit right after the 300 base rows"
    a = s["anomalies"]
    assert a["iqr"]["per_column"]["income"] == 6 and a["zscore"]["per_column"]["income"] == 6
    assert a["iqr"]["outlier_rows"] >= 6 and a["zscore"]["outlier_rows"] >= 6
    skew_cols = {x["column"] for x in a["skewed_columns"]}
    assert {"income", "spend"} <= skew_cols and "age" not in skew_cols, skew_cols
    # 8 planted state variants, plus the 5 upper-cased "PARIS " style city variants made for the near-duplicates
    inc = s["inconsistencies"]
    by = {b["column"]: b["cells"] for b in inc["by_column"]}
    assert by["state"] == 8 and by["city"] == 5 and inc["cells"] == 13, by
    ov = s["overview"]
    assert ov["rows"] == 317 and ov["columns"] == 6 and ov["file_bytes"] == 1234 and len(ov["sha256"]) == 64
    assert ov["type_counts"] == {"numeric": 3, "categorical": 3}, ov["type_counts"]
    assert len(s["not_covered"]) == 3 and any("health score" in x for x in s["not_covered"]), "limits must be stated"
    print("OK planted counts: missing/placeholders, exact and near duplicates, IQR and |Z| outliers, skew, spellings")


def formats_frame():
    n = 50
    rng = np.random.default_rng(1)
    return pd.DataFrame({
        "price": [f"${v:,}" for v in rng.choice([1200, 850, 3400, 15000, 99], n)],  # formatted: currency + thousands (repeats)
        "pct": [f"{v}%" for v in rng.integers(1, 90, n)],                          # formatted: percent
        "plain": [str(v) for v in rng.integers(1, 9, n)],                          # plain numbers stored as text
        "country": ["US"] * n,                                                     # constant
        "code": [f"id-{i:04d}" for i in range(n)],                                 # every value distinct text
        "seq": np.arange(1, n + 1),                                                # distinct integers
        "score": rng.normal(size=n).round(4),                                      # distinct floats: NOT an identifier
        "group": rng.choice(["x", "y", "z"], n)})


def test_h2_kinds_constant_and_identifier_flags():
    s = hs.build_snapshot(formats_frame())
    kinds = {c["name"]: c["kind"] for c in s["columns"]}
    assert kinds["price"] == "formatted number stored as text" and kinds["pct"] == "formatted number stored as text"
    assert kinds["plain"] == "numeric stored as text", kinds["plain"]
    assert kinds["seq"] == "numeric" and kinds["country"] == "categorical" and kinds["group"] == "categorical"
    a = s["anomalies"]
    assert a["constant_columns"] == ["country"], a["constant_columns"]
    assert set(a["possible_identifier_columns"]) == {"code", "seq"}, a["possible_identifier_columns"]
    flags = {c["name"]: " ".join(c["flags"]) for c in s["columns"]}
    assert "audit reads this as categorical" in flags["price"] and "identifier" not in flags["score"]
    assert s["overview"]["type_counts"]["formatted number stored as text"] == 2
    # a column of ordinary text that merely contains digits is not mistaken for a formatted number
    t = hs.build_snapshot(pd.DataFrame({"a": ["12 Main St", "5 Oak Rd", "9 Elm Ave"], "b": [1, 2, 3]}))
    assert {c["name"]: c["kind"] for c in t["columns"]}["a"] == "categorical"
    assert not hs._is_formatted_number_column(pd.Series(["1", "2", "3"]))             # all plain: not "formatted"
    assert hs._is_formatted_number_column(pd.Series(["$1,200", "850", None]))          # mixed plain + formatted, blanks ignored
    assert not hs._is_formatted_number_column(pd.Series(["$1,200", "abc"]))
    print("OK kinds: formatted vs plain numbers-as-text, constant and identifier rules (floats are not identifiers)")


def test_h1_snapshot_agrees_with_the_audit():
    df = planted_frame()
    s = hs.build_snapshot(df)
    st = audit_dataset(df, "label", detectors={"missing_values", "duplicates", "outliers", "inconsistency"})
    audit_cols = [c for c in df.columns if c != "label"]
    cm = st.detections["missing_values"].extra["cell_mask"]
    snap_missing = {c["name"]: c["missing"] for c in s["columns"]}
    assert {c: int(cm[c].sum()) for c in audit_cols} == {c: snap_missing[c] for c in audit_cols}
    assert s["duplicates"]["exact_rows"] == int(st.detections["duplicates"].row_mask.sum())
    assert s["anomalies"]["iqr"]["outlier_rows"] == int(st.detections["outliers"].row_mask.sum())
    assert s["inconsistencies"]["cells"] == len(st.detections["inconsistency"].extra["cells"])
    print("OK agreement: missing cells per column, duplicates, IQR outlier rows and spelling cells equal the audit's")


def test_h3_immutable_and_deterministic():
    df = planted_frame()
    before = int(pd.util.hash_pandas_object(df, index=True).sum())
    cols, dtypes = list(df.columns), df.dtypes.copy()
    a = hs.build_snapshot(df, {"d": 1}, "f.csv", b"abc")
    assert int(pd.util.hash_pandas_object(df, index=True).sum()) == before and list(df.columns) == cols
    assert (df.dtypes == dtypes).all()
    b = hs.build_snapshot(df, {"d": 1}, "f.csv", b"abc")
    a.pop("captured_at_utc"), b.pop("captured_at_utc")
    assert a == b, "same input must give the same snapshot"
    # a non-default index is reset for the snapshot's own row ids, but the caller's frame keeps it
    d2 = df.copy()
    d2.index = d2.index + 1000
    hs.build_snapshot(d2)
    assert d2.index[0] == 1000
    json.dumps(a)                                                                   # JSON-safe: no NaN objects, no numpy
    print("OK immutability and determinism: input unchanged, same snapshot twice, JSON-safe")


def test_h4_robustness():
    cases = {
        "all-missing column": pd.DataFrame({"a": [np.nan] * 10, "b": range(10)}),
        "single column": pd.DataFrame({"a": range(30)}),
        "one row": pd.DataFrame({"a": [1], "b": ["x"]}),
        "text only": pd.DataFrame({"a": ["u", "v", "u", "w"], "b": ["p", "q", "p", "q"]}),
        "no usable feature": pd.DataFrame({"a": [np.nan] * 5}),
        "zero rows": pd.DataFrame({"a": pd.Series([], dtype=float), "b": pd.Series([], dtype=object)}),
        "no duplicates": pd.DataFrame({"a": range(40), "b": list("xy") * 20}),
        "mixed types": pd.DataFrame({"a": [1, "x", 2.5, None, True] * 6, "b": range(30)}),
        "datetime column": pd.DataFrame({"t": pd.date_range("2020-01-01", periods=30), "b": range(30)}),
        "bool column": pd.DataFrame({"f": [True, False] * 15, "b": range(30)}),
    }
    for name, df in cases.items():
        s = hs.build_snapshot(df)
        json.dumps(s)
        assert s["overview"]["rows"] == len(df) and s["overview"]["columns"] == df.shape[1], name
        assert 0.0 <= s["completeness"]["missing_pct"] <= 1.0, name
    assert hs.build_snapshot(cases["all-missing column"])["completeness"]["columns_entirely_missing"] == ["a"]
    nu = hs.build_snapshot(cases["no usable feature"])
    assert nu["duplicates"] is None and nu["anomalies"]["iqr"] is None and any("could not run" in n for n in nu["notes"])
    assert hs.build_snapshot(cases["no duplicates"])["duplicates"]["exact_rows"] == 0
    assert hs.build_snapshot(cases["one row"])["duplicates"]["exact_rows"] == 0
    kinds = {c["name"]: c["kind"] for c in hs.build_snapshot(cases["datetime column"])["columns"]}
    assert kinds["t"] == "datetime"
    print("OK robustness: all-missing, single column, one row, zero rows, no usable feature, mixed, datetime, bool")


def test_target_profile():
    df = pd.DataFrame({"y": ["a"] * 70 + ["b"] * 25 + ["c"] * 3 + [np.nan] * 2, "x": range(100)})
    t = hs.target_profile(df, "y")
    assert t["rows_without_target"] == 2 and t["n_classes"] == 3 and t["usable"]
    assert abs(t["imbalance_ratio"] - 70 / 3) < 1e-9 and t["rare_classes"] == ["c"]
    assert sum(c["rows"] for c in t["classes"]) == 98 and abs(sum(c["share"] for c in t["classes"]) - 1) < 1e-9
    assert not hs.target_profile(pd.DataFrame({"y": range(50)}), "y")["usable"], "50 distinct values: not a class target"
    assert not hs.target_profile(pd.DataFrame({"y": ["a"] * 9}), "y")["usable"]
    try:
        hs.target_profile(df, "missing_col")
    except ValueError as e:
        assert "not found" in str(e)
    else:
        raise AssertionError("an unknown target must be rejected")
    print("OK target block: counts, imbalance, rare classes, unusable targets")


def test_csv_export_matches_the_snapshot():
    s = hs.build_snapshot(planted_frame(), {"encoding": "utf-8"}, "planted.csv", b"abc")
    t = hs.snapshot_rows(s)
    assert list(t.columns) == ["section", "item", "metric", "value"]
    get = lambda sec, item, metric: t[(t.section == sec) & (t.item == item) & (t.metric == metric)]["value"].iloc[0]  # noqa: E731
    assert get("overview", "", "rows") == 317 and get("completeness", "", "missing_cells") == 25
    assert get("duplicates", "", "exact_rows") == 12 and get("duplicates", "", "near_extra_rows") == 5
    assert get("outliers (zscore)", "income", "flagged_cells") == 6 and get("outliers (iqr)", "income", "flagged_cells") == 6
    assert get("inconsistent spellings", "state", "cells") == 8
    assert (t.section == "not covered").sum() == 3
    # no value in the table that is not in the snapshot: spot-check every per-column missing count
    for c in s["columns"]:
        assert get("columns", c["name"], "missing") == c["missing"]
    print("OK CSV export: one row per reported number, values equal the snapshot")


def test_backend_stores_the_snapshot_at_upload_and_never_recomputes_it():
    """Calls the backend functions directly (the HTTP client package is not installed); the HTTP path is checked in the browser."""
    import hashlib
    import sys
    sys.path.insert(0, ROOT)
    from fastapi import HTTPException
    from backend.main import DATASETS, dataset_payload, read_health, store_dataset
    csv = planted_frame().to_csv(index=False).encode("utf-8")
    df, info = load_csv(csv)
    ds_id = store_dataset(df, info, "planted.csv", raw=csv)
    assert DATASETS[ds_id]["snapshot"] is not None and DATASETS[ds_id]["snapshot_error"] is None
    assert dataset_payload(ds_id)["id"] == ds_id
    h1 = read_health(ds_id)
    snap = h1["snapshot"]
    assert snap["overview"]["sha256"] == hashlib.sha256(csv).hexdigest() and snap["overview"]["file_bytes"] == len(csv)
    assert snap["completeness"]["missing_cells"] == 25 and snap["duplicates"]["exact_rows"] == 12
    assert h1["target"] is None
    json.dumps(h1)                                                                   # what the API returns is JSON-safe
    h2 = read_health(ds_id, "label")
    assert h2["snapshot"] == snap, "the stored snapshot must be returned unchanged (same capture time and numbers)"
    assert h2["target"]["n_classes"] == 2 and h2["target"]["rows_without_target"] == 0
    for call, code in ((lambda: read_health(ds_id, "nope"), 400), (lambda: read_health("doesnotexist"), 404)):
        try:
            call()
        except HTTPException as e:
            assert e.status_code == code, e.status_code
        else:
            raise AssertionError("expected an HTTP error")
    # a snapshot failure must never block the upload: the dataset is still stored, with the error recorded
    import health_snapshot
    import backend.main as bm
    original = bm.build_snapshot
    bm.build_snapshot = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
    try:
        broken = bm.store_dataset(df, info, "x.csv", raw=csv)
    finally:
        bm.build_snapshot = original
    assert DATASETS[broken]["snapshot"] is None and "boom" in DATASETS[broken]["snapshot_error"]
    try:
        read_health(broken)
    except HTTPException as e:
        assert e.status_code == 500 and "boom" in e.detail
    else:
        raise AssertionError("a missing snapshot must be reported, not hidden")
    assert health_snapshot is not None
    print("OK backend: snapshot captured at upload with the file's SHA-256, returned unchanged, upload survives a failure")


def test_real_data_findings_telco_regressions():
    """Found by testing on IBM Telco churn (docs/PHASE18_SPEC.md, "Real-data test"): a 0/1 flag column was reported as
    heavily skewed, and 'No phone service' was reported as a spelling variant of 'No'."""
    rng = np.random.default_rng(3)
    n = 400
    senior = (rng.random(n) < 0.16).astype(int)               # binary, 16% ones: skew about +1.9
    assert abs(pd.Series(senior).skew()) > hs.SKEW_LIMIT, "test data must be skewed to prove the skip"
    lines = np.array(["No"] * 220 + ["Yes"] * 150 + ["No phone service"] * 30)
    rng.shuffle(lines)
    tenure = rng.integers(0, 72, n)
    df = pd.DataFrame({"SeniorCitizen": senior, "MultipleLines": lines, "tenure": tenure})
    s = hs.build_snapshot(df, {}, "telco_like.csv", b"x")
    assert "SeniorCitizen" not in {x["column"] for x in s["anomalies"]["skewed_columns"]}, "binary column is not 'skewed'"
    assert s["inconsistencies"]["cells"] == 0, s["inconsistencies"]
    # a three-valued skewed column is still reported (the skip is for two-valued columns only)
    tri = pd.DataFrame({"x": [0] * 340 + [1] * 40 + [5] * 20})
    s3 = hs.build_snapshot(tri, {}, "tri.csv", b"x")
    assert [x["column"] for x in s3["anomalies"]["skewed_columns"]] == ["x"], s3["anomalies"]["skewed_columns"]


def test_inconsistencies_are_broken_down_by_rule():
    df = pd.DataFrame({"sex": ["Female"] * 90 + ["female "] * 3 + ["Fem"] * 7 + ["Male"] * 100})
    s = hs.build_snapshot(df, {}, "sex.csv", b"x")
    inc = s["inconsistencies"]
    assert inc["by_rule"] == {"abbreviation": 7, "format_variant": 3}, inc["by_rule"]
    assert sum(inc["by_rule"].values()) == inc["cells"] == 10
    assert inc["by_column"][0]["by_rule"] == inc["by_rule"]
    assert all(e["rule"] in hs.INCONSISTENCY_RULE_LABELS for e in inc["by_column"][0]["examples"])
    t = hs.snapshot_rows(s)
    got = t[(t.section == "inconsistent spellings") & t.metric.str.startswith("cells by rule")]
    assert dict(zip(got.metric, got.value)) == {"cells by rule: abbreviation": 7, "cells by rule: format_variant": 3}


def test_h5_speed_on_a_real_file():
    path = os.path.join(ROOT, "data", "adult_original.csv")
    if not os.path.exists(path):
        print("skip speed test: adult_original.csv not present")
        return
    raw = open(path, "rb").read()
    df, info = load_csv(raw)
    t0 = time.monotonic()
    s = hs.build_snapshot(df, info, "adult_original.csv", raw)
    dt = time.monotonic() - t0
    assert dt < 5.0, f"snapshot took {dt:.1f}s on {len(df):,} rows (limit 5s)"
    assert s["overview"]["rows"] == len(df)
    print(f"OK speed: {len(df):,} rows in {dt:.1f}s")


if __name__ == "__main__":
    tests = [v for n, v in sorted(globals().items()) if n.startswith("test_") and callable(v)]
    for t in tests:
        t()
    print(f"\nALL PHASE 18 HEALTH-SNAPSHOT INVARIANTS HOLD ({len(tests)} tests)")
