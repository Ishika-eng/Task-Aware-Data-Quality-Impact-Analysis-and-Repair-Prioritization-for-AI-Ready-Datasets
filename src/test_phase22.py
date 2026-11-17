"""
Invariants for Excel intake (Phase 22; docs/PHASE22_SPEC.md).
Run:  .venv/bin/python src/test_phase22.py
E1 round trip, E2 sheets, E3 layout, E4 robustness, E5 end to end (Excel upload audits identically to the same CSV).
E0 (everything else unchanged) is the other suites passing. Every check can fail.
"""
import io
import os
import sys
import zipfile

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.abspath(ROOT))

import openpyxl

from intake import SUPPORTED, load_table


def frame(n=60, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"age": rng.integers(18, 80, n), "city": rng.choice(["Pune", "Delhi", "Goa"], n),
                         "score": rng.normal(size=n).round(3), "label": rng.choice(["yes", "no"], n),
                         "when": pd.date_range("2024-01-01", periods=n), "ok": rng.random(n) > 0.5})


def xlsx(sheets: dict, **kw) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        for name, (df, opts) in sheets.items():
            df.to_excel(w, sheet_name=name, index=False, **opts)
    return buf.getvalue()


def test_e1_round_trip():
    df = frame()
    df.loc[3, "age"] = np.nan
    df.loc[5, "city"] = None
    got, info = load_table(xlsx({"Data": (df, {})}), "t.xlsx")
    assert info["format"] == "xlsx" and info["sheet"] == "Data" and info["rows"] == len(df)
    assert got.equals(df), "numbers, text, missing cells, dates and booleans must survive an .xlsx round trip"
    assert got.dtypes["when"].kind == "M" and got.dtypes["ok"] == bool
    got2, info2 = load_table(xlsx({"Data": (df, {})}), "no_extension_given")            # recognised by content
    assert info2["format"] == "xlsx" and got2.equals(df)
    assert ".xlsx" in SUPPORTED and ".xlsm" in SUPPORTED
    print("OK E1: .xlsx round trip is exact (numbers, text, missing, dates, booleans); recognised by content too")


def test_e2_sheets():
    a, b = frame(30, 1), frame(40, 2)
    empty = pd.DataFrame()
    raw = xlsx({"Empty": (empty, {}), "First": (a, {}), "Second": (b, {})})
    got, info = load_table(raw, "t.xlsx")
    assert info["sheet"] == "First" and got.equals(a), "an empty first sheet is skipped"
    assert info["sheets"] == ["Empty", "First", "Second"] and any("3 sheets" in n for n in info["notes"])
    got, info = load_table(raw, "t.xlsx", sheet="Second")
    assert info["sheet"] == "Second" and got.equals(b)
    try:
        load_table(raw, "t.xlsx", sheet="Nope")
    except ValueError as e:
        assert "Nope" in str(e) and "First" in str(e), e
    else:
        raise AssertionError("an unknown sheet must be a clear error that lists the sheets")
    print("OK E2: default is the first sheet with data, a named sheet is read, an unknown name lists the sheets")


def test_e3_layout():
    df = pd.DataFrame({"a": [1, 2, None, 4], "b": ["x", "y", "z", "w"]})
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "S"
    ws.cell(row=3, column=2, value="alpha")                    # two blank rows and a blank column before the table
    ws.cell(row=3, column=3, value=None)                       # blank header cell
    ws.cell(row=3, column=4, value="alpha")                    # duplicate header
    for i, (x, y) in enumerate(zip([1, 2, 3, 4], ["p", "q", "r", "s"]), start=4):
        ws.cell(row=i, column=2, value=x)
        ws.cell(row=i, column=3, value=y)
        ws.cell(row=i, column=4, value=x * 10)
    ws.cell(row=6, column=2).value = None
    ws.cell(row=6, column=3).value = None
    ws.cell(row=6, column=4).value = None                       # a fully empty row inside the table
    buf = io.BytesIO()
    wb.save(buf)
    got, info = load_table(buf.getvalue(), "t.xlsx")
    assert list(got.columns) == ["alpha", "Unnamed: 1", "alpha.1"], list(got.columns)
    assert len(got) == 3 and any("empty row" in n for n in info["notes"]), (len(got), info["notes"])
    print("OK E3: leading blank rows/columns skipped, blank and duplicate headers made unique, empty rows dropped and counted")


def test_e3b_merged_and_formulas_are_reported():
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["g", "v", "calc"])
    ws.append(["A", 1, "=B2*2"])
    ws.append([None, 2, "=B3*2"])
    ws.merge_cells("A2:A3")
    buf = io.BytesIO()
    wb.save(buf)                                               # formulas were never calculated by Excel: no stored result
    got, info = load_table(buf.getvalue(), "t.xlsx")
    joined = " ".join(info["notes"])
    assert "merged" in joined and "formula" in joined, info["notes"]
    assert got["calc"].isna().all(), "an uncalculated formula reads as missing, and says so"
    print("OK E3b: merged ranges and never-calculated formulas are reported, not silently mangled")


def test_e4_robustness():
    for bad, name, msg in ((b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"0" * 600, "old.xls", "save it as .xlsx or CSV"),
                           (b"not a workbook at all", "t.xlsx", "could not read the Excel workbook"),
                           (xlsx({"S": (pd.DataFrame(), {})}), "t.xlsx", "no data"),
                           (xlsx({"S": (frame(), {})})[:200], "t.xlsx", "could not read")):
        try:
            load_table(bad, name)
        except ValueError as e:
            assert msg in str(e), (name, str(e))
        else:
            raise AssertionError(f"{name}: a bad workbook must be a clear ValueError")
    only_header = xlsx({"S": (pd.DataFrame({"a": [], "b": []}), {})})
    try:
        load_table(only_header, "t.xlsx")
    except ValueError as e:
        assert "no data" in str(e)
    else:
        raise AssertionError("a header-only sheet holds no data")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("hello.txt", "hi")
    try:
        load_table(buf.getvalue(), "t.csv")
    except Exception as e:                                  # a plain ZIP is not a workbook: falls through to the CSV reader
        assert not isinstance(e, (KeyError, AttributeError)), type(e)
    print("OK E4: .xls, garbage, empty, header-only and truncated workbooks are clear errors")


def test_e5_end_to_end_matches_csv():
    import backend.main as bm
    df = frame(700, 5).drop(columns=["when", "ok"])
    df.loc[::17, "score"] = np.nan
    csv_ds = bm.store_dataset(*load_table(df.to_csv(index=False).encode(), "d.csv"), "d.csv", raw=df.to_csv(index=False).encode())
    raw = xlsx({"Other": (frame(20, 9), {}), "Data": (df, {})})
    xl_df, xl_info = load_table(raw, "d.xlsx", sheet="Data")
    xl_ds = bm.store_dataset(xl_df, xl_info, "d.xlsx", raw=raw)
    other = bm.pick_sheet(xl_ds, bm.SheetRequest(sheet="Other"))
    assert other["input_format"]["sheet"] == "Other" and other["rows"] == 20 and other["id"] != xl_ds
    from taskclean import apply_repairs, audit_dataset
    r_csv = apply_repairs(audit_dataset(bm.DATASETS[csv_ds]["df"], "label", seed=3))
    r_xl = apply_repairs(audit_dataset(bm.DATASETS[xl_ds]["df"], "label", seed=3))
    assert r_csv.quality_report.drop(columns=["note"]).equals(r_xl.quality_report.drop(columns=["note"]))
    assert r_csv.summary == r_xl.summary and r_csv.cleaned.equals(r_xl.cleaned)
    try:
        bm.pick_sheet(csv_ds, bm.SheetRequest(sheet="Data"))
    except Exception as e:
        assert getattr(e, "status_code", None) == 400
    else:
        raise AssertionError("a non-Excel dataset has no sheets to pick")
    print("OK E5: the same data as .xlsx and as CSV audits identically; another sheet becomes its own dataset")


if __name__ == "__main__":
    tests = [v for n, v in sorted(globals().items()) if n.startswith("test_") and callable(v)]
    for t in tests:
        t()
    print(f"\nALL PHASE 22 EXCEL-INTAKE INVARIANTS HOLD ({len(tests)} tests)")
