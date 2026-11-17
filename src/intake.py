"""
File intake: CSV (and delimited text), Excel (.xlsx/.xlsm), JSON, JSON Lines and Parquet in, one pandas frame out.

`load_csv` (taskclean.py) stays the CSV reader (encoding, delimiter and decimal-comma detection). This module only
decides which reader a file needs, reads JSON and Parquet, and returns the same `(frame, info)` pair so everything
downstream (health snapshot, audit, repairs) is format-blind. `info["format"]` records what the file was.

JSON shapes accepted: a list of records `[{"a": 1}, ...]`; an object wrapping such a list (`{"data": [...]}`, any single
key); an object of equal-length columns `{"a": [1, 2], "b": [3, 4]}`; and JSON Lines (one record per line). Nested
objects are flattened to `parent.child` columns; a list inside a cell is kept as its JSON text so no information is lost.
"""
from __future__ import annotations

import io
import json
import os
import zipfile

import pandas as pd

from taskclean import load_csv

SUPPORTED = (".csv", ".tsv", ".txt", ".xlsx", ".xlsm", ".json", ".jsonl", ".ndjson", ".parquet", ".pq")
PARQUET_MAGIC = b"PAR1"
OLE_MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"          # legacy .xls (and other old Office files)
EXCEL_EXTENSIONS = (".xlsx", ".xlsm")
LEGACY_EXCEL_MESSAGE = "legacy .xls workbooks are not supported: open the file in Excel and save it as .xlsx or CSV"
EXTRAS_MAX_BYTES = 20_000_000                          # the merged-cell / formula notes are skipped above this size


def _is_xlsx(raw: bytes) -> bool:
    if raw[:4] != b"PK\x03\x04":
        return False
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            return "xl/workbook.xml" in z.namelist()
    except zipfile.BadZipFile:
        return False


def _unique_headers(values) -> list[str]:
    out, seen = [], {}
    for i, v in enumerate(values):
        name = "" if v is None or (isinstance(v, float) and pd.isna(v)) else str(v).strip()
        if not name:
            name = f"Unnamed: {i}"
        n = seen.get(name, 0)
        seen[name] = n + 1
        out.append(name if n == 0 else f"{name}.{n}")
    return out


def _excel_notes(raw: bytes, sheet: str) -> list[str]:
    """Merged ranges and formulas Excel never calculated, read from the same sheet (skipped for very large files)."""
    if len(raw) > EXTRAS_MAX_BYTES:
        return []
    try:
        import openpyxl
        wf = openpyxl.load_workbook(io.BytesIO(raw), read_only=False, data_only=False)[sheet]
        wv = openpyxl.load_workbook(io.BytesIO(raw), read_only=False, data_only=True)[sheet]
        notes = []
        if wf.merged_cells.ranges:
            notes.append(f"{len(wf.merged_cells.ranges)} merged cell range(s): only the top-left cell of each holds a value, "
                         "the rest read as missing")
        uncached = sum(1 for row_f, row_v in zip(wf.iter_rows(), wv.iter_rows()) for cf, cv in zip(row_f, row_v)
                       if isinstance(cf.value, str) and cf.value.startswith("=") and cv.value is None)
        if uncached:
            notes.append(f"{uncached:,} formula cell(s) have no stored result (the workbook was never calculated in Excel) "
                         "and read as missing")
        return notes
    except Exception:                                  # noqa: BLE001  extras are best effort, the data is already read
        return []


def _excel_frame(raw: bytes, sheet: str | None) -> tuple[pd.DataFrame, dict]:
    try:
        xl = pd.ExcelFile(io.BytesIO(raw), engine="openpyxl")
    except Exception as e:                             # noqa: BLE001  a bad workbook is user input
        raise ValueError(f"could not read the Excel workbook (it may be corrupt or password-protected): {e}") from e
    names = [str(n) for n in xl.sheet_names]
    if sheet is not None and sheet not in names:
        raise ValueError(f"the workbook has no sheet named '{sheet}' (sheets: {', '.join(names)})")
    for name in ([sheet] if sheet is not None else names):
        grid = xl.parse(name, header=None, dtype=object)
        n_blank_rows = int(grid.isna().all(axis=1).sum()) if len(grid) else 0
        grid = grid.dropna(how="all").dropna(axis=1, how="all")
        if len(grid) >= 2 or (sheet is not None and len(grid) >= 1):
            break
    else:
        raise ValueError("the workbook holds no data (every sheet is empty or has only a header row)")
    if len(grid) < 2:
        raise ValueError(f"sheet '{name}' has no data rows")
    header = _unique_headers(grid.iloc[0].tolist())
    body = grid.iloc[1:].copy()
    body.columns = header
    df = body.reset_index(drop=True).infer_objects()
    notes = []
    if len(names) > 1:
        notes.append(f"the workbook has {len(names)} sheets ({', '.join(names)}); '{name}' was read")
    if n_blank_rows:
        notes.append(f"{n_blank_rows} empty row(s) were skipped")
    notes += _excel_notes(raw, name)
    return df, {"format": "xlsx", "sheet": name, "sheets": names, "rows": len(df), "columns": df.shape[1], "notes": notes}


def _stringify_containers(df: pd.DataFrame) -> pd.DataFrame:
    """A list or dict left inside a cell is stored as JSON text (hashable, comparable, and nothing is dropped)."""
    for c in df.columns:
        if df[c].dtype == object and df[c].map(lambda v: isinstance(v, (list, dict))).any():
            df[c] = df[c].map(lambda v: json.dumps(v, sort_keys=True) if isinstance(v, (list, dict)) else v)
    return df


def _json_frame(text: str) -> tuple[pd.DataFrame, str]:
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        try:
            rows = [json.loads(line) for line in text.splitlines() if line.strip()]  # JSON Lines
        except json.JSONDecodeError as e:
            raise ValueError(f"the file is not valid JSON or JSON Lines ({e.msg}, line {e.lineno})") from e
        if not rows or not all(isinstance(r, dict) for r in rows):
            raise ValueError("the file is not valid JSON or JSON Lines of records")
        return pd.json_normalize(rows, sep="."), "jsonl"
    if isinstance(obj, dict):
        lists = [v for v in obj.values() if isinstance(v, list)]
        if len(obj) == 1 and lists and all(isinstance(r, dict) for r in lists[0]):
            obj = lists[0]                                                           # {"data": [records]}
        elif obj and all(isinstance(v, list) for v in obj.values()) and len({len(v) for v in obj.values()}) == 1:
            return pd.DataFrame(obj), "json"                                         # {"col": [values]}
        else:
            obj = [obj]                                                              # a single record
    if isinstance(obj, list) and obj and all(isinstance(r, dict) for r in obj):
        return pd.json_normalize(obj, sep="."), "json"
    raise ValueError("JSON must be a list of records, an object of equal-length columns, or JSON Lines")


def load_table(source, filename: str | None = None, sheet: str | None = None) -> tuple[pd.DataFrame, dict]:
    """Read bytes (or a path) of a CSV, Excel, JSON, JSON Lines or Parquet file. Returns (frame, info). `sheet` picks an Excel sheet."""
    if isinstance(source, (str, os.PathLike)):
        filename = filename or str(source)
        with open(source, "rb") as f:
            raw = f.read()
    else:
        raw = bytes(source)
    if not raw.strip():
        raise ValueError("the file is empty")
    ext = os.path.splitext((filename or "").lower())[1]

    if ext == ".xls" or (raw[:8] == OLE_MAGIC and ext not in EXCEL_EXTENSIONS):
        raise ValueError(LEGACY_EXCEL_MESSAGE)
    if ext in EXCEL_EXTENSIONS or _is_xlsx(raw):
        return _excel_frame(raw, sheet)

    if ext in (".parquet", ".pq") or raw[:4] == PARQUET_MAGIC:
        try:
            df = pd.read_parquet(io.BytesIO(raw))
        except Exception as e:                  # noqa: BLE001  (a corrupt file is user input)
            raise ValueError(f"could not read the Parquet file: {e}") from e
        notes = []
        if not isinstance(df.index, pd.RangeIndex):
            notes.append("the file's own index was dropped; rows are numbered from 0")
        df = _stringify_containers(df.reset_index(drop=True))
        df.columns = [str(c).strip() for c in df.columns]
        return df, {"format": "parquet", "rows": len(df), "columns": df.shape[1], "notes": notes}

    head = raw.lstrip()[:1]
    if ext in (".json", ".jsonl", ".ndjson") or (ext not in SUPPORTED and head in (b"[", b"{")):
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            text = raw.decode("latin-1")
        df, kind = _json_frame(text)
        df = _stringify_containers(df.reset_index(drop=True))
        df.columns = [str(c).strip() for c in df.columns]
        if df.empty:
            raise ValueError("the JSON file holds no records")
        return df, {"format": kind, "rows": len(df), "columns": df.shape[1], "notes": []}

    df, info = load_csv(raw)
    return df, {"format": "csv", **info}
