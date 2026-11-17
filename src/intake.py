"""
File intake: CSV (and delimited text), JSON, JSON Lines and Parquet in, one pandas frame out.

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

import pandas as pd

from taskclean import load_csv

SUPPORTED = (".csv", ".tsv", ".txt", ".json", ".jsonl", ".ndjson", ".parquet", ".pq")
PARQUET_MAGIC = b"PAR1"


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


def load_table(source, filename: str | None = None) -> tuple[pd.DataFrame, dict]:
    """Read bytes (or a path) of a CSV, JSON, JSON Lines or Parquet file. Returns (frame, info)."""
    if isinstance(source, (str, os.PathLike)):
        filename = filename or str(source)
        with open(source, "rb") as f:
            raw = f.read()
    else:
        raw = bytes(source)
    if not raw.strip():
        raise ValueError("the file is empty")
    ext = os.path.splitext((filename or "").lower())[1]

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
