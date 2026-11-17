"""
Baseline health report (Phase 18, docs/PHASE18_SPEC.md): a diagnostic snapshot of the RAW uploaded file, taken before any
cleaning, repair or target choice. Definitions are fixed in the spec; this module only implements them.

It reuses the audit's own preparation and detectors (placeholder masking, numbers-stored-as-text, exact duplicates, IQR
outliers, inconsistent values) so the snapshot's counts equal what the audit later reports; it adds the checks the audit
does not have (near-duplicates, |Z| outliers, skew, constant and identifier columns, formatted numbers, volume vitals) as
REPORT-ONLY. It never modifies its input and reports no score.
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from detect import detect_duplicates, detect_missing, detect_outliers_iqr
from inconsistency import detect_inconsistency_v2
from taskclean import (MAX_CAT_LEVELS, MAX_CLASSES, PLACEHOLDER_TOKENS, _looks_numeric, _mask_placeholders, _prepare)

SNAPSHOT_VERSION = 1
Z_LIMIT = 3.0                  # spec: |Z| > 3, population std
SKEW_LIMIT = 1.5               # spec: |skew| > 1.5
INCONSISTENCY_RULE_LABELS = {
    "format_variant": "same value, different case, spacing or punctuation (certain)",
    "abbreviation": "rare short form of a longer value, e.g. Fem for Female (likely: verify it is not a real category)",
    "ambiguous_abbreviation": "short form that could mean more than one value (flagged, not repaired)",
    "possible_typo": "one character away from a common value (flagged, not repaired)",
}
SKEW_MIN_VALUES = 8            # fewer non-missing values than this: skew is not reported
ID_MIN_VALUES = 20             # identifier rule needs at least this many non-missing values
RARE_CLASS_ROWS = 20           # target block: classes with fewer rows than this are called out
EXAMPLES = 4

# currency / thousands / percent / accounting-bracket formatting that pandas will not parse as a number
_NUM = r"\d+(?:\.\d+)?"
_FORMATTED = re.compile(
    r"^\(?\s*[-+]?\s*[$€£¥₹]\s*[-+]?\s*" + _NUM + r"\s*\)?$"                       # $12.50, (€ 3)
    r"|^\(?\s*[-+]?\s*[$€£¥₹]?\s*[-+]?\s*\d{1,3}(?:,\d{3})+(?:\.\d+)?\s*%?\s*\)?$"   # 1,200  $1,200.50
    r"|^[-+]?\s*" + _NUM + r"\s*%$")                                                 # 12%


def _py(v):
    """numpy / pandas scalar -> plain Python (NaN and inf -> None) so the payload is JSON-safe."""
    if v is None:
        return None
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if isinstance(v, (np.integer, int)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        f = float(v)
        return f if np.isfinite(f) else None
    if isinstance(v, pd.Timestamp):
        return v.isoformat()
    return v


def _is_formatted_number_column(masked: pd.Series) -> bool:
    """Text column whose every non-missing value is a plain or formatted number and at least one is formatted."""
    nn = masked.dropna()
    if nn.empty or not nn.map(lambda v: isinstance(v, str)).all():
        return False
    s = nn.str.strip()
    formatted = s.map(lambda v: bool(_FORMATTED.match(v)))
    plain = pd.to_numeric(s, errors="coerce").notna()
    return bool(formatted.any() and (formatted | plain).all())


def _normalise_text(X: pd.DataFrame) -> pd.DataFrame:
    """Near-duplicate view: text cells trimmed, inner whitespace collapsed, lower-cased; everything else untouched."""
    out = {}
    for c in X.columns:
        s = X[c]
        if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
            out[c] = s
        else:
            out[c] = s.astype(object).map(
                lambda v: re.sub(r"\s+", " ", v.strip()).lower() if isinstance(v, str) else v)
    return pd.DataFrame(out, index=X.index)


def _column_kind(raw: pd.Series, masked: pd.Series | None, in_audit_as_numeric: bool, n_unique: int) -> str:
    if pd.api.types.is_bool_dtype(raw):
        return "boolean"
    if pd.api.types.is_datetime64_any_dtype(raw) or pd.api.types.is_timedelta64_dtype(raw):
        return "datetime"
    if pd.api.types.is_numeric_dtype(raw):
        return "numeric"
    if masked is not None:
        if in_audit_as_numeric:
            return "numeric stored as text"
        if _is_formatted_number_column(masked):
            return "formatted number stored as text"
    return "categorical" if n_unique <= MAX_CAT_LEVELS else "text / high-cardinality"


def build_snapshot(df: pd.DataFrame, info: dict | None = None, file_name: str = "", raw_bytes: bytes | None = None) -> dict:
    """The baseline health report for one raw dataframe. Pure: `df` is not modified, and the result contains no object
    that refers back to it."""
    df = df.reset_index(drop=True)                      # same 0-based row ids as the audit uses
    n_rows, n_cols = df.shape
    notes: list[str] = []

    # --- the audit's own preparation, with no target (every column is a feature here) -----------------------------
    try:
        X, _, integer_like, excluded, prep_notes, _ = _prepare(df, None)
        notes += prep_notes
    except ValueError as e:                            # e.g. no usable feature column: report the structure anyway
        X, integer_like, excluded = pd.DataFrame(index=df.index), {}, {}
        notes.append(f"the problem checks could not run: {e}")

    # --- per-column profile ----------------------------------------------------------------------------------------
    masked_cols, placeholders = {}, {}
    for c in df.columns:
        s = df[c]
        if not (pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s)
                or pd.api.types.is_datetime64_any_dtype(s) or pd.api.types.is_timedelta64_dtype(s)):
            masked_cols[c], placeholders[c] = _mask_placeholders(s)
        else:
            placeholders[c] = 0

    columns, counts = [], {}
    constant, id_like, skewed = [], [], []
    for c in df.columns:
        raw = df[c]
        masked = masked_cols.get(c)
        analysed = c in X.columns
        in_audit_numeric = analysed and pd.api.types.is_numeric_dtype(X[c])
        eff = X[c] if analysed else (masked if masked is not None else raw)
        nn = eff.dropna()
        n_unique = int(nn.nunique())
        kind = _column_kind(raw, masked, in_audit_numeric and not pd.api.types.is_numeric_dtype(raw), n_unique)
        counts[kind] = counts.get(kind, 0) + 1
        blanks = int(raw.isna().sum())
        ph = int(placeholders.get(c, 0))
        flags = []
        entry = {"name": str(c), "kind": kind, "n_unique": n_unique, "missing": blanks + ph, "blank_cells": blanks,
                 "placeholder_cells": ph, "missing_pct": (blanks + ph) / n_rows if n_rows else 0.0,
                 "examples": [str(v) for v in raw.dropna().unique()[:EXAMPLES]]}
        if c in excluded:
            flags.append(f"not analysed by the problem checks: {excluded[c]}")
        if len(nn) and n_unique <= 1 and c not in excluded:
            flags.append("constant: carries no information")
            constant.append(str(c))
        if (len(nn) >= ID_MIN_VALUES and n_unique == len(nn)
                and (not pd.api.types.is_numeric_dtype(eff) or integer_like.get(c, False))):
            flags.append("possible identifier: every value is distinct")
            id_like.append(str(c))
        if kind == "formatted number stored as text":
            flags.append("numbers with currency, thousands or percent formatting: the audit reads this as categorical")
        if in_audit_numeric and len(nn):
            vals = nn.astype(float)
            # skew is not meaningful for a two-valued column (a 0/1 flag is "skewed" by definition of its balance)
            sk = float(vals.skew()) if len(vals) >= SKEW_MIN_VALUES and vals.nunique() > 2 else None
            entry.update(min=_py(vals.min()), max=_py(vals.max()), mean=_py(vals.mean()), median=_py(vals.median()),
                         std=_py(vals.std()), skew=_py(sk))
            if sk is not None and abs(sk) > SKEW_LIMIT:
                flags.append(f"heavily skewed (skew {sk:+.1f}): distorts linear and neural models, not tree models")
                skewed.append({"column": str(c), "skew": sk})
        elif kind in ("categorical", "text / high-cardinality") and len(nn):
            top = nn.astype(str).value_counts().head(3)
            entry["top_values"] = [{"value": str(k), "share": float(v / len(nn))} for k, v in top.items()]
        entry["flags"] = flags
        columns.append(entry)

    # --- completeness ----------------------------------------------------------------------------------------------
    total_cells = n_rows * n_cols
    blank_total = int(df.isna().sum().sum())
    ph_total = int(sum(placeholders.values()))
    cell_missing = pd.DataFrame({c: (df[c].isna() | (masked_cols[c].isna() if c in masked_cols else False)) for c in df.columns})
    rows_missing = int(cell_missing.any(axis=1).sum()) if n_cols else 0
    all_missing = [str(c) for c in df.columns if cell_missing[c].all()] if n_rows else []
    completeness = {
        "total_cells": total_cells, "missing_cells": blank_total + ph_total, "blank_cells": blank_total,
        "placeholder_cells": ph_total, "missing_pct": (blank_total + ph_total) / total_cells if total_cells else 0.0,
        "rows_with_missing": rows_missing, "rows_with_missing_pct": rows_missing / n_rows if n_rows else 0.0,
        "columns_entirely_missing": all_missing,
        "placeholder_tokens": sorted(t if t else "(empty text)" for t in PLACEHOLDER_TOKENS)}

    # --- duplicates ------------------------------------------------------------------------------------------------
    dup = near = None
    if X.shape[1]:
        exact = detect_duplicates(X).row_mask
        norm = _normalise_text(X).duplicated(keep="first")
        extra = norm & ~exact
        dup = {"exact_rows": int(exact.sum()), "exact_pct": float(exact.mean()) if n_rows else 0.0,
               "exact_examples": [int(i) for i in exact.index[exact][:EXAMPLES]],
               "near_extra_rows": int(extra.sum()), "near_extra_pct": float(extra.mean()) if n_rows else 0.0,
               "near_examples": [int(i) for i in extra.index[extra][:EXAMPLES]],
               "near_definition": "rows that become duplicates after trimming, collapsing inner spaces and lower-casing "
                                  "text, beyond the exact duplicates (report-only, not validated)"}

    # --- anomalies -------------------------------------------------------------------------------------------------
    anomalies = {"iqr": None, "zscore": None, "skewed_columns": skewed, "constant_columns": constant,
                 "possible_identifier_columns": id_like}
    num = X.select_dtypes("number")
    if num.shape[1]:
        iqr = detect_outliers_iqr(X)
        per_col = {str(c): int(f.sum()) for c, f in iqr.extra["per_column_flags"].items() if int(f.sum())}
        anomalies["iqr"] = {"outlier_rows": int(iqr.row_mask.sum()), "outlier_pct": float(iqr.row_mask.mean()),
                            "per_column": per_col, "percentile_rule_columns": [str(c) for c in iqr.extra["degenerate_columns"]],
                            "rule": "1.5 x IQR fences (existing audit detector; zero-inflated columns use a percentile rule)"}
        mu, sd = num.mean(), num.std(ddof=0)
        z = (num - mu) / sd.where(sd > 0)
        zflag = z.abs() > Z_LIMIT
        zrows = zflag.any(axis=1)
        anomalies["zscore"] = {"outlier_rows": int(zrows.sum()), "outlier_pct": float(zrows.mean()),
                               "per_column": {str(c): int(zflag[c].sum()) for c in num.columns if int(zflag[c].sum())},
                               "rule": f"|Z| > {Z_LIMIT:g} (report-only)"}

    # --- inconsistent spellings (existing detector) ---------------------------------------------------------------
    incons = None
    if X.shape[1]:
        r = detect_inconsistency_v2(X)
        cells = r.extra["cells"]
        by_col = []
        for col, g in cells.groupby("column") if len(cells) else []:
            by_col.append({"column": str(col), "cells": int(len(g)),
                           "by_rule": {str(k): int(v) for k, v in g["rule"].value_counts().items()},
                           "examples": [{"value": str(a), "canonical": None if b is None else str(b), "rule": str(r)}
                                        for a, b, r in zip(g["original_value"].head(EXAMPLES), g["proposed_value"].head(EXAMPLES),
                                                           g["rule"].head(EXAMPLES))]})
        by_rule = {str(k): int(v) for k, v in cells["rule"].value_counts().items()} if len(cells) else {}
        incons = {"rows": int(r.row_mask.sum()), "rows_pct": float(r.rate), "cells": int(len(cells)), "by_column": by_col,
                  "by_rule": by_rule, "rule_labels": INCONSISTENCY_RULE_LABELS,
                  "skipped_columns": {str(k): v for k, v in r.extra["skipped_columns"].items()}}

    file_bytes = len(raw_bytes) if raw_bytes is not None else None
    return {
        "version": SNAPSHOT_VERSION,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "overview": {"file_name": file_name, "file_bytes": file_bytes,
                     "file_mb": None if file_bytes is None else file_bytes / 1e6,
                     "sha256": None if raw_bytes is None else hashlib.sha256(raw_bytes).hexdigest(),
                     "rows": int(n_rows), "columns": int(n_cols),
                     "memory_mb": float(df.memory_usage(deep=True).sum() / 1e6),
                     "input_format": dict(info or {}), "type_counts": counts},
        "columns": columns, "completeness": completeness, "duplicates": dup, "anomalies": anomalies,
        "inconsistencies": incons, "notes": notes,
        "not_covered": [
            "Wrong labels and anomalous feature values need a target column and a trained model: the full audit checks them.",
            "How much each problem would hurt a model is estimated on the Impact page, not here.",
            "No single health score is given: the project's own test found an aggregate of detected rates unsupported."],
    }


def target_profile(df: pd.DataFrame, target: str) -> dict:
    """The target-dependent block, computed on request once a target is chosen (not part of the stored snapshot)."""
    if target not in df.columns:
        raise ValueError(f"target column '{target}' not found in the dataset")
    y = df[target].astype(object).map(lambda v: np.nan if pd.isna(v) else str(v))
    n = len(y)
    labelled = y.dropna()
    vc = labelled.value_counts()
    k = int(len(vc))
    out = {"target": str(target), "rows": int(n), "rows_without_target": int(n - len(labelled)),
           "rows_without_target_pct": float((n - len(labelled)) / n) if n else 0.0, "n_classes": k,
           "classes": [{"value": str(v), "rows": int(c), "share": float(c / len(labelled))} for v, c in vc.items()][:MAX_CLASSES + 1],
           "usable": True, "warnings": []}
    if k < 2:
        out["usable"] = False
        out["warnings"].append(f"fewer than 2 distinct values: nothing to classify")
    elif k > MAX_CLASSES:
        out["usable"] = False
        out["warnings"].append(f"{k} distinct values: TaskClean supports classification only (<= {MAX_CLASSES} classes); "
                               "this looks like a regression target or an identifier")
    if k >= 2:
        out["imbalance_ratio"] = float(vc.iloc[0] / vc.iloc[-1])
        out["minority_share"] = float(vc.iloc[-1] / len(labelled))
        rare = [str(v) for v, c in vc.items() if c < RARE_CLASS_ROWS]
        out["rare_classes"] = rare
        if rare:
            out["warnings"].append(f"{len(rare)} class(es) have fewer than {RARE_CLASS_ROWS} rows: {', '.join(rare[:5])}")
    return out


def snapshot_rows(s: dict) -> pd.DataFrame:
    """The snapshot as one flat table (section, item, metric, value) for the CSV download. Nothing is added or rounded:
    every value comes from the snapshot dictionary."""
    rows = []

    def add(section, item, metric, value):
        rows.append({"section": section, "item": item, "metric": metric, "value": value})

    ov = s["overview"]
    for k in ("file_name", "file_bytes", "file_mb", "sha256", "rows", "columns", "memory_mb"):
        add("overview", "", k, ov[k])
    for k, v in (ov["input_format"] or {}).items():
        add("overview", "input format", k, v)
    add("overview", "", "captured_at_utc", s["captured_at_utc"])
    for kind, n in ov["type_counts"].items():
        add("column types", kind, "columns", n)
    for k, v in s["completeness"].items():
        if not isinstance(v, (list, dict)):
            add("completeness", "", k, v)
    add("completeness", "", "columns_entirely_missing", "; ".join(s["completeness"]["columns_entirely_missing"]))
    for c in s["columns"]:
        for k in ("kind", "n_unique", "missing", "blank_cells", "placeholder_cells", "missing_pct",
                  "min", "max", "mean", "median", "std", "skew"):
            if k in c:
                add("columns", c["name"], k, c[k])
        add("columns", c["name"], "flags", "; ".join(c["flags"]))
    if s["duplicates"]:
        for k, v in s["duplicates"].items():
            add("duplicates", "", k, "; ".join(map(str, v)) if isinstance(v, list) else v)
    a = s["anomalies"]
    for name in ("iqr", "zscore"):
        if a[name]:
            for k in ("outlier_rows", "outlier_pct", "rule"):
                add(f"outliers ({name})", "", k, a[name][k])
            for col, n in a[name]["per_column"].items():
                add(f"outliers ({name})", col, "flagged_cells", n)
    for x in a["skewed_columns"]:
        add("skew", x["column"], "skew", x["skew"])
    for col in a["constant_columns"]:
        add("constant columns", col, "flagged", True)
    for col in a["possible_identifier_columns"]:
        add("possible identifier columns", col, "flagged", True)
    if s["inconsistencies"]:
        add("inconsistent spellings", "", "rows", s["inconsistencies"]["rows"])
        add("inconsistent spellings", "", "cells", s["inconsistencies"]["cells"])
        for rule, n in s["inconsistencies"].get("by_rule", {}).items():
            add("inconsistent spellings", "", f"cells by rule: {rule}", n)
        for b in s["inconsistencies"]["by_column"]:
            add("inconsistent spellings", b["column"], "cells", b["cells"])
    for n in s["notes"]:
        add("notes", "", "note", n)
    for n in s["not_covered"]:
        add("not covered", "", "limit", n)
    return pd.DataFrame(rows)
