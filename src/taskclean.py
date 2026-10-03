"""
TaskClean product layer.

    Upload CSV -> pick target -> audit -> task-specific impact ->
    repair policy -> (safe auto-repair | human review) -> cleaned dataset +
    repair log + reports

This module adds NO new detectors, metrics, or experiments. Every
detection routine, threshold, evidence number and wording comes from the
research phases (detect.py, phase9_repair_engine.py, phase10 evidence table,
phase11_readiness.py). What it adds is plumbing: handling arbitrary CSVs,
turning detector output into an auditable repair log, and applying the
Phase 9/10 evidence as an explicit apply-or-review policy.

Two stages, so a UI can change the repair plan without paying for the
(expensive, model-based) audit again:

    state  = audit_dataset(df, target)          # detect + propose
    result = apply_repairs(state, apply_issues) # cheap; writes nothing

Design rules inherited from the research (see README):
  * A detector flag is not a confirmed error, and "detected" != "safe to
    repair". Only repairs with zero observed harm and high repair precision
    in the controlled benchmark are applied automatically; everything else
    is logged as a proposal and left untouched.
  * The repair log records every change (and every non-change), so the
    system's behaviour is provable after the fact.
"""

from __future__ import annotations

import csv
import io
import re
import json
import os
import platform
import time
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable

import numpy as np
import pandas as pd
import sklearn
from sklearn.impute import KNNImputer

from detect import (DetectionResult, detect_duplicates, detect_feature_corruption_crossfeature,
                    detect_inconsistency, detect_label_noise, detect_missing, detect_outliers_iqr)
from phase9_repair_engine import (AUTO_REPAIR_FEATURE_CAT_THRESHOLD, AUTO_REPAIR_FEATURE_NUM_Z,
                                   AUTO_REPAIR_LABEL_CONFIDENCE, AUTO_REPAIR_OUTLIER_FALLBACK_Q,
                                   AUTO_REPAIR_OUTLIER_K)
from phase11_readiness import (EVIDENCE_NOTES, EVIDENCE_STATUS, LABELS, RECOMMENDATION_TEXT, RESULTS_DIR,
                                _estimate_impact, _load_baseline_rates, _status)

TASKCLEAN_VERSION = "1.0"

ISSUES = ["missing_values", "duplicates", "outliers", "label_errors", "feature_corruption"]
APPLY_ORDER = ["missing_values", "outliers", "feature_corruption", "label_errors", "duplicates"]

PROBLEM_TEXT = {
    "missing_values": "Missing value",
    "duplicates": "Duplicate row",
    "outliers": "Potential outlier",
    "label_errors": "Potential label error",
    "feature_corruption": "Potential feature anomaly",
}

# --- policy / safety parameters (documented, not hidden) --------------------
# A repair type is applied AUTOMATICALLY only if, in the Phase 9 controlled
# benchmark at the nearest evaluated corruption rate, (a) no seed ever showed
# net harm (risk == 0), (b) its repair precision was >= this value (i.e. it
# rarely overwrote legitimate data), and (c) its repairability status was not
# "negative". Everything else is flagged for human review.
SAFE_REPAIR_MIN_PRECISION = 0.95
# Categorical columns with more distinct values than this are excluded from
# the model-based detectors only (one-hot encoding an ID-like column would
# blow up memory and carries no signal). They are still checked for missing
# values, duplicates and inconsistency.
MAX_CAT_LEVELS = 50
MAX_CLASSES = 20            # classification only; more distinct targets looks like regression/IDs
KNN_MAX_ROWS = 40_000       # above this, numeric imputation falls back to the median (KNN is O(n^2))
# Never applied automatically, whatever the measured evidence says: label repair rewrites the target, and the
# benchmark injects RANDOM noise whereas real label errors are usually structured, so a measured precision
# would be optimistic; feature-anomaly repair was net-harmful.
NEVER_AUTO_APPLY = {"label_errors", "feature_corruption"}
# Above this many rows the slow, cross-validated detectors (labels, feature anomalies) run on a stratified
# random sample instead of every row, and repairs/flags are produced only for the sampled rows.
MAX_MODEL_ROWS = 30_000
# Cell contents treated as "missing" when analysing text columns, in addition to real NaN. The original value
# is never rewritten unless a missing-value repair is applied, and it is shown as-is in the repair log.
PLACEHOLDER_TOKENS = {"", "?", "-", "--", "n/a", "na", "n.a.", "#n/a", "null", "nan"}

Progress = Callable[[float, str], None]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class AuditState:
    df: pd.DataFrame                         # the uploaded frame (0-based RangeIndex = row_id)
    target: str | None                       # None = no-target mode (label checks are skipped)
    X: pd.DataFrame                          # analysis features (category / float64), rows with a target
    y: pd.Series | None                      # target as strings (None in no-target mode)
    integer_like: dict[str, bool]
    excluded_columns: dict[str, str]         # not analysed at all
    unmodeled_columns: dict[str, str]        # skipped by model-based detectors only
    detections: dict[str, DetectionResult]   # keyed missing_values/duplicates/outliers/inconsistency/label_noise/feature_corruption
    skipped: dict[str, str]                  # detector -> reason it did not run
    proposals: pd.DataFrame                  # every candidate repair / flag
    policy: pd.DataFrame                     # one row per issue type
    meta: dict = field(default_factory=dict)
    decimal_comma_columns: set = field(default_factory=set)
    selfbench: object | None = None          # dataset-specific evidence (see selfbench.py), if calibrated


@dataclass
class TaskCleanResult:
    cleaned: pd.DataFrame                    # evidence-based (or user-selected) repairs only
    repair_log: pd.DataFrame
    quality_report: pd.DataFrame
    impact_report: pd.DataFrame
    policy: pd.DataFrame
    readiness: dict
    summary: dict
    # The "aggressive" variant: every proposed repair EXCEPT feature-anomaly repairs (demonstrably harmful in the
    # benchmark). It is a model-ready file for people who want everything applied -- with the risks that implies.
    cleaned_aggressive: pd.DataFrame | None = None
    repair_log_aggressive: pd.DataFrame | None = None
    summary_aggressive: dict | None = None
    selfbench: object | None = None


# ---------------------------------------------------------------------------
# Preparation
# ---------------------------------------------------------------------------
def _to_category(s: pd.Series) -> pd.Series:
    """Strings with real NaN for missing, as a pandas category (the dtype the
    detectors were developed on; also sidesteps pandas 3's `str` dtype)."""
    out = s.astype(object).map(lambda v: np.nan if pd.isna(v) else str(v))
    return out.astype("category")


def _mask_placeholders(s: pd.Series) -> tuple[pd.Series, int]:
    """Text cells whose stripped, lower-cased content is a placeholder token ('?', 'N/A', '', ...) become NaN
    *in the analysis copy only*. Returns the masked series and how many cells were masked (not counting
    cells that were already NaN)."""
    obj = s.astype(object)
    is_ph = obj.map(lambda v: isinstance(v, str) and v.strip().lower() in PLACEHOLDER_TOKENS)
    return obj.where(~is_ph, np.nan), int((is_ph & s.notna()).sum())


_COMMA_DECIMAL = re.compile(r"^\s*-?\d+(,\d+)?\s*$")


def _looks_numeric(masked: pd.Series):
    """Parse a text column as numbers only if EVERY non-missing value parses (and none looks like an
    identifier with leading zeros, e.g. a ZIP code). Handles decimal commas ('3,5'). Returns
    (numbers, uses_decimal_comma), or (None, False) to keep the column categorical."""
    nn = masked.dropna()
    if nn.empty:
        return None, False
    if nn.map(lambda v: isinstance(v, str) and len(v.strip()) > 1 and v.strip()[0] == "0"
              and not v.strip().startswith("0.") and not v.strip().startswith("0,")).any():
        return None, False
    num = pd.to_numeric(masked, errors="coerce")
    if num.notna().sum() == masked.notna().sum():
        return num, False
    if nn.map(lambda v: isinstance(v, str) and bool(_COMMA_DECIMAL.match(v))).all() and nn.str.contains(",").any():
        num = pd.to_numeric(masked.map(lambda v: v.replace(",", ".") if isinstance(v, str) else v), errors="coerce")
        if num.notna().sum() == masked.notna().sum():
            return num, True
    return None, False


def _prepare(df: pd.DataFrame, target: str | None):
    if target is not None and target not in df.columns:
        raise ValueError(f"target column '{target}' not found in the dataset")

    excluded, feats, integer_like, notes = {}, {}, {}, []
    placeholder_counts, numeric_text, comma_cols = {}, [], set()
    for col in df.columns:
        if col == target:
            continue
        s = df[col]
        if s.isna().all():
            excluded[col] = "entirely missing"
        elif pd.api.types.is_datetime64_any_dtype(s) or pd.api.types.is_timedelta64_dtype(s):
            excluded[col] = "datetime column (not analysed)"
        elif pd.api.types.is_bool_dtype(s):
            feats[col] = _to_category(s)
        elif pd.api.types.is_numeric_dtype(s):
            num = pd.to_numeric(s, errors="coerce").astype(float)
            feats[col] = num
            nn = num.dropna()
            integer_like[col] = bool(len(nn) and (nn % 1 == 0).all())
        else:
            masked, n_ph = _mask_placeholders(s)
            if masked.isna().all():
                excluded[col] = "entirely missing (placeholders only)"
                continue
            if n_ph:
                placeholder_counts[col] = n_ph
            num, uses_comma = _looks_numeric(masked)
            if num is not None:
                feats[col] = num.astype(float)
                nn = feats[col].dropna()
                integer_like[col] = bool(len(nn) and (nn % 1 == 0).all())
                numeric_text.append(col)
                if uses_comma:
                    comma_cols.add(col)
            else:
                feats[col] = _to_category(masked)

    if not feats:
        raise ValueError("no usable feature columns after excluding the target")
    if placeholder_counts:
        top = sorted(placeholder_counts.items(), key=lambda kv: -kv[1])
        shown = ", ".join(f"{c} ({n:,})" for c, n in top[:6]) + (f", and {len(top) - 6} more" if len(top) > 6 else "")
        notes.append(f"{sum(placeholder_counts.values()):,} placeholder cell(s) such as '?', 'N/A' or empty text are "
                     f"treated as missing: {shown}")
    if numeric_text:
        notes.append(f"{len(numeric_text)} column(s) are stored as text but every value is a number, so they are "
                     f"analysed as numeric: {', '.join(numeric_text[:6])}" + (", ..." if len(numeric_text) > 6 else "")
                     + (f" (decimal comma in {len(comma_cols)})" if comma_cols else ""))

    if target is None:
        return pd.DataFrame(feats, index=df.index), None, integer_like, excluded, notes, comma_cols

    # y covers every row (NaN where the target is missing) so the cheap checks can still clean unlabeled rows;
    # only the label / model-based detectors restrict themselves to labeled rows.
    y = df[target].astype(object).map(lambda v: np.nan if pd.isna(v) else str(v))
    n_classes = y.nunique()
    if n_classes < 2:
        raise ValueError(f"target '{target}' has fewer than 2 distinct values -- nothing to classify")
    if n_classes > MAX_CLASSES:
        raise ValueError(
            f"target '{target}' has {n_classes} distinct values; TaskClean supports classification only "
            f"(<= {MAX_CLASSES} classes). This looks like a regression target or an ID column.")

    return pd.DataFrame(feats, index=df.index), y, integer_like, excluded, notes, comma_cols


def _full_record(X: pd.DataFrame, y: pd.Series | None) -> pd.DataFrame:
    """Duplicates are judged on the WHOLE record (features + target). Two
    rows with identical features but different labels are not duplicates --
    dropping one would delete information. (In the benchmark, injected
    duplicates were full-row copies, so this matches what was validated.)
    With no target, the record is just the features."""
    return X.copy() if y is None else pd.concat([X, y.rename("__target__")], axis=1)


def load_csv(source) -> tuple[pd.DataFrame, dict]:
    """Read a CSV from a path, bytes or file-like object, detecting the text encoding, the delimiter
    (, ; tab |) and -- for ';'-delimited files -- whether decimals use a comma. Returns (frame, info)."""
    if isinstance(source, (str, os.PathLike)):
        with open(source, "rb") as f:
            raw = f.read()
    elif isinstance(source, (bytes, bytearray)):
        raw = bytes(source)
    else:
        raw = source.read()
        raw = raw.encode("utf-8") if isinstance(raw, str) else raw
    if not raw.strip():
        raise ValueError("the file is empty")

    for encoding in ("utf-8-sig", "cp1252", "latin-1"):    # latin-1 accepts any byte sequence
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue

    head = text[:65536]
    try:
        delimiter = csv.Sniffer().sniff(head, delimiters=",;\t|").delimiter
    except csv.Error:
        first = head.splitlines()[0] if head.strip() else ""
        delimiter = max(",;\t|", key=first.count)

    def read(sep, decimal="."):
        return pd.read_csv(io.StringIO(text), sep=sep, decimal=decimal, low_memory=False)

    df = read(delimiter)
    if df.shape[1] == 1:                                   # sniffer guessed wrong: take the delimiter that splits most
        candidates = {d: read(d) for d in ",;\t|" if d != delimiter and d in head}
        if candidates:
            best = max(candidates, key=lambda d: candidates[d].shape[1])
            if candidates[best].shape[1] > 1:
                delimiter, df = best, candidates[best]
    decimal = "."
    if delimiter == ";":                                   # European convention: 3,5 means 3.5
        alt = read(delimiter, ",")
        n_numeric = lambda d: sum(pd.api.types.is_numeric_dtype(d[c]) for c in d.columns)
        if n_numeric(alt) > n_numeric(df):
            df, decimal = alt, ","
    df.columns = [str(c).strip() for c in df.columns]
    return df, {"encoding": encoding, "delimiter": {"\t": "tab"}.get(delimiter, delimiter), "decimal": decimal,
                "rows": len(df), "columns": df.shape[1]}


# ---------------------------------------------------------------------------
# Proposal builders (turn detector output into candidate repairs)
# ---------------------------------------------------------------------------
def _records(issue, rows, column, original, proposed, method, tier, score) -> pd.DataFrame:
    n = len(rows)
    return pd.DataFrame({
        "issue_key": issue, "row_id": np.asarray(rows), "column": column,
        "problem": PROBLEM_TEXT[issue],
        "original_value": pd.Series(np.asarray(original, dtype=object) if np.ndim(original) else [original] * n, dtype=object),
        "proposed_value": pd.Series(np.asarray(proposed, dtype=object) if np.ndim(proposed) else [proposed] * n, dtype=object),
        "method": method, "tier": tier,
        "detector_score": np.asarray(score, dtype=float) if np.ndim(score) else np.full(n, score, dtype=float),
    })


def _propose_missing(X, integer_like, cell_mask, original: pd.DataFrame) -> list[pd.DataFrame]:
    frames = []
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    miss_num = [c for c in num_cols if cell_mask[c].any()]
    if miss_num:
        if len(X) <= KNN_MAX_ROWS and len(num_cols) >= 2:
            # Standardise first so KNN distance isn't dominated by large-scale
            # columns (same approach as the Phase 9 evaluation).
            means = X[num_cols].mean()
            stds = X[num_cols].std().replace(0, 1).fillna(1)
            scaled = (X[num_cols] - means) / stds
            imputed = pd.DataFrame(KNNImputer(n_neighbors=5).fit_transform(scaled),
                                   index=X.index, columns=num_cols) * stds + means
            method = "KNN imputation (k=5, standardised numeric columns)"
        else:
            imputed = X[num_cols].fillna(X[num_cols].median())
            method = "median imputation" + (" (KNN skipped: dataset too large)" if len(X) > KNN_MAX_ROWS else "")
        for c in miss_num:
            rows = X.index[cell_mask[c]]
            vals = imputed.loc[rows, c]
            if integer_like.get(c):
                vals = vals.round()
            frames.append(_records("missing_values", rows, c, original.loc[rows, c].values, vals.values,
                                   method, "exact", np.nan))
    for c in [c for c in X.columns if c not in num_cols and cell_mask[c].any()]:
        mode = X[c].mode(dropna=True)
        if len(mode) == 0:
            continue
        rows = X.index[cell_mask[c]]
        frames.append(_records("missing_values", rows, c, original.loc[rows, c].values, str(mode.iloc[0]),
                               "most-frequent value", "exact", np.nan))
    return frames


def _propose_duplicates(full: pd.DataFrame, dup_mask: pd.Series) -> list[pd.DataFrame]:
    rows = full.index[dup_mask]
    if len(rows) == 0:
        return []
    h = pd.Series(pd.util.hash_pandas_object(full, index=False).values, index=full.index)
    first = h[~h.duplicated(keep="first")]
    first_idx = pd.Series(first.index, index=first.values)
    originals = [f"exact copy of row {first_idx.loc[h.loc[r]]}" for r in rows]
    return [_records("duplicates", rows, "", originals, "drop row", "exact full-record duplicate removal",
                     "exact", np.nan)]


def _propose_outliers(X, integer_like, loose: DetectionResult) -> list[pd.DataFrame]:
    strict = detect_outliers_iqr(X, k=AUTO_REPAIR_OUTLIER_K,
                                 fallback_lower_q=AUTO_REPAIR_OUTLIER_FALLBACK_Q[0],
                                 fallback_upper_q=AUTO_REPAIR_OUTLIER_FALLBACK_Q[1])
    lower, upper = loose.extra["bounds"]
    degenerate = set(loose.extra["degenerate_columns"])
    frames = []
    for col, flags in loose.extra["per_column_flags"].items():
        flags = flags.reindex(X.index, fill_value=False)
        sflags = strict.extra["per_column_flags"][col].reindex(X.index, fill_value=False)
        rows_strict = X.index[flags & sflags]
        rows_loose = X.index[flags & ~sflags]
        lo, hi = lower[col], upper[col]
        basis = "0.5-99.5 percentile bounds" if col in degenerate else "1.5xIQR bounds"
        if len(rows_strict):
            orig = X.loc[rows_strict, col]
            new = orig.clip(lo, hi)
            if integer_like.get(col):
                new = new.round()
            frames.append(_records("outliers", rows_strict, col, orig.values, new.values,
                                   f"clip to [{lo:.6g}, {hi:.6g}] ({basis})", "strict", np.nan))
        if len(rows_loose):
            frames.append(_records("outliers", rows_loose, col, X.loc[rows_loose, col].values, np.nan,
                                   f"flag only: outside {basis}, below the high-confidence threshold",
                                   "loose", np.nan))
    return frames


def _propose_labels(y, res: DetectionResult) -> list[pd.DataFrame]:
    rows = res.row_mask[res.row_mask].index
    if len(rows) == 0:
        return []
    conf = res.extra["oof_confidence"].loc[rows]
    pred = res.extra["oof_pred"].loc[rows].astype(object)
    strict = conf >= AUTO_REPAIR_LABEL_CONFIDENCE
    proposed = pred.where(strict)               # NaN where only flagged
    tier = np.where(strict, "strict", "loose")
    method = np.where(strict, "relabel to cross-validated model prediction",
                      "flag only: model disagrees, below the high-confidence threshold")
    out = _records("label_errors", rows, "", y.loc[rows].values, proposed.values, method, tier, conf.values)
    return [out]


def _propose_features(X, integer_like, res: DetectionResult) -> list[pd.DataFrame]:
    frames = []
    for col, info in res.extra["per_column"].items():
        flags = res.extra["per_column_flags"][col] & X[col].notna()   # a missing cell is missing, not anomalous
        if not flags.any():
            continue
        score = info["score"]
        is_cat = info["kind"] == "categorical"
        strict_mask = (score < AUTO_REPAIR_FEATURE_CAT_THRESHOLD) if is_cat else (score > AUTO_REPAIR_FEATURE_NUM_Z)
        rows = X.index[flags]
        strict = strict_mask.loc[rows]
        pred = info["predicted_value"].loc[rows]
        if not is_cat and integer_like.get(col):
            pred = pred.round()
        proposed = pred.astype(object).where(strict)
        tier = np.where(strict, "strict", "loose")
        method = np.where(strict, "replace with cross-feature model prediction",
                          "flag only: disagrees with the other columns, below the high-confidence threshold")
        frames.append(_records("feature_corruption", rows, col, X.loc[rows, col].astype(object).values,
                               proposed.values, method, tier, score.loc[rows].values))
    return frames


# ---------------------------------------------------------------------------
# Policy (Phase 9/10 evidence -> apply or review)
# ---------------------------------------------------------------------------
def _nearest_evidence_row(ev: pd.DataFrame, issue: str, observed_rate: float):
    sub = ev[ev.error_type == issue]
    if sub.empty:
        return None
    return sub.iloc[(sub["rate"] - observed_rate).abs().argsort().iloc[0]]


def _evidence_for(issue: str, observed_rate: float, selfbench=None) -> dict | None:
    """Repair evidence for an issue at the nearest evaluated corruption rate. Uses this dataset's own
    self-benchmark when it measured the issue; otherwise falls back to the UCI Adult benchmark."""
    sources = []
    if selfbench is not None and getattr(selfbench, "ok", False):
        sources.append(("this dataset (self-benchmark)", selfbench.evidence))
    path = os.path.join(RESULTS_DIR, "phase10_evidence_table.csv")
    if os.path.exists(path):
        sources.append(("UCI Adult benchmark" + (" (not measured on this dataset)" if sources else ""),
                        pd.read_csv(path)))
    for source, ev in sources:
        row = _nearest_evidence_row(ev, issue, observed_rate)
        if row is not None:
            # dataset-specific evidence also carries a noise-aware "material" harm risk; use it when present
            material = row.get("material_risk")
            use_material = material is not None and not pd.isna(material)
            return {"rate": float(row["rate"]), "status": row["repairability_status"],
                    "risk": float(material if use_material else row["risk"]),
                    "risk_kind": "harm beyond the noise floor" if use_material else "observed harm",
                    "precision": float(row["repair_precision"]), "recall": float(row["repair_recall"]),
                    "source": source}
    return None


def _impact_curve_estimate(curve: pd.DataFrame, issue: str, rate: float):
    """Same interpolation as the Adult version (a (0,0) anchor; beyond the highest measured rate the value is a
    boundary estimate), over this dataset's own measured damage curve. Returns None if the issue wasn't measured."""
    c = curve[curve.error_type == issue].sort_values("rate")
    if c.empty:
        return None
    rates = np.concatenate([[0.0], c["rate"].values])
    damages = np.concatenate([[0.0], c["mean_damage"].values])
    return float(np.interp(rate, rates, damages)), bool(rate > rates.max())


def _build_policy(detections, proposals, skipped, selfbench=None) -> pd.DataFrame:
    rows = []
    for issue in ISSUES:
        det_key = "label_noise" if issue == "label_errors" else issue
        det = detections.get(det_key)
        sub = proposals[proposals.issue_key == issue]
        n_auto = int(sub.tier.isin(["exact", "strict"]).sum())
        n_flag = int((sub.tier == "loose").sum())
        if det is None:
            rows.append({"issue_key": issue, "issue": PROBLEM_TEXT[issue], "detector_ran": False,
                         "detected_rate": np.nan, "n_repair_candidates": 0, "n_flag_only": 0,
                         "repairability_status": "not_assessed", "benchmark_risk": np.nan,
                         "benchmark_repair_precision": np.nan, "safe_auto": False,
                         "default_action": "not assessed", "evidence_source": "-",
                         "reason": skipped.get(det_key, "detector did not run")})
            continue
        ev = _evidence_for(issue, det.rate, selfbench)
        source = ev["source"] if ev else "-"
        if ev is None:
            safe, reason = False, "no repair evidence available"
            status, risk, prec = "unknown", np.nan, np.nan
        else:
            status, risk, prec = ev["status"], ev["risk"], ev["precision"]
            safe = risk <= 1e-9 and prec >= SAFE_REPAIR_MIN_PRECISION and status != "negative"
            guarded = issue in NEVER_AUTO_APPLY
            reason = (f"{source}, nearest evaluated rate ({ev['rate']:.0%}): repair precision "
                      f"{prec:.2f}, {ev['risk_kind']} risk {risk:.4f}, status '{status}' -- "
                      + ("always human review: this repair type is never applied automatically "
                         "(it rewrites the target or was harmful in testing)" if guarded else
                         "meets the bar for safe automatic repair" if safe else
                         f"does not meet the safe-auto bar (risk must be 0 and precision >= "
                         f"{SAFE_REPAIR_MIN_PRECISION:.2f}, status not 'negative')"))
            safe = safe and not guarded
        if n_auto == 0 and n_flag == 0:
            action = "nothing to repair"
        elif safe and n_auto > 0:
            action = "auto-repair"
        else:
            action = "flag for human review"
        rows.append({"issue_key": issue, "issue": PROBLEM_TEXT[issue], "detector_ran": True,
                     "detected_rate": det.rate, "n_repair_candidates": n_auto, "n_flag_only": n_flag,
                     "repairability_status": status, "benchmark_risk": risk,
                     "benchmark_repair_precision": prec, "safe_auto": bool(safe),
                     "default_action": action, "evidence_source": source, "reason": reason})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Stage 1: audit
# ---------------------------------------------------------------------------
def _stratified_sample_index(y: pd.Series | None, index: pd.Index, n: int, seed: int) -> pd.Index:
    """Random sample of `n` rows; stratified by class when there is a target (every class keeps >= 5 rows
    where it has them) so the cross-validated label detector still sees all classes."""
    if y is None:
        return index[np.sort(np.random.default_rng(seed).choice(len(index), size=n, replace=False))]
    frac = n / len(y)
    parts = [g.sample(n=min(len(g), max(5, int(round(len(g) * frac)))), random_state=seed)
             for _, g in y.groupby(y)]
    return pd.concat(parts).index.sort_values()


def audit_dataset(df: pd.DataFrame, target: str | None, include_feature_anomalies: bool = False, seed: int = 42,
                  progress: Progress | None = None, max_model_rows: int = MAX_MODEL_ROWS,
                  input_info: dict | None = None, detectors: set[str] | None = None) -> AuditState:
    """Detect problems and propose repairs. Expensive (model-based detectors).

    target=None runs in no-target mode: missing values, duplicates, outliers and inconsistent values are
    checked; the label detector cannot run. Datasets with more than `max_model_rows` rows run the slow
    cross-validated detectors on a stratified sample (cheap checks always use every row).

    `detectors` restricts which detectors run (names: missing_values, duplicates, outliers, inconsistency,
    label_noise, feature_corruption); None runs all of them. Used by the self-benchmark, which only needs the
    detector matching the error it injected."""
    say = progress or (lambda f, m: None)
    want = lambda name: detectors is None or name in detectors     # noqa: E731
    t0 = time.monotonic()
    df = df.reset_index(drop=True).copy()
    if df.empty:
        raise ValueError("the dataset has no rows")
    X, y, integer_like, excluded, notes, comma_cols = _prepare(df, target)

    unmodeled = {c: f"{X[c].nunique()} distinct values (> {MAX_CAT_LEVELS}); skipped by model-based detectors"
                 for c in X.columns if not pd.api.types.is_numeric_dtype(X[c]) and X[c].nunique() > MAX_CAT_LEVELS}
    labeled = y.notna() if y is not None else pd.Series(True, index=X.index)
    X_model = X.loc[labeled].drop(columns=list(unmodeled))
    y_lab = y.loc[labeled] if y is not None else None
    n_unlabeled = int((~labeled).sum()) if y is not None else 0
    if n_unlabeled:
        notes.append(f"{n_unlabeled:,} row(s) have no target value: they are still checked for missing values, "
                     "duplicates and outliers, but are excluded from the label and model-based detectors")

    sampled = len(X_model) > max_model_rows
    if sampled:
        sample_idx = _stratified_sample_index(y_lab, X_model.index, max_model_rows, seed)
        X_model = X_model.loc[sample_idx]
        y_model = None if y_lab is None else y_lab.loc[sample_idx]
        notes.append(f"{len(labeled):,} rows exceed the {max_model_rows:,}-row limit for the model-based detectors: they ran "
                     f"on a stratified random sample of {len(sample_idx):,} rows, so label/feature findings and "
                     "proposals cover only those rows (the cheap checks use every row)")
    else:
        y_model = y_lab

    detections, skipped, timings = {}, {}, {}

    def timed(name, fn):
        t = time.monotonic()
        out = fn()
        timings[name] = round(time.monotonic() - t, 2)
        return out

    full = _full_record(X, y)
    if want("missing_values"):
        say(0.05, "Checking missing values")
        detections["missing_values"] = timed("missing_values", lambda: detect_missing(X))
    if want("duplicates"):
        say(0.10, "Checking duplicate records")
        detections["duplicates"] = timed("duplicates", lambda: detect_duplicates(full))
    if want("outliers"):
        say(0.15, "Checking outliers")
        detections["outliers"] = timed("outliers", lambda: detect_outliers_iqr(X))
    if want("inconsistency"):
        detections["inconsistency"] = timed("inconsistency", lambda: detect_inconsistency(X))
    for name in ("missing_values", "duplicates", "outliers", "inconsistency"):
        if name not in detections:
            skipped[name] = "not requested"

    if not want("label_noise"):
        skipped["label_noise"] = "not requested"
    elif y_model is None:
        skipped["label_noise"] = "no target column selected"
    elif X_model.shape[1] >= 1 and y_model.value_counts().min() >= 2:
        say(0.25, "Estimating label reliability (cross-validated model)")
        detections["label_noise"] = timed("label_noise", lambda: detect_label_noise(
            X_model, y_model, confidence_threshold=0.9, seed=seed))
    else:
        skipped["label_noise"] = ("no modellable feature columns" if X_model.shape[1] == 0
                                  else "a target class has fewer than 2 rows")

    if not want("feature_corruption"):
        skipped["feature_corruption"] = "not requested"
    elif include_feature_anomalies and X_model.shape[1] >= 2:
        say(0.55, "Checking feature anomalies (one model per column -- slow)")
        detections["feature_corruption"] = timed("feature_corruption", lambda: detect_feature_corruption_crossfeature(
            X_model, seed=seed))
    else:
        skipped["feature_corruption"] = ("not run (disabled: slow, low benchmark precision, and its automatic "
                                         "repair was found harmful)" if not include_feature_anomalies
                                         else "needs at least 2 modellable feature columns")

    say(0.90, "Building repair proposals")
    frames: list[pd.DataFrame] = []
    if "missing_values" in detections:
        frames += _propose_missing(X, integer_like, detections["missing_values"].extra["cell_mask"], df)
    if "duplicates" in detections:
        frames += _propose_duplicates(full, detections["duplicates"].row_mask)
    if "outliers" in detections:
        frames += _propose_outliers(X, integer_like, detections["outliers"])
    if "label_noise" in detections:
        frames += _propose_labels(y, detections["label_noise"])
    if "feature_corruption" in detections:
        frames += _propose_features(X_model, integer_like, detections["feature_corruption"])
    proposals = (pd.concat(frames, ignore_index=True) if frames else
                 pd.DataFrame(columns=["issue_key", "row_id", "column", "problem", "original_value",
                                       "proposed_value", "method", "tier", "detector_score"]))
    policy = _build_policy(detections, proposals, skipped)

    meta = {
        "seed": seed, "n_rows_uploaded": len(df), "n_rows_analysed": len(X), "n_rows_missing_target": n_unlabeled,
        "n_columns": df.shape[1], "include_feature_anomalies": include_feature_anomalies,
        "no_target_mode": target is None,
        "model_detector_rows": len(X_model), "model_detectors_sampled": sampled,
        "input_format": input_info or {}, "notes": notes,
        "detector_seconds": timings, "audit_seconds": round(time.monotonic() - t0, 2),
    }
    say(1.0, "Audit complete")
    return AuditState(df=df, target=target, X=X, y=y, integer_like=integer_like, excluded_columns=excluded,
                      unmodeled_columns=unmodeled, detections=detections, skipped=skipped,
                      proposals=proposals, policy=policy, meta=meta, decimal_comma_columns=comma_cols)


# ---------------------------------------------------------------------------
# Stage 2: apply repairs
# ---------------------------------------------------------------------------
def attach_self_benchmark(state: AuditState, selfbench) -> AuditState:
    """Make the policy, impact estimates and reports use this dataset's own measured evidence. If the
    self-benchmark could not run (`ok` False) the Adult evidence stays in force and the reason is reported."""
    state.selfbench = selfbench
    state.policy = _build_policy(state.detections, state.proposals, state.skipped, selfbench)
    return state


def default_apply_issues(state: AuditState) -> set[str]:
    """Issue types the evidence says are safe to repair automatically."""
    p = state.policy
    return set(p.loc[(p.default_action == "auto-repair"), "issue_key"])


def _assign(cleaned: pd.DataFrame, col: str, ids, values, label_map=None, decimal_comma=False):
    """Write `values` into cleaned[col] at row labels `ids`, respecting dtype."""
    ids = list(ids)
    s = cleaned[col]
    if label_map is not None:
        values = [label_map.get(str(v), v) for v in values]
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        values = pd.to_numeric(pd.Series(values, dtype=object)).values
        if pd.api.types.is_integer_dtype(s) and not np.all(np.mod(values, 1) == 0):
            cleaned[col] = s.astype(float)
        elif pd.api.types.is_integer_dtype(s):
            values = values.astype(s.dtype)
        cleaned.loc[ids, col] = values
    else:
        # text / category column: a numeric proposal (e.g. an imputed value for a numeric-looking text column)
        # is written as text so the column keeps a consistent type
        values = [v if isinstance(v, str) else (f"{float(v):.10g}".replace(".", ",") if decimal_comma
                                                 else f"{float(v):.10g}") for v in values]
        if isinstance(s.dtype, pd.CategoricalDtype):
            new_cats = [v for v in set(values) if v not in s.cat.categories]
            if new_cats:
                cleaned[col] = s.cat.add_categories(new_cats)
        cleaned.loc[ids, col] = values


def _apply(state: AuditState, apply_issues: set[str], overrides: list[str]):
    """Apply `apply_issues` to a copy of the uploaded frame. Returns (cleaned, repair_log)."""
    cleaned = state.df.copy()
    prop = state.proposals.copy()
    prop["action"] = np.where(prop.tier == "loose", "flagged_only", "flagged_for_review")
    prop["applied_value"] = pd.Series([np.nan] * len(prop), dtype=object)
    modified: set[tuple] = set()
    drop_ids: list[int] = []
    label_map = ({str(v): v for v in state.df[state.target].dropna().unique()} if state.target is not None else {})
    # Rows that will be removed as duplicates: repairing cells on a row we are
    # about to delete would make the log claim a change that does not exist in
    # the output file, so those repairs are skipped (and logged as such).
    drop_set: set[int] = set()
    if "duplicates" in apply_issues:
        dup_idx = prop.index[(prop.issue_key == "duplicates") & prop.tier.isin(["exact", "strict"])]
        drop_set = set(prop.loc[dup_idx, "row_id"])

    for issue in APPLY_ORDER:
        if issue not in apply_issues:
            continue
        idx = prop.index[(prop.issue_key == issue) & prop.tier.isin(["exact", "strict"])]
        if len(idx) == 0:
            continue
        sub = prop.loc[idx]
        if issue == "duplicates":
            drop_ids = sub.row_id.tolist()
            prop.loc[idx, "action"] = "applied"
            prop.loc[idx, "applied_value"] = "row dropped"
            continue
        is_label = issue == "label_errors"
        col_key = pd.Series(state.target, index=sub.index) if is_label else sub["column"]
        for col, g in sub.groupby(col_key):
            in_drop = g.row_id.isin(drop_set).values
            conflict = np.array([(r, col) in modified for r in g.row_id], dtype=bool)
            prop.loc[g.index[in_drop], "action"] = "skipped_row_dropped"
            prop.loc[g.index[~in_drop & conflict], "action"] = "skipped_conflict"
            g = g[~in_drop & ~conflict]
            if g.empty:
                continue
            _assign(cleaned, col, g.row_id.values, g.proposed_value.values, label_map if is_label else None,
                    decimal_comma=col in state.decimal_comma_columns)
            modified |= {(r, col) for r in g.row_id}
            prop.loc[g.index, "action"] = "applied"
            # record what was actually written to the file (e.g. '38,2' in a decimal-comma column)
            prop.loc[g.index, "applied_value"] = [cleaned.at[r, col] for r in g.row_id]

    if drop_ids:
        cleaned = cleaned.drop(index=drop_ids)

    pol = state.policy.set_index("issue_key")
    prop["repair_evidence_status"] = prop.issue_key.map(pol["repairability_status"])
    prop["human_override"] = prop.issue_key.isin(overrides) & (prop.action == "applied")
    prop.insert(0, "repair_id", [f"R{i:06d}" for i in range(1, len(prop) + 1)])
    repair_log = prop[["repair_id", "row_id", "column", "problem", "original_value", "proposed_value",
                       "applied_value", "action", "method", "tier", "detector_score",
                       "repair_evidence_status", "human_override"]].copy()
    repair_log["column"] = repair_log["column"].replace("", np.nan)
    if state.target is not None:
        repair_log.loc[(repair_log.problem == PROBLEM_TEXT["label_errors"]), "column"] = state.target
    return cleaned, repair_log


def apply_repairs(state: AuditState, apply_issues: set[str] | None = None) -> TaskCleanResult:
    """Apply the chosen repair types. `apply_issues=None` means "exactly what the
    evidence-based policy approves". Passing extra issue types is an explicit
    human override and is recorded as such in the log and report.

    Always also builds the "aggressive" variant (every proposal except feature-anomaly
    repairs) so a model-ready file exists even when the policy approves almost nothing."""
    policy_default = default_apply_issues(state)
    apply_issues = set(policy_default if apply_issues is None else apply_issues)
    overrides = sorted(apply_issues - policy_default)
    cleaned, repair_log = _apply(state, apply_issues, overrides)

    aggressive_issues = {i for i in ISSUES if i != "feature_corruption"}
    aggressive_overrides = sorted(aggressive_issues - policy_default)
    cleaned_agg, log_agg = _apply(state, aggressive_issues, aggressive_overrides)
    summary_agg = _summary(state, log_agg, cleaned_agg, aggressive_issues, aggressive_overrides)

    after = _after_cleaning_rates(cleaned, state.target)
    after_agg = _after_cleaning_rates(cleaned_agg, state.target)
    quality = _quality_report(state)
    impact = _impact_report(state, apply_issues)
    summary = _summary(state, repair_log, cleaned, apply_issues, overrides)
    readiness = _readiness(state, quality, impact, apply_issues, overrides, after, summary)
    readiness["aggressive_variant"] = {
        "description": ("Every proposed repair except feature-anomaly repairs (harmful in the benchmark). Intended "
                        "as a model-ready file; each applied repair outside the evidence-based policy is a human-"
                        "override-style risk and is marked as such in repair_log_aggressive.csv."),
        "issues_applied": sorted(aggressive_issues), "changes_summary": summary_agg,
        "rates_after_cleaning": after_agg,
    }
    return TaskCleanResult(cleaned=cleaned, repair_log=repair_log, quality_report=quality,
                           impact_report=impact, policy=state.policy, readiness=readiness, summary=summary,
                           cleaned_aggressive=cleaned_agg, repair_log_aggressive=log_agg,
                           summary_aggressive=summary_agg, selfbench=state.selfbench)


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------
def _after_cleaning_rates(cleaned: pd.DataFrame, target: str | None) -> dict:
    """Cheap, deterministic detectors only -- the model-based ones are not re-run."""
    try:
        X, y, _, _, _, _ = _prepare(cleaned.reset_index(drop=True), target)
    except ValueError:
        return {}
    return {"missing_values": detect_missing(X).rate,
            "duplicates": detect_duplicates(_full_record(X, y)).rate,
            "outliers": detect_outliers_iqr(X).rate,
            "inconsistency": detect_inconsistency(X).rate}


def _ok_selfbench(state: AuditState):
    sb = state.selfbench
    return sb if sb is not None and getattr(sb, "ok", False) else None


def _evidence_status(issue: str, tracking: pd.DataFrame | None) -> tuple[str, str]:
    """(status, note) for how far a detector's flagged rate can be trusted as a severity signal: measured on this
    dataset when calibrated, else the UCI Adult validation (Phase 11B)."""
    if tracking is not None and issue in tracking.index:
        return tracking.loc[issue, "evidence_status"], tracking.loc[issue, "note"]
    return EVIDENCE_STATUS.get(issue, "not_estimable"), EVIDENCE_NOTES.get(issue, "")


def _quality_report(state: AuditState) -> pd.DataFrame:
    p5_path = os.path.join(RESULTS_DIR, "phase5_detection_scores.csv")
    p5 = pd.read_csv(p5_path).set_index("error_type") if os.path.exists(p5_path) else None
    sb = _ok_selfbench(state)
    reliability = sb.detector_reliability.set_index("error_type") if sb is not None else None
    tracking = sb.detector_tracking.set_index("error_type") if sb is not None else None
    rows = []
    order = [("missing_values", "missing_values", "exact check", "cells"),
             ("duplicates", "duplicates", "exact check", "rows"),
             ("outliers", "outliers", "statistical estimate", "rows"),
             ("inconsistency", "inconsistency", "exact check", "rows"),
             ("label_noise", "label_errors", "statistical estimate", "rows"),
             ("feature_corruption", "feature_corruption", "statistical estimate", "rows")]
    for det_key, issue, dtype, unit in order:
        det = state.detections.get(det_key)
        rec = {"dimension": LABELS.get(issue, issue), "issue_key": issue, "detector_type": dtype}
        if det is None:
            rec.update(detected_rate=np.nan, detected_count=np.nan, count_unit=unit, benchmark_precision=np.nan,
                       benchmark_recall=np.nan, reliability_source=None, evidence_status="not_run",
                       note=state.skipped.get(det_key, "detector did not run"))
        else:
            n_rows = len(state.X)
            count = (int(det.extra["cell_mask"].values.sum()) if det_key == "missing_values"
                     else int(det.row_mask.sum()))
            prec = rec_ = np.nan
            rel_source = None
            if reliability is not None and issue in reliability.index:
                prec, rec_ = float(reliability.loc[issue, "precision"]), float(reliability.loc[issue, "recall"])
                rel_source = "this dataset (self-benchmark)"
            elif p5 is not None and issue in p5.index:
                prec, rec_ = float(p5.loc[issue, "precision"]), float(p5.loc[issue, "recall"])
                rel_source = "UCI Adult benchmark"
            ev_status, ev_note = _evidence_status(issue, tracking)
            note = ("report-only: no repair evidence exists for this issue type, so no repair is proposed"
                    if issue == "inconsistency" else
                    "flagged rate, not a confirmed error rate" if dtype != "exact check" else "")
            rec.update(detected_rate=det.rate, detected_count=count, count_unit=unit,
                       benchmark_precision=prec, benchmark_recall=rec_, reliability_source=rel_source,
                       evidence_status=ev_status, note=note)
        rows.append(rec)
    return pd.DataFrame(rows)


def _impact_report(state: AuditState, apply_issues: set[str]) -> pd.DataFrame:
    pol = state.policy.set_index("issue_key")
    sb = _ok_selfbench(state)
    rows = []
    for issue in ISSUES:
        p = pol.loc[issue]
        if not p.detector_ran:
            rows.append({"issue": PROBLEM_TEXT[issue], "issue_key": issue, "detected_rate": np.nan,
                         "benchmark_impact_estimate": np.nan, "is_boundary_estimate": False,
                         "impact_source": "-", "impact_unit": "-",
                         "repairability_status": "not_assessed", "benchmark_risk": np.nan,
                         "policy_decision": "not assessed", "recommendation": p.reason})
            continue
        own = _impact_curve_estimate(sb.impact_curve, issue, p.detected_rate) if sb is not None else None
        if own is not None:
            (est, boundary), source, unit = own, "this dataset (self-benchmark)", sb.metric
        else:
            est, boundary = _estimate_impact(issue, p.detected_rate)
            source, unit = "UCI Adult benchmark", "F1 (positive class)"
        if issue in apply_issues and p.n_repair_candidates > 0:
            decision = "applied" + ("" if p.safe_auto else " (human override)")
        elif p.n_repair_candidates + p.n_flag_only == 0:
            decision = "nothing to repair"
        else:
            decision = "flagged for human review (not applied)"
        rows.append({"issue": PROBLEM_TEXT[issue], "issue_key": issue, "detected_rate": p.detected_rate,
                     "benchmark_impact_estimate": est, "is_boundary_estimate": boundary,
                     "impact_source": source, "impact_unit": unit,
                     "repairability_status": p.repairability_status, "benchmark_risk": p.benchmark_risk,
                     "policy_decision": decision,
                     "recommendation": RECOMMENDATION_TEXT.get(p.repairability_status, RECOMMENDATION_TEXT["unknown"])})
    return pd.DataFrame(rows)


def _summary(state, repair_log, cleaned, apply_issues, overrides) -> dict:
    applied = repair_log[repair_log.action == "applied"]
    return {
        "rows_uploaded": len(state.df), "rows_after_cleaning": len(cleaned),
        "rows_dropped": len(state.df) - len(cleaned),
        "cells_modified": int((applied.problem != PROBLEM_TEXT["duplicates"]).sum()),
        "repairs_applied_by_issue": applied.groupby("problem").size().to_dict(),
        "flagged_for_review": int((repair_log.action == "flagged_for_review").sum()),
        "flagged_only": int((repair_log.action == "flagged_only").sum()),
        "skipped_conflicts": int((repair_log.action == "skipped_conflict").sum()),
        "skipped_row_dropped": int((repair_log.action == "skipped_row_dropped").sum()),
        "issues_applied": sorted(apply_issues), "human_overrides": overrides,
    }


def _readiness(state, quality, impact, apply_issues, overrides, after, summary) -> dict:
    baseline = _load_baseline_rates()
    sb = _ok_selfbench(state)
    tracking = sb.detector_tracking.set_index("error_type") if sb is not None else None
    dims = []
    for _, q in quality.iterrows():
        issue = q.issue_key
        ran = not pd.isna(q.detected_rate)
        evidence, evidence_note = _evidence_status(issue, tracking) if ran else ("not_run", "")
        exact = q.detector_type == "exact check"
        own_floor = sb is not None and issue in sb.baseline_floors
        base = sb.baseline_floors[issue] if own_floor else baseline.get(issue)
        if exact:
            base = 0.0   # an exact check has no false positives by construction, on any dataset
        status = _status(issue, q.detected_rate, base, evidence) if ran else "Not assessed"
        floor_source = ("0 by construction (exact check)" if exact else
                        "this dataset's clean-ish reference (self-benchmark)" if own_floor else
                        "Adult benchmark floor -- NOT recalibrated for this dataset")
        dims.append({
            "dimension": q.dimension, "issue_key": issue,
            "observed_rate": None if not ran else float(q.detected_rate),
            "baseline_false_positive_floor": base,
            "baseline_source": floor_source if ran else None,
            "evidence_status": evidence, "evidence_note": evidence_note, "status": status,
            "rate_after_cleaning": after.get(issue),
        })
    meta = state.meta
    return {
        "taskclean_version": TASKCLEAN_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset_overview": {
            "rows_uploaded": meta["n_rows_uploaded"], "rows_analysed": meta["n_rows_analysed"],
            "rows_without_target_excluded_from_label_checks": meta["n_rows_missing_target"],
            "columns": meta["n_columns"], "target": state.target,
            "task": "classification" if state.target is not None else "none (no-target mode: label checks skipped)",
            "n_classes": int(state.y.nunique()) if state.y is not None else None,
            "input_format": meta.get("input_format", {}),
            "notes": meta.get("notes", []),
            "model_based_detectors_ran_on_rows": meta["model_detector_rows"],
            "model_based_detectors_sampled": meta["model_detectors_sampled"],
            "columns_not_analysed": state.excluded_columns,
            "columns_skipped_by_model_based_detectors": state.unmodeled_columns,
            "model_used_for_benchmark_validation": (
                f"RandomForestClassifier (n_estimators={sb.n_estimators}) on this dataset's clean-ish reference; "
                f"{sb.metric}" if sb is not None else "RandomForestClassifier (n_estimators=300), UCI Adult"),
        },
        "dimensions": dims,
        "benchmark_task_impact": [{
            "issue": r.issue, "detected_rate": None if pd.isna(r.detected_rate) else float(r.detected_rate),
            "benchmark_impact_estimate": None if pd.isna(r.benchmark_impact_estimate) else float(r.benchmark_impact_estimate),
            "is_boundary_estimate": bool(r.is_boundary_estimate), "impact_source": r.impact_source,
            "impact_unit": r.impact_unit, "repairability_status": r.repairability_status,
            "policy_decision": r.policy_decision, "recommendation": r.recommendation,
        } for r in impact.itertuples()],
        "impact_estimate_note": (
            "Self-benchmark impact estimate: the detected rate is mapped onto a damage curve measured on a "
            "clean-ish reference built from THIS dataset (corruption injected at 5/10/20%, macro-F1 on held-out "
            "reference rows). It is an estimate of relative harm for this data and a Random Forest, not a "
            "validated prediction of your model's loss. Beyond the highest measured rate the value is a boundary "
            "estimate, not an extrapolation. Issues the self-benchmark did not measure use the UCI Adult curve."
            if sb is not None else
            "Benchmark-based task-impact estimate: the detected rate is mapped onto the controlled Phase 7 "
            "damage curve (UCI Adult / Random Forest, rates 5-20%). It is not a measured or validated "
            "prediction of F1 loss for this dataset. Where the observed rate exceeds the validated 0-20% "
            "range the value is a boundary estimate (the 20% benchmark), not an extrapolation."),
        "repair_policy": {
            "safe_auto_requires": (f"observed harm risk == 0, repair precision >= {SAFE_REPAIR_MIN_PRECISION}, "
                                   "status != negative -- evaluated on "
                                   + ("this dataset's self-benchmark where measured, else the UCI Adult benchmark"
                                      if sb is not None else "the UCI Adult benchmark")),
            "issues_applied": sorted(apply_issues), "human_overrides": overrides,
            "decisions": state.policy[["issue", "default_action", "repairability_status", "evidence_source",
                                       "reason"]].to_dict("records"),
        },
        "self_benchmark": (state.selfbench.summary() if state.selfbench is not None else
                           {"ran": False, "reason": "not requested: all evidence comes from the UCI Adult benchmark"}),
        "changes_summary": summary,
        "run": {**meta, "python": platform.python_version(), "pandas": pd.__version__,
                "scikit_learn": sklearn.__version__},
        "limitations": _limitations(state),
    }


LIMITATIONS = [
    "Estimate-based audit: for an uploaded dataset the true error locations are unknown, so every rate comes "
    "from a detector with known precision/recall limits (Phase 5). A flagged rate is not a confirmed error rate.",
    "Benchmark-based impact estimates come from UCI Adult + Random Forest and do not transfer automatically to "
    "another dataset or model.",
    "The false-positive floors used for 'Attention/Good' status on statistical detectors (outliers, feature "
    "anomalies, labels) were measured on clean Adult data and are NOT recalibrated for this dataset.",
    "The label-error detector's flagged rate showed an INVERTED relationship with real damage in validation; a "
    "low flagged label-error rate must not be read as reassurance.",
    "Validation evidence per error type rests on 3 corruption rates (n=3): directional, not statistically significant.",
    "No single AI-readiness score is reported: the tested aggregate of detected rates was not supported by the "
    "evidence (pooled r = -0.24). This does not prove no composite could ever work.",
    "No before/after model metrics are reported: an uploaded dataset has no clean held-out test set, and "
    "evaluating on a held-out slice of the same dirty data would be misleading.",
    "Automatic repair is applied only for issue types whose benchmark repair showed zero observed harm and high "
    "precision. 'Repair not recommended' means the evaluated repair strategy was harmful in testing -- not that "
    "the underlying error is inherently unrepairable.",
    "Classification targets only (or no target: label checks are skipped). Duplicates are full-record duplicates "
    "(features + target).",
    "Placeholder tokens ('?', 'N/A', '-', empty, ...) are counted as missing in the analysis; the original cell is "
    "left as-is unless a missing-value repair is applied. A token that is a genuine category in your data will "
    "be over-counted as missing.",
    "Files above the model-detector row limit are audited on a stratified sample for the label/feature "
    "detectors; their findings and proposals then cover only the sampled rows.",
    "The 'aggressive' variant applies every proposed repair except feature-anomaly repairs. In the benchmark only "
    "duplicate removal met the safe-auto bar, so every other repair in that file carries the risks described above.",
]


SELFBENCH_LIMITATIONS = [
    "Self-benchmark evidence comes from a CLEAN-ISH reference built from your own data (complete rows, no "
    "duplicates, high-confidence label problems removed), not true ground truth. Undetected problems left in the "
    "reference, and selection bias from dropping incomplete rows, can make damage look smaller or repairs look "
    "easier than they are.",
    "Each measurement uses only a few seeds and 3 corruption rates, a Random Forest, and macro-F1: 'no harm "
    "observed' means none in those runs, not a guarantee. Your own model may react differently.",
    "Issues the self-benchmark did not measure (feature anomalies, consistency) still use the UCI Adult "
    "evidence, and each decision names its evidence source.",
    "The self-benchmark injects RANDOM errors. Real errors are often structured (systematic mislabeling, a "
    "faulty sensor), so measured repair precision can be optimistic. That is why label and feature-anomaly "
    "repairs are never applied automatically, whatever the evidence.",
]


def _limitations(state: AuditState) -> list[str]:
    adult_specific = (1, 2, 3)      # benchmark-transfer / Adult-floor / Adult label-detector statements
    sb = state.selfbench
    if sb is not None and getattr(sb, "ok", False):
        return SELFBENCH_LIMITATIONS + [t for i, t in enumerate(LIMITATIONS) if i not in adult_specific]
    out = list(LIMITATIONS)
    if sb is not None:
        out.insert(0, f"Self-benchmark did not run on this dataset ({sb.reason}); all evidence comes from the "
                      "UCI Adult benchmark.")
    else:
        out.append("Calibrating on your own data (self-benchmark) was not requested; enabling it replaces the "
                   "Adult evidence with measurements from this dataset.")
    return out


# ---------------------------------------------------------------------------
# Output files
# ---------------------------------------------------------------------------
def _sanitize(o):
    if isinstance(o, dict):
        return {str(k): _sanitize(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [_sanitize(v) for v in o]
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return float(o) if np.isfinite(o) else None
    if o is pd.NA:
        return None
    return o


def outputs_as_bytes(result: TaskCleanResult) -> dict[str, bytes]:
    csv = lambda df: df.to_csv(index=False).encode("utf-8")
    dims = pd.DataFrame(result.readiness["dimensions"])
    extra = {}
    sb = result.selfbench
    if sb is not None and getattr(sb, "ok", False):
        extra = {"selfbenchmark_evidence.csv": csv(sb.evidence), "selfbenchmark_runs.csv": csv(sb.runs)}
    return {**extra,
        "dataset_cleaned.csv": csv(result.cleaned),
        "repair_log.csv": csv(result.repair_log),
        "dataset_cleaned_aggressive.csv": csv(result.cleaned_aggressive),
        "repair_log_aggressive.csv": csv(result.repair_log_aggressive),
        "quality_report.csv": csv(result.quality_report),
        "impact_report.csv": csv(result.impact_report),
        "readiness_report.json": json.dumps(_sanitize(result.readiness), indent=2).encode("utf-8"),
        "readiness_report.csv": csv(dims),
    }


def write_outputs(result: TaskCleanResult, out_dir: str) -> dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    paths = {}
    for name, data in outputs_as_bytes(result).items():
        paths[name] = os.path.join(out_dir, name)
        with open(paths[name], "wb") as f:
            f.write(data)
    return paths


def outputs_zip(result: TaskCleanResult) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in outputs_as_bytes(result).items():
            z.writestr(name, data)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    import argparse

    ap = argparse.ArgumentParser(description="TaskClean: task-aware data quality audit and safe repair")
    ap.add_argument("csv")
    ap.add_argument("--target", default=None,
                    help="target column (classification). Omit for no-target mode: missing values, duplicates, "
                         "outliers and inconsistencies only")
    ap.add_argument("--out", default="taskclean_output")
    ap.add_argument("--feature-anomalies", action="store_true",
                    help="also run the slow feature-anomaly detector (its auto-repair is never recommended)")
    ap.add_argument("--apply", default=None,
                    help="comma-separated issue keys to apply, overriding the evidence-based policy "
                         f"(choices: {', '.join(ISSUES)}); default = what the policy approves")
    ap.add_argument("--max-model-rows", type=int, default=MAX_MODEL_ROWS,
                    help="above this many rows the slow detectors run on a stratified sample")
    ap.add_argument("--calibrate", action="store_true",
                    help="self-benchmark on this dataset: measure damage, repair effectiveness and harm risk on a "
                         "clean-ish reference built from your own data, and base the policy on that (needs a target)")
    ap.add_argument("--calibrate-seeds", type=int, default=3, help="seeds per measurement (3 = quick, 5 = thorough)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df, info = load_csv(args.csv)
    print(f"Read {info['rows']:,} rows x {info['columns']} columns "
          f"(encoding {info['encoding']}, delimiter '{info['delimiter']}', decimal '{info['decimal']}')")
    state = audit_dataset(df, args.target, include_feature_anomalies=args.feature_anomalies, seed=args.seed,
                          progress=lambda f, m: print(f"  [{f:4.0%}] {m}"), max_model_rows=args.max_model_rows,
                          input_info=info)
    for note in state.meta["notes"]:
        print("  note:", note)
    if args.calibrate:
        from selfbench import run_self_benchmark       # local import: selfbench imports this module
        print("\nCalibrating on this dataset (self-benchmark)...")
        sb = run_self_benchmark(state, n_seeds=args.calibrate_seeds,
                                progress=lambda f, m: print(f"  [{f:4.0%}] {m}") if m.endswith("(seed 42)") else None)
        attach_self_benchmark(state, sb)
        if sb.ok:
            print(f"  done in {sb.seconds:.0f}s; noise floor {sb.noise_floor:.4f} {sb.metric}; "
                  f"reference {sb.reference['rows_used']:,} rows")
        else:
            print(f"  skipped: {sb.reason}\n  (the policy keeps using the UCI Adult evidence)")
    apply_issues = None if args.apply is None else {s.strip() for s in args.apply.split(",") if s.strip()}
    result = apply_repairs(state, apply_issues)
    paths = write_outputs(result, args.out)

    print("\nREPAIR POLICY")
    print(state.policy[["issue", "detected_rate", "n_repair_candidates", "n_flag_only", "default_action",
                        "evidence_source"]].round(4).to_string(index=False))
    print("\nEVIDENCE-BASED FILE:", json.dumps(_sanitize(result.summary)))
    print("AGGRESSIVE FILE:    ", json.dumps(_sanitize(result.summary_aggressive)))
    print("\nWrote:")
    for name, path in paths.items():
        print(f"  {path}")


if __name__ == "__main__":
    main()
