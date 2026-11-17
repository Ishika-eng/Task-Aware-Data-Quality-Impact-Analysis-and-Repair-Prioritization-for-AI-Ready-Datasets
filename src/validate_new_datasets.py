"""
Phase 23 validation on genuine datasets the project has never used (docs/PHASE23_SPEC.md). No product code is changed here.

  python validate_new_datasets.py T        Titanic (classification)
  python validate_new_datasets.py A        Ames Housing (regression)
  python validate_new_datasets.py R        Real Estate Valuation .xlsx (regression, under the 600-row minimum)
  python validate_new_datasets.py O        Online Retail II .xlsx (large, two sheets, no target)
  python validate_new_datasets.py C        Concrete .xls (must be refused clearly)

Each run writes results/phase23_<id>.json. It calls the same backend functions the API calls (load_table, store_dataset,
run_audit_job, run_panel_job), so it is the product path, not a re-implementation.
"""
from __future__ import annotations

import json
import os
import resource
import sys
import time
import traceback

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path[:0] = [os.path.join(ROOT, "src"), ROOT]

import backend.main as bm                      # noqa: E402
from intake import load_table                  # noqa: E402

NEW = os.path.join(ROOT, "data", "new")
OUT = os.path.join(ROOT, "results")
FILES = {
    "T": dict(path="titanic.csv", target="Survived", task="classification", drop=["PassengerId"]),
    "A": dict(path="ames_housing.csv", target="SalePrice", task="regression", drop=["Id"]),
    "R": dict(path="realestate/Real estate valuation data set.xlsx", target="Y house price of unit area", task="regression",
              drop=["No"]),
    "O": dict(path="retail2/online_retail_II.xlsx", target=None, task="classification", drop=[]),
    "C": dict(path="concrete/Concrete_Data.xls", target=None, task="regression", drop=[]),
}
SPOT_N, SPOT_SEED = 20, 0


def rss_mb() -> float:
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 1)          # bytes on macOS


def strings(obj, out=None):
    """Every string VALUE in a nested payload (keys are internal names, not page text)."""
    out = [] if out is None else out
    if isinstance(obj, dict):
        for v in obj.values():
            strings(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            strings(v, out)
    elif isinstance(obj, str):
        out.append(obj)
    return out


def jsonable(o):
    return json.loads(json.dumps(o, default=lambda x: x.item() if hasattr(x, "item") else str(x)))


def same_cell(a, b) -> bool:
    return (pd.isna(a) and pd.isna(b)) or a == b


def spot_check(audit: dict, df: pd.DataFrame) -> dict:
    """20 random flagged items per problem type, with the context needed to judge them by the PHASE23_SPEC rules."""
    log = audit["result"].repair_log
    state = audit["state"]
    index = state.X.index
    out = {}
    for problem, g in log.groupby("problem"):
        n_flagged = len(g)
        s = g.sample(min(SPOT_N, n_flagged), random_state=SPOT_SEED)
        items = []
        for r in s.itertuples():
            pos = int(r.row_id)
            row_label = index[pos] if 0 <= pos < len(index) else pos
            item = {"row": str(row_label), "column": r.column, "original": None if pd.isna(r.original_value) else str(r.original_value),
                    "proposed": None if pd.isna(r.proposed_value) else str(r.proposed_value), "action": r.action, "method": r.method}
            col = r.column if r.column in df.columns else None
            if col is not None and pd.api.types.is_numeric_dtype(df[col]):
                x = pd.to_numeric(df[col], errors="coerce")
                q1, q3 = x.quantile(.25), x.quantile(.75)
                item["column_stats"] = {"min": float(x.min()), "q1": float(q1), "median": float(x.median()), "q3": float(q3),
                                        "max": float(x.max()), "fence_lo": float(q1 - 3 * (q3 - q1)), "fence_hi": float(q3 + 3 * (q3 - q1)),
                                        "missing_share": round(float(x.isna().mean()), 3)}
            elif col is not None:
                vc = df[col].value_counts(dropna=True)
                item["column_stats"] = {"missing_share": round(float(df[col].isna().mean()), 3), "n_unique": int(df[col].nunique()),
                                        "top_values": {str(k): int(v) for k, v in vc.head(5).items()}}
            items.append(item)
        out[problem] = {"n_flagged": int(n_flagged), "actions": jsonable(g.action.value_counts().to_dict()), "sample": items}
    # duplicates are mechanical: is the flagged row really an exact copy of another row?
    full = df.copy()
    if audit["state"].target is not None and audit["state"].target in df.columns:
        pass
    dups = int(df.duplicated(keep=False).sum())
    out["_mechanical"] = {"rows_in_exact_duplicate_groups_in_raw_file": dups}
    return out


def v2_mechanical(audit: dict, df: pd.DataFrame, task: str) -> dict:
    """(a) every changed cell/row between the raw file and the evidence-based file is in the repair log as applied."""
    res, state = audit["result"], audit["state"]
    cleaned, log = res.cleaned, res.repair_log
    applied = log[log.action == "applied"]
    index = state.X.index
    n_removed = len(index.difference(cleaned.index)) if cleaned.index.is_unique else None
    common = cleaned.index.intersection(df.index)
    changed = 0
    unexplained = 0
    applied_cells = {(index[int(p)], c) for p, c in zip(applied.row_id, applied.column) if 0 <= int(p) < len(index)}
    for col in cleaned.columns:
        if col not in df.columns:
            continue
        a, b = df.loc[common, col], cleaned.loc[common, col]
        diff = ~((a == b) | (a.isna() & b.isna()))
        changed += int(diff.sum())
        for i in a.index[diff.values]:
            if (i, col) not in applied_cells:
                unexplained += 1
    removed_logged = int(applied[applied.problem.str.contains("uplicate", na=False)].shape[0])
    return {"rows_raw": int(len(df)), "rows_cleaned": int(len(cleaned)), "rows_removed": n_removed,
            "duplicate_rows_logged_applied": removed_logged, "cells_changed": changed, "cells_changed_not_in_log": unexplained,
            "applied_log_rows": int(len(applied)), "actions": jsonable(log.action.value_counts().to_dict()),
            "pass_a": bool(unexplained == 0 and (n_removed in (0, None) or n_removed == removed_logged))}


def run_tabular(fid: str) -> dict:
    spec = FILES[fid]
    path = os.path.join(NEW, spec["path"])
    out = {"id": fid, "file": spec["path"], "task": spec["task"], "target": spec["target"], "errors": []}
    raw = open(path, "rb").read()
    out["file_bytes"] = len(raw)
    t0 = time.time()
    df, info = load_table(raw, os.path.basename(path))
    out["load_seconds"] = round(time.time() - t0, 2)
    out["input_format"] = jsonable(info)
    df = df.drop(columns=[c for c in spec["drop"] if c in df.columns])
    out["rows"], out["columns"] = int(len(df)), int(df.shape[1])
    ds_id = bm.store_dataset(df, info, os.path.basename(path), raw=raw)
    ds = bm.get_dataset(ds_id)
    out["snapshot_ok"] = ds["snapshot"] is not None
    out["snapshot_error"] = ds["snapshot_error"]
    req = bm.AuditRequest(dataset_id=ds_id, target=spec["target"], task=spec["task"], calibrate=spec["target"] is not None,
                          model_panel=spec["target"] is not None, seeds=3)
    jid = "val" + fid
    bm.JOBS[jid] = bm.JobDict(id=jid, status="queued", progress=0.0, message="", stage="queued", error=None, audit_id=None,
                              started=time.time(), finished=None, calibrate=req.calibrate)
    t0 = time.time()
    bm.run_audit_job(jid, req)
    job = bm.JOBS[jid]
    out["audit_seconds"] = round(time.time() - t0, 1)
    out["job_status"], out["job_error"] = job["status"], job["error"]
    if job["status"] != "done":
        return out
    audit = bm.AUDITS[job["audit_id"]]
    state = audit["state"]
    payload = bm.dashboard_payload(audit)
    out["quality"] = jsonable(payload["quality"])
    out["policy"] = jsonable(payload["policy"])
    out["impact"] = jsonable(payload["impact"])
    out["calibration"] = jsonable({k: v for k, v in payload["calibration"].items() if k != "impact_curve"})
    out["selection"] = jsonable(payload["selection"])
    out["summary"] = jsonable(payload["summary"])
    out["review"] = jsonable(payload["review"])
    out["notes"] = jsonable(state.meta.get("notes"))
    out["n_rows_analysed"] = int(state.meta["n_rows_analysed"])
    out["limitations"] = jsonable(payload["limitations"])
    # V2
    out["v2_mechanical"] = v2_mechanical(audit, df, spec["task"])
    texts = strings(payload)
    bad = ["F1" in t for t in texts] if spec["task"] == "regression" else ["R²" in t or "R-squared" in t for t in texts]
    out["v2_wrong_metric_strings"] = [t[:160] for t, b in zip(texts, bad) if b][:10]
    # V5 spot check material
    out["spot_check"] = spot_check(audit, df)
    # V3 injected truth: the calibration's own runs inject known errors into the file's reference rows
    sb = state.selfbench
    if sb is not None and sb.ok:
        runs = sb.runs
        r5 = runs[runs.rate == 0.05].groupby("issue")[["detector_precision", "detector_recall", "damage", "recovery"]].mean()
        out["v3_detection_at_5pct"] = jsonable(r5.reset_index().to_dict("records"))
        out["v3_detector_false_positive_floor_on_reference"] = jsonable(sb.baseline_floors)
        out["v3_noise_floor"] = float(sb.noise_floor)
    # V6 panel
    panel = audit.get("panel") or {"status": "none"}
    t0 = time.time()
    while panel.get("status") in ("queued", "running") and time.time() - t0 < 1800:
        time.sleep(5)
        panel = audit["panel"]
    out["panel_seconds"] = round(time.time() - t0, 1)
    out["panel_status"] = panel.get("status")
    out["panel"] = jsonable(panel)
    out["peak_rss_mb"] = rss_mb()
    return out


def run_O() -> dict:
    spec = FILES["O"]
    path = os.path.join(NEW, spec["path"])
    out = {"id": "O", "file": spec["path"], "errors": []}
    raw = open(path, "rb").read()
    out["file_bytes"] = len(raw)
    t0 = time.time()
    df, info = load_table(raw, os.path.basename(path))
    out["load_seconds"] = round(time.time() - t0, 1)
    out["peak_rss_mb_after_load"] = rss_mb()
    out["input_format"] = jsonable(info)
    out["rows"], out["columns"] = int(len(df)), int(df.shape[1])
    out["dtypes"] = {c: str(t) for c, t in df.dtypes.items()}
    out["dates_are_dates"] = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    t0 = time.time()
    ds_id = bm.store_dataset(df, info, os.path.basename(path), raw=raw)
    out["store_seconds"] = round(time.time() - t0, 1)
    out["snapshot_ok"] = bm.DATASETS[ds_id]["snapshot"] is not None
    out["snapshot_error"] = bm.DATASETS[ds_id]["snapshot_error"]
    others = [s for s in info["sheets"] if s != info["sheet"]]
    out["sheets"] = info["sheets"]
    if others:
        t0 = time.time()
        try:
            other = bm.pick_sheet(ds_id, bm.SheetRequest(sheet=others[0]))
            out["other_sheet"] = {"sheet": others[0], "rows": other["rows"], "columns": [c["name"] for c in other["columns"]],
                                  "same_columns_as_first": [c["name"] for c in other["columns"]] == list(df.columns),
                                  "seconds": round(time.time() - t0, 1)}
        except Exception as e:                  # noqa: BLE001
            out["other_sheet"] = {"error": f"{type(e).__name__}: {e}"}
    req = bm.AuditRequest(dataset_id=ds_id, target=None, task="classification", calibrate=False, model_panel=False)
    jid = "valO"
    bm.JOBS[jid] = bm.JobDict(id=jid, status="queued", progress=0.0, message="", stage="queued", error=None, audit_id=None,
                              started=time.time(), finished=None, calibrate=False)
    t0 = time.time()
    bm.run_audit_job(jid, req)
    job = bm.JOBS[jid]
    out["audit_seconds"], out["job_status"], out["job_error"] = round(time.time() - t0, 1), job["status"], job["error"]
    if job["status"] == "done":
        audit = bm.AUDITS[job["audit_id"]]
        payload = bm.dashboard_payload(audit)
        out["quality"] = jsonable(payload["quality"])
        out["notes"] = jsonable(audit["state"].meta.get("notes"))
        out["n_rows_analysed"] = int(audit["state"].meta["n_rows_analysed"])
        out["raw_duplicate_rows"] = int(df.duplicated().sum())
        out["raw_missing_by_column"] = {c: int(n) for c, n in df.isna().sum().items() if n}
        out["spot_check"] = spot_check(audit, df.loc[audit["state"].X.index] if len(audit["state"].X) < len(df) else df)
    out["peak_rss_mb"] = rss_mb()
    return out


def count_rows_independently() -> dict:
    """V7: row counts read straight from the workbook with openpyxl, independent of intake.py."""
    import openpyxl
    path = os.path.join(NEW, FILES["O"]["path"])
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    counts = {}
    for ws in wb.worksheets:
        n = 0
        for row in ws.iter_rows(values_only=True):
            if any(v is not None for v in row):
                n += 1
        counts[ws.title] = n - 1                  # minus the header row
    return counts


def run_C() -> dict:
    path = os.path.join(NEW, FILES["C"]["path"])
    raw = open(path, "rb").read()
    out = {"id": "C", "file_bytes": len(raw)}
    try:
        load_table(raw, os.path.basename(path))
        out["refused"] = False
    except ValueError as e:
        out["refused"], out["message"] = True, str(e)
    except Exception as e:                      # noqa: BLE001
        out["refused"], out["message"], out["wrong_exception"] = False, f"{type(e).__name__}: {e}", True
    return out


if __name__ == "__main__":
    fid = sys.argv[1]
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    try:
        if fid == "O" and "--count" in sys.argv:
            res = {"id": "O_count", "openpyxl_rows_per_sheet": count_rows_independently()}
            fid = "O_count"
        else:
            res = (run_C() if fid == "C" else run_O() if fid == "O" else run_tabular(fid))
    except Exception as e:                      # noqa: BLE001  a crash is itself a finding: record it, never hide it
        res = {"id": fid, "crash": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()}
    res["wall_seconds"] = round(time.time() - t0, 1)
    with open(os.path.join(OUT, f"phase23_{fid}.json"), "w") as f:
        json.dump(jsonable(res), f, indent=1)
    print(f"{fid}: wrote results/phase23_{fid}.json in {res['wall_seconds']}s",
          "CRASH: " + res["crash"] if "crash" in res else "")
