"""
TaskClean web API (FastAPI).

A thin HTTP layer over src/taskclean.py and src/selfbench.py. It adds NO analysis of its own: every number,
threshold and sentence comes from the research code, which is imported read-only. The React front end in
frontend/ talks to this API.

Run (from the project folder):
    .venv/bin/python -m uvicorn backend.main:app --port 8000

State is kept in memory (single local user): uploaded datasets, audit jobs, and finished audits.
"""
from __future__ import annotations

import json
import sys
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from selfbench import IMBALANCE_WARNING_SHARE, NULL_DROP_FRACTION, run_self_benchmark  # noqa: E402
from taskclean import (ISSUES, MAX_CAT_LEVELS, MAX_CERTIFIABLE_NOISE_FLOOR, MAX_CLASSES, MAX_MODEL_ROWS,  # noqa: E402
                       NEVER_AUTO_APPLY, PROBLEM_TEXT, SAFE_REPAIR_MIN_PRECISION, TASKCLEAN_VERSION, _sanitize,
                       apply_repairs, attach_self_benchmark, audit_dataset, default_apply_issues, load_csv,
                       outputs_as_bytes, outputs_zip)

DEMO_PATH = ROOT / "data" / "demo_dirty_adult.csv"
RESULTS = ROOT / "results"
DIST = ROOT / "frontend" / "dist"

app = FastAPI(title="TaskClean API", version=TASKCLEAN_VERSION)
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                   allow_methods=["*"], allow_headers=["*"])

DATASETS: dict[str, dict] = {}
AUDITS: dict[str, dict] = {}
JOBS: dict[str, dict] = {}
EXECUTOR = ThreadPoolExecutor(max_workers=1)      # the detectors use every core; run one audit at a time
LOCK = threading.Lock()


# ----------------------------------------------------------------------------------------------- helpers
def records(df: pd.DataFrame) -> list[dict]:
    """DataFrame -> JSON-safe list of dicts (NaN becomes null, numpy types become plain numbers)."""
    return json.loads(df.to_json(orient="records", date_format="iso"))


def safe(obj):
    return _sanitize(obj)


def get_dataset(dataset_id: str) -> dict:
    ds = DATASETS.get(dataset_id)
    if ds is None:
        raise HTTPException(404, "dataset not found (the server may have restarted: upload it again)")
    return ds


def get_audit(audit_id: str) -> dict:
    a = AUDITS.get(audit_id)
    if a is None:
        raise HTTPException(404, "audit not found (the server may have restarted: run it again)")
    return a


def column_kind(s: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(s):
        return "boolean"
    if pd.api.types.is_datetime64_any_dtype(s):
        return "datetime"
    if pd.api.types.is_numeric_dtype(s):
        return "numeric"
    return "text"


def dataset_payload(dataset_id: str) -> dict:
    ds = DATASETS[dataset_id]
    df: pd.DataFrame = ds["df"]
    cols = []
    for c in df.columns:
        s = df[c]
        n_unique = int(s.nunique(dropna=True))
        cols.append({"name": str(c), "kind": column_kind(s), "n_unique": n_unique,
                     "n_missing": int(s.isna().sum()), "target_candidate": 2 <= n_unique <= MAX_CLASSES,
                     "examples": [str(v) for v in s.dropna().unique()[:4]]})
    lowered = {str(c).lower(): str(c) for c in df.columns}
    suggested = next((lowered[k] for k in ("class", "target", "label", "y", "outcome") if k in lowered), None)
    if suggested is None:
        last = df.columns[-1]
        suggested = str(last) if 2 <= df[last].nunique(dropna=True) <= MAX_CLASSES else None
    head = df.head(10)
    return {"id": dataset_id, "name": ds["name"], "rows": int(len(df)), "n_columns": int(df.shape[1]),
            "input_format": safe(ds["info"]), "columns": cols, "suggested_target": suggested,
            "preview": json.loads(head.to_json(orient="records")), "demo": bool(ds.get("demo"))}


def store_dataset(df: pd.DataFrame, info: dict, name: str, demo: bool = False) -> str:
    dataset_id = uuid.uuid4().hex[:10]
    DATASETS[dataset_id] = {"df": df, "info": info, "name": name, "demo": demo}
    return dataset_id


# ----------------------------------------------------------------------------------------------- datasets
@app.get("/api/health")
def health():
    return {"ok": True, "version": TASKCLEAN_VERSION, "demo_available": DEMO_PATH.exists()}


@app.get("/api/config")
def config():
    return {"issues": ISSUES, "problem_text": PROBLEM_TEXT, "never_auto_apply": sorted(NEVER_AUTO_APPLY),
            "safe_repair_min_precision": SAFE_REPAIR_MIN_PRECISION,
            "max_certifiable_noise_floor": MAX_CERTIFIABLE_NOISE_FLOOR, "max_model_rows": MAX_MODEL_ROWS,
            "max_classes": MAX_CLASSES, "max_cat_levels": MAX_CAT_LEVELS,
            "null_drop_fraction": NULL_DROP_FRACTION, "imbalance_warning_share": IMBALANCE_WARNING_SHARE}


@app.post("/api/datasets/upload")
async def upload_dataset(file: UploadFile = File(...)):
    raw = await file.read()
    try:
        df, info = load_csv(raw)
    except Exception as e:                      # a malformed CSV is user input, not a server bug
        raise HTTPException(400, f"Could not read that file as CSV: {e}")
    return dataset_payload(store_dataset(df, info, file.filename or "upload.csv"))


@app.post("/api/datasets/demo")
def demo_dataset():
    if not DEMO_PATH.exists():
        raise HTTPException(404, "demo dataset missing (run src/make_demo_dataset.py)")
    df, info = load_csv(str(DEMO_PATH))
    return dataset_payload(store_dataset(df, info, "demo_dirty_adult.csv", demo=True))


@app.get("/api/datasets/{dataset_id}")
def read_dataset(dataset_id: str):
    get_dataset(dataset_id)
    return dataset_payload(dataset_id)


# ----------------------------------------------------------------------------------------------- audit jobs
class AuditRequest(BaseModel):
    dataset_id: str
    target: str | None = None
    include_feature_anomalies: bool = False
    include_joint_outliers: bool = False      # Phase 15: flag-only, off by default
    calibrate: bool = False
    seeds: int = Field(3, ge=2, le=5)
    max_model_rows: int = Field(MAX_MODEL_ROWS, ge=2000, le=500_000)


def run_audit_job(job_id: str, req: AuditRequest):
    job = JOBS[job_id]
    try:
        ds = get_dataset(req.dataset_id)
        df = ds["df"]
        job.update(status="running", message="Starting", stage="audit")
        share = 0.4 if (req.calibrate and req.target) else 1.0

        def on_audit(f: float, m: str):
            job.update(progress=min(f, 1.0) * share, message=m, stage="audit")

        state = audit_dataset(df, req.target, include_feature_anomalies=req.include_feature_anomalies,
                              include_joint_outliers=req.include_joint_outliers, max_model_rows=req.max_model_rows, progress=on_audit, input_info=ds["info"])
        if req.calibrate and req.target:
            def on_cal(f: float, m: str):
                job.update(progress=0.4 + 0.6 * min(f, 1.0), message=f"Calibrating: {m}", stage="calibration")

            sb = run_self_benchmark(state, n_seeds=req.seeds, progress=on_cal)
            attach_self_benchmark(state, sb)
        job.update(message="Applying the evidence-based policy", progress=0.99, stage="finalising")
        result = apply_repairs(state)
        audit_id = uuid.uuid4().hex[:10]
        applied = default_apply_issues(state)
        AUDITS[audit_id] = {"id": audit_id, "dataset_id": req.dataset_id, "dataset_name": ds["name"], "state": state,
                            "result": result, "applied": applied, "label": "policy default", "request": req.model_dump(),
                            "created": time.time()}
        job.update(status="done", progress=1.0, message="Audit complete", audit_id=audit_id, finished=time.time())
    except ValueError as e:                    # data the tool does not support: tell the user why
        job.update(status="error", error=str(e), message="Audit could not run", finished=time.time())
    except Exception as e:                     # noqa: BLE001
        traceback.print_exc()
        job.update(status="error", error=f"{type(e).__name__}: {e}", message="Audit failed", finished=time.time())


class JobDict(dict):
    def update(self, **kw):                    # keep writes simple and atomic enough for a single polling reader
        with LOCK:
            super().update(**kw)


@app.post("/api/audits")
def start_audit(req: AuditRequest):
    ds = get_dataset(req.dataset_id)
    if req.target is not None and req.target not in ds["df"].columns:
        raise HTTPException(400, f"target column '{req.target}' not found in the dataset")
    if req.calibrate and req.target is None:
        raise HTTPException(400, "calibration needs a target column")
    job_id = uuid.uuid4().hex[:10]
    JOBS[job_id] = JobDict(id=job_id, status="queued", progress=0.0, message="Queued", stage="queued", error=None,
                           audit_id=None, started=time.time(), finished=None, calibrate=req.calibrate)
    EXECUTOR.submit(run_audit_job, job_id, req)
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
def read_job(job_id: str):
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    with LOCK:
        out = dict(job)
    out["elapsed"] = round((out["finished"] or time.time()) - out["started"], 1)
    return out


# ----------------------------------------------------------------------------------------------- audit payloads
def calibration_payload(state) -> dict:
    sb = state.selfbench
    if sb is None:
        return {"status": "not_run"}
    if not sb.ok:
        return {"status": "declined", "reason": sb.reason, "reference": safe(sb.reference)}
    ref = safe(sb.reference)
    share = ref.get("minority_class_share")
    return {
        "status": "ok", "metric": sb.metric, "n_seeds": sb.n_seeds, "rates": list(sb.rates),
        "n_estimators": sb.n_estimators, "seconds": round(sb.seconds, 1), "reference": ref,
        "noise_floor": float(sb.noise_floor), "noise_floor_auc": float(sb.noise_floor_auc) if sb.has_auc else None,
        "has_auc": bool(sb.has_auc), "clean_macro_f1": float(np.mean(list(sb.baseline_f1.values()))),
        "chance_f1": float(sb.chance_f1), "minority_share": share,
        "imbalance_warning": bool(share is not None and share < IMBALANCE_WARNING_SHARE),
        "underpowered": bool(sb.evidence["underpowered"].any()),
        "evidence": records(sb.evidence), "impact_curve": records(sb.impact_curve),
        "detector_reliability": records(sb.detector_reliability), "detector_tracking": records(sb.detector_tracking),
        "baseline_floors": safe(sb.baseline_floors), "skipped_issues": safe(sb.skipped_issues),
        "metric_disagreement": records(sb.metric_disagreement) if sb.metric_disagreement is not None else [],
    }


REVIEW_ACTIONS = ("flagged_for_review", "flagged_only")


def review_summary(result) -> dict:
    """What a human still has to look at in the evidence-based file: proposed-but-not-applied repairs and flag-only rows."""
    log = result.repair_log
    by_problem = {v: k for k, v in PROBLEM_TEXT.items()}
    rows = []
    for problem, g in log[log.action.isin(REVIEW_ACTIONS)].groupby("problem"):
        rows.append({"issue_key": by_problem.get(problem, problem), "problem": problem,
                     "proposed": int((g.action == "flagged_for_review").sum()),
                     "flag_only": int((g.action == "flagged_only").sum())})
    return {"proposed": sum(r["proposed"] for r in rows), "flag_only": sum(r["flag_only"] for r in rows),
            "total": sum(r["proposed"] + r["flag_only"] for r in rows), "by_issue": rows}


def dashboard_payload(audit: dict) -> dict:
    state, result = audit["state"], audit["result"]
    rd = result.readiness
    files = outputs_as_bytes(result)
    default = default_apply_issues(state)
    return safe({
        "id": audit["id"], "dataset_id": audit["dataset_id"], "dataset_name": audit["dataset_name"],
        "request": audit["request"], "version": TASKCLEAN_VERSION,
        "overview": {**rd["dataset_overview"], "audit_seconds": state.meta["audit_seconds"],
                     "detector_seconds": state.meta["detector_seconds"],
                     "generated_at": rd["generated_at_utc"]},
        "quality": records(result.quality_report),
        "policy": records(state.policy),
        "impact": records(result.impact_report),
        "impact_note": rd["impact_estimate_note"],
        "readiness": rd["dimensions"],
        "limitations": rd["limitations"],
        "calibration": calibration_payload(state),
        "selection": {"applied": sorted(audit["applied"]), "default": sorted(default), "label": audit["label"]},
        "summary": {"evidence_based": result.summary, "aggressive": result.summary_aggressive},
        "review": review_summary(result),
        "features": records(result.feature_report) if result.feature_report is not None else None,
        "files": [{"name": n, "bytes": len(b)} for n, b in files.items()],
        "never_auto_apply": sorted(NEVER_AUTO_APPLY),
    })


@app.get("/api/audits/{audit_id}")
def read_audit(audit_id: str):
    return dashboard_payload(get_audit(audit_id))


class ApplyRequest(BaseModel):
    issues: list[str]


@app.post("/api/audits/{audit_id}/apply")
def apply_selection(audit_id: str, req: ApplyRequest):
    audit = get_audit(audit_id)
    unknown = [i for i in req.issues if i not in ISSUES]
    if unknown:
        raise HTTPException(400, f"unknown issue type(s): {', '.join(unknown)}")
    selection = set(req.issues)
    default = default_apply_issues(audit["state"])
    audit["result"] = apply_repairs(audit["state"], selection)
    audit["applied"] = selection
    audit["label"] = "policy default" if selection == default else \
        "custom selection (" + (", ".join(sorted(selection)) or "none") + ")"
    return dashboard_payload(audit)


@app.get("/api/audits/{audit_id}/repair-log")
def repair_log(audit_id: str, variant: str = Query("evidence", pattern="^(evidence|aggressive)$"),
               action: str | None = None, issue: str | None = None, page: int = Query(1, ge=1),
               page_size: int = Query(50, ge=10, le=500)):
    result = get_audit(audit_id)["result"]
    log = result.repair_log if variant == "evidence" else result.repair_log_aggressive
    view = log
    if action:
        view = view[view.action.isin(action.split(","))]
    if issue:
        if issue not in PROBLEM_TEXT:
            raise HTTPException(400, "unknown issue")
        view = view[view.problem == PROBLEM_TEXT[issue]]
    total = int(len(view))
    chunk = view.iloc[(page - 1) * page_size: page * page_size]
    return {"total": total, "page": page, "page_size": page_size, "rows": records(chunk),
            "actions": safe(log.action.value_counts().to_dict()), "problems": safe(log.problem.value_counts().to_dict())}


@app.get("/api/audits/{audit_id}/review.csv")
def review_csv(audit_id: str, variant: str = Query("evidence", pattern="^(evidence|aggressive)$")):
    """Only the rows a human still has to review (flagged, not applied), as a standalone worklist."""
    result = get_audit(audit_id)["result"]
    log = result.repair_log if variant == "evidence" else result.repair_log_aggressive
    work = log[log.action.isin(REVIEW_ACTIONS)]
    return Response(work.to_csv(index=False), media_type="text/csv",
                    headers={"Content-Disposition": 'attachment; filename="review_queue.csv"'})


MEDIA = {"csv": "text/csv", "json": "application/json"}


@app.get("/api/audits/{audit_id}/download/{name}")
def download_file(audit_id: str, name: str):
    files = outputs_as_bytes(get_audit(audit_id)["result"])
    if name not in files:
        raise HTTPException(404, "unknown output file")
    return Response(files[name], media_type=MEDIA[name.rsplit(".", 1)[1]],
                    headers={"Content-Disposition": f'attachment; filename="{name}"'})


@app.get("/api/audits/{audit_id}/download.zip")
def download_zip(audit_id: str):
    return Response(outputs_zip(get_audit(audit_id)["result"]), media_type="application/zip",
                    headers={"Content-Disposition": 'attachment; filename="taskclean_output.zip"'})


# ----------------------------------------------------------------------------------------------- research results
def _read(name: str):
    p = RESULTS / name
    return records(pd.read_csv(p, low_memory=False)) if p.exists() else []


@app.get("/api/benchmark")
def benchmark():
    """The frozen UCI Adult research results the policy falls back on, plus the Bank Marketing validation."""
    base = "phase13_bank_marketing_groundtruth_evidence_"
    baseline_rates = {}
    if (RESULTS / "phase11_baseline_rates.json").exists():
        baseline_rates = json.loads((RESULTS / "phase11_baseline_rates.json").read_text())
    return safe({
        "baseline": _read("phase2_clean_baseline.csv"), "impact": _read("phase7_impact_summary_full.csv"),
        "detection": _read("phase5_detection_scores.csv"), "repair": _read("phase9_repair_summary_full.csv"),
        "evidence": _read("phase10_evidence_table.csv"), "priority": _read("phase10_priority_headline.csv"),
        "sensitivity": _read("phase10_sensitivity_all_rates.csv"), "tracking": _read("phase11b_validation.csv"),
        "correlation": _read("phase11b_correlation.csv"), "baseline_rates": baseline_rates,
        "bank": {"posf1": _read(base + "posf1.csv"), "macrof1": _read(base + "macrof1.csv"), "auc": _read(base + "auc.csv"),
                 "decisions": _read("phase13_bank_marketing_comparison_summary.csv"),
                 "damage_corr": _read("phase13_bank_marketing_comparison_damage_corr.csv"),
                 "regimes": _read("phase13_regime_checks.csv")},
    })


# ----------------------------------------------------------------------------------------------- built front end
if DIST.exists():
    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str):
        if path.startswith("api/"):
            raise HTTPException(404)
        target = (DIST / path).resolve()
        if path and target.is_file() and DIST in target.parents:
            return FileResponse(target)
        return FileResponse(DIST / "index.html")
