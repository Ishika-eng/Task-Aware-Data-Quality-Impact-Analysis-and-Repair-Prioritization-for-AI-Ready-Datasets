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

import io
import json
import sys
import threading
import time
import traceback
import uuid
import zipfile
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

from health_snapshot import build_snapshot, snapshot_rows, target_profile  # noqa: E402
from intake import SUPPORTED, load_table  # noqa: E402
from controlled_proof import SEEDS as PROOF_SEEDS, run_controlled_proof
from controlled_proof_regression import run_controlled_proof_regression  # noqa: E402
from model_panel import DEFAULT_SEEDS, panel_rows, run_model_panel  # noqa: E402
from model_panel_regression import panel_rows_regression, run_model_panel_regression  # noqa: E402
from regression import TASKS, suggest_task, target_profile_regression  # noqa: E402
from selfbench_regression import run_self_benchmark_regression  # noqa: E402
from selfbench import IMBALANCE_WARNING_SHARE, NULL_DROP_FRACTION, run_self_benchmark  # noqa: E402
from taskclean import (ISSUES, MAX_CAT_LEVELS, MAX_CERTIFIABLE_NOISE_FLOOR, MAX_CLASSES, MAX_MODEL_ROWS,  # noqa: E402
                       NEVER_AUTO_APPLY, PROBLEM_TEXT, PROBLEM_TEXT_REGRESSION, SAFE_REPAIR_MIN_PRECISION, TASKCLEAN_VERSION, _sanitize,
                       _clean_declarations, apply_repairs, attach_self_benchmark, audit_dataset, default_apply_issues, load_csv,
                       outputs_as_bytes)

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


def store_dataset(df: pd.DataFrame, info: dict, name: str, demo: bool = False, raw: bytes | None = None) -> str:
    """Store the upload and capture the baseline health report NOW, before anything touches the data. The snapshot is
    stored with the dataset and never recomputed. A failure to build it must never block the upload."""
    dataset_id = uuid.uuid4().hex[:10]
    snapshot, error = None, None
    try:
        snapshot = safe(build_snapshot(df, info, name, raw))
    except Exception as e:                      # noqa: BLE001
        traceback.print_exc()
        error = f"{type(e).__name__}: {e}"
    DATASETS[dataset_id] = {"df": df, "info": info, "name": name, "demo": demo, "snapshot": snapshot,
                            "snapshot_error": error,
                            "raw": raw if info.get("format") == "xlsx" else None}     # kept so another sheet can be read
    return dataset_id


# ----------------------------------------------------------------------------------------------- datasets
@app.get("/api/health")
def health():
    return {"ok": True, "version": TASKCLEAN_VERSION, "demo_available": DEMO_PATH.exists()}


@app.get("/api/config")
def config():
    return {"issues": ISSUES, "problem_text": PROBLEM_TEXT, "problem_text_regression": PROBLEM_TEXT_REGRESSION,
            "tasks": list(TASKS), "never_auto_apply": sorted(NEVER_AUTO_APPLY),
            "safe_repair_min_precision": SAFE_REPAIR_MIN_PRECISION,
            "max_certifiable_noise_floor": MAX_CERTIFIABLE_NOISE_FLOOR, "max_model_rows": MAX_MODEL_ROWS,
            "max_classes": MAX_CLASSES, "max_cat_levels": MAX_CAT_LEVELS,
            "null_drop_fraction": NULL_DROP_FRACTION, "imbalance_warning_share": IMBALANCE_WARNING_SHARE}


@app.post("/api/datasets/upload")
async def upload_dataset(file: UploadFile = File(...)):
    raw = await file.read()
    name = file.filename or "upload.csv"
    try:
        df, info = load_table(raw, name)
    except Exception as e:                      # a malformed file is user input, not a server bug
        raise HTTPException(400, f"Could not read that file ({', '.join(SUPPORTED)} are supported): {e}")
    return dataset_payload(store_dataset(df, info, name, raw=raw))


class SheetRequest(BaseModel):
    sheet: str


@app.post("/api/datasets/{dataset_id}/sheet")
def pick_sheet(dataset_id: str, req: SheetRequest):
    """Read another sheet of an uploaded workbook. It becomes a new dataset with its own baseline health report."""
    ds = get_dataset(dataset_id)
    if not ds.get("raw"):
        raise HTTPException(400, "this dataset is not an Excel workbook")
    try:
        df, info = load_table(ds["raw"], ds["name"], sheet=req.sheet)
    except Exception as e:                      # noqa: BLE001
        raise HTTPException(400, str(e))
    return dataset_payload(store_dataset(df, info, ds["name"], raw=ds["raw"]))


@app.post("/api/datasets/demo")
def demo_dataset():
    if not DEMO_PATH.exists():
        raise HTTPException(404, "demo dataset missing (run src/make_demo_dataset.py)")
    raw = DEMO_PATH.read_bytes()
    df, info = load_csv(raw)
    return dataset_payload(store_dataset(df, info, "demo_dirty_adult.csv", demo=True, raw=raw))


@app.get("/api/datasets/{dataset_id}")
def read_dataset(dataset_id: str):
    get_dataset(dataset_id)
    return dataset_payload(dataset_id)


@app.get("/api/datasets/{dataset_id}/health")
def read_health(dataset_id: str, target: str | None = None, task: str = "classification"):
    """The baseline health report captured at upload (never recomputed), plus, if a target is given, its profile."""
    ds = get_dataset(dataset_id)
    if ds.get("snapshot") is None:
        raise HTTPException(500, f"the baseline health report could not be built: {ds.get('snapshot_error')}")
    out = {"snapshot": ds["snapshot"], "target": None}
    if target is not None:
        if target not in ds["df"].columns:
            raise HTTPException(400, f"target column '{target}' not found in the dataset")
        if task not in TASKS:
            raise HTTPException(400, f"task must be one of {list(TASKS)}")
        out["target"] = safe(target_profile_regression(ds["df"], target) if task == "regression"
                             else target_profile(ds["df"], target))
        out["suggested_task"] = suggest_task(ds["df"][target])
    return out


# ----------------------------------------------------------------------------------------------- audit jobs
class AuditRequest(BaseModel):
    dataset_id: str
    target: str | None = None
    task: str = "classification"              # Phase 21: "classification" or "regression" (a numeric target)
    include_feature_anomalies: bool = False
    include_joint_outliers: bool = False      # Phase 15: flag-only, off by default
    calibrate: bool = False
    model_panel: bool = True                  # Phase 19: 4-model before/after on the user's own data (background, report-only)
    seeds: int = Field(3, ge=2, le=5)
    max_model_rows: int = Field(MAX_MODEL_ROWS, ge=2000, le=500_000)
    declared_missing: dict[str, list[str]] = {}   # Phase 26 F2: values the user says mean "missing", per column
    keep_values: list[str] = []                   # Phase 26 F2: columns whose automatic placeholder detection is switched off


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
                              include_joint_outliers=req.include_joint_outliers, max_model_rows=req.max_model_rows, progress=on_audit,
                              input_info=ds["info"], task=req.task,
                              declared_missing=req.declared_missing, keep_values=req.keep_values)
        if req.calibrate and req.target:
            def on_cal(f: float, m: str):
                job.update(progress=0.4 + 0.6 * min(f, 1.0), message=f"Calibrating: {m}", stage="calibration")

            sb = (run_self_benchmark_regression if req.task == "regression" else run_self_benchmark)(
                state, n_seeds=req.seeds, progress=on_cal)
            attach_self_benchmark(state, sb)
        job.update(message="Applying the evidence-based policy", progress=0.99, stage="finalising")
        result = apply_repairs(state)
        audit_id = uuid.uuid4().hex[:10]
        applied = default_apply_issues(state)
        AUDITS[audit_id] = {"id": audit_id, "dataset_id": req.dataset_id, "dataset_name": ds["name"], "state": state,
                            "result": result, "applied": applied, "label": "policy default", "request": req.model_dump(),
                            "created": time.time(), "baseline": ds.get("snapshot"),
                            "baseline_target": (safe(target_profile_regression(df, req.target) if req.task == "regression"
                                                 else target_profile(df, req.target)) if req.target else None)}
        job.update(status="done", progress=1.0, message="Audit complete", audit_id=audit_id, finished=time.time())
        if req.model_panel and req.target:
            queue_panel(audit_id, req.seeds)
    except ValueError as e:                    # data the tool does not support: tell the user why
        job.update(status="error", error=str(e), message="Audit could not run", finished=time.time())
    except Exception as e:                     # noqa: BLE001
        traceback.print_exc()
        job.update(status="error", error=f"{type(e).__name__}: {e}", message="Audit failed", finished=time.time())


def run_panel_job(audit_id: str, n_seeds: int):                # noqa: C901
    audit = AUDITS.get(audit_id)
    if audit is None:
        return
    panel = audit["panel"]

    def put(**kw):
        with LOCK:
            panel.update(**kw)

    def on(f: float, m: str):
        put(progress=round(f, 3), message=m)

    try:
        put(status="running", message="Starting", progress=0.0, started=time.time())
        seeds = tuple(range(42, 42 + max(2, min(5, n_seeds)))) if n_seeds != 3 else DEFAULT_SEEDS
        run = run_model_panel_regression if audit["state"].task == "regression" else run_model_panel
        result = run(audit["state"], seeds=seeds, progress=on)
        with LOCK:
            panel.clear()
            panel.update(result, progress=1.0, finished=time.time())
    except Exception as e:                     # noqa: BLE001
        traceback.print_exc()
        with LOCK:
            panel.clear()
            panel.update(status="error", reason=f"{type(e).__name__}: {e}", finished=time.time())


def panel_snapshot(audit: dict) -> dict:
    with LOCK:
        return dict(audit.get("panel") or {"status": "none"})


def queue_panel(audit_id: str, n_seeds: int):
    audit = AUDITS[audit_id]
    audit["panel"] = {"status": "queued", "progress": 0.0, "message": "Waiting for the audit to finish", "started": time.time()}
    EXECUTOR.submit(run_panel_job, audit_id, n_seeds)


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
    try:                                                          # Phase 26: a bad declaration is a 400, never silently ignored
        _clean_declarations(ds["df"], req.declared_missing, req.keep_values)
    except ValueError as e:
        raise HTTPException(400, str(e))
    if req.task not in TASKS:
        raise HTTPException(400, f"task must be one of {list(TASKS)}")
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
        "task": sb.task, "has_auc": bool(sb.has_auc), "clean_macro_f1": float(np.mean(list(sb.baseline_f1.values()))),
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
    by_problem = {v: k for k, v in {**PROBLEM_TEXT_REGRESSION, **PROBLEM_TEXT}.items()}
    rows = []
    for problem, g in log[log.action.isin(REVIEW_ACTIONS)].groupby("problem"):
        rows.append({"issue_key": by_problem.get(problem, problem), "problem": problem,
                     "proposed": int((g.action == "flagged_for_review").sum()),
                     "flag_only": int((g.action == "flagged_only").sum())})
    return {"proposed": sum(r["proposed"] for r in rows), "flag_only": sum(r["flag_only"] for r in rows),
            "total": sum(r["proposed"] + r["flag_only"] for r in rows), "by_issue": rows}


def held_back_summary(audit: dict) -> dict:
    """Phase 23 A1 + Phase 25, for the UI: per issue type, the proposals that stayed in the review queue although the type is
    auto-applied, so the Repair plan and Review queue can say why. `calibration` counts the ones in columns the calibration did not cover
    (too empty to benchmark); `codes` counts the cells read as missing only because they hold a code such as -1, which are never
    imputed automatically. The two never overlap (a column outside the calibration holds all of its proposals under `calibration`).
    Shape: {issue: {count, columns, calibration: {count, columns}, codes: {count, columns}}}; `count`/`columns` cover both."""
    state, result = audit["state"], audit["result"]
    sb = state.selfbench
    left = set(sb.reference.get("columns_left_out", {})) if (sb is not None and getattr(sb, "ok", False)) else set()
    code_rows = {c: set(r) for c, r in state.meta.get("code_cells", {}).items()}
    if not left and not code_rows:
        return {}
    texts = PROBLEM_TEXT_REGRESSION if state.task == "regression" else PROBLEM_TEXT
    log = result.repair_log
    out = {}
    for issue in audit["applied"] & default_apply_issues(state):          # a human override of the whole type applies everything
        if issue in ("label_errors", "duplicates") or issue not in texts:
            continue
        g = log[(log.problem == texts[issue]) & (log.action == "flagged_for_review")]
        cal = g[g.column.isin(left)]
        cod = g[~g.column.isin(left)]
        if issue == "missing_values" and code_rows:
            cod = cod[[r in code_rows.get(c, ()) for r, c in zip(cod.row_id, cod.column)]]
        else:
            cod = cod.iloc[0:0]
        if len(cal) or len(cod):
            both = pd.concat([cal, cod])
            out[issue] = {"count": int(len(both)),
                          "columns": {str(c): int(n) for c, n in both.column.value_counts().items()},
                          "calibration": {"count": int(len(cal)), "columns": {str(c): int(n) for c, n in cal.column.value_counts().items()}},
                          "codes": {"count": int(len(cod)), "columns": {str(c): int(n) for c, n in cod.column.value_counts().items()}}}
    return out


def dashboard_payload(audit: dict) -> dict:
    state, result = audit["state"], audit["result"]
    rd = result.readiness
    files = audit_files(audit)
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
        "baseline": audit.get("baseline"), "baseline_target": audit.get("baseline_target"),
        "selection": {"applied": sorted(audit["applied"]), "default": sorted(default), "label": audit["label"]},
        "summary": {"evidence_based": result.summary, "aggressive": result.summary_aggressive},
        "review": review_summary(result),
        "features": records(result.feature_report) if result.feature_report is not None else None,
        "files": [{"name": n, "bytes": len(b)} for n, b in files.items()],
        "never_auto_apply": sorted(NEVER_AUTO_APPLY),
        "held_back": held_back_summary(audit),
    })


@app.get("/api/audits/{audit_id}")
def read_audit(audit_id: str):
    return dashboard_payload(get_audit(audit_id))


@app.get("/api/audits/{audit_id}/panel")
def read_panel(audit_id: str):
    """The model panel: {status: none|queued|running|ok|declined|error, ...}. `ok` carries the full result."""
    return safe(panel_snapshot(get_audit(audit_id)))


class PanelRequest(BaseModel):
    seeds: int = Field(3, ge=2, le=5)


@app.post("/api/audits/{audit_id}/panel")
def start_panel(audit_id: str, req: PanelRequest):
    audit = get_audit(audit_id)
    if audit["state"].target is None:
        raise HTTPException(400, "the model panel needs a target column")
    if panel_snapshot(audit).get("status") in ("queued", "running"):
        raise HTTPException(409, "the model panel is already running")
    queue_panel(audit_id, req.seeds)
    return {"status": "queued"}


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
    audit = get_audit(audit_id)
    result = audit["result"]
    texts = PROBLEM_TEXT_REGRESSION if audit["state"].task == "regression" else PROBLEM_TEXT
    log = result.repair_log if variant == "evidence" else result.repair_log_aggressive
    view = log
    if action:
        view = view[view.action.isin(action.split(","))]
    if issue:
        if issue not in texts:
            raise HTTPException(400, "unknown issue")
        view = view[view.problem == texts[issue]]
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


def audit_files(audit: dict) -> dict[str, bytes]:
    """Every output file of an audit, plus the baseline health report captured at upload (JSON and flat CSV)."""
    files = outputs_as_bytes(audit["result"])
    if audit.get("baseline") is not None:
        files["baseline_health_report.json"] = json.dumps(
            {"baseline": audit["baseline"], "target": audit.get("baseline_target")}, indent=2).encode("utf-8")
        files["baseline_health_report.csv"] = snapshot_rows(audit["baseline"]).to_csv(index=False).encode("utf-8")
    panel = panel_snapshot(audit)
    if panel.get("status") == "ok":
        files["model_panel.json"] = json.dumps(safe(panel), indent=2).encode("utf-8")
        files["model_panel.csv"] = (panel_rows_regression if panel.get("task") == "regression" else panel_rows)(panel).to_csv(index=False).encode("utf-8")
    return files


@app.get("/api/audits/{audit_id}/download/{name}")
def download_file(audit_id: str, name: str):
    files = audit_files(get_audit(audit_id))
    if name not in files:
        raise HTTPException(404, "unknown output file")
    return Response(files[name], media_type=MEDIA[name.rsplit(".", 1)[1]],
                    headers={"Content-Disposition": f'attachment; filename="{name}"'})


@app.get("/api/audits/{audit_id}/download.zip")
def download_zip(audit_id: str):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in audit_files(get_audit(audit_id)).items():
            z.writestr(name, data)
    return Response(buf.getvalue(), media_type="application/zip",
                    headers={"Content-Disposition": 'attachment; filename="taskclean_output.zip"'})


# ----------------------------------------------------------------------------------------------- controlled proof (Phase 20)
PROOF: dict = {"live": None, "live_regression": None}


def _proof_key(task: str) -> str:
    return "live_regression" if task == "regression" else "live"


def _saved_proof(task: str = "classification"):
    p = RESULTS / ("phase21_proof.json" if task == "regression" else "phase20_proof.json")
    return json.loads(p.read_text()) if p.exists() else None


def run_proof_job(n_seeds: int, task: str = "classification"):
    key = _proof_key(task)

    def on(f: float, m: str):
        with LOCK:
            PROOF[key].update(progress=round(f, 3), message=m)

    try:
        run = run_controlled_proof_regression if task == "regression" else run_controlled_proof
        result = run(PROOF_SEEDS[:n_seeds], progress=on)
        with LOCK:
            PROOF[key] = {**result, "source": "live re-run"}
    except Exception as e:                      # noqa: BLE001
        traceback.print_exc()
        with LOCK:
            PROOF[key] = {"status": "error", "reason": f"{type(e).__name__}: {e}"}


@app.get("/api/proof")
def read_proof(task: str = "classification"):
    """The controlled proof (clean held-out set, known injected errors). A finished live re-run wins over the saved registered run."""
    if task not in TASKS:
        raise HTTPException(400, f"task must be one of {list(TASKS)}")
    with LOCK:
        live = dict(PROOF[_proof_key(task)]) if PROOF[_proof_key(task)] else None
    if live and live.get("status") in ("running", "queued", "error", "ok"):
        return safe(live)
    saved = _saved_proof(task)
    return safe({**saved, "source": "saved registered run"}) if saved else {"status": "none"}


class ProofRequest(BaseModel):
    seeds: int = Field(3, ge=2, le=5)
    task: str = "classification"


@app.post("/api/proof")
def start_proof(req: ProofRequest):
    if req.task not in TASKS:
        raise HTTPException(400, f"task must be one of {list(TASKS)}")
    key = _proof_key(req.task)
    with LOCK:
        if PROOF[key] and PROOF[key].get("status") in ("queued", "running"):
            raise HTTPException(409, "the controlled proof is already running")
        PROOF[key] = {"status": "queued", "progress": 0.0, "message": "Waiting for the previous job to finish"}
    EXECUTOR.submit(lambda: (PROOF[key].update(status="running", message="Starting"), run_proof_job(req.seeds, req.task)))
    return {"status": "queued"}


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
        "model_check": {"evidence": _read("phase16_gb_evidence_table.csv"), "priority": _read("phase16_gb_priority_headline.csv"),
                        "damage": _read("phase16_gb_damage_summary.csv"),
                        "criteria": json.loads((RESULTS / "phase16_gb_criteria.json").read_text())
                        if (RESULTS / "phase16_gb_criteria.json").exists() else {}},
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
