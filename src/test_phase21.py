"""
Invariants for regression as a second ML task (Phase 21; docs/PHASE21_SPEC.md).
Run:  .venv/bin/python src/test_phase21.py

R1 target handling, R2 the target-error detector on clean and injected data, R3 no silent repair, R4 split integrity,
R5 the clean-file outcome (as amended, A2), R6 honest labels, C5 tuned selection reads only dirty training rows, R8 end to end
through the backend. Every check can fail. R0 (classification unchanged) is the other eight suites passing as they are.
"""
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.abspath(ROOT))

import controlled_proof as cp
import controlled_proof_regression as cpr
import model_panel_regression as mpr
from corruption.target_errors import corrupt as corrupt_target
from regression import detect_target_errors, prepare_target, suggest_task
from taskclean import _prepare, apply_repairs, attach_self_benchmark, audit_dataset

CH = os.path.join(ROOT, "data", "california_housing.csv")


def housing(n=2500, seed=7):
    return pd.read_csv(CH).sample(n, random_state=seed).reset_index(drop=True)


# ----------------------------------------------------------------------------------------------------- R1 target handling
def test_r1_target_handling():
    df = housing(300)
    X, y, *_ = _prepare(df, "MedHouseVal", "regression")
    assert y.dtype == float and y.notna().all() and len(y) == len(df)
    bad = df.copy()
    bad["MedHouseVal"] = bad["MedHouseVal"].astype(object)
    bad.loc[:4, "MedHouseVal"] = "n/a"
    yv, notes = prepare_target(bad, "MedHouseVal")
    assert yv.isna().sum() == 5 and any("not numbers" in n for n in notes), notes
    for broken, msg in ((df.assign(MedHouseVal=1.0), "constant"), (df.assign(MedHouseVal="abc"), "no numeric")):
        try:
            prepare_target(broken, "MedHouseVal")
        except ValueError as e:
            assert msg in str(e), e
        else:
            raise AssertionError("an unusable target must be a clear error")
    assert suggest_task(df["MedHouseVal"])["task"] == "regression"
    assert suggest_task(pd.Series(["a", "b", "a", "b"] * 20))["task"] == "classification"
    assert suggest_task(pd.Series(np.arange(5).tolist() * 30))["task"] == "classification"      # few distinct numbers: classes
    try:                                                                                            # classification still rejects it
        _prepare(df, "MedHouseVal", "classification")
    except ValueError as e:
        assert "choose the regression task" in str(e)
    else:
        raise AssertionError("a continuous target must be rejected under classification, with a pointer to regression")
    print("OK R1: numeric target accepted, non-numeric entries counted, constant/text targets are clear errors, task suggested")


# ------------------------------------------------------------------------------------------------------- R2 the detector
def test_r2_target_error_detector():
    from sklearn.datasets import load_diabetes
    df = housing(8000)
    X, y = df.drop(columns="MedHouseVal"), df["MedHouseVal"]
    clean = detect_target_errors(X, y, seed=42)
    assert clean.rate <= 0.02, f"R2: a clean file must flag at most 2% of rows, got {clean.rate:.4f}"
    d = load_diabetes(as_frame=True)
    assert detect_target_errors(d.data, d.target, seed=42).rate <= 0.02, "R2 on a second, small clean dataset"
    Xd, yd, log = corrupt_target(X, y, 0.05, seed=1)
    det = detect_target_errors(Xd, yd, seed=42)
    truth, got = set(log.row_id), set(det.row_mask[det.row_mask].index)
    precision, recall = len(truth & got) / max(len(got), 1), len(truth & got) / len(truth)
    assert precision > 0.5 and recall > 0.3, f"the detector must find injected target errors (precision {precision:.2f}, recall {recall:.2f})"
    print(f"OK R2: clean flag rate {clean.rate:.2%}; at 5% injected errors precision {precision:.2f}, recall {recall:.2f}")


# ------------------------------------------------------------------------------------------------- R3 no silent repair
def test_r3_no_repair_is_applied_silently():
    df = housing(2500)
    X, y = df.drop(columns="MedHouseVal"), df["MedHouseVal"]
    Xd, yd, _ = corrupt_target(X, y, 0.08, seed=3)
    d = Xd.copy()
    d["MedHouseVal"] = yd
    st = audit_dataset(d, "MedHouseVal", task="regression")
    assert (st.policy.evidence_source == "-").all(), "regression must not borrow the Adult (classification) evidence"
    res = apply_repairs(st)
    assert res.summary["cells_modified"] == 0 and res.summary["rows_dropped"] == 0 and not res.summary["issues_applied"], \
        "without calibration nothing is applied automatically"
    assert res.cleaned.equals(st.df), "the default file must equal the upload when nothing is approved"
    from selfbench_regression import run_self_benchmark_regression
    sb = run_self_benchmark_regression(st, n_seeds=2, rates=(0.05, 0.20))
    assert sb.ok and sb.task == "regression" and sb.metric == "R-squared", sb.reason
    attach_self_benchmark(st, sb)
    row = st.policy.set_index("issue_key").loc["label_errors"]
    assert row.default_action != "auto-repair", "rewriting the target is never automatic, whatever the evidence says"
    print("OK R3: no Adult evidence, nothing applied without calibration, target rewrite never automatic after calibration")


# ---------------------------------------------------------------------------------------------------- R4 split integrity
def test_r4_split_integrity():
    df = housing(1500)
    st = audit_dataset(df, "MedHouseVal", task="regression")
    seen = []
    real = mpr.audit_dataset
    mpr.audit_dataset = lambda frame, *a, **k: (seen.append(frame.copy()), real(frame, *a, **k))[1]
    try:
        out = mpr.run_model_panel_regression(st, seeds=(42, 43), models=("ridge",))
    finally:
        mpr.audit_dataset = real
    assert out["status"] == "ok" and len(seen) == 2
    assert all(len(f) == out["train_rows"] for f in seen), "the audit must see exactly the training rows"
    assert out["train_rows"] + out["test_rows"] == out["rows_used"]
    first = mpr._record_hash(seen[0])
    a, b = mpr.run_model_panel_regression(st, seeds=(42,), models=("ridge",)), mpr.run_model_panel_regression(st, seeds=(42,), models=("ridge",))
    assert a["results"] == b["results"], "the panel must be deterministic"
    assert len(first) == out["train_rows"]
    print(f"OK R4: audit saw only the {out['train_rows']} training rows (test rows {out['test_rows']} never audited); deterministic")


# ------------------------------------------------------------- R5 the clean-file outcome (Amendment A2) and R6 labels
def test_r5_r6_clean_file_and_honest_labels():
    df = housing(2500, seed=11)
    st = audit_dataset(df, "MedHouseVal", task="regression")
    out = mpr.run_model_panel_regression(st, seeds=(42, 43, 44))
    assert out["status"] == "ok" and out["task"] == "regression" and "R-squared" in out["metric"]
    arms = {a["key"]: a for a in out["arms"]}
    for r in out["results"]:
        assert "delta_pct" not in r, "no percentage change is reported for regression deltas"
        if r["arm"] == "before":
            continue
        changed = arms.get(r["arm"], {}).get("cells_changed", 0) + arms.get(r["arm"], {}).get("rows_removed", 0)
        if r["verdict"] in ("improved", "worse"):
            assert changed > 0, f"{r['model']}/{r['arm']} reads {r['verdict']} although it changed nothing"
            assert r["seeds_agree"] >= 0.8 and abs(r["delta_score"]) > r["noise_floor"], "a verdict must clear the rule it claims"
        if changed == 0:
            assert r["verdict"] == "no change", f"{r['model']}/{r['arm']} changed nothing but reads {r['verdict']}"
    n_imp = sum(r["verdict"] == "improved" for r in out["results"])
    n_worse = sum(r["verdict"] == "worse" for r in out["results"])
    print(f"OK R5/R6 (A2): on a CLEAN file the panel still reports {n_imp} improved and {n_worse} worse arm(s), each only for an arm "
          "that changed cells, each beyond the floor with >= 80% seed agreement; no-op arms read 'no change'")


# --------------------------------------------------------------------- C5 tuned selection reads only dirty training rows
def test_c5_tuned_selection_reads_only_dirty_rows():
    dirty = cpr.inject(cpr.clean_frame().sample(1200, random_state=3).reset_index(drop=True), 42)
    audited, scored = [], []
    real_audit, real_fit, real_read = cpr.audit_dataset, cpr._fit_score, pd.read_csv
    cpr.audit_dataset = lambda df, *a, **k: (audited.append(len(df)), real_audit(df, *a, **k))[1]
    cpr._fit_score = lambda n, Xt, yt, Xe, ye, *a, **k: (scored.append((len(Xt), len(Xe))), real_fit(n, Xt, yt, Xe, ye, *a, **k))[1]

    def boom(*a, **k):
        raise AssertionError("selection must not read the clean data file")
    pd.read_csv = boom
    try:
        out = cpr.select_repair(dirty, ["ridge"], 42, [], True)
    finally:
        cpr.audit_dataset, cpr._fit_score, pd.read_csv = real_audit, real_fit, real_read
    assert audited and max(audited) <= int(len(dirty) * (1 - cp.VAL_FRACTION)) + 2
    assert scored and all(te <= int(len(dirty) * cp.VAL_FRACTION) + 2 for _, te in scored)
    assert out["ridge"]["arm"] in out["ridge"]["val"]
    print(f"OK C5: tuned selection audited {audited[0]} rows, scored on <= {max(te for _, te in scored)} validation rows, never read the clean file")


def test_summarise_accepts_regression_labels():
    models = ["ridge", "random_forest"]
    rows, nulls, choices = [], [], []
    for seed in (1, 2, 3):
        for m in models:
            for arm in cp.ARM_ORDER:
                f1 = {"clean": 0.80, "dirty": 0.70}.get(arm, 0.70)
                if arm in ("tuned", "tuned_v0") and m == "ridge":
                    f1 = 0.73
                rows.append({"seed": seed, "model": m, "arm": arm, "same_as_dirty": arm in ("dirty", "default") or (arm.startswith("tuned") and m != "ridge"),
                             "f1": f1 + 0.0005 * seed, "auc": 0.3, "acc": np.nan})
            nulls.append({"seed": seed, "model": m, "d_f1": 0.0005})
            choices.append({"seed": seed, "model": m, "variant": "tuned", "chosen": "dirty", "best_candidate": "outliers", "best_val_gain": 0.0})
            choices.append({"seed": seed, "model": m, "variant": "tuned_v0", "chosen": "dirty", "best_candidate": "outliers", "best_val_gain": 0.0})
    out = cp.summarise(pd.DataFrame(rows), pd.DataFrame(nulls), pd.DataFrame(choices), [1, 2, 3], models, set(), 1.0,
                       labels=mpr.MODEL_LABELS, metric="R-squared on CLEAN held-out rows", extra={"task": "regression"})
    assert out["task"] == "regression" and out["metric"].startswith("R-squared") and out["models"][0]["label"] == "Ridge regression"
    assert next(c for c in out["results"] if c["model"] == "ridge" and c["arm"] == "tuned")["verdict"] == "improved"
    print("OK summary logic is shared: regression labels and metric flow through")


# -------------------------------------------------------------------------------------------------------- R8 end to end
def test_r8_backend_end_to_end():
    import backend.main as bm
    from intake import load_table
    df = housing(2200)
    buf = df.to_csv(index=False).encode()
    parsed, info = load_table(buf, "housing.csv")
    ds_id = bm.store_dataset(parsed, info, "housing.csv", raw=buf)
    health = bm.read_health(ds_id, "MedHouseVal", "regression")
    assert health["target"]["task"] == "regression" and health["suggested_task"]["task"] == "regression"
    req = bm.AuditRequest(dataset_id=ds_id, target="MedHouseVal", task="regression", model_panel=False, calibrate=True, seeds=2)
    job_id = "t21"
    bm.JOBS[job_id] = bm.JobDict(id=job_id, status="queued", progress=0.0, message="", stage="", error=None, audit_id=None,
                                 started=time.time(), finished=None, calibrate=True)
    bm.run_audit_job(job_id, req)
    assert bm.JOBS[job_id]["status"] == "done", bm.JOBS[job_id].get("error")
    aid = bm.JOBS[job_id]["audit_id"]
    dash = bm.dashboard_payload(bm.AUDITS[aid])
    assert dash["overview"]["task"] == "regression" and dash["calibration"]["status"] == "ok" and dash["calibration"]["metric"] == "R-squared"
    assert {r["impact_unit"] for r in dash["impact"]} <= {"R-squared", "-"}

    def strings(o):
        if isinstance(o, str):
            yield o
        elif isinstance(o, dict):
            for v in o.values():
                yield from strings(v)
        elif isinstance(o, (list, tuple)):
            for v in o:
                yield from strings(v)
    mentions = [s for s in strings({k: dash[k] for k in ("overview", "quality", "policy", "impact", "impact_note", "limitations")})
                if "F1" in s or "macro" in s.lower()]
    assert not mentions, f"a regression audit must not talk about F1: {mentions[:2]}"
    bm.AUDITS[aid]["panel"] = {"status": "queued", "progress": 0.0}
    bm.run_panel_job(aid, 2)
    panel = bm.read_panel(aid)
    assert panel["status"] == "ok" and panel["task"] == "regression"
    flat = pd.read_csv(__import__("io").BytesIO(bm.audit_files(bm.AUDITS[aid])["model_panel.csv"]))
    assert "score_mean" in flat.columns and len(flat) == len(panel["results"])
    assert bm.repair_log(aid, "evidence", None, "label_errors", 1, 10)["total"] > 0
    print("OK R8: regression upload -> health, audit, calibration, ranked impact in R-squared, panel, downloads; no F1 wording")


def test_backend_serves_the_saved_regression_proof():
    import json
    import backend.main as bm
    saved = os.path.join(ROOT, "results", "phase21_proof.json")
    bm.PROOF["live_regression"] = None
    out = bm.read_proof("regression")
    if os.path.exists(saved):
        want = json.load(open(saved))
        assert out["status"] == "ok" and out["task"] == "regression" and out["source"].startswith("saved") and out["results"] == want["results"]
        assert bm.read_proof("classification").get("task") != "regression", "the two tasks keep separate proofs"
    else:
        assert out["status"] == "none"
    print("OK backend: /api/proof?task=regression serves the saved registered run, separately from classification")


if __name__ == "__main__":
    tests = [v for n, v in sorted(globals().items()) if n.startswith("test_") and callable(v)]
    for t in tests:
        t()
    print(f"\nALL PHASE 21 REGRESSION INVARIANTS HOLD ({len(tests)} tests)")
