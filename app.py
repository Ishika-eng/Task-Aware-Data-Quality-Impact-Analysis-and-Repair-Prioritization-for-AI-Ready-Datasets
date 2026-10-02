"""
TaskClean -- Streamlit front end.

    streamlit run app.py

Upload a CSV, pick the target, run the audit, review the evidence-based
repair plan, and download the cleaned dataset with its repair log and
reports. All logic lives in src/taskclean.py; this file is only the UI.
"""

import os
import sys

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from taskclean import (ISSUES, MAX_CLASSES, PROBLEM_TEXT, apply_repairs, audit_dataset,  # noqa: E402
                       default_apply_issues, outputs_as_bytes, outputs_zip)

DEMO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "demo_dirty_adult.csv")

st.set_page_config(page_title="TaskClean", page_icon=None, layout="wide")


def pct(x, digits=1):
    return "-" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x * 100:.{digits}f}%"


def num(x, digits=4):
    return "-" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:+.{digits}f}"


# ------------------------------------------------------------------ header
st.title("TaskClean")
st.caption("Task-aware data quality audit, evidence-based repair, and a multidimensional readiness report.")
with st.expander("How to read this tool"):
    st.markdown(
        "- **A flagged rate is not a confirmed error rate.** Statistical detectors have known false-positive and "
        "false-negative behaviour, measured in a controlled benchmark (UCI Adult, Random Forest).\n"
        "- **Detected does not mean safe to repair.** Only repair types that never showed harm and had high repair "
        "precision in that benchmark are applied automatically; everything else is flagged for human review and "
        "recorded in the repair log without being changed.\n"
        "- **Impact numbers are benchmark-based estimates**, not predictions for your dataset or model.\n"
        "- **There is no single readiness score**: the evidence did not justify collapsing the dimensions into one "
        "number. Each dimension carries its own evidence status.")

# ------------------------------------------------------------------ 1. dataset
st.header("1. Dataset")
c1, c2 = st.columns([3, 2])
with c1:
    uploaded = st.file_uploader("Upload a CSV file", type=["csv"])
with c2:
    st.write("")
    st.write("")
    use_demo = st.button("Use demo dataset (Adult, 6,300 rows, errors injected)")

if uploaded is not None and st.session_state.get("source") != f"upload:{uploaded.name}:{uploaded.size}":
    try:
        st.session_state.update(df=pd.read_csv(uploaded), source=f"upload:{uploaded.name}:{uploaded.size}",
                                name=uploaded.name, state=None, result=None)
    except Exception as e:  # malformed CSV is user input, not a bug
        st.error(f"Could not read that file as CSV: {e}")
if use_demo:
    st.session_state.update(df=pd.read_csv(DEMO_PATH), source="demo", name="demo_dirty_adult.csv",
                            state=None, result=None)

df = st.session_state.get("df")
if df is None:
    st.info("Upload a CSV or try the demo dataset to begin.")
    st.stop()

st.write(f"**{st.session_state['name']}** -- {len(df):,} rows x {df.shape[1]} columns")
st.dataframe(df.head(10), width="stretch")

# ------------------------------------------------------------------ 2. configure
st.header("2. Target and task")
cols = list(df.columns)
default_target = cols.index("class") if "class" in cols else len(cols) - 1
c1, c2, c3 = st.columns([2, 2, 3])
with c1:
    target = st.selectbox("Target column", cols, index=default_target)
with c2:
    task = st.selectbox("ML task", ["Classification", "Regression (not supported)"])
with c3:
    feat = st.checkbox(
        "Also run the feature-anomaly detector (slow)", value=False,
        help="One model per column. In the benchmark this detector had ~28% precision and its automatic repair "
             "was harmful, so it is off by default; if enabled, its findings are flag-only unless you override.")

if task != "Classification":
    st.error("Only classification is supported: the benchmark evidence behind every threshold and impact "
             "estimate was gathered on a classification task.")
    st.stop()
if len(df) > 50_000:
    st.warning("Large dataset: the model-based detectors may take several minutes.")

if st.button("Run audit", type="primary"):
    bar = st.progress(0.0, text="Starting")
    try:
        state = audit_dataset(df, target, include_feature_anomalies=feat,
                              progress=lambda f, m: bar.progress(min(f, 1.0), text=m))
        st.session_state["state"] = state
        st.session_state["audit_id"] = st.session_state.get("audit_id", 0) + 1
        st.session_state["result"] = apply_repairs(state)          # evidence-based default policy
        st.session_state["applied_set"] = default_apply_issues(state)
        st.session_state["selection_label"] = "policy default"
        st.session_state["view"] = "Overview"
        bar.empty()
    except ValueError as e:
        bar.empty()
        st.error(str(e))

state = st.session_state.get("state")
result = st.session_state.get("result")
if state is None or result is None:
    st.stop()

# ------------------------------------------------------------------ 3. results
VIEWS = ["Overview", "Quality audit", "Readiness", "Task impact", "Repair plan", "Outputs"]


def apply_selection():
    """Button callback (runs before the rerun, so it may set the widget-backed
    `view` key): apply exactly the checked repair types, then jump to Outputs."""
    st_state = st.session_state["state"]
    aid = st.session_state["audit_id"]
    selection = {i for i in ISSUES if st.session_state.get(f"apply_{aid}_{i}", False)}
    default = default_apply_issues(st_state)
    st.session_state["result"] = apply_repairs(st_state, selection)
    st.session_state["applied_set"] = selection
    st.session_state["selection_label"] = ("policy default" if selection == default else
                                           "custom selection (" + (", ".join(sorted(selection)) or "none") + ")")
    st.session_state["view"] = "Outputs"


st.header("3. Results")
# A radio rather than st.tabs: st.tabs snaps back to the first tab on every
# rerun, which made ticking a checkbox in "Repair plan" jump to "Overview".
view = st.radio("View", VIEWS, horizontal=True, key="view", label_visibility="collapsed")

if view == "Overview":
    ov = result.readiness["dataset_overview"]
    m = st.columns(5)
    m[0].metric("Rows analysed", f"{ov['rows_analysed']:,}")
    m[1].metric("Columns", ov["columns"])
    m[2].metric("Target", ov["target"])
    m[3].metric("Classes", ov["n_classes"])
    m[4].metric("Audit time", f"{state.meta['audit_seconds']:.0f}s")
    if ov["rows_missing_target_excluded_from_analysis"]:
        st.warning(f"{ov['rows_missing_target_excluded_from_analysis']:,} rows have no target value and were "
                   "excluded from the analysis (they are left untouched in the output).")
    if ov["columns_not_analysed"]:
        st.write("Columns not analysed:", ov["columns_not_analysed"])
    if ov["columns_skipped_by_model_based_detectors"]:
        st.write("Skipped by the model-based detectors (too many distinct values):",
                 ov["columns_skipped_by_model_based_detectors"])
    st.caption("Model used for the benchmark validation: " + ov["model_used_for_benchmark_validation"])

elif view == "Quality audit":
    q = result.quality_report.copy()
    show = pd.DataFrame({
        "Dimension": q.dimension, "Detector": q.detector_type,
        "Flagged": [("-" if pd.isna(c) else f"{int(c):,} {u}") for c, u in zip(q.detected_count, q.count_unit)],
        "Rate": [pct(r) for r in q.detected_rate],
        "Benchmark precision": [pct(p, 0) for p in q.benchmark_precision],
        "Benchmark recall": [pct(r, 0) for r in q.benchmark_recall],
        "Evidence": q.evidence_status, "Note": q.note})
    st.dataframe(show, width="stretch", hide_index=True)
    st.caption("Flagged rate is not a confirmed error rate. Benchmark precision/recall are from the controlled "
               "Phase 5 experiment (known corruption injected into clean Adult data).")

elif view == "Readiness":
    dims = pd.DataFrame(result.readiness["dimensions"])
    show = pd.DataFrame({
        "Dimension": dims.dimension, "Observed": [pct(x) for x in dims.observed_rate],
        "False-positive floor": [pct(x) for x in dims.baseline_false_positive_floor],
        "After cleaning": [pct(x) for x in dims.rate_after_cleaning],
        "Evidence": dims.evidence_status, "Status": dims.status})
    st.dataframe(show, width="stretch", hide_index=True)
    st.caption("'After cleaning' is recomputed only for the deterministic detectors (missing, duplicates, "
               "outliers, consistency); model-based detectors are not re-run. The false-positive floor for "
               "statistical detectors comes from clean Adult data and is not recalibrated for your dataset.")
    with st.expander("Evidence notes per dimension"):
        for d in result.readiness["dimensions"]:
            st.markdown(f"**{d['dimension']}** ({d['evidence_status']}): {d['evidence_note']}")
    with st.expander("Limitations"):
        for item in result.readiness["limitations"]:
            st.markdown(f"- {item}")

elif view == "Task impact":
    imp = result.impact_report.copy()
    show = pd.DataFrame({
        "Issue": imp.issue, "Detected rate": [pct(x) for x in imp.detected_rate],
        "Benchmark impact estimate (F1)": [
            ("-" if pd.isna(e) else num(e) + ("  [boundary estimate]" if b else ""))
            for e, b in zip(imp.benchmark_impact_estimate, imp.is_boundary_estimate)],
        "Repair evidence": imp.repairability_status, "Decision": imp.policy_decision,
        "Recommendation": imp.recommendation})
    st.dataframe(show, width="stretch", hide_index=True)
    st.caption(result.readiness["impact_estimate_note"])

elif view == "Repair plan":
    st.subheader("Repair plan")
    st.write("Each issue type is either **auto-repaired** (the benchmark showed zero observed harm and high repair "
             "precision) or **flagged for human review** (proposals are logged but nothing is changed). You can "
             "override this; overrides are recorded in the repair log.")
    pol = state.policy
    default_set = default_apply_issues(state)
    applied_set = st.session_state.get("applied_set", default_set)
    audit_id = st.session_state["audit_id"]
    selection = set()
    for _, p in pol.iterrows():
        left, right = st.columns([2, 5])
        n_prop = int(p.n_repair_candidates)
        with left:
            checked = st.checkbox(f"Apply: {p.issue}", value=p.issue_key in applied_set, disabled=n_prop == 0,
                                  key=f"apply_{audit_id}_{p.issue_key}")
            if checked and n_prop > 0:
                selection.add(p.issue_key)
        with right:
            st.markdown(f"**{p.default_action}** -- {n_prop:,} repair candidate(s), {int(p.n_flag_only):,} flag-only")
            st.caption(p.reason)
            if p.repairability_status == "negative" and p.issue_key in selection:
                st.warning("Benchmark testing found this repair strategy net-harmful. This does not mean the error "
                           "is unrepairable -- only that this strategy made models worse in testing.")
            elif p.issue_key in selection and p.issue_key not in default_set:
                st.warning("Human override: the evidence does not support applying this automatically.")
    st.button("Apply selected repairs", on_click=apply_selection, type="primary")

elif view == "Outputs":
    s = result.summary
    st.caption(f"Repairs applied using: {st.session_state.get('selection_label', 'policy default')}")
    m = st.columns(4)
    m[0].metric("Rows", f"{s['rows_after_cleaning']:,}", delta=f"-{s['rows_dropped']:,}" if s["rows_dropped"] else None,
                delta_color="off")
    m[1].metric("Cells modified", f"{s['cells_modified']:,}")
    m[2].metric("Proposals flagged for review", f"{s['flagged_for_review']:,}")
    m[3].metric("Flag-only (low confidence)", f"{s['flagged_only']:,}")
    if s["human_overrides"]:
        st.warning("Human overrides applied: " + ", ".join(s["human_overrides"]))
    if s["skipped_row_dropped"]:
        st.caption(f"{s['skipped_row_dropped']:,} cell repairs were skipped because their row was removed as a duplicate.")

    st.subheader("Repair log")
    log = result.repair_log
    actions = sorted(log.action.unique())
    chosen = st.multiselect("Filter by action", actions, default=actions)
    st.dataframe(log[log.action.isin(chosen)].head(1000), width="stretch", hide_index=True)
    st.caption(f"Showing up to 1,000 of {int(log.action.isin(chosen).sum()):,} rows. row_id is the 0-based row "
               "position in the uploaded file.")

    st.subheader("Download")
    files = outputs_as_bytes(result)
    mimes = {"json": "application/json", "csv": "text/csv"}
    cols_dl = st.columns(3)
    for i, (name, data) in enumerate(files.items()):
        cols_dl[i % 3].download_button(name, data, file_name=name, mime=mimes[name.rsplit(".", 1)[1]],
                                       key=f"dl_{name}")
    st.download_button("Download everything (.zip)", outputs_zip(result), file_name="taskclean_output.zip",
                       mime="application/zip", type="primary")
