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

from selfbench import NULL_DROP_FRACTION, run_self_benchmark  # noqa: E402
from taskclean import (ISSUES, MAX_MODEL_ROWS, PROBLEM_TEXT, apply_repairs, attach_self_benchmark,  # noqa: E402
                       audit_dataset, default_apply_issues, load_csv, outputs_as_bytes, outputs_zip)

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
        loaded, info = load_csv(uploaded.getvalue())
        st.session_state.update(df=loaded, input_info=info, source=f"upload:{uploaded.name}:{uploaded.size}",
                                name=uploaded.name, state=None, result=None)
    except Exception as e:  # malformed CSV is user input, not a bug
        st.error(f"Could not read that file as CSV: {e}")
if use_demo:
    loaded, info = load_csv(DEMO_PATH)
    st.session_state.update(df=loaded, input_info=info, source="demo", name="demo_dirty_adult.csv",
                            state=None, result=None)

df = st.session_state.get("df")
if df is None:
    st.info("Upload a CSV or try the demo dataset to begin.")
    st.stop()

info = st.session_state.get("input_info", {})
st.write(f"**{st.session_state['name']}** -- {len(df):,} rows x {df.shape[1]} columns")
if info:
    st.caption(f"Detected: {info['encoding']} text, '{info['delimiter']}' delimiter, "
               f"'{info['decimal']}' decimal mark.")
st.dataframe(df.head(10), width="stretch")

# ------------------------------------------------------------------ 2. configure
st.header("2. Target and task")
NO_TARGET = "(no target -- unlabeled dataset)"
cols = list(df.columns)
default_target = cols.index("class") + 1 if "class" in cols else len(cols)
c1, c2, c3 = st.columns([2, 2, 3])
with c1:
    chosen_target = st.selectbox("Target column", [NO_TARGET] + cols, index=default_target)
    target = None if chosen_target == NO_TARGET else chosen_target
with c2:
    task = st.selectbox("ML task", ["Classification", "Regression (not supported)"])
with c3:
    feat = st.checkbox(
        "Also run the feature-anomaly detector (slow)", value=False,
        help="One model per column. In the benchmark this detector had ~28% precision and its automatic repair "
             "was harmful, so it is off by default; if enabled, its findings are flag-only unless you override.")

calibrate = st.checkbox(
    "Calibrate on this dataset (self-benchmark)", value=False, disabled=target is None,
    help="Builds a clean-ish reference from your own data, injects errors into it, and measures how much each "
         "hurts a model and whether repair helps or harms -- then bases the repair policy on THOSE numbers "
         "instead of the UCI Adult benchmark. Adds roughly 20 seconds to a few minutes. Needs a target.")
with st.expander("Advanced"):
    seeds_n = st.radio("Calibration seeds", [3, 5], horizontal=True,
                       help="More seeds give steadier harm/benefit estimates but take longer.")
    max_rows = st.number_input(
        "Row limit for the slow model-based detectors", min_value=2_000, max_value=500_000,
        value=MAX_MODEL_ROWS, step=5_000,
        help="Above this many rows the label and feature detectors run on a stratified random sample; the cheap "
             "checks (missing values, duplicates, outliers) always use every row.")
if target is None:
    st.info("No target selected: missing values, duplicates, outliers and inconsistent values are checked; the "
            "label-error check needs a target and is skipped.")

if task != "Classification" and target is not None:
    st.error("Only classification is supported: the benchmark evidence behind every threshold and impact "
             "estimate was gathered on a classification task.")
    st.stop()
if len(df) > 50_000:
    st.warning("Large dataset: the model-based detectors may take several minutes.")

if st.button("Run audit", type="primary"):
    bar = st.progress(0.0, text="Starting")
    try:
        state = audit_dataset(df, target, include_feature_anomalies=feat, max_model_rows=int(max_rows),
                              progress=lambda f, m: bar.progress(min(f, 1.0), text=m),
                              input_info=st.session_state.get("input_info"))
        if calibrate and target is not None:
            bar.progress(0.0, text="Calibrating on your data")
            sb = run_self_benchmark(state, n_seeds=int(seeds_n),
                                    progress=lambda f, m: bar.progress(min(f, 1.0), text=f"Calibrating: {m}"))
            attach_self_benchmark(state, sb)
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
VIEWS = ["Overview", "Quality audit", "Calibration", "Readiness", "Task impact", "Repair plan", "Outputs"]


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
    m[2].metric("Target", ov["target"] or "none")
    m[3].metric("Classes", ov["n_classes"] if ov["n_classes"] is not None else "-")
    m[4].metric("Audit time", f"{state.meta['audit_seconds']:.0f}s")
    for note in ov["notes"]:
        st.info(note)
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
        "Detector precision": [pct(p, 0) for p in q.benchmark_precision],
        "Detector recall": [pct(r, 0) for r in q.benchmark_recall],
        "Measured on": q.reliability_source.fillna("-"),
        "Evidence": q.evidence_status, "Note": q.note})
    st.dataframe(show, width="stretch", hide_index=True)
    st.caption("Flagged rate is not a confirmed error rate. Detector precision/recall come from controlled "
               "experiments with known injected errors: on your own data if you calibrated, otherwise on UCI Adult.")

elif view == "Calibration":
    sb = state.selfbench
    if sb is None:
        st.info("Not calibrated: every decision currently uses the UCI Adult benchmark evidence. Tick "
                "'Calibrate on this dataset' (needs a target) and rerun the audit to replace it with measurements "
                "from your own data.")
    elif not sb.ok:
        st.warning(f"Calibration did not run: {sb.reason}. Decisions keep using the UCI Adult benchmark evidence.")
        if sb.reference:
            st.caption("Reference build: " + ", ".join(f"{k}: {v}" for k, v in sb.reference.items()
                                                       if k in ("rows_labeled", "rows_complete", "rows_after_dedup")))
    else:
        ref = sb.reference
        m = st.columns(4)
        m[0].metric("Reference rows", f"{ref['rows_used']:,}")
        m[1].metric("Clean-trained macro-F1", f"{np.mean(list(sb.baseline_f1.values())):.3f}")
        m[2].metric("Chance level", f"{sb.chance_f1:.3f}")
        m[3].metric("Noise floor", f"{sb.noise_floor:.4f}")
        st.caption(
            f"Built from your data: {ref['rows_labeled']:,} labeled rows -> {ref['rows_complete']:,} complete -> "
            f"{ref['rows_after_dedup']:,} without duplicates -> {ref['rows_used']:,} used "
            f"({ref['rows_train']:,} train / {ref['rows_test']:,} held-out test; the test rows are never corrupted). "
            f"The noise floor ({sb.noise_floor:.4f}) is the typical F1 change from randomly dropping "
            f"{NULL_DROP_FRACTION:.0%} of the training rows: effects smaller than that are treated as noise, so a "
            "repair only counts as harmful if it costs more than that. This is a clean-ish reference, not ground truth.")
        if ref["columns_left_out"]:
            st.write("Left out of the reference:", ref["columns_left_out"])
        if sb.skipped_issues:
            st.warning("Not measured: " + "; ".join(f"{k} ({v})" for k, v in sb.skipped_issues.items())
                       + ". These keep using the UCI Adult evidence.")
        st.subheader("Damage and repair evidence measured on your data")
        ev = sb.evidence
        st.dataframe(pd.DataFrame({
            "Error": ev.error_type, "Rate": [pct(r, 0) for r in ev.rate],
            "Damage (macro-F1)": [num(x) for x in ev.mean_damage],
            "Repair precision": [pct(x, 0) for x in ev.repair_precision],
            "Repair recall": [pct(x, 0) for x in ev.repair_recall],
            "Recovery": [num(x) for x in ev.mean_recovery], "Repair evidence": ev.repairability_status,
            "Material harm risk": [f"{x:.4f}" for x in ev.material_risk]}), width="stretch", hide_index=True)
        st.caption(f"Means over {sb.n_seeds} seeds. 'Damage' is how much each error lowers macro-F1 versus a model "
                   "trained on the clean reference; 'Recovery' is how much of that the real TaskClean repair wins "
                   "back; 'Material harm risk' counts only harm larger than the noise floor.")
        st.subheader("Does each detector's flagged rate track real damage on your data?")
        tr = sb.detector_tracking
        st.dataframe(pd.DataFrame({"Error": tr.error_type, "Evidence": tr.evidence_status, "Note": tr.note}),
                     width="stretch", hide_index=True)
        st.subheader("Detector false-positive floors on the reference")
        st.write({k: pct(v) for k, v in sb.baseline_floors.items()})
        st.caption("What each detector flags on data that is clean-ish by construction: an observed rate close to "
                   "this is not evidence of a problem.")

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
        "Impact estimate": [
            ("-" if pd.isna(e) else num(e) + ("  [boundary estimate]" if b else ""))
            for e, b in zip(imp.benchmark_impact_estimate, imp.is_boundary_estimate)],
        "Unit": imp.impact_unit, "Based on": imp.impact_source,
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
    sa = result.summary_aggressive
    st.write("TaskClean produces **two cleaned files**. Pick the one that matches how much risk you accept:")
    left, right = st.columns(2)
    with left:
        st.markdown("**Evidence-based** (`dataset_cleaned.csv`)")
        st.caption(f"Repairs applied using: {st.session_state.get('selection_label', 'policy default')}. Only "
                   "repairs that never showed harm in the benchmark; everything else is logged, not changed.")
        m = st.columns(3)
        m[0].metric("Rows", f"{s['rows_after_cleaning']:,}",
                    delta=f"-{s['rows_dropped']:,}" if s["rows_dropped"] else None, delta_color="off")
        m[1].metric("Cells modified", f"{s['cells_modified']:,}")
        m[2].metric("Flagged for review", f"{s['flagged_for_review']:,}")
    with right:
        st.markdown("**Aggressive** (`dataset_cleaned_aggressive.csv`)")
        st.caption("Every proposed repair except feature-anomaly repairs (harmful in the benchmark): missing values "
                   "imputed, outliers clipped, high-confidence label flips applied. Model-ready, but most of these "
                   "repairs did not clear the safety bar -- review the log before trusting it.")
        m = st.columns(3)
        m[0].metric("Rows", f"{sa['rows_after_cleaning']:,}",
                    delta=f"-{sa['rows_dropped']:,}" if sa["rows_dropped"] else None, delta_color="off")
        m[1].metric("Cells modified", f"{sa['cells_modified']:,}")
        m[2].metric("Left flag-only", f"{sa['flagged_only']:,}")
    if s["human_overrides"]:
        st.warning("Human overrides applied to the evidence-based file: " + ", ".join(s["human_overrides"]))
    if s["skipped_row_dropped"]:
        st.caption(f"{s['skipped_row_dropped']:,} cell repairs were skipped because their row was removed as a duplicate.")

    st.subheader("Repair log")
    which = st.radio("Log for", ["Evidence-based file", "Aggressive file"], horizontal=True)
    log = result.repair_log if which == "Evidence-based file" else result.repair_log_aggressive
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
