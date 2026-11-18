"""
Phase 25 validation: numeric placeholder codes read as missing values (docs/PHASE25_SPEC.md). No product code is changed here.

  python validate_sentinels.py D                 Uncleaned_DS_jobs.csv, Rating as regression target (S1, S4, S6)  -> results/phase25_D.json
  python validate_sentinels.py D --out PATH      the same, written to PATH (used to run the committed code in a git worktree = "before")
  python validate_sentinels.py S0 T|A|R|O|C      re-run a Phase 23 file and compare it with results/phase23_<id>.json (S0)
  python validate_sentinels.py S2                planted codes in clean Adult / California columns and in an Excel workbook (S2)
  python validate_sentinels.py S0demo            byte comparison of the cleaned files on demo_dirty_adult.csv (S0)

It calls the same backend functions the API calls (load_table, store_dataset, run_audit_job), so it is the product path.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import traceback

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path[:0] = [HERE, ROOT]

import validate_new_datasets as vnd              # noqa: E402  (Phase 23 runner: same product path, same file specs)
import backend.main as bm                        # noqa: E402

DATA = os.environ.get("TASKCLEAN_DATA", os.path.join(ROOT, "data"))
OUT = os.path.join(ROOT, "results")
vnd.NEW = os.path.join(DATA, "new")
vnd.FILES["D"] = dict(path="Uncleaned_DS_jobs.csv", target="Rating", task="regression", drop=[])


def jsonable(o):
    return vnd.jsonable(o)


def analyse_D(audit: dict, df: pd.DataFrame) -> dict:
    """Facts about the DS-jobs audit that do not depend on the code version, so the same function reads the 'before' and 'after' runs."""
    st, res = audit["state"], audit["result"]
    agg = res.repair_log_aggressive
    ev = res.repair_log
    minus1 = set(np.flatnonzero((df["Rating"] == -1).to_numpy()).tolist())
    wt = agg[agg.problem == "Potential wrong target value"]
    founded = agg[(agg.problem == "Potential outlier") & (agg.column == "Founded")]
    out = {
        "codes": jsonable(st.meta.get("codes")),
        "code_majority": jsonable(st.meta.get("code_majority")),
        "n_rows_missing_target": int(st.meta.get("n_rows_missing_target", 0)),
        "n_rows_analysed": int(st.meta["n_rows_analysed"]),
        "missing_cells_audit_features": int(sum(len(v) for v in st.meta.get("code_cells", {}).values())),
        "outlier_proposals_founded_total": int(len(founded)),
        "outlier_proposals_founded_on_minus1": int((pd.to_numeric(founded.original_value, errors="coerce") == -1).sum()),
        "wrong_target_flags": int(len(wt)),
        "wrong_target_flags_on_rating_minus1": int(wt.row_id.isin(minus1).sum()),
        "wrong_target_flags_on_5_0": int((pd.to_numeric(wt.original_value, errors="coerce") == 5.0).sum()),
        "rating_minus1_rows": len(minus1),
        "proposals_for_rating_minus1_rows_in_target_column": int(len(agg[agg.row_id.isin(minus1) & (agg.column == "Rating")])),
        "proposals_by_problem_aggressive": jsonable(agg.groupby("problem").size().to_dict()),
        "proposals_by_problem_evidence": jsonable(ev.groupby("problem").size().to_dict()),
        "evidence_applied": int((ev.action == "applied").sum()),
        "aggressive_applied": int((agg.action == "applied").sum()),
    }
    cleaned, cagg = res.cleaned, res.cleaned_aggressive
    code_cols = ["Headquarters", "Size", "Founded", "Type of ownership", "Industry", "Sector", "Revenue"]
    raw_is_code = {c: (df[c].astype(str).str.strip().isin(["-1", "-1.0"])) for c in code_cols}
    out["evidence_file_code_cells_left"] = {c: int((cleaned[c].astype(str).str.strip().isin(["-1", "-1.0"]) & raw_is_code[c]).sum()) for c in code_cols}
    out["aggressive_file_code_cells_left"] = {c: int((cagg[c].astype(str).str.strip().isin(["-1", "-1.0"]) & raw_is_code[c]).sum()) for c in code_cols}
    out["rating_cells_unchanged_in_both_files"] = bool((cleaned["Rating"].reset_index(drop=True).iloc[sorted(minus1)] == -1).all()
                                                       and (cagg["Rating"].reset_index(drop=True).iloc[sorted(minus1)] == -1).all())
    out["competitors_minus1_left_in_aggressive"] = int((cagg["Competitors"].astype(str) == "-1").sum())
    return out


def run_D(out_path: str) -> dict:
    spec = vnd.FILES["D"]
    # run_tabular drops nothing and writes nothing; it returns the dict we extend with the DS-specific analysis
    res = vnd.run_tabular("D")
    audit = None
    for a in bm.AUDITS.values():
        if a["dataset_name"] == os.path.basename(spec["path"]):
            audit = a
    if audit is not None:
        df = pd.read_csv(os.path.join(vnd.NEW, spec["path"]))
        res["D_analysis"] = analyse_D(audit, df)
        res["baseline_completeness"] = jsonable({k: v for k, v in (audit["baseline"]["completeness"]).items() if k != "placeholder_tokens"})
        res["overview_notes_with_code_or_target"] = [n for n in jsonable(audit["state"].meta["notes"])
                                                       if "code" in n.lower() or "no target" in n.lower() or "held back" in n.lower()]
    res["code_version"] = os.popen(f'git -C "{ROOT}" rev-parse --short HEAD').read().strip()
    res["code_is_phase25"] = hasattr(sys.modules["taskclean"], "CODE_CANDIDATES")
    with open(out_path, "w") as f:
        json.dump(jsonable(res), f, indent=1)
    return res


# ------------------------------------------------------------------------------------------------------------- S0
def digest(o) -> str:
    return hashlib.sha256(json.dumps(jsonable(o), sort_keys=True, default=str).encode()).hexdigest()[:12]


def drop_volatile(d: dict) -> dict:
    """Timing and memory are not results; limitations text was extended by Phase 25 on purpose and is compared separately."""
    skip = {"load_seconds", "audit_seconds", "panel_seconds", "peak_rss_mb", "wall_seconds", "store_seconds", "limitations",
            "peak_rss_mb_after_load", "calibration_seconds"}
    def walk(x):
        if isinstance(x, dict):
            return {k: walk(v) for k, v in x.items() if k not in skip and not k.endswith("_seconds") and k not in ("audit_seconds", "detector_seconds")}
        if isinstance(x, list):
            return [walk(v) for v in x]
        if isinstance(x, float):
            return round(x, 6)
        return x
    return walk(d)


import re


def _round_numbers(text: str) -> str:
    return re.sub(r"-?\d+\.\d+", lambda m: f"{float(m.group()):.2f}", text)


FLOAT_TOL = 2e-3          # run-to-run wobble of the parallel forests (seen on the committed code too); counts, strings, decisions are exact
VOLATILE = ("finished", "seconds")


def diff(a, b, path="", out=None, stats=None, limit=40):
    """Exact for ints, strings, bools and structure; floats within FLOAT_TOL (the largest deviation is kept in stats)."""
    out = [] if out is None else out
    stats = {"max_float_dev": 0.0, "float_cells": 0} if stats is None else stats
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if any(v in k for v in VOLATILE):
                continue
            if k not in a or k not in b:
                out.append(f"{path}/{k}: only in {'new' if k in a else 'old'}")
            else:
                diff(a[k], b[k], f"{path}/{k}", out, stats, limit)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f"{path}: length new={len(a)} old={len(b)}")
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                diff(x, y, f"{path}[{i}]", out, stats, limit)
    elif isinstance(a, float) or isinstance(b, float):
        try:
            fa, fb = float(a), float(b)
        except (TypeError, ValueError):
            if a != b:
                out.append(f"{path}: new={str(a)[:60]!r} old={str(b)[:60]!r}")
            return out
        if np.isnan(fa) and np.isnan(fb):
            return out
        stats["float_cells"] += 1
        dev = abs(fa - fb) / max(1.0, abs(fb))
        stats["max_float_dev"] = max(stats["max_float_dev"], dev)
        if not dev <= FLOAT_TOL:
            out.append(f"{path}: new={fa:.6g} old={fb:.6g}")
    elif isinstance(a, str) and isinstance(b, str) and _round_numbers(a) == _round_numbers(b):
        pass                                           # numbers printed inside a sentence agree to two decimals
    elif a != b and not (str(a) == "nan" and str(b) == "nan"):
        out.append(f"{path}: new={str(a)[:80]!r} old={str(b)[:80]!r}")
    return out


def run_S0(fid: str) -> dict:
    new = vnd.run_C() if fid == "C" else vnd.run_O() if fid == "O" else vnd.run_tabular(fid)
    old = json.load(open(os.path.join(OUT, f"phase23_{fid}.json")))
    a, b = drop_volatile(jsonable(new)), drop_volatile(old)
    stats = {"max_float_dev": 0.0, "float_cells": 0}
    d = diff(a, b, stats=stats)
    return {"id": fid, "same_decisions_and_counts": not d, "differences": d, "float_cells_compared": stats["float_cells"],
            "max_relative_float_deviation": round(stats["max_float_dev"], 6), "float_tolerance": FLOAT_TOL,
            "code_is_phase25": hasattr(sys.modules["taskclean"], "CODE_CANDIDATES"),
            "limitations_changed": jsonable(new.get("limitations")) != old.get("limitations"),
            "new_digest": digest(a), "old_digest": digest(b)}


def run_S0demo() -> dict:
    """Byte-identical cleaned files on demo_dirty_adult.csv, before (committed code, via --out in a worktree) and after."""
    from taskclean import apply_repairs, audit_dataset
    df = pd.read_csv(os.path.join(DATA, "demo_dirty_adult.csv"))
    st = audit_dataset(df, "class", detectors={"missing_values", "duplicates", "outliers", "inconsistency"})
    res = apply_repairs(st, {"missing_values", "duplicates", "outliers", "inconsistency"})
    h = lambda d: hashlib.sha256(d.to_csv(index=True).encode()).hexdigest()
    return {"rows": len(df), "cleaned_sha256": h(res.cleaned), "aggressive_sha256": h(res.cleaned_aggressive),
            "repair_log_sha256": hashlib.sha256(res.repair_log.drop(columns=["method"]).to_csv(index=False).encode()).hexdigest(),
            "code_is_phase25": hasattr(sys.modules["taskclean"], "CODE_CANDIDATES")}


# ------------------------------------------------------------------------------------------------------------- S2
def planted(df: pd.DataFrame, col: str, target: str, task: str, code, rate: float, seed: int) -> dict:
    from taskclean import _prepare
    d = df.copy()
    rng = np.random.default_rng(seed)
    rows = np.sort(rng.choice(len(d), max(3, int(rate * len(d))), replace=False))
    d[col] = pd.to_numeric(d[col]).astype(float)
    d.loc[rows, col] = float(code)
    X, y, *_ = _prepare(d.reset_index(drop=True), target, task)
    codes = X.attrs["codes"]
    got = set(codes.get(col, {}).get("rows", []))
    want = set(rows.tolist())
    other_cols = sorted(c for c in codes if c != col)
    return {"column": col, "code": code, "rate": rate, "seed": seed, "planted": len(want), "found": len(got),
            "recall": len(got & want) / len(want), "precision": (len(got & want) / len(got)) if got else 0.0,
            "other_columns_flagged": other_cols, "extra_cells_in_column": len(got - want)}


def run_S2() -> dict:
    from intake import load_table
    cases = []
    adult = pd.read_csv(os.path.join(DATA, "adult_clean.csv"))
    cal = pd.read_csv(os.path.join(DATA, "california_housing.csv"))
    a_target = [c for c in adult.columns if c.lower() in ("income", "class", "label", "target")][0]
    c_target = "MedHouseVal"
    for code in (-1, -999):
        for rate in (0.01, 0.05, 0.20):
            for seed in (0, 1, 2):
                cases.append({"file": "adult_clean", **planted(adult, "age", a_target, "classification", code, rate, seed)})
                cases.append({"file": "california_housing", **planted(cal, "Population", c_target, "regression", code, rate, seed)})
    # one Excel workbook case: the code is a number in a spreadsheet, so the numeric rule applies
    import openpyxl, io
    wb = openpyxl.Workbook(); ws = wb.active
    rng = np.random.default_rng(0)
    ws.append(["age", "city", "y"])
    codes_at = set(rng.choice(400, 20, replace=False).tolist())
    for i in range(400):
        ws.append([-1 if i in codes_at else int(rng.integers(18, 80)), str(rng.choice(["Pune", "Goa"])), str(rng.choice(["p", "q"]))])
    buf = io.BytesIO(); wb.save(buf)
    x, _ = load_table(buf.getvalue(), "codes.xlsx")
    from taskclean import _prepare
    X, *_ = _prepare(x, "y", "classification")
    xl = {"planted": len(codes_at), "found_rows_match": X.attrs["codes"]["age"]["rows"] == sorted(codes_at)}
    bad = [c for c in cases if c["recall"] < 1.0 or c["precision"] < 1.0 or c["other_columns_flagged"]]
    return {"cases": cases, "n_cases": len(cases), "failures": bad, "excel": xl,
            "all_recall_and_precision_1": not bad and xl["found_rows_match"]}


if __name__ == "__main__":
    stage = sys.argv[1]
    out_path = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else None
    t0 = time.time()
    try:
        if stage == "D":
            res = run_D(out_path or os.path.join(OUT, "phase25_D.json"))
            fid = "D"
        elif stage == "S0":
            res = run_S0(sys.argv[2]); fid = f"S0_{sys.argv[2]}"
        elif stage == "S0demo":
            res = run_S0demo(); fid = "S0demo"
        elif stage == "S2":
            res = run_S2(); fid = "S2"
        else:
            raise SystemExit(f"unknown stage {stage}")
    except Exception as e:                      # noqa: BLE001  a crash is itself a finding: record it, never hide it
        res = {"crash": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()}
        fid = stage
    res["wall_seconds"] = round(time.time() - t0, 1)
    if stage != "D" or "crash" in res:
        path = out_path or os.path.join(OUT, f"phase25_{fid}.json")
        with open(path, "w") as f:
            json.dump(jsonable(res), f, indent=1)
    print(f"{fid}: done in {res['wall_seconds']}s", "CRASH: " + res["crash"] if "crash" in res else "")
