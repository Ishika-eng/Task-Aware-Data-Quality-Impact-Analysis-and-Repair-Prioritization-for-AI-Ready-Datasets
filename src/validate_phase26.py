"""
Phase 26 validation on real files and planted truth (docs/PHASE26_SPEC.md). No product code is changed here.

  python validate_phase26.py D                  Uncleaned_DS_jobs.csv, Rating as regression target (S1, S6)     -> results/phase26_D.json
  python validate_phase26.py S0 T|A|R|O         re-run a Phase 23 file and compare with results/phase23_<id>.json -> results/phase26_S0_<id>.json
  python validate_phase26.py S0files            A/B on every CSV/Excel file: Phase 26 features on vs off          -> results/phase26_S0files.json
  python validate_phase26.py S0demo             byte comparison of the cleaned files on demo_dirty_adult.csv    -> results/phase26_S0demo.json
  python validate_phase26.py S2                 planted words / declared values / keep_values (S2)              -> results/phase26_S2.json
  python validate_phase26.py B                  Bank Marketing through the product path (S6)                    -> results/phase26_B.json

"Off" in S0files means the same code with the word list emptied and the F3 diagnostics returned empty: that is Phase 25 behaviour for an
audit without declarations (nothing else in the product is switched), so it is a like-for-like A/B on one code base. It reuses the
Phase 25 runner for D and S0 (same product path: load_table, store_dataset, run_audit_job).
"""
from __future__ import annotations

import contextlib
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

import validate_sentinels as vs                  # noqa: E402  (Phase 25 runner)
import validate_new_datasets as vnd              # noqa: E402
import taskclean as tc                           # noqa: E402

DATA, OUT, jsonable = vs.DATA, vs.OUT, vs.jsonable
CHEAP = {"missing_values", "duplicates", "outliers", "inconsistency"}
vnd.FILES["B"] = dict(path="../bank_marketing_original.csv", target="Class", task="classification", drop=[])


@contextlib.contextmanager
def features_off():
    """Phase 26's two automatic features off: no word placeholders, no F3 diagnostics (declarations default to none anyway)."""
    saved = tc.WORD_PLACEHOLDERS, tc._diagnostic_notes
    tc.WORD_PLACEHOLDERS, tc._diagnostic_notes = frozenset(), lambda *a, **k: []
    try:
        yield
    finally:
        tc.WORD_PLACEHOLDERS, tc._diagnostic_notes = saved


def sha(df: pd.DataFrame) -> str:
    return hashlib.sha256(df.to_csv(index=True).encode()).hexdigest()[:16]


def cheap_run(df: pd.DataFrame, target: str | None, task: str) -> dict:
    st = tc.audit_dataset(df, target, detectors=set(CHEAP), task=task)
    res = tc.apply_repairs(st, set(CHEAP))
    log = res.repair_log
    dres = tc.apply_repairs(st)                                    # the default evidence-based file (what a user downloads), no override
    dlog = dres.repair_log
    word_rows = {c: set(r) for c, r in st.meta.get("code_cells", {}).items() if st.meta["codes"][c].get("n_word", 0)}
    kept = int(sum((dres.cleaned.loc[list(r), c].astype(str).str.strip().str.lower() == "unknown").sum() for c, r in word_rows.items()
                   if set(r) <= set(dres.cleaned.index)))
    return {"default_cleaned": sha(dres.cleaned), "default_applied": int((dlog.action == "applied").sum()),
            "default_word_cells_still_unknown": kept, "default_issues": sorted(tc.default_apply_issues(st)),
            "cleaned": sha(res.cleaned), "aggressive": sha(res.cleaned_aggressive),
            "log_without_method": hashlib.sha256(log.drop(columns=["method"]).to_csv(index=False).encode()).hexdigest()[:16],
            "log_with_method": hashlib.sha256(log.to_csv(index=False).encode()).hexdigest()[:16],
            "proposals": int(len(st.proposals)), "applied": int((log.action == "applied").sum()),
            "word_columns": {c: v.get("n_word", 0) for c, v in st.meta["codes"].items() if v.get("n_word", 0)},
            "word_cells": int(sum(v.get("n_word", 0) for v in st.meta["codes"].values())),
            "notes_sha": hashlib.sha256("\n".join(map(str, st.meta["notes"])).encode()).hexdigest()[:16],
            "notes": [str(n)[:200] for n in st.meta["notes"] if any(k in str(n) for k in ("placeholder word", "ranges", "second line"))]}


def files() -> dict:
    from intake import load_table
    out = {}
    for name, target, task in (("adult_original.csv", "class", "classification"), ("adult_clean.csv", "class", "classification"),
                               ("california_housing.csv", "MedHouseVal", "regression"), ("credit_g_original.csv", "class", "classification"),
                               ("heart_c_original.csv", "num", "classification"), ("telco_churn_original.csv", "Churn", "classification"),
                               ("bank_marketing_original.csv", "Class", "classification")):
        out[name] = (pd.read_csv(os.path.join(DATA, name), low_memory=False), target, task)
    for name, target, task, drop in (("new/titanic.csv", "Survived", "classification", ["PassengerId"]),
                                     ("new/ames_housing.csv", "SalePrice", "regression", ["Id"])):
        d = pd.read_csv(os.path.join(DATA, name), low_memory=False)
        out[name] = (d.drop(columns=[c for c in drop if c in d.columns]), target, task)
    p = os.path.join(DATA, "new", "realestate", "Real estate valuation data set.xlsx")
    d, _ = load_table(open(p, "rb").read(), os.path.basename(p))
    out["realestate.xlsx"] = (d.drop(columns=["No"]), "Y house price of unit area", "regression")
    return out


def run_S0files() -> dict:
    rows = {}
    for name, (df, target, task) in files().items():
        t0 = time.time()
        with features_off():
            off = cheap_run(df, target, task)
        on = cheap_run(df, target, task)
        same = {k: on[k] == off[k] for k in ("cleaned", "aggressive", "log_without_method", "proposals", "applied", "notes_sha",
                                             "default_cleaned", "default_applied")}
        rows[name] = {"rows": len(df), "byte_identical_all": all(same.values()), "same": same, "word_cells_on": on["word_cells"],
                      "word_columns_on": on["word_columns"], "new_notes": on["notes"], "cleaned_on": on["cleaned"], "cleaned_off": off["cleaned"],
                      "aggressive_on": on["aggressive"], "aggressive_off": off["aggressive"],
                      "proposals_on": on["proposals"], "proposals_off": off["proposals"], "applied_on": on["applied"],
                      "applied_off": off["applied"], "default_cleaned_on": on["default_cleaned"], "default_cleaned_off": off["default_cleaned"],
                      "default_applied_on": on["default_applied"], "default_applied_off": off["default_applied"],
                      "default_word_cells_still_unknown_on": on["default_word_cells_still_unknown"], "default_issues": on["default_issues"],
                      "seconds": round(time.time() - t0, 1)}
    unchanged = [n for n, r in rows.items() if r["word_cells_on"] == 0 and not r["new_notes"]]
    return {"files": rows, "no_pattern_files": unchanged,
            "no_pattern_files_all_identical": all(rows[n]["byte_identical_all"] for n in unchanged),
            "files_with_patterns": [n for n in rows if n not in unchanged],
            "concrete": "not run: xlrd is not installed in the venv (as in Phase 25); its columns are physical quantities and the Phase 25 scan found no pattern, but this phase did not scan it"}


def run_S0demo() -> dict:
    d = vs.run_S0demo()
    old = json.load(open(os.path.join(OUT, "phase25_S0demo.json")))
    d["phase25"] = {k: old[k] for k in ("cleaned_sha256", "aggressive_sha256", "repair_log_sha256")}
    d["byte_identical_to_phase25"] = all(d[k] == old[k] for k in ("cleaned_sha256", "aggressive_sha256", "repair_log_sha256"))
    return d


def run_S0(fid: str) -> dict:
    d = vs.run_S0(fid)
    d["expected_to_change"] = fid == "O"          # Online Retail II has 'Unspecified' / 'missing' cells (scan)
    return d


def run_D(out_path: str) -> dict:
    res = vs.run_D(out_path)
    a = res.get("D_analysis", {})
    codes = a.get("codes", {})
    feat = {c: v for c, v in codes.items() if not v["is_target"]}
    res["phase26"] = {
        "feature_placeholder_cells": int(sum(v["count"] for v in feat.values())),
        "per_column": {c: {"count": v["count"], "n_code": v.get("n_code"), "n_word": v.get("n_word")} for c, v in codes.items()},
        "baseline_missing_cells": res.get("baseline_completeness", {}).get("missing_cells"),
        "notes_f3": [n for n in res.get("notes", []) if "numeric ranges" in n or "second line" in n],
        "calibration_status": res.get("calibration", {}).get("status"),
    }
    with open(out_path, "w") as f:
        json.dump(jsonable(res), f, indent=1)
    return res


# ------------------------------------------------------------------------------------------------------------- S2
def planted_words(df: pd.DataFrame, col: str, target: str, task: str, word: str, rate: float, seed: int) -> dict:
    d = df.copy().reset_index(drop=True)
    rng = np.random.default_rng(seed)
    rows = np.sort(rng.choice(len(d), max(3, int(rate * len(d))), replace=False))
    d[col] = d[col].astype(object)
    d.loc[rows, col] = word
    before = tc._prepare(df.reset_index(drop=True), target, task)[0].attrs["codes"]
    X = tc._prepare(d, target, task)[0]
    codes = X.attrs["codes"]
    got, want = set(codes.get(col, {}).get("rows", [])), set(rows.tolist())
    already = set(before.get(col, {}).get("rows", []))                      # cells the clean column already had flagged (none expected)
    return {"column": col, "word": word, "rate": rate, "seed": seed, "planted": len(want), "found": len(got),
            "recall": len(got & want) / len(want), "precision": (len(got & want) / len(got)) if got else 0.0,
            "other_columns_flagged": sorted(c for c in codes if c != col), "already_flagged_in_clean": len(already)}


def run_S2() -> dict:
    adult = pd.read_csv(os.path.join(DATA, "adult_clean.csv"))
    credit = pd.read_csv(os.path.join(DATA, "credit_g_original.csv"))
    cases = []
    for word in ("Unknown", "unspecified", "Missing"):
        for rate in (0.01, 0.05, 0.20):
            for seed in (0, 1, 2):
                cases.append({"file": "adult_clean", **planted_words(adult, "workclass", "class", "classification", word, rate, seed)})
                cases.append({"file": "credit_g", **planted_words(credit, "purpose", "class", "classification", word, rate, seed)})
    # numeric-as-text column: words go into a column of numbers stored as text
    num_text = adult.copy()
    num_text["age_text"] = num_text["age"].astype(str)
    for seed in (0, 1, 2):
        for rate in (0.01, 0.05, 0.20):
            cases.append({"file": "adult_clean (numeric text)", **planted_words(num_text, "age_text", "class", "classification", "Unknown", rate, seed)})
    bad = [c for c in cases if c["recall"] < 1.0 or c["precision"] < 1.0 or c["other_columns_flagged"] or c["already_flagged_in_clean"]]

    # declared: plant 0 where 0 is impossible in one column of California, legitimate zeros live in other columns of Adult
    declared = []
    cal = pd.read_csv(os.path.join(DATA, "california_housing.csv"))
    for rate in (0.01, 0.05, 0.20):
        for seed in (0, 1, 2):
            d = cal.copy()
            rows = np.sort(np.random.default_rng(seed).choice(len(d), int(rate * len(d)), replace=False))
            d.loc[rows, "AveBedrms"] = 0.0
            base = tc._prepare(cal.copy(), "MedHouseVal", "regression")[0]
            X = tc._prepare(d, "MedHouseVal", "regression", {"AveBedrms": ["0"]})[0]
            got = set(X.attrs["declared"]["AveBedrms"]["rows"])
            others_same = all(X[c].equals(base[c]) for c in X.columns if c != "AveBedrms")
            declared.append({"rate": rate, "seed": seed, "planted": len(rows), "found": len(got), "exact": got == set(rows.tolist()),
                             "others_unchanged": others_same, "codes_flagged": sorted(X.attrs["codes"])})
    ad = adult.copy()
    zeros_gain = int((ad["capital-gain"] == 0).sum())
    X = tc._prepare(ad, "class", "classification", {"capital-gain": ["0"]})[0]
    legit = {"declared_zero_cells_in_capital_gain": X.attrs["declared"]["capital-gain"]["count"], "equals_real_zero_count": zeros_gain,
             "note": "declaring 0 in a column where 0 is a real value treats every real zero as missing: that is the user's decision, and it is applied exactly"}

    # keep_values on a column with genuine -1 and 'Unknown'
    kv = pd.DataFrame({"x": np.random.default_rng(0).integers(0, 40, 300).astype(float),
                       "g": np.random.default_rng(1).choice(["a", "b", "c"], 300), "y": np.random.default_rng(2).choice(["p", "q"], 300)})
    kv.loc[:7, "x"] = -1
    kv.loc[10:15, "g"] = "Unknown"
    on = tc._prepare(kv, "y", "classification")[0]
    off = tc._prepare(kv, "y", "classification", None, {"x", "g"})[0]
    keep = {"flagged_without_keep": sorted(on.attrs["codes"]), "flagged_with_keep": sorted(off.attrs["codes"]),
            "x_values_kept": int((off["x"] == -1).sum()), "g_unknown_kept": int((off["g"] == "Unknown").sum())}
    ok = (not bad and all(c["exact"] and c["others_unchanged"] and not c["codes_flagged"] for c in declared)
          and keep["flagged_without_keep"] == ["g", "x"] and keep["flagged_with_keep"] == [])
    return {"words": {"n_cases": len(cases), "failures": bad, "cases": cases}, "declared": {"cases": declared, "adult_capital_gain": legit},
            "keep_values": keep, "all_ok": ok}


if __name__ == "__main__":
    stage = sys.argv[1]
    out_path = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else None
    t0 = time.time()
    try:
        if stage == "D":
            res = run_D(out_path or os.path.join(OUT, "phase26_D.json")); fid = "D"
        elif stage == "B":
            res = vnd.run_tabular("B"); fid = "B"
        elif stage == "Boff":                       # the same full product run with the Phase 26 automatic features switched off
            with features_off():
                res = vnd.run_tabular("B")
            fid = "Boff"
        elif stage == "S0":
            res = run_S0(sys.argv[2]); fid = f"S0_{sys.argv[2]}"
        elif stage == "S0files":
            res = run_S0files(); fid = "S0files"
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
    res["code_has_phase26"] = hasattr(tc, "WORD_PLACEHOLDERS")
    if stage != "D" or "crash" in res:
        with open(out_path or os.path.join(OUT, f"phase26_{fid}.json"), "w") as f:
            json.dump(jsonable(res), f, indent=1)
    print(f"{fid}: done in {res['wall_seconds']}s", "CRASH: " + res["crash"] if "crash" in res else "")
