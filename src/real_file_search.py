"""
Real-file search for a cleaning-improves-the-model result (exploratory, NOT pre-registered; written 2026-11-18 at the user's request).

Runs the product's Model panel (repairs fitted on training rows only, models scored on untouched held-out rows of the user's own file, a change is
"improved" only if it clears the measured noise floor AND >= 80% of seeds agree) on EVERY real file in the project with 5 seeds, and records every
result, not only the good ones. The held-out rows are as dirty as the file, so an "improved" verdict means "training on the repaired rows predicts
this file's own rows better", not "the repair was correct".

  .venv/bin/python src/real_file_search.py [name ...]     -> results/real_search_<name>.json
"""
from __future__ import annotations
import json, os, sys, time, traceback
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path[:0] = [HERE, ROOT]
from intake import load_table                                      # noqa: E402
from model_panel import PANEL_DETECTORS, run_model_panel           # noqa: E402
from model_panel_regression import run_model_panel_regression      # noqa: E402
from taskclean import audit_dataset                                # noqa: E402

D = os.path.join(ROOT, "data")
SEEDS = (42, 43, 44, 45, 46)
FILES = {   # name: (path, target, task, columns to drop)
    "bank": ("bank_marketing_original.csv", "Class", "classification", []),
    "adult_original": ("adult_original.csv", "class", "classification", []),
    "credit_g": ("credit_g_original.csv", "class", "classification", []),
    "telco": ("telco_churn_original.csv", "Churn", "classification", ["customerID"]),
    "titanic": ("new/titanic.csv", "Survived", "classification", ["PassengerId"]),
    "heart": ("heart_c_original.csv", "num", "classification", []),
    "california": ("california_housing.csv", "MedHouseVal", "regression", []),
    "ames": ("new/ames_housing.csv", "SalePrice", "regression", ["Id"]),
    "realestate": ("new/realestate/Real estate valuation data set.xlsx", "Y house price of unit area", "regression", ["No"]),
    "dsjobs": ("new/Uncleaned_DS_jobs.csv", "Rating", "regression", []),
}


def run(name: str) -> dict:
    path, target, task, drop = FILES[name]
    raw = open(os.path.join(D, path), "rb").read()
    df, info = load_table(raw, os.path.basename(path))
    df = df.drop(columns=[c for c in drop if c in df.columns])
    t0 = time.time()
    st = audit_dataset(df, target, detectors=PANEL_DETECTORS, task=task, max_model_rows=4000)
    res = (run_model_panel_regression if task == "regression" else run_model_panel)(st, seeds=SEEDS)
    out = {"name": name, "task": task, "rows": len(df), "status": res.get("status"), "reason": res.get("reason"), "seconds": round(time.time() - t0, 1),
           "rows_used": res.get("rows_used"), "default_issues": res.get("default_issues"), "results": res.get("results", [])}
    return out


if __name__ == "__main__":
    for name in (sys.argv[1:] or list(FILES)):
        try:
            r = run(name)
        except Exception as e:                      # noqa: BLE001  a crash is a result too
            r = {"name": name, "status": "crash", "reason": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-600:]}
        json.dump(r, open(os.path.join(ROOT, "results", f"real_search_{name}.json"), "w"), default=str)
        print(name, r.get("status"), r.get("seconds"), flush=True)
