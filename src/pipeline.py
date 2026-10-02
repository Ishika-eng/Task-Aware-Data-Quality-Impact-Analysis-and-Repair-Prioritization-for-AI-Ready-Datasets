"""
End-to-end pipeline, mirroring the project diagram:

    Dataset -> Data Quality Audit -> Inject errors -> Train -> Measure impact
    -> Rank by impact/effort -> Repair top issues -> Retrain & validate
    -> Assess AI-readiness

Run this file directly (`python src/pipeline.py`) to execute the whole
study on the Breast Cancer Wisconsin dataset and print a report.
"""

import os

import pandas as pd
from sklearn.datasets import load_breast_cancer

from errors import ERROR_INJECTORS
from train import make_clean_split, train_and_evaluate
from impact import compute_impact, compute_priority
from repair import REPAIR_FUNCTIONS
from readiness import readiness_score, readiness_verdict

INJECTION_RATE = 0.15  # how aggressively we corrupt data for the experiment
SEED = 42
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def load_dataset():
    data = load_breast_cancer(as_frame=True)
    return data.data, data.target


def run_study():
    print("=" * 70)
    print("STEP 1: Load dataset + fixed clean train/test split")
    print("=" * 70)
    X, y = load_dataset()
    X_train, X_test, y_train, y_test = make_clean_split(X, y, seed=SEED)
    print(f"Dataset: breast cancer (binary classification), "
          f"{X.shape[0]} rows x {X.shape[1]} features")
    print(f"Train: {len(X_train)} rows | Test (always clean, held out): {len(X_test)} rows")

    print("\n" + "=" * 70)
    print("STEP 2: Train baseline model on CLEAN data")
    print("=" * 70)
    clean_f1 = train_and_evaluate(X_train, y_train, X_test, y_test, seed=SEED)
    print(f"Clean baseline macro-F1: {clean_f1:.4f}")

    print("\n" + "=" * 70)
    print(f"STEP 3: Inject each error type independently (rate={INJECTION_RATE}) "
          f"and measure its ISOLATED impact")
    print("=" * 70)
    dirty_f1_by_error = {}
    dirty_data_by_error = {}
    for name, inject_fn in ERROR_INJECTORS.items():
        X_dirty, y_dirty = inject_fn(X_train, y_train, rate=INJECTION_RATE, seed=SEED)
        f1 = train_and_evaluate(X_dirty, y_dirty, X_test, y_test, seed=SEED)
        dirty_f1_by_error[name] = f1
        dirty_data_by_error[name] = (X_dirty, y_dirty)
        print(f"  {name:22s} -> macro-F1 = {f1:.4f}  (impact = {clean_f1 - f1:+.4f})")

    print("\n" + "=" * 70)
    print("STEP 4: Rank errors by impact AND by effort-aware priority")
    print("=" * 70)
    impact_df = compute_impact(clean_f1, dirty_f1_by_error)
    priority_df = compute_priority(impact_df)
    print("\nRanked by raw impact (worst first):")
    print(impact_df.to_string(index=False))
    print("\nRanked by PRIORITY = impact / effort (fix this first):")
    print(priority_df[["error_type", "impact", "effort", "priority"]].to_string(index=False))

    print("\n" + "=" * 70)
    print("STEP 5: Repair the top-priority issue, retrain, validate, assess readiness")
    print("=" * 70)
    top_error = priority_df.iloc[0]["error_type"]
    print(f"Top priority repair target: '{top_error}'")

    X_dirty, y_dirty = dirty_data_by_error[top_error]
    repair_fn = REPAIR_FUNCTIONS[top_error]
    X_repaired, y_repaired = repair_fn(X_dirty, y_dirty, clean_X=X_train, clean_y=y_train)
    repaired_f1 = train_and_evaluate(X_repaired, y_repaired, X_test, y_test, seed=SEED)

    dirty_f1 = dirty_f1_by_error[top_error]
    score = readiness_score(clean_f1, dirty_f1, repaired_f1)
    verdict = readiness_verdict(score)

    print(f"  Dirty F1 (before repair):    {dirty_f1:.4f}")
    print(f"  Repaired F1 (after repair):  {repaired_f1:.4f}")
    print(f"  Clean baseline F1:           {clean_f1:.4f}")
    print(f"  AI-readiness score:          {score:.3f}  ({score*100:.1f}% of lost performance recovered)")
    print(f"  Verdict:                     {verdict}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    priority_df.to_csv(os.path.join(RESULTS_DIR, "priority_ranking.csv"), index=False)
    pd.DataFrame([{
        "top_error": top_error,
        "clean_f1": clean_f1,
        "dirty_f1": dirty_f1,
        "repaired_f1": repaired_f1,
        "readiness_score": score,
        "readiness_verdict": verdict,
    }]).to_csv(os.path.join(RESULTS_DIR, "repair_summary.csv"), index=False)
    print(f"\nSaved results/priority_ranking.csv and results/repair_summary.csv")

    return {
        "clean_f1": clean_f1,
        "impact_df": impact_df,
        "priority_df": priority_df,
        "top_error": top_error,
        "repaired_f1": repaired_f1,
        "readiness_score": score,
        "readiness_verdict": verdict,
    }


if __name__ == "__main__":
    run_study()
