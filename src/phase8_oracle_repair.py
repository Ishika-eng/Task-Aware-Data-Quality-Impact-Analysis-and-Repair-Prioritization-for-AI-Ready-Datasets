"""
Oracle-repair ceiling experiment (Phase 8).

For each error type x rate x seed, after measuring dirty F1 (same
procedure as Phase 7), apply an ORACLE repair -- using the corruption log
to restore the EXACT original values -- then retrain and measure repaired
F1. This is not a real repair method (a real system never has the
corruption log) -- it answers a different, narrower question: "if this
error type were perfectly fixed, what's the maximum F1 we could possibly
recover?" That ceiling is what Phase 9's real (imperfect) repair engine
gets graded against.

Metrics recorded per run, kept separate rather than collapsed into one
number (per the project's reviewer's explicit guidance):

    Dirty F1
    Repaired F1
    Clean (baseline) F1
    Recovery      = Repaired F1 - Dirty F1
    Recovery Rate = Recovery / (Clean F1 - Dirty F1)   (fraction of the
                    damage this repair undid; 1.0 = fully recovered)

No ranking/prioritization happens here -- that's deliberately deferred
until a real (non-oracle) repair engine exists with its own cost, in a
later phase.
"""

import os

import numpy as np
import pandas as pd

from baseline import evaluate
from corruption import RATES, corrupt_with_id
from data import load_adult_clean, three_way_split
from phase7_impact import SPLIT_SEED, build_rf_pipeline
from training_log import log_training

ERROR_TYPES = ["missing_values", "label_errors", "duplicates", "outliers", "feature_corruption"]


def oracle_repair(error_type: str, X_dirty: pd.DataFrame, y_dirty: pd.Series, log_df: pd.DataFrame):
    """Invert exactly the corruption recorded in log_df, using the logged
    original_value for every corrupted cell/row. Dispatch differs by error
    type because each one changed the data differently (see corruption/*.py):
    cell-level value swaps (missing/outliers/feature_corruption), a whole-row
    label flip (label_errors), or appended rows (duplicates).
    """
    X_repaired, y_repaired = X_dirty.copy(), y_dirty.copy()

    if error_type in ("missing_values", "outliers", "feature_corruption"):
        for row_id, col, original in zip(log_df["row_id"], log_df["column"], log_df["original_value"]):
            X_repaired.at[row_id, col] = original

    elif error_type == "label_errors":
        for row_id, original in zip(log_df["row_id"], log_df["original_value"]):
            y_repaired.loc[row_id] = original

    elif error_type == "duplicates":
        # the logged row_ids ARE the newly appended duplicate rows -- an
        # oracle repair simply removes them, restoring the original rows.
        duplicate_ids = log_df["row_id"].tolist()
        X_repaired = X_repaired.drop(index=duplicate_ids)
        y_repaired = y_repaired.drop(index=duplicate_ids)

    else:
        raise ValueError(f"no oracle repair defined for {error_type}")

    return X_repaired, y_repaired


def run_experiment(seeds: list[int]) -> pd.DataFrame:
    X, y = load_adult_clean()
    X_train, X_val, X_test, y_train, y_val, y_test = three_way_split(X, y, seed=SPLIT_SEED)

    print(f"Computing clean baseline F1 for {len(seeds)} seed(s)...")
    baseline_f1 = {}
    for seed in seeds:
        pipeline = build_rf_pipeline(X_train, seed=seed)
        with log_training("phase8", f"clean_baseline seed={seed}", pipeline, X_train, seed=seed):
            pipeline.fit(X_train, y_train)
        baseline_f1[seed] = evaluate(pipeline, X_test, y_test)["f1"]
        print(f"  seed={seed}: clean F1 = {baseline_f1[seed]:.4f}")

    print(f"\nRunning {len(ERROR_TYPES)} error types x {len(RATES)} rates x {len(seeds)} seed(s) "
          f"= {len(ERROR_TYPES) * len(RATES) * len(seeds)} oracle-repair experiments...\n")
    rows = []
    for error_type in ERROR_TYPES:
        for rate in RATES:
            for seed in seeds:
                X_dirty, y_dirty, log_df = corrupt_with_id(error_type, X_train, y_train, rate=rate, seed=seed)
                context = f"{error_type} rate={rate:.0%} seed={seed}"

                dirty_pipeline = build_rf_pipeline(X_dirty, seed=seed)
                with log_training("phase8", f"{context} (dirty)", dirty_pipeline, X_dirty, seed=seed):
                    dirty_pipeline.fit(X_dirty, y_dirty)
                dirty_f1 = evaluate(dirty_pipeline, X_test, y_test)["f1"]

                X_repaired, y_repaired = oracle_repair(error_type, X_dirty, y_dirty, log_df)
                repaired_pipeline = build_rf_pipeline(X_repaired, seed=seed)
                with log_training("phase8", f"{context} (oracle_repaired)", repaired_pipeline, X_repaired, seed=seed):
                    repaired_pipeline.fit(X_repaired, y_repaired)
                repaired_f1 = evaluate(repaired_pipeline, X_test, y_test)["f1"]

                clean_f1 = baseline_f1[seed]
                recovery = repaired_f1 - dirty_f1
                damage = clean_f1 - dirty_f1
                recovery_rate = recovery / damage if abs(damage) > 1e-9 else np.nan

                rows.append({
                    "error_type": error_type, "rate": rate, "seed": seed,
                    "clean_f1": clean_f1, "dirty_f1": dirty_f1, "repaired_f1": repaired_f1,
                    "damage": damage, "recovery": recovery, "recovery_rate": recovery_rate,
                })
                print(f"  {error_type:20s} rate={rate:4.0%} seed={seed:3d}  "
                      f"dirty={dirty_f1:.4f}  repaired={repaired_f1:.4f}  "
                      f"recovery={recovery:+.4f}  recovery_rate={recovery_rate:.2f}")

    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    summary = results.groupby(["error_type", "rate"]).agg(
        mean_dirty_f1=("dirty_f1", "mean"),
        mean_repaired_f1=("repaired_f1", "mean"),
        mean_damage=("damage", "mean"),
        mean_recovery=("recovery", "mean"),
        mean_recovery_rate=("recovery_rate", "mean"),
        std_recovery_rate=("recovery_rate", "std"),
        n_seeds=("seed", "nunique"),
    ).reset_index()
    return summary.sort_values(["error_type", "rate"])


if __name__ == "__main__":
    import sys

    seeds = [42, 43, 44, 45, 46] if "--full" in sys.argv else [42]

    results = run_experiment(seeds)
    summary = summarize(results)

    print("\n" + "=" * 100)
    print("PHASE 8 SUMMARY -- Oracle-repair ceiling per error type / rate" +
          (" (single seed)" if len(seeds) == 1 else ""))
    print("=" * 100)
    print(summary.round(4).to_string(index=False))

    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(results_dir, exist_ok=True)
    suffix = "full" if len(seeds) > 1 else "single_seed"
    results.to_csv(os.path.join(results_dir, f"phase8_oracle_repair_raw_{suffix}.csv"), index=False)
    summary.to_csv(os.path.join(results_dir, f"phase8_oracle_repair_summary_{suffix}.csv"), index=False)
    print(f"\nSaved phase8_oracle_repair_raw_{suffix}.csv and phase8_oracle_repair_summary_{suffix}.csv")
