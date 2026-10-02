"""
Controlled per-error impact experiment (Phase 7).

For each of the 5 error types x 3 corruption rates (5%/10%/20%), inject
ONLY that error into the clean training data, train Random Forest with the
EXACT SAME preprocessing/model config/test set as the Phase 2 baseline,
and measure the F1 drop relative to a clean-data baseline trained the same
way. No detectors are involved here at all -- this is an oracle experiment
(we know exactly what we corrupted because we did it), measuring ground-
truth impact, not detectability.

Design choices, and why:

- Everything (split, preprocessing, RF hyperparameters, test set,
  evaluation code) is held IDENTICAL across every run, reusing baseline.py's
  build_preprocessor/evaluate directly rather than reimplementing them --
  only error_type, rate, and seed are allowed to vary. This is what makes
  the resulting deltas attributable to the corruption alone.

- A single `seed` per run drives BOTH the corruption's randomness and the
  Random Forest's random_state. This means each of the N repetitions is a
  fully independent trial (not just "noisy corruption, identical model"),
  which is the right design for estimating variance across repetitions.
  The train/test SPLIT seed is held fixed at 42 for every run, including
  the baseline -- the test set must be the exact same rows every time.

- The clean baseline is recomputed PER SEED (not reused from Phase 2's
  single number) and paired with that seed's dirty run:
      damage(error, rate, seed) = baseline_f1(seed) - dirty_f1(error, rate, seed)
  This paired-difference design cancels out run-to-run RF training noise
  that has nothing to do with the corruption, giving a tighter estimate of
  the corruption's true effect than comparing against one fixed baseline
  number would.

- We rerun the clean baseline fresh here (seed=42) rather than trusting
  Phase 2's recorded 0.680, and explicitly check it matches, since Phase 7
  must use the identical procedure -- if anything had drifted, the
  recorded number would be wrong to lean on.
"""

import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from baseline import build_preprocessor, evaluate
from corruption import CORRUPTORS, RATES
from data import load_adult_clean, three_way_split
from training_log import log_training

SPLIT_SEED = 42  # train/test split is held fixed across every single run


def build_rf_pipeline(X: pd.DataFrame, seed: int) -> Pipeline:
    """Same preprocessing as baseline.py; Random Forest only (per the
    project's model decision from Phase 2: RF now, LogReg validation later).
    """
    preprocessor = build_preprocessor(X)
    return Pipeline([
        ("prep", preprocessor),
        ("clf", RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)),
    ])


def run_experiment(seeds: list[int]) -> pd.DataFrame:
    X, y = load_adult_clean()
    X_train, X_val, X_test, y_train, y_val, y_test = three_way_split(X, y, seed=SPLIT_SEED)

    print(f"Computing clean baseline F1 for {len(seeds)} seed(s) (paired reference per seed)...")
    baseline_f1 = {}
    for seed in seeds:
        pipeline = build_rf_pipeline(X_train, seed=seed)
        with log_training("phase7", f"clean_baseline seed={seed}", pipeline, X_train, seed=seed):
            pipeline.fit(X_train, y_train)
        metrics = evaluate(pipeline, X_test, y_test)
        baseline_f1[seed] = metrics["f1"]
        print(f"  seed={seed}: clean F1 = {metrics['f1']:.4f}")

    if SPLIT_SEED in baseline_f1:
        recorded = 0.6804  # from results/phase2_clean_baseline.csv (random_forest row)
        fresh = baseline_f1[SPLIT_SEED]
        match = "MATCHES" if abs(fresh - recorded) < 1e-3 else "DOES NOT MATCH"
        print(f"  Sanity check vs Phase 2 recorded baseline ({recorded}): {match} (fresh={fresh:.4f})")

    print(f"\nRunning {len(CORRUPTORS)} error types x {len(RATES)} rates x {len(seeds)} seed(s) "
          f"= {len(CORRUPTORS) * len(RATES) * len(seeds)} experiments...\n")
    rows = []
    for error_type, corrupt_fn in CORRUPTORS.items():
        for rate in RATES:
            for seed in seeds:
                X_dirty, y_dirty, _log_df = corrupt_fn(X_train, y_train, rate=rate, seed=seed)
                pipeline = build_rf_pipeline(X_dirty, seed=seed)
                context = f"{error_type} rate={rate:.0%} seed={seed}"
                with log_training("phase7", context, pipeline, X_dirty, seed=seed):
                    pipeline.fit(X_dirty, y_dirty)
                metrics = evaluate(pipeline, X_test, y_test)

                dirty_f1 = metrics["f1"]
                damage = baseline_f1[seed] - dirty_f1
                rows.append({
                    "error_type": error_type, "rate": rate, "seed": seed,
                    "baseline_f1": baseline_f1[seed], "dirty_f1": dirty_f1,
                    "f1_damage": damage,
                    **{f"dirty_{k}": v for k, v in metrics.items() if k != "f1"},
                })
                print(f"  {error_type:20s} rate={rate:4.0%} seed={seed:3d}  "
                      f"F1={dirty_f1:.4f}  damage={damage:+.4f}")

    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    summary = results.groupby(["error_type", "rate"]).agg(
        mean_f1=("dirty_f1", "mean"), std_f1=("dirty_f1", "std"),
        mean_damage=("f1_damage", "mean"), std_damage=("f1_damage", "std"),
        n_seeds=("seed", "nunique"),
    ).reset_index()
    return summary.sort_values(["error_type", "rate"])


if __name__ == "__main__":
    import sys

    # Start with a single seed to validate the pipeline (fast); pass
    # --full on the command line to run the full 5-seed experiment.
    seeds = [42, 43, 44, 45, 46] if "--full" in sys.argv else [42]

    results = run_experiment(seeds)
    summary = summarize(results)

    print("\n" + "=" * 90)
    print("PHASE 7 SUMMARY -- Impact per error type / rate" + (" (single seed, not yet averaged)" if len(seeds) == 1 else ""))
    print("=" * 90)
    print(summary.round(4).to_string(index=False))

    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(results_dir, exist_ok=True)
    suffix = "full" if len(seeds) > 1 else "single_seed"
    results.to_csv(os.path.join(results_dir, f"phase7_impact_raw_{suffix}.csv"), index=False)
    summary.to_csv(os.path.join(results_dir, f"phase7_impact_summary_{suffix}.csv"), index=False)
    print(f"\nSaved phase7_impact_raw_{suffix}.csv and phase7_impact_summary_{suffix}.csv")
