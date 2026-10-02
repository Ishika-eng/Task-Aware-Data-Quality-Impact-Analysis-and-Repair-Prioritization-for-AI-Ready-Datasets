"""
Automated Repair Engine (Phase 9).

Unlike Phase 8's oracle repair (which cheats by reading the corruption
log), this is the real system: detect using Phase 5's detectors (no ground
truth), then repair ONLY the cells the detector is confident about,
leaving uncertain ones flagged rather than touched. Graded on TWO
independent axes, since a repair method could score well on one while
failing the other:

    Repair correctness          Task impact
    - repair precision          - repaired F1
    - repair recall             - recovery        (repaired_f1 - dirty_f1)
    - correct-value restoration - recovery_rate_vs_clean  (recovery / damage)
                                 - recovery_rate_vs_oracle (recovery / oracle's
                                   recovery -- how much of the THEORETICAL
                                   ceiling from Phase 8 did we actually reach)

Conservative two-tier policy (per the project's explicit design goal: don't
blindly modify every detector-flagged cell):

    - missing_values / duplicates: detection is EXACT (a cell either is NaN
      or it isn't; a row either is an exact copy or it isn't) -- no
      ambiguity, so every detected case is auto-repaired.
    - outliers / label_errors / feature_corruption: detection is a
      STATISTICAL ESTIMATE. We use Phase 5's threshold as the "flag for
      review" boundary, and a STRICTER secondary threshold as the
      "confident enough to auto-repair" boundary. Only the stricter subset
      gets modified; everything else is flagged but left alone. Both
      thresholds are read off the SAME underlying score (oof_confidence,
      IQR distance, or the cross-feature prediction's score) rather than
      re-running the expensive cross-validated detectors twice.
"""

import os

import numpy as np
import pandas as pd

from sklearn.impute import KNNImputer

from baseline import evaluate
from corruption import RATES, corrupt_with_id
from data import load_adult_clean, three_way_split, column_types
from detect import detect_missing, detect_duplicates, detect_outliers_iqr, detect_label_noise, \
    detect_feature_corruption_crossfeature
from phase7_impact import SPLIT_SEED, build_rf_pipeline
from phase8_oracle_repair import oracle_repair
from training_log import log_training

ERROR_TYPES = ["missing_values", "label_errors", "duplicates", "outliers", "feature_corruption"]

# error types whose corruption/repair is scoped to a specific CELL rather
# than a whole row -- must be graded cell-by-cell (same lesson as Phase 5).
CELL_LEVEL_ERRORS = {"outliers", "feature_corruption"}

# Stricter secondary thresholds for the "confident enough to auto-repair"
# tier, vs. Phase 5's (looser) "flag for review" thresholds.
AUTO_REPAIR_LABEL_CONFIDENCE = 0.97      # flag threshold (Phase 5) was 0.90
AUTO_REPAIR_OUTLIER_K = 3.0              # flag threshold (Phase 5) was 1.5
AUTO_REPAIR_OUTLIER_FALLBACK_Q = (0.001, 0.999)  # flag fallback was (0.005, 0.995)
AUTO_REPAIR_FEATURE_CAT_THRESHOLD = 0.01  # flag threshold (Phase 5) was 0.05
AUTO_REPAIR_FEATURE_NUM_Z = 6.0            # flag threshold (Phase 5) was 4.0


def detect_and_repair(error_type: str, X_dirty: pd.DataFrame, y_dirty: pd.Series, seed: int):
    """Returns (X_repaired, y_repaired, repaired_set, flagged_count).

    repaired_set is what actually got MODIFIED (the auto-repair tier only):
    a set of row_ids for row-level errors, or (row_id, column) tuples for
    cell-level errors. flagged_count is how many additional cases were
    flagged for review but NOT touched (useful to report, not graded).
    """
    categorical_cols, numeric_cols = column_types(X_dirty)
    X_repaired, y_repaired = X_dirty.copy(), y_dirty.copy()
    X_repaired[numeric_cols] = X_repaired[numeric_cols].astype(float)  # int cols can't hold repaired floats

    if error_type == "missing_values":
        result = detect_missing(X_dirty)
        cell_mask = result.extra["cell_mask"]
        repaired_set = set()

        # Numeric columns: KNNImputer, not plain median. This matters: the
        # shared preprocessing pipeline (baseline.py) ALREADY median-imputes
        # internally (needed since RandomForest can't take NaN), so a
        # median-based "repair" here would be a complete no-op -- the dirty
        # pipeline would silently do the exact same thing on its own,
        # making dirty_f1 and repaired_f1 identical by construction rather
        # than by the repair actually doing nothing useful. KNN imputation
        # (predict each missing value from similar rows' numeric columns)
        # is a genuinely different, more informed estimate.
        numeric_missing_mask = cell_mask[numeric_cols]
        if numeric_missing_mask.values.any():
            # Standardize first (ignoring NaN) so KNN distance isn't
            # dominated by large-scale columns like fnlwgt (~100,000s) vs
            # age (~10s); KNNImputer itself handles the NaN entries,
            # computing distance only over each pair's shared non-missing
            # dimensions.
            means = X_repaired[numeric_cols].mean()
            stds = X_repaired[numeric_cols].std().replace(0, 1)
            scaled = (X_repaired[numeric_cols] - means) / stds
            imputer = KNNImputer(n_neighbors=5)
            imputed_scaled = pd.DataFrame(
                imputer.fit_transform(scaled), index=X_repaired.index, columns=numeric_cols)
            imputed_numeric = imputed_scaled * stds + means
            for col in numeric_cols:
                flagged_rows = cell_mask.index[cell_mask[col]]
                if len(flagged_rows) == 0:
                    continue
                X_repaired.loc[flagged_rows, col] = imputed_numeric.loc[flagged_rows, col]
                repaired_set |= {(r, col) for r in flagged_rows}

        for col in categorical_cols:
            flagged_rows = cell_mask.index[cell_mask[col]]
            if len(flagged_rows) == 0:
                continue
            fill_value = X_dirty[col].mode(dropna=True)
            fill_value = fill_value.iloc[0] if len(fill_value) else "missing"
            X_repaired.loc[flagged_rows, col] = fill_value
            repaired_set |= {(r, col) for r in flagged_rows}
        return X_repaired, y_repaired, repaired_set, 0

    if error_type == "duplicates":
        result = detect_duplicates(X_dirty)
        flagged_rows = set(result.row_mask[result.row_mask].index)
        X_repaired = X_repaired.drop(index=list(flagged_rows))
        y_repaired = y_repaired.drop(index=list(flagged_rows))
        return X_repaired, y_repaired, flagged_rows, 0

    if error_type == "outliers":
        flagged = detect_outliers_iqr(X_dirty)
        auto = detect_outliers_iqr(X_dirty, k=AUTO_REPAIR_OUTLIER_K,
                                    fallback_lower_q=AUTO_REPAIR_OUTLIER_FALLBACK_Q[0],
                                    fallback_upper_q=AUTO_REPAIR_OUTLIER_FALLBACK_Q[1])
        n_flagged_only = int(flagged.row_mask.sum() - auto.row_mask.sum())

        numeric = X_dirty.select_dtypes("number")
        q1, q3 = numeric.quantile(0.25), numeric.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr  # repair TO the looser (flag-tier) bound
        repaired_set = set()
        for col, col_flags in auto.extra["per_column_flags"].items():
            flagged_rows = col_flags.index[col_flags]
            for row_id in flagged_rows:
                value = X_dirty.at[row_id, col]
                X_repaired.at[row_id, col] = min(max(value, lower[col]), upper[col])
                repaired_set.add((row_id, col))
        return X_repaired, y_repaired, repaired_set, n_flagged_only

    if error_type == "label_errors":
        result = detect_label_noise(X_dirty, y_dirty, confidence_threshold=0.9, seed=seed)
        oof_pred, oof_conf = result.extra["oof_pred"], result.extra["oof_confidence"]
        flagged_rows = set(result.row_mask[result.row_mask].index)
        auto_rows = set(oof_conf.index[(oof_conf >= AUTO_REPAIR_LABEL_CONFIDENCE) &
                                        (oof_pred != y_dirty)])
        n_flagged_only = len(flagged_rows - auto_rows)
        for row_id in auto_rows:
            y_repaired.loc[row_id] = oof_pred.loc[row_id]
        return X_repaired, y_repaired, auto_rows, n_flagged_only

    if error_type == "feature_corruption":
        result = detect_feature_corruption_crossfeature(X_dirty, cat_confidence_threshold=0.05,
                                                          num_z_threshold=4.0, seed=seed)
        repaired_set = set()
        n_flagged_only = 0
        for col, info in result.extra["per_column"].items():
            score = info["score"]
            if info["kind"] == "categorical":
                auto_mask = score < AUTO_REPAIR_FEATURE_CAT_THRESHOLD
                flag_mask = result.extra["per_column_flags"][col]
            else:
                auto_mask = score > AUTO_REPAIR_FEATURE_NUM_Z
                flag_mask = result.extra["per_column_flags"][col]
            n_flagged_only += int(flag_mask.sum() - auto_mask.sum())
            flagged_rows = auto_mask.index[auto_mask]
            for row_id in flagged_rows:
                X_repaired.at[row_id, col] = info["predicted_value"].loc[row_id]
                repaired_set.add((row_id, col))
        return X_repaired, y_repaired, repaired_set, n_flagged_only

    raise ValueError(f"no repair defined for {error_type}")


def grade_repair(error_type: str, log_df: pd.DataFrame, repaired_set: set,
                  X_dirty: pd.DataFrame, X_repaired: pd.DataFrame) -> dict:
    """Repair correctness: precision/recall of WHICH cells we touched, plus
    correct-value restoration (of the cells we correctly identified as
    corrupted AND repaired, how many did we restore close to the truth)."""
    if error_type in CELL_LEVEL_ERRORS:
        true_set = set(zip(log_df["row_id"], log_df["column"]))
    elif error_type == "label_errors":
        true_set = set(log_df["row_id"])
    elif error_type == "duplicates":
        true_set = set(log_df["row_id"])
    else:  # missing_values
        true_set = set(zip(log_df["row_id"], log_df["column"]))

    tp_set = true_set & repaired_set
    fp = len(repaired_set - true_set)
    fn = len(true_set - repaired_set)
    tp = len(tp_set)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0

    # Correct-value restoration: among true positives, did the repaired
    # value end up close to the TRUE original value? (duplicates has no
    # "value" to compare -- restoration there just means the row is gone.)
    restored_correctly = 0
    if error_type == "duplicates":
        restored_correctly = tp  # dropping IS the correct restoration
        n_checkable = tp
    elif tp > 0:
        n_checkable = tp
        if error_type in CELL_LEVEL_ERRORS or error_type == "missing_values":
            log_lookup = log_df.set_index(["row_id", "column"])["original_value"]
            for row_id, col in tp_set:
                true_val = log_lookup.loc[(row_id, col)]
                repaired_val = X_repaired.at[row_id, col]
                if _values_close(true_val, repaired_val):
                    restored_correctly += 1
        else:  # label_errors: tp_set is a set of row_ids, grading done by caller (needs y)
            n_checkable = 0
    else:
        n_checkable = 0

    return {"precision": precision, "recall": recall, "tp": tp, "fp": fp, "fn": fn,
            "restored_correctly": restored_correctly, "n_checkable": n_checkable}


def _values_close(true_val, repaired_val) -> bool:
    if pd.isna(true_val) or pd.isna(repaired_val):
        return False
    if isinstance(true_val, str) or isinstance(repaired_val, str):
        return str(true_val) == str(repaired_val)
    try:
        true_val, repaired_val = float(true_val), float(repaired_val)
    except (TypeError, ValueError):
        return str(true_val) == str(repaired_val)
    if true_val == 0:
        return abs(repaired_val) < 1e-6
    return abs(repaired_val - true_val) / abs(true_val) < 0.10  # within 10%


def run_experiment(seeds: list[int]) -> pd.DataFrame:
    X, y = load_adult_clean()
    X_train, X_val, X_test, y_train, y_val, y_test = three_way_split(X, y, seed=SPLIT_SEED)

    print(f"Computing clean baseline F1 for {len(seeds)} seed(s)...")
    baseline_f1 = {}
    for seed in seeds:
        pipeline = build_rf_pipeline(X_train, seed=seed)
        with log_training("phase9", f"clean_baseline seed={seed}", pipeline, X_train, seed=seed):
            pipeline.fit(X_train, y_train)
        baseline_f1[seed] = evaluate(pipeline, X_test, y_test)["f1"]
        print(f"  seed={seed}: clean F1 = {baseline_f1[seed]:.4f}")

    print(f"\nRunning {len(ERROR_TYPES)} error types x {len(RATES)} rates x {len(seeds)} seed(s) "
          f"= {len(ERROR_TYPES) * len(RATES) * len(seeds)} repair experiments...\n")
    rows = []
    for error_type in ERROR_TYPES:
        for rate in RATES:
            for seed in seeds:
                X_dirty, y_dirty, log_df = corrupt_with_id(error_type, X_train, y_train, rate=rate, seed=seed)

                context = f"{error_type} rate={rate:.0%} seed={seed}"

                dirty_pipeline = build_rf_pipeline(X_dirty, seed=seed)
                with log_training("phase9", f"{context} (dirty)", dirty_pipeline, X_dirty, seed=seed):
                    dirty_pipeline.fit(X_dirty, y_dirty)
                dirty_f1 = evaluate(dirty_pipeline, X_test, y_test)["f1"]

                # Real repair
                X_repaired, y_repaired, repaired_set, n_flagged_only = detect_and_repair(
                    error_type, X_dirty, y_dirty, seed=seed)
                repaired_pipeline = build_rf_pipeline(X_repaired, seed=seed)
                with log_training("phase9", f"{context} (auto_repaired)", repaired_pipeline, X_repaired, seed=seed):
                    repaired_pipeline.fit(X_repaired, y_repaired)
                repaired_f1 = evaluate(repaired_pipeline, X_test, y_test)["f1"]

                # Oracle ceiling, for recovery_rate_vs_oracle
                X_oracle, y_oracle = oracle_repair(error_type, X_dirty, y_dirty, log_df)
                oracle_pipeline = build_rf_pipeline(X_oracle, seed=seed)
                with log_training("phase9", f"{context} (oracle_repaired)", oracle_pipeline, X_oracle, seed=seed):
                    oracle_pipeline.fit(X_oracle, y_oracle)
                oracle_f1 = evaluate(oracle_pipeline, X_test, y_test)["f1"]

                clean_f1 = baseline_f1[seed]
                damage = clean_f1 - dirty_f1
                recovery = repaired_f1 - dirty_f1
                oracle_recovery = oracle_f1 - dirty_f1
                recovery_rate_vs_clean = recovery / damage if abs(damage) > 1e-9 else np.nan
                recovery_rate_vs_oracle = recovery / oracle_recovery if abs(oracle_recovery) > 1e-9 else np.nan

                correctness = grade_repair(error_type, log_df, repaired_set, X_dirty, X_repaired)
                if error_type == "label_errors" and correctness["n_checkable"] == 0 and correctness["tp"] > 0:
                    tp_rows = set(log_df["row_id"]) & repaired_set
                    log_lookup = log_df.set_index("row_id")["original_value"]
                    restored = sum(1 for r in tp_rows if str(log_lookup.loc[r]) == str(y_repaired.loc[r]))
                    correctness["restored_correctly"] = restored
                    correctness["n_checkable"] = len(tp_rows)

                rows.append({
                    "error_type": error_type, "rate": rate, "seed": seed,
                    "clean_f1": clean_f1, "dirty_f1": dirty_f1, "repaired_f1": repaired_f1,
                    "oracle_f1": oracle_f1, "damage": damage, "recovery": recovery,
                    "recovery_rate_vs_clean": recovery_rate_vs_clean,
                    "recovery_rate_vs_oracle": recovery_rate_vs_oracle,
                    "repair_precision": correctness["precision"], "repair_recall": correctness["recall"],
                    "tp": correctness["tp"], "fp": correctness["fp"], "fn": correctness["fn"],
                    "restored_correctly": correctness["restored_correctly"],
                    "n_checkable": correctness["n_checkable"],
                    "n_flagged_not_repaired": n_flagged_only,
                })
                restoration_pct = (correctness["restored_correctly"] / correctness["n_checkable"] * 100
                                    if correctness["n_checkable"] else float("nan"))
                print(f"  {error_type:20s} rate={rate:4.0%} seed={seed:3d}  "
                      f"dirty={dirty_f1:.4f} repaired={repaired_f1:.4f} oracle={oracle_f1:.4f}  "
                      f"repair_P={correctness['precision']:.2f} repair_R={correctness['recall']:.2f}  "
                      f"restoration={restoration_pct:.0f}%  recovery_vs_oracle={recovery_rate_vs_oracle:.2f}")

    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    summary = results.groupby(["error_type", "rate"]).agg(
        mean_dirty_f1=("dirty_f1", "mean"), mean_repaired_f1=("repaired_f1", "mean"),
        mean_oracle_f1=("oracle_f1", "mean"),
        mean_recovery=("recovery", "mean"),
        mean_recovery_rate_vs_clean=("recovery_rate_vs_clean", "mean"),
        mean_recovery_rate_vs_oracle=("recovery_rate_vs_oracle", "mean"),
        mean_repair_precision=("repair_precision", "mean"), mean_repair_recall=("repair_recall", "mean"),
        total_restored_correctly=("restored_correctly", "sum"), total_checkable=("n_checkable", "sum"),
        n_seeds=("seed", "nunique"),
    ).reset_index()
    summary["restoration_rate"] = summary["total_restored_correctly"] / summary["total_checkable"].replace(0, np.nan)
    return summary.sort_values(["error_type", "rate"])


if __name__ == "__main__":
    import sys

    seeds = [42, 43, 44, 45, 46] if "--full" in sys.argv else [42]

    results = run_experiment(seeds)
    summary = summarize(results)

    print("\n" + "=" * 110)
    print("PHASE 9 SUMMARY -- Automated repair vs. dirty baseline and oracle ceiling" +
          (" (single seed)" if len(seeds) == 1 else ""))
    print("=" * 110)
    print(summary.round(3).to_string(index=False))

    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(results_dir, exist_ok=True)
    suffix = "full" if len(seeds) > 1 else "single_seed"
    results.to_csv(os.path.join(results_dir, f"phase9_repair_raw_{suffix}.csv"), index=False)
    summary.to_csv(os.path.join(results_dir, f"phase9_repair_summary_{suffix}.csv"), index=False)
    print(f"\nSaved phase9_repair_raw_{suffix}.csv and phase9_repair_summary_{suffix}.csv")
