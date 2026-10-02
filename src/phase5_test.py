"""
Grades the Phase 5 detectors against Phase 3-4's ground-truth corruption
log: for each error type, corrupt X_train with ONLY that error (rate=10%),
run the matching detector on the resulting dirty data (which has no idea
what was corrupted), and compute precision/recall against the rows we
know for a fact were corrupted.
"""

import os

import pandas as pd

from corruption import CORRUPTORS
from data import load_adult_clean, three_way_split
from detect import (detect_missing, detect_duplicates, detect_outliers_iqr,
                     detect_label_noise, detect_feature_corruption_crossfeature, detect_inconsistency)

SEED = 42
RATE = 0.10

# corruption engine name -> matching detector function
DETECTORS = {
    "missing_values": lambda X, y: detect_missing(X),
    "duplicates": lambda X, y: detect_duplicates(X),
    "outliers": lambda X, y: detect_outliers_iqr(X),
    "label_errors": lambda X, y: detect_label_noise(X, y, seed=SEED),
    "feature_corruption": lambda X, y: detect_feature_corruption_crossfeature(X, seed=SEED),
}


# error types whose corruption (and therefore ground truth) is naturally
# scoped to a specific CELL (row, column) rather than the whole row -- these
# must be graded cell-by-cell, or a detector gets undeserved credit for
# flagging the right row for the wrong reason (e.g. a different column
# happened to look anomalous). missing_values is also cell-level in how
# it's injected, but its detector is a cell-exact check by construction
# (isna() on the literal same cells) so row- vs cell-level give identical
# scores -- no need to complicate it. label_errors/duplicates have no
# "column" axis (one label / one row-copy per event), so row-level is the
# only sensible unit there.
CELL_LEVEL_ERRORS = {"outliers", "feature_corruption"}


def precision_recall_f1(true_set: set, detected_set: set) -> dict:
    tp = len(true_set & detected_set)
    fp = len(detected_set - true_set)
    fn = len(true_set - detected_set)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def main():
    X, y = load_adult_clean()
    X_train, X_val, X_test, y_train, y_val, y_test = three_way_split(X, y, seed=SEED)

    print(f"Evaluating Phase 5 detectors at corruption rate={RATE:.0%}\n")
    rows = []
    for error_type, corrupt_fn in CORRUPTORS.items():
        X_dirty, y_dirty, log_df = corrupt_fn(X_train, y_train, rate=RATE, seed=SEED)
        detector = DETECTORS[error_type]
        result = detector(X_dirty, y_dirty)

        if error_type in CELL_LEVEL_ERRORS:
            true_set = set(zip(log_df["row_id"], log_df["column"]))
            detected_set = set()
            for col, flags in result.extra["per_column_flags"].items():
                for row_id in flags[flags].index:
                    detected_set.add((row_id, col))
            unit = "cells"
        else:
            true_set = set(log_df["row_id"].unique())
            detected_set = set(result.row_mask[result.row_mask].index)
            unit = "rows"

        metrics = precision_recall_f1(true_set, detected_set)
        metrics["error_type"] = error_type
        metrics["unit"] = unit
        metrics["true_count"] = len(true_set)
        metrics["detected_count"] = len(detected_set)
        rows.append(metrics)

        print(f"{error_type:20s} [{unit:5s}] true={len(true_set):5d}  detected={len(detected_set):5d}  "
              f"precision={metrics['precision']:.3f}  recall={metrics['recall']:.3f}  f1={metrics['f1']:.3f}")

    # inconsistency has no corruptor to grade against -- just show its
    # natural (expected near-zero) rate on this already-canonical dataset
    X_dirty_any, _, _ = CORRUPTORS["missing_values"](X_train, y_train, rate=RATE, seed=SEED)
    inconsistency_result = detect_inconsistency(X_train)
    print(f"\n{'inconsistency':20s} (no injected ground truth -- Adult's categoricals are "
          f"already canonical) rate={inconsistency_result.rate:.4f}")

    report = pd.DataFrame(rows)[
        ["error_type", "unit", "true_count", "detected_count", "precision", "recall", "f1", "tp", "fp", "fn"]
    ]
    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, "phase5_detection_scores.csv")
    report.to_csv(out_path, index=False)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
