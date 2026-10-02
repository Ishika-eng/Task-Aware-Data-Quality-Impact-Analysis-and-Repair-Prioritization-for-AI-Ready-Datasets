"""
Sanity-checks the corruption engine (Phase 3) and ground-truth logging
(Phase 4): runs every corruptor at every rate on X_train/y_train only,
verifies the resulting dataset and log look right, and saves one combined
corruption_log.csv to results/.
"""

import os

import pandas as pd

from corruption import CORRUPTORS, RATES
from data import load_adult_clean, three_way_split

SEED = 42


def main():
    X, y = load_adult_clean()
    X_train, X_val, X_test, y_train, y_val, y_test = three_way_split(X, y, seed=SEED)
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}  (test set NOT touched)")

    all_logs = []
    for error_type, corrupt_fn in CORRUPTORS.items():
        for rate in RATES:
            X_dirty, y_dirty, log_df = corrupt_fn(X_train, y_train, rate=rate, seed=SEED)
            log_df = log_df.copy()
            log_df["rate"] = rate
            all_logs.append(log_df)

            print(f"{error_type:20s} rate={rate:4.0%} -> "
                  f"X_dirty {X_dirty.shape}, log rows: {len(log_df)}")

            # sanity checks
            assert len(X_dirty) == len(y_dirty)
            if error_type == "duplicates":
                assert len(X_dirty) == len(X_train) + int(len(X_train) * rate)
            else:
                assert len(X_dirty) == len(X_train)
                assert len(log_df) > 0

    combined_log = pd.concat(all_logs, ignore_index=True)

    # Unique ID per corruption event, assigned globally across the whole
    # combined log -- lets us trace cases where the same row was hit by
    # multiple corruptors (e.g. a row that got both a missing value AND a
    # label flip), since row_id alone isn't unique across error types/rates.
    combined_log.insert(0, "corruption_id", [f"C{i:06d}" for i in range(1, len(combined_log) + 1)])

    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, "corruption_log.csv")
    combined_log.to_csv(out_path, index=False)
    print(f"\nSaved {len(combined_log)} log rows to {out_path}")
    print("\nSample log rows:")
    print(combined_log.sample(8, random_state=SEED).to_string(index=False))


if __name__ == "__main__":
    main()
