"""
Duplicate-row corruption.

Appends copies of randomly chosen rows to the bottom of the dataset.
Unlike the other corruptors, this changes the dataset's SHAPE (more rows),
so its log is structured a little differently: rather than one
original-value/corrupted-value pair, each log row records which new row
id was created and which original row it was copied from.

Rate definition: `rate` fraction of the ORIGINAL training rows get
duplicated (e.g. rate=0.10 on 100,000 rows -> 10,000 unique rows are each
duplicated once -> 110,000 total rows). We sample source rows WITHOUT
replacement so this is a clean "unique 10% of rows, one duplicate each" --
not "random draws with replacement," which could duplicate the same row
multiple times while leaving other rows untouched.
"""

import numpy as np
import pandas as pd


def corrupt(X: pd.DataFrame, y: pd.Series, rate: float, seed: int = 0):
    rng = np.random.default_rng(seed)
    n_dupes = int(len(X) * rate)
    source_idx = rng.choice(X.index, size=n_dupes, replace=False)

    X_dupes = X.loc[source_idx].reset_index(drop=True)
    y_dupes = y.loc[source_idx].reset_index(drop=True)
    # New IDs must be guaranteed not to collide with X's EXISTING index
    # labels. range(len(X), len(X)+n_dupes) looks safe but isn't: X here is
    # typically a train split whose index is a scattered subset of the
    # original full dataset's row labels (train_test_split doesn't reset
    # the index), so values like 31655 can already exist as a genuine row
    # label even though X only has 31655 rows. Starting strictly above the
    # index's actual max is the only version of this that's always correct.
    next_id = (X.index.max() if len(X) else -1) + 1
    new_row_ids = range(next_id, next_id + n_dupes)
    X_dupes.index = new_row_ids
    y_dupes.index = new_row_ids

    X_dirty = pd.concat([X, X_dupes])
    y_dirty = pd.concat([y, y_dupes])

    log_rows = [{
        "row_id": new_id, "error_type": "duplicates", "column": None,
        "original_value": f"copy_of_row_{src}", "corrupted_value": None,
    } for new_id, src in zip(new_row_ids, source_idx)]

    log_df = pd.DataFrame(log_rows)
    return X_dirty, y_dirty, log_df
