"""
Builds the demo dataset used by the TaskClean app and the end-to-end test:
a 6,000-row sample of the clean Adult data with all five error types injected
at 5% (Phase 3 corruptors, via quality_report.build_combined_dirty). The
corruption log is saved alongside so the product's behaviour can be checked
against ground truth.
"""

import os

from data import load_adult_clean
from quality_report import build_combined_dirty

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

if __name__ == "__main__":
    X, y = load_adult_clean()
    sample = X.sample(6000, random_state=7).reset_index(drop=True)
    y_sample = y.loc[X.sample(6000, random_state=7).index].reset_index(drop=True)

    X_dirty, y_dirty, log = build_combined_dirty(sample, y_sample, rate=0.05, seed=42)
    demo = X_dirty.copy()
    demo["class"] = y_dirty.values
    demo = demo.reset_index(drop=True)

    demo.to_csv(os.path.join(DATA_DIR, "demo_dirty_adult.csv"), index=False)
    log.to_csv(os.path.join(DATA_DIR, "demo_dirty_adult_corruption_log.csv"), index=False)
    print(f"demo_dirty_adult.csv: {demo.shape}")
    print(log.groupby("error_type").size())
