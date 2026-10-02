"""
Dataset Quality Report (Phase 6).

Packages Phase 5's detectors into the summary profile from the brief:

    DATA QUALITY REPORT
    Rows: 32,561
    Columns: 14
    Missing Values:         7.4%
    Duplicates:             4.2%
    Potential Outliers:     3.8%
    Potential Label Errors: 2.1%
    Inconsistencies:        1.7%

One addition beyond the brief: since Phase 5 established that these
detectors are NOT equally trustworthy (missing/duplicates are exact
checks; outliers/label-errors/feature-corruption are statistical
estimates with known, imperfect precision/recall), the report annotates
each uncertain line with that detector's measured reliability instead of
presenting all five numbers with false equal confidence.
"""

import os

import pandas as pd

from corruption import CORRUPTORS
from data import load_adult_clean, three_way_split
from detect import run_full_audit

SEED = 42

# Which detectors are exact checks vs. statistical estimates (from Phase 5
# results) -- controls how the report labels each line.
EXACT_DETECTORS = {"missing_values", "duplicates"}

# Apply duplicates LAST since it's the only corruptor that changes the
# dataset's row count -- applying it first would mean every other
# corruptor also operates on (and re-corrupts) the duplicated rows.
COMBINED_APPLY_ORDER = ["missing_values", "outliers", "label_errors", "feature_corruption", "duplicates"]


def build_combined_dirty(X: pd.DataFrame, y: pd.Series, rate: float = 0.10, seed: int = SEED):
    """Inject all 5 error types into the SAME dataset, one after another, to
    simulate a realistically messy dataset (real dirty data rarely has just
    one problem). Returns the combined dirty (X, y) and a combined log.

    IMPORTANT SCOPE NOTE: this combined dataset is a STRESS TEST for the
    Quality Report (Phase 6) -- it shows the report handling multiple
    simultaneous, interacting problems. It is NOT used to measure any
    single error type's isolated impact on the model: once 5 error types
    coexist, they interact (e.g. an injected missing value changes what the
    feature-corruption detector sees; an outlier shifts the label-noise
    model's decision boundary), so a flagged rate here isn't directly
    comparable to the rate we injected. Phase 7's controlled, one-error-at-
    a-time experiments are the right tool for measuring isolated impact --
    this function is deliberately not used there.
    """
    X_dirty, y_dirty = X.copy(), y.copy()
    logs = []
    for error_type in COMBINED_APPLY_ORDER:
        X_dirty, y_dirty, log_df = CORRUPTORS[error_type](X_dirty, y_dirty, rate=rate, seed=seed)
        logs.append(log_df)
    combined_log = pd.concat(logs, ignore_index=True)
    return X_dirty, y_dirty, combined_log


def load_detector_reliability() -> pd.DataFrame | None:
    """Phase 5's graded precision/recall per error type, if available."""
    path = os.path.join(os.path.dirname(__file__), "..", "results", "phase5_detection_scores.csv")
    if os.path.exists(path):
        return pd.read_csv(path).set_index("error_type")
    return None


# detect.py's run_full_audit() names its label detector "label_noise" (it
# describes what it looks for); corruption/ and phase5_test.py name the
# corresponding error "label_errors" (it describes what was injected). Same
# detector, two vocabularies -- map between them here rather than renaming
# either module, since both names are individually well-justified.
AUDIT_NAME_TO_BENCHMARK_NAME = {"label_noise": "label_errors"}


def generate_quality_report(X: pd.DataFrame, y: pd.Series, seed: int = SEED) -> dict:
    audit = run_full_audit(X, y, seed=seed)
    reliability = load_detector_reliability()

    report = {
        "n_rows": len(X),
        "n_columns": X.shape[1],
        "issues": {},
    }
    for name, result in audit.items():
        entry = {"rate": result.rate, "exact": name in EXACT_DETECTORS}
        benchmark_name = AUDIT_NAME_TO_BENCHMARK_NAME.get(name, name)
        if reliability is not None and benchmark_name in reliability.index:
            entry["precision"] = reliability.loc[benchmark_name, "precision"]
            entry["recall"] = reliability.loc[benchmark_name, "recall"]
        report["issues"][name] = entry
    return report


# Deliberately "anomalies"/"potential", not "corruption"/"errors": these
# detectors are statistical estimates, not confirmed findings. Calling the
# feature-corruption detector's output "Potential Feature Corruption" (as
# an earlier version of this report did) implies more certainty than a
# detector with 0.28 precision has earned.
LABELS = {
    "missing_values": "Missing Values",
    "duplicates": "Duplicate Rows",
    "outliers": "Potential Outliers",
    "label_noise": "Potential Label Errors",
    "feature_corruption": "Potential Feature Anomalies",
    "inconsistency": "Inconsistencies",
}


def print_report(report: dict, title: str = "DATA QUALITY REPORT"):
    """Two separate sections, deliberately not merged into one table:

    1. DATA QUALITY -- what was flagged (an observed rate).
    2. DETECTOR RELIABILITY -- how much to trust each flag (precision/recall
       from the Phase 5 benchmark).

    Keeping these apart (rather than e.g. multiplying rate by precision
    into a single "estimated true rate") is intentional: precision was
    measured on a specific synthetic corruption distribution and doesn't
    necessarily transfer to this dataset's actual error composition. A
    single blended number would quietly smuggle in that assumption.
    """
    print("=" * 70)
    print(title)
    print("=" * 70)
    print(f"Rows: {report['n_rows']:,}")
    print(f"Columns: {report['n_columns']}")
    print()

    exact_lines, uncertain_lines, reliability_lines = [], [], []
    for name, entry in report["issues"].items():
        label = LABELS.get(name, name)
        if entry["exact"]:
            exact_lines.append(f"{label:28s} {entry['rate']*100:5.1f}%")
        else:
            marker = "*" if "precision" in entry else ""
            uncertain_lines.append(f"{label:28s} {entry['rate']*100:5.1f}% flagged{marker}")
            if "precision" in entry:
                reliability_lines.append(f"{label:28s} precision={entry['precision']*100:5.1f}%   "
                                          f"recall={entry['recall']*100:5.1f}%")

    for line in exact_lines:
        print(line)
    print()
    for line in uncertain_lines:
        print(line)
    print()

    if reliability_lines:
        print("-" * 70)
        print("DETECTOR RELIABILITY (from the Phase 5 benchmark)")
        print("-" * 70)
        for line in reliability_lines:
            print(line)
        print()
        print("* Precision/recall measured on controlled synthetic corruption at a")
        print("  known rate (Phase 5) -- it may not generalize exactly to this")
        print("  dataset's actual error composition. A flagged rate is NOT the same")
        print("  as a confirmed error rate; do not read '47.9% flagged' as '47.9% of")
        print("  the dataset is actually corrupted.'")
    print()


def main():
    X, y = load_adult_clean()
    X_train, X_val, X_test, y_train, y_val, y_test = three_way_split(X, y, seed=SEED)

    print("Building a combined dirty dataset (all 5 error types injected together, rate=10%)\n")
    X_dirty, y_dirty, combined_log = build_combined_dirty(X_train, y_train, rate=0.10, seed=SEED)

    report = generate_quality_report(X_dirty, y_dirty, seed=SEED)
    print_report(report, title="DATA QUALITY REPORT -- Synthetic Combined-Dirty Training Set")

    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(results_dir, exist_ok=True)

    flat_rows = [{"issue": name, **entry} for name, entry in report["issues"].items()]
    pd.DataFrame(flat_rows).to_csv(os.path.join(results_dir, "phase6_quality_report_synthetic.csv"), index=False)

    # Second example: the REAL original Adult data (before we dropped its
    # native missing rows) -- a genuinely messy real-world dataset, not one
    # we corrupted ourselves. duplicates/outliers/label-noise/feature-corruption
    # detectors still run on it (they don't need injected ground truth to
    # operate), only their results can't be graded against a known answer here.
    print("\nRunning the same report on the REAL (uncleaned) Adult dataset for comparison\n")
    from sklearn.datasets import fetch_openml
    raw = fetch_openml("adult", version=2, as_frame=True).frame
    y_raw = raw["class"].astype(str)
    X_raw = raw.drop(columns=["class"])
    real_report = generate_quality_report(X_raw, y_raw, seed=SEED)
    print_report(real_report, title="DATA QUALITY REPORT -- Real UCI Adult (uncleaned)")

    flat_rows_real = [{"issue": name, **entry} for name, entry in real_report["issues"].items()]
    pd.DataFrame(flat_rows_real).to_csv(os.path.join(results_dir, "phase6_quality_report_real.csv"), index=False)
    print(f"Saved phase6_quality_report_synthetic.csv and phase6_quality_report_real.csv to {results_dir}")


if __name__ == "__main__":
    main()
