"""
Demonstrates the Phase 11C multidimensional readiness report on two
datasets: a synthetic combined-dirty training set (all 5 errors injected
together, reusing Phase 6's stress-test construction) and the real
uncleaned UCI Adult dataset (genuine missingness, no injected corruption).
"""

from data import load_adult_clean, three_way_split
from phase11_readiness import generate_readiness_report, print_readiness_report, SPLIT_SEED
from quality_report import build_combined_dirty

if __name__ == "__main__":
    X, y = load_adult_clean()
    X_train, X_val, X_test, y_train, y_val, y_test = three_way_split(X, y, seed=SPLIT_SEED)

    print("Building combined-dirty dataset (all 5 errors injected together, rate=10%)...\n")
    X_dirty, y_dirty, _log = build_combined_dirty(X_train, y_train, rate=0.10, seed=SPLIT_SEED)
    report = generate_readiness_report(X_dirty, y_dirty, dataset_name="Synthetic combined-dirty (rate=10%)",
                                        target_name="class (income >50K / <=50K)", seed=SPLIT_SEED)
    print_readiness_report(report)

    print("\n\n")
    from sklearn.datasets import fetch_openml
    raw = fetch_openml("adult", version=2, as_frame=True).frame
    y_raw = raw["class"].astype(str)
    X_raw = raw.drop(columns=["class"])
    report_real = generate_readiness_report(X_raw, y_raw, dataset_name="Real UCI Adult (uncleaned)",
                                              target_name="class (income >50K / <=50K)", seed=SPLIT_SEED)
    print_readiness_report(report_real)
