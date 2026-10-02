"""
End-to-end invariant test for the TaskClean product layer.

The repair log is the product's proof of what it did, so the central claim
under test is:

    every cell that differs between the uploaded file and the cleaned file
    is a logged, applied repair -- and nothing else changed.

Run:  python3 test_taskclean.py        (uses data/demo_dirty_adult.csv)
"""

import json
import os

import numpy as np
import pandas as pd

from taskclean import apply_repairs, audit_dataset, default_apply_issues, outputs_as_bytes

DEMO = os.path.join(os.path.dirname(__file__), "..", "data", "demo_dirty_adult.csv")


def changed_cells(original: pd.DataFrame, cleaned: pd.DataFrame) -> set:
    """(row_id, column) pairs whose value differs, over rows that survive."""
    common = original.loc[cleaned.index]
    diff = set()
    for col in original.columns:
        a, b = common[col], cleaned[col]
        both_nan = a.isna() & b.isna()
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            same = np.isclose(a.astype(float), b.astype(float), equal_nan=True)
        else:
            same = (a.astype(object).astype(str) == b.astype(object).astype(str)) | both_nan
        diff |= {(r, col) for r in common.index[~np.asarray(same)]}
    return diff


def values_match(cleaned_value, applied_value) -> bool:
    try:
        return bool(np.isclose(float(cleaned_value), float(applied_value)))
    except (TypeError, ValueError):
        return str(cleaned_value) == str(applied_value)


def main():
    df = pd.read_csv(DEMO)
    state = audit_dataset(df, "class", include_feature_anomalies=False,
                          progress=lambda f, m: print(f"  [{f:4.0%}] {m}"))
    print("\nPolicy:\n", state.policy[["issue", "detected_rate", "n_repair_candidates", "n_flag_only",
                                       "default_action"]].round(4).to_string(index=False))

    # 1. Default policy: only evidence-approved repairs are applied.
    approved = default_apply_issues(state)
    print("\nPolicy-approved issues:", sorted(approved))
    assert approved == {"duplicates"}, "only exact duplicate removal should clear the safe-auto bar"

    result = apply_repairs(state)
    log, cleaned = result.repair_log, result.cleaned
    assert list(cleaned.columns) == list(df.columns), "columns/order must be preserved"

    applied = log[log.action == "applied"]
    dropped = set(df.index) - set(cleaned.index)
    assert dropped == set(applied.row_id), "dropped rows must be exactly the logged duplicate removals"
    assert changed_cells(df, cleaned) == set(), "default policy must not modify any surviving cell"
    assert not cleaned.duplicated().any(), "no full-record duplicates may remain after duplicate removal"
    print(f"OK default policy: {len(dropped)} duplicate rows dropped, no other cell changed")

    # 2. Human override: apply the review-tier repairs too; the log must still
    #    account for every changed cell, and say they were human overrides.
    override = {"duplicates", "missing_values", "outliers", "label_errors"}
    result2 = apply_repairs(state, override)
    log2, cleaned2 = result2.repair_log, result2.cleaned
    applied2 = log2[(log2.action == "applied") & (log2.problem != "Duplicate row")]
    assert set(applied2.row_id) <= set(cleaned2.index), \
        "the log must never claim a repair on a row that is absent from the cleaned file"
    logged_cells = {(int(r), c) for r, c in zip(applied2.row_id, applied2.column)}
    actual_cells = changed_cells(df, cleaned2)
    # a repair that writes the same value it replaced is a no-op: allowed
    assert actual_cells <= logged_cells, f"{len(actual_cells - logged_cells)} changed cells missing from the log"
    for r, c, v in zip(applied2.row_id, applied2.column, applied2.applied_value):
        assert values_match(cleaned2.at[r, c], v), f"log/cleaned mismatch at row {r}, column {c}"
    assert log2.loc[log2.action == "applied", "human_override"].sum() == len(applied2), \
        "every non-policy repair must be flagged as a human override"
    print(f"OK override run: {len(actual_cells)} changed cells, all present in the log "
          f"({len(applied2)} logged cell repairs)")

    # 3. Ground-truth sanity against the injected corruption.
    truth = pd.read_csv(DEMO.replace(".csv", "_corruption_log.csv"))
    injected_dupes = (truth.error_type == "duplicates").sum()
    detected_dupes = int(state.detections["duplicates"].row_mask.sum())
    injected_missing_cells = int(df.drop(columns="class").isna().values.sum())
    detected_missing_cells = int(state.detections["missing_values"].extra["cell_mask"].values.sum())
    print(f"duplicates: injected {injected_dupes}, detected {detected_dupes}")
    print(f"missing cells: present {injected_missing_cells}, detected {detected_missing_cells}")
    assert detected_missing_cells == injected_missing_cells
    assert abs(detected_dupes - injected_dupes) <= 0.1 * injected_dupes

    print("\nALL TASKCLEAN INVARIANTS HOLD")


def test_generic_csv():
    """A non-Adult CSV with awkward columns: the pipeline must not crash, must
    keep the log honest, and must reject non-classification targets."""
    rng = np.random.default_rng(0)
    n = 1500
    df = pd.DataFrame({
        "customer_id": np.arange(n),                                   # unique ID
        "full_name": [f"person_{i}" for i in range(n)],                 # high-cardinality text
        "age": rng.integers(18, 80, n).astype(float),
        "spend": rng.gamma(2.0, 50.0, n),
        "is_member": rng.random(n) < 0.4,                               # bool
        "city": rng.choice(["NYC", "nyc", "Boston", "Chicago"], n),     # inconsistent casing
        "churned": rng.integers(0, 2, n).astype(float),                 # numeric 0/1 target
    })
    df.loc[rng.choice(n, 90, replace=False), "age"] = np.nan
    df.loc[rng.choice(n, 40, replace=False), "churned"] = np.nan        # unlabeled rows
    df.loc[rng.choice(n, 10, replace=False), "spend"] *= 80             # outliers
    df = pd.concat([df, df.iloc[:60]], ignore_index=True)               # 60 exact duplicates

    state = audit_dataset(df, "churned")
    assert "customer_id" not in state.unmodeled_columns, "a numeric ID is not a high-cardinality categorical"
    assert "full_name" in state.unmodeled_columns, "high-cardinality text must skip the model-based detectors"
    assert state.meta["n_rows_missing_target"] > 0
    assert state.detections["inconsistency"].rate > 0, "NYC/nyc casing must be flagged"

    for issues in (None, {"duplicates", "missing_values", "outliers", "label_errors"}):
        res = apply_repairs(state, issues)
        assert list(res.cleaned.columns) == list(df.columns)
        unlabeled = df.index[df["churned"].isna() & df.index.isin(res.cleaned.index)]
        assert res.cleaned.loc[unlabeled, "churned"].isna().all(), "unlabeled rows must be left untouched"
        logged = res.repair_log[(res.repair_log.action == "applied") & (res.repair_log.problem != "Duplicate row")]
        assert set(logged.row_id) <= set(res.cleaned.index)
        assert changed_cells(df, res.cleaned) <= {(int(r), c) for r, c in zip(logged.row_id, logged.column)}
        json.loads(outputs_as_bytes(res)["readiness_report.json"])        # strict JSON (no NaN)
    print("OK generic CSV: ID/text/bool/numeric-target/unlabeled rows handled, log honest, JSON valid")

    for bad_target, msg in (("spend", "classification only"), ("customer_id", "classification only")):
        try:
            audit_dataset(df, bad_target)
        except ValueError as e:
            assert msg in str(e)
        else:
            raise AssertionError(f"{bad_target} should have been rejected as a non-classification target")
    print("OK continuous / ID targets rejected with a clear message")


if __name__ == "__main__":
    main()
    test_generic_csv()
