"""
End-to-end invariant tests for the TaskClean product layer.

The repair log is the product's proof of what it did, so the central claim
under test is:

    every cell that differs between the uploaded file and a cleaned file is a
    logged, applied repair -- and nothing else changed.

It is checked for BOTH output variants (evidence-based and aggressive), on the
demo dataset and on awkward generic CSVs.

Run:  python3 test_taskclean.py        (uses data/demo_dirty_adult.csv)
"""

import json
import os

import numpy as np
import pandas as pd

from taskclean import (apply_repairs, audit_dataset, default_apply_issues, load_csv, outputs_as_bytes)

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


def values_match(a, b) -> bool:
    """Equal as numbers (tolerating a decimal comma) or as text."""
    try:
        return bool(np.isclose(float(str(a).replace(",", ".")), float(str(b).replace(",", "."))))
    except (TypeError, ValueError):
        return str(a) == str(b)


def assert_log_honest(df, cleaned, log, label):
    """The invariant, for one (cleaned file, log) pair."""
    applied = log[log.action == "applied"]
    dups = set(applied.loc[applied.problem == "Duplicate row", "row_id"])
    assert set(df.index) - set(cleaned.index) == dups, f"{label}: dropped rows must be exactly the logged duplicates"
    cells = applied[applied.problem != "Duplicate row"]
    assert set(cells.row_id) <= set(cleaned.index), f"{label}: log claims a repair on a row absent from the file"
    logged = {(int(r), c) for r, c in zip(cells.row_id, cells.column)}
    unlogged = changed_cells(df, cleaned) - logged
    assert not unlogged, f"{label}: {len(unlogged)} changed cells are missing from the log"
    for r, c, v, proposed in zip(cells.row_id, cells.column, cells.applied_value, cells.proposed_value):
        assert str(cleaned.at[r, c]) == str(v), f"{label}: log/cleaned mismatch at row {r}, column {c}"
        assert values_match(v, proposed), f"{label}: applied value differs from the proposal at row {r}, column {c}"


def test_demo():
    df = pd.read_csv(DEMO)
    state = audit_dataset(df, "class", include_feature_anomalies=False,
                          progress=lambda f, m: print(f"  [{f:4.0%}] {m}"))
    print("\nPolicy:\n", state.policy[["issue", "detected_rate", "n_repair_candidates", "n_flag_only",
                                       "default_action"]].round(4).to_string(index=False))

    approved = default_apply_issues(state)
    assert approved == {"duplicates"}, "only exact duplicate removal should clear the safe-auto bar"
    result = apply_repairs(state)
    assert list(result.cleaned.columns) == list(df.columns)
    assert changed_cells(df, result.cleaned) == set(), "default policy must not modify any surviving cell"
    assert not result.cleaned.duplicated().any(), "no full-record duplicates may remain"
    assert_log_honest(df, result.cleaned, result.repair_log, "evidence-based")
    print(f"OK evidence-based file: {len(df) - len(result.cleaned)} duplicates dropped, no other cell changed")

    # The aggressive variant is a real, different, model-ready file ...
    assert result.cleaned_aggressive.drop(columns=[]).isna().values.sum() < df.isna().values.sum(), \
        "the aggressive file must have imputed the missing values"
    assert_log_honest(df, result.cleaned_aggressive, result.repair_log_aggressive, "aggressive")
    assert "feature_corruption" not in result.readiness["aggressive_variant"]["issues_applied"]
    print(f"OK aggressive file: {len(changed_cells(df, result.cleaned_aggressive))} cells changed, all logged")

    # A human override on the primary file is also fully accounted for.
    result2 = apply_repairs(state, {"duplicates", "missing_values", "outliers", "label_errors"})
    assert_log_honest(df, result2.cleaned, result2.repair_log, "override")
    flagged = result2.repair_log[result2.repair_log.action == "applied"]
    assert flagged.loc[flagged.problem != "Duplicate row", "human_override"].all()
    print("OK override run: log honest, overrides marked")

    truth = pd.read_csv(DEMO.replace(".csv", "_corruption_log.csv"))
    injected_dupes = (truth.error_type == "duplicates").sum()
    detected_dupes = int(state.detections["duplicates"].row_mask.sum())
    assert detected_dupes == injected_dupes == 300
    assert int(state.detections["missing_values"].extra["cell_mask"].values.sum()) == \
        int(df.drop(columns="class").isna().values.sum())
    print("OK ground truth: 300/300 duplicates and every missing cell detected")


def awkward_frame(n=1500, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "customer_id": np.arange(n),                                   # unique ID
        "full_name": [f"person_{i}" for i in range(n)],                 # high-cardinality text
        "zip": [f"{z:05d}" for z in rng.integers(1000, 9999, n)],       # leading zeros: must stay text
        "age": rng.integers(18, 80, n).astype(float),
        "spend": rng.gamma(2.0, 50.0, n),
        "score": [f"{v:.1f}" for v in rng.uniform(0, 100, n)],          # numbers stored as text
        "is_member": rng.random(n) < 0.4,                               # bool
        "city": rng.choice(["NYC", "nyc", "Boston", "Chicago"], n),     # inconsistent casing
        "region": rng.choice(["north", "south", "east"], n),
        "price": [f"{v:.2f}".replace(".", ",") for v in rng.uniform(1, 500, n)],   # decimal-comma text
        "churned": rng.integers(0, 2, n).astype(float),                 # numeric 0/1 target
    })
    df.loc[rng.choice(n, 90, replace=False), "age"] = np.nan
    df.loc[rng.choice(n, 40, replace=False), "churned"] = np.nan        # unlabeled rows
    df.loc[rng.choice(n, 10, replace=False), "spend"] *= 80             # outliers
    df.loc[rng.choice(n, 50, replace=False), "score"] = "?"             # placeholder in a numeric-looking column
    df.loc[rng.choice(n, 60, replace=False), "region"] = "N/A"          # placeholder in a categorical column
    df.loc[rng.choice(n, 45, replace=False), "price"] = "?"             # placeholder in a decimal-comma column
    return pd.concat([df, df.iloc[:60]], ignore_index=True)             # 60 exact duplicates


def test_generic_csv():
    df = awkward_frame()
    state = audit_dataset(df, "churned")
    assert "customer_id" not in state.unmodeled_columns, "a numeric ID is not a high-cardinality categorical"
    assert "full_name" in state.unmodeled_columns, "high-cardinality text must skip the model-based detectors"
    assert state.meta["n_rows_missing_target"] > 0
    assert state.detections["inconsistency"].rate > 0, "NYC/nyc casing must be flagged"

    # placeholders count as missing; numeric-looking text is analysed as numbers; ZIP codes stay text
    assert pd.api.types.is_numeric_dtype(state.X["score"]), "'score' (numbers as text) must be analysed as numeric"
    assert pd.api.types.is_numeric_dtype(state.X["price"]) and "price" in state.decimal_comma_columns, \
        "decimal-comma text must be analysed as numeric"
    assert not pd.api.types.is_numeric_dtype(state.X["zip"]), "leading-zero ZIP codes must stay categorical"
    cell_mask = state.detections["missing_values"].extra["cell_mask"]
    assert cell_mask["score"].sum() == (df["score"] == "?").sum(), "every '?' must count as missing (all rows)"
    assert cell_mask["region"].sum() == (df["region"] == "N/A").sum(), "every 'N/A' must count as missing (all rows)"
    prop = state.proposals
    placeholder_rows = prop[(prop.issue_key == "missing_values") & (prop.column == "region")]
    assert "N/A" in set(placeholder_rows.original_value), "the log must show the cell's real original content"

    res = apply_repairs(state)
    assert "?" in set(res.cleaned["score"]), "the evidence-based file must leave placeholders untouched"
    assert_log_honest(df, res.cleaned, res.repair_log, "generic/evidence-based")
    assert not set(res.cleaned_aggressive["score"].dropna()) & {"?"}, "the aggressive file must impute placeholders"
    assert not set(res.cleaned_aggressive["region"].dropna()) & {"N/A"}
    was_placeholder = (df["price"] == "?").reindex(res.cleaned_aggressive.index)
    imputed_price = res.cleaned_aggressive.loc[was_placeholder, "price"].dropna()
    assert len(imputed_price) and imputed_price.str.contains(",").all(), \
        "imputed values in a decimal-comma column must be written with a comma"
    assert_log_honest(df, res.cleaned_aggressive, res.repair_log_aggressive, "generic/aggressive")
    unlabeled = df.index[df["churned"].isna() & df.index.isin(res.cleaned_aggressive.index)]
    assert res.cleaned_aggressive.loc[unlabeled, "churned"].isna().all(), "unlabeled rows must be left untouched"
    for name, data in outputs_as_bytes(res).items():
        if name.endswith(".json"):
            json.loads(data)                                              # strict JSON (no NaN)
    print("OK generic CSV: placeholders, numeric-looking text, ZIPs, IDs, bool, unlabeled rows; both files honest")

    for bad_target in ("spend", "customer_id"):
        try:
            audit_dataset(df, bad_target)
        except ValueError as e:
            assert "classification only" in str(e)
        else:
            raise AssertionError(f"{bad_target} should have been rejected as a non-classification target")
    print("OK continuous / ID targets rejected with a clear message")


def test_load_csv():
    """European-style file: ';' delimiter, decimal comma, cp1252 accents."""
    text = "name;price;qty\nCafé;3,5;2\nNaïve;4,25;7\nPlain;1,0;3\n"
    df, info = load_csv(text.encode("cp1252"))
    assert info["delimiter"] == ";" and info["decimal"] == "," and info["encoding"] == "cp1252", info
    assert list(df.columns) == ["name", "price", "qty"] and np.isclose(df["price"].iloc[1], 4.25)
    assert df["name"].iloc[0] == "Café"
    df2, info2 = load_csv(b"a\tb\n1\t2\n3\t4\n")
    assert info2["delimiter"] == "tab" and df2.shape == (2, 2)
    try:
        load_csv(b"   ")
    except ValueError:
        pass
    else:
        raise AssertionError("an empty file must be rejected")
    print("OK loader: delimiter, decimal comma, encoding and empty-file handling")


def test_no_target_and_sampling():
    df = awkward_frame()
    state = audit_dataset(df, None)
    assert state.target is None and "label_noise" not in state.detections
    assert state.policy.set_index("issue_key").loc["label_errors", "default_action"] == "not assessed"
    res = apply_repairs(state)
    assert_log_honest(df, res.cleaned, res.repair_log, "no-target/evidence-based")
    assert_log_honest(df, res.cleaned_aggressive, res.repair_log_aggressive, "no-target/aggressive")
    json.loads(outputs_as_bytes(res)["readiness_report.json"])
    print("OK no-target mode: label checks skipped, both files honest")

    state = audit_dataset(df, "churned", max_model_rows=400)
    assert state.meta["model_detectors_sampled"] and state.meta["model_detector_rows"] < 600
    assert state.detections["duplicates"].row_mask.shape[0] == state.meta["n_rows_analysed"], \
        "cheap checks must still use every row"
    res = apply_repairs(state, {"duplicates", "missing_values", "outliers", "label_errors"})
    assert_log_honest(df, res.cleaned, res.repair_log, "sampled")
    assert any("sample" in n for n in state.meta["notes"])
    print(f"OK sampling: slow detectors ran on {state.meta['model_detector_rows']} of "
          f"{state.meta['n_rows_analysed']} rows, log honest")


if __name__ == "__main__":
    test_demo()
    test_generic_csv()
    test_load_csv()
    test_no_target_and_sampling()
    print("\nALL TASKCLEAN INVARIANTS HOLD")
