"""
Invariants for the Phase 14 detectors (inconsistency repair proposals and irrelevant-feature detection).
Run:  .venv/bin/python src/test_phase14.py

Every check asserts something that can fail: planted ground truth, and the cases the guards exist for.
"""

import numpy as np
import pandas as pd

from inconsistency import detect_inconsistency_v2
from phase14_injectors import IRRELEVANT_KINDS, KINDS, add_irrelevant_columns, inject_inconsistency
from relevance import assess_features


def cells_of(values):
    X = pd.DataFrame({"c": pd.Series(values, dtype=object)})
    r = detect_inconsistency_v2(X)
    c = r.extra["cells"]
    return r, X, {(row.original_value, row.proposed_value, row.rule) for row in c.itertuples()} if len(c) else set()


def test_inconsistency_found():
    # the case the old detector missed: abbreviations
    r, X, got = cells_of(["Male"] * 40 + ["male "] * 3 + ["M"] * 4 + ["Female"] * 40 + ["female"] * 3 + ["F"] * 4)
    assert got == {("male ", "Male", "format_variant"), ("M", "Male", "abbreviation"),
                   ("female", "Female", "format_variant"), ("F", "Female", "abbreviation")}, got
    assert r.extra["n_strict"] == 14 and r.extra["n_loose"] == 0
    assert abs(r.rate - 14 / 94) < 1e-12, "rate must be the share of rows holding a non-canonical cell"
    assert r.row_mask.sum() == 14 and r.row_mask.index.equals(X.index)

    # punctuation, underscores, accents
    _, _, got = cells_of(["Café"] * 50 + ["Cafe"] * 3 + ["café."] * 2 + ["New York"] * 40 + ["new_york"] * 3 + ["NY"] * 2)
    assert got == {("Cafe", "Café", "format_variant"), ("café.", "Café", "format_variant"),
                   ("new_york", "New York", "format_variant"), ("NY", "New York", "abbreviation")}, got

    # typos are flagged but NEVER given a proposed value
    r, _, got = cells_of(["Private"] * 100 + ["Prvate"] * 2 + ["Federal-gov"] * 60)
    assert got == {("Prvate", None, "possible_typo")}, got
    assert (r.extra["cells"].tier == "loose").all() and r.extra["cells"].proposed_value.isna().all()
    print("OK inconsistency: case/space/punctuation/accents/abbreviations repaired, typos flag-only")


def test_inconsistency_guards():
    for name, vals in {
        "two real categories that share a prefix": ["No"] * 50 + ["None"] * 50,
        "digit-only difference is a different value": ["Grade 3"] * 50 + ["Grade 4"] * 3 + ["Grade 5"] * 50,
        "long names are never abbreviations": ["Other"] * 40 + ["Other-service"] * 60,
        "clean column": ["A"] * 30 + ["B"] * 30 + ["C"] * 30,
        "single value": ["X"] * 50,
        # a symbol that carries meaning is not decoration (found by the real-data check: it rewrote '>50K' as '<=50K')
        "comparison operators": [">50K"] * 70 + ["<=50K"] * 30,
        "grades with signs": ["A"] * 40 + ["A-"] * 30 + ["A+"] * 30,
        "blood types": ["A+"] * 20 + ["A-"] * 20 + ["B+"] * 20 + ["B-"] * 20 + ["O+"] * 20 + ["O-"] * 20 + ["AB+"] * 10 + ["AB-"] * 10,
    }.items():
        r, _, got = cells_of(vals)
        assert got == set() and r.rate == 0.0, f"{name}: false positive {got}"
    # an abbreviation that could stand for two values must not be repaired
    _, _, got = cells_of(["Self-emp-inc"] * 30 + ["Self-emp-not-inc"] * 30 + ["Sel"] * 2)
    assert got == {("Sel", None, "ambiguous_abbreviation")}, got
    print("OK inconsistency guards: no false positives on the traps they exist for")


def test_abbreviation_rule_does_not_run_backwards():
    # Amendment A1: a FREQUENT short key next to a rarer long key is two real categories (found on Telco churn, where
    # "No phone service" was rewritten to "No"). Several mixes, including shares well under the 0.30 limit.
    for n_short, n_long in [(900, 100), (700, 300), (3390, 682), (95, 5)]:
        _, _, got = cells_of(["No"] * n_short + ["No phone service"] * n_long + ["Yes"] * 800)
        assert got == set(), f"majority short key must not be merged, {n_short}/{n_long}: {got}"
    # the same words with the roles reversed: the rare short form IS the abbreviation and is still repaired
    r, _, got = cells_of(["Female"] * 900 + ["Fem"] * 100 + ["Male"] * 800)
    assert got == {("Fem", "Female", "abbreviation")}, got
    assert r.extra["cells"].shape[0] == 100
    # a short key that is the rarer one of an equal-sized pair stays unflagged (share 0.5 > limit)
    _, _, got = cells_of(["No"] * 50 + ["No phone service"] * 50)
    assert got == set(), got
    # a tie at the limit goes to the short key as the minority: 30 / (30 + 70) = 0.30
    _, _, got = cells_of(["Female"] * 70 + ["Fem"] * 30)
    assert got == {("Fem", "Female", "abbreviation")}, got
    print("OK abbreviation direction: frequent short key untouched, rare short key repaired")


def test_vocabulary_guard_counts_keys_not_spellings():
    # a genuine identifier column (every value distinct) must still be skipped, and must produce nothing
    ids = pd.DataFrame({"occ": pd.Series([f"user-{i:06d}" for i in range(3000)], dtype=object)})
    r2 = detect_inconsistency_v2(ids)
    assert "occ" in r2.extra["skipped_columns"] and r2.rate == 0.0, "a real identifier column must still be skipped"
    # many case/space variants of a few values: distinct RAW spellings far above 500, distinct keys tiny
    base = ["Alpha", "Beta", "Gamma"]
    many = [f"{b.upper() if i % 2 else b.lower()}{' ' * (i % 700)}" for i, b in enumerate(base * 1000)]
    X3 = pd.DataFrame({"c": pd.Series(many + ["Alpha"] * 500 + ["Beta"] * 500 + ["Gamma"] * 500, dtype=object)})
    assert X3["c"].nunique() > 500, "precondition: more than 500 distinct raw spellings"
    r3 = detect_inconsistency_v2(X3)
    assert r3.extra["skipped_columns"] == {}, r3.extra["skipped_columns"]
    assert r3.extra["n_strict"] > 0 and (r3.extra["cells"].proposed_value.isin(base)).all()
    print(f"OK vocabulary guard: {X3['c'].nunique()} raw spellings of 3 values are checked; real identifiers still skipped")


def test_inconsistency_invariants():
    rng = np.random.default_rng(1)
    base = rng.choice(["Private", "Federal-gov", "Local-gov", "State-gov"], 2000)
    vals = base.astype(object).copy()
    vals[:60] = [v.upper() for v in vals[:60]]
    vals[60:90] = [" " + v for v in vals[60:90]]
    X = pd.DataFrame({"w": vals, "n": rng.normal(size=2000)})
    X.loc[5, "w"] = np.nan
    r = detect_inconsistency_v2(X)
    c = r.extra["cells"]
    allowed = set(X["w"].dropna())
    prop = c.proposed_value.dropna()
    assert prop.isin(allowed).all(), "a proposed value must already exist in the column"
    assert not c.row_id.isin([5]).any(), "a missing cell is missing-value territory, not an inconsistency"
    assert (c.column == "w").all(), "numeric columns are never touched"
    # 90 cells were planted (rows 0-89) but row 5 was then blanked, so exactly 89 may be reported
    assert len(c) == 89 and (c.tier == "strict").all(), len(c)
    # deterministic
    c2 = detect_inconsistency_v2(X).extra["cells"]
    assert c.reset_index(drop=True).equals(c2.reset_index(drop=True))
    print("OK inconsistency invariants: proposals come from the column, nulls ignored, deterministic")


def planted_frame(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    group = rng.choice(["a", "b", "c", "d"], n)
    logit = 1.6 * x1 - 1.2 * x2 + np.where(group == "a", 1.0, -0.3)
    y = pd.Series(np.where(rng.random(n) < 1 / (1 + np.exp(-logit)), "pos", "neg"))
    X = pd.DataFrame({
        "signal_1": x1, "signal_2": x2, "signal_cat": group,
        "noise_gauss": rng.normal(size=n),
        "noise_cat": rng.choice(["u", "v", "w"], n),
        "const": ["same"] * n,
        "row_id": np.arange(n).astype(float),
        "copy_of_signal_1": x1.copy(),
        "scaled_signal_2": x2 * 0.5 + 3.0,
        "code_of_signal_cat": pd.Series(group).map({"a": "A1", "b": "B2", "c": "C3", "d": "D4"}),
    })
    return X, y


def test_relevance_planted():
    X, y = planted_frame()
    ii = {"row_id": True}
    # structural checks are deterministic: they do not depend on the probe test
    out = assess_features(X, y, seed=0, integer_like=ii).set_index("column")
    v = out.verdict
    assert v["const"] == "constant" and v["row_id"] == "id_like"
    assert v["copy_of_signal_1"] == "redundant" and out.redundant_with["copy_of_signal_1"] == "signal_1"
    assert v["scaled_signal_2"] == "redundant" and out.redundant_with["scaled_signal_2"] == "signal_2"
    assert v["code_of_signal_cat"] == "redundant" and out.redundant_with["code_of_signal_cat"] == "signal_cat", \
        "a one-to-one recoding of another column is redundant"
    assert v["signal_1"] != "redundant" and v["signal_2"] != "redundant", "the EARLIER column of a pair is the one kept"

    # the probe test is statistical, so assert its measured behaviour over several seeds, not one draw:
    # an informative column must never be flagged; a pure-noise column is expected to be caught ~85-90% of the time
    sig, noise = ("signal_1", "signal_2", "signal_cat"), ("noise_gauss", "noise_cat")
    wrong_flags, caught, trials = 0, 0, 0
    for seed in range(8):
        vv = assess_features(X, y, seed=seed, integer_like=ii).set_index("column").verdict
        wrong_flags += sum(vv[c] != "relevant" for c in sig)
        caught += sum(vv[c] == "no_detectable_relevance" for c in noise)
        trials += len(noise)
    assert wrong_flags == 0, f"informative columns flagged {wrong_flags} time(s) in 8 seeds"
    assert caught / trials >= 0.75, f"noise columns caught only {caught}/{trials} times"
    print(f"OK relevance: structural flags exact; informative columns never flagged; noise caught {caught}/{trials}")


def test_relevance_no_target_and_small():
    X, _ = planted_frame()
    out = assess_features(X, None, seed=0, integer_like={"row_id": True}).set_index("column")
    assert out.verdict["const"] == "constant" and out.verdict["copy_of_signal_1"] == "redundant", \
        "structural checks need no target"
    assert out.verdict["noise_gauss"] == "not_assessed" and "no target" in out.reason["noise_gauss"]
    Xs, ys = planted_frame(n=120)
    small = assess_features(Xs, ys, seed=0)
    assert (small[small.verdict.isin(["relevant", "no_detectable_relevance"])].shape[0] == 0), \
        "too few rows: the probe test must decline, not guess"
    print("OK relevance: structural flags work without a target; probe test declines on tiny data")


def test_injectors_are_honest():
    rng = np.random.default_rng(3)
    n = 4000
    X = pd.DataFrame({
        "workclass": rng.choice(["Private", "Federal-gov", "Local-gov", "State-gov", "Self-emp-inc"], n),
        "education": rng.choice(["Bachelors", "HS-grad", "Masters", "Doctorate", "Some-college"], n),
        "sex": rng.choice(["Male", "Female"], n),
        "age": rng.integers(18, 80, n).astype(float),
    }, index=rng.permutation(n) + 500)                        # scattered index, like a train split
    y = pd.Series(rng.choice(["a", "b"], n), index=X.index)
    X_before = X.copy()
    Xd, yd, log = inject_inconsistency(X, y, rate=0.10, seed=7)
    assert X.equals(X_before), "the input must never be mutated"
    assert yd.equals(y) and Xd["age"].equals(X["age"]), "labels and numeric columns are untouched"
    # the log is exactly the set of changed cells
    changed = {(r, c) for c in ["workclass", "education", "sex"] for r in X.index if Xd.at[r, c] != X.at[r, c]}
    logged = {(r, c) for r, c in zip(log.row_id, log.column)}
    assert changed == logged, f"log disagrees with the data: {len(changed ^ logged)} cell(s)"
    assert all(Xd.at[r, c] == cv and X.at[r, c] == ov for r, c, ov, cv in
               zip(log.row_id, log.column, log.original_value, log.corrupted_value))
    assert (log.original_value != log.corrupted_value).all()
    assert abs(len(log) / (n * 3) - 0.10) < 0.005, "rate is a fraction of categorical cells"
    counts = log.kind.value_counts()
    assert set(counts.index) <= set(KINDS) and counts.min() > 0.5 * counts.max(), counts.to_dict()
    # the same call twice is identical
    assert inject_inconsistency(X, y, rate=0.10, seed=7)[0].equals(Xd)

    aug, truth = add_irrelevant_columns(X, seed=1)
    assert aug[list(X.columns)].equals(X) and set(truth.values()) == set(IRRELEVANT_KINDS)
    assert set(truth) == set(aug.columns) - set(X.columns)
    assert aug["x_constant"].nunique() == 1 and aug["x_row_id"].is_monotonic_increasing
    assert np.allclose(aug["x_linear_of_age"], X["age"] * 0.5 + 3.0)
    print(f"OK injectors: log == changed cells ({len(log)}), rate exact, kinds balanced {counts.to_dict()}")


if __name__ == "__main__":
    test_injectors_are_honest()
    test_inconsistency_found()
    test_inconsistency_guards()
    test_abbreviation_rule_does_not_run_backwards()
    test_vocabulary_guard_counts_keys_not_spellings()
    test_inconsistency_invariants()
    test_relevance_planted()
    test_relevance_no_target_and_small()
    print("\nALL PHASE 14 DETECTOR INVARIANTS HOLD")
