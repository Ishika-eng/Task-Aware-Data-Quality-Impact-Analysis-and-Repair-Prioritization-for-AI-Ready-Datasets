"""
Invariants for Phase 23 Amendment A1 (docs/PHASE23_SPEC.md): evidence-based repairs are only applied to columns the calibration
covered. Run:  .venv/bin/python src/test_phase23.py        Every check can fail.
"""
import os
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from taskclean import _apply, audit_dataset


def frame(n=400, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n), "grade": rng.choice(["x", "y", "z"], n),
                       "pool": rng.choice(["Gd", "Ex"], n), "y": rng.choice(["p", "q"], n)})
    df.loc[rng.choice(n, 20, replace=False), "a"] = np.nan                    # an ordinary, covered blank
    df.loc[rng.random(n) < 0.7, "pool"] = np.nan                              # 70% empty: "no pool", out of the calibration's scope
    return df


def fake_selfbench(left_out):
    return SimpleNamespace(ok=True, reference={"columns_left_out": {c: "70% missing" for c in left_out}})


def applied_by_column(log):
    ap = log[log.action == "applied"]
    return ap.groupby("column").size().to_dict()


def test_out_of_scope_columns_stay_proposals():
    st = audit_dataset(frame(), "y", detectors={"missing_values"})
    st.selfbench = fake_selfbench(["pool"])
    _, log_scoped = _apply(st, {"missing_values"}, [], respect_scope=True)
    _, log_plain = _apply(st, {"missing_values"}, [])
    assert applied_by_column(log_plain).get("pool", 0) > 100, "without the rule the 70%-empty column is imputed"
    got = applied_by_column(log_scoped)
    assert "pool" not in got and got.get("a", 0) == 20, f"only the covered column may be applied: {got}"
    held = log_scoped[(log_scoped.column == "pool") & (log_scoped.problem == log_plain.problem.iloc[0])]
    assert len(held) > 100 and set(held.action) == {"flagged_for_review"}, "held-back cells stay in the review queue"
    print("OK A1: the calibration-uncovered column stays a proposal, the covered column is still repaired")


def test_human_override_and_other_callers_are_unchanged():
    st = audit_dataset(frame(), "y", detectors={"missing_values"})
    st.selfbench = fake_selfbench(["pool"])
    _, log_override = _apply(st, {"missing_values"}, ["missing_values"], respect_scope=True)
    assert applied_by_column(log_override).get("pool", 0) > 100, "a human override of the issue type applies everything"
    c0, l0 = _apply(st, {"missing_values"}, [])
    st.selfbench = None
    c1, l1 = _apply(st, {"missing_values"}, [])
    assert c0.equals(c1) and l0.equals(l1), "respect_scope off (the default for every other caller) ignores the calibration"
    c2, l2 = _apply(st, {"missing_values"}, [], respect_scope=True)
    assert c2.equals(c1), "with no calibration there is nothing to scope: the output is unchanged"
    print("OK A1: human override applies everything; callers without the flag and uncalibrated runs are unchanged")


def test_the_note_says_what_was_held_back_and_is_not_duplicated():
    st = audit_dataset(frame(), "y", detectors={"missing_values"})
    st.selfbench = fake_selfbench(["pool"])
    for _ in range(3):                                           # applying repeatedly (the apply endpoint) must not stack notes
        _apply(st, {"missing_values"}, [], respect_scope=True)
    notes = [n for n in st.meta["notes"] if n.startswith("Held back for review:")]
    assert len(notes) == 1 and "pool" in notes[0] and "did not cover" in notes[0], notes
    st.selfbench = None
    _apply(st, {"missing_values"}, [], respect_scope=True)
    assert not [n for n in st.meta["notes"] if n.startswith("Held back for review:")], "no held-back cells, no note"
    print("OK A1: one explanatory note, replaced rather than repeated, removed when nothing is held back")


if __name__ == "__main__":
    tests = [v for n, v in sorted(globals().items()) if n.startswith("test_") and callable(v)]
    for t in tests:
        t()
    print(f"\nALL PHASE 23 AMENDMENT A1 INVARIANTS HOLD ({len(tests)} tests)")
