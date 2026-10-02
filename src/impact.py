"""
Task-aware impact analysis and effort-aware repair prioritization.

This is the heart of the project's research idea:

    Impact_e   = P_clean - P_dirty_e              (how much did error e hurt?)
    Priority_e = Gain_e / (Effort_e + eps)         (how much F1 do I regain per
                                                     unit of cleaning effort?)

`Gain_e` here is just `Impact_e`: the F1 we'd get back by fully fixing error
e. Effort is NOT something we can measure from the data itself -- it's a
property of the real-world repair process (does fixing it need a script, or
a human re-labeling thousands of rows?). So we encode it as explicit,
editable assumptions rather than hiding it inside a formula. You should feel
free to argue with these numbers; that's the point of making them visible.
"""

import pandas as pd

EPSILON = 1e-6

# Relative cleaning effort per error type, on an arbitrary 1-10 scale.
# Rationale for each number is the comment beside it -- these are the
# project's modeling assumptions, not measured quantities.
DEFAULT_EFFORT = {
    "duplicates": 1,            # one-line dedupe, no domain knowledge needed
    "outliers": 3,               # needs a statistical rule (e.g. clip/winsorize) + some tuning
    "missing_values": 3,         # imputation strategy choice, but mostly automatable
    "label_noise": 8,            # needs human re-annotation / domain expert review
    "feature_corruption": 9,     # needs rebuilding the data pipeline / tracing the bug upstream
}


def compute_impact(clean_f1: float, dirty_f1_by_error: dict[str, float]) -> pd.DataFrame:
    """Build a table of Impact_e for each error type.

    Parameters
    ----------
    clean_f1 : F1 of the model trained on the untouched dataset.
    dirty_f1_by_error : {error_name: F1 of the model trained on data with
        only that error injected}.
    """
    rows = []
    for error, dirty_f1 in dirty_f1_by_error.items():
        rows.append({
            "error_type": error,
            "clean_f1": clean_f1,
            "dirty_f1": dirty_f1,
            "impact": clean_f1 - dirty_f1,
        })
    return pd.DataFrame(rows).sort_values("impact", ascending=False).reset_index(drop=True)


def compute_priority(impact_df: pd.DataFrame, effort: dict[str, float] | None = None) -> pd.DataFrame:
    """Add Effort_e and Priority_e columns, ranked by priority descending.

    Priority answers "which repair gives the most F1 back per unit of
    effort spent" -- NOT "which error is worst". A hugely damaging error
    that's extremely expensive to fix can rank below a moderately damaging
    error that's nearly free to fix.
    """
    effort = effort or DEFAULT_EFFORT
    df = impact_df.copy()
    df["effort"] = df["error_type"].map(effort)
    df["priority"] = df["impact"] / (df["effort"] + EPSILON)
    return df.sort_values("priority", ascending=False).reset_index(drop=True)
