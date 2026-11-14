"""
Diagnostic for one Phase 14 finding: the inconsistency repair is exactly correct (precision 1.000, zero false repairs)
yet the measured F1 recovery is negative. This restores different SUBSETS of the injected cells exactly (using the
injection log, i.e. an oracle) and measures recovery for each, to see which subset drives the sign. It changes no
detector, policy or result file used by the product; it only reports.

    .venv/bin/python src/phase14_diagnostic.py
"""

import os

import numpy as np
import pandas as pd

from phase14_injectors import inject_inconsistency
from phase14_validation import RATES, RESULTS_DIR, fit_scores, load_split, log

SEEDS = [42, 43, 44, 45, 46]
EASY = {"case", "whitespace", "punctuation"}          # what the product repairs with certainty
HARD = {"typo", "abbreviation"}                       # flag-only, or repaired only when the match is unique


def restore(X_d, X_clean, inj, mask):
    out = X_d.copy()
    for (row_id, col) in inj.loc[mask, ["row_id", "column"]].itertuples(index=False):
        out.at[row_id, col] = X_clean.at[row_id, col]
    return out


if __name__ == "__main__":
    X_tr, X_te, y_tr, y_te = load_split()
    repair_recall = pd.read_csv(os.path.join(RESULTS_DIR, "phase14_inconsistency_runs.csv")).groupby("rate").repair_recall.mean()
    rows = []
    for rate in RATES:
        for seed in SEEDS:
            X_d, y_d, inj = inject_inconsistency(X_tr, y_tr, rate=rate, seed=seed)
            rng = np.random.default_rng([seed, 99])
            variants = {
                "random_same_fraction_as_product": pd.Series(rng.random(len(inj)) < repair_recall[rate], index=inj.index),
                "only_easy_kinds": inj.kind.isin(EASY),
                "only_hard_kinds": inj.kind.isin(HARD),
                "everything_oracle": pd.Series(True, index=inj.index),
            }
            dirty = fit_scores(X_d, y_d, X_te, y_te, seed)
            for name, mask in variants.items():
                s = fit_scores(restore(X_d, X_tr, inj, mask), y_d, X_te, y_te, seed)
                rows.append({"rate": rate, "seed": seed, "variant": name, "fraction_restored": float(mask.mean()),
                             "recovery_f1": s["f1"] - dirty["f1"], "recovery_auc": s["roc_auc"] - dirty["roc_auc"]})
            r = [x for x in rows if x["rate"] == rate and x["seed"] == seed]
            log(f"rate={rate:.0%} seed={seed}: " + "  ".join(f"{x['variant'].split('_')[0]}={x['recovery_f1']:+.4f}" for x in r))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, "phase14_inconsistency_diagnostic.csv"), index=False)
    print(df.groupby(["variant", "rate"]).agg(fraction=("fraction_restored", "mean"), f1_mean=("recovery_f1", "mean"),
                                              f1_std=("recovery_f1", "std"), f1_neg_seeds=("recovery_f1", lambda x: int((x < 0).sum())),
                                              auc_mean=("recovery_auc", "mean")).round(4).to_string())
