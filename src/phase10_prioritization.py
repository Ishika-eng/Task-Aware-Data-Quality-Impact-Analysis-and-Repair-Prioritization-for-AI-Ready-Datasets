"""
Repair Prioritization (Phase 10).

Combines three independently-measured dimensions into a priority score,
WITHOUT collapsing the underlying evidence -- the full per-error-type
table (mean_damage, repair precision/recall, recovery, status) is kept
alongside the final score, not discarded once a number exists.

    Impact          (Phase 7, oracle corruption experiment)
        How much does this error hurt F1?
    Repairability   (Phase 9, real non-oracle repair)
        Can we recover the damage? Four-state evidence model:

        IMPORTANT: repairability_status is Status(error_type, rate), NOT
        Status(error_type) -- it is determined independently for each
        corruption rate, not once per error type. An error type's status
        can legitimately differ across 5%/10%/20% (e.g. label_errors is
        "weak" at 5% and 20% but "positive" at 10% in this project's
        actual results) -- that instability IS a finding worth reporting,
        not noise to average away into one global label per error type.
          positive / negative : confident sign (>=80% of 5 seeds agree)
                                 AND magnitude above the noise floor
          weak                 : inconsistent sign OR magnitude at/below
                                 the noise floor -- we have data, it's just
                                 not a confident signal in either direction
          not_estimable        : no repair data exists at all (doesn't
                                 occur for our 15 error/rate combos, kept
                                 in the schema for generality)
    Risk            (Phase 9, same raw per-seed recovery data)
        Could attempting repair make things worse?
        Risk = P(harm) x E[harm magnitude | harm occurred]   (always >= 0)

Priority(e,r) = w_impact*Impact_norm - w_risk*Risk_norm + w_repair*Repairability_norm
(repairability estimable)
Priority(e,r) = w_impact'*Impact_norm - w_risk'*Risk_norm   (not_estimable;
w_impact,w_risk renormalized to sum to 1 -- the term is DROPPED, not zeroed)

All three normalized components are min-max scaled across the 5 error
types WITHIN a fixed rate, so error types are compared at matched
corruption severity rather than blending different severities together.

NOISE_FLOOR = 0.003 and MAJORITY_THRESHOLD = 0.8 are explicit, documented
parameters -- not hidden magic numbers -- chosen from the observed
run-to-run std of baseline F1 itself across seeds (~0.0025 in Phase 7/8),
i.e. effects smaller than this aren't distinguishable from pure RF
training noise.
"""

import itertools
import os

import numpy as np
import pandas as pd

NOISE_FLOOR = 0.003
MAJORITY_THRESHOLD = 0.8
HEADLINE_RATE = 0.10
ALL_RATES = [0.05, 0.10, 0.20]

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def load_evidence_table() -> pd.DataFrame:
    """Build the consolidated per (error_type, rate) evidence table from
    Phase 7's impact summary and Phase 9's raw per-seed repair results.
    Nothing here is reduced to a single score yet.
    """
    impact = pd.read_csv(os.path.join(RESULTS_DIR, "phase7_impact_summary_full.csv"))
    repair_raw = pd.read_csv(os.path.join(RESULTS_DIR, "phase9_repair_raw_full.csv"))

    rows = []
    for (error_type, rate), g in repair_raw.groupby(["error_type", "rate"]):
        n = len(g)
        n_pos = int((g["recovery"] > 0).sum())
        n_neg = int((g["recovery"] < 0).sum())
        mean_recovery = g["recovery"].mean()
        agreement = max(n_pos, n_neg) / n

        if agreement >= MAJORITY_THRESHOLD and mean_recovery >= NOISE_FLOOR:
            status = "positive"
        elif agreement >= MAJORITY_THRESHOLD and mean_recovery <= -NOISE_FLOOR:
            status = "negative"
        else:
            status = "weak"
        evidence_status = {"positive": "complete", "negative": "complete",
                            "weak": "partial", "not_estimable": "none"}[status]

        harmful_seeds = g.loc[g["recovery"] < 0, "recovery"]
        p_harm = n_neg / n
        harm_magnitude = abs(harmful_seeds.mean()) if len(harmful_seeds) else 0.0
        risk = p_harm * harm_magnitude

        damage_row = impact[(impact.error_type == error_type) & (impact.rate == rate)].iloc[0]

        rows.append({
            "error_type": error_type, "rate": rate,
            "mean_damage": damage_row["mean_damage"], "damage_std": damage_row["std_damage"],
            "repair_precision": g["repair_precision"].mean(), "repair_recall": g["repair_recall"].mean(),
            "mean_recovery": mean_recovery, "sign_agreement": agreement,
            "n_pos_seeds": n_pos, "n_neg_seeds": n_neg,
            "repairability_status": status, "evidence_status": evidence_status,
            "p_harm": p_harm, "harm_magnitude": harm_magnitude, "risk": risk,
        })

    return pd.DataFrame(rows).sort_values(["rate", "error_type"]).reset_index(drop=True)


def _minmax(series: pd.Series) -> pd.Series:
    lo, hi = series.min(), series.max()
    if hi - lo < 1e-12:
        return pd.Series(0.5, index=series.index)  # all equal -- no information, stay neutral
    return (series - lo) / (hi - lo)


def compute_priority(evidence: pd.DataFrame, rate: float, w_impact: float, w_risk: float,
                      w_repair: float) -> pd.DataFrame:
    """Priority scores for all 5 error types at a single rate, for one
    weight triple. w_impact + w_risk + w_repair should sum to 1, but this
    isn't enforced here -- the sensitivity grid intentionally explores off
    that constraint too, see run_sensitivity_grid().
    """
    df = evidence[evidence["rate"] == rate].copy()
    df["impact_norm"] = _minmax(df["mean_damage"])
    df["risk_norm"] = _minmax(df["risk"])

    estimable = df["repairability_status"] != "not_estimable"
    df["repairability_norm"] = np.nan
    if estimable.any():
        df.loc[estimable, "repairability_norm"] = _minmax(df.loc[estimable, "mean_recovery"])

    priority = pd.Series(index=df.index, dtype=float)
    for idx, row in df.iterrows():
        if pd.isna(row["repairability_norm"]):
            total = w_impact + w_risk
            wi, wr = (w_impact / total, w_risk / total) if total > 0 else (0.5, 0.5)
            priority[idx] = wi * row["impact_norm"] - wr * row["risk_norm"]
        else:
            priority[idx] = (w_impact * row["impact_norm"] - w_risk * row["risk_norm"]
                              + w_repair * row["repairability_norm"])
    df["priority_score"] = priority
    return df.sort_values("priority_score", ascending=False).reset_index(drop=True)


def run_sensitivity_grid(evidence: pd.DataFrame, rate: float,
                          weight_values=(0.2, 0.3, 0.4, 0.5, 0.6)) -> pd.DataFrame:
    """Every (w_impact, w_risk, w_repair) combo from weight_values that sums
    to 1 (within tolerance). For each, compute the ranking and record each
    error type's RANK POSITION (1=highest priority). The question this
    answers: does the priority ORDER actually depend on which weights you
    picked, or is it robust across a wide range of reasonable choices?
    """
    combos = [w for w in itertools.product(weight_values, repeat=3) if abs(sum(w) - 1.0) < 1e-9]
    error_types = sorted(evidence["error_type"].unique())
    rank_records = {et: [] for et in error_types}

    for w_impact, w_risk, w_repair in combos:
        ranked = compute_priority(evidence, rate, w_impact, w_risk, w_repair)
        for rank, (_, row) in enumerate(ranked.iterrows(), start=1):
            rank_records[row["error_type"]].append(rank)

    rows = []
    for et in error_types:
        ranks = rank_records[et]
        rows.append({
            "error_type": et, "n_weight_combos": len(ranks),
            "mean_rank": np.mean(ranks), "min_rank": min(ranks), "max_rank": max(ranks),
            "rank_1_pct": sum(r == 1 for r in ranks) / len(ranks),
            "rank_stable": max(ranks) - min(ranks) <= 1,  # never moves by more than one position
        })
    return pd.DataFrame(rows).sort_values("mean_rank")


if __name__ == "__main__":
    evidence = load_evidence_table()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    evidence.to_csv(os.path.join(RESULTS_DIR, "phase10_evidence_table.csv"), index=False)

    print("=" * 110)
    print("PHASE 10 EVIDENCE TABLE (full, all rates) -- underlying metrics retained, not reduced yet")
    print("=" * 110)
    print(evidence[["error_type", "rate", "mean_damage", "repair_precision", "repair_recall",
                     "mean_recovery", "repairability_status", "evidence_status", "risk"]]
          .round(4).to_string(index=False))

    print(f"\n{'=' * 110}")
    print(f"HEADLINE PRIORITY RANKING at rate={HEADLINE_RATE:.0%} (equal weights 1/3, 1/3, 1/3 -- provisional)")
    print("=" * 110)
    headline = compute_priority(evidence, HEADLINE_RATE, w_impact=1 / 3, w_risk=1 / 3, w_repair=1 / 3)
    print(headline[["error_type", "impact_norm", "risk_norm", "repairability_norm",
                     "priority_score", "repairability_status", "evidence_status"]].round(3).to_string(index=False))
    headline.to_csv(os.path.join(RESULTS_DIR, "phase10_priority_headline.csv"), index=False)

    print(f"\n{'=' * 110}")
    print("SENSITIVITY ANALYSIS -- rank position across the full weight grid (w in {0.2..0.6}, sum=1)")
    print("=" * 110)
    sensitivity = run_sensitivity_grid(evidence, HEADLINE_RATE)
    print(sensitivity.round(3).to_string(index=False))
    sensitivity.to_csv(os.path.join(RESULTS_DIR, "phase10_sensitivity.csv"), index=False)

    print(f"\nSaved phase10_evidence_table.csv, phase10_priority_headline.csv, phase10_sensitivity.csv")
