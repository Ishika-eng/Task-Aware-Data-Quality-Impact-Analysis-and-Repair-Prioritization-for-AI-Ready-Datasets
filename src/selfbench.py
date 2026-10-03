"""
Self-benchmark: calibrate TaskClean on the uploaded dataset itself.

The research phases (7-9) measured, on UCI Adult, how much each error type hurts a model, how well repair
recovers it, and whether repair ever makes things worse. Those numbers do not automatically transfer to another
dataset. This module re-runs a small version of those experiments on the user's own data:

  1. Build a CLEAN-ISH REFERENCE from the data: complete rows, no duplicates, high-confidence label problems
     removed. It is the best clean data we can derive, not true ground truth.
  2. Split it into train/test. The test rows are never corrupted.
  3. For each error type x rate (5/10/20%) x seed: inject ONLY that error into the training rows (known ground
     truth), train a Random Forest, measure the macro-F1 damage against a clean-trained model with the same
     seed (paired, as in Phase 7); then run the REAL TaskClean path (audit_dataset -> _apply) on the corrupted
     training data and measure recovery, repair precision/recall and harm (as in Phase 9).
  4. Aggregate into the same evidence table the Adult phases produced (repairability status, risk, repair
     precision/recall), plus this dataset's damage curve, detector precision/recall, detector false-positive
     floors, and whether each detector's flagged rate tracks real damage (as in Phase 11B).

The product then applies the SAME safe-auto rule to this evidence instead of the Adult evidence.

It deliberately declines to produce evidence when it cannot be meaningful (too little clean-ish data, or a target
that is barely predictable from the features): "no harm observed" would then be vacuous. In that case the Adult
evidence stays in force and the reason is reported.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from baseline import build_preprocessor
from corruption import CORRUPTORS
from taskclean import AuditState, _apply, _full_record, _stratified_sample_index, audit_dataset

RATES = (0.05, 0.10, 0.20)
BENCH_ISSUES = ("missing_values", "duplicates", "outliers", "label_errors")
DETECTOR_OF = {"label_errors": "label_noise"}                      # taskclean issue key -> detector name
UNIT_OF = {"missing_values": "cells", "outliers": "cells", "duplicates": "rows", "label_errors": "rows"}

# --- documented parameters ---------------------------------------------------------------------------------
MIN_REFERENCE_ROWS = 600         # below this the held-out test set is too small to measure small effects
MIN_CLASS_ROWS = 20              # classes rarer than this are dropped from the reference (can't be split/scored)
MAX_MISSING_COLUMN = 0.30        # columns missing more than this are left out of the reference
MAX_REFERENCE_ROWS = 6000        # stratified cap, for speed
LEARNABLE_MARGIN = 0.05          # clean-trained macro-F1 must beat the chance level (see below) by this much
NOISE_FLOOR_MIN = 0.003          # same minimum as Phase 10 (RF run-to-run std on Adult was ~0.0025)
NULL_DROP_FRACTION = 0.10        # null experiment: retrain on a random 90% of the training rows (should change nothing
                                 # that matters): a perturbation the size of the corruptions being studied
NULL_DRAWS = 4                   # null draws per seed
SIGN_AGREEMENT = 0.8             # share of seeds that must agree on the sign of recovery (Phase 10)
TEST_FRACTION = 0.30
SEED0 = 42

Progress = Callable[[float, str], None]


@dataclass
class SelfBench:
    ok: bool
    reason: str | None = None
    metric: str = "macro-F1"
    n_estimators: int = 100
    n_seeds: int = 0
    rates: tuple = RATES
    reference: dict = field(default_factory=dict)
    noise_floor: float = NOISE_FLOOR_MIN
    null_noise: float = float("nan")
    baseline_f1: dict = field(default_factory=dict)
    majority_f1: float = float("nan")
    chance_f1: float = float("nan")
    runs: pd.DataFrame | None = None
    evidence: pd.DataFrame | None = None
    impact_curve: pd.DataFrame | None = None
    detector_reliability: pd.DataFrame | None = None
    detector_tracking: pd.DataFrame | None = None
    baseline_floors: dict = field(default_factory=dict)
    skipped_issues: dict = field(default_factory=dict)
    seconds: float = 0.0

    def summary(self) -> dict:
        if not self.ok:
            return {"ran": False, "reason": self.reason, "reference": self.reference}
        rec = lambda df: df.to_dict("records")          # noqa: E731
        return {
            "ran": True, "metric": self.metric, "model": f"RandomForestClassifier (n_estimators={self.n_estimators})",
            "seeds": self.n_seeds, "rates": list(self.rates), "reference": self.reference,
            "noise_floor": self.noise_floor, "null_experiment_rms_f1_change": self.null_noise,
            "baseline_f1": {str(k): v for k, v in self.baseline_f1.items()},
            "majority_class_f1": self.majority_f1, "chance_level_f1": self.chance_f1,
            "evidence": rec(self.evidence), "impact_curve": rec(self.impact_curve),
            "detector_reliability": rec(self.detector_reliability),
            "detector_tracking": rec(self.detector_tracking),
            "detector_false_positive_floors_on_reference": self.baseline_floors,
            "issues_not_measured": self.skipped_issues, "seconds": round(self.seconds, 1),
        }


# ---------------------------------------------------------------------------------------------------------------
# Evidence rules (kept as plain functions so they can be checked against the Adult results)
# ---------------------------------------------------------------------------------------------------------------
def classify_repairability(recoveries, noise_floor: float = NOISE_FLOOR_MIN,
                           agreement_needed: float = SIGN_AGREEMENT) -> str:
    """Phase 10's rule: 'positive'/'negative' only if >= 80% of seeds agree on the sign of recovery AND the mean
    recovery clears the noise floor; otherwise 'weak' (evidence exists but is not a confident signal)."""
    rec = np.asarray(recoveries, dtype=float)
    agreement = max((rec > 0).sum(), (rec < 0).sum()) / len(rec)
    mean = rec.mean()
    if agreement >= agreement_needed and mean >= noise_floor:
        return "positive"
    if agreement >= agreement_needed and mean <= -noise_floor:
        return "negative"
    return "weak"


def material_harm_risk(recoveries, noise_floor: float) -> tuple[float, float]:
    """(P(material harm), risk) where harm only counts if recovery < -noise_floor. On small datasets the F1
    noise is larger than on Adult, and counting every tiny negative as harm would reject even perfectly harmless
    repairs (e.g. exact duplicate removal) -- so the product's safe-auto rule uses this, while the Phase 10 risk
    is still reported next to it."""
    rec = np.asarray(recoveries, dtype=float)
    material = rec[rec < -noise_floor]
    p = len(material) / len(rec)
    return p, p * (float(abs(material.mean())) if len(material) else 0.0)


def harm_risk(recoveries) -> tuple[float, float, float]:
    """(P(harm), mean harm magnitude when harm occurred, risk = product) -- Phase 10's definition."""
    rec = np.asarray(recoveries, dtype=float)
    harmful = rec[rec < 0]
    p_harm = len(harmful) / len(rec)
    magnitude = float(abs(harmful.mean())) if len(harmful) else 0.0
    return p_harm, magnitude, p_harm * magnitude


# ---------------------------------------------------------------------------------------------------------------
# Reference set
# ---------------------------------------------------------------------------------------------------------------
def _build_reference(state: AuditState, max_rows: int, seed: int):
    """Returns (X_ref, y_ref, info) or (None, None, {"reason": ...})."""
    if state.y is None:
        return None, None, {"reason": "no target column: the self-benchmark trains a classifier and needs labels"}
    X, y = state.X, state.y
    labeled = y.notna()
    keep, dropped = [], {}
    for c in X.columns:
        if c in state.unmodeled_columns:
            dropped[c] = "high-cardinality text"
        elif X.loc[labeled, c].isna().mean() > MAX_MISSING_COLUMN:
            dropped[c] = f"{X.loc[labeled, c].isna().mean():.0%} missing"
        else:
            keep.append(c)
    if not keep:
        return None, None, {"reason": "no feature columns are complete enough to build a reference"}

    Xr, yr = X.loc[labeled, keep], y.loc[labeled]
    n_labeled = len(Xr)
    complete = ~Xr.isna().any(axis=1)
    Xr, yr = Xr[complete], yr[complete]
    n_complete = len(Xr)
    dup = _full_record(Xr, yr).duplicated(keep="first")
    Xr, yr = Xr[~dup], yr[~dup]
    n_dedup = len(Xr)
    prop = state.proposals
    n_label_removed = 0
    if not prop.empty:
        bad = Xr.index.intersection(pd.Index(prop.loc[(prop.issue_key == "label_errors")
                                                      & (prop.tier == "strict"), "row_id"]))
        n_label_removed = len(bad)
        Xr, yr = Xr.drop(index=bad), yr.drop(index=bad)
    counts = yr.value_counts()
    rare = list(counts[counts < MIN_CLASS_ROWS].index)
    if rare:
        keep_rows = ~yr.isin(rare)
        Xr, yr = Xr[keep_rows], yr[keep_rows]
    info = {"rows_labeled": n_labeled, "rows_complete": n_complete, "rows_after_dedup": n_dedup,
            "rows_removed_label_problems": n_label_removed, "rare_classes_dropped": [str(c) for c in rare],
            "columns_left_out": dropped, "columns_used": keep}
    if yr.nunique() < 2 or len(Xr) < MIN_REFERENCE_ROWS:
        info["reason"] = (f"only {len(Xr):,} clean-ish rows remain after removing incomplete rows, duplicates and "
                          f"high-confidence label problems (need at least {MIN_REFERENCE_ROWS:,} and 2 classes)")
        return None, None, info
    if len(Xr) > max_rows:
        idx = _stratified_sample_index(yr, Xr.index, max_rows, seed)
        Xr, yr = Xr.loc[idx], yr.loc[idx]
        info["capped_to_rows"] = len(Xr)
    info["rows_used"] = len(Xr)
    return Xr, yr, info


# ---------------------------------------------------------------------------------------------------------------
# Modelling and grading helpers
# ---------------------------------------------------------------------------------------------------------------
def _fit_eval(X_tr, y_tr, X_te, y_te, seed: int, n_estimators: int) -> float:
    pipe = Pipeline([("prep", build_preprocessor(X_tr)),
                     ("clf", RandomForestClassifier(n_estimators=n_estimators, random_state=seed, n_jobs=-1))])
    pipe.fit(X_tr, y_tr)
    return float(f1_score(y_te, pipe.predict(X_te), average="macro", zero_division=0))


def _same_data(Xa: pd.DataFrame, ya: pd.Series, Xb: pd.DataFrame, yb: pd.Series) -> bool:
    """True if two training sets hold exactly the same rows in the same order (ignoring row labels)."""
    return (len(Xa) == len(Xb)
            and Xa.reset_index(drop=True).equals(Xb.reset_index(drop=True))
            and ya.astype(str).reset_index(drop=True).equals(yb.astype(str).reset_index(drop=True)))


def _prf(true_set: set, got_set: set) -> tuple[float, float]:
    tp = len(true_set & got_set)
    precision = tp / len(got_set) if got_set else float("nan")
    recall = tp / len(true_set) if true_set else float("nan")
    return precision, recall


def _detected_set(issue: str, st: AuditState, orig_index: pd.Index) -> set:
    """What the (blind) detector flagged, in the corruptor's row-label space."""
    if issue == "missing_values":
        cm = st.detections["missing_values"].extra["cell_mask"]
        return {(orig_index[i], c) for c in cm.columns for i in np.flatnonzero(cm[c].values)}
    if issue == "outliers":
        flags = st.detections["outliers"].extra["per_column_flags"]
        return {(orig_index[i], c) for c, f in flags.items()
                for i in np.flatnonzero(f.reindex(st.X.index, fill_value=False).values)}
    key = DETECTOR_OF.get(issue, issue)
    mask = st.detections[key].row_mask
    return {orig_index[i] for i in np.flatnonzero(mask.reindex(st.X.index, fill_value=False).values)}


def _repaired_set(issue: str, rlog: pd.DataFrame, orig_index: pd.Index) -> set:
    applied = rlog[rlog.action == "applied"]
    pos = applied.row_id.astype(int).values
    if UNIT_OF[issue] == "cells":
        return {(orig_index[p], c) for p, c in zip(pos, applied.column)}
    return {orig_index[p] for p in pos}


def _true_set(issue: str, log: pd.DataFrame) -> set:
    if UNIT_OF[issue] == "cells":
        return set(zip(log.row_id, log.column))
    return set(log.row_id)


# ---------------------------------------------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------------------------------------------
def _evidence_table(runs: pd.DataFrame, noise_floor: float) -> pd.DataFrame:
    """Same columns as the Adult Phase 10 table, plus noise-aware material harm.

    Exact restoration: when a repair reproduces the clean reference EXACTLY (checked by comparing the data), a
    negative 'recovery' only means the corruption happened to help by chance; restoring the clean data cannot
    be harmful. Such runs never count as harm, and their status is 'positive' (the repair recovers real damage)
    or 'weak', never 'negative'."""
    rows = []
    for (issue, rate), g in runs.groupby(["issue", "rate"]):
        rec = g["recovery"].values
        exact = g["restored_exactly"].values.astype(bool)
        n_pos, n_neg = int((rec > 0).sum()), int((rec < 0).sum())
        p_harm, magnitude, risk = harm_risk(rec)                       # Phase 10's definition, reported as-is
        if exact.all():
            dmg = g["damage"].values
            agree = max((dmg > 0).sum(), (dmg < 0).sum()) / len(dmg)
            gain_agree = (dmg > 0).sum() / len(dmg)
            status = "positive" if gain_agree >= SIGN_AGREEMENT and dmg.mean() >= noise_floor else "weak"
            p_material, material_risk = 0.0, 0.0
        else:
            status = classify_repairability(rec, noise_floor)
            harmed = (rec < -noise_floor) & ~exact
            p_material = harmed.mean()
            material_risk = p_material * (float(abs(rec[harmed].mean())) if harmed.any() else 0.0)
        rows.append({
            "error_type": issue, "rate": rate, "mean_damage": g["damage"].mean(),
            "damage_std": g["damage"].std(ddof=1) if len(g) > 1 else float("nan"),
            "repair_precision": g["repair_precision"].mean(), "repair_recall": g["repair_recall"].mean(),
            "mean_recovery": rec.mean(), "sign_agreement": max(n_pos, n_neg) / len(rec),
            "n_pos_seeds": n_pos, "n_neg_seeds": n_neg, "repairability_status": status,
            "evidence_status": {"positive": "complete", "negative": "complete", "weak": "partial"}[status],
            "p_harm": p_harm, "harm_magnitude": magnitude, "risk": risk,
            "p_material_harm": p_material, "material_risk": material_risk,
            "restored_exactly": bool(exact.all()),
        })
    return pd.DataFrame(rows).sort_values(["rate", "error_type"]).reset_index(drop=True)


def _tracking_table(runs: pd.DataFrame, noise_floor: float) -> pd.DataFrame:
    """Does each detector's flagged rate track real damage on THIS dataset? (Phase 11B on the user's data.)
    Three rates only, so this is directional evidence, never 'validated'."""
    rows = []
    for issue, g in runs.groupby("issue"):
        pts = g.groupby("rate").agg(det=("detected_rate", "mean"), dmg=("damage", "mean")).reset_index()
        r = rho = float("nan")
        if len(pts) < 3 or pts["det"].std() < 1e-9 or pts["dmg"].abs().max() < noise_floor \
                or pts["dmg"].std() < 1e-12:
            status = "not_estimable"
            note = ("On this dataset the damage from this error stays below the noise floor (or too few rates were "
                    "measured), so the detected rate cannot be checked against real damage.")
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                r = float(pearsonr(pts["det"], pts["dmg"])[0])
                rho = float(spearmanr(pts["det"], pts["dmg"])[0])
            if r <= -0.8:
                status = "inverted"
                note = (f"INVERTED on this dataset: the detected rate fell as true corruption and measured damage "
                        f"rose (r = {r:.2f} over {len(pts)} rates). Do not read it as a severity signal.")
            elif r >= 0.8 and rho >= 0.5:
                status = "weak"
                note = (f"The detected rate rose with measured damage on this dataset (r = {r:.2f}, "
                        f"{len(pts)} rates: directional only, not statistically significant).")
            else:
                status = "weak"
                note = (f"No clear relationship between detected rate and measured damage on this dataset "
                        f"(r = {r:.2f}, {len(pts)} rates).")
        rows.append({"error_type": issue, "pearson_r": r, "spearman_r": rho, "n_points": len(pts),
                     "evidence_status": status, "note": note})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------------------------------------------
def run_self_benchmark(state: AuditState, n_seeds: int = 3, rates=RATES, max_rows: int = MAX_REFERENCE_ROWS,
                       n_estimators: int = 100, progress: Progress | None = None) -> SelfBench:
    """Measure impact, repair effectiveness, harm risk and detector reliability on this dataset. Never raises
    for 'data not suitable' conditions: returns SelfBench(ok=False, reason=...) so the caller can fall back."""
    if n_seeds < 2:
        raise ValueError("n_seeds must be at least 2 (sign agreement and noise are meaningless with one run)")
    say = progress or (lambda f, m: None)
    t0 = time.monotonic()
    rates = tuple(rates)

    say(0.0, "Building a clean-ish reference from your data")
    Xr, yr, info = _build_reference(state, max_rows, SEED0)
    if Xr is None:
        return SelfBench(ok=False, reason=info.pop("reason"), reference=info, n_estimators=n_estimators)

    X_tr, X_te, y_tr, y_te = train_test_split(Xr, yr, test_size=TEST_FRACTION, stratify=yr, random_state=SEED0)
    info.update(rows_train=len(X_tr), rows_test=len(X_te))
    seeds = [SEED0 + i for i in range(n_seeds)]

    say(0.02, "Training clean reference models")
    baseline = {s: _fit_eval(X_tr, y_tr, X_te, y_te, s, n_estimators) for s in seeds}
    majority = y_tr.mode()[0]
    majority_f1 = float(f1_score(y_te, [majority] * len(y_te), average="macro", zero_division=0))
    # Chance level = the same model trained on SHUFFLED labels. (Always guessing the majority class is a poor
    # yardstick for macro-F1: a coin-flip classifier already scores ~0.5 on a balanced binary target.)
    say(0.03, "Checking the target is predictable at all")
    perm_rng = np.random.default_rng(SEED0 + 1)
    shuffled = [_fit_eval(X_tr, pd.Series(perm_rng.permutation(y_tr.values), index=y_tr.index), X_te, y_te, s,
                          n_estimators) for s in seeds]
    chance_f1 = max(majority_f1, float(np.mean(shuffled)))
    if np.mean(list(baseline.values())) - chance_f1 < LEARNABLE_MARGIN:
        return SelfBench(
            ok=False, reference=info, n_estimators=n_estimators, n_seeds=n_seeds, baseline_f1=baseline,
            majority_f1=majority_f1, chance_f1=chance_f1, seconds=time.monotonic() - t0,
            reason=(f"the target is barely predictable from the features (clean-trained macro-F1 "
                    f"{np.mean(list(baseline.values())):.3f} vs {chance_f1:.3f} for the same model trained on "
                    "shuffled labels), so data-quality damage cannot be measured above the noise"))
    # Noise floor from a NULL experiment: how much does F1 move when we perturb the training data in a way that
    # should not matter (drop a random 10% of rows)? Effects smaller than that are indistinguishable from noise.
    say(0.04, "Measuring the noise floor (null experiment)")
    rng = np.random.default_rng(SEED0)
    null_diffs = []
    for seed in seeds:
        for _ in range(NULL_DRAWS):
            keep = rng.random(len(X_tr)) >= NULL_DROP_FRACTION
            null_diffs.append(_fit_eval(X_tr[keep], y_tr[keep], X_te, y_te, seed, n_estimators) - baseline[seed])
    null_noise = float(np.sqrt(np.mean(np.square(null_diffs))))
    noise_floor = max(NOISE_FLOOR_MIN, null_noise)

    target = state.target
    has_numeric = any(pd.api.types.is_numeric_dtype(X_tr[c]) for c in X_tr.columns)
    issues, skipped = [], {}
    for issue in BENCH_ISSUES:
        if issue == "outliers" and not has_numeric:
            skipped[issue] = "no numeric columns to corrupt"
        else:
            issues.append(issue)

    total = len(issues) * len(rates) * n_seeds + 1
    done = 0
    rows = []
    for issue in issues:
        try:
            for rate in rates:
                for seed in seeds:
                    say(0.05 + 0.9 * done / total, f"{issue} at {rate:.0%} (seed {seed})")
                    X_d, y_d, log = CORRUPTORS[issue](X_tr, y_tr, rate=rate, seed=seed)
                    dirty_f1 = _fit_eval(X_d, y_d, X_te, y_te, seed, n_estimators)

                    # the REAL product path on the corrupted training data
                    d = X_d.copy()
                    d[target] = y_d.values
                    st = audit_dataset(d, target, detectors={DETECTOR_OF.get(issue, issue)}, seed=seed)
                    cleaned, rlog = _apply(st, {issue}, [])
                    X_rep, y_rep = cleaned.drop(columns=[target]), cleaned[target].astype(str)
                    repaired_f1 = _fit_eval(X_rep, y_rep, X_te, y_te, seed, n_estimators)
                    restored_exactly = _same_data(X_rep, y_rep, X_tr, y_tr)

                    orig_index = X_d.index
                    truth = _true_set(issue, log)
                    det_p, det_r = _prf(truth, _detected_set(issue, st, orig_index))
                    rep_p, rep_r = _prf(truth, _repaired_set(issue, rlog, orig_index))
                    det = st.detections[DETECTOR_OF.get(issue, issue)]
                    rows.append({
                        "issue": issue, "rate": rate, "seed": seed, "baseline_f1": baseline[seed],
                        "dirty_f1": dirty_f1, "repaired_f1": repaired_f1,
                        "damage": baseline[seed] - dirty_f1, "recovery": repaired_f1 - dirty_f1,
                        "detected_rate": det.rate, "detector_precision": det_p, "detector_recall": det_r,
                        "repair_precision": rep_p, "repair_recall": rep_r,
                        "restored_exactly": restored_exactly,
                    })
                    done += 1
        except Exception as e:                      # one failing issue must not sink the whole calibration
            skipped[issue] = f"failed during calibration: {type(e).__name__}: {e}"
            rows = [r for r in rows if r["issue"] != issue]

    if not rows:
        return SelfBench(ok=False, reference=info, n_estimators=n_estimators, n_seeds=n_seeds,
                         baseline_f1=baseline, majority_f1=majority_f1, skipped_issues=skipped,
                         seconds=time.monotonic() - t0, reason="no error type could be benchmarked on this dataset")
    runs = pd.DataFrame(rows)

    say(0.96, "Measuring detector false-positive floors on the reference")
    ref_df = Xr.copy()
    ref_df[target] = yr.values
    floors_state = audit_dataset(ref_df, target, detectors={"missing_values", "duplicates", "outliers", "label_noise"},
                                 seed=SEED0)
    floors = {("label_errors" if k == "label_noise" else k): float(v.rate) for k, v in floors_state.detections.items()}

    evidence = _evidence_table(runs, noise_floor)
    impact_curve = (runs.groupby(["issue", "rate"])["damage"].agg(["mean", "std"]).reset_index()
                    .rename(columns={"issue": "error_type", "mean": "mean_damage", "std": "damage_std"}))
    reliability = (runs.groupby("issue")[["detector_precision", "detector_recall"]].mean().reset_index()
                   .rename(columns={"issue": "error_type", "detector_precision": "precision",
                                    "detector_recall": "recall"}))
    reliability.insert(1, "unit", reliability["error_type"].map(UNIT_OF))
    say(1.0, "Self-benchmark complete")
    return SelfBench(ok=True, n_estimators=n_estimators, n_seeds=n_seeds, rates=rates, reference=info,
                     noise_floor=noise_floor, null_noise=null_noise, baseline_f1=baseline, majority_f1=majority_f1,
                     chance_f1=chance_f1, runs=runs,
                     evidence=evidence, impact_curve=impact_curve, detector_reliability=reliability,
                     detector_tracking=_tracking_table(runs, noise_floor), baseline_floors=floors,
                     skipped_issues=skipped, seconds=time.monotonic() - t0)
