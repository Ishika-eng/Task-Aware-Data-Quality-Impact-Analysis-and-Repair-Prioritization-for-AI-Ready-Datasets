"""
Phase 15 validation (method and failure criteria: docs/PHASE15_SPEC.md, written before this ran).

Detection and attribution are scored against an injection log; there is no repair, so there is no model-recovery
measurement. Datasets: synthetic data with known correlation (rho 0 / 0.5 / 0.8, Gaussian and skewed), plus real data
(German credit, Diabetes, Breast Cancer as evidence; Heart, Bank Marketing, Adult, Wine as limited / gating checks).

`--design original` is the pre-registered method (average ranks for ties). `--design amended` is Amendment 2 (random
tie-breaking). Both are always reported side by side; the original criteria are never re-labelled as met.

Usage (from the project root):
    .venv/bin/python src/phase15_validation.py --design original          # one seed: smoke test
    .venv/bin/python src/phase15_validation.py --design original --full   # seeds 42-46 (the real run)
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_diabetes, load_wine
from sklearn.ensemble import IsolationForest

import sklearn
from data import load_adult_clean
from detect import detect_feature_corruption_crossfeature, detect_outliers_iqr
from joint_outliers import UNIVARIATE_K, copula, detect_joint_outliers, eligible_columns
from phase15_injectors import KINDS, inject_joint

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RATES = [0.02, 0.05, 0.10]
SEEDS_FULL = [42, 43, 44, 45, 46]
DESIGNS = {"original": "average", "amended": "random"}
SWAPS = {"swap_one", "swap_block"}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ------------------------------------------------------------------------------------------------------ datasets
def synthetic(rho: float, skew: bool, seed: int, n: int = 5000, d: int = 6) -> pd.DataFrame:
    """Two blocks of d/2 columns, latent Pearson correlation rho inside a block and 0 between blocks."""
    rng = np.random.default_rng([seed, 7])
    half = d // 2
    block = np.full((half, half), rho)
    np.fill_diagonal(block, 1.0)
    L = np.linalg.cholesky(block)
    Z = np.hstack([rng.normal(size=(n, half)) @ L.T for _ in range(2)])
    if skew:
        Z = np.exp(Z)
    return pd.DataFrame(Z, columns=[f"s{i}" for i in range(d)])


def _csv(name):
    return pd.read_csv(os.path.join(DATA_DIR, name))


def real_datasets() -> dict[str, tuple[pd.DataFrame, str]]:
    """name -> (frame, group). `evidence` datasets decide F1/F3; `limited` ones are reported as gating checks."""
    adult, _ = load_adult_clean()
    return {
        "german_credit": (_csv("credit_g_original.csv"), "evidence"),
        "diabetes": (load_diabetes(as_frame=True).data, "evidence"),
        "breast_cancer": (load_breast_cancer(as_frame=True).data, "evidence"),
        "heart": (_csv("heart_c_original.csv"), "limited"),
        "bank_marketing": (_csv("bank_marketing_original.csv"), "limited"),
        "adult": (adult.reset_index(drop=True), "limited"),
        "wine": (load_wine(as_frame=True).data, "gating"),
    }


def all_datasets(seed: int):
    out = {}
    for rho in (0.0, 0.5, 0.8):
        for skew in (False, True):
            out[f"synthetic_rho{rho}{'_skew' if skew else ''}"] = (synthetic(rho, skew, seed), "evidence")
    out.update(real_datasets())
    return out


# ------------------------------------------------------------------------------------------------------- measuring
def _rate(hit, total):
    return hit / total if total else np.nan


def isolation_forest_flags(Xd, cols, budget, seed, ties):
    """Isolation Forest on the same copula-transformed columns, flagging exactly `budget` rows (the equal-budget check)."""
    sub = Xd[cols].dropna()
    if budget == 0 or len(sub) < 200 or len(cols) < 3:
        return set()
    Z = copula(sub.to_numpy(dtype=float), ties, seed)
    score = IsolationForest(n_estimators=300, random_state=seed, n_jobs=-1).fit(Z).score_samples(Z)
    return set(sub.index[np.argsort(score)[:budget]])


def run_one(name, X, group, rate, seed, ties, with_overlap=False):
    cols = eligible_columns(X)[0]
    Xd, _, inj = inject_joint(X, pd.Series(np.zeros(len(X))), rate=rate, seed=seed, columns=cols)
    res = detect_joint_outliers(Xd, seed=seed, ties=ties)
    rec = res.extra["records"]
    F = set(res.extra["row_mask_any"][res.extra["row_mask_any"]].index)
    F_multi = set(rec.row_id[rec.scope == "multivariate"])
    F_bi = set(rec.row_id[rec.scope != "multivariate"])

    row = inj.groupby("row_id").agg(kind=("kind", "first"), invisible=("invisible", "first"))
    R = set(row.index)
    inv_swap = {r for r in R if row.at[r, "kind"] in SWAPS and row.at[r, "invisible"]}
    by_kind = {k: {r for r in R if row.at[r, "kind"] == k} for k in KINDS}

    iqr = detect_outliers_iqr(Xd[cols], k=UNIVARIATE_K).row_mask if cols else pd.Series(False, index=Xd.index)
    I = set(iqr[iqr].index)
    budget = len(F)
    IF = isolation_forest_flags(Xd, cols, budget, seed, ties) if res.extra["scopes"] else set()

    out = {"dataset": name, "group": group, "rate": rate, "seed": seed, "design": ties,
           "n_rows": len(Xd), "n_eligible_columns": len(cols), "n_pairs_tested": len(res.extra["pairs_tested"]),
           "scopes_ran": len(res.extra["scopes"]), "n_injected_rows": len(R), "n_flagged": len(F),
           "n_flagged_multi": len(F_multi), "n_flagged_bi": len(F_bi), "n_joint_only": res.extra["n_joint_only"],
           "precision": _rate(len(F & R), len(F)), "recall": _rate(len(F & R), len(R)),
           "recall_multi": _rate(len(F_multi & R), len(R)), "recall_bi": _rate(len(F_bi & R), len(R)),
           "n_invisible_swaps": len(inv_swap), "hits_invisible_swaps": len(F & inv_swap),
           "n_inflate": len(by_kind["inflate"]), "hits_inflate": len(F & by_kind["inflate"]),
           "iqr_precision": _rate(len(I & R), len(I)), "iqr_recall": _rate(len(I & R), len(R)),
           "iqr_hits_invisible_swaps": len(I & inv_swap), "iqr_hits_inflate": len(I & by_kind["inflate"]),
           "if_budget": budget, "if_recall": _rate(len(IF & R), len(R)) if res.extra["scopes"] else np.nan,
           "if_hits_invisible_swaps": len(IF & inv_swap)}
    for k in KINDS:
        out[f"n_{k}"], out[f"hits_{k}"] = len(by_kind[k]), len(F & by_kind[k])

    # attribution: swap_one rows that were flagged. The record's column must be the injected column.
    one = inj[inj.kind == "swap_one"].set_index("row_id")["column"].to_dict()
    for tag, want in (("multi", rec.scope == "multivariate"), ("bi", rec.scope != "multivariate")):
        sub = rec[want & rec.row_id.isin(one)]
        out[f"attr_{tag}_n"] = len(sub)
        out[f"attr_{tag}_correct"] = int(sum(c == one[r] for r, c in zip(sub.row_id, sub["column"])))

    if with_overlap:
        cf = detect_feature_corruption_crossfeature(Xd, seed=seed).row_mask
        C = set(cf[cf].index)
        out["overlap_cf_n"], out["overlap_cf_of_joint"] = len(C), _rate(len(C & F), len(F))
        out["cf_recall"] = _rate(len(C & R), len(R))
    return out


def run_clean(name, X, group, seed, ties):
    res = detect_joint_outliers(X, seed=seed, ties=ties)
    rec = res.extra["records"]
    n = len(X)
    return {"dataset": name, "group": group, "seed": seed, "design": ties, "n_rows": n,
            "multi_ran": any(s["scope"] == "multivariate" for s in res.extra["scopes"]),
            "bi_ran": any(s["scope"] != "multivariate" for s in res.extra["scopes"]),
            "n_eligible_columns": len(res.extra["eligible_columns"]), "n_pairs_tested": len(res.extra["pairs_tested"]),
            "clean_flag_rate_any": res.extra["n_flagged_any"] / n,
            "clean_flag_rate_multi": rec.row_id[rec.scope == "multivariate"].nunique() / n,
            "clean_flag_rate_bi": rec.row_id[rec.scope != "multivariate"].nunique() / n,
            "clean_flag_rate_joint_only": res.extra["n_joint_only"] / n,
            "declined": json.dumps(res.extra["declined"]), "skipped_columns": json.dumps(res.extra["skipped_columns"])}


# --------------------------------------------------------------------------------------------------------- criteria
def evaluate(runs: pd.DataFrame, clean: pd.DataFrame) -> dict:
    """The spec's section D, applied mechanically. Returns every number the verdict rests on."""
    out = {}
    ev = clean[clean.group != "gating"]
    ran_any = ev[ev.multi_ran | ev.bi_ran]

    # F1: clean false-positive rate above 1% of rows, per scope, per dataset
    f1 = ran_any.groupby("dataset").agg(multi=("clean_flag_rate_multi", "mean"), multi_max=("clean_flag_rate_multi", "max"),
                                         bi=("clean_flag_rate_bi", "mean"), bi_max=("clean_flag_rate_bi", "max"),
                                         any=("clean_flag_rate_any", "mean"), group=("group", "first"))
    f1["multi_fail"] = f1.multi_max > 0.01
    f1["bi_fail"] = f1.bi_max > 0.01
    out["F1_clean_false_positive"] = {"fails_multivariate": f1.index[f1.multi_fail].tolist(),
                                      "fails_bivariate": f1.index[f1.bi_fail].tolist(), "table": f1.round(5).reset_index().to_dict("records"),
                                      "passed": not (f1.multi_fail.any() or f1.bi_fail.any())}

    syn = runs[runs.dataset.str.startswith("synthetic")]
    g = syn[syn.dataset.str.contains("rho0.8") & syn.rate.isin([0.02, 0.05])].groupby(["dataset", "rate"]).apply(
        lambda d: pd.Series({"recall_invisible_swaps": d.hits_invisible_swaps.sum() / max(d.n_invisible_swaps.sum(), 1),
                             "recall_invisible_swaps_per_seed_min": (d.hits_invisible_swaps / d.n_invisible_swaps.clip(lower=1)).min()}),
        include_groups=False)
    out["F2_recall_invisible_swaps_rho0.8"] = {"table": g.round(4).reset_index().to_dict("records"),
                                               "passed": bool((g.recall_invisible_swaps >= 0.5).all())}

    s3 = syn[~syn.dataset.str.contains("rho0.0") & (syn.rate <= 0.05)]
    p = s3.groupby(["dataset", "rate"]).agg(precision=("precision", "mean"), n_seeds_with_flags=("precision", "count"),
                                            n_seeds=("seed", "nunique"))
    out["F3_precision_synthetic_rho>=0.5"] = {"table": p.round(4).reset_index().to_dict("records"),
                                              "passed": bool((p.precision >= 0.8).all()),
                                              "note": "NaN precision (nothing flagged) is excluded from the mean"}

    a = syn[~syn.dataset.str.contains("rho0.0")]
    attr = {tag: {"correct": int(a[f"attr_{tag}_correct"].sum()), "flagged_swap_one_rows": int(a[f"attr_{tag}_n"].sum())} for tag in ("multi", "bi")}
    for tag in attr:
        attr[tag]["share"] = attr[tag]["correct"] / attr[tag]["flagged_swap_one_rows"] if attr[tag]["flagged_swap_one_rows"] else None
    out["F4_attribution_synthetic_rho>=0.5"] = {**attr, "passed": all(v["share"] is not None and v["share"] >= 0.7 for v in attr.values())}

    ad = clean[clean.dataset == "adult"]
    wi = clean[clean.dataset == "wine"]
    out["F5_gating"] = {
        "adult_bivariate_declined": bool((~ad.bi_ran).all()), "adult_multivariate_clean_flag_rate_max": float(ad.clean_flag_rate_multi.max()),
        "wine_both_declined": bool(((~wi.multi_ran) & (~wi.bi_ran)).all()),
        "passed": bool((~ad.bi_ran).all() and ad.clean_flag_rate_multi.max() <= 0.01 and ((~wi.multi_ran) & (~wi.bi_ran)).all())}

    cells = runs[runs.group != "gating"].groupby(["dataset", "rate"]).agg(
        main=("recall", "mean"), iforest=("if_recall", "mean"), ran=("scopes_ran", "max")).reset_index()
    cells = cells[cells.ran > 0]
    beats = cells.iforest > cells.main
    out["F6_isolation_forest_beats_detector"] = {"share_of_cells": float(beats.mean()) if len(cells) else None, "n_cells": int(len(cells)),
                                                 "passed": bool(len(cells) == 0 or beats.mean() < 0.6)}

    multi_fail = set(f1.index[f1.multi_fail]) | ({"adult"} if out["F5_gating"]["adult_multivariate_clean_flag_rate_max"] > 0.01 else set())
    out["consequence"] = {
        "multivariate_datasets_failing_F1_or_F5": sorted(multi_fail),
        "multivariate_ships_off_by_default": len(multi_fail) >= 2,
        "bivariate_F1_failures": f1.index[f1.bi_fail].tolist(),
        "F2_failed_so_detector_cannot_find_marginally_invisible_anomalies": not out["F2_recall_invisible_swaps_rho0.8"]["passed"]}
    return out


def summarise(runs: pd.DataFrame) -> pd.DataFrame:
    def agg(d):
        sw = d.n_invisible_swaps.sum()
        return pd.Series({
            "n_seeds": d.seed.nunique(), "scopes_ran": d.scopes_ran.max(), "n_pairs_tested": d.n_pairs_tested.max(),
            "precision": d.precision.mean(), "recall": d.recall.mean(), "recall_std": d.recall.std(),
            "recall_multi": d.recall_multi.mean(), "recall_bi": d.recall_bi.mean(),
            "recall_swap_one": d.hits_swap_one.sum() / max(d.n_swap_one.sum(), 1),
            "recall_swap_block": d.hits_swap_block.sum() / max(d.n_swap_block.sum(), 1),
            "recall_inflate": d.hits_inflate.sum() / max(d.n_inflate.sum(), 1),
            "recall_invisible_swaps": d.hits_invisible_swaps.sum() / sw if sw else np.nan,
            "iqr_recall": d.iqr_recall.mean(), "iqr_recall_invisible_swaps": d.iqr_hits_invisible_swaps.sum() / sw if sw else np.nan,
            "iqr_recall_inflate": d.iqr_hits_inflate.sum() / max(d.n_inflate.sum(), 1),
            "if_recall_equal_budget": d.if_recall.mean(),
            "if_recall_invisible_swaps": d.if_hits_invisible_swaps.sum() / sw if sw else np.nan,
            "attr_multi_share": d.attr_multi_correct.sum() / d.attr_multi_n.sum() if d.attr_multi_n.sum() else np.nan,
            "attr_bi_share": d.attr_bi_correct.sum() / d.attr_bi_n.sum() if d.attr_bi_n.sum() else np.nan,
            "mean_flagged": d.n_flagged.mean(), "mean_joint_only": d.n_joint_only.mean()})
    return runs.groupby(["design", "group", "dataset", "rate"]).apply(agg, include_groups=False).reset_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", choices=list(DESIGNS), required=True)
    ap.add_argument("--full", action="store_true")
    a = ap.parse_args()
    ties = DESIGNS[a.design]
    seeds = SEEDS_FULL if a.full else [42]
    tag = "" if a.full else "_single_seed"
    log(f"design={a.design} (ties='{ties}')  seeds={seeds}  scikit-learn {sklearn.__version__}")

    runs, cleans = [], []
    for seed in seeds:
        for name, (X, group) in all_datasets(seed).items():
            X = X.reset_index(drop=True)
            cleans.append(run_clean(name, X, group, seed, ties))
            for rate in RATES:
                runs.append(run_one(name, X, group, rate, seed, ties, with_overlap=(name == "german_credit" and rate == 0.05)))
            r = runs[-1]
            log(f"seed={seed} {name:28s} ({group:8s}) clean-flag={cleans[-1]['clean_flag_rate_any']:.4f} "
                f"@10%: P={r['precision']:.2f} R={r['recall']:.2f} scopes={r['scopes_ran']}")
    runs, clean = pd.DataFrame(runs), pd.DataFrame(cleans)
    summary = summarise(runs)
    verdict = evaluate(runs, clean)
    verdict["design"], verdict["ties"], verdict["seeds"] = a.design, ties, seeds
    verdict["scikit_learn"] = sklearn.__version__
    for name, df in (("runs", runs), ("clean", clean), ("summary", summary)):
        df.to_csv(os.path.join(RESULTS_DIR, f"phase15_{a.design}_{name}{tag}.csv"), index=False)
    with open(os.path.join(RESULTS_DIR, f"phase15_{a.design}_criteria{tag}.json"), "w") as f:
        json.dump(verdict, f, indent=2, default=str)
    log("saved results")
    for k in ("F1_clean_false_positive", "F2_recall_invisible_swaps_rho0.8", "F3_precision_synthetic_rho>=0.5",
              "F4_attribution_synthetic_rho>=0.5", "F5_gating", "F6_isolation_forest_beats_detector"):
        log(f"{k}: {'PASS' if verdict[k]['passed'] else 'FAIL'}")
    log(f"consequence: {json.dumps(verdict['consequence'])}")


if __name__ == "__main__":
    main()
