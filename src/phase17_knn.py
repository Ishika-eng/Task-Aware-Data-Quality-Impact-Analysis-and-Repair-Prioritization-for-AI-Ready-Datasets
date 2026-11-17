"""
Phase 17: why does KNN imputation hurt gradient boosting? Method, hypotheses, criteria: docs/PHASE17_SPEC.md (written
before this module existed).

One variable per arm: how the blanked cells are filled before training. The corruption draw, the model, the test set and
the single KNN run are shared by every arm of a (rate, seed) cell, so arms are paired. The frozen detect/repair code is
called unchanged and compared against the composed frame (K0); this module never edits a frozen file.

Writes only results/phase17_*.

    .venv/bin/python src/phase17_knn.py --stage a            main GB arms, K0-K3, K5, mechanism table
    .venv/bin/python src/phase17_knn.py --stage a --smoke    one cell (rate 0.10, seed 42), prints K0 checks, writes nothing
    .venv/bin/python src/phase17_knn.py --stage b            secondary arms: shallow GB and Random Forest (K4, P6)
    .venv/bin/python src/phase17_knn.py --stage c            Bank Marketing: the guard rule, unchanged, out of sample (K6)
    .venv/bin/python src/phase17_knn.py --stage d            the shipped product code path, guard on vs off (amendment A3: F0-F2)
"""

from __future__ import annotations

import argparse
import json
import os
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer

import phase16_gb as g
from corruption import corrupt_with_id
from data import column_types
from phase9_repair_engine import detect_and_repair
from taskclean import POINT_MASS_SHARE        # the product's own constant: stage D validates exactly what ships

warnings.filterwarnings("ignore")

RESULTS_DIR = g.RESULTS_DIR
PATHS = {k: os.path.join(RESULTS_DIR, f"phase17_knn_{k}") for k in
         ("raw.csv", "summary.csv", "mechanism.csv", "checks.json", "criteria.json",
          "secondary_raw.csv", "secondary_summary.csv", "bank_raw.csv", "bank_summary.csv", "bank_checks.json",
          "product_adult_raw.csv", "product_bank_raw.csv", "product_checks.json")}

MISSING = "missing_values"
RATES = [0.05, 0.10, 0.20]
EVAL_RATES = [0.10, 0.20]                  # spec: the rates K1-K5 are evaluated at (5% is reported, not tested)
SEEDS = g.SEEDS
CAPITAL = ["capital-gain", "capital-loss"]
GUARD_MODE_SHARE = 0.5                     # spec: a numeric column is "point-mass" if its top value covers >= 50% of observed cells
K1_SHARE = 0.75                            # spec thresholds
K3_TOLERANCE = 0.003
K4_RATIO = 0.50
TOL = 1e-9
METRICS = ("posf1", "macrof1", "auc")

ARMS = ["dirty", "median_explicit", "knn_all", "knn_no_capital", "knn_capital_only", "knn_rounded", "knn_guarded"]
SINGLE = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
ALL_ARMS = ARMS + [f"knn_{c}" for c in SINGLE]
SECONDARY_ARMS = ["dirty", "knn_all", "knn_no_capital", "knn_capital_only"]


# ---------------------------------------------------------------------------------------------------------------
# Imputation pieces (the numeric KNN is a line-for-line copy of phase9_repair_engine's, verified against it by K0a)
# ---------------------------------------------------------------------------------------------------------------
def knn_values(X_dirty: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    """KNN-imputed numeric columns exactly as the frozen repair computes them: standardise ignoring NaN, KNNImputer,
    un-standardise. Returns a full numeric frame (observed cells unchanged)."""
    _, numeric = column_types(X_dirty)
    Xn = X_dirty[numeric].astype(float)
    means = Xn.mean()
    stds = Xn.std().replace(0, 1)
    scaled = (Xn - means) / stds
    imputed = pd.DataFrame(KNNImputer(n_neighbors=k).fit_transform(scaled), index=Xn.index, columns=numeric)
    return imputed * stds + means


def point_mass_columns(X: pd.DataFrame, threshold: float = GUARD_MODE_SHARE) -> list[str]:
    """The guard rule (stated in the spec, not a list of names): numeric columns whose most frequent observed value covers
    at least `threshold` of the observed cells. KNN averages are not meaningful for these."""
    _, numeric = column_types(X)
    out = []
    for c in numeric:
        obs = X[c].dropna()
        if len(obs) and obs.value_counts(normalize=True).iloc[0] >= threshold:
            out.append(c)
    return out


def fill_frame(X_dirty: pd.DataFrame, imputed: pd.DataFrame, knn_cols, round_cols=(), explicit: bool = False) -> pd.DataFrame:
    """Blanked numeric cells in `knn_cols` take the KNN value (rounded for `round_cols`). Categorical cells take the column
    mode (what the frozen repair and the pipeline both do). Other numeric cells stay NaN, so the pipeline's median fills
    them, or, with explicit=True, are filled here with the dirty column's median (the control)."""
    categorical, numeric = column_types(X_dirty)
    X = X_dirty.copy()
    X[numeric] = X[numeric].astype(float)
    for c in numeric:
        mask = X_dirty[c].isna()
        if not mask.any():
            continue
        if c in knn_cols:
            vals = imputed.loc[mask, c]
            X.loc[mask, c] = vals.round() if c in round_cols else vals
        elif explicit:
            X.loc[mask, c] = X_dirty[c].median()
    for c in categorical:
        mask = X_dirty[c].isna()
        if mask.any():
            mode = X_dirty[c].mode(dropna=True)
            X.loc[mask, c] = mode.iloc[0] if len(mode) else "missing"
    return X


def build_arms(X_dirty: pd.DataFrame, imputed: pd.DataFrame) -> tuple[dict, dict]:
    """{arm: training frame} for one corruption draw, plus meta (the columns the guard routes to the median)."""
    _, numeric = column_types(X_dirty)
    guard = point_mass_columns(X_dirty)
    knn_guarded = [c for c in numeric if c not in guard]
    int_cols = [c for c in numeric if (X_dirty[c].dropna() % 1 == 0).all()]
    frames = {
        "dirty": X_dirty.copy(),
        "median_explicit": fill_frame(X_dirty, imputed, [], explicit=True),
        "knn_all": fill_frame(X_dirty, imputed, numeric),
        "knn_no_capital": fill_frame(X_dirty, imputed, [c for c in numeric if c not in CAPITAL]),
        "knn_capital_only": fill_frame(X_dirty, imputed, [c for c in CAPITAL if c in numeric]),
        "knn_rounded": fill_frame(X_dirty, imputed, numeric, round_cols=int_cols),
        "knn_guarded": fill_frame(X_dirty, imputed, knn_guarded),
    }
    for c in numeric:
        frames[f"knn_{c}"] = fill_frame(X_dirty, imputed, [c])
    return frames, {"guard_cols": guard, "int_cols": int_cols}


# ---------------------------------------------------------------------------------------------------------------
# Mechanism table: what each imputation does to the blanked cells, with no classifier involved
# ---------------------------------------------------------------------------------------------------------------
def mechanism_rows(X_true: pd.DataFrame, X_dirty: pd.DataFrame, imputed: pd.DataFrame, rate: float, seed: int) -> list[dict]:
    _, numeric = column_types(X_dirty)
    rows = []
    for c in numeric:
        mask = X_dirty[c].isna() & X_true[c].notna()
        if not mask.any():
            continue
        truth = X_true.loc[mask, c].astype(float)
        knn = imputed.loc[mask, c]
        med = float(X_dirty[c].median())
        observed = set(X_dirty[c].dropna().astype(float).unique())
        rows.append({"rate": rate, "seed": seed, "column": c, "n_blanked": int(mask.sum()),
                     "median": med, "truth_mean": float(truth.mean()), "knn_mean": float(knn.mean()),
                     "knn_exact_share": float((np.abs(knn - truth) < 1e-9).mean()),
                     "median_exact_share": float((np.abs(med - truth) < 1e-9).mean()),
                     "knn_mae": float(np.abs(knn - truth).mean()), "median_mae": float(np.abs(med - truth).mean()),
                     "knn_novel_share": float((~knn.round(9).isin(observed)).mean())})
    return rows


# ---------------------------------------------------------------------------------------------------------------
# Scoring one cell
# ---------------------------------------------------------------------------------------------------------------
def score_arms(factory, frames: dict, y: pd.Series, X_te, y_te, seed: int, arms) -> dict:
    out = {}
    for arm in arms:
        pipe = factory(frames[arm], seed)
        pipe.fit(frames[arm], y)
        out[arm] = g.scores(pipe, X_te, y_te)
    return out


def k0_cell_checks(frames: dict, X_rep_frozen: pd.DataFrame, X_dirty: pd.DataFrame, meta: dict) -> dict:
    """K0a (composed knn_all frame == the frozen repair's frame) and the guard selection for one cell."""
    _, numeric = column_types(X_dirty)
    num_diff = float(np.abs(frames["knn_all"][numeric].values - X_rep_frozen[numeric].astype(float).values).max())
    cats = [c for c in X_dirty.columns if c not in numeric]
    cat_equal = bool(all((frames["knn_all"][c].astype(str).values == X_rep_frozen[c].astype(str).values).all() for c in cats))
    return {"max_abs_numeric_diff": num_diff, "categorical_equal": cat_equal,
            "guard_cols": sorted(meta["guard_cols"]), "guard_ok": sorted(meta["guard_cols"]) == sorted(CAPITAL)}


def phase16_recorded() -> pd.DataFrame:
    raw = pd.read_csv(g.PATHS["repair_raw.csv"])
    return raw[raw.error_type == MISSING].set_index(["rate", "seed"])


# ---------------------------------------------------------------------------------------------------------------
# Criteria (pure functions of the summary so they can be tested on synthetic numbers)
# ---------------------------------------------------------------------------------------------------------------
def recovery_table(summary: pd.DataFrame, metric: str = "posf1") -> dict:
    """{(arm, rate): mean recovery vs dirty} from a summary with columns arm, rate, mean_recovery_<metric>."""
    col = f"mean_recovery_{metric}"
    return {(r.arm, round(float(r.rate), 4)): float(getattr(r, col)) for r in summary.itertuples()}


def share_removed(rec: dict, arm: str, rate: float, ref: str = "knn_all", floor: float = g.G0_TOLERANCE):
    """Fraction of the reference arm's harm that `arm` removes: (rec[arm]-rec[ref]) / (0-rec[ref]). None if the reference
    arm shows no harm to remove (recovery >= 0)."""
    harm = -rec[(ref, rate)]
    if harm <= floor:
        return None
    return (rec[(arm, rate)] - rec[(ref, rate)]) / harm


def share_kept(rec: dict, arm: str, rate: float, ref: str = "knn_all", floor: float = g.G0_TOLERANCE):
    """Fraction of the reference arm's harm that `arm` still has: rec[arm] / rec[ref]."""
    if -rec[(ref, rate)] <= floor:
        return None
    return rec[(arm, rate)] / rec[(ref, rate)]


def criterion_k1(rec: dict, rates=EVAL_RATES) -> dict:
    removed = {r: share_removed(rec, "knn_no_capital", r) for r in rates}
    kept = {r: share_kept(rec, "knn_capital_only", r) for r in rates}
    if any(v is None for v in list(removed.values()) + list(kept.values())):
        return {"verdict": "not evaluable", "removed": removed, "kept": kept}
    a = all(v >= K1_SHARE for v in removed.values())
    b = all(v >= K1_SHARE for v in kept.values())
    return {"verdict": "pass" if a and b else ("partly" if a or b else "H1 rejected"),
            "removed_by_no_capital": removed, "kept_by_capital_only": kept}


def criterion_k2(rec: dict, rates=EVAL_RATES) -> dict:
    harms = {r: {c: -rec[(f"knn_{c}", r)] for c in SINGLE} for r in rates}
    top = {r: max(h, key=h.get) for r, h in harms.items()}
    return {"pass": all(t == "capital-gain" for t in top.values()), "largest": top, "harms": harms}


def criterion_k3(rec: dict, rates=EVAL_RATES) -> dict:
    d = {r: abs(rec[("knn_rounded", r)] - rec[("knn_all", r)]) for r in rates}
    return {"H2_rejected": all(v < K3_TOLERANCE for v in d.values()), "abs_difference": d}


def criterion_k4(rec_deep: dict, rec_shallow: dict, rates=EVAL_RATES) -> dict:
    ratio = {}
    for r in rates:
        deep, shallow = -rec_deep[("knn_all", r)], -rec_shallow[("knn_all", r)]
        ratio[r] = None if deep <= g.G0_TOLERANCE else shallow / deep
    if any(v is None for v in ratio.values()):
        return {"verdict": "not evaluable", "shallow_over_deep": ratio}
    return {"verdict": "pass" if all(v <= K4_RATIO for v in ratio.values()) else "fail", "shallow_over_deep": ratio}


def criterion_k5(rec: dict, rates=EVAL_RATES) -> dict:
    removed = {r: share_removed(rec, "knn_guarded", r) for r in rates}
    if any(v is None for v in removed.values()):
        return {"verdict": "not evaluable", "removed": removed}
    return {"verdict": "pass" if all(v >= K1_SHARE for v in removed.values()) else "fail", "removed": removed}


def k0_verdict(cells: list[dict], recorded_diffs: dict) -> dict:
    """K0 over all cells. `recorded_diffs` = max |F1 difference| for the three score checks."""
    a = all(c["max_abs_numeric_diff"] <= TOL and c["categorical_equal"] for c in cells)
    d = all(c["guard_ok"] for c in cells)
    scores_ok = all(v <= TOL for v in recorded_diffs.values())
    return {"pass": bool(a and d and scores_ok), "K0a_frame_equal": bool(a), "K0d_guard_selects_capital": bool(d),
            "K0bc_scores_equal": bool(scores_ok), "n_cells": len(cells), "score_max_abs_diff": recorded_diffs,
            "max_numeric_frame_diff": max(c["max_abs_numeric_diff"] for c in cells)}


# ---------------------------------------------------------------------------------------------------------------
# Stage A: main GB arms
# ---------------------------------------------------------------------------------------------------------------
def _summarise(raw: pd.DataFrame) -> pd.DataFrame:
    return raw.groupby(["rate", "arm"], sort=False).agg(
        **{f"mean_{m}": (m, "mean") for m in METRICS},
        **{f"mean_recovery_{m}": (f"recovery_{m}", "mean") for m in METRICS},
        std_recovery_posf1=("recovery_posf1", "std"), n_seeds=("seed", "nunique")).reset_index()


def stage_a(smoke: bool = False):
    config = g.require_config()
    params = g.selected_params(config)
    X_tr, _, X_te, y_tr, _, y_te = g.load_split()
    factory = g.gb_factory(params)
    rec16 = phase16_recorded()
    cells = [(0.10, 42)] if smoke else [(r, s) for r in RATES for s in SEEDS]
    print(f"Phase 17 stage A: GB {params} | {len(cells)} cells x {len(ALL_ARMS)} arms | spec docs/PHASE17_SPEC.md", flush=True)

    partial = PATHS["raw.csv"] + ".partial"
    rows = pd.read_csv(partial).to_dict("records") if (os.path.exists(partial) and not smoke) else []
    done = {(round(float(r["rate"]), 4), int(r["seed"])) for r in rows}
    k0_cells, mech, diffs = [], [], {"dirty_vs_phase16": 0.0, "knn_all_vs_phase16": 0.0, "median_explicit_vs_dirty": 0.0}
    cache = PATHS["checks.json"] + ".cells.json"
    if os.path.exists(cache) and not smoke:
        saved = json.load(open(cache))
        k0_cells, mech, diffs = saved["k0_cells"], saved["mech"], saved["diffs"]
    t0 = time.monotonic()
    for i, (rate, seed) in enumerate(cells, 1):
        if (round(rate, 4), seed) in done:
            continue
        tc = time.monotonic()
        X_dirty, y_dirty, _ = corrupt_with_id(MISSING, X_tr, y_tr, rate=rate, seed=seed)
        X_frozen, _, _, _ = detect_and_repair(MISSING, X_dirty, y_dirty, seed=seed)
        imputed = knn_values(X_dirty)
        frames, meta = build_arms(X_dirty, imputed)
        k0 = k0_cell_checks(frames, X_frozen, X_dirty, meta)
        sc = score_arms(factory, frames, y_dirty, X_te, y_te, seed, ALL_ARMS)
        for arm in ALL_ARMS:
            row = {"rate": rate, "seed": seed, "arm": arm, **sc[arm]}
            for m in METRICS:
                row[f"recovery_{m}"] = sc[arm][m] - sc["dirty"][m]
            rows.append(row)
        k0_cells.append({"rate": rate, "seed": seed, **k0})
        mech += mechanism_rows(X_tr, X_dirty, imputed, rate, seed)
        if (rate, seed) in rec16.index:
            r16 = rec16.loc[(rate, seed)]
            diffs["dirty_vs_phase16"] = max(diffs["dirty_vs_phase16"], abs(sc["dirty"]["posf1"] - float(r16["dirty_posf1"])))
            diffs["knn_all_vs_phase16"] = max(diffs["knn_all_vs_phase16"], abs(sc["knn_all"]["posf1"] - float(r16["repaired_posf1"])))
        diffs["median_explicit_vs_dirty"] = max(diffs["median_explicit_vs_dirty"],
                                                abs(sc["median_explicit"]["posf1"] - sc["dirty"]["posf1"]))
        print(f"[{i:2d}/{len(cells)}] rate {rate:.0%} seed {seed} ({time.monotonic() - tc:.0f}s) | K0a numeric diff "
              f"{k0['max_abs_numeric_diff']:.1e} cat_equal {k0['categorical_equal']} guard {k0['guard_cols']} | "
              f"knn_all {sc['knn_all']['posf1'] - sc['dirty']['posf1']:+.4f} no_capital "
              f"{sc['knn_no_capital']['posf1'] - sc['dirty']['posf1']:+.4f} capital_only "
              f"{sc['knn_capital_only']['posf1'] - sc['dirty']['posf1']:+.4f} rounded "
              f"{sc['knn_rounded']['posf1'] - sc['dirty']['posf1']:+.4f}", flush=True)
        if not smoke:
            pd.DataFrame(rows).to_csv(partial, index=False)
            json.dump({"k0_cells": k0_cells, "mech": mech, "diffs": diffs}, open(cache, "w"), default=float)
    if smoke:
        print("smoke diffs vs Phase 16:", {k: f"{v:.1e}" for k, v in diffs.items()})
        return None
    raw = pd.DataFrame(rows)
    raw.to_csv(PATHS["raw.csv"], index=False)
    summary = _summarise(raw)
    summary.to_csv(PATHS["summary.csv"], index=False)
    pd.DataFrame(mech).to_csv(PATHS["mechanism.csv"], index=False)
    checks = k0_verdict(k0_cells, diffs)
    json.dump(checks, open(PATHS["checks.json"], "w"), indent=1, default=float)
    for p in (partial, cache):
        if os.path.exists(p):
            os.remove(p)
    print(f"\nK0: {json.dumps(checks, indent=1, default=float)}")
    if not checks["pass"]:
        print("K0 FAILED: no criterion below is to be believed until the harness is fixed", flush=True)
    crit = write_criteria()
    print(json.dumps(crit, indent=1, default=float))
    print(f"({time.monotonic() - t0:.0f}s) Saved phase17_knn_raw.csv, _summary.csv, _mechanism.csv, _checks.json, _criteria.json")
    return summary


def write_criteria() -> dict:
    """K1-K3 and K5 from the main summary; K4 from the secondary summary if present. Only evaluated if K0 passed."""
    checks = json.load(open(PATHS["checks.json"]))
    out = {"K0": checks}
    if not checks["pass"]:
        return out
    rec = recovery_table(pd.read_csv(PATHS["summary.csv"]))
    out.update(K1=criterion_k1(rec), K2=criterion_k2(rec), K3=criterion_k3(rec), K5=criterion_k5(rec))
    if os.path.exists(PATHS["secondary_summary.csv"]):
        sec = pd.read_csv(PATHS["secondary_summary.csv"])
        shallow = recovery_table(sec[sec.model == "gb_shallow"])
        out["K4"] = criterion_k4(rec, shallow)
    json.dump(out, open(PATHS["criteria.json"], "w"), indent=1, default=float)
    return out


# ---------------------------------------------------------------------------------------------------------------
# Stage B: secondary arms under a shallow booster and the Random Forest (K4, P6)
# ---------------------------------------------------------------------------------------------------------------
def stage_b():
    g.require_config()
    if not os.path.exists(PATHS["checks.json"]) or not json.load(open(PATHS["checks.json"]))["pass"]:
        raise SystemExit("run stage A first: stage B is only meaningful after K0 passed")
    X_tr, _, X_te, y_tr, _, y_te = g.load_split()
    models = {"gb_shallow": g.gb_factory(g.CAPACITY_ARMS["shallow"]), "random_forest": g.rf_factory}
    cells = [(r, s) for r in EVAL_RATES for s in SEEDS]
    partial = PATHS["secondary_raw.csv"] + ".partial"
    rows = pd.read_csv(partial).to_dict("records") if os.path.exists(partial) else []
    done = {(round(float(r["rate"]), 4), int(r["seed"])) for r in rows}
    print(f"Phase 17 stage B: {list(models)} | {len(cells)} cells x {len(SECONDARY_ARMS)} arms each", flush=True)
    t0 = time.monotonic()
    for i, (rate, seed) in enumerate(cells, 1):
        if (round(rate, 4), seed) in done:
            continue
        tc = time.monotonic()
        X_dirty, y_dirty, _ = corrupt_with_id(MISSING, X_tr, y_tr, rate=rate, seed=seed)
        frames, _ = build_arms(X_dirty, knn_values(X_dirty))
        msg = []
        for name, factory in models.items():
            sc = score_arms(factory, frames, y_dirty, X_te, y_te, seed, SECONDARY_ARMS)
            for arm in SECONDARY_ARMS:
                row = {"model": name, "rate": rate, "seed": seed, "arm": arm, **sc[arm]}
                for m in METRICS:
                    row[f"recovery_{m}"] = sc[arm][m] - sc["dirty"][m]
                rows.append(row)
            msg.append(f"{name}: knn_all {sc['knn_all']['posf1'] - sc['dirty']['posf1']:+.4f} no_capital "
                       f"{sc['knn_no_capital']['posf1'] - sc['dirty']['posf1']:+.4f}")
        print(f"[{i:2d}/{len(cells)}] rate {rate:.0%} seed {seed} ({time.monotonic() - tc:.0f}s) | " + " | ".join(msg), flush=True)
        pd.DataFrame(rows).to_csv(partial, index=False)
    raw = pd.DataFrame(rows)
    raw.to_csv(PATHS["secondary_raw.csv"], index=False)
    summ = raw.groupby(["model", "rate", "arm"], sort=False).agg(
        **{f"mean_recovery_{m}": (f"recovery_{m}", "mean") for m in METRICS},
        std_recovery_posf1=("recovery_posf1", "std"), n_seeds=("seed", "nunique")).reset_index()
    summ.to_csv(PATHS["secondary_summary.csv"], index=False)
    if os.path.exists(partial):
        os.remove(partial)
    crit = write_criteria()
    print(f"\nK4: {json.dumps(crit.get('K4'), indent=1, default=float)}")
    print(summ.round(4).to_string(index=False))
    print(f"({time.monotonic() - t0:.0f}s) Saved phase17_knn_secondary_raw.csv, _secondary_summary.csv")


# ---------------------------------------------------------------------------------------------------------------
# Stage C: Bank Marketing, the guard rule out of sample (K6; clarifications in spec section K-bis)
# ---------------------------------------------------------------------------------------------------------------
BANK_ARMS = ["dirty", "median_explicit", "knn_all", "knn_guarded", "knn_pointmass_only"]


def criterion_k6(rec: dict, floor: float, rates=EVAL_RATES) -> dict:
    """Spec K-bis: evaluable at a rate if knn_all harm exceeds the Bank noise floor there. Pass = at least one evaluable rate
    and the guard removes >= 75% of the harm at every evaluable rate; not evaluable if no rate is; fail otherwise."""
    harm = {r: -rec[("knn_all", r)] for r in rates}
    evaluable = [r for r in rates if harm[r] > floor]
    removed = {r: share_removed(rec, "knn_guarded", r) for r in evaluable}
    if not evaluable:
        return {"verdict": "not evaluable", "knn_all_harm": harm, "noise_floor": floor, "evaluable_rates": []}
    ok = all(v is not None and v >= K1_SHARE for v in removed.values())
    return {"verdict": "pass" if ok else "fail", "knn_all_harm": harm, "noise_floor": floor,
            "evaluable_rates": evaluable, "removed_by_guard": removed}


def stage_c():
    g.require_config()
    if not os.path.exists(PATHS["criteria.json"]):
        raise SystemExit("run stage A first")
    crit = json.load(open(PATHS["criteria.json"]))
    if not (crit["K0"]["pass"] and crit["K1"]["verdict"] == "pass" and crit["K5"]["verdict"] == "pass"):
        raise SystemExit("spec: K6 is only run if K0, K1 and K5 passed")
    import phase13_transfer as p13
    from data import three_way_split
    params = g.selected_params(g.require_config())
    meta16 = json.load(open(g.PATHS["bank_meta.json"]))
    if not meta16.get("harness_check_pass"):
        raise SystemExit("Phase 16 Stage 6 harness check did not pass; the Bank scorer cannot be trusted")
    floor = float(meta16["gb_noise_floor"]["macrof1"])
    df, target = p13.load_raw(g.BANK_SLUG)
    X, y = p13.clean_dataset(df, target, p13.CAP_ROWS)
    pos = y.value_counts().idxmin()
    X_tr, _, X_te, y_tr, _, y_te = three_way_split(X, y, seed=42)
    factory = g.gb_factory(params)
    runs16 = pd.read_csv(g.PATHS["bank_runs.csv"])
    runs16 = runs16[runs16.issue == MISSING].set_index(["rate", "seed"])
    cells = [(r, s) for r in p13.RATES for s in p13.SEEDS]
    print(f"Phase 17 stage C: Bank Marketing {len(X):,} rows (train {len(X_tr):,}), positive {pos!r}, GB {params}, "
          f"macro-F1 noise floor {floor:.4f} | {len(cells)} cells", flush=True)
    rows, k0_cells, guard_seen = [], [], set()
    diffs = {"dirty_vs_phase16": 0.0, "knn_all_vs_phase16": 0.0, "median_explicit_vs_dirty": 0.0}
    t0 = time.monotonic()
    for i, (rate, seed) in enumerate(cells, 1):
        tc = time.monotonic()
        X_d, y_d, _ = p13.CORRUPTORS[MISSING](X_tr, y_tr, rate=rate, seed=seed)
        X_frozen, _, _, _ = p13.detect_and_repair(MISSING, X_d, y_d, seed)
        imputed = knn_values(X_d)
        frames, meta = build_arms(X_d, imputed)
        frames["knn_pointmass_only"] = fill_frame(X_d, imputed, meta["guard_cols"])
        _, numeric = column_types(X_d)
        guard_seen.add(tuple(sorted(meta["guard_cols"])))
        num_diff = float(np.abs(frames["knn_all"][numeric].values - X_frozen[numeric].astype(float).values).max())
        cats = [c for c in X_d.columns if c not in numeric]
        cat_eq = bool(all((frames["knn_all"][c].astype(str).values == X_frozen[c].astype(str).values).all() for c in cats))
        k0_cells.append({"rate": rate, "seed": seed, "max_abs_numeric_diff": num_diff, "categorical_equal": cat_eq,
                         "guard_cols": sorted(meta["guard_cols"]), "guard_ok": True})   # no prediction for Bank's guard set
        sc = {}
        for arm in BANK_ARMS:
            sc[arm] = g.scores(factory(frames[arm], seed).fit(frames[arm], y_d), X_te, y_te, pos)
            row = {"rate": rate, "seed": seed, "arm": arm, **sc[arm]}
            rows.append(row)
        for r in rows[-len(BANK_ARMS):]:
            for m in METRICS:
                r[f"recovery_{m}"] = r[m] - sc["dirty"][m]
        r16 = runs16.loc[(rate, seed)]
        diffs["dirty_vs_phase16"] = max(diffs["dirty_vs_phase16"], abs(sc["dirty"]["macrof1"] - float(r16["dirty_macrof1"])))
        diffs["knn_all_vs_phase16"] = max(diffs["knn_all_vs_phase16"], abs(sc["knn_all"]["macrof1"] - float(r16["repaired_macrof1"])))
        diffs["median_explicit_vs_dirty"] = max(diffs["median_explicit_vs_dirty"], abs(sc["median_explicit"]["macrof1"] - sc["dirty"]["macrof1"]))
        print(f"[{i:2d}/{len(cells)}] rate {rate:.0%} seed {seed} ({time.monotonic() - tc:.0f}s) | guard {sorted(meta['guard_cols'])} | "
              f"macro-F1 recovery: knn_all {sc['knn_all']['macrof1'] - sc['dirty']['macrof1']:+.4f} guarded "
              f"{sc['knn_guarded']['macrof1'] - sc['dirty']['macrof1']:+.4f} pointmass_only "
              f"{sc['knn_pointmass_only']['macrof1'] - sc['dirty']['macrof1']:+.4f}", flush=True)
    raw = pd.DataFrame(rows)
    raw.to_csv(PATHS["bank_raw.csv"], index=False)
    summ = _summarise(raw)
    summ.to_csv(PATHS["bank_summary.csv"], index=False)
    checks = k0_verdict([{**c, "guard_ok": True} for c in k0_cells], diffs)
    checks["guard_columns_seen"] = [list(t) for t in sorted(guard_seen)]
    out = {"K0_bank": checks}
    if checks["pass"]:
        out["K6"] = criterion_k6(recovery_table(summ, "macrof1"), floor)
        out["K6_positive_f1_report"] = criterion_k6(recovery_table(summ, "posf1"), floor)   # reported only; macro-F1 is the test
        out["K6_auc_report"] = criterion_k6(recovery_table(summ, "auc"), floor)
    json.dump(out, open(PATHS["bank_checks.json"], "w"), indent=1, default=float)
    print(json.dumps(out, indent=1, default=float))
    print(summ.pivot(index="arm", columns="rate", values="mean_recovery_macrof1").round(4).to_string())
    print(f"({time.monotonic() - t0:.0f}s) Saved phase17_knn_bank_raw.csv, _bank_summary.csv, _bank_checks.json")


# ---------------------------------------------------------------------------------------------------------------
# Stage D: the shipped product code path (spec amendment A3): taskclean._propose_missing with the guard on and off
# ---------------------------------------------------------------------------------------------------------------
F1_MIN_RECOVERY = -0.005
F1_SHARE = 0.75


def product_frame(X_dirty: pd.DataFrame, share: float) -> pd.DataFrame:
    """The training frame after applying the product's own missing-value proposals (`share` > 1 turns the guard off)."""
    import taskclean as tc
    _, numeric = column_types(X_dirty)
    integer_like = {c: bool((X_dirty[c].dropna() % 1 == 0).all()) for c in numeric}
    frames = tc._propose_missing(X_dirty, integer_like, X_dirty.isna(), X_dirty, point_mass_share=share)
    X = X_dirty.copy()
    X[numeric] = X[numeric].astype(float)
    for f in frames:
        for col, grp in f.groupby("column"):
            vals = grp["proposed_value"]
            X.loc[grp["row_id"].values, col] = vals.astype(float).values if col in numeric else vals.values
    return X


def criterion_f1(rec_new: dict, rec_old: dict, rates=EVAL_RATES) -> dict:
    """Spec A3: mean recovery >= -0.005 at both rates AND >= 75% of the old product's harm removed at both rates."""
    out = {"recovery_new": {r: rec_new[r] for r in rates}, "recovery_old": {r: rec_old[r] for r in rates}}
    removed = {r: (rec_new[r] - rec_old[r]) / -rec_old[r] if rec_old[r] < -g.G0_TOLERANCE else None for r in rates}
    out["removed"] = removed
    if any(v is None for v in removed.values()):
        out["verdict"] = "not evaluable"
    else:
        out["verdict"] = "pass" if all(rec_new[r] >= F1_MIN_RECOVERY for r in rates) and all(v >= F1_SHARE for v in removed.values()) else "fail"
    return out


def criterion_f2(rec_new: dict, rec_old: dict, floor: float, rates=EVAL_RATES) -> dict:
    """Spec A3, report only: a caution if the shipped repair is worse than the old by more than the dataset's noise floor."""
    worse = {r: rec_old[r] - rec_new[r] for r in rates}          # positive = shipped is worse
    return {"caution": any(v > floor for v in worse.values()), "shipped_minus_old": {r: -v for r, v in worse.items()},
            "noise_floor": floor}


def stage_d():
    g.require_config()
    if not os.path.exists(PATHS["criteria.json"]):
        raise SystemExit("run stage A first")
    params = g.selected_params(g.require_config())
    factory = g.gb_factory(params)
    import phase13_transfer as p13
    from data import three_way_split
    t0 = time.monotonic()
    # ---- Adult: F0 and F1
    X_tr, _, X_te, y_tr, _, y_te = g.load_split()
    rec17 = pd.read_csv(PATHS["raw.csv"]).set_index(["rate", "seed", "arm"])
    rows, f0 = [], {"old_vs_knn_rounded_max_f1_diff": 0.0, "dirty_vs_phase17_max_f1_diff": 0.0, "max_numeric_frame_diff": 0.0}
    print(f"Phase 17 stage D (Adult): shipped product path vs guard-off, GB {params}", flush=True)
    for rate in EVAL_RATES:
        for seed in SEEDS:
            tc = time.monotonic()
            X_d, y_d, _ = corrupt_with_id(MISSING, X_tr, y_tr, rate=rate, seed=seed)
            new, old = product_frame(X_d, POINT_MASS_SHARE), product_frame(X_d, 1.01)
            frames, meta = build_arms(X_d, knn_values(X_d))
            _, numeric = column_types(X_d)
            nonpm = [c for c in numeric if c not in meta["guard_cols"]]
            ref = fill_frame(X_d, knn_values(X_d), nonpm, round_cols=meta["int_cols"])      # pipeline median fills the rest
            gc = meta["guard_cols"]
            d_guard = float(np.abs(new[gc].values - X_d[gc].fillna(X_d[gc].median()).values).max())   # median on point-mass cols
            d_knn = float(np.abs(new[nonpm].values - ref[nonpm].values).max())                          # KNN (rounded) elsewhere
            f0["max_numeric_frame_diff"] = max(f0["max_numeric_frame_diff"], d_guard, d_knn)
            sc = {a: g.scores(factory(f, seed).fit(f, y_d), X_te, y_te) for a, f in (("dirty", X_d), ("old", old), ("new", new))}
            f0["old_vs_knn_rounded_max_f1_diff"] = max(f0["old_vs_knn_rounded_max_f1_diff"],
                                                       abs(sc["old"]["posf1"] - rec17.loc[(rate, seed, "knn_rounded"), "posf1"]))
            f0["dirty_vs_phase17_max_f1_diff"] = max(f0["dirty_vs_phase17_max_f1_diff"],
                                                     abs(sc["dirty"]["posf1"] - rec17.loc[(rate, seed, "dirty"), "posf1"]))
            for a in ("old", "new"):
                rows.append({"rate": rate, "seed": seed, "arm": a, **{f"recovery_{m}": sc[a][m] - sc["dirty"][m] for m in METRICS}})
            print(f"  [{rate:.0%} seed {seed}] ({time.monotonic() - tc:.0f}s) recovery old "
                  f"{sc['old']['posf1'] - sc['dirty']['posf1']:+.4f} shipped {sc['new']['posf1'] - sc['dirty']['posf1']:+.4f}", flush=True)
    adult = pd.DataFrame(rows)
    adult.to_csv(PATHS["product_adult_raw.csv"], index=False)
    m = adult.groupby(["arm", "rate"])["recovery_posf1"].mean()
    rec_new = {r: float(m[("new", r)]) for r in EVAL_RATES}
    rec_old = {r: float(m[("old", r)]) for r in EVAL_RATES}
    f0["pass"] = bool(f0["old_vs_knn_rounded_max_f1_diff"] <= TOL and f0["dirty_vs_phase17_max_f1_diff"] <= TOL
                      and f0["max_numeric_frame_diff"] <= TOL)
    out = {"F0": f0}
    out["F1"] = criterion_f1(rec_new, rec_old) if f0["pass"] else {"verdict": "not evaluated: F0 failed"}

    # ---- Bank: F2 (report only)
    meta16 = json.load(open(g.PATHS["bank_meta.json"]))
    floor = float(meta16["gb_noise_floor"]["macrof1"])
    df, target = p13.load_raw(g.BANK_SLUG)
    X, y = p13.clean_dataset(df, target, p13.CAP_ROWS)
    pos = y.value_counts().idxmin()
    Xb_tr, _, Xb_te, yb_tr, _, yb_te = three_way_split(X, y, seed=42)
    brows = []
    print("Phase 17 stage D (Bank, report only)", flush=True)
    for rate in EVAL_RATES:
        for seed in p13.SEEDS:
            X_d, y_d, _ = p13.CORRUPTORS[MISSING](Xb_tr, yb_tr, rate=rate, seed=seed)
            sc = {a: g.scores(factory(f, seed).fit(f, y_d), Xb_te, yb_te, pos) for a, f in
                  (("dirty", X_d), ("old", product_frame(X_d, 1.01)), ("new", product_frame(X_d, POINT_MASS_SHARE)))}
            for a in ("old", "new"):
                brows.append({"rate": rate, "seed": seed, "arm": a, **{f"recovery_{mt}": sc[a][mt] - sc["dirty"][mt] for mt in METRICS}})
    bank = pd.DataFrame(brows)
    bank.to_csv(PATHS["product_bank_raw.csv"], index=False)
    mb = bank.groupby(["arm", "rate"])["recovery_macrof1"].mean()
    out["F2_bank_macrof1"] = criterion_f2({r: float(mb[("new", r)]) for r in EVAL_RATES},
                                          {r: float(mb[("old", r)]) for r in EVAL_RATES}, floor)
    out["F2_bank_macrof1"]["recovery_old"] = {r: float(mb[("old", r)]) for r in EVAL_RATES}
    out["F2_bank_macrof1"]["recovery_new"] = {r: float(mb[("new", r)]) for r in EVAL_RATES}
    mp = bank.groupby(["arm", "rate"])["recovery_posf1"].mean()
    out["F2_bank_posf1_report"] = {"recovery_old": {r: float(mp[("old", r)]) for r in EVAL_RATES},
                                   "recovery_new": {r: float(mp[("new", r)]) for r in EVAL_RATES}}
    json.dump(out, open(PATHS["product_checks.json"], "w"), indent=1, default=float)
    print(json.dumps(out, indent=1, default=float))
    print(f"({time.monotonic() - t0:.0f}s) Saved phase17_knn_product_*.csv, _product_checks.json")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["a", "b", "c", "d"], required=True)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(argv)
    if a.stage == "a":
        stage_a(smoke=a.smoke)
    elif a.stage == "b":
        stage_b()
    elif a.stage == "c":
        stage_c()
    else:
        stage_d()


if __name__ == "__main__":
    main()
