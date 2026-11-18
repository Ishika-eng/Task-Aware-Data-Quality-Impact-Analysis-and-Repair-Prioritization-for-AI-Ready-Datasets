"""
Phase 27 EXPLORATION (post-hoc, NOT pre-registered). Question: does a more aggressive noisy-label filter recover model performance that the
conservative label detector leaves on the table? Protocol that keeps the result honest without pre-registration: strategies and thresholds
are explored on SELECTION seeds only; the single chosen rule is then run, unchanged, on CONFIRMATION seeds and on a second dataset that the
choice never saw. Everything tried is written to the output, not only the winner.

  .venv/bin/python src/explore_phase27.py select      -> results/phase27_select.json
  .venv/bin/python src/explore_phase27.py confirm TAU -> results/phase27_confirm.json
"""
from __future__ import annotations
import json, os, sys, time
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict, train_test_split, StratifiedKFold

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path[:0] = [HERE, ROOT]
import controlled_proof as cp                                    # noqa: E402
from model_panel import MODELS, _fit_score, _typed               # noqa: E402

N_ROWS = 8000
SELECT_SEEDS, CONFIRM_SEEDS = (42, 43, 44), (100, 101, 102, 103, 104)
NOISE = (0.10, 0.20)
TAUS = (0.5, 0.4, 0.3, 0.2)


def load(name: str) -> tuple[pd.DataFrame, str]:
    if name == "adult":
        return cp.clean_frame(), cp.TARGET
    d = pd.read_csv(os.path.join(ROOT, "data", "bank_marketing_original.csv"), low_memory=False)
    d = d.sample(N_ROWS, random_state=7).reset_index(drop=True)
    return d, "Class"


def flip_labels(df: pd.DataFrame, target: str, rate: float, seed: int):
    rng = np.random.default_rng(seed)
    d = df.copy(); flip = rng.random(len(d)) < rate
    cls = sorted(d[target].astype(str).unique()); other = {cls[0]: cls[1], cls[1]: cls[0]}
    d[target] = d[target].astype(str)
    d.loc[flip, target] = d.loc[flip, target].map(other)
    return d, flip


def given_label_prob(d: pd.DataFrame, target: str, seed: int, drop_cols) -> np.ndarray:
    """Out-of-fold probability the model gives to each row's own (possibly wrong) label. Uses only the dirty training rows."""
    X, y = _typed(d, target, drop_cols)
    X = X.copy()
    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = X[c].astype("category").cat.codes.astype(float)
    X = X.fillna(-1)
    yy = y.astype(str).to_numpy()
    rf = RandomForestClassifier(n_estimators=200, min_samples_leaf=3, n_jobs=-1, random_state=seed)
    P = cross_val_predict(rf, X, yy, cv=StratifiedKFold(5, shuffle=True, random_state=seed), method="predict_proba")
    classes = np.array(sorted(set(yy)))
    return P[np.arange(len(yy)), np.searchsorted(classes, yy)]


def run(dataset: str, seeds, taus, noises, models) -> list[dict]:
    clean, target = load(dataset)
    rows = []
    for seed in seeds:
        tr_i, te_i = train_test_split(clean.index, test_size=0.2, random_state=seed, stratify=clean[target])
        clean_tr, clean_te = clean.loc[tr_i].reset_index(drop=True), clean.loc[te_i].reset_index(drop=True)
        clean_tr[target] = clean_tr[target].astype(str); clean_te[target] = clean_te[target].astype(str)
        for rate in noises:
            d, flip = flip_labels(clean_tr, target, rate, seed)
            drop_cols = []
            p = given_label_prob(d, target, seed, drop_cols)
            X_te, y_te = _typed(clean_te, target, drop_cols); ones = np.ones(len(y_te), dtype=bool)
            frames = {"clean": clean_tr, "dirty": d}
            for tau in taus:
                keep = p >= tau
                frames[f"drop_p<{tau}"] = d[keep].reset_index(drop=True)
                rows.append({"dataset": dataset, "seed": seed, "noise": rate, "tau": tau, "kind": "filter", "n_dropped": int((~keep).sum()),
                             "precision": float(flip[~keep].mean()) if (~keep).any() else None, "recall": float(flip[~keep].sum() / max(flip.sum(), 1))})
            for arm, fr in frames.items():
                X_tr, y_tr = _typed(fr, target, drop_cols)
                for m in models:
                    f1 = _fit_score(m, X_tr, y_tr, X_te, y_te, seed, ones)["f1"]
                    rows.append({"dataset": dataset, "seed": seed, "noise": rate, "model": m, "arm": arm, "f1": float(f1), "kind": "score"})
            print(dataset, "seed", seed, "noise", rate, "done", flush=True)
    return rows


def summarise(rows: list[dict]) -> pd.DataFrame:
    s = pd.DataFrame([r for r in rows if r["kind"] == "score"])
    g = s.groupby(["dataset", "noise", "model", "arm"]).f1.mean().unstack("arm")
    out = g.copy()
    for a in g.columns:
        out[a] = g[a] - g["dirty"]
    out["clean_gap"] = g["clean"] - g["dirty"]
    return out.round(4)


GATE_SEEDS, GATE_MARGIN, GATE_TAU, VAL_FRACTION = (200, 201, 202, 203, 204), 0.005, 0.4, 0.25


def run_gated(dataset: str, seeds, noises, models) -> list[dict]:
    """The filter as a GATED candidate: per model it is used only if it beats the raw training rows on a validation slice of the DIRTY
    training data by more than GATE_MARGIN (the Phase 20 'tuned' rule, same margin, nothing re-tuned); otherwise the raw rows are kept."""
    clean, target = load(dataset)
    rows = []
    for seed in seeds:
        tr_i, te_i = train_test_split(clean.index, test_size=0.2, random_state=seed, stratify=clean[target])
        clean_tr, clean_te = clean.loc[tr_i].reset_index(drop=True), clean.loc[te_i].reset_index(drop=True)
        clean_tr[target] = clean_tr[target].astype(str); clean_te[target] = clean_te[target].astype(str)
        X_te, y_te = _typed(clean_te, target, []); ones = np.ones(len(y_te), dtype=bool)
        for rate in noises:
            d, flip = flip_labels(clean_tr, target, rate, seed)
            a_i, v_i = train_test_split(d.index, test_size=VAL_FRACTION, random_state=seed, stratify=d[target])
            a, v = d.loc[a_i].reset_index(drop=True), d.loc[v_i].reset_index(drop=True)
            a_f = a[given_label_prob(a, target, seed, []) >= GATE_TAU].reset_index(drop=True)
            d_f = d[given_label_prob(d, target, seed, []) >= GATE_TAU].reset_index(drop=True)
            Xv, yv = _typed(v, target, []); onesv = np.ones(len(yv), dtype=bool)
            for m in models:
                raw_v = _fit_score(m, *_typed(a, target, []), Xv, yv, seed, onesv)["f1"]
                fil_v = _fit_score(m, *_typed(a_f, target, []), Xv, yv, seed, onesv)["f1"]
                use = (fil_v - raw_v) > GATE_MARGIN
                f_dirty = _fit_score(m, *_typed(d, target, []), X_te, y_te, seed, ones)["f1"]
                f_filt = _fit_score(m, *_typed(d_f, target, []), X_te, y_te, seed, ones)["f1"]
                f_clean = _fit_score(m, *_typed(clean_tr, target, []), X_te, y_te, seed, ones)["f1"]
                rows.append({"dataset": dataset, "seed": seed, "noise": rate, "model": m, "kind": "gated", "val_gain": float(fil_v - raw_v),
                             "used_filter": bool(use), "dirty": float(f_dirty), "always_filter": float(f_filt), "gated": float(f_filt if use else f_dirty),
                             "clean": float(f_clean)})
            print(dataset, "seed", seed, "noise", rate, "done", flush=True)
    return rows


def summarise_gated(rows: list[dict]) -> pd.DataFrame:
    g = pd.DataFrame(rows)
    g["d_always"] = g.always_filter - g.dirty; g["d_gated"] = g.gated - g.dirty; g["gap"] = g.clean - g.dirty
    return g.groupby(["dataset", "noise", "model"]).agg(gap=("gap", "mean"), d_always=("d_always", "mean"), d_gated=("d_gated", "mean"),
                                                         filter_used=("used_filter", "mean")).round(4)


if __name__ == "__main__":
    stage = sys.argv[1]
    t0 = time.time()
    models = ("logistic_regression", "random_forest", "gradient_boosting") if stage == "select" else MODELS
    if stage == "gated":
        noises = (0.0, 0.05, 0.10, 0.20)
        rows = run_gated("adult", GATE_SEEDS, noises, MODELS) + run_gated("bank", GATE_SEEDS, noises, MODELS)
        summ = summarise_gated(rows); print(summ.to_string())
        json.dump({"stage": stage, "rows": rows, "seconds": round(time.time() - t0, 1)}, open(os.path.join(ROOT, "results", "phase27_gated.json"), "w"), default=str)
        summ.to_csv(os.path.join(ROOT, "results", "phase27_gated_summary.csv")); sys.exit(0)
    if stage == "select":
        rows = run("adult", SELECT_SEEDS, TAUS, NOISE, models)
    else:
        tau = float(sys.argv[2]); noises = (0.0, 0.05, 0.10, 0.20)          # 0% and 5% added: a filter that hurts clean or lightly noisy data would be unacceptable
        rows = run("adult", CONFIRM_SEEDS, (tau,), noises, models) + run("bank", CONFIRM_SEEDS, (tau,), noises, models)
    summ = summarise(rows)
    print(summ.to_string())
    json.dump({"stage": stage, "rows": rows, "seconds": round(time.time() - t0, 1)}, open(os.path.join(ROOT, "results", f"phase27_{stage}.json"), "w"), default=str)
    summ.to_csv(os.path.join(ROOT, "results", f"phase27_{stage}_summary.csv"))
