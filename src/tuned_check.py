"""
Tuned-cleaning check on ANY file you give it (Phase 27 tool; exploratory, not pre-registered).

    .venv/bin/python src/tuned_check.py FILE TARGET [--task classification|regression] [--seeds 5] [--rows 8000] [--drop colA,colB]

What it does, for each of N seeds: split the file's rows 80/20; on the 80% training rows ONLY, let the tool choose a repair per model (the Phase 20 rule: repairs
fitted on 75% of the training rows, candidates scored on the other 25%, kept only if they beat the raw rows by more than 0.005, otherwise the raw rows are kept);
then train each of four models on the raw training rows and on the chosen cleaned rows and score both on the SAME untouched 20% test rows. The test rows are
raw rows of your own file, as dirty as the file, so a gain means "this file's own rows are predicted better", not "every repair was correct".
A change is called improved / worse only if it clears the measured noise floor (random 10% of training rows dropped, 3 draws per seed) with at least 80% of
seeds agreeing; otherwise "within noise". "no change" means the tool kept the raw rows in every seed. The result is whatever it is.

Writes results/tuned_check_<file>.json and prints a table.
"""
from __future__ import annotations

import argparse, json, os, sys, time, warnings
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path[:0] = [HERE, ROOT]
from sklearn.model_selection import train_test_split          # noqa: E402
import controlled_proof as cp                                  # noqa: E402
import controlled_proof_regression as cr                       # noqa: E402
from intake import load_table                                  # noqa: E402
from model_panel import MODELS as CLS_MODELS, MODEL_LABELS as CLS_LABELS, _fit_score as cls_fit, _typed as cls_typed   # noqa: E402
from model_panel_regression import MODELS as REG_MODELS, MODEL_LABELS as REG_LABELS, _fit_score as reg_fit, _typed as reg_typed  # noqa: E402
from regression import strat_labels, suggest_task             # noqa: E402
from selfbench import NOISE_FLOOR_MIN, classify_repairability  # noqa: E402

warnings.filterwarnings("ignore")
VERDICT = {"positive": "IMPROVED", "negative": "WORSE", "weak": "within noise"}


def run(path: str, target: str, task: str | None, seeds: int, rows: int, drop: list[str]) -> dict:
    t0 = time.time()
    df, info = load_table(open(path, "rb").read(), os.path.basename(path))
    if target not in df.columns:
        raise SystemExit(f"target '{target}' not in columns: {list(df.columns)[:12]}...")
    df = df.drop(columns=[c for c in drop if c in df.columns]).dropna(subset=[target]).reset_index(drop=True)
    task = task or suggest_task(df[target])["task"]
    if len(df) > rows:
        df = df.sample(rows, random_state=1).reset_index(drop=True)
    reg = task == "regression"
    mod = cr if reg else cp
    mod.TARGET = target
    models, labels = (REG_MODELS, REG_LABELS) if reg else (CLS_MODELS, CLS_LABELS)
    fit, typed, key = (reg_fit, reg_typed, "score") if reg else (cls_fit, cls_typed, "f1")
    metric = "R-squared" if reg else "macro-F1"
    seed_list = [100 + i for i in range(seeds)]
    out_rows, nulls, default_seen = [], [], set()
    for seed in seed_list:
        strata = strat_labels(df[target]) if reg else df[target].astype(str)
        try:
            tr_i, te_i = train_test_split(df.index, test_size=0.2, random_state=seed, stratify=strata)
        except ValueError:
            tr_i, te_i = train_test_split(df.index, test_size=0.2, random_state=seed)
        tr, te = df.loc[tr_i].reset_index(drop=True), df.loc[te_i].reset_index(drop=True)
        frames, st, default_issues = mod.repair_frames(tr, seed)
        default_seen |= set(default_issues)
        drop_cols = list(st.unmodeled_columns)
        pick = mod.select_repair(tr, models, seed, drop_cols, True)
        Xte, yte = typed(te, target, drop_cols); ones = np.ones(len(yte), dtype=bool)
        for m in models:
            arm = pick[m]["arm"]
            raw = fit(m, *typed(frames["dirty"], target, drop_cols), Xte, yte, seed, ones)[key]
            got = raw if arm == "dirty" else fit(m, *typed(frames[arm], target, drop_cols), Xte, yte, seed, ones)[key]
            out_rows.append({"seed": seed, "model": m, "chosen": arm, "raw": float(raw), "tuned": float(got), "delta": float(got - raw)})
            rng = np.random.default_rng(seed)
            for _ in range(3):                                     # noise floor: drop a random 10% of the training rows
                keep = rng.random(len(tr)) >= 0.10
                nulls.append({"model": m, "d": float(fit(m, *typed(tr.loc[keep], target, drop_cols), Xte, yte, seed, ones)[key] - raw)})
        print(f"seed {seed} done ({time.time() - t0:.0f}s)", flush=True)
    R, N = pd.DataFrame(out_rows), pd.DataFrame(nulls)
    summary = []
    for m in models:
        d = R[R.model == m].delta.to_numpy()
        floor = max(NOISE_FLOOR_MIN, float(np.sqrt(np.mean(np.square(N[N.model == m].d)))))
        chosen = R[R.model == m].chosen
        if (chosen == "dirty").all():
            verdict = "no change (kept the raw rows)"
        elif not np.any(d):
            verdict = "no effect"
        else:
            verdict = VERDICT[classify_repairability(d, floor)]
        summary.append({"model": labels[m], "raw_mean": float(R[R.model == m].raw.mean()), "tuned_mean": float(R[R.model == m].tuned.mean()),
                        "mean_change": float(d.mean()), "seeds_better": int((d > 0).sum()), "seeds_worse": int((d < 0).sum()),
                        "cleaned_in_seeds": int((chosen != "dirty").sum()), "noise_floor": floor, "verdict": verdict,
                        "repairs_chosen": chosen[chosen != "dirty"].value_counts().to_dict()})
    res = {"file": os.path.basename(path), "target": target, "task": task, "metric": metric, "rows_used": len(df), "seeds": seed_list,
           "default_file_applies": sorted(default_seen), "summary": summary, "per_seed": out_rows, "seconds": round(time.time() - t0, 1)}
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    json.dump(res, open(os.path.join(ROOT, "results", f"tuned_check_{os.path.splitext(os.path.basename(path))[0]}.json"), "w"), indent=1)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("file"); ap.add_argument("target")
    ap.add_argument("--task", choices=["classification", "regression"]); ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--rows", type=int, default=8000); ap.add_argument("--drop", default="")
    a = ap.parse_args()
    r = run(a.file, a.target, a.task, a.seeds, a.rows, [c for c in a.drop.split(",") if c])
    print(f"\n{r['file']}  target={r['target']}  task={r['task']}  rows={r['rows_used']}  metric={r['metric']} on untouched raw held-out rows")
    print(f"default (evidence-based) file applies: {r['default_file_applies'] or 'nothing'}")
    print(f"{'model':22s} {'raw':>7s} {'tuned':>7s} {'change':>8s} {'better/worse':>13s} {'cleaned':>8s} {'floor':>7s}  verdict")
    for s in r["summary"]:
        print(f"{s['model']:22s} {s['raw_mean']:7.4f} {s['tuned_mean']:7.4f} {s['mean_change']:+8.4f} {s['seeds_better']:>6d}/{s['seeds_worse']:<6d} {s['cleaned_in_seeds']:>5d}/{len(r['seeds'])} {s['noise_floor']:7.4f}  {s['verdict']}  {s['repairs_chosen'] or ''}")
