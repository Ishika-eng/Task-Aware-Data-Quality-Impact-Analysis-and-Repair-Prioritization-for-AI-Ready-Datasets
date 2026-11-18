"""
Confirmation of the one candidate found by src/real_file_search.py (exploratory, NOT pre-registered; claim fixed BEFORE this run):
"the tool's TUNED cleaning (Phase 20 rule, margin 0.005, nothing re-tuned) improves at least one model on real Bank Marketing rows in at
least 4 of 5 seeds, and no model gets worse by more than its noise floor". Fresh row sample (sample seed 99, not the 7 used before) and fresh
seeds 100-104. The repair is chosen per model on a validation slice of the TRAINING rows only; the 20% test rows are never looked at until scoring.
They are raw rows of the real file, as dirty as the file itself.
"""
import json, os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path[:0] = [HERE, ROOT]
from sklearn.model_selection import train_test_split
import controlled_proof as cp
from model_panel import MODELS, _fit_score, _typed
cp.TARGET = "Class"
df = pd.read_csv(os.path.join(ROOT, "data", "bank_marketing_original.csv"), low_memory=False).sample(8000, random_state=99).reset_index(drop=True)
rows = []
for seed in (100, 101, 102, 103, 104):
    tr_i, te_i = train_test_split(df.index, test_size=0.2, random_state=seed, stratify=df["Class"])
    tr, te = df.loc[tr_i].reset_index(drop=True), df.loc[te_i].reset_index(drop=True)
    frames, st, _ = cp.repair_frames(tr, seed)
    drop = list(st.unmodeled_columns)
    pick = cp.select_repair(tr, MODELS, seed, drop, True)
    Xte, yte = _typed(te, "Class", drop); ones = np.ones(len(yte), dtype=bool)
    for m in MODELS:
        arm = pick[m]["arm"]
        f_raw = _fit_score(m, *_typed(frames["dirty"], "Class", drop), Xte, yte, seed, ones)["f1"]
        f_pick = f_raw if arm == "dirty" else _fit_score(m, *_typed(frames[arm], "Class", drop), Xte, yte, seed, ones)["f1"]
        rows.append({"seed": seed, "model": m, "chosen": arm, "val_gain": pick[m]["best_gain"], "raw_f1": float(f_raw), "tuned_f1": float(f_pick), "delta": float(f_pick - f_raw)})
    print("seed", seed, "done", flush=True)
r = pd.DataFrame(rows)
s = r.groupby("model").agg(mean_delta=("delta", "mean"), seeds_improved=("delta", lambda x: int((x > 0.0).sum())), seeds_worse=("delta", lambda x: int((x < 0.0).sum())),
                           chose_cleaning=("chosen", lambda x: int((x != "dirty").sum())), most_chosen=("chosen", lambda x: x.value_counts().index[0]))
print(s.round(4).to_string())
json.dump({"rows": rows}, open(os.path.join(ROOT, "results", "real_confirm_bank.json"), "w"))
s.to_csv(os.path.join(ROOT, "results", "real_confirm_bank_summary.csv"))
