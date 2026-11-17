"""
Invariants for file intake (CSV/JSON/JSONL/Parquet) and the model panel (Phase 19, Stage U; docs/PHASE19_SPEC.md).
Run:  .venv/bin/python src/test_phase19.py

U0 split integrity and determinism, U1 the null file (no improvements manufactured from a clean file), U3 the
leakage count, U4 honest labels (no percentage for a within-noise delta), U5 robustness, plus the backend wiring.
Every check can fail.
"""
import io
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.abspath(ROOT))

import model_panel as mp
from intake import load_table
from taskclean import audit_dataset


# ---------------------------------------------------------------------------------------------------------- intake
def sample_frame(n=120, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"age": rng.integers(18, 80, n), "city": rng.choice(["Pune", "Delhi", "Goa"], n),
                         "score": rng.normal(size=n).round(3), "label": rng.choice(["yes", "no"], n)})


def test_intake_formats_round_trip():
    df = sample_frame()
    buf = io.BytesIO()
    df.to_parquet(buf)
    got, info = load_table(buf.getvalue(), "t.parquet")
    assert info["format"] == "parquet" and got.equals(df), "parquet must round-trip exactly"
    got, info = load_table(buf.getvalue(), "no_extension")                      # recognised by its magic bytes
    assert info["format"] == "parquet"
    got, info = load_table(df.to_json(orient="records").encode(), "t.json")
    assert info["format"] == "json" and got.shape == df.shape and (got["age"] == df["age"]).all()
    got, info = load_table(df.to_json(orient="records", lines=True).encode(), "t.jsonl")
    assert info["format"] == "jsonl" and got.shape == df.shape
    got, info = load_table(json.dumps({"data": df.head(5).to_dict("records")}).encode(), "wrapped.json")
    assert got.shape == (5, 4), "an object wrapping the records is unwrapped"
    got, _ = load_table(json.dumps({"a": [1, 2, 3], "b": ["x", "y", "z"]}).encode(), "cols.json")
    assert got.shape == (3, 2) and list(got.a) == [1, 2, 3], "an object of columns is read as columns"
    nested = json.dumps([{"a": 1, "b": {"c": 2}, "t": [1, 2]}, {"a": 3, "b": {"c": 4}, "t": []}]).encode()
    got, _ = load_table(nested, "n.json")
    assert set(got.columns) == {"a", "b.c", "t"} and got.loc[0, "t"] == "[1, 2]", "nested objects flatten, lists keep their text"
    got, info = load_table(df.to_csv(index=False).encode(), "t.csv")
    assert info["format"] == "csv" and info["delimiter"] == "," and got.shape == df.shape
    for bad, name in [(b"{not json", "b.json"), (b"[1, 2, 3]", "b.json"), (b"   ", "b.csv"), (b"PAR1garbage", "b.parquet")]:
        try:
            load_table(bad, name)
        except ValueError:
            continue
        raise AssertionError(f"{name} {bad!r} must be rejected with a ValueError")
    print("OK intake: CSV, JSON (list, wrapped, columns, nested), JSON Lines and Parquet read; bad files rejected cleanly")


# ---------------------------------------------------------------------------------------------------------- panel
def learnable_frame(n=1500, seed=1, dup_share=0.0, noise=0.1):
    """A classification frame whose label depends on the features; optionally a share of rows are exact copies."""
    rng = np.random.default_rng(seed)
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    g = rng.choice(["a", "b", "c"], n)
    y = ((x1 + 0.8 * x2 + (g == "a") * 0.7 + rng.normal(scale=0.5, size=n)) > 0.3).astype(int)
    y = np.where(rng.random(n) < noise, 1 - y, y)
    df = pd.DataFrame({"x1": x1.round(4), "x2": x2.round(4), "g": g, "uid": np.arange(n), "label": np.where(y == 1, "hi", "lo")})
    if dup_share:
        k = int(n * dup_share)
        df = pd.concat([df, df.sample(k, random_state=seed)], ignore_index=True)
    return df


FAST = dict(models=("logistic_regression", "random_forest"), seeds=(42, 43))


def test_u0_split_integrity_and_determinism():
    df = learnable_frame()
    st = audit_dataset(df, "label", detectors=mp.PANEL_DETECTORS)
    seen = []
    real = mp.audit_dataset

    def spy(frame, *a, **k):
        seen.append(mp._record_hash(frame))
        return real(frame, *a, **k)

    mp.audit_dataset = spy
    try:
        p1 = mp.run_model_panel(st, **FAST)
    finally:
        mp.audit_dataset = real
    assert p1["status"] == "ok", p1
    # rebuild the held-out rows exactly as the panel does, then require that no audit/cleaner ever saw one of them
    y = st.y.reset_index(drop=True)
    for seed, hashes in zip(FAST["seeds"], seen):
        _, te = train_test_split(df.index, test_size=mp.TEST_SIZE, random_state=seed, stratify=y)
        test_hashes = set(mp._record_hash(df.loc[te]))
        assert not (set(hashes) & test_hashes), f"seed {seed}: a held-out row reached the cleaner"
        assert len(hashes) == len(df) - len(te), "the cleaner must see exactly the training rows"
    p2 = mp.run_model_panel(st, **FAST)
    keep = lambda p: [(r["model"], r["arm"], round(r["f1_mean"], 12), r["verdict"]) for r in p["results"]]   # noqa: E731
    assert keep(p1) == keep(p2), "same data and seeds must give the same table"
    # the original frame is not modified
    assert df.equals(learnable_frame()), "the panel must not change the uploaded frame"
    print("OK U0: cleaners see training rows only, the held-out rows never reach them, same seeds give the same table")


def test_u4_honest_labels():
    df = learnable_frame(dup_share=0.1)
    st = audit_dataset(df, "label", detectors=mp.PANEL_DETECTORS)
    p = mp.run_model_panel(st, **FAST)
    assert p["status"] == "ok"
    for r in p["results"]:
        if r["verdict"] in ("improved", "worse"):
            assert abs(r["delta_f1"]) > r["noise_floor"] and r["seeds_agree"] >= mp.SIGN_AGREEMENT, r
            assert r["delta_pct"] is not None
        else:
            assert r["delta_pct"] is None, f"a '{r['verdict']}' delta must carry no percentage: {r}"
        if r["verdict"] in ("no change", "no effect"):
            assert r["delta_f1"] == 0.0 and r["delta_pct"] is None
    assert {r["verdict"] for r in p["results"] if r["arm"] == "before"} == {"baseline"}
    assert p["leakage"]["test_rows_with_exact_copy_in_train_pct"] > 0
    flat = mp.panel_rows(p)
    assert len(flat) == len(p["results"]) and "delta_pct" in flat.columns
    print("OK U4: improved/worse need the noise floor and 80% of seeds; everything else carries no percentage")


def test_u3_leakage_count_and_inflation():
    # label is mostly noise, so a model can only score well on test rows whose copy it has memorised
    df = learnable_frame(n=1500, seed=5, dup_share=0.30, noise=0.45)
    st = audit_dataset(df, "label", detectors=mp.PANEL_DETECTORS)
    p = mp.run_model_panel(st, models=("random_forest",), seeds=(42, 43, 44))
    assert p["status"] == "ok"
    pct = p["leakage"]["test_rows_with_exact_copy_in_train_pct"]
    # independent count with the same splits
    y = st.y.reset_index(drop=True)
    exp = []
    for seed in (42, 43, 44):
        tr, te = train_test_split(df.index, test_size=mp.TEST_SIZE, random_state=seed, stratify=y)
        exp.append(mp._record_hash(df.loc[te]).isin(set(mp._record_hash(df.loc[tr]))).mean() * 100)
    assert abs(pct - float(np.mean(exp))) < 1e-9, (pct, exp)
    assert pct > 15, f"30% of the rows are copies, so well over 15% of held-out rows should have a copy in training, got {pct:.1f}"
    before = next(r for r in p["results"] if r["arm"] == "before")
    assert before["f1_noleak_mean"] is not None and before["f1_mean"] > before["f1_noleak_mean"] + 0.03, before
    print(f"OK U3: {pct:.1f}% of held-out rows have a training copy (matches an independent count); "
          f"raw score {before['f1_mean']:.3f} vs {before['f1_noleak_mean']:.3f} without copies")


def test_u5_robustness():
    base = learnable_frame()
    st = audit_dataset(base, "label", detectors=mp.PANEL_DETECTORS)
    # unlabelled
    st0 = audit_dataset(base, None, detectors={"missing_values", "duplicates"})
    p = mp.run_model_panel(st0, **FAST)
    assert p["status"] == "declined" and "target" in p["reason"], p
    # too few rows
    small = learnable_frame(n=300)
    p = mp.run_model_panel(audit_dataset(small, "label", detectors=mp.PANEL_DETECTORS), **FAST)
    assert p["status"] == "declined" and str(mp.MIN_PANEL_ROWS) in p["reason"], p
    # a rare class is dropped with a note, the rest still runs
    rare = learnable_frame(n=900)
    rare.loc[:5, "label"] = "tiny"
    p = mp.run_model_panel(audit_dataset(rare, "label", detectors=mp.PANEL_DETECTORS), **FAST)
    assert p["status"] == "ok" and any("tiny" in n for n in p["notes"]), p.get("notes")
    # an all-missing feature column and a single usable feature
    odd = base.copy()
    odd["empty"] = np.nan
    p = mp.run_model_panel(audit_dataset(odd, "label", detectors=mp.PANEL_DETECTORS), **FAST)
    assert p["status"] == "ok"
    one = base[["x1", "label"]]
    p = mp.run_model_panel(audit_dataset(one, "label", detectors=mp.PANEL_DETECTORS), **FAST)
    assert p["status"] == "ok" and len(p["models"]) == 2
    # the neural net is skipped, with the reason, when there are too few training rows
    p = mp.run_model_panel(audit_dataset(learnable_frame(n=620), "label", detectors=mp.PANEL_DETECTORS),
                           models=("logistic_regression", "neural_net"), seeds=(42, 43))
    assert p["status"] == "ok" and [m["key"] for m in p["models"]] == ["logistic_regression"], p["models"]
    assert any("Neural net" in n and "skipped" in n for n in p["notes"]), p["notes"]
    # identifier-like text columns are left out of the models, with a note
    ident = base.copy()
    ident["customer"] = [f"C{i:05d}" for i in range(len(ident))]
    p = mp.run_model_panel(audit_dataset(ident, "label", detectors=mp.PANEL_DETECTORS), **FAST)
    assert p["status"] == "ok" and any("customer" in n for n in p["notes"]), p["notes"]
    # a numeric-text column and placeholders still go through the audit's own preparation
    txt = base.copy()
    txt["amount"] = txt["x1"].map(lambda v: f"{v:.2f}")
    txt.loc[::17, "g"] = "?"
    p = mp.run_model_panel(audit_dataset(txt, "label", detectors=mp.PANEL_DETECTORS), **FAST)
    assert p["status"] == "ok"
    print("OK U5: no target, too few rows, rare class, empty column, one feature, small-data MLP skip, ID column, text numbers")


def test_u1_null_file_manufactures_no_improvements():
    path = os.path.join(ROOT, "data", "adult_clean.csv")
    if not os.path.exists(path):
        print("skip U1: adult_clean.csv not present")
        return
    df = pd.read_csv(path).drop_duplicates().reset_index(drop=True)          # a file with nothing to repair
    st = audit_dataset(df.sample(5000, random_state=0).reset_index(drop=True), "class",
                       detectors=mp.PANEL_DETECTORS)
    p = mp.run_model_panel(st, seeds=(42, 43, 44))
    assert p["status"] == "ok"
    rows = [r for r in p["results"] if r["arm"] != "before"]
    quiet = [r for r in rows if r["verdict"] in ("within noise", "no change", "no effect")]
    share = len(quiet) / len(rows)
    loud = [(r["model"], r["arm"], round(r["delta_f1"], 4), r["verdict"]) for r in rows if r not in quiet]
    assert share >= 0.90, f"only {share:.0%} of deltas on a clean file are within noise: {loud}"
    print(f"OK U1: on a clean file {share:.0%} of {len(rows)} (model, repair) deltas are within noise or unchanged {loud or ''}")


# ---------------------------------------------------------------------------------------------------------- backend
def test_backend_panel_job_and_downloads():
    import backend.main as bm
    df = learnable_frame(n=900, dup_share=0.1)
    buf = io.BytesIO()
    df.to_parquet(buf)
    parsed, info = load_table(buf.getvalue(), "t.parquet")
    ds_id = bm.store_dataset(parsed, info, "t.parquet", raw=buf.getvalue())
    snap = bm.DATASETS[ds_id]["snapshot"]
    assert snap is not None and snap["overview"]["input_format"]["format"] == "parquet", "the health report reads Parquet too"
    req = bm.AuditRequest(dataset_id=ds_id, target="label", model_panel=False)
    job_id = "j" + ds_id
    bm.JOBS[job_id] = bm.JobDict(id=job_id, status="queued", progress=0.0, message="", stage="", error=None, audit_id=None,
                                 started=0.0, finished=None, calibrate=False)
    bm.run_audit_job(job_id, req)
    assert bm.JOBS[job_id]["status"] == "done", bm.JOBS[job_id]
    audit_id = bm.JOBS[job_id]["audit_id"]
    assert bm.read_panel(audit_id) == {"status": "none"}, "model_panel=False must not start a panel"
    bm.AUDITS[audit_id]["panel"] = {"status": "queued", "progress": 0.0}
    bm.run_panel_job(audit_id, 2)                                                 # what the executor runs after an audit
    out = bm.read_panel(audit_id)
    assert out["status"] == "ok" and out["n_seeds"] == 2 and len(out["models"]) == 4, out.get("reason", out.get("status"))
    names = {f["name"] for f in bm.dashboard_payload(bm.AUDITS[audit_id])["files"]}
    assert {"model_panel.csv", "model_panel.json"} <= names, names
    flat = pd.read_csv(io.BytesIO(bm.audit_files(bm.AUDITS[audit_id])["model_panel.csv"]))
    assert len(flat) == len(out["results"])
    # the panel does not change the audit, the repair plan or the cleaned files (U7)
    before = {n: b for n, b in bm.audit_files(bm.AUDITS[audit_id]).items() if not n.startswith("model_panel")}
    bm.AUDITS[audit_id].pop("panel")
    after = {n: b for n, b in bm.audit_files(bm.AUDITS[audit_id]).items()}
    assert before.keys() == after.keys() and all(before[k] == after[k] for k in before), "panel on/off changes the audit outputs"
    print("OK backend: Parquet upload, panel job, status endpoint, downloads; the panel does not change the audit's files")


if __name__ == "__main__":
    tests = [v for n, v in sorted(globals().items()) if n.startswith("test_") and callable(v)]
    for t in tests:
        t()
    print(f"\nALL PHASE 19 INTAKE AND MODEL-PANEL INVARIANTS HOLD ({len(tests)} tests)")
