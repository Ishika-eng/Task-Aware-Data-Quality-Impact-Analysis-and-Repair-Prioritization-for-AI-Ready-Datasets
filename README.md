# TaskClean

**An impact-aware data quality assessment, repair, and AI-readiness framework.**
TaskClean measures how specific data-quality problems (missing values, duplicates, outliers, label errors,
feature anomalies) actually affect a machine-learning task, decides which repairs are safe to apply
automatically, and reports its confidence honestly, including where repair makes things worse.

It is both a research project (controlled experiments on UCI Adult) and a usable tool (upload a CSV,
get a cleaned dataset, a repair log, and a multidimensional readiness report).

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

streamlit run app.py          # the app: upload CSV -> audit -> review plan -> download
```

Or from the command line:

```bash
cd src
python3 taskclean.py ../data/demo_dirty_adult.csv --target class --out ../taskclean_output_demo
# --feature-anomalies   also run the slow feature-anomaly detector
# --apply missing_values,label_errors   explicit human override of the evidence-based policy
```

> If your checkout path contains a space, run Streamlit as `.venv/bin/python -m streamlit run app.py`;
> the `streamlit` console script's shebang breaks on spaces.

## What the product produces

| File | Contents |
|---|---|
| `dataset_cleaned.csv` | the dataset after the repairs that were applied |
| `repair_log.csv` | every proposed, applied, skipped, and flagged-only change (row, column, original, proposed, applied value, method, tier, detector score, evidence status, human-override flag) |
| `quality_report.csv` | flagged counts/rates per dimension with benchmark precision/recall of each detector |
| `impact_report.csv` | benchmark-based task-impact estimate, repair evidence, and decision per issue |
| `readiness_report.json` / `.csv` | the multidimensional readiness report with evidence status and limitations |

The repair log is the product's proof of what it did: `src/test_taskclean.py` checks that every cell that
differs between the uploaded and cleaned file is a logged, applied repair, and that nothing else changed.

## How it decides what to repair

A repair type is applied **automatically** only if, in the controlled benchmark, it showed **zero observed
harm** and **repair precision >= 0.95**. On the benchmark evidence that is exact duplicate removal only.
Everything else is logged as a proposal and flagged for human review; you can override in the app, and
overrides are recorded. Feature-anomaly repair was demonstrably harmful, so its override carries an explicit warning.

## The research arc (what each phase established)

| Phase | Question | Result |
|---|---|---|
| 1-2 | Clean baseline (UCI Adult, 70/15/15 split, untouched test set) | Random Forest F1 = 0.680, LogReg F1 = 0.662 |
| 3-4 | Controlled corruption with a ground-truth log | 5 error types x 5/10/20%, every changed cell logged with a unique id |
| 5 | Can each error be detected, blind? | exact for missing/duplicates; label noise P=0.77/R=0.28; outliers P=0.18/R=0.67; feature anomalies P=0.28/R=0.26 (cell-level) |
| 6 | Quality report | flagged rate is not a true rate; detector reliability reported separately |
| 7 | How much does each error hurt F1? (5 seeds) | at 20%: label errors 0.065 >> feature corruption 0.014 > outliers 0.009 > missing 0.007 > duplicates 0.002 |
| 8 | Theoretical recoverability (oracle repair) | ceiling = clean F1 for every type |
| 9 | Real automated repair | only duplicate removal is reliably safe; feature-corruption repair is net-harmful at every rate; label repair precision is fine but recall is 0.01-0.18 |
| 10 | Repair prioritization (impact, repairability, risk) | at 10%, label errors rank first and feature corruption last in 100% of a weight grid; the ordering is **severity-dependent** (less stable at 5% and 20%) |
| 11 | AI-readiness | no single score is justified (pooled r = -0.24); the label-error detector's flagged rate is **inverted** relative to real damage |

Notes that matter when citing these results:
- Per-error-type validation uses n = 3 corruption rates: directional evidence, not significance.
- Impact estimates are *benchmark-based* (Adult + Random Forest) and are flagged as boundary estimates beyond the validated 0-20% range.
- Logistic Regression validation and a second dataset (Heart Disease) are **not yet done**.

## Known limitations

- Estimate-based audit: on an uploaded dataset the true errors are unknown; every rate comes from a detector with known limits.
- False-positive floors for statistical detectors come from clean Adult data and are not recalibrated per dataset.
- No before/after model metrics: an uploaded dataset has no clean held-out test set, and evaluating on a slice of the same dirty data would mislead.
- Classification targets only (<= 20 classes). Duplicates are full-record duplicates (features + target).
- Inconsistent-value detection is report-only: no repair evidence exists for it.

## Project layout

```
app.py                       Streamlit front end
src/taskclean.py             product layer: audit -> policy -> repair -> reports (CLI too)
src/test_taskclean.py        end-to-end invariants + generic-CSV robustness
src/data.py  baseline.py     Phase 1-2: data, split, clean baseline
src/corruption/              Phase 3-4: error injection with ground-truth logs
src/detect.py                Phase 5: detectors
src/quality_report.py        Phase 6
src/phase7_impact.py  phase8_oracle_repair.py  phase9_repair_engine.py
src/phase10_prioritization.py  phase11_readiness.py  phase11c_report_demo.py
src/training_log.py          provenance log of every model fit (results/training_log.csv)
src/make_demo_dataset.py     builds data/demo_dirty_adult.csv (+ its corruption log)
data/                        Adult (original + clean) and the demo dataset
results/                     every experiment's raw and summary CSVs
```

`src/errors.py`, `train.py`, `impact.py`, `repair.py`, `readiness.py`, and `pipeline.py` are the superseded v1
prototype (breast-cancer data) and are not used by the current pipeline.
