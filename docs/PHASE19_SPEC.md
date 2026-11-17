# Phase 19: a model panel (four model families) for the evidence and for before/after on the user's own data

Written **2026-11-17, before any code or experiment**. Thresholds are fixed here; results do not change them. Anything that
has to change after results are seen is appended as a visible **Amendment**, never edited in place. Phases 1-11, 13-18 and every
file in `results/` that exists today are untouched; new code is additive and writes only `results/phase19_*`.

This is a specification, not a result. Nothing in it has been run.

## Why this phase exists

Two gaps, both stated in the README or found in earlier phases.

1. **The evidence rests on two model families.** Damage and repair safety were measured with a Random Forest (Phases 7-9, 13) and
   re-measured with gradient boosting (Phase 16). Phase 16 showed the answer depends on the model (the missing-value repair hurt
   boosting but not the forest; the inconsistency repair is "model-dependent"). A linear model and a small neural net are the
   families expected to react *most* to dirty numbers (outliers, scale, skew), and neither has been measured. Logistic regression
   was deferred in Phase 2 ("RF now, LogReg validation later") and is on the open follow-up list.
2. **The user never sees a before/after on their own file by default.** The default evidence is the UCI Adult benchmark.
   "Calibrate on this dataset" re-measures damage and repair safety on the user's data, but with one model (Random Forest) and it
   does not show what the downloaded files would do to a model trained on that data.

## What is built (two stages, in this order)

### Stage S: do the evidence and the safe-repair rule survive four model families? (experiment)

The same harness as Phase 16 (`src/phase16_gb.py`: inject one error type at 5/10/20% into the **training split only**, train,
score on the untouched clean test set, 5 seeds 42-46, paired against a clean-trained model per seed, then run the real
`audit_dataset` -> `_apply` path and measure recovery), with the model swapped. New models:

| Family | Model | Settings (fixed now) |
|---|---|---|
| Linear | `LogisticRegression` | `max_iter=1000`, default `C=1`, the shared preprocessor (median / most-frequent fill, one-hot, standard scaling) |
| Neural net | `MLPClassifier` | one hidden layer of 64, `early_stopping=True`, `max_iter=200`, same preprocessor |

Random Forest (300 trees) and `HistGradientBoostingClassifier` stay as measured. Datasets: **Adult** (the reference) and **Bank
Marketing** (Phase 13), error types missing values, duplicates, outliers, label errors. Inconsistencies are excluded (the
product's own Phase 14 evidence is injected-variant based and model-dependent; re-measuring it for two more models is out of
scope here). The primary metric is macro-F1 (what the self-benchmark uses); positive-class F1 and ROC-AUC are reported.

**Criteria (fixed now):**

* **S0, harness check.** The harness, run with Random Forest on Adult, reproduces the recorded Phase 7 damage and clean baseline
  (clean F1 within 0.001 of the stored value, as Phase 16 checked). If not, nothing else in Stage S is read.
* **S1, noise floor first.** A null experiment (retrain on a random 90% of the training rows, 4 draws per seed) is run per model and
  dataset before any effect is read. A damage or recovery smaller than that model's noise floor is reported "within noise" and
  never interpreted. Models whose own noise floor is above `MAX_CERTIFIABLE_NOISE_FLOOR` (0.01) cannot certify any non-exact repair
  safe, as the product already says.
* **S2, evaluable.** A (model, dataset, issue, rate) cell is evaluable only if the clean-trained model beats chance by the
  self-benchmark's `LEARNABLE_MARGIN` (0.05) and the cell's damage is above the noise floor. Cells that are not evaluable are
  reported as such, not as "no harm".
* **S3, agreement table (the deliverable).** For every issue: damage sign and size, and repair status
  (`positive` / `zero` / `negative` by the existing `classify_repairability`), per model, per dataset. The product consequence
  rule is Phase 16's: a repair is only as safe as its **worst** evaluated family, and the larger risk is taken.
* **S4, "does cleaning still help a strong model" (the user's question).** For each model, the share of evaluable
  (issue, rate) cells where the repair recovers more than the noise floor. Reported per model. No expectation is set.
* **S5, integrity.** Frozen result files are byte-identical before and after (hash check, as in Phase 16/17); Stage S writes only
  `results/phase19_suite_*`.

**What counts as a finding.** Agreement across all four families: "measured on four model families". Disagreement: the
disagreement is the result and is shown as such; **it is not tuned away**. If a repair that the product currently proposes
is `negative` under logistic regression or the MLP, the product's status becomes the stricter one and the README says which
model drove it (this changes the product only through the existing stricter-status rule; no new threshold).

**Predictions (recorded before the run, not criteria; wrong predictions are reported):** (1) logistic regression is hurt more
than the forest by outliers and recovers more from outlier handling; (2) the MLP has the largest seed-to-seed spread of the four;
(3) label errors hurt boosting more than logistic regression (boosting fits residuals, the linear model cannot memorise);
(4) duplicate removal stays "safe" under all four.

### Stage U: the model panel on the user's own data (product feature)

After a target is chosen and the audit has run, a **model panel** trains the four families on the user's file twice, with the
same split and seed, and shows the difference. Classification only (the product's existing limit: 2-20 classes; a numeric target
with many values still gets the existing "regression is not supported" message and no panel).

**Protocol (fixed now).**
1. **Split first.** A stratified 80/20 split of the **raw** uploaded rows, per seed. The test rows are never removed or edited.
2. **Cleaning is fitted on the training split only.** Repairs are proposed and applied to the training rows; nothing about the
   test rows is read by any cleaner (imputer values, outlier fences, duplicate removal, spellings all come from training data).
3. **Before (control)** = the shared "survival" preprocessor (the same one used by every phase: median / most-frequent fill,
   one-hot, standard scaling) on the raw training rows. No outlier removal, no deduplication, no spelling repair.
   **After** = the same preprocessor and model on the training rows after repair. The preprocessor is part of the survival
   pipeline, not a cleaning step; scaling is kept because logistic regression and the MLP need it.
4. **Arms**: *before*; *default file* (the repairs the product applies without review); *aggressive file* (everything
   proposed); and *each applicable repair type alone*. One model, one seed, one split per row of results.
5. **Seeds**: 5 (42-46); each seed is a different split and a different model seed; per-seed paired differences.
6. **Models**: logistic regression, random forest (300 trees), `HistGradientBoostingClassifier`, MLP (settings as in Stage S).
   The MLP is skipped, with the reason shown, when there are fewer than 500 training rows or the time budget would be exceeded.
   LightGBM / XGBoost are not added: the sklearn boosting model is the same family and adds no dependency.
7. **Metric**: macro-F1 (primary), ROC-AUC (secondary, one-vs-rest for more than two classes). Mean, standard deviation across
   seeds, and the delta against *before*, per model and arm.
8. **Noise and labelling.** The null experiment of S1 (computed on this file) gives each model its noise floor. A delta is
   labelled **beyond noise** only if its absolute mean exceeds that floor **and** at least 80% of the seeds agree on its sign
   (`SIGN_AGREEMENT`, as in Phase 10). Otherwise it is **within noise**, and the report says so rather than showing a
   percentage improvement. No "improvement %" is shown for a within-noise delta.
9. **Leakage check.** The report states how many test rows have an exact copy in the training split, and gives a second set of
   scores on the test rows that have **no** exact copy in training. Duplicates across the split inflate the *before* score and
   would otherwise make deduplication look like a degradation.
10. **Interaction note.** The *all repairs* arm is shown next to the sum of the single-repair arms, with the note that effects
    do not add.
11. **Attribution is per repair type, not per row.** The repair log already lists the cells changed. The panel does **not**
    claim which rows caused a score change.
12. **Size cap.** Panel fits use a stratified sample of at most 10,000 rows (the full file is still audited). If the estimated
    time exceeds the budget, the seed count falls to 3 and the report says so.
13. **Opt-out, never blocking.** The panel runs as a background job after the audit (the existing jobs mechanism) and never
    delays the audit; the page shows "running" and then the table. It can be switched off on the Start page.
14. **Report-only.** In this phase the panel **does not change which repairs the product applies or recommends.** It is
    information. (Using it to decide repairs would be a new, unvalidated feedback loop.)
15. **Not part of the baseline health report.** That snapshot is immutable and taken at upload, before a target exists. The
    panel is stored with the audit, shown on its own page next to the health report, and downloadable as `model_panel.csv` /
    `model_panel.json` in the zip.

**What it does not claim.** A score change does not show that a repair was *correct*: the test set is as dirty as the user's file,
so a gain means "training on cleaned data predicted the real, dirty rows better", and a loss can mean the repair was right and the
test labels are noisy. This is stated on the page.

**Criteria (fixed now):**

* **U0, split integrity (unit tests).** Test rows are identical before and after (hash), no cleaner reads them (checked by
  corrupting the test rows and showing the cleaned training rows are unchanged), same seed gives the same table.
* **U1, null file.** On Adult *with no corruption and nothing to repair*, at least 90% of (model, arm) deltas are labelled
  **within noise**. This guards against a panel that manufactures improvements.
* **U2, oracle agreement.** On Adult with injected errors at 10% and 20% (Phase 7 injectors), the panel is run **as a user would
  run it** (corrupted training rows, dirty test rows) and compared with the Stage S truth (clean test set). Among the (model,
  issue) cells that are beyond noise in the truth, at least 80% have the same sign in the panel. Reported per model.
* **U3, leakage check.** Planted: a file where 30% of rows are exact copies. The panel's leakage count equals the planted number;
  the *before* score on the whole test set is higher than on the no-copy test rows (the inflation is visible).
* **U4, honest labels.** A within-noise delta never shows a percentage improvement (UI test).
* **U5, robustness.** No crash, a sensible message, and no panel (not an error) for: an unlabelled file, a regression-looking
  target, fewer than 600 rows, a class with fewer than 20 rows, an all-missing feature column, a single feature column, an MLP
  that fails to converge (recorded, other models still reported).
* **U6, speed.** Reported, not assumed: wall time for the full panel on Adult (10,000 sampled rows) and on Telco (7,043 rows).
  Pass if under 5 minutes each with 5 seeds; otherwise the 3-seed fallback is used and that is the shipped behaviour.
* **U7, it does not change decisions.** With the panel on versus off, the audit, the repair plan and the downloaded files are
  byte-identical (hash check).

## Consequence rules

* If **S0 fails**, Stage S is not read and the harness is fixed first.
* If **U1 fails** (the panel reports gains on a clean file), the panel is not shown in the product until fixed or its labelling
  made stricter by amendment; the experiment results are still published.
* If **U2 fails** for a model, that model is shown in the panel with a "does not track the clean-test truth" note, not hidden.
* If a Stage S finding makes a currently proposed repair `negative` under any family, that repair's product status becomes the
  stricter one (existing rule). The README states the model that caused it.

## Explicitly out of scope

Regression targets (Ridge/SVR/regression forest: a separate, larger phase), LightGBM/XGBoost, Parquet and JSON upload, any change
to the detectors or to which repairs are applied automatically, a composite score (still rejected: pooled r = -0.24), and
per-row attribution.

## Files (planned)

`src/phase19_suite.py` (Stage S), `src/model_panel.py` (Stage U), `src/test_phase19.py`, `results/phase19_suite_*`, backend job and
download wiring in `backend/main.py`, a "Model panel" page in the React dashboard, README and review-guide sections, and a
Results section appended here.

## Order of work

1. Stage S harness, S0 and the null experiments, then the full runs (long: Adult and Bank, 4 models, 4 issues, 3 rates, 5 seeds,
   in stages as Phases 16 and 17 were), then the S3 table. 2. Unit tests for Stage U (U0, U3, U5). 3. `model_panel.py`, then U1,
   U2, U6, U7. 4. Backend and dashboard. Stage U's product wiring starts only after Stage S's table exists, because the panel's
   statuses and notes depend on it.

---

# Status (2026-11-17): Stage U built, Stage S not run

Built: `src/intake.py` (CSV, JSON, JSON Lines, Parquet), `src/model_panel.py`, the backend job and downloads
(`GET/POST /api/audits/{id}/panel`, `model_panel.csv/.json`), the Model panel page, an Overview card and a Start-page switch,
and `src/test_phase19.py` (7 tests). Two small departures from the plan, both for speed: the panel uses 3 seeds by default
(5 on request), a random forest of 100 trees, and 3 null draws per seed; and it caps the fits at 8,000 rows.

| Criterion | Verdict | Evidence |
|---|---|---|
| U0 split integrity, determinism | PASS | the audit/cleaner never receives a held-out row (checked by recomputing the split), same seeds give the same table, and the same numbers came back after a server restart |
| U1 null file | PASS | Adult with nothing to repair: 24 of 24 (model, repair) deltas within noise or unchanged |
| U3 leakage | PASS | planted 30% copies: the reported count equals an independent count; raw score 0.688 vs 0.492 without copied rows |
| U4 honest labels | PASS | improved/worse only beyond the noise floor with >= 80% seed agreement; no percentage otherwise |
| U5 robustness | PASS | no target, <600 rows, rare class, empty column, one feature, small-data MLP skip, ID column, text numbers |
| U6 speed | PASS | Adult (48,842 rows, sampled to 8,000): audit 14 s + panel 46 s; Telco 7,043 rows: panel 37 s; dirty demo 6,300 rows: 48 s |
| U7 no change to decisions | PASS | audit outputs are byte-identical with the panel present or removed |
| **U2 oracle agreement** | **NOT RUN** | needs the Stage S ground-truth harness; the panel is not shown to track the clean-test truth yet |

**Stage S (the four-family evidence experiment) was not run.** The product's repair-safety evidence is still Random Forest plus
gradient boosting only; the panel is report-only and changes no decision.

Found while testing: the model-dependent pattern the spec predicted appears on the dirty demo. Capping outliers helps logistic
regression (+0.015, beyond noise) and hurts gradient boosting (-0.031) and the neural net (-0.039); removing training duplicates
helps logistic regression but not the forest. A model that has memorised duplicated rows also scores higher on held-out rows
that are copies of training rows, which the leakage column makes visible. The panel's verdict "no effect" means cells were repaired
but the trained model is identical to the raw one (e.g. the fill equals the model's own default fill).
