# Phase 16: does the evidence depend on the model? Gradient boosting as a second model family

Written **before** any code or experiment. Thresholds below are fixed in advance; results do not change them. Anything
that has to change after results are seen is appended as a visible **Amendment**, never edited in place. Phases 1-11,
13, 14, 15 and every file in `results/` that exists today are untouched; everything here is additive and writes only
`results/phase16_*`.

The standing decision "Random Forest only for experiments" (Phase 2) is lifted **for new experiments only**. The frozen
Random Forest results stay the reference and are never overwritten or re-labelled.

## Why this phase exists

Every number the product shows about *damage* and *repair safety* was measured with one model: a 300-tree Random
Forest on UCI Adult (and, in Phase 13, on Bank Marketing). The README says so, but the product's behaviour rests on it:

| Product behaviour | Rests on RF-measured evidence |
|---|---|
| "Task-impact estimate" per issue | Phase 7 damage curve (F1 drop vs clean) |
| Which repairs are proposed, applied or held | Phase 9 recovery, risk, repairability status per (issue, rate) |
| Safe-auto-apply gate (risk <= 1e-9, precision >= 0.95, status not negative) | passes **duplicate removal only**, on RF evidence |
| Priority ranking (label errors first, feature corruption last) | Phase 10, built from the two rows above |

The open question is whether these are facts about **the errors** or facts about **the Random Forest**. They differ in
a known direction: bagging averages away label noise; boosting fits the residual and therefore tends to chase noisy
labels. If label-error damage is much larger under boosting, or if a repair that looks harmless to a forest hurts a
boosted model, the product's evidence is model-specific and must say so. If it is the same, the claim becomes
"measured on two model families", which is materially stronger. Either answer is useful; this phase reports whichever
it gets.

## A. Model and harness

**Model.** `sklearn.ensemble.HistGradientBoostingClassifier` (histogram gradient-boosted trees; the same algorithm
family as LightGBM and XGBoost's `hist` method). Chosen over XGBoost/LightGBM because it is in the installed
scikit-learn 1.9.1 (no compiled dependency, nothing new to ship), is deterministic given the data, and is the
recommended scikit-learn estimator for this data size. It is **one implementation of one family**; section G says what
that does and does not license.

**One variable.** Everything except the classifier is held identical to Phases 7-9: the Adult clean frame
(`load_adult_clean`, 45,222 rows), the 70/15/15 split with seed 42 (31,655 / 6,783 / 6,784), the corruptors and RATES
(5/10/20%), the preprocessing (`build_preprocessor`: impute, one-hot, scale), seeds 42-46, the metric set, and the
detect-and-repair step (`detect_and_repair`, detectors included, run unchanged; the repaired frames are model-independent
so they are computed once per (issue, rate, seed) and reused by every arm).
* The only change to the pipeline is a densifying step after the preprocessor, because
  `HistGradientBoostingClassifier` rejects sparse input. It changes no value.
* **Native missing-value handling is not used.** The pipeline imputes, as in the RF runs; using GB's native NaN support
  would change what "missing-value damage" means and break the comparison. (A follow-up, not an arm.)
* The frozen modules are **not edited**. The harness is a new module (`src/phase16_gb.py`) that re-implements the
  Phase 7 / Phase 9 loops with the model injected as a factory, exactly as Phase 14 did. Its validity is gated by G0.

**Metrics.** Primary: positive-class F1 (as Phases 2-9, so the damage numbers are comparable). Also recorded for every
run: macro-F1 and ROC-AUC (Phase 13 showed positive-class F1 can *rise* under label noise on imbalanced targets;
Adult is 24.8% positive, but the other two are recorded anyway). All criteria below use positive-class F1 on Adult
unless stated; AUC and macro-F1 versions are reported in the results and are not criteria.

## B. Hyperparameters: chosen once, on clean data, before any dirty run

A boosted model's noise sensitivity depends heavily on its capacity, so the configuration cannot be a free choice made
after seeing results.

* Fixed: `early_stopping=False` (the default `"auto"` switches on a random internal validation split for n > 10,000,
  which would add a noise-sensitive stopping rule and a source of seed variance), `min_samples_leaf=20`,
  `l2_regularization=0`, `max_leaf_nodes=31`, `random_state=seed`.
* Grid: `max_iter` in {100, 200, 400} x `learning_rate` in {0.05, 0.1}: six candidates.
* Selection: highest **positive-class F1 on the clean validation split** (6,783 rows), clean training data only. Ties:
  smaller `max_iter`, then smaller `learning_rate`. The test split is not consulted.
* The selected configuration is written to `results/phase16_gb_config.json` **before** any dirty run. The runner
  refuses every later stage if that file is absent, so the order is enforced by code, not by this paragraph.
* One consequence stated now: with no early stopping and no subsampling, `HistGradientBoostingClassifier` uses its
  `random_state` for almost nothing, so the **clean baseline will be (near-)identical across seeds**, unlike the forest.
  Seed-to-seed variation under GB then comes from the corruption draw alone. This is checked, not assumed (G0).

**Noise floor.** The RF floor (0.003) came from RF training noise (~0.0025). It cannot be transplanted to a model with
no training noise. The floor for GB is `max(0.003, null_noise)`, where `null_noise` is measured by the existing null
experiment of `selfbench` (retrain on a random 90% of the training rows, 4 draws per seed; noise = the quantity
`selfbench` already uses). Status is also recomputed at the fixed 0.003 and both are reported; the primary is the
measured floor. The status function is the existing `classify_repairability` with `MAJORITY_THRESHOLD = 0.8`, unchanged.

## C. Stages (each runnable and reportable alone)

| Stage | What | Output |
|---|---|---|
| 0 Harness check | RF through the new harness vs the frozen CSVs (G0) | `phase16_gb_harness_check.csv` |
| 1 Selection | Section B; GB clean test scores; LR and RF clean scores for context | `phase16_gb_config.json`, `phase16_gb_clean.csv` |
| 2 Damage | Phase 7 analogue: 5 issues x 3 rates x 5 seeds, GB, plus the null experiment | `phase16_gb_damage_{raw,summary}.csv` |
| 3 Capacity | Stage 2 repeated for two more configurations (below), damage only | `phase16_gb_capacity.csv` |
| 4 Repair | Phase 9 analogue: dirty / repaired / oracle per cell, status, risk, evidence table | `phase16_gb_repair_{raw,summary}.csv`, `phase16_gb_evidence_table.csv` |
| 5 Priority | Phase 10 machinery on the GB evidence table: headline + 15-combination weight grid | `phase16_gb_priority_*.csv` |
| 6 Second dataset | Phase 13 ground-truth protocol on Bank Marketing, GB instead of RF | `phase16_gb_bank_*.csv` |
| 7 Phase 14 under GB | Inconsistency repair recovery re-measured with GB (report only, no criterion) | `phase16_gb_inconsistency.csv` |

**Capacity arms (Stage 3).** `shallow`: `max_leaf_nodes=8, max_iter=100, learning_rate=0.1`. `deep`:
`max_leaf_nodes=63, max_iter=400, learning_rate=0.1`. The selected configuration is the third arm. These are fixed now,
independent of what Stage 1 selects.

**Stage 6** reuses the Phase 13 protocol as it stands (12,000-row cap, 70/15/15, 5 seeds, macro-F1 / ROC-AUC / positive
F1, the same noise-floor definition) with the model swapped; the Phase 13 RF numbers it is compared with are read
from the frozen `results/phase13_bank_marketing_groundtruth_*` files.

**Compute (estimate, not measured).** The detectors are the slow part (the label-noise and feature-corruption detectors
fit Random Forests internally) and are run once and cached. The GB fits are cheap (seconds each). Stages 0-5: well under
two hours on 12 cores; Stage 6 comparable. Long stages run in the background.

## D. Expectations, written down before the results so a surprise is visible

These are predictions, **not criteria**. A wrong prediction is reported as wrong.

* P1. Label-error damage is **larger** under GB than RF, most at 20% (boosting chases noisy labels).
* P2. Outlier and feature-corruption damage is similar in size (both are tree ensembles, invariant to monotone transforms).
* P3. Duplicate damage stays indistinguishable from zero, and duplicate removal stays safe.
* P4. Feature-corruption repair stays net-harmful (the substitutes are wrong values, whatever the model).
* P5. Many (issue, rate) statuses stay `weak`, so agreement is partly chance. Section E gives the chance baseline so
  that "agreement" is not over-read.

## E. Pre-registered criteria (mechanical; a table is produced, nothing is judged by eye)

RF reference values are read from the frozen `results/phase7_impact_summary_full.csv`,
`results/phase10_evidence_table.csv`, `results/phase10_priority_headline.csv` and `results/phase10_sensitivity.csv`.
A **cell** is an (issue, rate) pair: 15 on Adult. An **informative cell** is one where the RF mean damage is at least
0.005 (computed from the frozen file; it is 8 cells today). Ratios on near-zero damage are meaningless, which is why
only informative cells enter G3.

| # | Criterion | Pass if |
|---|---|---|
| G0 | **Harness validity (gate).** RF through the new harness reproduces the frozen RF numbers: `dirty_f1` for 3 damage cells (label_errors 10%, duplicates 5%, outliers 20%) and `repaired_f1` for 1 repair cell (feature_corruption 10%), all seed 42. A second check: two GB clean fits with different `random_state` give identical predictions. | RF values match to 1e-9. If this fails **nothing else is reported as a finding**; the harness is fixed through a visible amendment. |
| G1 | **The comparison is against a competent model.** GB clean test positive-F1 | >= 0.6704 (RF 0.6804 minus 0.01) |
| G2 | **Damage ordering transfers.** Spearman correlation of GB vs RF mean damage over the 15 cells, **and** label errors have the largest GB damage at 10% and at 20% | rho >= 0.8 **and** both rates |
| G3 | **Damage magnitude transfers.** Median of GB/RF damage over the informative cells | between 0.5 and 2.0 |
| G4 | **Repairability status transfers.** Cells with the same status (positive / negative / weak) under both models | >= 11 of 15. Chance baseline stated beside it: "everything weak" agrees on 8 of 15 (the RF weak count); Cohen's kappa is reported |
| G5 | **The auto-apply decision survives.** Duplicate removal passes the existing safe-auto gate under GB (status not negative, risk <= 1e-9, repair precision >= 0.95) | at all three rates |
| G6 | **No new harm.** Cells where RF status is positive or weak and GB status is negative ("GB-only harm") | none for duplicates and at most 1 across all 15 cells; the cells are listed whatever the count |
| G7 | **Priority transfers.** At 10% with equal weights: Kendall tau between the RF and GB rankings of the five issues, **and** the share of the 15 weight combinations in which label errors rank first and feature corruption last | tau >= 0.6 **and** share >= 0.8 (RF: 1.0) |
| G8 | **Dominance does not hinge on capacity.** Label errors have the largest damage at 10% and 20% in all three capacity arms | all six cases |
| G9 | **A second dataset agrees.** On Bank Marketing (macro-F1, the Phase 13 primary): G2's Spearman rule, G4's agreement rule, and G5 for duplicates | all three |

## F. Consequence rules (fixed now; what the outcome does to the product)

The rule that the evidence may only ever make the product **more cautious** applies throughout. Nothing here can unlock
an automatic repair.

| Outcome | Consequence |
|---|---|
| G0 fails | Stop. Amend the harness. No finding is written. |
| G5 fails | Duplicate removal is no longer applied automatically on Adult-sourced evidence; it becomes a proposal. Documented as a limit. |
| G4 or G6 fails | Each disagreeing cell is labelled **model-dependent**; its effective status is the stricter of the two (`negative` if either is negative, `positive` only if both are). The safe-auto gate requires a pass under **both** models. |
| G2, G3 or G8 fails | Impact estimates are labelled with the model and state the GB-to-RF ratio where it is outside the range; the label-error claim is reworded ("dominant under a Random Forest"). |
| G7 fails | The headline ranking is shown with a "model-dependent ranking" note and both rankings. |
| G9 fails but Adult passes | The finding is stated as Adult-specific. |
| Everything passes | The README, review guide and product text say "measured on two model families (Random Forest, gradient boosting)". No gate changes. |

**Product integration, if and only if a rule above triggers or everything passes:** the policy table gains a
"model check" column (agree / model-dependent / not tested) sourced from `phase16_gb_evidence_table.csv`. There is **no
model selector**, the product does not re-label RF evidence as GB evidence, and dataset-calibrated evidence
(`selfbench`) stays Random-Forest-only and says so: the GB check applies to the Adult benchmark evidence only. Wiring
the calibration to a second model is not part of this phase.

## G. Known limits, stated up front

* **One implementation of one family.** `HistGradientBoostingClassifier` is representative of histogram GBDT, not of
  XGBoost's exact method, neural networks or linear models. "Holds for two model families" would be accurate; "holds
  for all models" would not. Logistic Regression (still the Phase 2 follow-up) is not in this phase; the harness takes
  a model factory, so adding it later needs no new design, only its own pre-registration.
* **Detectors are untouched.** The label-error and feature-corruption detectors are Random-Forest-based *inside*. This
  phase varies the **downstream** model only. Whether the Phase 11 finding that the label-error flagged rate *falls* as
  corruption rises is a property of the forest-based detector is a different question and is not answered here.
* **Five seeds.** Per-cell damage standard deviations are 0.002-0.011, so cells with damage below about 0.005 cannot be
  told from zero under either model. That is why G3 uses informative cells only and why G4 is read against its chance
  baseline. The 0.8 sign-agreement rule needs 4 of 5 seeds.
* **Tuned on clean data, then frozen.** Capacity is chosen for clean accuracy, which is the realistic situation, and
  also the one most likely to overfit noise. G8 probes it; it does not remove it.
* **Adult is the anchor, Bank Marketing the replication.** Both are tabular, mixed-type, mid-sized; the conclusion does
  not extend to text, images or very small tables.
* **Equal preprocessing is a choice.** One-hot encoding is what the forest was given; boosting with native categorical
  support would be a different, arguably better, setup. It is held equal to keep the model the only variable.

## G-bis. Clarifications fixed before Stage 6 was run (2026-11-16)

Stages 0-5 had been run; **no Bank Marketing result under GB existed** when these were written. They resolve places where
section E could be read two ways. They change no criterion threshold.

1. **G9, the "G2 Spearman rule".** Only the Spearman part: rho >= 0.8 between GB and RF mean macro-F1 damage over the 15
   cells (RF from the frozen `phase13_bank_marketing_groundtruth_evidence_macrof1.csv`). Which issue is largest at 10% and
   20% is reported under both models but is not part of G9, because the frozen RF Bank table is not required to put
   label errors first.
2. **G9, the "G5 for duplicates".** Bank evidence comes from `selfbench._evidence_table`, which has a noise-aware
   `material_risk` and treats a repair that restores the clean data exactly as harmless. The product's rule applied to
   that evidence is `phase13_transfer.safe_decisions` (material risk <= 1e-9, repair precision >= 0.95, status not
   negative). G9 uses that function, unchanged, on the GB table at all three rates. The strict Phase 10 `risk` is
   reported beside it. (On Adult, whose table has no material risk, G5 used the strict risk; the reason is that the Adult
   gate in the product reads that table.)
3. **Bank hyperparameters.** The configuration selected on clean Adult data (`phase16_gb_config.json`) is used unchanged.
   Selecting again on Bank Marketing would add a tuning step the Adult run did not have. GB's clean Bank scores are
   reported next to RF's; if GB is clearly worse than RF on clean data there, that is stated beside G9.
4. **Harness check for Stage 6.** The RF arm is re-run through the same cells for seed 42 (15 cells) using the frozen
   `phase13_transfer._scores`; dirty and repaired scores for all three metrics must match
   `phase13_bank_marketing_groundtruth_runs.csv` to 1e-9. If any differs, G9 is not reported.

## H. Deliverables

* `src/phase16_gb.py` (harness, stages, criteria table; CLI `--stage {0..7}`), `src/test_phase16.py` (real assertions:
  RF reproduction, determinism, factory injection, criteria arithmetic on synthetic inputs, refusal without the config file).
* `results/phase16_gb_*` as listed in section C, plus the criteria table `phase16_gb_criteria.json`.
* A **Results** section appended to this file, with the pre-registered criteria table filled in as it came out,
  predictions P1-P5 marked right or wrong, and the consequence rules applied mechanically.
* README, REVIEW_GUIDE and (if triggered) product text updated to match. No frozen file is edited.

---

# Results (written 2026-11-16, after Stages 0-6; Stage 7 was run afterwards and is in R8)

Everything below is read from `results/phase16_gb_*`. `phase16_gb_criteria.json` recomputes G0-G9 mechanically from the
files on disk. Nothing was judged by eye, and no criterion threshold was changed after a result was seen. Section G-bis
fixed four interpretation choices before Stage 6 produced any Bank number.

## R1. Setup as run

* GB configuration (selected once on clean Adult validation data, before any dirty run): `max_iter=400`,
  `learning_rate=0.1`, plus the fixed `min_samples_leaf=20, max_leaf_nodes=31, l2_regularization=0, early_stopping=False`.
  It sits at the top corner of the grid and the top four candidates were within 0.0022 validation positive-F1, so the
  choice among them was close to arbitrary; Stage 3 probes it.
* GB has no training noise under this setup: the clean baseline is identical across the five seeds (std 0), so
  seed-to-seed variation comes from the corruption draw alone. Its noise floor is the measured null noise,
  0.0028 (positive-F1), 0.0017 (macro-F1), 0.0011 (AUC), so every metric uses the 0.003 minimum.
* Clean test positive-F1: GB 0.7148, RF 0.6804, LR 0.6615. GB is the stronger clean model by 0.034.

## R2. Criteria as they came out

| # | Criterion | Result | Value |
|---|---|---|---|
| G0 | Harness reproduces frozen RF | **PASS** | 5/5, every difference 0.0 |
| G1 | GB is a competent model | **PASS** | 0.7148 against a threshold of 0.6704 |
| G2 | Damage ordering transfers | **PASS** | Spearman 0.857; label errors largest at 10% and 20% |
| G3 | Damage magnitude transfers | **FAIL** | median GB/RF over 8 informative cells 0.350 (needed 0.5-2.0) |
| G4 | Repairability status transfers | **PASS, on the edge** | 11 of 15 agree (needed 11); "all weak" agrees on 8; kappa 0.55 |
| G5 | Duplicate removal passes the safe-auto gate | **FAIL** | passes at 10% only; fails at 5% and 20% on strict risk |
| G6 | No new harm | **FAIL** | 2 GB-only harm cells (missing values 5%, 10%); none are duplicates; limit is 1 |
| G7 | Priority transfers | **FAIL** | tau 0.80 passes; label-first/feature-last share 0.0 (needed 0.8) |
| G8 | Dominance does not hinge on capacity | **PASS** | label errors largest in all six cases |
| G9 | A second dataset agrees (Bank, macro-F1) | **FAIL** | Spearman 0.004; agreement 9/15; duplicates gate passes at all rates |

Five pass and five fail. G4 passes at exactly its threshold, so it should not be read as strong agreement.

## R3. What was found

**Damage (Stages 2-3).**

* The ordering transfers, the size does not. Mean positive-F1 damage under GB against RF on Adult: label errors
  0.0038 / 0.0111 / 0.0295 against 0.0098 / 0.0248 / 0.0649 at 5 / 10 / 20% (ratios 0.38 / 0.45 / 0.45). Feature
  corruption, outliers and missing values are 0.2-0.7 times the forest's.
* Only label errors at 10% and 20% clearly clear the GB noise floor. The other GB cells are within or near 0.003, so
  their ratios compare two small numbers.
* Capacity matters but does not change the ordering. Label-error damage at 20%: shallow 0.0260, selected 0.0295, deep
  0.0415 (monotone). At 5% and 10% it is not monotone. Even the deep arm is 0.64 times the forest's 0.0649.
* GB is a stronger clean model than RF. Whether that explains its smaller damage was not tested; it is a candidate
  explanation, not a finding.

**Repair (Stage 4).** GB statuses: 8 weak, 6 negative, 1 positive. RF: 8 weak, 4 negative, 3 positive.

* Feature-corruption repair is net-harmful under both models, and about 2-4 times more harmful under GB
  (recovery -0.014 to -0.019 against -0.005 to -0.008).
* The missing-values repair (KNN imputation) is harmful under GB at 10% and 20% (-0.018, -0.021 positive-F1;
  ROC-AUC also falls, -0.008 and -0.010) although it touches exactly the right cells (precision 1.0) and the oracle
  recovers about 0. Under RF it is near zero. This was not predicted. The cause was not tested; a candidate
  explanation is that GB responds differently to the imputed values than to the median fill the pipeline already
  applies.
* Duplicate removal: the repaired model is the same in every seed, so recovery varies only with the dirty model,
  which moves by about 0.003-0.006 with the corruption draw. The gate counts any negative seed as risk, so 5% and 20%
  fail. With the noise-aware material risk, 10% and 20% would pass and 5% would still fail by a hair (-0.0031 against
  0.003). The pre-registered strict risk decides G5. Duplicate damage under GB is not distinguishable from zero, so
  "weak" means nothing detectable to fix, not that removal hurts.
* Label-error repair recovers about zero under GB at every rate, although the oracle would recover the full damage
  (0.0295 at 20%).
* Only one cell is safe to auto-apply under both models: duplicates at 10%.

**Priority (Stage 5).** At 10% the top three are identical under both models (label errors, duplicates, outliers) and do
not move across the 15 weight combinations. Under GB, missing values drops to last because its repair hurts, so
feature corruption and missing values swap places. They are nearly tied (recovery -0.0184 and -0.0178), so G7's failure
rests on a swap between two close cells.

**Second dataset (Stage 6).** The Stage 6 harness check passed (135/135 values, max difference 1e-16). On clean Bank
data GB has positive-F1 0.500 and macro-F1 0.720 against RF's 0.405 and 0.673, so GB is competent there.

* Macro-F1 on Bank is a weak test. The test set has 1,800 rows (about 210 positives); GB's macro-F1 noise floor is
  0.010 (RF 0.007); the forest's own label-error macro-F1 damage is negative at all three rates (label noise appears to
  help, a threshold effect on a 12% minority class); only 5 of 15 forest cells clear the forest's floor. The mismatch is
  a mismatch on a metric whose own baseline is unstable.
* **Exploratory, after seeing the macro-F1 result, not part of G9:** on ROC-AUC the two models agree closely
  (Spearman 0.87; status agreement 14/15 against an all-weak baseline of 11; label errors largest at every rate under
  both). On positive-F1 they do not (Spearman 0.09; 9/15).
* On Bank the label-error damage is larger under GB, about twice the forest's on AUC (0.105 against 0.050 at 20%).
  On Adult it was 0.45 times. The size of the damage therefore depends on the dataset and the model together.

## R4. Predictions D marked

| # | Prediction | Outcome |
|---|---|---|
| P1 | Label-error damage is larger under GB | **Wrong** on Adult (about 0.45 times). Right in direction on Bank (about 2 times on AUC), which was not what P1 predicted |
| P2 | Outlier and feature-corruption damage similar in size | **Wrong** (GB is 0.2-0.3 times the forest's) |
| P3 | Duplicate damage stays indistinguishable from zero, and removal stays safe | **Half right.** Damage is within the noise (-0.0014 to +0.0031). "Stays safe" failed the strict gate at 5% and 20% |
| P4 | Feature-corruption repair stays net-harmful | **Right** (negative at all rates, larger under GB) |
| P5 | Many statuses stay weak, so agreement is partly chance | **Right** (8 of 15 weak under both; the chance baseline is why G4's 11/15 is modest) |

## R5. Consequence rules (section F) applied

| Outcome | Rule fired | Applied as |
|---|---|---|
| G5 fails | Yes | Duplicate removal is no longer applied automatically on Adult-sourced evidence; it becomes a proposal. A documented limit |
| G6 fails (G4 passed) | Yes | The two missing-values cells are labelled model-dependent. Effective status is the stricter of the two; the safe-auto gate needs a pass under both models. Under that rule only duplicates at 10% passes |
| G3 fails | Yes, first clause | Impact estimates are labelled with the model they came from, and the GB-to-RF ratio is stated where outside 0.5-2.0. The second clause (reword the label-error claim) conflicts with G2 and G8 passing: see Amendment A1 |
| G7 fails | Yes | The headline ranking is shown with a "model-dependent ranking" note and both rankings |
| G9 fails | Premise not met | The rule assumes Adult passes, and it does not: see Amendment A2 |
| Everything passes | No | Not triggered |

**Applied 2026-11-16** (after the amendments below were accepted). In `src/taskclean.py`: the policy gains a `model_check`
column (agree / model-dependent / not tested) read from `phase16_gb_evidence_table.csv`; Adult-sourced evidence takes the
stricter status and the larger risk of the two models and must pass the safe-auto gate under **both**; duplicate removal
is proposal-only on Adult-sourced evidence (a human can approve it, and a calibration on the user's own data can still
certify it); impact estimates carry a model note with the GB-to-RF ratio where outside 0.5-2.0; the limitations text and
the report notes name the models. Dataset-calibrated evidence is unchanged and labelled Random Forest only; the Phase 14
inconsistency repair is labelled "model-dependent" after Stage 7 (see R8). The React and Streamlit front ends show the model check; the Research page has the
criteria table and both priority rankings. README and REVIEW_GUIDE updated. No frozen file was edited. One existing
product test (`test_taskclean.test_demo`) changed by design: it asserted that duplicate removal clears the bar on Adult
evidence, and now asserts that nothing does and that a human can still approve it. The rule that evidence may only make
the product more cautious held throughout; nothing here unlocks an automatic repair.

## R6. Amendments (visible, made after the results; **both accepted by the user on 2026-11-16**)

* **A1. G3 wording.** Section F says a G3 failure rewords the label-error claim as "dominant under a Random Forest".
  G2 and G8 passed, so label errors dominate under both models and under all three capacities on Adult, and that wording
  would understate the result. Proposed instead: "label errors are the largest damage under both model families
  (ordering robust); the *size* of the damage is model-specific (about 0.45 times as large under GB on Adult, about 2
  times as large on Bank by AUC)". The first clause of the G3 rule (label impact numbers with their model and ratio)
  stands unchanged.
* **A2. G9 rule.** Section F's row reads "G9 fails but Adult passes -> state the finding as Adult-specific". Adult does
  not fully pass (G3, G5, G6, G7 failed). Proposed reading: G9 failed, so the repair-safety and magnitude findings are
  recorded as not replicated on Bank Marketing under macro-F1, and as Adult-specific until shown otherwise. The
  exploratory AUC agreement is recorded beside it and does not rescue G9.

## R7. What this does and does not show

* **Holds up:** label errors are the largest damage on Adult under both families and every capacity tried; feature-
  corruption repair is net-harmful under both; the *order* of the top three priorities is stable.
* **Does not hold up:** the *size* of the damage, the status of several repairs, duplicate removal's automatic
  approval on Adult-sourced evidence, and the bottom of the priority ranking are model- and dataset-specific.
* **Limits:** two model families, one implementation of each; five seeds; Adult is the anchor and Bank is a noisy
  replication; GB's configuration was fixed from clean Adult and not re-tuned for Bank; Stage 7 (Phase 14 under GB) was
  run afterwards and is reported in R8, with no criterion; the detectors are unchanged and Random-Forest-based inside, so this phase varies only the downstream model.
* **Untested explanations** (stated as such): that GB's smaller Adult damage reflects its higher clean accuracy; that
  the missing-values harm reflects how GB treats KNN-imputed values.

## R8. Stage 7: Phase 14 (inconsistent-value repair) under GB (run 2026-11-16, after R1-R7 were written; report only)

The spec attaches no criterion to this stage. Same Phase 14 injector, detector and repair (the product path:
`audit_dataset` then `_apply`), seeds 42-46, rates 5/10/20%, GB configuration unchanged, noise floor 0.003. The harness
check (the frozen Random Forest scorer and the repair counts on the same frames, seed 42) passed: 15 of 15 values
identical (max difference 0.0).

| Rate | GB damage | RF damage | GB recovery | RF recovery | GB status | RF status | Repair precision / recall |
|---|---|---|---|---|---|---|---|
| 5% | -0.0003 | 0.0127 | -0.0002 | -0.0035 | weak | negative | 1.00 / 0.64 |
| 10% | +0.0007 | 0.0104 | -0.0002 | -0.0045 | weak | negative | 1.00 / 0.64 |
| 20% | -0.0048 | 0.0103 | -0.0030 | -0.0068 | weak | negative | 1.00 / 0.57 |

* Injected inconsistencies (case, spacing, punctuation, abbreviations) cost the Random Forest about 0.010-0.013 F1 but cost
  gradient boosting nothing measurable (|damage| below the noise at 5% and 10%; at 20% the dirty model scores *higher*
  than the clean one, which is noise, not benefit).
* With nothing to recover, repair recovery under GB is about zero and the status is `weak` at all three rates (sign
  agreement 0.6), where the forest gave `negative` at all three. Repair precision (1.00) and recall (0.57-0.64) are properties of
  the detector and are the same under both models.
* The consequence for the product is none: the stricter of the two statuses is still `negative`, so the repair stays in
  the review queue, as before. The product still labels this repair "not tested under gradient boosting"; that label is
  now out of date (see below).
* Why GB is untouched is not tested. A candidate explanation is that a one-hot-encoded spelling variant is a new
  category that a boosted model gives little weight; this is a hypothesis.
* Files: `results/phase16_gb_inconsistency.csv`, `_inconsistency_summary.csv`, `_inconsistency_harness_check.csv`.
* **Applied afterwards (2026-11-17, closing Phase 16):** the product's model check for the inconsistency repair now
  reads "model-dependent" (RF negative / GB weak, effective status negative) and its impact note gives the RF and GB
  damage. No behaviour changes: the stricter status stays negative, so the repair stays in review. Covered by
  `test_taskclean.test_demo`.
