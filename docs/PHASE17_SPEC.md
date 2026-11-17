# Phase 17: why does KNN imputation hurt gradient boosting?

Written **2026-11-17, before any code or experiment**. Thresholds are fixed here; results do not change them. Anything that
has to change afterwards is appended as a visible **Amendment**, never edited in place. Phases 1-11, 13-16 and every
existing file in `results/` are untouched; this phase is additive and writes only `results/phase17_*`.

## The open question

Phase 16 (Stage 4) found that the missing-values repair (KNN imputation of numeric cells, mode for categorical cells)
*lowers* gradient-boosting (GB) positive-class F1 relative to leaving the cells blank:

| Rate | GB dirty - clean (damage) | GB repaired - dirty (recovery) | RF recovery (frozen Phase 10) |
|---|---|---|---|
| 5% | +0.0008 | -0.0042 | +0.0026 |
| 10% | -0.0008 | -0.0178 | -0.0003 |
| 20% | +0.0053 | -0.0212 | -0.0054 |

"Dirty" here is not broken: the shared pipeline median-imputes NaN itself, so GB loses almost nothing to the missing
cells (damage is within the 0.003 noise floor at 5% and 10%). The repair then makes it **worse than doing nothing** by
about 0.02, which is 6-7x the noise floor. Phase 16 recorded the cause as *untested*. This phase tests it.

## Facts established from the code, not assumed

* `phase9_repair_engine.detect_and_repair("missing_values", ...)` standardises the six numeric columns (age, fnlwgt,
  education-num, capital-gain, capital-loss, hours-per-week), runs `KNNImputer(n_neighbors=5)` over all of them at once and
  writes the imputed value into each flagged numeric cell. Categorical cells get the column mode. Imputed numeric
  values are non-integers.
* The shared pipeline's `SimpleImputer(strategy="median")` / `most_frequent` already fills any NaN left in a frame, so the
  "dirty" arm is median/mode imputation by construction. Leaving a cell NaN and imputing it with the training median are
  the same thing.
* capital-gain is 0 in 91.6% of Adult rows (median 0, max 99,999); capital-loss is similar. KNN averages five
  neighbours, so an imputed capital value is usually 0 but sometimes a *fraction of a rare large value*, a number that
  appears nowhere in the real data. The other four numeric columns are not zero-inflated.
* The corruptor blanks cells MCAR at the stated rate across all 14 columns.

## Hypotheses (candidates for the cause; each is tested by a one-variable arm)

* **H1 (zero-inflated columns).** The harm comes from KNN-imputing capital-gain and capital-loss: GB makes sharp splits on
  these (a few thousand dollars of gain is a strong >50K signal), so invented in-between values create false thresholds
  that the median (0, the real mode) does not.
* **H2 (non-integer values).** The harm comes from fractional imputed values in integer columns (age 37.4).
* **H3 (distance quality).** KNN neighbours are simply poor (fnlwgt is a sampling weight with no relation to the
  target), so the harm is spread over all columns.
* **H4 (capacity).** The harm needs GB's capacity to exploit fine thresholds, so a shallow booster shows far less.

H1 and H3 are rival explanations; H2 and H4 are secondary. My prior is H1 most likely, but I have not looked at any
per-column result.

## Design

One variable per arm: **how the blanked cells are filled before training**. Everything else is Phase 16's: the Adult
clean frame, split seed 42, `corruption.CORRUPTORS["missing_values"]`, rates 5/10/20%, seeds 42-46, the selected GB
configuration (`max_iter=400`, `learning_rate=0.1`, `GB_FIXED`), the same test set. For each (rate, seed) one corruption
draw and **one** KNN run over all six columns are shared by every arm, so differences between arms are paired and
come only from which cells receive KNN values.

| Arm | Cells filled by KNN | Everything else |
|---|---|---|
| `dirty` | none | pipeline median/mode (the Phase 16 dirty arm) |
| `median_explicit` | none (explicit median/mode written into the frame) | control: must equal `dirty` |
| `knn_all` | all six numeric columns | frozen repair; must equal Phase 16 `repaired` |
| `knn_<col>` x 6 | one numeric column | pipeline median |
| `knn_no_capital` | age, fnlwgt, education-num, hours-per-week | median for capital-gain, capital-loss |
| `knn_capital_only` | capital-gain, capital-loss | median for the rest |
| `knn_rounded` | all six, integer-valued columns rounded to int | |
| `knn_guarded` | numeric columns whose most frequent value covers < 50% of the observed values | median for the rest |

Categorical cells are mode-filled in every arm (identical to the pipeline). `knn_guarded` is a general rule stated here
in advance, not a list of column names; on Adult it must select exactly {capital-gain, capital-loss} for the *median*
branch (checked by K0), making it the same frame as `knn_no_capital`.

Secondary arms (cheap): the 4 arms `dirty`, `knn_all`, `knn_no_capital`, `knn_capital_only` under the Phase 16 `shallow`
capacity configuration (H4) and under the frozen 300-tree Random Forest at 10% and 20% (does the RF's smaller harm sit
in the same columns?).

**Mechanism table (no model).** For the cells actually blanked, per numeric column: the share of imputed values that
equal the truth exactly, the MAE, and the share of imputed values that appear nowhere in the real training column, for
KNN versus the median. This shows what each imputation does to the data, independent of any classifier.

**Primary metric:** positive-class F1, `recovery = arm - dirty`; harm = -recovery, averaged over the five seeds.
Noise floor for GB: **0.003** (Phase 16 null experiment). Rates 10% and 20% are the evaluated rates (Phase 16 harm exceeds
the floor there); 5% is reported, not tested. ROC-AUC is reported as a second metric, not tested.

## Criteria (fixed now)

* **K0, faithfulness (must pass before anything is believed).** At all 15 cells: (a) the composed all-column frame equals the
  frozen `detect_and_repair` frame on every numeric cell (max abs difference <= 1e-9); (b) `dirty` and `knn_all` F1 equal
  Phase 16's recorded `dirty_posf1` / `repaired_posf1` (<= 1e-9); (c) `median_explicit` equals `dirty` (<= 1e-9);
  (d) the guard rule selects exactly {capital-gain, capital-loss}. A failure is a bug to report and fix, not a result.
* **K1, H1 (capital columns are the cause).** At **both** 10% and 20%: share of `knn_all` harm removed by `knn_no_capital`
  >= 75% **and** share of `knn_all` harm kept by `knn_capital_only` >= 75%. Pass = both. One of the two = "partly". Neither =
  H1 rejected.
* **K2, attribution.** Capital-gain is the single column with the largest harm at both rates. Reported with all six
  single-column harms; the claim "the harm is concentrated" needs K1 as well.
* **K3, H2 (rounding).** `|recovery(knn_rounded) - recovery(knn_all)|` < 0.003 at both rates means H2 is rejected; otherwise
  rounding matters and is reported.
* **K4, H4 (capacity).** Shallow-GB `knn_all` harm <= 50% of deep-GB harm at both rates = the harm needs capacity.
* **K5, the guard.** `knn_guarded` recovers >= 75% of `knn_all` harm at both rates (on Adult this is K1's first half by
  construction, so it is the check that the *general rule* reproduces it).
* **K6, out of sample (only if K1 and K5 pass).** The guard rule, unchanged, on Bank Marketing under the same GB
  configuration and arms `dirty`, `knn_all`, `knn_guarded`, rates 10% and 20%, five seeds. If `knn_all` harm on Bank is
  below the Bank noise floor the criterion is **not evaluable** and is reported as such, not passed. Otherwise pass =
  guard recovers >= 75% of the harm.

## Predictions, written before the run

P1: K0 passes. P2: K1 passes (capital columns carry >= 75% each way). P3: K2 holds. P4: K3 rejects H2 (rounding does not
matter). P5: K4 is the least certain; I expect the shallow booster to show less harm but do not expect a clean 50% cut.
P6: the RF's harm sits in the same columns but is smaller.

## Consequence rules (fixed now)

* The product (`taskclean._propose_missing`) imputes numeric cells with KNN for any user dataset. It changes **only if**
  K1, K5 and K6 all pass (K6 not evaluable counts as not passed): then zero-inflated numeric columns use median/mode
  imputation instead. This can only make the repair more conservative. If K6 is not evaluable or fails, the finding is
  recorded as an **Adult-specific diagnosis** and the product is left unchanged, with the limitation stated.
* Whatever the outcome, the README and review guide state what was measured. A rejected hypothesis is kept in the results,
  not removed.

## What this phase cannot show

It diagnoses one dataset's repair under one booster implementation. A mechanism confirmed on Adult is a mechanism for
Adult's columns; K6 is the only test that it travels. It does not test other imputers (MICE, indicators), and it does
not ask whether the better imputation helps beyond "no worse than the median".

## K-bis. Clarifications fixed before Stage C (Bank) was run (2026-11-17)

Stage A (Adult, K0-K3, K5) had finished and Stage B was running when these were written; nothing from Bank had been run in
this phase. They resolve two points K6 left open, so the choice is not made after seeing Bank numbers. (Phase 16
Stage 6 already holds GB macro-F1 `dirty` and `repaired` scores for Bank missing values; Stage C uses them only as the
consistency check below, and I am not choosing the metric or the rule with them in view.)

* **Metric.** K6 uses **macro-F1**, Bank Marketing's primary metric in Phase 13 and in Phase 16's G9, with the **Bank GB
  noise floor for macro-F1** recorded in `results/phase16_gb_bank_meta.json`. Positive-class F1 and ROC-AUC are reported,
  not tested.
* **Evaluable per rate.** For each of 10% and 20%, `knn_all` harm must exceed that floor for the rate to be evaluable. K6
  **passes** if at least one rate is evaluable and the guard removes >= 75% of the harm at *every* evaluable rate. It is
  **not evaluable** if neither rate is evaluable (KNN does not hurt on Bank, which is itself reported), and it **fails**
  otherwise.
* **K0 for Bank.** The composed `knn_all` frame must equal the frozen `detect_and_repair` frame on every numeric cell, and
  `dirty` / `knn_all` macro-F1 must equal Phase 16's recorded `dirty_macrof1` / `repaired_macrof1` for the missing-values
  rows (<= 1e-9). The guard's selected columns are reported; no prediction is attached to which they are.
* **Exploratory arm (not a criterion).** `knn_pointmass_only` (KNN only on the columns the guard routes to the median) is
  added to see whether the mechanism travels in the same direction.

---

# Results (2026-11-17; all three stages run once, no re-runs, no threshold changed)

Artifacts: `results/phase17_knn_{raw,summary,mechanism,checks,criteria,secondary_raw,secondary_summary,bank_raw,bank_summary,bank_checks}`.
Tests: `src/test_phase17.py` (11 tests). Code: `src/phase17_knn.py`.

## R1. Faithfulness (K0): PASS on Adult and on Bank

Over all 15 Adult cells the composed `knn_all` frame equals the frozen `detect_and_repair` frame on every numeric cell
(max difference 0), `dirty` and `knn_all` F1 equal Phase 16's recorded values (max difference 0), and the explicit-median
control equals `dirty` (difference 0). The guard rule selected exactly {capital-gain, capital-loss} in every cell. The same
holds for Bank (15 cells, differences 0), where the guard selected {V14, V15}.

## R2. Adult: the harm is in the capital columns (positive-class F1 recovery vs `dirty`, mean of 5 seeds)

| Arm | 5% | 10% | 20% |
|---|---|---|---|
| `knn_all` (the frozen repair) | -0.0042 | **-0.0178** | **-0.0212** |
| `knn_no_capital` (= `knn_guarded`, the same frame here) | -0.0011 | -0.0032 | +0.0002 |
| `knn_capital_only` | -0.0027 | -0.0180 | -0.0202 |
| `knn_rounded` | -0.0015 | -0.0145 | -0.0195 |
| one column at a time: capital-gain | -0.0048 | -0.0130 | -0.0133 |
| capital-loss | -0.0029 | -0.0058 | -0.0021 |
| fnlwgt / hours-per-week / education-num / age | within -0.004..+0.002 | -0.0043 / -0.0033 / -0.0015 / +0.0011 | -0.0004 / -0.0017 / +0.0024 / -0.0008 |

ROC-AUC tells the same story (`knn_all` -0.0076 and -0.0096 at 10% and 20%; `knn_no_capital` -0.0004 and -0.0002).

| Criterion | Verdict | Numbers |
|---|---|---|
| K0 | **PASS** | R1 |
| K1 (H1: capital columns are the cause) | **PASS** | removed by `knn_no_capital`: 82% (10%), 101% (20%); kept by `knn_capital_only`: 101%, 95%. All >= 75% |
| K2 (capital-gain is the largest single column) | **PASS** | harm 0.0130 vs next 0.0058 at 10%; 0.0133 vs 0.0021 at 20% |
| K3 (H2: rounding is not the cause) | **not rejected** | rounding changes recovery by 0.0034 at 10% (just over the 0.003 line) and 0.0017 at 20%. It removes at most 19% of the harm (10%) and 8% (20%) |
| K4 (H4: the harm needs capacity) | **FAIL** | shallow-GB harm is 62% (10%) and 91% (20%) of the deep-GB harm, above the 50% line |
| K5 (the general guard rule reproduces K1) | **PASS** | removes 82% and 101% |
| K6 (guard out of sample, Bank) | **NOT EVALUABLE** | see R4 |

## R3. What the mechanism table shows (no classifier involved; pooled over the 15 Adult draws)

| Column | Blanked cells imputed exactly right: KNN / median | MAE: KNN / median | Imputed values that appear nowhere in the real column |
|---|---|---|---|
| capital-gain | 49% / 92% | 2,518 / 1,166 | 48% |
| capital-loss | 64% / 95% | 245 / 95 | 33% |
| hours-per-week | 3% / 48% | 9.4 / 7.2 | 51% |
| age | 0.5% / 2.7% | 11.5 / 10.7 | 81% |
| education-num | 4% / 21% | 2.18 / 1.89 | 79% |
| fnlwgt | 0% / 0% | 91,243 / 77,240 | 99% |

Mean imputed capital-gain is 1,599 against a true mean of 1,166 (biased high by 37%). **KNN is a worse point estimate than
the median on every numeric column**, so the comment in `phase9_repair_engine.py` calling it "a genuinely different, more
informed estimate" does not hold for Adult: the standardised neighbours carry almost no information about a blanked
cell. That is a fact about the imputations; it matters to the classifier only in the capital columns (R2).

## R4. Bank Marketing (K6): NOT EVALUABLE, and the post-hoc report does not rescue the guard

The Bank GB macro-F1 noise floor is 0.0101. `knn_all` macro-F1 harm is 0.0090 (10%) and 0.0059 (20%), **below the floor
at both rates**, so by the rule fixed in K-bis K6 is not evaluable. Per-seed recoveries range from -0.027 to +0.029 on a
test set of about 1,800 rows: this run cannot see an effect of this size on Bank.

Reported, not tested (K-bis names macro-F1 as the test): on positive-class F1 the harm is above the floor (0.0172, 0.0132)
but the guard removes only 18% of it at 10% and -43% at 20% (it makes it worse). AUC harm is nil. So on Bank, by the one
metric where KNN visibly hurts, the guard (which routed V14/V15 to the median) does **not** explain the harm. Bank's
macro-F1 K6 stays "not evaluable"; I am not upgrading it either way.

## R5. Secondary arms (K4, P6)

| | 10% `knn_all` | 10% `knn_no_capital` | 20% `knn_all` | 20% `knn_no_capital` |
|---|---|---|---|---|
| deep GB (selected config) | -0.0178 | -0.0032 | -0.0212 | +0.0002 |
| shallow GB | -0.0110 | -0.0021 | -0.0194 | -0.0019 |
| Random Forest, 300 trees | -0.0003 | +0.0027 | -0.0054 (seed std 0.010) | +0.0004 |

A shallow booster is still hurt, concentrated in the same columns (`knn_capital_only` -0.0091 and -0.0158). The Random Forest
is not hurt at 10% and shows -0.0054 at 20% with a seed standard deviation about twice that, so its repair evidence
(Phase 10: `weak` / `negative`) is neither confirmed nor contradicted here.

## R6. Predictions

P1 (K0 passes): right. P2 (capital columns carry the harm both ways): right. P3 (capital-gain largest): right.
P4 (rounding does not matter): **wrong by the rule**: it is not rejected, though the effect is small and mixed (above
the line at 10%, well below it at 20%). P5 (less harm when shallow, no clean 50% cut): right in direction; the cut was
38% at 10% and 9% at 20%. P6 (the Random Forest's harm sits in the same columns but is smaller): **not testable**, its
harm is within noise.

## R7. What holds up and what does not

* **Holds:** the GB harm of the KNN repair is localised to the two zero-inflated columns. Replacing KNN by the median in
  just those two columns removes 82-101% of it (K1, K5), and KNN on only those two reproduces 95-101% of it. This is
  a paired design on one corruption draw per cell with a frozen-code faithfulness check, and the full harm is 6-7x the GB noise floor (0.003).
* **Holds, descriptive:** KNN fills those cells with values that are exactly right half as often as the median
  (49% vs 92%), about twice as far off on average, biased high, and often numbers that exist nowhere in the data (R3).
* **Not shown:** *why* GB reacts. The arms say **where** the harm is, not which property of the imputed values causes it.
  "Invented values create false split thresholds" is plausible; so is "the upward bias shifts the zero/non-zero split".
  A further arm (e.g. KNN values snapped to the nearest observed value, or a mean-shifted median) would separate them.
  It was not run.
* **Does not travel, so far:** Bank is too noisy to evaluate K6, and the positive-F1 report there points the other way.
  This is an **Adult-specific diagnosis**.
* **Not shown:** that the guard improves anything *beyond* "no worse than the median": `knn_no_capital` is about 0.000 to
  -0.003, never positive.

## R8. Consequences (the rules fixed above, applied)

* K6 is not evaluable, which the consequence rule counts as not passed. **The product is unchanged**:
  `taskclean._propose_missing` still imputes numeric cells with KNN. The finding is recorded as an Adult-specific
  diagnosis. The product already does not apply missing-value repair automatically on Adult evidence (Phase 16), so
  the exposure is the "aggressive" file, which the README already describes as containing repairs that did not clear
  the safety bar.
* **Superseded by amendment A3 below:** the user then asked for the fix, and the product now uses the median for
  point-mass columns. The bullets above are kept as written: they record what the pre-registered rule alone would have done.
* Documentation: README, review guide and the Phase 16 spec now point to this result in place of "cause untested".
* Open, for a later decision (not part of this phase): whether to route point-mass columns to the median anyway, on
  mechanism grounds, as an Adult-informed precaution. It is cheap and can only make the repair more conservative, but it
  would be a product change that the pre-registered rule did not license.

---

# Amendment A3: the product fix (2026-11-17), a decision outside the pre-registered rule

The consequence rule above said the product changes **only if** K1, K5 and K6 all pass, and K6 was not evaluable. By that
rule the product stays as it is. **The user then asked for the problem to be fixed**, so this change is a deliberate
departure from the rule, recorded as such, justified by the mechanism (R2, R3) and not by out-of-sample evidence (there
is none: R4). It can only make the repair more conservative.

**The change.** In `taskclean._propose_missing`, a numeric column whose most frequent observed value covers at least
`POINT_MASS_SHARE = 0.5` of its observed cells is imputed with the **column median** (which equals that value, since it covers at least half the cells)
instead of by KNN. Every other numeric column is unchanged: KNN (k=5, standardised), still using **all** numeric columns as
distance features, integer-like columns still rounded. The proposal's `method` text says which rule applied and why.

**Validation criteria, fixed before the shipped code path was run** (Adult, GB, 5 seeds, rates 10% and 20%, positive-class
F1 recovery against `dirty`; "old product" = the same code with the guard off):

* **F0, faithfulness.** The guard-off path reproduces Phase 17's `knn_rounded` arm exactly (F1 difference <= 1e-9 in all 10
  cells): that proves the product's KNN code equals the harness's. The guard-on frame equals the reference frame built from
  the harness functions (KNN values on the non-point-mass columns, rounded where integer-like; pipeline median on the
  rest) on every numeric cell.
* **F1, the fix works.** The shipped repair's mean recovery is >= -0.005 at **both** rates **and** it removes >= 75% of the
  old product's harm at both rates. If F0 or F1 fails, the product change is reverted and that is reported.
* **F2, Bank, report only, no-regression.** (Bank GB macro-F1; the guard selects V14/V15 there.) Shipped minus old recovery,
  mean over 5 seeds, at 10% and 20%. If the shipped repair is worse than the old by more than Bank's noise floor (0.0101)
  at either rate, that is flagged as a **caution** in the documentation. It does not undo the decision above; the
  Bank test is underpowered (R4).

## A3 results (the shipped code path, same day; run once)

Artifacts: `results/phase17_knn_product_{adult_raw,bank_raw}.csv`, `results/phase17_knn_product_checks.json`; code
`taskclean._propose_missing` (`POINT_MASS_SHARE = 0.5`, `_point_mass_columns`), tests in `test_phase17.py` (13).

**F0 (faithfulness): PASS.** With the guard off, the product's missing-value proposals reproduce Phase 17's `knn_rounded`
arm to the last digit in all 10 cells (F1 difference 0), the dirty arm equals Phase 17's (0), and the guard-on frame equals
the reference frame built from the harness functions on every numeric cell (0).

**F1 (the fix works): PASS.** Adult, GB, positive-class F1 recovery against leaving the cells blank, mean of 5 seeds:

| Rate | Old product (KNN everywhere, integer rounding) | Shipped (median for point-mass columns) | Harm removed |
|---|---|---|---|
| 10% | -0.0145 | **-0.0034** | **76%** (the line is 75%: a narrow pass) |
| 20% | -0.0195 | **-0.0014** | 93% |

Both are above the -0.005 line. ROC-AUC: -0.0063 -> -0.0007 (10%), -0.0101 -> -0.0003 (20%). Macro-F1: -0.0087 -> -0.0019
and -0.0117 -> -0.0004. Per seed the shipped repair is better than the old in all 10 cells. It is still not *positive*:
at 10% it is about one noise floor (0.003) below leaving the cells blank. The old product's "recovery" of the clean
score is therefore replaced by "about no worse than doing nothing", not by an improvement.

**F2 (Bank, report only): no caution.** Macro-F1 recovery, old -> shipped: 10%: -0.0052 -> -0.0087 (shipped worse by 0.0034);
20%: -0.0126 -> -0.0057 (shipped better by 0.0068). Both differences are within Bank's noise floor (0.0101) and have opposite
signs, so this shows no regression and no gain. Positive-class F1 is the same pattern (10%: -0.0105 -> -0.0170; 20%: -0.0252 -> -0.0133).

**What this does and does not establish.** The fix removes the Adult harm in the real code path, measured on the same
corruption draws, with the old behaviour reproduced exactly as the control. It is **validated on Adult, where it was
diagnosed**. It is not shown to help on any other dataset; on Bank it is neutral within noise. The decision to ship it
was the user's, taken outside the pre-registered rule (K6 was not evaluable), and is recorded as such.
