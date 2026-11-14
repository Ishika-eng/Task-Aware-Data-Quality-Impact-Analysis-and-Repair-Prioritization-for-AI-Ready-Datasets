# Phase 15: bivariate and multivariate outlier detection (flag-only)

Written **before** any code or validation run. Thresholds below are fixed in advance; validation results do not change
them. Anything that has to change after results are seen is appended as a visible **Amendment**, never edited in place.
Phases 1-11, 13, 14 and their `results/` files are untouched; everything here is additive.

## Why this phase exists

| What exists | Limit |
|---|---|
| `detect_outliers_iqr` (Phase 5) | **Univariate**: one column at a time. A value that is typical for its column but wrong for its row (a 19-year-old earning a senior salary, a 3-month loan of the largest amount) is invisible. |
| `detect_feature_corruption_crossfeature` | Cell-level, Random-Forest-based, **off by default**: precision 0.28, recall 0.26, repair status `negative` at every rate. Its job is cells that do not fit their row's other values, including categorical ones. It is not a numeric distance method. |
| nothing | No bivariate relationship check, no multivariate distance. |

This phase adds **one detector run at two scopes**, and nothing that edits data.

## Data-eligibility pre-check (done before designing the validation; not a detector result)

Spearman |rho| between numeric columns with more than 10 distinct values, computed on the raw files in `data/`:

| Dataset | Numeric columns | Largest |rho| | Pairs with |rho| >= 0.5 |
|---|---|---|---|
| Adult | 6 (age, fnlwgt, education-num, capital-gain, capital-loss, hours-per-week) | 0.17 | **0** |
| German credit | 3 (duration, credit_amount, age) | 0.62 (duration x credit_amount) | 1 |
| Heart (Cleveland) | 5 | 0.43 (oldpeak x thalach) | 0 |
| Bank Marketing | 7 | 0.99 (V14 x V15, a near-duplicate) | 1, but redundant |
| **Diabetes** (scikit-learn bundled, added by Amendment 1) | 9 | 0.88 | **6** (0.5 <= rho < 0.98) |
| **Breast Cancer Wisconsin** (scikit-learn bundled, Amendment 1) | 30 | 1.00 | **146** (9 more are >= 0.98 near-duplicates) |
| Wine (scikit-learn bundled, Amendment 1) | 13 | 0.88 | 12, but only 178 rows (below the 200-row minimum) |

Consequences, stated up front rather than discovered later:
* **Adult, the project's anchor dataset, cannot validate a bivariate detector**: there is no correlated pair, and two
  of its numeric columns (capital-gain, capital-loss) are zero-inflated. The same is true of the multivariate detector in
  any meaningful sense: columns that are nearly independent give a joint distance that adds nothing to the marginals.
  Adult is therefore the **negative control** in this phase, not the evidence base.
* The evidence base is **synthetic data with known correlation** (ground truth by construction) plus the **real
  datasets where correlated pairs exist**: German credit, and, after Amendment 1, **Diabetes** and **Breast Cancer
  Wisconsin**. The rest are reported honestly as "declined" or limited.
* All are small (n of 442, 569 and 1,000 rows). This is still a narrower evidence base than the earlier phases had.
  It is stated again in section E.

## A. Detector (`src/joint_outliers.py`): flag-only

Applies to numeric columns only. Rows are used listwise (a row needs all columns of the scope present; missing cells
are the missing-value detector's territory). Minimum 200 usable rows, otherwise the detector **declines** with a data note.

**Column eligibility.** More than 10 distinct values; not degenerate (IQR = 0, or one value covers >= 50% of non-null
rows, which is what makes `capital-gain` unusable); not constant; not identifier-like (the Phase 14 `id_like` rule).

**Transform.** Gaussian copula: replace each value by `norm.ppf((rank - 0.5) / n)` using average ranks for ties. This
removes skew and heavy tails from the marginals so that one distance cut-off is meaningful, and it is monotone, so a
relationship that is monotone stays monotone. It cannot represent a non-monotone relationship (see E).

**Distance.** Robust squared Mahalanobis distance from a Minimum Covariance Determinant fit
(`sklearn.covariance.MinCovDet`, `random_state=seed`, default support fraction), fitted on at most **10,000** rows
(deterministic sample) and used to score all usable rows. If the fit is singular the detector declines with a data
note; there is no silent fallback to a non-robust covariance.

**Cut-off.** A row is flagged when its squared distance exceeds `chi2.ppf(1 - alpha, df)`, where `df` is the number of
columns in the scope. The cut-off has a stated null rate (alpha) under normality, which is why this is the primary
method. It will not hold exactly on real data; the clean-data false-positive rate is measured, not assumed (section C).

**Two scopes, one method.**
1. **Multivariate.** All eligible numeric columns together, needs >= 3 columns, `alpha = 0.001`.
2. **Bivariate.** Every eligible pair with 0.5 <= |Spearman rho| < 0.98 (>= 0.98 is a near-duplicate, left to the Phase 14
   `redundant` check; a singular pair is skipped, not repaired). At most the **20** strongest pairs are tested.
   `alpha = 0.001 / P`, where P is the number of pairs actually tested (Bonferroni, family-wise 0.1%), `df = 2`.
   Bivariate exists as its own scope for a reason this project has already met: **one disturbed pair is diluted in
   a many-column distance** (the PCA feature detector had ~25x lower recall than the targeted cross-feature predictor
   for exactly this reason). The pair gate (|rho| >= 0.5) is what stops it testing relationships that do not exist.

**Output.** One record per flagged row: row id, scope (`multivariate` or the pair name), squared distance, the cut-off, the
**column that contributes most** and the direction. Attribution uses the Gaussian conditional residual of each column
given the others, which has a closed form from the precision matrix: `z_j = (Sigma^-1 (x - mu))_j / sqrt((Sigma^-1)_jj)`,
the column with the largest `|z_j|`. The record says whether the value is higher or lower than the other columns imply. It
**never proposes a replacement value.**

**Relation to the univariate outlier detector.** Each flagged row carries `also_univariate`: whether `detect_outliers_iqr`
(k = 1.5) already flags it. The headline rate counts **joint-only** rows (flagged here, not flagged there), so the two issue
types do not double-count the same row.

**Policy: flag-only, always.** Issue key `joint_outliers`, shown as "Unusual combination of values". It is added to
`NEVER_AUTO_APPLY`; every record has action `flagged_only` and appears in the Review queue. No repair is built, for two
reasons: the existing evidence says repairing cell anomalies has been **harmful** (feature anomalies: `negative` at all three
rates, precision 0.24-0.42), and a rare combination is often a genuine record that only a person can judge.
Because nothing is repaired, **no repair-gate evidence is produced or claimed** for this issue.

## B. Comparison baselines (validation only, never in the product)

* **Univariate IQR baseline**: the existing `detect_outliers_iqr`, k = 1.5, on the same data. This is what the new detector
  must add to.
* **Isolation Forest benchmark**: `IsolationForest(n_estimators=300, random_state=seed)` on the same copula-transformed
  columns, scored on the same rows. It has no calibrated cut-off, so it is compared at an **equal flag budget**: it flags
  exactly as many rows as the main detector did. It is a ranking-quality check ("did the simple, interpretable method
  leave detection quality on the table?"), not a candidate for the product.

## C. Validation design (`src/phase15_validation.py`)

Seeds 42-46. Detection and attribution are scored against an injection log; no model-recovery measurement is made (there
is no repair to recover from). The real product path (`audit_dataset(..., detectors={"joint_outliers"})`) is used.

**Datasets.**
* **Synthetic, known structure** (n = 5,000, 6 numeric columns, block-correlated Gaussian with rho in {0.0, 0.5, 0.8}), plus a
  **skewed** version (each column exponentiated to a lognormal) to test the copula step. rho = 0.0 is a **negative control**:
  swapped values are statistically indistinguishable from the rest by construction, so recall near 0 there is the
  *correct* answer and is not counted as a failure.
* **Real, with correlated pairs (evidence):** German credit (3 columns, 1 pair); **Diabetes** (9 columns, 6 pairs);
  **Breast Cancer Wisconsin** (30 columns, 146 eligible pairs, of which the 20 strongest are tested). Diabetes and Breast
  Cancer are loaded from the copies bundled with scikit-learn 1.9.1 (`load_diabetes`, `load_breast_cancer`), so there is no
  download and no new file in `data/`; the validation records the scikit-learn version.
* **Real, reported as declined / limited (gating checks):** Heart (no eligible pair, multivariate only), Bank Marketing
  (one redundant pair), Wine (178 rows: both scopes must decline on the 200-row minimum), Adult (negative control: bivariate
  must decline).

**Injection** (`src/phase15_injectors.py`, not registered in `corruption.CORRUPTORS`, so frozen phases and selfbench are
unchanged). Row rates **2%, 5%, 10%** (a "rate" above ~10% is not an outlier population), equal mix of three kinds, which
is an **assumption**, not a measurement of real data:
1. `swap_one`: one numeric cell takes the value of the same column in another random row. Marginally typical, jointly wrong.
2. `swap_block`: two or more numeric cells of a row are replaced together by those of one other random row. Each cell is plausible
   alone and the block is internally consistent, but it does not fit the row's untouched columns.
3. `inflate`: one cell pushed 4 standard deviations beyond the column median, in the direction it already lies **(amended from
   "multiplied by 3": see Amendment 1)**. A **control**: visible to the univariate check, expected to be caught by both.
The log records row, column(s) and kind, and whether the injected value lies **inside the univariate IQR fences** (k = 1.5),
i.e. whether it is *marginally invisible*.

**Measured:** row-level precision and recall per kind and per scope; **recall on marginally-invisible injections** (the number
that justifies the phase); attribution accuracy; the false-positive rate on **clean** data (no injection) for every dataset;
the same quantities for the two baselines; for the Isolation Forest, precision and recall at the equal budget; and, on German credit
only, the overlap with `detect_feature_corruption_crossfeature` (informational).

**Floors for the product.** The false-positive floor used for "Attention / Good" status on this issue is the **largest
clean-data flag rate observed across the eligible validation datasets** (conservative). Status stays "indicative", as for the
other statistical detectors. Written to `results/phase15_baseline_rates.json`.

## D. What would count as a failure (reported, not hidden)

* **F1.** Clean-data false-positive rate above **1% of rows** on any eligible validation dataset.
* **F2.** Recall on marginally-invisible `swap_*` injections below **50%** on synthetic data at rho = 0.8 (rates 2% and 5%).
  (At rho = 0.5 it is reported only; at rho = 0.0 it is expected to be near 0.)
* **F3.** Precision below **0.80** at rates up to 5% on synthetic data at rho >= 0.5.
* **F4.** Attribution: the top-contributing column is an injected column for fewer than **70%** of correctly flagged `swap_one`
  rows at rho >= 0.5.
* **F5.** Gating: on Adult the bivariate scope must decline, and the multivariate scope must flag at most 1% of clean rows;
  on Wine (178 rows) both scopes must decline on the 200-row minimum **(Wine added by Amendment 1)**.
* **F6.** The Isolation Forest, at equal budget, beats the main detector on recall in **60% or more** of the dataset x rate cells.
  This would not be hidden or switched silently: it is recorded and, if the simple method is clearly worse, handled by a visible
  amendment.

**Consequence rules, fixed now.**
* If F1 or F5 fails for the multivariate scope on **two or more** datasets, that scope ships **off by default** (like the feature
  anomaly detector), documented as a negative result. The bivariate scope is decided separately on its own evidence.
* If F2 fails, the detector is described as unable to find marginally-invisible anomalies, which removes its reason to exist; it
  stays available only as an off-by-default tool.
* F3 or F4 failing does not by itself switch the detector off; the limit goes in the README and the Review page text.
* Any threshold change after results are seen is a visible Amendment and is **not** used to declare the original criteria met.

## E. Known limits, stated up front

* **Narrow real-data evidence.** Three real datasets with eligible pairs (German credit, Diabetes, Breast Cancer), all small and
  all curated, clinical or financial. Breast Cancer's pairs are mostly geometric near-identities (radius, perimeter, area), an
  easier case than typical business data. The synthetic data is where the structure is ours. A strong result on either is a
  statement about the method, not about the data you will upload.
* **Numeric columns only.** An unusual combination that involves a categorical column (age 12 and "Divorced") is the feature
  anomaly detector's job and is not covered here.
* **Monotone dependence only.** The copula transform cannot see a non-monotone relationship (a U-shape between two columns), and a
  correlation low in rank terms can still hide a strong dependence.
* **Rare is not wrong.** A flag means "unusual combination", not "error". This is why nothing is repaired.
* **Complete rows only.** Rows with a missing cell in the scope are skipped for that scope; heavy missingness lowers coverage, and
  the audit reports how many rows were usable.
* **Clean-data rate is empirical.** The chi-square cut-off assumes normality after the copula; real data will not satisfy it exactly,
  which is why F1 exists.
* **The injected mix is an assumption.** Equal shares of `swap_one`, `swap_block` and `inflate` are the author's choice.
* Calibration (`selfbench.py`) is **not** extended to this type; calibrated audits keep the Phase 15 evidence and say so.

## F. Deliverables

`src/joint_outliers.py` (detector); `src/phase15_injectors.py`; `src/phase15_validation.py` (CLI with `--full`, `--only`);
`src/test_phase15.py` (real assertions, no vacuous ones); `results/phase15_*.csv/json`; integration into `taskclean.py` (new issue
key `joint_outliers`, listed in `ISSUES`, `PROBLEM_TEXT`, `NEVER_AUTO_APPLY`, data notes for declines), the API payload, and the
Review page; README and `docs/REVIEW_GUIDE.md` updated with the measured numbers, including any negative ones.

Planned unit tests: planted joint anomalies are found and a null (rho = 0) dataset is not over-flagged; a degenerate column is
excluded; the near-duplicate pair is skipped; the detector is deterministic for a fixed seed; the input is never mutated; a singular
fit declines with a note; fewer than 200 usable rows declines; a flagged record never carries a proposed value; and the product never
auto-applies the issue.

## Amendment 1 (made before any code or validation run: more real datasets, and a scale-free control)

**Why.** The pre-check above showed Adult and Heart cannot test the bivariate scope (no pair reaches |rho| 0.5) and German
credit offers a single pair. The user asked for another real dataset where both scopes can be tested. The datasets were chosen
by **data eligibility only** (at least 200 rows, at least 3 eligible numeric columns, at least one pair with 0.5 <= |rho| < 0.98),
looking at correlation structure and **not** at any detector output, because the detector does not exist yet.

**What was added.**
* **Diabetes** (442 rows, 9 eligible numeric columns, 6 eligible pairs): real clinical measurements, moderate correlation, a
  realistic case for both scopes.
* **Breast Cancer Wisconsin** (569 rows, 30 numeric columns, 146 eligible pairs, 9 near-duplicates skipped): a high-dimensional,
  heavily dependent, non-Gaussian real dataset. It is the **hardest test of the chi-square cut-off** (criterion F1) and of the
  20-pair cap, so a clean-data false-positive rate above 1% there is a plausible outcome and would be reported, not tuned away.
* **Wine** (178 rows) as a **gating check only**: it has correlated pairs but is below the 200-row minimum, so both scopes must
  decline with a data note (added to criterion F5).

**Consequence for the failure criteria.** F1 and F3 apply to every dataset in the evidence group (synthetic, German credit,
Diabetes, Breast Cancer). The consequence rule "multivariate scope ships off if F1 or F5 fails on two or more datasets" now counts
over that larger group. Nothing else in section D changed.

**Control definition changed.** `inflate` was "one cell multiplied by 3". Diabetes ships mean-centred (values near 0), where
multiplying by 3 barely moves anything, so the control would silently stop being visible to the univariate check on that
dataset. It is now **"one cell pushed 4 standard deviations beyond the column median, in the direction it already lies"**, which is
scale-free and identical on every dataset. The log still records whether each injected value falls inside the univariate IQR
fences, so any injection that is not univariate-visible is counted as such rather than assumed.

## Amendment 2 (made after the detector's unit tests exposed a failure on clean Adult, BEFORE the validation was run)

**What was seen.** While writing the tests, the pre-registered detector flagged **19.5% (seed 0) and 51.5% (seed 42)** of
*clean* Adult rows in the multivariate scope, against a nominal 0.1%, and the number moved with the random fit sample. A
diagnostic (no detector change) found the mechanism: `hours-per-week` has **47.2% of its rows at one value (40)**, just under the
50% degeneracy rule in section A, so it was eligible. Under average-rank tie handling that mass becomes a **point mass** in
normal-score space; the robust covariance fits the tied half of the data, its variance along that axis collapses, and every
other row is far away. That one column drove 100% of the flags. Without it the same three other columns flagged 0.05-0.09% of
clean rows, matching the nominal rate.

**What this amendment does NOT do.** It does not edit section A, and the original design is **still run in full and judged against
F1-F6 exactly as written**. The consequence rules in section D are applied to the original design's results. Run it first; its
results are not re-labelled by anything below.

**The second design (`ties="random"`).** In the rank transform, tied values are ordered at random (seeded) instead of sharing an
average rank, a randomised probability-integral transform. The marginal becomes exactly normal, so no column is a point mass. Rows
that share a value then get different scores: that is noise by construction, so power on heavily tied columns falls, but the null
stays calibrated. Nothing else changes: same eligibility rules, same cut-offs, same alphas.

**Honesty about independence.** This fix was motivated by an observation on **Adult**, the negative-control dataset. Adult is
therefore **not independent evidence for the amended design**. The independent checks of the amended design are the datasets it was
*not* fitted to: the synthetic data, German credit, Diabetes, Breast Cancer, Heart and Bank Marketing. Results are reported for both
designs side by side with a `design` column. If the amended design passes where the original fails, that is recorded as "an amended
design passes", never as "the criteria were met".

## Amendment 3 (implementation note, made before the validation was run)

Section C says the real product path (`audit_dataset(..., detectors={"joint_outliers"})`) is used. The validation calls
`detect_joint_outliers` directly instead, because on the 45,000-row datasets the product path adds per-audit overhead (label-noise
and feature-relevance steps are skipped, but the prepare/encode steps are not) that does not change the flags. To keep the spec's
intent, an equivalence test (`test_phase15.py`) asserts that the product path flags exactly the same rows as the direct call on the
same frame and seed, once the issue is integrated. Everything else in section C is as written.

## Results (written after both validation runs; `results/phase15_*`, seeds 42-46, scikit-learn 1.9.1)

Run twice, identical except for tie handling: `original` (pre-registered, average ranks) and `amended` (Amendment 2, random
tie-breaking). Criteria are the ones in section D, evaluated mechanically (`phase15_*_criteria.json`).

| Criterion | Original | Amended |
|---|---|---|
| F1 clean false-positive rate <= 1% | **FAIL**: Adult 46.6%, Breast Cancer 29.2%, Diabetes 8.4% (multivariate); Breast Cancer 9.8%, Diabetes 1.8% (bivariate) | **FAIL**: Breast Cancer 29.9%, Diabetes 7.2%; bivariate 9.9% / 1.8%. Adult fixed (0.40%) |
| F2 recall on marginally-invisible swaps at rho 0.8 >= 50% | **FAIL**: 10-14% | **FAIL**: 10-14% (identical) |
| F3 precision >= 0.80 at rates <= 5%, rho >= 0.5 | **FAIL** in one cell (skewed rho 0.5 at 2%: 0.74); 0.88-0.98 elsewhere | same |
| F4 attribution >= 70% | **FAIL**: multivariate 96%, bivariate 51% (about chance) | same |
| F5 gating (Adult, Wine) | **FAIL** (Adult multivariate 47-53%) | **PASS** (Adult 0.41%; Wine declines) |
| F6 Isolation Forest beats the detector in >= 60% of cells | pass (14%) | pass (3%) |

**Consequence rules applied.** Multivariate scope fails F1/F5 on two or more datasets, so it ships **off by default**
(both designs). F2 fails, so the detector "cannot find marginally-invisible anomalies" and stays available only as an
off-by-default tool. The bivariate scope fails F1 on Breast Cancer and Diabetes. F3/F4 failing does not by itself switch it
off; the limits go in the README and the Review text.

**What the numbers mean.**
* Calibrated on synthetic Gaussian data (0.07% of clean rows flagged, precision 0.95-0.98 at rho 0.8), not on real data.
* A random swap in a correlated pair is mostly jointly plausible. The one-pair cut-off's theoretical power is 25% at rho 0.9
  (the unit test asserts the implementation matches it); recall of 10-14% at rho 0.8 is the measured counterpart.
* **Adult's high original recall (0.46) was an artifact:** the original design flagged about 16,000 of 45,000 rows. Amended:
  recall 0.03, precision 0.44, about 147 rows flagged. Adult is not independent evidence for the amended design (Amendment 2).
* **Bivariate attribution is structurally ambiguous.** In a two-column scope the standardised residuals satisfy
  z_a^2 - z_b^2 proportional to a^2 - b^2, so the "top contributor" is simply the more extreme of the two, not the column that
  was disturbed (measured 51% = chance; unit-tested). In a three-column scope with an asymmetric dependency it named the disturbed
  column about 57-62% of the time (chance 33%). The 96% multivariate figure in F4 is a selection effect: only rows with an extreme
  injected value are flagged.
* The rank transform bounds the influence of any single extreme value (about +-3.6 at n = 3,000), so a lone huge value in one
  column is left to the univariate detector. Intended, and unit-tested.
* German credit (informational): 77% of the rows this detector flagged were also flagged by the cross-feature anomaly detector,
  which has 53% recall of the injected rows there against this detector's 13%.

**Diagnostic, not saved to a file (one-off):** perfectly Gaussian data with each dataset's own correlation matrix and size
gave a flag rate of 2.5% (Breast Cancer shape, n=569, d=30) and 0.5% (Diabetes shape, n=442, d=9), against 29.4% and 7.2% on the
real data. So the small-sample behaviour of the MCD cut-off explains a little of Breast Cancer's excess, and the rest is
non-Gaussian dependence that a per-column rank transform cannot remove. This is a diagnosis by elimination; no specific
alternative was tested.

**Product integration.** `joint_outliers` is in `ISSUES`, `PROBLEM_TEXT` and `NEVER_AUTO_APPLY`, is **off by default**
(`audit_dataset(..., include_joint_outliers=True)`, `--joint-outliers`, an "experimental" switch on the Start page), uses the
amended design (random tie-breaking), is flag-only (tier `loose`, no proposed value; a human override changes nothing),
reports status "Evidence-limited" (never "Good"), counts only joint-only rows, and declines visibly (data note plus policy reason)
when there is too little data. The Phase 15 clean-flag-rate floor in section C was **not** used for the quality status: the
conservative floor (the largest clean rate, 30%) would make every rate look "Good", so the status is "Evidence-limited" instead.
Calibration (`selfbench.py`) does not cover this issue.
