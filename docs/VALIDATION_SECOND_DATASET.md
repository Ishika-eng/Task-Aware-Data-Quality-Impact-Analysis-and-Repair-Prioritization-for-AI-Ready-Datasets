# Validation on a second dataset (bank-marketing)

Everything before this was measured on UCI Adult. This report asks what survives on a different real dataset,
and whether the "Calibrate on this dataset" feature gives better advice than reusing the Adult numbers.

**Short answer:** the product is robust and its detector-reliability estimates transfer well, and the Adult
prioritization transfers exactly once damage is measured with a threshold-free metric. The calibration did **not**
demonstrably improve the safe-to-repair decisions, and the dirty-upload variant made one false approval. The most
useful discovery is that **F1-based damage is unreliable on heavily imbalanced targets**, which affects the whole
project.

Code: [`src/phase13_transfer.py`](../src/phase13_transfer.py). Results: `results/phase13_*.csv`.

## Set-up

| | |
|---|---|
| Second dataset | bank-marketing (OpenML 1461): 45,211 rows, 16 features (9 categorical, 7 numeric), no missing values, no duplicates; **11.7% positive class** (Adult: 24.8%) |
| Ground truth | the Phase 7-10 protocol on the cleaned data: 12,000 complete, de-duplicated rows, 70/15/15 split (train 8,400 / test 1,800), 5 seeds, 300-tree Random Forest, repairs through the **Phase 9 engine** (a code path independent of the product layer), harm judged against a noise floor from a null experiment |
| Clean baseline | positive-class F1 0.410, macro-F1 0.676, ROC-AUC 0.912 |
| Calibration under test | the quick self-benchmark (3 seeds), run on (a) the **raw 45,211-row file** (33 s; reference 6,000 rows; noise floor 0.0092) and (b) a **dirty upload**: a 12,600-row sample with all five error types injected at 5% (32 s; reference 5,239 rows; noise floor 0.0161) |
| Regime checks | UCI Heart Disease (`heart-c`, 303 rows) and German credit (`credit-g`, 1,000 rows) through the whole product |

Four questions were fixed before running: **Q1** does the product run correctly; **Q2** do the Adult findings
transfer; **Q3** does the calibration agree with the independent ground truth; **Q4** does calibrated evidence beat
the Adult evidence transferred to this dataset, where the dangerous failure is a *false approval* (judging a repair
safe to apply automatically that the ground truth says is not).

### Deviations, disclosed

- **Metrics.** Positive-class F1 was pre-specified (as in Phases 2-9). After a first partial run showed label noise
  *raising* F1, the run was restarted recording **macro-F1** (so Q3/Q4 compare like with like, since the calibration
  measures macro-F1) and **ROC-AUC**. The AUC-based results below are therefore **exploratory**, not pre-specified.
- Per-error-type conclusions rest on 3 corruption rates and 5 (ground truth) or 3 (calibration) seeds. Only 9
  decision cells are comparable. Treat everything as directional.

## Q1 - Does the product run on a different dataset? Yes

- bank-marketing (raw, 45,211 rows): audited and calibrated in 33 s; both cleaned files have honest repair logs.
- **Heart Disease (303 rows): calibration correctly declined** (only 296 clean-ish rows; the minimum is 600) and fell
  back to the Adult evidence. It still dropped its 1 duplicate and imputed its 7 missing cells in the aggressive file.
- **credit-g (1,000 rows): calibration ran with a noise floor of 0.021**, nearly 7 times Adult's, so on small data
  it demands much larger effects before calling anything safe or harmful. Nothing was auto-repaired (no duplicates).

## Q2 - Do the Adult findings transfer? It depends on the metric

Mean damage at 20% corruption (largest first):

| Metric | Order of damage | Rank correlation with Adult (15 points) |
|---|---|---|
| Adult, positive-class F1 | label errors, feature corruption, outliers, missing, duplicates | - |
| bank-marketing, positive-class F1 | feature corruption, outliers, missing, duplicates, **label errors (negative)** | -0.00 |
| bank-marketing, macro-F1 | feature corruption, outliers, missing, duplicates, **label errors (negative)** | -0.01 |
| bank-marketing, **ROC-AUC** | **label errors**, feature corruption, missing, outliers, duplicates | **+0.74** |

**Label noise looks beneficial under F1 and is the most damaging error under AUC.** At 20%, label-error damage
per seed was (positive-class F1) -0.031, -0.058, -0.026, +0.044, -0.019 but (AUC) +0.065, +0.037, +0.046, +0.051,
+0.052. Flipped labels push a model toward predicting the rare class more often, which raises minority-class F1
while the model's ability to *rank* cases gets worse. F1 and macro-F1 depend on the decision threshold; with an
11.7% positive class that dependence swamps the damage signal. On Adult (24.8% positive) the effect did not appear.

Prioritization (the Phase 10 rules, 10% corruption, equal weights):

| Evidence | Order | Stability across 15 weight combinations |
|---|---|---|
| Adult | label errors > duplicates > outliers > missing > feature corruption | label errors first in 100%, feature corruption always last |
| bank-marketing, F1-based | outliers > feature corruption > missing > label errors > duplicates | no stable first or last place |
| bank-marketing, **AUC-based** | **label errors > missing > duplicates > outliers > feature corruption** | **label errors first in 100%, feature corruption always last** |

So the headline prioritization **transfers exactly under AUC** (exploratory) and fails under F1. The same pattern
holds for repair: feature-anomaly repair was net-harmful on Adult; on bank-marketing it looks helpful by F1
(recovery +0.0095 macro-F1) but harmful by AUC (-0.0039, harmful in 4 of 5 seeds), again agreeing with Adult.
Exact duplicate removal is the one finding that transfers under every metric: precision and recall 1.00, and it
restores the clean data exactly.

## Q3 - Does the calibration agree with the ground truth?

**Detector reliability: yes, closely.** Precision / recall at 10% corruption:

| Detector | Ground truth | Calibration, raw upload | Calibration, dirty upload | Adult (for reference) |
|---|---|---|---|---|
| Missing values | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |
| Duplicates | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 0.99 / 1.00 |
| Label errors | 0.96 / 0.30 | 0.93 / 0.29 | 0.93 / 0.25 | 0.77 / 0.28 |
| Outliers | 0.22 / 0.72 | 0.25 / 0.82 | 0.22 / 0.77 | 0.18 / 0.67 |

**Damage ordering: partly.** Spearman correlation with the macro-F1 ground truth: calibration on the raw upload
**0.70**, on the dirty upload **0.35**, Adult evidence transferred **-0.01**. Calibration captures this dataset's own
ordering far better than Adult's numbers do (but it inherits the same F1 blind spot for label noise).

**Safe-to-auto-repair decisions: no better than Adult.** The 9 comparable cells (missing, duplicates, outliers at
three rates; label and feature-anomaly repairs are never auto-applied by design):

| Source | Agrees with ground truth | False approvals | Missed safe repairs |
|---|---|---|---|
| Adult evidence, transferred | 7 / 9 | 0 | 2 |
| Calibration, raw upload | 7 / 9 | 0 | 2 |
| Calibration, dirty upload | 7 / 9 | **1** | 1 |

Ground truth says duplicate removal is safe at all rates, missing-value imputation safe at 5% and 10% but not 20%
(recovery -0.010, harmful in 4 of 5 seeds), and outlier repair never safe (repair precision 0.22-0.54). Both Adult and the
raw calibration miss the two safe imputation cells (they are conservative). **The dirty-upload calibration
approved imputation at 20%, which the ground truth shows is harmful** - a false approval. Its noise floor (0.0161)
was twice the ground truth's (0.0074), so harm the ground truth saw fell below its threshold.

## Q4 - Is calibrated evidence better than transferred Adult evidence?

**Not demonstrated.** For the decision that matters (what to apply automatically) calibration tied with Adult
on a raw upload and was worse on safety for the dirty upload. It was clearly better at estimating detector
reliability and the dataset's own damage ordering. With 9 cells and 3 seeds this is weak evidence either way.

## What this changes

1. **F1-based damage is unreliable on imbalanced targets.** The self-benchmark's impact numbers (macro-F1) would
   have told a bank-marketing user that label noise is harmless. *Recommended:* also measure ROC-AUC (one-vs-rest for
   multi-class) in the self-benchmark and warn when F1 and AUC disagree. Not yet implemented.
2. **Approvals from a noisy calibration are weak evidence.** A large noise floor makes "no material harm" easy to
   satisfy. *Recommended:* refuse to certify a repair as safe when the noise floor is large (for example above 0.01)
   and offer 5 seeds. Not yet implemented.
3. **Repair harm is dataset- and metric-dependent.** The "feature-anomaly repair is net-harmful" statement is not
   universal; the never-auto guard is justified by its low precision (repair precision 0.25-0.42, detector precision 0.28), not by universal harm. The product's
   limitations text now says so.
4. The Adult findings are **not** safe to quote as general without stating the metric: label errors first and feature
   corruption last holds under AUC on both datasets, and not under F1 on the imbalanced one.

## Not tested

Only one extra dataset with a full ground truth; a binary target only; Random Forest only (Logistic Regression
validation is still pending); random (not structured) injected errors; and 3 calibration seeds. A second dataset is
not "general"; this is one more data point.
