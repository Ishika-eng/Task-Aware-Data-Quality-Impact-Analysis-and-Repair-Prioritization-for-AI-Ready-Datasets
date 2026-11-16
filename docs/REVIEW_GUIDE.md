# TaskClean: review study guide (from scratch)

Every number below is taken from the repo (`README.md`, `docs/VALIDATION_SECOND_DATASET.md`, `results/`, `src/`).
Where something is an inference or was not tested, it says so. Say the same in the review: it is the strongest position.

Contents: 0 Pitch · 1 Problem · 2 Data · 3 Concepts you must own · 4 The pipeline, phase by phase · 5 The product ·
6 Calibration · 7 Second dataset · 8 Honest limits · 9 Question bank · 10 Number cheat sheet · 11 Demo script

---

## 0. The pitch

**30 seconds.** Most data-cleaning tools ask "is this data wrong?". TaskClean asks "is this problem actually hurting my
ML task, can it be repaired safely, and is it worth it?". We injected five kinds of errors into clean data one at a time,
measured how much each one hurts a model, built detectors and repair methods that never see the ground truth, and
measured whether repairing helps or hurts. A repair is applied automatically only if it showed zero harm and at least
0.95 precision in testing, and on Adult evidence it must pass under two model families (Random Forest and gradient
boosting). Under the Random Forest alone that was exact duplicate removal only; gradient boosting did not confirm it at
every rate, so on Adult evidence nothing is applied automatically and duplicate removal is a one-click proposal. We also
report where our own ideas failed: one detector is inverted, one repair makes things worse, and several numbers turned
out to depend on the model (Section 7b).

**The one-sentence thesis.** *Detected is not the same as harmful, and harmful is not the same as safely repairable.*

**The five findings to be able to recite**

1. Label errors hurt the most (F1 loss 0.065 at 20% with a Random Forest; about 0.03 with gradient boosting, so the
   ordering is robust and the size is not), and they are the hardest to repair automatically.
2. Duplicates barely hurt (loss smaller than its own noise), and removing them did no harm with the Random Forest.
   With gradient boosting it failed the strict safe-auto gate at 5% and 20% (noise, not harm), so it is now a proposal.
3. Feature-corruption repair makes things worse at every rate (-0.007 F1 at 10%), even though the detector finds it.
4. The label-error detector is inverted: its flagged rate falls as real corruption rises (r = -0.95). So a low
   label-error rate must not be read as reassurance, and no single "readiness score" is justified.
5. F1 is unreliable on imbalanced data. On a second dataset (11.7% positive class) label noise *raised* F1 while
   ROC-AUC fell. ROC-AUC fixed it, and the product now measures both.

---

## 1. The problem and research question

**Problem.** Real datasets contain missing values, duplicates, outliers, wrong labels and misaligned features. Standard
practice is to detect everything and clean everything. Two issues:

- Not every defect matters for the model. Cleaning costs effort.
- Automatic repair can make data *worse* (it overwrites correct values, or imputes bad guesses).

**Research question.** For a given ML task, which data-quality problems actually cost performance, which repairs are
safe to automate, and how confident can we be?

**What "task-aware" means.** Damage is measured by the *model's F1 on an untouched clean test set*, not by a statistic
about the data. Same dirty rate, different effect depending on error type.

**What "AI-ready" means here.** A *multidimensional* report (observed rate, false-positive floor, evidence status,
after-cleaning rate) per issue. Not a single number, because Phase 11 showed a single number is not justified.

---

## 2. The data

| Item | Value |
|---|---|
| Dataset | UCI Adult (Census Income), loaded through OpenML (`fetch_openml("adult", version=2)`) |
| Task | binary classification: does a person earn `>50K` or `<=50K` per year |
| Rows | 45,222 after dropping rows with natively missing values (about 7% were missing, concentrated in workclass, occupation and native-country) |
| Class balance | about 24.8% `>50K` (so a model that always says `<=50K` is about 75% accurate, which is why accuracy is a poor metric) |
| Features | 14: 6 numeric (`age`, `fnlwgt`, `education-num`, `capital-gain`, `capital-loss`, `hours-per-week`), 8 categorical (workclass, education, marital-status, occupation, relationship, race, sex, native-country) |
| Split | 70 / 15 / 15 train / validation / test, **stratified** on the label (every split keeps the class ratio). Train is 31,655 rows |

**Why drop the natively-missing rows?** The core experiment needs a *zero-defect starting point*. If we inject exactly
10% missing values, the result must be exactly 10% and we must know where. Native missingness would confound that. It is
a design choice, not a judgement that missingness is unimportant. (Cost: selection bias, since people with complete
records may differ. Listed in limitations.)

**Why stratified?** With a 25% minority class, a random split could shift the ratio and change F1 for reasons unrelated to
the model. Stratification removes that source of variation.

**The rule that keeps the experiments honest.** Corruption and repair are applied to **training data only**. The clean
test set is never corrupted or repaired. Damage is therefore "how much worse does a model trained on dirty data do on clean
data".

---

## 3. Concepts you must own

### 3.1 Supervised classification, and the two models

- **Supervised classification:** learn a function from features to a known label using labelled examples.
- **Random Forest (the model used throughout, 300 trees).** An ensemble of decision trees. Each tree is trained on a
  *bootstrap sample* (rows drawn with replacement) and, at each split, considers only a *random subset of features*
  (for classifiers, the default is the square root of the feature count). Predictions are averaged across trees
  (`predict_proba`) or voted. Averaging many de-correlated high-variance trees gives a low-variance model.
  A single deep tree overfits; the forest does not, to the same degree.
- **Decision tree split:** chooses the feature and threshold that most reduce impurity (Gini by default).
- **Logistic Regression:** a linear model, `P(y=1) = sigmoid(w.x + b)`. It is the simple baseline. F1 = 0.662 versus
  Random Forest 0.680. It needs scaled inputs; trees do not (we share one pipeline anyway, so both see identical inputs).
- **Why only Random Forest for the experiments?** To isolate the effect of data quality, one model was fixed. It is
  strong on tabular data and has few settings. The consequence, stated openly: findings are for Random Forest. **Phase 16
  re-measured the key results with gradient boosting (Section 7b); Logistic Regression validation is still not done.**

### 3.2 Preprocessing (`baseline.py`)

One shared `ColumnTransformer`:

- Categorical: impute most frequent, then **one-hot encode** (`handle_unknown="ignore"` so a category seen only at
  test time does not crash).
- Numeric: impute median, then `StandardScaler`.
- Imputation is a no-op on clean data, but needed later because Random Forest cannot accept NaN.

Why one-hot rather than integer codes: integer codes invent an order (Private = 1 < State-gov = 2) that does not exist.

### 3.3 Metrics (this is where teachers dig)

Confusion matrix for the positive class `>50K`:

```
                    predicted >50K     predicted <=50K
actual >50K              TP                 FN
actual <=50K             FP                 TN
```

| Metric | Formula | Meaning | Use here |
|---|---|---|---|
| Accuracy | (TP+TN)/all | share correct | misleading at 75/25 balance |
| Precision | TP/(TP+FP) | of those we called >50K, how many were | |
| Recall | TP/(TP+FN) | of the real >50K, how many we caught | |
| **F1** | 2PR/(P+R) | harmonic mean of precision and recall | the main damage metric (positive class) |
| **Macro-F1** | mean of per-class F1 | treats both classes equally | added for the product's self-benchmark |
| **ROC-AUC** | area under TPR vs FPR | probability a random positive is ranked above a random negative | threshold-free |

- Harmonic mean punishes imbalance between P and R (high P with near-zero R gives a low F1).
- **F1 depends on a decision threshold** (0.5 on the probability). **ROC-AUC does not.** This difference is the core of
  the second-dataset finding (Section 7).
- Clean baseline: Random Forest accuracy 0.851, F1 0.680, ROC-AUC 0.903.

### 3.4 Variance, seeds, paired differences, noise floor

- A Random Forest is random (bootstrap, feature subsets), so one run is one draw. Five seeds (42 to 46) give a mean and a
  standard deviation.
- **Paired difference:** `damage(error, rate, seed) = baseline_f1(seed) - dirty_f1(error, rate, seed)`. Subtracting the
  same seed's clean run cancels training noise that has nothing to do with the corruption.
- **Noise floor = 0.003.** Baseline F1 itself varies by about 0.0025 across seeds. Any effect smaller than 0.003 is not
  distinguishable from training noise. This is why duplicates, whose damage is below its own std, count as "about zero".
- **Sign agreement:** a repair is "positive" or "negative" only if at least 80% of the 5 seeds agree on the sign *and*
  the mean magnitude is at least 0.003. Otherwise "weak".

### 3.5 Cross-validation and out-of-fold predictions

k-fold CV splits the data into k parts, trains on k-1, predicts the held-out part, and repeats. The collected predictions
are **out-of-fold**: every row is predicted by a model that never saw it. The label-error and feature-corruption
detectors rely on this. Predicting a row with a model trained on it would just memorise the (possibly wrong) label.

### 3.6 Missingness types

- **MCAR** (missing completely at random): independent of everything. *What we inject.*
- **MAR** (at random given other columns) and **MNAR** (depends on the missing value itself): common in real data.
  Our results for missing values apply to MCAR. Listed as a limitation.

### 3.7 Outlier rules

- **Tukey / IQR fences:** flag values outside `[Q1 - k*IQR, Q3 + k*IQR]`, k = 1.5 for "flag", k = 3.0 for "auto-repair".
- **Degenerate IQR:** `capital-gain` is about 92% zeros, so Q1 = Q3 = 0 and IQR = 0. The fence collapses to [0, 0] and
  would flag every non-zero value. Fix: for such columns, fall back to percentile bounds (0.5% / 99.5%).
- **Top-coding:** about 6% of people with capital gains are capped at exactly 99,999, a real value. So an injected extreme
  value is indistinguishable from a legitimate one on that column. This explains the low outlier precision (0.18). It is
  a property of the data, not a tuning failure.

### 3.8 Correlation

- **Pearson r:** linear association. **Spearman:** rank-based, robust to monotone non-linearity and outliers.
- A p-value says how surprising r would be if there were no relationship. With **3 points per error type**, any r is
  directional evidence only. The pooled r over 15 points was -0.24 (p = 0.39: not significant).

### 3.9 Min-max normalisation

`x' = (x - min) / (max - min)`, scaling to 0..1. Used in Phase 10 so impact, risk and repairability, which have different
units, can be combined.

---

## 4. The pipeline, phase by phase

```
Phase 1-2   clean baseline            Phase 7   impact of each error
Phase 3-4   corruption engine         Phase 8   oracle repair (ceiling)
Phase 5     blind detectors           Phase 9   real automated repair
Phase 6     quality report            Phase 10  prioritisation
                                      Phase 11  AI-readiness
                                      Phase 13  second-dataset validation
```

### Phase 1-2: clean baseline
Load, drop missing rows, 70/15/15 stratified split, train both models, evaluate on test.
**Result:** RF F1 0.680, accuracy 0.851, ROC-AUC 0.903; LR F1 0.662. Every later "damage" is relative to this.

### Phase 3-4: controlled corruption (`src/corruption/`)
One error type at a time, at rates 5 / 10 / 20%, with a **ground-truth log of every changed cell** (row, column, original,
corrupted). This is what lets us grade detectors and repairs later.

| Error | How it is injected |
|---|---|
| Missing values | blank `rate` of all cells uniformly at random (MCAR) |
| Label errors | flip `rate` of labels to the other class |
| Duplicates | append exact copies of `rate` of rows (sampled without replacement, so a clean "10% duplicated once") |
| Outliers | pick `rate` of rows, push one random numeric column by +/- 6 standard deviations |
| Feature corruption | pick `rate` of cells and shuffle their values among themselves within the same column. Every value stays plausible, but the row stops making sense (a broken join) |

Why cell-level for feature corruption: with 14 columns, "shuffle 5% of columns" and "10% of columns" both round to one
column, so the rates would be indistinguishable.

### Phase 5: blind detectors (`src/detect.py`)
They see only the dirty data, like a real system. Graded against the ground-truth log.

| Detector | How it works |
|---|---|
| Missing | `isna()` (exact) |
| Duplicates | `duplicated()` (exact) |
| Outliers | IQR fences, with the percentile fallback above |
| Label errors | 5-fold stratified CV Random Forest; flag a row if the out-of-fold prediction **disagrees** with the label and the model is at least 0.90 confident. A lightweight "confident learning" (the idea behind cleanlab) |
| Feature corruption | for each column, predict it from all the other columns (cross-validated); flag a cell whose actual value is very unlikely under that prediction (categorical: predicted probability < 0.05; numeric: robust z-score > 4) |
| Inconsistency (original) | values equal after strip/lowercase but written differently ("Male"/"male"). Report-only. Superseded in Phase 14 by `src/inconsistency.py` (below) |

**Iteration story (good to tell):** the first feature-corruption detector used PCA reconstruction error. It had very low
recall, because a one-column disruption averages out over the whole feature space. The cross-feature predictor targets the
mechanism directly and raised recall roughly 25 times.

| Error | Unit | Precision | Recall |
|---|---|---|---|
| Missing | rows | 1.00 | 1.00 |
| Duplicates | rows | 0.99 | 1.00 |
| Label errors | rows | 0.77 | 0.28 |
| Outliers | cells | 0.18 | 0.67 |
| Feature anomalies | cells | 0.28 | 0.26 |

**Grading level matters:** an earlier row-level grading inflated the feature-anomaly score from 0.26 to 0.67. We report
cell-level because the corruption is cell-level.

**Why label recall is low (0.28):** the 0.90 confidence bar is deliberately strict. Flipped labels near the decision
boundary look like normal examples, and Adult's classes overlap (the model is only 85% accurate), so most flips never
reach 0.90 confidence.

### Phase 6: quality report
Flagged rate per dimension, kept **separate** from detector reliability (precision/recall), so a reader sees both "how
much was flagged" and "how much to trust that".

### Phase 7: impact (`phase7_impact.py`)
5 error types x 3 rates x 5 seeds = 75 experiments. Same split, preprocessing, model and test set every time; only
error type, rate and seed vary. This is a *controlled experiment*: any difference is attributable to the corruption.

Mean F1 damage vs the clean baseline (mean +/- std over 5 seeds):

| Error | 5% | 10% | 20% |
|---|---|---|---|
| Label errors | 0.0098 +/- 0.0054 | 0.0248 +/- 0.0032 | **0.0649 +/- 0.0105** |
| Feature corruption | 0.0036 +/- 0.0042 | 0.0059 +/- 0.0018 | 0.0144 +/- 0.0063 |
| Outliers | 0.0028 +/- 0.0035 | 0.0051 +/- 0.0036 | 0.0091 +/- 0.0027 |
| Missing values | 0.0042 +/- 0.0051 | 0.0006 +/- 0.0046 | 0.0071 +/- 0.0032 |
| Duplicates | 0.0005 +/- 0.0040 | 0.0020 +/- 0.0050 | 0.0020 +/- 0.0052 |

Reading (Random Forest): label noise dominates and grows faster than linearly (0.010, 0.025, 0.065); with gradient boosting
the same ordering holds at about 0.45 times the size (0.004, 0.011, 0.030). Duplicates' damage is smaller than its
own std: indistinguishable from zero. A likely reason (a hypothesis, not tested): copies add no false information, they
only reweight some rows slightly.

### Phase 8: oracle repair (the ceiling)
Use the corruption log to restore exact original values, retrain, measure F1. Not a real method (no real system has the
log). It answers: *if this error were perfectly fixed, what is the most F1 we could ever recover?* Phase 9 is graded
against this ceiling.

### Phase 9: real repair (`phase9_repair_engine.py`)
Detect blind, repair only what the detector is confident about, leave the rest flagged. Two tiers:

- **Exact errors (missing, duplicates):** repair every detected case.
- **Statistical errors (outliers, labels, features):** the "flag for review" threshold stays loose; a **stricter**
  threshold decides what is auto-repaired (label confidence 0.97 vs 0.90; outlier k = 3.0 vs 1.5; feature categorical
  0.01 vs 0.05; feature numeric z 6.0 vs 4.0).

Repair method per type: duplicates, drop; missing, KNN imputation (k = 5, on standardised numeric columns) and the mode for
categoricals; outliers, clip to the IQR fence; labels, set to the out-of-fold prediction; features, replace with the
cross-feature model's prediction.

**Why KNN, not median:** the shared pipeline already median-imputes internally. A median "repair" would be a no-op, making
dirty and repaired F1 identical by construction. KNN is a genuinely different, more informed estimate.

Graded on **two independent axes**, because a repair can win one and lose the other:

- **Repair correctness:** precision (of the cells we changed, how many were truly corrupted) and recall (of the truly
  corrupted, how many did we change).
- **Task impact:** `recovery = repaired_F1 - dirty_F1`.

| Error (10%) | ML recovery | Repair precision | Repair recall |
|---|---|---|---|
| Duplicates | +0.0032 | 0.99 | 1.00 |
| Missing values | -0.0003 | 1.00 | 1.00 |
| Label errors | +0.0031 | 0.86 | 0.06 |
| Outliers | +0.0017 | 0.31 | 0.57 |
| **Feature corruption** | **-0.0071** | 0.33 | 0.12 |

Findings:

- **Detected is not safe to repair.** Feature-corruption repair is net-harmful at every rate: two-thirds of what it
  "fixes" was never corrupted (precision 0.33), and the substitutes are worse than the original values.
- **The most damaging error is the hardest to fix.** Label repair is precise (0.86) but has recall 0.01 to 0.18 under the
  conservative rule, so it recovers almost nothing.
- Missing-value imputation turns net-harmful at 20% (-0.0054). The *cause* was not tested.
- Recovery ratios (recovery / damage) are not reported for duplicates, outliers, missing: numerator and denominator are both
  near the noise floor, so the ratio is unstable.

### Phase 10: prioritisation (`phase10_prioritization.py`)
Three separately measured dimensions:

- **Impact** (Phase 7): mean damage.
- **Repairability** (Phase 9): status positive / negative / weak / not_estimable (rules in 3.4).
- **Risk** = P(harm) x E[harm magnitude | harm]: the probability a repair makes things worse across seeds, times how much
  worse. Always >= 0.

`Priority = w_impact*Impact - w_risk*Risk + w_repair*Repairability`, each min-max normalised within a corruption rate
(so error types are compared at matched severity).

Headline at 10%, equal weights: **label errors 0.66 > duplicates 0.35 > outliers 0.31 > missing 0.16 > feature
corruption -0.26.** Across 15 weight combinations label errors rank first and feature corruption last in 100% (Random Forest;
under gradient boosting missing values, not feature corruption, ranks last: a "model-dependent ranking").
That is **severity-dependent**: at 5% label errors rank first in only 53%; at 20% imputation turns harmful and competes
with feature corruption for last place. Repairability status can differ across rates for the same error type
(label errors are "weak" at 5% and 20% but "positive" at 10%). That instability is a finding, not noise.

### Phase 11: AI-readiness
**11A:** define measurable dimensions from machinery already built (not an invented metric), keep them separate.
**11B:** test whether the *detected* rate tracks *real* damage, because on a real dataset only detector output exists.

| Error | r (detected rate vs damage) | Reading |
|---|---|---|
| Outliers | 0.99 | tracks damage |
| Feature corruption | 0.95 | tracks, but about 46% flagged even on clean data |
| Duplicates | 0.78 | tracks |
| Missing values | 0.61 | weak |
| **Label errors** | **-0.95** | **inverted** |

Pooled r = -0.24 (p = 0.39). **Conclusion: no single score.** The label detector is inverted, and the baseline rates sit
on incomparable scales.

**Why inverted (most likely, not proven):** the detector's cross-validated model is itself trained on the noisy labels. As
noise rises, the model becomes less confident, fewer rows reach the 0.90 bar, and the flagged rate *falls* while true
damage rises. A low flagged label-error rate is therefore *not* reassurance. Each correlation rests on 3 rates.

---

## 5. The product (what the app does)

`app.py` (Streamlit) over `src/taskclean.py` (library and CLI).

```
upload CSV + target -> audit (6 detectors) -> impact estimate + repair proposals
   -> evidence-based policy -> apply automatically  OR  flag for review
   -> cleaned dataset + repair log + readiness report
```

**The policy.** A repair type is applied automatically only if, in the benchmark at the nearest rate: no seed ever showed
net harm (risk = 0), repair precision >= 0.95, and status is not "negative". On Adult evidence it must pass under **both**
a Random Forest and gradient boosting (the "model check" column: agree / model-dependent / not tested; the stricter status
is used when they differ), and duplicate removal is proposal-only there. So with Adult evidence alone `dataset_cleaned.csv`
changes nothing *by design*; a calibration on your own data (Random Forest) can still certify duplicate removal.
Everything else is logged as a proposal. A second file,
`dataset_cleaned_aggressive.csv`, applies the other proposals (never feature-anomaly) so a user who wants a model-ready file
can judge the risk from its log. Label and feature-anomaly repairs are never auto-applied whatever the evidence.

**Impact estimate.** The detected rate is mapped onto the measured damage curve by linear interpolation (with a (0,0)
anchor so small rates scale toward zero). Above 20% it is labelled a **boundary estimate**, not an extrapolation. It is
worded "benchmark-based task-impact estimate", never "your model will lose X".

**Proof of honesty.** `test_taskclean.py` checks that every cell that differs from the upload is a logged applied
repair, the log never claims a repair on a row absent from the output, and logged values match the file. It also grades
detection on the demo against the injected ground truth and runs awkward CSVs (ID columns, booleans, numbers stored as
text, ZIP codes with leading zeros, placeholders like `?`, unlabeled rows). Both test scripts pass.

**Honest gap:** an uploaded dataset has no clean held-out set, so the product gives **no before/after model metrics for
uploads**. Evaluating on a slice of the same dirty data would mislead.

---

## 6. Calibration (the self-benchmark)

**Problem it solves:** the Adult evidence need not transfer to your data.

**What it does (`selfbench.py`):** (1) builds a clean-ish **reference** from your data (complete rows, no duplicates,
confident label problems removed; mostly-missing or high-cardinality columns dropped); (2) splits it 70/30; (3) injects
each error at 5/10/20% into the training part only, trains a Random Forest, measures damage vs a clean-trained model, then
runs the **real** repair code and measures recovery and harm; (4) applies the same safe-auto rule to *your* evidence.

**Noise floor, measured on your data (null experiment):** retrain on a random 90% of the training rows. That should change
nothing, so whatever it does change *is* noise. A repair counts as harmful only if it loses more than that. Small datasets
get a bigger floor than Adult's 0.003 (German credit: 0.021).

**What it refuses to do:**

- **Declines** if fewer than 600 clean-ish rows remain, or if the target is barely predictable (clean-trained macro-F1 not
  at least 0.05 above a model trained on shuffled labels). "No harm observed" would otherwise be vacuous.
- **Power guard (new):** if noise on either metric exceeds **0.01**, imputation and outlier clipping cannot be certified
  safe (harm smaller than that would be invisible). Exact duplicate removal is exempt: it is safe by construction.
- **Two metrics (new):** damage is measured by macro-F1 *and* ROC-AUC. A repair is safe only if neither shows material harm;
  disagreement is flagged.
- **Exact restoration is never harm.** When a repair reproduces the clean reference exactly, a negative recovery means the
  corruption happened to help by chance.

**The honest caveat:** calibration injects *random* errors; real errors are often structured, so measured repair precision
can be optimistic. And the 0.01 cap was set *after* seeing a failure (see 7), so it is in-sample.

---

## 7. Second dataset: bank-marketing (Phase 13)

**Why:** one dataset proves nothing about generality. Bank-marketing: 45,211 rows, 16 features, **11.7% positive** (Adult
24.8%). Ground truth: Phases 7-10 protocol on 12,000 clean rows, 5 seeds. Four questions fixed *before* running: does the
product run; do Adult findings transfer; does calibration agree with ground truth; does calibration beat transferred Adult
evidence.

**Results**

| Question | Answer |
|---|---|
| Does the product run? | Yes. On Heart Disease (303 rows) calibration *correctly declined* (only 296 clean-ish rows). On German credit (1,000 rows) it ran with a noise floor of 0.021 |
| Do detector reliabilities transfer? | Yes, closely. Label errors 0.93/0.29 (calibration) vs 0.96/0.30 (truth) |
| Does Adult prioritisation transfer? | **Not under F1; exactly under ROC-AUC** (label errors first in 100%, feature corruption last) |
| Does calibration beat Adult evidence? | **Not demonstrated** (7/9 agreement, same as Adult) |

**The big finding.** Under F1, label noise looked *beneficial* (F1 went up in 4 of 5 seeds). Under ROC-AUC it was the most
damaging error in every seed. Why: flipped labels push the model to predict the rare class more often, which raises
minority-class F1, while the model's ability to *rank* gets worse. F1 depends on the threshold, and at 11.7% positive
that dependence swamps the signal.

**The failure and the fix.** A calibration on a noisy "dirty upload" (noise floor 0.0161, twice the truth's) made one
**false approval**: it called imputation at 20% safe when the truth showed it harmful. Two fixes were implemented: measure
AUC alongside F1, and add the power guard. Both fixes were re-validated: the false approval is gone.

**What this does *not* prove (say this unprompted):**

- After the fixes, all three evidence sources tie at 9/9 against the *stricter* two-metric truth. So the fixes remove the
  failures found; they do not show calibration beats Adult evidence.
- The ground truth got stricter because AUC revealed imputation harm that F1 hid, and Adult evidence happens to agree.
- The 0.01 cap and the AUC analysis were decided after seeing the data: exploratory, in-sample.
- One extra dataset with full ground truth; binary target only; Random Forest only.

---

## 7b. Is it the errors or the model? Gradient boosting (Phase 16)

Source: `docs/PHASE16_SPEC.md` (pre-registered, with Results and two visible amendments at the end), `src/phase16_gb.py`,
`results/phase16_gb_*`. Know this one well: it is the strongest answer to "does it hold for another model?".

- **Design.** One variable: the classifier (scikit-learn `HistGradientBoostingClassifier`, chosen over XGBoost for no
  compiled dependency and determinism). Same Adult data, split, corruptors, rates, preprocessing, seeds, detectors and
  repairs. Hyperparameters chosen once on clean validation data and frozen before any dirty run. Ten criteria (G0-G9)
  and five predictions written down first. G0 proved the harness reproduces the frozen Random Forest numbers exactly.
- **Result: 5 pass, 5 fail.** Pass: harness (G0), competent model (G1: 0.715 vs 0.680), damage ordering (G2, Spearman
  0.86, label errors first), repair-status agreement (G4, 11 of 15, the minimum; "everything weak" agrees on 8), dominance
  across three capacities (G8). Fail: damage size (G3, median 0.35 of the forest's), duplicate auto-approval (G5), no new
  harm (G6: KNN imputation hurts gradient boosting by about 0.02), priority ranking (G7: the bottom two swap), Bank
  Marketing macro-F1 replication (G9).
- **Predictions.** P1 (label damage larger under GB) wrong on Adult; P2 (outliers and feature corruption similar) wrong;
  P3 half right; P4 and P5 right.
- **What changed in the product.** Impact estimates carry the model and the GB ratio; Adult-sourced repairs need both
  models to pass; disagreements are `model-dependent` (stricter status); duplicate removal is proposal-only on Adult
  evidence; the ranking carries a model-dependent note. Evidence can only make the product more cautious.
- **Honest caveats.** The Bank Marketing agreement on ROC-AUC (Spearman 0.87, 14 of 15) was found after seeing the macro-F1
  failure, so it is exploratory and does not rescue G9. On Bank, GB's label-error damage is about twice the forest's
  (opposite of Adult), so the size of the damage is dataset-and-model specific. G5's failure is partly a property of the
  strict risk definition on a model with no training noise, not evidence that removal harms. GB's smaller Adult damage may
  reflect its higher clean accuracy (untested). Two amendments were made after the results (G3 wording, G9 premise) and are
  recorded as such. Logistic Regression and Phase 14 under GB were not done.

---

## 8. Honest limits (list these yourself)

1. **Two model families, not all models.** Random Forest throughout, gradient boosting for the Adult evidence (Phase 16);
   Logistic Regression validation not done, and the on-your-data calibration is still Random Forest only.
2. **Random, synthetic errors.** Real errors are structured, so repair precision may be optimistic. Missing values are MCAR.
3. **Estimates, not truth.** On uploads, every rate is a detector estimate with the precision/recall above.
4. **Three corruption rates per error type.** Detector-vs-damage correlations are directional.
5. **Dropping incomplete rows** at the start creates selection bias.
6. **Test set reuse.** The same clean test set measures every experiment. It is never used for tuning, and damage is a
   paired difference, but it is still one test set.
7. **No before/after metrics for uploaded data.**
8. **Classification only.** Duplicates are full-row duplicates. Inconsistency repairs are exact in validation but the
   measured model recovery was negative, so they stay in the review queue; typos are flag-only (recall about 2% on
   dense injected typos). The irrelevant-feature check is report-only. Phase 14 selfbench calibration does not cover
   the new issue types, so calibrated audits use the Adult evidence for them and say so.
8b. **Unusual combinations (Phase 15) is opt-in, flag-only and failed its validation on real data.** Clean-data flag rate
   0.07% on synthetic data, 7% (Diabetes) and 30% (Breast Cancer) on real data; recall 10-14% on hidden swaps; bivariate
   attribution is at chance. It never edits the cleaned file and its status is always "Evidence-limited". Details and both
   design variants: `docs/PHASE15_SPEC.md`.
9. **No composite score**, and the evidence did not support one. That does not prove none could ever work.
10. **Placeholders:** a real category called `-` or `N/A` is over-counted as missing.

---

## 9. Question bank (practise answering out loud)

### Data science fundamentals

**Q1. Why F1 and not accuracy?**
With a 75/25 split, always predicting `<=50K` is about 75% accurate and useless. F1 on the positive class balances
precision and recall on the class we care about.

**Q2. Precision versus recall, with an example from this project.**
The label detector: precision 0.77 (of rows it flags, 77% really are wrong), recall 0.28 (it finds 28% of the wrong
labels). It is conservative by design (confidence >= 0.90).

**Q3. What is overfitting, and how did you guard against it?**
Fitting noise so training performance beats unseen performance. Random Forest averages many de-correlated trees to reduce
variance. We never tuned hyperparameters on the test set, and all damage is measured on a held-out clean test set.

**Q4. Why a validation set at all?**
For model selection and tuning. We did no tuning, so the validation split is reserved rather than heavily used. Say that
plainly.

**Q5. What is data leakage, and did you avoid it?**
Information from the evaluation data influencing training. The test set is never corrupted, repaired, or fitted on.
Detectors that use a model are cross-validated so a row is never judged by a model that saw it. Preprocessing is fitted
inside the pipeline on training data.

**Q6. Why stratified splitting and stratified k-fold?**
They keep the 25% minority share in every part, so differences are not artefacts of an unlucky split.

**Q7. Explain k-fold cross-validation and why you used out-of-fold predictions.**
Section 3.5. A model that has seen a row will reproduce its label, noisy or not, so it cannot reveal label errors.

**Q8. Bias-variance trade-off in Random Forest?**
Individual deep trees have low bias and high variance. Bagging plus random feature subsets de-correlates them, and
averaging cuts variance while keeping bias low.

**Q9. Why one-hot encoding?** Section 3.2. No false ordering. `handle_unknown="ignore"` for unseen categories.

**Q10. Why scale numeric features when Random Forest does not need it?**
We share one pipeline across Logistic Regression (which does need it) and Random Forest so both see identical inputs.

**Q11. How do you handle class imbalance?**
We did not resample. We chose metrics robust to it (F1 on the positive class, macro-F1, ROC-AUC). Alternatives (class
weights, SMOTE, threshold tuning) are not used.

**Q12. What is ROC-AUC, intuitively?**
The probability that a randomly chosen positive is scored higher than a randomly chosen negative. It does not depend on a
threshold, which is why it rescued the second-dataset analysis.

**Q13. MCAR, MAR, MNAR?** Section 3.6. We inject MCAR.

**Q14. Why IQR instead of z-scores for outliers?**
Mean and std are themselves distorted by outliers; quartiles are robust. Tukey fences also do not assume normality.

**Q15. What goes wrong with the IQR rule on `capital-gain`?**
About 92% zeros so IQR = 0 and every non-zero value would be flagged; we fall back to percentile bounds. Also the value
99,999 is a real top-code, so injected extremes cannot be separated there.

**Q16. Pearson versus Spearman?** Section 3.8.

**Q17. Why is r = 0.99 with three points not strong evidence?**
With only three points r is very unstable: a large correlation can arise by chance, the confidence interval is huge, and
the p-value cannot be small. We label it directional evidence, not significance.

### Methodology and design

**Q18. Why inject errors into clean data rather than study real dirty data?**
Because real data has no ground truth. Injection gives a known cause, so we can measure effect, grade detectors and grade
repairs. The price is realism (random, not structured, errors).

**Q19. Why paired seeds?** Cancels training noise; a tighter estimate of the true effect.

**Q20. Why is the noise floor 0.003?** Baseline F1 varies about 0.0025 across seeds, so smaller effects are not separable
from noise. It is a documented parameter, not a tuned magic number.

**Q21. Difference between detector precision and repair precision?**
Detector precision: of flagged items, how many are truly corrupted. Repair precision: of cells we *changed*, how many were
truly corrupted. Repair uses the stricter tier, so the two can differ (labels: detector precision about 0.77, repair precision 0.86 at
10%).

**Q22. Difference between damage, recovery and recovery rate?**
Damage = baseline - dirty. Recovery = repaired - dirty. Recovery rate = recovery / damage (unstable when damage is tiny).

**Q23. Why a two-tier threshold?**
Flagging for review is cheap and reversible; overwriting data is not. So a repair needs a stricter bar than a flag.

**Q24. Why is the safe-repair bar 0.95 precision?**
It is a documented policy parameter (`SAFE_REPAIR_MIN_PRECISION`): a repair must rarely overwrite legitimate data. It is a
judgement, not derived from a theorem. Be honest about that.

**Q25. Why is duplicate removal "safe" if duplicates barely hurt?**
"Safe" is about *harm*, not benefit. Exact removal restores the data exactly (precision 0.99-1.00, recall 1.00), so it
cannot hurt. The benefit is small because damage is small. (Phase 16 caveat: under gradient boosting the strict gate counted
tiny negative recoveries at 5% and 20% as risk, so on Adult-only evidence the product proposes removal instead of applying it.)

**Q26. Why does imputation hurt at 20%?**
Observed: recovery -0.0054. The mechanism was not tested. A plausible reading is that imputed values are guesses that
distort the joint distribution once many cells are filled, but treat that as a hypothesis.

**Q27. Why is the label detector inverted?** Section 4, Phase 11. "Most likely", with the reason.

**Q28. Why not combine everything into one readiness score?**
We tested it: pooled r = -0.24 (p = 0.39). A score built on an inverted detector would be misleading. We chose to report
evidence status per dimension instead.

**Q29. Why does repair precision not equal ML benefit?**
Precision counts correct cells. Benefit depends on whether the *model* is affected. Feature corruption repair is a good
example of both being poor, but missing-value repair has precision 1.00 and recovery about 0.

### Results and honesty

**Q30. What is your single most important result?**
Detected does not mean harmful, and harmful does not mean safely repairable. Label errors are the most damaging and least
repairable; duplicates the safest. Plus the F1-versus-AUC finding on imbalanced data.

**Q31. What is your negative result?**
Feature-corruption repair is net-harmful; the label detector is inverted; calibration did not beat the transferred
evidence; no composite score. We kept all of them.

**Q32. Did calibration work?**
Mixed. It estimates detector reliability well, and the dataset's own damage ordering far better than transferred Adult
evidence (Spearman with the macro-F1 ground truth: 0.70 on the raw upload and 0.35 on the dirty upload, versus -0.01 for
the transferred Adult evidence). For the safe-to-repair decision it tied with Adult evidence and, before the fixes, made
one false approval.

**Q33. What did the second dataset change?**
It exposed that F1-based damage is unreliable at 11.7% positive, and that a noisy calibration can falsely approve a repair.
Both were fixed and re-tested.

**Q34. Is the 0.01 cap overfitted?**
Yes, partially: it was chosen after seeing one dirty upload's noise floor (0.0161 vs truth 0.0074). It is round and
conservative, but in-sample and untested elsewhere. It also means small datasets (for example German credit, 0.021) can
never auto-approve imputation.

**Q35. Would your findings hold for a neural network or logistic regression?**
Partly known. Phase 16 re-tested the key results with gradient boosting: the ordering (label errors first) held, but the size
of the damage, several repair verdicts and the bottom of the priority ranking did not (5 of 10 pre-registered criteria
failed), and the Bank Marketing check did not replicate on macro-F1. Neural networks and Logistic Regression were not
tested; Logistic Regression is the next step and the harness takes any model.

**Q36. What would you do with more time?**
Logistic Regression validation; more datasets; structured (non-random) error injection; a calibration test on more
datasets; investigating *why* imputation turns harmful at 20%.

**Q37. What is original here?**
Not the detectors (standard ideas). It is the experimental design that grades detection, repair and ML impact
*separately*, the evidence-based auto-repair rule, the honest status model, and the self-calibration with a noise floor and
decline rules.

### Traps

- *"Your tool cleans the data."* Correct it: it **audits** and repairs only what the evidence supports; the default file
  changes little by design.
- *"Is accuracy 85% good?"* Context: baseline-by-majority is about 75%.
- *"Your result proves X."* Say "directional evidence, 5 seeds, 3 rates".
- *"Did you tune things to get these numbers?"* The research phases were frozen: no re-tuning or re-running to improve
  results, and weak or negative results were kept. The two fixes made after the second-dataset validation (AUC, the 0.01
  cap) were decided after seeing that data, and are disclosed as exploratory. Phase 16's criteria were written before its
  results; its two amendments (G3 wording, G9 premise) were made after and are recorded as amendments, and the Bank AUC
  comparison is labelled exploratory.

---

## 10. Number cheat sheet

| Item | Value |
|---|---|
| Adult rows (after dropping missing) | 45,222; train 31,655 |
| Positive class share | about 24.8% (bank-marketing 11.7%) |
| Split | 70 / 15 / 15, stratified |
| RF | 300 trees; clean F1 0.680, accuracy 0.851, AUC 0.903; LR F1 0.662 |
| Seeds | 42 to 46 (5); split seed fixed 42 |
| Rates | 5 / 10 / 20% |
| Experiments, Phase 7 | 5 x 3 x 5 = 75 |
| Noise floor | 0.003 (Adult); cap for certification 0.01 |
| Max damage | label errors, 0.0649 F1 at 20% |
| Repair status rule | 80% of seeds agree on sign and abs(mean recovery) >= 0.003 |
| Auto-repair rule | risk = 0, repair precision >= 0.95, status not negative |
| Feature repair at 10% | -0.0071 F1 (harmful at every rate) |
| Label detector | precision 0.77, recall 0.28; r = -0.95 (inverted) |
| Pooled readiness r | -0.24, p = 0.39 |
| Prioritisation 10% | label 0.66 > dup 0.35 > outliers 0.31 > missing 0.16 > feature -0.26 |
| Calibration declines | fewer than 600 rows, or macro-F1 < shuffled + 0.05 |
| Bank-marketing | 45,211 rows; ground truth on 12,000 clean rows; AUC prioritisation transfers |
| Detector thresholds | label 0.90 (auto 0.97); IQR k 1.5 (auto 3.0); feature cat 0.05 (0.01), z 4 (6) |

---

## 11. Three-minute live demo script

1. Open the app (`.venv/bin/python -m streamlit run app.py`, path has a space, so use `python -m`).
2. Tick **Use demo dataset** (6,300 Adult rows, all five errors injected at 5%).
3. Tick **Calibrate on this dataset**, press **Run audit** (about 30 s).
4. **Quality audit:** each detector's flagged rate next to its benchmark precision/recall.
5. **Calibration:** the noise floors (F1 and AUC), the evidence table: label-error damage is about three times larger under
   AUC (+0.066) than F1 (+0.022) at 20%.
6. **Repair plan / Outputs:** only duplicates auto-repaired; everything else flagged; open the repair log.
7. Say the thesis: *detected is not harmful, harmful is not safely repairable.*

If the app will not open, show `taskclean_output_demo/` (sample outputs) and `results/` instead.

---

## 12. Syllabus bridge

If the teacher goes by the course units, the companion notebooks in `notebooks/` cover each topic on this same dataset
(`notebooks/README.md` maps every syllabus topic to a section): probability and inference (Unit I), attribute types and
descriptive statistics (II), cleaning, outliers, transformation, encoding (III), NumPy, PCA and pandas (IV), EDA,
cross-validation, confusion matrix (V), visualisation and regression (VI).
