# TaskClean: the complete study guide (from zero)

Read this top to bottom once. Part 17 (questions and answers) and Part 16 (what to say and what never to say) are the two parts to re-read the night before.
Every number here comes from the project's own README, specs or result files. Where something is **not proven**, this guide says so, because that is what a good examiner will probe.

---

## Part 0. The pitch (memorise these)

**One sentence.**
TaskClean is a tool that looks at a data file, finds what is wrong with it, and then *measures* how much each problem actually hurts a machine-learning model, so it only repairs what is safe and worth repairing, instead of cleaning everything blindly.

**Thirty seconds.**
Most cleaning tools ask "is this data wrong?" and fix it. We ask three questions: is it wrong, is it *hurting my model*, and is the repair *safe*? We test these with controlled experiments: we take clean data, break it on purpose in known ways, train models, and see which problems really cost performance and which repairs really win it back. The tool then applies only repairs the evidence supports, logs every change, and tells you honestly when the evidence is weak.

**Two minutes.** Add: it works on CSV and Excel files, for classification (predict a category) and regression (predict a number). It has six detectors, a repair engine, a safety policy, an optional "calibration" that re-runs the experiments on the user's own data, a panel that trains four model types before and after cleaning, and a web dashboard. It was built as research (27 phases of experiments with written-down success criteria) and as a usable product. The honest headline result: it measures impact and refuses unsafe repairs reliably; it has shown a real model improvement on one real file (Bank Marketing, logistic regression), and no improvement on the other files we tried.

---

## Part 1. The problem, in plain words

**Data science in one paragraph.** You have a table. Each row is one thing (a customer, a house, a loan). Each column is a fact about it (age, price, city). One column is the thing you want to predict; that is the **target** (also called the label). A **model** is a program that learns, from rows where the answer is known, how to guess the answer for new rows. This is **machine learning (ML)**.

**Two kinds of prediction.**
- **Classification**: the target is a category ("will leave / will stay", "approved / rejected").
- **Regression**: the target is a number (house price, rating).

**Why dirty data matters.** A model learns whatever is in the table. If the table contains mistakes, the model learns the mistakes. "Garbage in, garbage out."

**What is wrong with normal cleaning tools.** They fix everything they can find: fill every blank, delete every odd value. But:
1. Some problems barely affect the model, so fixing them is wasted effort.
2. Some "fixes" make the model *worse* (for example, filling blanks with invented numbers can add false information).
3. Nobody measures the effect. You never learn whether cleaning helped.

**Our objective** (the project's main idea): *clean the data according to its impact on model performance*, and measure that impact.

---

## Part 2. Data science building blocks (every concept used)

Read these slowly. Each one is used somewhere in the project.

### 2.1 Train/test split (and why the test set is sacred)
We split rows into a **training set** (the model learns from it) and a **test set** (we judge the model on it). The model must never see test rows while learning, otherwise it could just memorise the answers and the score would be fake. This is the most important rule in ML.

- In the research (Phases 1-11) the split on UCI Adult was **70% train / 15% validation / 15% test**, and the test set is never touched or corrupted.
- In the Model panel the split is **80/20**.

### 2.2 Overfitting
A model that memorises the training rows instead of learning the general pattern. It scores great on training rows and badly on new rows. This is why we always score on rows the model has not seen.

### 2.3 Random seed
Many steps use randomness (shuffling rows, building trees). A **seed** is a number that fixes the randomness so a run can be repeated exactly. We run the same experiment with several seeds (42 to 46) and look at the average and the spread, because one run can be lucky.

### 2.4 Metrics (how we score a model)
Suppose the model predicts "will leave" or "will stay".
- **Accuracy**: share of predictions that are right. Misleading when one class is rare (a model that always says "stay" is 90% accurate if 90% stay).
- **Precision**: of the rows the model *said* were positive, how many really were. ("When it accuses, is it right?")
- **Recall**: of the rows that really were positive, how many the model found. ("Does it catch them all?")
- **F1**: one number that combines precision and recall (their harmonic mean). High only if both are high.
- **Macro-F1**: compute F1 for each class separately, then average them. Treats rare classes as equally important. This is the project's main classification score.
- **ROC-AUC**: how well the model *ranks* positives above negatives, independent of a cut-off. 0.5 = coin flip, 1.0 = perfect. We use it as a second opinion because F1 can mislead on imbalanced data (the Bank Marketing check showed this).
- **R-squared (R²)**: for regression. 1.0 = perfect, 0 = no better than always guessing the average, **negative = worse than guessing the average**. This is the project's main regression score.

**Chance level** (very important idea): the score a model gets with no real knowledge. For 4 balanced classes, macro-F1 is about 0.25; for 3 classes about 0.33. If a model scores at chance, the columns carry no information about the target. (This is exactly what happened on the Messy Employee file.)

### 2.3b Imbalanced data
When one class is much rarer than the other. Bank Marketing has 11.7% "yes". F1 and AUC can then disagree, which is why the project looks at both.

### 2.4b The models we use
You do not need the maths, only the idea and one-line differences:
- **Logistic regression**: draws a straight dividing boundary. Simple, fast, *sensitive to outliers and to scale*. Treat it as the "linear" family.
- **Random forest (RF)**: many decision trees, each trained on a random piece of the data, voting together. Robust, handles messy data well. Research used 300 trees; the panel uses 100 to be fast.
- **Gradient boosting (GB)**: trees built one after another, each fixing the previous one's mistakes. Usually the most accurate on tables. We use scikit-learn's `HistGradientBoostingClassifier`.
- **Neural net (MLP)**: a small neural network (one hidden layer of 64 units, early stopping). Needs scaled inputs, noisy on small data.

Why four? Because the effect of dirty data **depends on the model**. A repair that helps logistic regression can do nothing for a random forest. (Phase 16 proved this.)

### 2.5 Preprocessing
Models need numbers. So we:
- **Impute** blanks (fill them with something).
- **One-hot encode** categories (turn "red/green/blue" into three 0/1 columns).
- **Scale** numbers (put them on a similar range so one big-valued column does not dominate).
These are inside the model pipelines, so they are *fitted on training rows only* (never on test rows, which would leak information).

### 2.6 Cross-validation and "out-of-fold" predictions
To judge a model, split the data into k folds (we use 5). Train on 4 folds, predict the 5th; rotate so every row gets predicted by a model that **never saw it**. These predictions are called **out-of-fold**. We use them in detectors so a row is never judged by a model that memorised it.

### 2.7 Controlled experiment (the heart of the research)
Real dirty data has no answer key: you do not know which cells are wrong. So we create one:
1. Start with a **clean** dataset (UCI Adult, a census income dataset).
2. **Inject** one error type at a known rate (5%, 10%, 20%) and write a **ground-truth log** of every cell changed.
3. Train a model on the dirty training rows, score on the clean test rows. The drop in score versus the clean baseline is the **damage**, which measures *impact*.
4. Run the detector and repair; compare against the log to grade them (precision/recall), and re-train to see how much score is **recovered**.

This is the same idea as testing a smoke alarm by lighting a controlled fire.

### 2.8 Precision and recall *of a detector*
Same words as for models, applied to flagging errors. A missing-value detector has precision 1.00 / recall 1.00 (it can see blanks). The label-error detector has precision 0.77 / recall 0.28 (when it accuses it is usually right, but it finds only about a quarter of the wrong labels).

### 2.9 Noise floor (second most important idea)
Scores wobble a bit by pure chance (a different random split changes the number). To avoid claiming a gain that is just wobble, we measure the wobble with a **null experiment**: retrain on a random 90% of the training rows, which *should change nothing*, and see how much the score moves. That is the **noise floor** (minimum 0.003 in the code). A change smaller than the floor is "within noise" and we do not claim it. We also require **at least 80% of seeds to agree** on the direction before calling something improved or worse.

### 2.10 Statistics ideas used
- **IQR rule for outliers**: a value is an outlier if it is below Q1 − 1.5×IQR or above Q3 + 1.5×IQR (Q1, Q3 = 25th and 75th percentiles; IQR = Q3 − Q1). The Baseline report also uses |Z| > 3 (more than 3 standard deviations from the mean) and skewness.
- **Robust z-score**: like Z but based on the median, so a few huge values do not distort it.
- **Correlation (r)**: how closely two things move together (−1 to +1). Phase 11 used it to test whether "how much error a detector reports" tracks "how much damage".
- **Chi-square / ANOVA** (used in the last Employee diagnosis): tests of whether a column is related to the target. A large p-value means "looks unrelated".
- **MCAR** (missing completely at random): blanks are placed uniformly at random, the simplest assumption; our missing-value corruptor uses it.

### 2.11 Imputation methods
Filling blanks:
- **Median** for numbers (the middle value; robust).
- **Most frequent value (mode)** for categories.
- **KNN imputation (k=5)**: find the 5 most similar rows and average their value. Better in theory, but Phase 17 found it **hurt gradient boosting** on zero-heavy columns (capital-gain), so now columns where one value covers ≥50% of cells use the median.

### 2.12 Confident learning (how wrong labels are found)
Idea from the `cleanlab` library. Train a model with cross-validation. If, for a row, the model that never saw it is **very confident** (≥ 0.9) about a class that **disagrees** with the given label, the label is suspect. Our detector uses a 200-tree random forest and 5 folds.

### 2.13 PCA (principal component analysis)
Compresses data into a few directions that capture most variation. If a row cannot be rebuilt well from those directions (high "reconstruction error"), it is odd. It was the first feature-anomaly detector; the final one is cross-feature prediction (see Part 5).

### 2.14 Pre-registration
Writing down, **before** running an experiment, what you will measure and what counts as success. It prevents "tuning until it looks good". **Honest caveat:** in this project it is a *self-imposed convention* (a spec file first, fixed criteria, amendments recorded); it is not independently timestamped. In Phase 27 you explicitly set it aside for exploration and those results are labelled "exploratory, not pre-registered".

---

## Part 3. The six data problems the tool looks for

| Problem | What it means | Example |
|---|---|---|
| **Missing values** | blank cells, or placeholders like `?`, `N/A`, `-`, `null` | Age is empty |
| **Duplicates** | the same full record twice | row 5 identical to row 2 |
| **Outliers** | a number far outside the usual range | age 999 |
| **Wrong labels / wrong target values** | the target column is wrong for that row | customer marked "stayed" who really left |
| **Feature corruption / anomalies** | a cell value that does not fit its row | a swapped value |
| **Inconsistent values** | same thing written differently | `NYC`, `nyc`, `New York` |

Plus extras added later: placeholder **codes** (`-1`, `-99`), placeholder **words** (`Unknown`), user-declared missing values, joint outliers (flag-only), irrelevant-feature check (report-only).

---

## Part 4. How the project is organised (the big picture)

```
            upload file (CSV / Excel / JSON / Parquet)
                          |
                  intake.py  (detect encoding, delimiter, sheets)
                          |
              health_snapshot.py  -> Baseline health report (frozen, never changes)
                          |
          choose TARGET and TASK (classification or regression)
                          |
        taskclean.audit_dataset()   <-- expensive stage
             |  _prepare()   (placeholders -> missing, numbers-as-text -> numbers, IDs set aside)
             |  detectors (detect.py, inconsistency.py, regression.py, ...)
             |  repair PROPOSALS (taskclean._propose_*)
             |  evidence + policy (taskclean._build_policy)
                          |
        taskclean.apply_repairs()   <-- cheap stage, can be re-run with different choices
             |  applies only approved repairs, logs every change
                          |
   outputs: cleaned file, aggressive file, repair logs, quality/impact/readiness reports
                          |
   optional:  calibration (selfbench.py)       - measures impact/safety on YOUR data
              model panel (model_panel.py)     - 4 models, before vs after, on held-out rows
              controlled proof (controlled_proof.py) - clean-test proof on Adult
```

**Two-stage design** (good thing to mention): `audit_dataset` (slow, model-based) and `apply_repairs` (fast) are separate, so the user can change the repair plan in the app without paying for the audit again.

### Code map (what lives where)
| File | Job |
|---|---|
| `src/taskclean.py` (~2000 lines) | The product: `_prepare`, proposals, policy, `audit_dataset`, `_apply`, `apply_repairs`, reports, CLI |
| `src/detect.py` | Detectors: missing, duplicates, IQR outliers, label noise, feature corruption (PCA and cross-feature) |
| `src/inconsistency.py` | Inconsistent-spelling detector and rules |
| `src/regression.py` | Regression support: task suggestion, wrong-target-value detector |
| `src/joint_outliers.py` | Unusual combinations of numbers (flag-only, off by default) |
| `src/relevance.py` | Irrelevant-feature check (constant, ID-like, redundant, no better than random probes) |
| `src/corruption/` | The error injectors used in experiments (with ground-truth logs) |
| `src/selfbench.py` | Calibration: self-benchmark on your own data |
| `src/model_panel.py`, `model_panel_regression.py` | Four-model before/after panel |
| `src/controlled_proof.py`, `controlled_proof_regression.py` | Clean-test proof; tuned cleaning (`repair_frames`, `select_repair`) |
| `src/health_snapshot.py` | Baseline health report |
| `src/intake.py` | File reading (CSV, Excel, JSON, Parquet) |
| `src/tuned_check.py` | Run the tuned-cleaning check on any file you give it |
| `src/phase*.py`, `explore_phase27.py`, `validate_*.py` | The research experiments and validations |
| `src/test_*.py` | Automated tests (invariants and mutation checks) |
| `backend/main.py` | FastAPI server (the web API) |
| `frontend/` | React + TypeScript dashboard (pages: Start, Overview, Baseline, Quality, Impact, Repair plan, Review, Outputs, Calibration, Model panel, Controlled proof, Readiness, Features, Research) |
| `app.py` | The older Streamlit version of the app |
| `docs/PHASE*_SPEC.md` | The written plan and results for each phase |

Note: `errors.py`, `train.py`, `impact.py`, `repair.py`, `readiness.py`, `pipeline.py` are a **superseded first prototype** (breast-cancer data); say so if asked.

---

## Part 5. The detectors, one by one

**1. Missing values** (`detect_missing`). Any blank cell or placeholder token (`""`, `?`, `-`, `n/a`, `null`, `nan`...). Precision and recall 1.00.

**2. Duplicates** (`detect_duplicates`). A row identical (features **and** target) to an earlier one. Precision 0.99, recall 1.00.

**3. Outliers** (`detect_outliers_iqr`). The IQR rule with k = 1.5 on every numeric column. Special case: a column like Adult's `capital-gain` is 92% zeros, so IQR = 0 and every non-zero would be flagged; for such columns the code uses a percentile rule (0.5% / 99.5%) instead. **Weakness:** precision is low (0.18 on Adult; 0.09-0.21 on real files) because real data has legitimate extreme values (capital-gain is capped at 99,999 in real census data). So say: "outlier flags mean *unusual values to look at*, not confirmed errors."

**4. Wrong labels** (`detect_label_noise`), classification. Confident learning (Part 2.12). Precision 0.77, recall 0.28. For **regression**, the equivalent is `detect_target_errors`: an out-of-fold random-forest regression, and a target value is flagged when its residual is more than 4 robust sigmas away (scaled within 5 bands of the predicted value).

**5. Feature corruption** (`detect_feature_corruption_crossfeature`). For each column, predict it from all the other columns (cross-validated), and flag a cell whose actual value disagrees strongly with the prediction. Roughly 25 times better recall than PCA. Precision 0.28 / recall 0.26 — weak, so it is **off by default** and its repair is never applied automatically.

**6. Inconsistent spellings** (`detect_inconsistency_v2`). Builds a normalised "key" for each value (lower-case, trimmed, separators and trailing punctuation removed). Rules: case/spacing variants = certain; short abbreviations (≤3 letters, the minority form ≤30%) = verify; typos = flag only. A real bug was found in Phase 14: an abbreviation rule that ran backwards and proposed rewriting `No phone service` to `No`; it was fixed.

**Report-only extras:** near-duplicates, |Z| outliers, skew, constant columns, ID-like columns, numbers with currency symbols, joint (multivariate) outliers (its own validation *failed* on real data, so it is opt-in and flag-only), irrelevant features.

---

## Part 6. How `_prepare` makes real files readable (Phases 25-26)

Before any detector runs, `_prepare` converts the raw file into an analysable form:
- Placeholder tokens (`?`, `N/A`, ...) become real missing values.
- Numbers stored as text (`"3,5"`, `"$1,200"`) become numbers; imputed values are written back in the same format.
- **Placeholder codes** (Phase 25): `-1, -9, -99, -999, -9999` are read as missing when a column has **at least 3** of them, they cover **at most 50%** of the column, and every other value is non-negative. (A code covering more than half the column, like `pdays = -1`, is a category, not missingness.)
- **Placeholder words** (Phase 26): `unknown, unspecified, not available, not applicable, undefined, missing, tbd, no data`... same three rules. `none` and `other` are real categories and untouched.
- **User-declared missing values**: the Start page lets you say "treat `0` in column X as missing", and a switch "treat these as real values" for genuine `-1`/`Unknown`.
- Why `0` and `999` are **not** guessed: a scan of every project file found real zeros in more than 20 columns, so any automatic rule would flag correct data.
- Codes and words are **never filled automatically** (held back for the human), because only the user knows whether `-1` is a real value.
- Two **report-only notes**: numeric ranges stored as text (`$137K-$171K`), and a number glued onto a text column that equals another column (company name carrying the rating).
- ID-like and very-high-cardinality columns (more than 50 distinct text values) are still checked for blanks and duplicates but skipped by model-based detectors.

---

## Part 7. The repairs

| Problem | Repair | Details |
|---|---|---|
| Missing numbers | **KNN imputation** (k=5, standardised columns) | median instead when ≥50% of a column is one value, or the dataset has more than 40,000 rows; integer columns are rounded |
| Missing categories | most frequent value | |
| Duplicates | drop the repeated rows (keep the first) | exact, so precision 0.99 |
| Outliers | **clip** (winsorize) to a stricter bound | a stricter rule than the one used to flag |
| Wrong labels | flip to the model's confident prediction | only high-confidence; rewrites the target, so **never automatic** |
| Feature anomalies | replace with the cross-feature model's prediction | harmful in testing, **never automatic** |
| Inconsistent values | rewrite to the canonical spelling | exact (precision 1.0) but model recovery measured negative, so review queue |

Apply order: inconsistency → missing → outliers → feature corruption → labels → duplicates.

Two output files:
- `dataset_cleaned.csv` — **evidence-based file**: only repairs the evidence approves (often unchanged).
- `dataset_cleaned_aggressive.csv` — every proposed repair except feature-anomaly ones; model-ready but riskier.
- `repair_log.csv` — every change *and* every non-change: row, column, original, proposed and applied value, method, tier, score, evidence status. Tests check that every cell that differs from the upload is a logged repair.

---

## Part 8. The safety policy (the "brain" of the product)

A repair type is applied **automatically** only if **all** of these hold:
1. **Risk = 0**: no seed ever showed net harm in the benchmark.
2. **Repair precision ≥ 0.95** (`SAFE_REPAIR_MIN_PRECISION`): it rarely overwrites legitimate data.
3. Status is not **negative**.
4. On Adult-sourced evidence it must pass under **both** random forest and gradient boosting (Phase 16); if they disagree the stricter verdict is used and the row is marked `model-dependent`.
5. It is not on the **never-auto** list: label errors, feature corruption, joint outliers.
6. It is not **underpowered** (the measured noise floor must be ≤ 0.01, otherwise small harm would be invisible).
7. Duplicate removal on Adult evidence is "proposal only" (it failed the GB gate at 5% and 20%), so it needs one click or a calibration.

Everything else is **flagged for human review**, logged, and not changed. On default Adult evidence **nothing** is applied automatically.

**Evidence status** (a label per problem type, separate from its observed rate): *validated*, *weak*, *inverted*, *not estimable*. **There is no single "readiness score"**, because Phase 11 showed the aggregate was not supported by evidence.

---

## Part 9. Calibration (measuring on the user's own data)

The Adult results do not automatically apply to a new file. **Calibrate on this dataset** (`selfbench.py`) re-runs a small version of the experiments on the user's data:
1. **Reference**: build a "clean-ish" subset (complete rows, no duplicates, high-confidence label problems removed). It is the best clean data derivable, not true ground truth.
2. **Experiments**: split 70/30; for each error type (missing, duplicates, outliers, label errors) at 5/10/20% and several seeds, inject only that error into the training rows, train, and measure the damage; then run the real repair and measure recovery, precision, recall and harm. The test rows are never corrupted.
3. **Noise floor** from a null experiment.
4. **Decisions** by the same safe-auto rule.

When it **refuses** (important to say): fewer than 600 clean-ish rows; target barely learnable (not at least 0.05 above shuffled-label performance); noise > 0.01 (non-exact repairs cannot be certified); F1 and AUC disagree (flagged); label and feature-anomaly repairs are never applied automatically. Honest limits: Random Forest only; injects random errors, so precision can be optimistic for structured real errors; not shown to beat the transferred evidence for the safe-repair decision.

---

## Part 10. The three ways we measure "did cleaning help?"

| Tool | Test rows | What it proves | What it cannot prove |
|---|---|---|---|
| **Controlled proof** (Phase 20/21) | **clean** held-out rows (errors injected only into training rows) | cleaning improved a model, with a known clean answer | works on benchmark data, not your file |
| **Model panel** (Phase 19) | untouched held-out rows of **your own file** (as dirty as the file) | the model predicts *this file's* rows better/worse | that a repair was *correct* |
| **tuned_check.py** (Phase 27) | same as panel, any file | whether per-model tuned cleaning helps on that file | a gain is not "proof the repair was right" |

**Tuned cleaning** (the method that produced the one real gain): for each model, try repairs (default, aggressive, each repair alone) on a validation slice (25%) of the *training rows only*; keep a repair only if it beats the raw rows by more than 0.005; otherwise keep the raw rows. Then score on held-out test rows. Choosing per model matters because repairs help models differently.

**Results of controlled proof:**
- Adult (classification, 5 seeds): logistic regression **+0.015** macro-F1 (5/5 seeds, 44% of the damage recovered); RF, GB, MLP only within noise. The pre-registered criterion (improve at least two families) was **not met** — reported, not hidden.
- California Housing (regression): Ridge **+0.037** R², gradient boosting **+0.016**, none worse; the criterion **passed**.

---

## Part 11. The research story (phases in plain words)

| Phase | What | Key result |
|---|---|---|
| 1-2 | Clean baseline on UCI Adult (predict income >50K) | F1 0.680, accuracy 0.851, AUC 0.903 |
| 3-4 | Corruption engine with ground-truth logs | five error types, rates 5/10/20% |
| 5 | Detectors graded against ground truth | table in Part 5 |
| 6 | Quality report | flagged rate kept separate from detector reliability |
| 7 | **Impact**: how much each error hurts F1 | label errors dominate (0.065 loss at 20%); duplicates indistinguishable from zero |
| 8 | Oracle repair: how much is recoverable in theory | |
| 9 | Real automated repair | feature-corruption repair is net-harmful; label repair precise but recall tiny |
| 10 | Prioritisation: Priority = impact − risk + repairability | label errors first, feature corruption last (RF) |
| 11 | AI-readiness | no composite score; label-error detection rate is *inverted* (r = −0.95) |
| 13 | Second dataset (Bank Marketing) | transfer works with ROC-AUC, not F1 (F1 unreliable on imbalanced targets) |
| 14 | Inconsistent values; irrelevant-feature check | repair exact but model recovery negative |
| 15 | Joint outliers | validation **failed** on real data → opt-in, flag-only |
| 16 | **Is it the errors or the model?** Re-ran with gradient boosting | 5 of 10 pre-registered criteria passed, 5 failed: damage *size* is model-dependent |
| 17 | Why KNN hurts GB | harm sits in zero-heavy columns; fixed by median there |
| 18 | Baseline health report | frozen snapshot at upload, SHA-256, no health score |
| 19 | File formats + four-model Model panel | |
| 20 | Controlled proof | see Part 10 |
| 21 | **Regression** support | passed its criterion |
| 22 | Excel support | |
| 23 | Validation on 5 new real files | found a real bug (calibrated default auto-applied 7,829 Ames imputations, 6,969 outside calibration scope), fixed |
| 24 | Clustering | **deliberately out of scope** (no ground truth); spec deferred |
| 25 | Placeholder codes (`-1`) | found by you on DS-jobs: tool said 0 missing; now finds 372 feature cells + 50 target rows |
| 26 | Words, declared values, two diagnostics | DS jobs 606 cells (6.51%); recall/precision 1.00 on 63+9 planted cases |
| 27 | Exploration: does cleaning improve a model? | see Part 12 |

---

## Part 12. Phase 27 and the independent-file results (be exact here)

- Searched all real files with the Model panel (217 comparisons): 3 improvements and 11 harms.
- The strongest, **Bank Marketing logistic regression**, was confirmed on a fresh sample and seeds with the claim written first: **+0.030 macro-F1, 5 of 5 seeds**, about 5× the noise floor. Re-run with the generic runner on a different split: **+0.036, 5 of 5**. RF and GB did not gain.
- Caveats: one dataset; held-out rows as dirty as the file; the default evidence-based file applies **nothing** on Bank — the gain comes from the validation-based selection; it is the best of 217 comparisons (selection effect).
- A stronger noisy-label filter helped at high noise but hurt clean data → **not added** to the product (a negative result, kept).

Independent files run with `tuned_check.py`:

| File | Target | Result | Why |
|---|---|---|---|
| Messy Employee (yours) | 5 targets | **No gain** | labels unrelated to the columns (random-looking data); models at chance |
| UCI Credit Approval | class | **No gain** (macro-F1 0.83-0.87) | only 5% of rows have a blank; little to fix |
| UCI Horse Colic | outcome | **No gain** | 19% missing but the validation slice (~74 rows) is so small the repair choice is noise |

**The honest one-liner:** "The tool reliably *measures* impact and *refuses* repairs that don't help. A real model improvement is shown once (Bank logistic regression); on three other independent files it correctly reported no gain rather than inventing one."

**When a gain appears you need three things together:** a *learnable* target, *real damage* in the data, and a *model that is hurt by that damage*. Employee had damage but no signal; Credit Approval had signal but little damage.

---

## Part 13. The app

- **Backend** `backend/main.py`: FastAPI. Receives the upload, stores the dataset, runs the audit as a background job, and serves results (the model panel runs in the background after the audit, about a minute).
- **Frontend** `frontend/`: React + Vite + TypeScript; built into `frontend/dist` (git-ignored) and served by the backend at `http://localhost:8000`.
- **Pages in order:** Start (upload, choose target/task, switches, placeholders card) → Overview → Baseline health → Quality audit → Impact → Repair plan → Review (human override) → Outputs (downloads) → Calibration → Model panel → Controlled proof → Readiness → Features → Research.
- **Run:**
  ```
  cd "/Users/ishikamahadar/Documents/Documents/Projects /DSc CP"
  .venv/bin/python -m uvicorn backend.main:app --port 8000
  ```
  (Open http://localhost:8000. The folder name contains a space, so the project launches Streamlit with `python -m streamlit`; the same `python -m` form is used for uvicorn.)
- Audits are kept **in memory**: restarting the server loses them, so re-upload.

---

## Part 14. Testing and quality of the engineering

- ~15 test scripts (`src/test_*.py`); `test_taskclean.py` checks end-to-end invariants (every changed cell is logged, logs match the file, detection matches injected ground truth).
- **Mutation checks**: deliberately break a rule and confirm a test fails (for example, Phase 26's test M5 initially *survived* a changed threshold, so a "half-matching" column was added to catch it).
- Real bugs found by validation (good to mention): backwards abbreviation rule (Phase 14), Ames over-imputation (Phase 23 A1), F1-vs-AUC disagreement (Bank), a hold-back annotation bug, a glued-number regex edge case.
- Everything run is recorded in `results/training_log.csv`.

---

## Part 15. What is proven and what is not

**Solid (say with confidence):**
- It detects and logs problems; missing and duplicate detection is exact on injected errors.
- It measures per-error impact in controlled experiments; label errors are the most damaging on Adult (under both RF and GB).
- Its safety policy is conservative; on default evidence it applies almost nothing and says why.
- It reads real-world quirks (encodings, `?`, `-1`, `Unknown`, Excel, currency strings).
- It reports no-gain honestly.

**Not proven (never overclaim):**
- That it improves models in general: shown on **one dataset, one model** (Bank logistic regression).
- That the held-out gain means a repair was *correct* (held-out rows are as dirty as the file).
- That calibration beats the transferred Adult evidence.
- Outlier and feature-anomaly detectors are weak (precision 0.09-0.28).
- Damage *size* is model-dependent (Phase 16); the ranking at the bottom is model-dependent.
- Clustering is unsupported (a deliberate scope decision).
- Pre-registration is a self-imposed discipline, not independently verifiable.

---

## Part 16. How to present it (and what never to say)

**Structure for the talk (8-10 minutes):**
1. The problem: blind cleaning, no measurement (1 min).
2. The idea: impact-aware cleaning (1 min).
3. Method: controlled experiments, detectors, repair, policy (3 min, use the diagram in Part 4).
4. Live demo (3 min): upload the demo file or Bank Marketing, show Quality, Impact, Repair plan, Model panel.
5. Results and honest limits (2 min).

**Demo plan:** `Use demo dataset` for a one-click run (6,300 Adult rows with all five errors injected), or upload Bank Marketing and show the Model panel with logistic regression improved. Start the server first and check http://localhost:8000 loads.

**Never say:**
- "It improves any model." → say "It measures whether cleaning helps and gates repairs; one confirmed gain so far."
- "It is proven." for the Bank gain → say "confirmed on a fresh sample, one dataset".
- "The tool cleaned the Employee file." → it filled blanks and found no other problems; it correctly reported no gain.
- "Pre-registered" as if audited → say "specified in advance in our own spec files".
- "No bugs." → mention bugs found and fixed; examiners trust that more.

---

## Part 17. Likely questions and answers

**Basics**

1. *What does your project do?* See Part 0.
2. *What is the target column?* The column you want to predict.
3. *Classification vs regression?* Category vs number; the tool supports both; not clustering.
4. *Why four models?* The effect of dirty data depends on the model; a repair can help one and not another.
5. *Why do you not clean everything?* Because "detected" does not mean "safe to repair" and some repairs hurt (feature-corruption repair was net-harmful at every rate).

**Method**

6. *How do you know a problem hurts the model?* Controlled experiment: inject one error into clean training data at a known rate, retrain, score on clean test rows, compare with the clean baseline (the drop is the impact).
7. *Why a clean test set?* So we measure true performance, not performance on equally dirty data.
8. *How do you detect wrong labels without knowing the truth?* Confident learning with out-of-fold predictions (Part 2.12).
9. *Why out-of-fold?* So a row is not judged by a model that memorised it.
10. *Why F1 and not accuracy?* Accuracy misleads on imbalanced classes; macro-F1 treats classes equally. We also check ROC-AUC because F1 misled on Bank Marketing.
11. *What is the noise floor and why?* The amount a score moves by chance, measured by a null experiment; effects smaller than it are not claimed.
12. *What is a null experiment?* Retrain on a random 90% of the training rows; nothing was cleaned, so any change is noise.
13. *Why 5 seeds and 80% agreement?* One run can be lucky; requiring most seeds to agree guards against that.
14. *Why median instead of KNN for some columns?* Phase 17: KNN invented values in zero-heavy columns and hurt gradient boosting.
15. *How do you decide a repair is safe?* The 7 conditions in Part 8.
16. *Why is nothing applied automatically on Adult evidence?* Duplicate removal failed the gradient-boosting gate at 5% and 20%; the pre-registered rule says the stricter verdict wins.
17. *Why are label repairs never automatic?* They rewrite the target; calibration injects random errors, while real label errors are structured; recall is low.
18. *What does calibration do?* Re-runs the experiments on your own data to base decisions on measurements (Part 9).
19. *When does calibration refuse?* Under 600 clean-ish rows, unlearnable target, noise above 0.01.

**Results**

20. *Does your tool improve model performance?* On one real file and one model, yes (Bank Marketing logistic regression +0.03 macro-F1, 5/5 seeds). On the other independent files it found no gain and said so. The Adult controlled proof improved logistic regression (+0.015); the regression proof improved Ridge and GB.
21. *Why didn't it work on the employee file?* The labels are unrelated to the columns (models score at chance), so there is nothing for cleaning to protect.
22. *Why not on Horse Colic?* Lots of missing data, but the validation slice for choosing a repair is about 74 rows, so the choice was noise; honest result: no gain.
23. *Isn't the Bank result cherry-picked?* It was the best of 217 comparisons, so we **confirmed it on a fresh sample and fresh seeds with the claim fixed in advance**, then re-ran it with the generic runner. Still one dataset, so we call it one confirmed case.
24. *Which error hurts most?* Label errors (0.065 F1 at 20% under RF; about 0.45 times that under GB).
25. *Which detector is weakest?* Outliers (precision 0.09-0.21 on real files) and feature anomalies (0.28).
26. *What failed?* Phase 16 (5 of 10 criteria), Phase 20's criterion (only logistic regression improved), joint outliers validation, the noisy-label filter, and the independent files. All reported.

**Honesty and design**

27. *What does "pre-registered" mean here?* We wrote criteria in a spec before running; it is self-imposed and not independently timestamped.
28. *Why no readiness score?* Phase 11 found no correlation between detected rate and damage (r = −0.24, p = 0.39), and label-error detection was inverted.
29. *Why is clustering excluded?* No ground truth, so a before/after comparison could only measure stability; a deliberate scope decision.
30. *Can the held-out gain be trusted?* It shows the model predicts this file's rows better, not that a repair is correct, because the held-out rows are as dirty as the file.
31. *What if the file has a `-1` that is real?* It is flagged but never filled; the Start page has a switch to keep those values as real.
32. *Why isn't 0 or 999 detected automatically?* Real zeros appear in more than 20 columns across the project's files; guessing would flag correct data. You declare them.
33. *How do you avoid data leakage?* Repairs are fitted on training rows only; test rows are never repaired in the panel; identifier columns are dropped from models.
34. *What are the limitations?* Part 15.
35. *What would you do next?* Noise-aware repair selection (several validation splits, margin scaled to noise), more real datasets, a pre-registered run on a new file.

**Technical**

36. *Why FastAPI + React?* Separate backend (Python, the ML code) and frontend (dashboard); the older Streamlit app remains.
37. *Why two stages (audit then apply)?* So repairs can be re-chosen without re-running slow detectors.
38. *How big a file can it take?* The cheap checks use every row; slow detectors run on a stratified sample above 30,000 rows; Excel with half a million rows read in 11 s.
39. *Which file types?* CSV, TSV, Excel (.xlsx/.xlsm), JSON, JSON Lines, Parquet; legacy `.xls` is refused with a message.
40. *What did validation on real files find?* A real over-imputation bug on Ames (fixed), and the outlier detector's weakness.

---

## Part 18. Glossary (quick lookup)

- **Target / label**: the column to predict.
- **Feature**: any other column used to predict.
- **Impact / damage**: score lost because of a data problem.
- **Recovery**: score won back by a repair.
- **Ground truth**: the known correct answer, available only in controlled experiments.
- **Imputation**: filling blanks.
- **Winsorizing / clipping**: capping extreme values at a bound.
- **Out-of-fold**: a prediction from a model that never saw that row.
- **Noise floor**: chance wobble in a score.
- **Seed**: number that fixes randomness.
- **Evidence status**: validated / weak / inverted / not estimable.
- **Safe-auto**: a repair the evidence supports applying without asking.
- **Flag-only / report-only**: shown to the user, never changes the file.
- **Held-out rows**: rows kept aside for scoring.
- **Arm**: one version of the training data (raw, default, aggressive, a single repair, tuned).
- **Tuned cleaning**: choosing the repair per model on a validation slice of the training rows, keeping raw if nothing clearly wins.
- **Leakage**: information from the test rows reaching the training step.

---

## Part 19. Key numbers cheat sheet

| Item | Number |
|---|---|
| Adult clean baseline (RF) | F1 0.680, accuracy 0.851, ROC-AUC 0.903 |
| Corruption rates | 5%, 10%, 20% |
| Seeds | 42-46 (5) |
| Split (research) | 70/15/15 |
| Label-error damage at 20% (RF) | 0.065 F1 |
| Detector precision / recall | missing 1.00/1.00; duplicates 0.99/1.00; labels 0.77/0.28; outliers 0.18/0.67; features 0.28/0.26 |
| Safe-auto rule | risk 0, precision ≥ 0.95, noise floor ≤ 0.01 |
| Noise floor minimum | 0.003 |
| Seed agreement needed | ≥ 80% |
| Tuned-selection margin / validation share | 0.005 / 25% |
| Placeholder codes rule | ≥3 occurrences, ≤50% of column, others ≥ 0 |
| Calibration minimum | 600 clean-ish rows |
| Label detector | RF 200 trees, 5-fold, confidence ≥ 0.9 |
| Wrong-target detector (regression) | residual > 4 robust sigmas, 5 bands |
| Model-row limit | 30,000 (sample above) |
| Adult controlled proof | logistic regression +0.015 macro-F1 (5/5 seeds) |
| California regression proof | Ridge +0.037 R², GB +0.016 |
| Bank Marketing (real file) | logistic regression +0.030 (confirmation), +0.036 (re-run), 5/5 seeds each |
| Independent files | Employee, Credit Approval, Horse Colic: no gain |
| DS-jobs missing | 606 cells + 50 target rows = 6.51% (was 0% before Phase 25) |

---

## Part 20. Suggested study order for tonight

1. Part 0 (say the pitch aloud three times).
2. Parts 1-2 (concepts). If short on time, make sure you can explain: train/test split, F1, noise floor, out-of-fold, controlled experiment.
3. Parts 4, 8, 10 (the architecture, the policy, the three measurements).
4. Part 12 and Part 15 (what is proven and what is not).
5. Part 17 (questions). Cover the answers, say yours aloud, compare.
6. Run the demo once end to end (Part 16).
