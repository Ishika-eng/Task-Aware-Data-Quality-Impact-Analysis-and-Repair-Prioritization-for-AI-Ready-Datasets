# Phase 26: word placeholders, placeholders you declare, and two diagnostics

Written **2026-11-18, before any code**. Fixed in advance; outcomes are reported whatever they are. Anything that has to change afterwards
is appended as a visible Amendment. Additive: files without any of these patterns must produce byte-identical results (S0), frozen research
files untouched, new results go to `results/phase26_*`. Status: **built as specified (2026-11-18); results and amendments below.** Builds on Phase 25 (uncommitted at the time of writing).

## Why

Phase 25 closed with a list of things the tool still does not handle on `Uncleaned_DS_jobs.csv`: words such as `Unknown`, the values `0` and
`999`, `Salary Estimate` ranges, the rating glued onto `Company Name`, a genuine `-1` that gets flagged, and no per-column opt-out. The user
asked for them to be handled. They are not one problem, and the evidence below shows they cannot all be handled the same way.

## Inspection that preceded this spec (read-only, no code, recorded so the rules are not tuned afterwards)

A scan of every dataset in the project (Adult original/clean/demo, Bank Marketing, California, Credit-g, Heart, Telco, Titanic, Ames, Real
Estate, Online Retail II, DS jobs) found:

| Pattern | What the scan showed | Consequence |
|---|---|---|
| Word placeholders | `unknown`: Bank `V2` 288, `V4` 1,857, `V9` 13,020 (28.8%), `V16` 36,959 (81.7%); DS jobs `Size` 17, `Type of ownership` 4, `Revenue` 213 (`Unknown / Non-Applicable`, 31.7%); Online Retail II `Country` `Unspecified` 310, `Description` `missing`/`MISSING` 26. **Not placeholders:** `none` (Credit-g `other_parties` 90.7%, `own_telephone` 59.6%), `other`, `Other-service`, `Other-relative`. | Auto-detect a short, conservative word list. `none` and `other` are legitimate categories and must stay out of it. |
| `0` | Real zeros everywhere: Heart `oldpeak` 32.7%, Telco `tenure` 0.2% (new customers), Real Estate `house age` 4.1%, Titanic `Fare` 1.7%, Ames `TotalBsmtSF` 2.5% (no basement), `PoolArea` 99.5%, Bank `V15`/Adult `capital-gain` 82-92%. A "spike far from the other values" rule would fire on Ames `TotalBsmtSF` (zero, nearest other value 105) which is correct data. | **Not detectable from the data alone.** `0` cannot be auto-detected without producing false alarms on real columns. |
| `999` | California `Population` 22 rows, Ames `1stFlrSF` and `GrLivArea` 3 rows each: all real values. | Same: not auto-detectable. |
| Numeric range text | `Salary Estimate` is 100% `$75K-$131K (Glassdoor est.)`-style text, 30 distinct values. `Size` and `Revenue` are bands with words ("1001 to 5000 employees") and must not match a strict numeric-range rule. No other file has it. | Detect and report. Converting is feature engineering (see below). |
| Number glued onto a text column | `Company Name`: 622 of 672 cells (92.6%) end with `\n<number>`; that number equals the `Rating` column in **99.5%** of those rows; the 50 cells without it are exactly the 50 rows where `Rating` is `-1`. Distinct names are **432 with and without** the suffix (a company has one rating), so the suffix changes no grouping. | A leak and a readability defect, not a category-splitting one. Detect and report; the wording is stronger when the matching column is the target. |

Two conclusions this forces, stated before building so they are not rationalised later:

1. **`0` and `999` will not be auto-detected.** The only sound way to handle them is to let the user declare them, per column. The same
   mechanism also answers "a genuine `-1` gets flagged" and "no opt-out".
2. **Ranges are not converted to numbers.** Midpoint conversion would either write numbers into a column of text ranges (the cleaned file
   would mix types) or change what the model sees without any evidence it helps; the tool's own panel measures cleaning, not feature
   engineering. The honest handling is to detect it and say so.

## What is built

### F1. Word placeholders (automatic, held back like codes)

* **Word list (whole cell, case-insensitive, stripped):** `unknown`, `unspecified`, `not available`, `not applicable`, `non-applicable`,
  `undefined`, `missing`, `tbd`, `no data`. A cell also matches when it is such words joined by `/` (`Unknown / Non-Applicable`). `none`,
  `other`, `n/a` (already a token), anything with extra words ("missing (wrongly coded?)") and the `-1` family do not belong here.
* **Same conditions as numeric codes (Phase 25 R2/R5):** the column needs at least **3** such cells, and they may fill at most **50%** of it; above
  that they are a category and the column gets the existing "kept as a category" note (Bank `V16`).
* **Behaviour = Phase 25 codes:** counted as missing everywhere in the audit and the Baseline, analysis copy only, never auto-applied to the
  evidence-based file (held back, same Data note / Repair plan alert / Review queue card), filled by the aggressive file and by a human
  override. They share the existing `codes` bookkeeping, so the UI and `held_back_summary` need only their wording widened: "a code such as
  -1 or a word such as Unknown".
* Words and `-1` codes in the same column are combined into one count.
* A text column that is numeric apart from such words now parses as numeric (as it already does for `?` and `N/A`).

### F2. Placeholders you declare (answers `0`, `999`, any other value, and the opt-out)

* **Request:** `declared_missing: {column: [value, ...]}` and `keep_values: [column, ...]` on the audit request; the Start page gets a
  "Placeholders" card with, per column, a box for values that mean missing, and a list of the columns auto-detection found with a "treat as
  real values" switch.
* **Declared values = known missing.** A cell is declared when its stripped text equals a declared value (case-insensitively), or, in a numeric
  column, when it equals the declared number. These cells are *ordinary missing values*: no code hold-back (the human already made the
  decision), so the evidence-based file imputes them exactly when it imputes other missing cells in that column. The log keeps the original cell.
* **`keep_values`:** for the listed columns F1 and Phase 25 detection is switched off, including their notes; the values stay real values.
  Declared values for the same column still apply.
* **Targets:** a declared value in the target makes that row a **missing target** (regression: existing Phase 25 path; classification: a
  missing label). Never imputed, never relabelled, excluded from training and evaluation, as for Phase 25 R4.
* **Where it applies:** the audit, the calibration reference's completeness test, the Model panel, the Overview Data note ("You declared
  `0` as missing in `Cholesterol` (172 cells)"), and a "Declared by you" line in the outputs `summary`. **It does not apply to the stored
  Baseline snapshot**, which is computed at upload before the user can declare anything; the Baseline page says so.
* Limits: at most 20 values per column, each at most 40 characters; a column or value that does not exist is a 400, not silently ignored; a
  declared value that matches no cell is reported ("matched no cells") and not an error.

### F3. Two diagnostics (report only; the data and the repairs are untouched)

* **Numeric ranges stored as text.** A text column where at least **90%** of non-missing cells match `^[$£€]?\d[\d,.]*[KkMm]?\s*[-–]\s*[$£€]?\d[\d,.]*[KkMm]?(\s*\(.*\))?$`
  gets a Data note: "`Salary Estimate` holds numeric ranges stored as text (30 distinct values), so it is analysed as a category; split
  it into a low and a high column to use it as a number." No conversion, no repair.
* **A number glued onto a text column.** A text column where at least **50%** of non-missing cells end in a line break plus a number, and that
  number equals the value of another numeric column of the same row in at least **95%** of those rows, gets a Data note naming that column
  ("`Company Name` carries `Rating` as a second line in 93% of cells"). If that column is the **target**, the note adds that the feature
  contains the answer and should be dropped before modelling. A text column with the suffix but no matching column gets no note (no evidence).

## Not built, and why

* **Auto-detecting `0`, `999` and other positive codes:** the scan shows real zeros in more than 20 columns and real 999s in three, across nine files; any rule that
  catches Cholesterol-style zeros would also fire on these. Left to the user (F2).
* **Converting ranges, splitting `Company Name`, rewriting the glued rating in the file:** feature engineering or cell rewriting without
  calibrated evidence (F3 explains).
* **A `0` or `999` suggestion list in the UI** ("this column looks like it might use 0 as missing"): any threshold is a guess I cannot
  validate; the Placeholders card only lists what the tool actually detected.

## Pre-registered criteria

* **S0 nothing else moves.** All test scripts pass (`test_taskclean`, `test_selfbench`, `test_phase14`..`test_phase25`). Files with no word
  placeholders, no declaration and no F3 pattern reproduce byte-identical `dataset_cleaned*.csv` (Titanic, Ames, Real Estate, Concrete, Adult,
  California, Credit-g, Heart, Telco, demo dirty Adult) and identical counts. Credit-g `none`/`other`, Adult `Other-service`, and Ames
  `Exterior2nd = Other` are **not** flagged.
* **S1 mechanical, direct counts.** DS jobs (target `Rating`, regression): feature placeholder cells grow from 372 to **606** (`Size` 44,
  `Type of ownership` 31, `Revenue` 240, the others as in Phase 25) with the target's 50 unchanged, so Baseline missing cells are **656** (6.51% of
  10,080). Bank Marketing: `V2` 288, `V4` 1,857, `V9` 13,020 flagged; `V16` (81.7%) **not** flagged and given the category note. Online
  Retail II: `Country` 310, `Description` 26 (the cell "missing (wrongly coded?)" is not).
* **S2 injected truth.** Plant `Unknown`, `unspecified` and `Missing` at 1%, 5%, 20% of a clean categorical column (Adult `workclass`, Credit-g
  `purpose`) and into a numeric-as-text column, 3 seeds: **recall 1.00, precision 1.00**. Must **not** fire: 2 cells only; 60% of the column;
  `none`; `other`; "Unknown Soldier" (extra words). Declared:
  plant `0` into a numeric column that also holds legitimate zeros in another column; declaring it in the first yields recall 1.00 and
  precision 1.00 for that column and **zero** change anywhere else. `keep_values` on a column with genuine `-1` and `Unknown` produces no flag,
  no held-back note and results identical to the Phase 25 behaviour on a file without them.
* **S3 policy.** Word placeholders: the evidence-based file leaves every flagged cell unchanged, the aggressive file fills them, the log shows
  the original word, a human override applies them. Declared cells: never counted as held back for codes; applied exactly when the other
  missing cells of that column are applied; the cleaned file at declared-and-untouched cells equals the original.
* **S4 targets.** A declared value in a regression target and in a classification target both give rows without a target, never repaired and
  never counted as wrong-target or label errors.
* **S5 user-side check (browser, real file).** DS jobs through the Start page: the Placeholders card lists the auto-detected columns; the
  Overview, Repair plan and Review queue show the widened wording and the new counts; declaring `-1` or `Unknown` in `Competitors` and
  toggling `keep_values` on `Revenue` change the audit as the criteria say; the two F3 notes appear on `Salary Estimate` and `Company Name`
  (with the stronger target wording when `Rating` is the target). I report exactly which pages I read.
* **S6 reported, not claimed.** What the new counts do to calibration (it may decline further), to the Model panel on DS jobs and on Bank
  Marketing, and the Bank change in the evidence-based file. No improvement is claimed from this phase: it fixes what the audit *sees*.
* **Predictions (scored in the Results section):**
  P1 Bank `V16` gets the majority note and no other Bank column fires beyond `V2`, `V4`, `V9`. P2 DS jobs Baseline missing cells equal 656.
  P3 calibration still declines on DS jobs (fewer than 600 clean-ish rows). P4 the Bank evidence-based cleaned file changes only where
  Phase 25 did not already change it, i.e. `unknown` cells stay `unknown`. P5 `Company Name` is reported with `Rating` and the 99.5%
  figure is reproduced by the tool. P6 no file in S0 changes. P7 the Model panel shows no gain from this phase on DS jobs (it is a
  detection phase).

## Known limits stated in advance

* Declared values apply to the audit and not to the stored Baseline.
* The word list is English only and short on purpose; a user with `inconnu` or `N.D.` uses F2.
* A real category literally called `Unknown` in a column below 50% is treated as missing (and held back, not filled): `keep_values` is the way out.
* F3 diagnostics need a numeric sibling column; a rating glued onto a name with no matching column is not reported.


---

# Results (2026-11-18)

Code: `src/taskclean.py` (`_is_word_placeholder`, `_word_codes`, `_declared_mask`, `_clean_declarations`, `_diagnostic_notes`, `_prepare`, `_apply`, `audit_dataset`),
`backend/main.py` (`AuditRequest.declared_missing`/`keep_values`, a 400 on a bad declaration), `frontend/src` (the Start page "Placeholders" card, wording, the Baseline footnote).
Tests: `src/test_phase26.py` (11 tests). Mutation check: eight deliberate breakages of the rules (the word list gaining `none`/`other`, the majority cap removed for words,
declared numbers never matching, `keep_values` ignored, the glued-number threshold removed, a classification target declaration ignored, the minimum count 3 lowered to 1, declared cells held
back like codes) were each run against the tests: **seven were caught at once, one survived** (Amendment A2) and the test was strengthened until it was caught. Validation:
`src/validate_phase26.py`, results in `results/phase26_*.json`. All 15 test scripts pass on the final code (`test_taskclean`, `test_selfbench`, `test_phase14`-`23`, `25`, `26`).

| Criterion | Result |
|---|---|
| **S0** nothing else moves | **Pass for every file without the patterns; the three files with them change exactly as scoped.** Cleaned, aggressive and repair-log hashes, proposal and applied counts and the Data notes are **identical** with the Phase 26 features on and off for Adult (original and clean), California, Credit-g, Heart, Telco, Titanic, Ames and Real Estate; `demo_dirty_adult.csv` cleaned files are byte-identical to Phase 25. The Phase 23 re-runs of Titanic, Ames and Real Estate give the same decisions and counts (largest float deviation 0.083%, Ames). `Credit-g none/other`, `Adult Other-service` and `Ames Exterior2nd = Other` are not flagged. **Not run: Concrete** (`xlrd` is not installed, as in Phase 25). "Off" means the same code with the word list emptied and the F3 diagnostics returning nothing (Phase 25 behaviour for an audit without declarations), a like-for-like A/B on one code base. |
| **S1** mechanical, direct counts | **Pass, exact.** DS jobs: feature placeholder cells **372 to 606** (`Size` 44, `Type of ownership` 31, `Revenue` 240, the others unchanged), target 50 unchanged, Baseline missing cells **656 = 6.51%** of 10,080 (read on the Start page). Bank Marketing: `V2` 288, `V4` 1,857, `V9` 13,020 flagged; `V16` (81.7% `unknown`) and `V14` (81.7% `-1`) **not** flagged and given the category note. Online Retail II: `Country` 310, `Description` 26; the re-run of the Phase 23 comparison differs only by those +336 missing cells (111,236 vs 110,900) and the sampled rows that follow from them. |
| **S2** injected truth | **Pass.** Words: **63 cases** (`Unknown`, `unspecified`, `Missing` into Adult `workclass`, Credit-g `purpose`, and a numeric-as-text column; 1%, 5%, 20%; 3 seeds), **recall 1.00, precision 1.00, no other column flagged**. Declared: 9 cases (zero planted into California `AveBedrms`, 1/5/20%, 3 seeds): the declared cells are exactly the planted ones and **no other column changes**. `keep_values` on a column with genuine `-1` and one with `Unknown`: both flagged without it, neither with it. Unit counter-cases that must not fire: 2 cells, 60% of a column, `none`, `other`, `Other-service`, "Unknown Soldier", "Unknown, TX", "missing (wrongly coded?)". |
| **S3** policy | **Pass in unit tests** (30 word cells kept in the evidence-based file, filled by the aggressive file and by an override, log shows the original `Unknown` and "a word such as 'Unknown'"; declared cells imputed, logged "declared missing by you" with the original `0`, never in the held-back note; one idempotent note). **Not shown on screen:** the Ames file with 10 planted `Unknown` cells, run through the browser under calibration, got "weak" evidence for missing values (harm risk 0.003), so nothing was auto-applied and there was nothing to hold back. The hold-back screens were not seen for words. |
| **S4** targets | **Pass in unit tests:** a declared value in a regression target (15 rows) and in a classification target (`Unknown` labels) gives rows without a target, excluded from the wrong-target/label detectors, never imputed or relabelled even in the aggressive path. |
| **S5** user-side check | **Partial, and only these pages were read.** Real file `Uncleaned_DS_jobs.csv`, through the browser: the Start page after upload (Baseline 6.51%, the Placeholders card listing the eight detected columns with `Unknown / -1` counts), a run with `Rating` as regression target, `-1` declared in `Competitors` and `Revenue` ticked as real values: the Overview Data notes (345 code cells and 21 word cells with `Revenue` absent, 501 declared cells in `Competitors`, the two F3 notes, the stronger wording for `Rating`), and the Repair plan counts (867 candidates = 345 + 21 + 501). **Not read** for this run: Review queue, Outputs, Model panel, Quality, Readiness. The Baseline footnote about declarations was added after that browser pass; it was then viewed on the Baseline page of the DS-jobs audit (no other change to that page was checked). |
| **S6** reported, not claimed | See Findings. |

## Predictions scored

P1 **right** (Bank `V16` gets the category note; only `V2`, `V4`, `V9` are flagged). P2 **right** (656). P3 **right** (calibration still declines on DS jobs: 541 clean-ish rows, 600 needed). P4 **right, but vacuous**: the Bank evidence-based file is unchanged (0 cells modified) with the features on or off, because
calibration approved no missing-value repair on Bank either way; the held-back path was therefore not exercised on a real file. P5 **right** (`Company Name` is reported with `Rating`, 93% and 99.5% reproduced by the tool). P6 **right** for every file in S0 without the patterns. P7 **right**: on DS jobs every arm of the Model panel is "within noise".

## Findings

* **F1 (what Bank Marketing shows).** 15,165 cells that were read as the category `unknown` are now missing cells, so the aggressive file changes 20,422 cells instead of 5,230 (the extra 15,165 are the word cells, filled by the aggressive file only). The evidence-based file is unchanged. In the Model panel (Bank, 8,000-row sample)
  logistic regression is "improved" with and without this phase (F1 +0.024 off, +0.034 on), random forest and gradient boosting "within noise" (-0.007/+0.002 off, -0.004/+0.001 on), the neural net "within noise" off (+0.070) and "improved" on (+0.026). These are noisy single comparisons; **no gain is claimed from this phase**. Whether `unknown` in Bank is information rather than a gap is a domain question the tool cannot answer; it only stops hiding it.
* **F2 (DS jobs).** The model panel's raw-row scores moved a little because the words are now missing in the features too (Ridge 0.033 unchanged, random forest 0.200 to 0.206, gradient boosting 0.167 to 0.138). Cleaning still does not improve any model there (aggressive arm: Ridge -0.008, random forest -0.008, gradient boosting -0.016, all "within noise"); wrong-target flags 9 to 8.
* **F3 (the glued rating is a leak, not a category problem).** `Company Name` has 432 distinct values with and without the appended rating, so it changes no grouping; it matters because with `Rating` as the target it contains the answer. The tool now says so; it does not alter the file.
* **F4 (one test was too weak and is now stronger).** See A2.

## Amendments

* **A1 (implementation detail not in the spec).** The F3 diagnostics judge the pattern on at most the first 5,000 non-missing cells of a column (`DIAG_SAMPLE`), to keep the cost bounded on million-row files. The only file in the project where an F3 note fires is DS jobs (672 rows), so the cap changed no result; on a larger file a pattern present only after the first 5,000 cells would be missed.
* **A2 (a test survived its mutation).** Removing the 95% sibling-match threshold of the glued-number note was not caught, because the test's non-matching column matched 0% of rows. A column matching about half the rows was added; the mutation is now caught.
* **A3 (S0 method).** The first S0 comparison applied the repair types explicitly, which is a human override and fills coded cells; for Bank that showed the word cells being filled and was not the default file. It was replaced by two measurements: the explicit path (hashes, for files without the patterns) and the full product run with calibration (Bank), which is what a user gets.
* **A4 (wording).** The held-back note now ends with "run the audit again with that column set to keep its values", pointing to the new opt-out; the Phase 25 texts say "code such as -1 or a word such as 'Unknown'".
* **A5 (not covered).** Concrete was not scanned or re-run (no `xlrd`).

## What is still not handled

`0` and `999` and any other value are handled only by declaration. The word list is English only. A declared value applies to the audit, not to the Baseline. Ranges and the glued rating are reported, not converted. The hold-back of words was exercised in unit tests and not on a real file.
