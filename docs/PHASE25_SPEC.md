# Phase 25: numeric placeholder codes (`-1`, `-999`, ...) read as missing values

Written **2026-11-17, before any code**. Fixed in advance; outcomes are reported whatever they are. Anything that has to change afterwards
is appended as a visible Amendment. Additive: classification and regression behaviour on files without such codes must stay
byte-identical (S0), frozen research files untouched, new results go to `results/phase25_*`. Status: **built as specified (2026-11-17); results below.** Amendments are listed at the end.

## Why (a defect found by using the product, not by a benchmark)

The user uploaded `Uncleaned_DS_jobs.csv` (672 job listings scraped from Glassdoor, 15 columns) through the front end and chose `Rating` as a
regression target. The file marks "unknown" with `-1` in about nine columns. The tool reported **0 missing cells** (Overview, Baseline, Quality,
Readiness: "Good") because its placeholder list (`PLACEHOLDER_TOKENS`: `''`, `?`, `-`, `--`, `n/a`, `na`, `n.a.`, `#n/a`, `null`, `nan`) holds only
*text* tokens and the `-1` columns are either numeric or text `"-1"`. Consequences measured in that audit:

* Completeness read as "Good" on a file where roughly 4% of cells are unknown: a false assurance.
* `Founded = -1` (118 rows) was flagged as an outlier and **"clip to 1781"** was proposed, i.e. an invented founding year. The aggressive file
  applies it and the Model panel showed Ridge **-0.89** and the neural net **-3.79** R-squared against the raw rows. The evidence-based file
  applied nothing, so the safety gate protected the user; the defect is that the right diagnosis was never offered.
* The regression target `Rating` holds 50 `-1` rows. The wrong-target detector caught 20 of them and 15 legitimate ratings (7 of them 5.0).
* Same placeholder, different outcome: this is the same blind spot as the Ames "blank means absent" limitation (Phase 23 F2), but here the
  *mechanical* part (a code that cannot be a real value) is detectable.

## Inspection that preceded this spec (read-only, no code, recorded so the rule is not tuned afterwards)

A scan with a candidate rule over every dataset used so far, run **before** writing the rule below, found: `Uncleaned_DS_jobs` fires in 8
columns; `Bank Marketing` column `V14` (`pdays`) is **81.7% `-1`** and fires; no other file fires (Adult, California, Credit-g, Heart, Telco, Titanic,
Ames, Real Estate, Online Retail II both sheets, Demo dirty Adult). `Concrete_Data.xls` could not be scanned because `xlrd` is not installed in
the venv (its columns are physical quantities; S0 re-checks it through the Phase 23 script). The Bank finding is why rule R5 exists: in `pdays`
`-1` is a documented code for "not previously contacted", a category, not missingness. `Competitors` in the DS file (`-1` in 74.6%) is the same
situation. A rule without a majority limit would have turned both into false alarms.

## The rule (applies to the analysis copy only; the file and the repair log keep the original cell)

* **R1 candidate codes:** `-1, -9, -99, -999, -9999`. Positive codes (`9, 99, 999, 9999`) and `0` are *not* candidates: they collide with
  real ages, counts and amounts.
* **R2 numeric columns** (not boolean): let C be the candidate codes that occur at least **3 times** in the column. The column is flagged
  when (a) C is non-empty, (b) every remaining non-missing value is **>= 0** (so a column with genuine negatives is never touched), and
  (c) the cells holding a code are **at most 50%** of the column (R5).
* **R3 text columns** (after the existing token masking, and only when the column does not parse as all-numeric, in which case R2 applies to the
  parsed numbers): a stripped cell equal to `str(v)` or `str(v) + ".0"` for a candidate code, same conditions: at least 3 cells, at most 50% of
  the column, and the remaining non-missing values are not all numeric.
* **R4 target:** regression targets only (R2 on the numeric target). A flagged target cell becomes a **missing target**, the same existing
  path as a non-numeric target value ("treated as a missing target"): the row is excluded from training, evaluation and the wrong-target
  detector, and is **never imputed** (Phase 21: no silent target repair). Classification targets are never touched: `-1/1` class labels are legitimate.
* **R5 majority code:** when the cells holding a code exceed 50% of the column, nothing is flagged; an informational data note says
  "N% of `column` is `-1`; kept as a category ('not applicable'), not treated as missing".
* **R6 pandas 3:** string columns are detected with `is_string_dtype`, not `dtype == object` (known environment gotcha; my first scan
  missed them for exactly that reason).

## Behaviour once a cell is flagged

* It counts as a missing value everywhere the audit counts them (Overview, Quality, Readiness, Baseline "Missing cells", the placeholder
  detail line, the Model panel's feature preparation, the calibration reference's completeness test). Outlier and model-based detectors no
  longer see it, so the `Founded = -1` clip proposal disappears.
* **Never auto-applied.** Imputing a cell whose code *might* be a real value is exactly the guess the tool must not make alone, and the
  calibration measures random missingness, not codes. The evidence-based file leaves every code cell byte-identical and logs a proposal
  (`flagged_for_review`, method "`-1` read as a missing-value code"). The aggressive file imputes them like other missing cells. A human
  override of the missing-values repair on the Repair plan applies them. This is Amendment A1's `respect_scope` mechanism extended with a
  second hold-back reason, and it is shown the same way: a Data note, the Repair plan alert, the Review queue card, `held_back_summary`.
* The Overview Data note names the columns and counts ("422 cell(s) hold a code such as -1 and are treated as missing: Founded (118), ...").
  The Baseline snapshot applies the same rule before a target is chosen and so counts the would-be target as well; its existing footnote is
  extended to say so (as it already does for outliers).
* Text that currently says "placeholder cells (such as ?, N/A, -)" is updated to mention codes such as -1.

## Pre-registered criteria

* **S0 nothing else moves.** All existing test scripts pass unchanged (`test_taskclean`, `test_selfbench`, `test_phase14` to `test_phase23`).
  Files the scan found clean give identical results: the Phase 23 results for Titanic, Ames, Real Estate and Online Retail reproduce the same
  flagged counts and applied counts, Concrete is re-run through its Phase 23 script, and a run on `demo_dirty_adult.csv` produces byte-identical
  `dataset_cleaned.csv` and `dataset_cleaned_aggressive.csv` before and after. Bank Marketing `V14` is **not** flagged (R5) and the file produces an
  informational note instead.
* **S1 mechanical, DS jobs** (`Rating` as regression target). The audit flags exactly the cells a direct count finds: `Headquarters 31`,
  `Size 27`, `Founded 118`, `Type of ownership 27`, `Industry 71`, `Sector 71`, `Revenue 27` as features (**372 cells**) and `Rating 50` as the target (50 rows
  without a target); `Competitors` (501, 74.6%) is **not** flagged and gets the R5 note. Baseline missing cells equal **422** (4.19% of 10,080).
* **S2 injected truth, clean data.** Into a non-negative numeric column of clean data (Adult `age`, California `population`) plant `-1` and `-999` at 1%,
  5% and 20% of cells, 3 seeds each: **recall 1.00 and precision 1.00** (nothing else in the column or any other column flagged). Counter-cases
  that must **not** fire: a column with genuine negatives that contains `-1`; a column with only 2 cells equal to `-1`; a column that is 60% `-1`;
  a column where `-1` occurs and the rest contains `-5`; a classification target with labels `{-1, 1}`; an all-numeric text column of ZIP-like codes.
  Text columns get the same planted test with `"-1"` in a categorical column.
* **S3 policy.** On every file that fires: the evidence-based file leaves all flagged cells unchanged, the aggressive file leaves none, the
  log shows the original `-1`, the Model panel's default arm equals the raw arm on those cells, and a human override of the missing-values
  repair applies them.
* **S4 target.** DS jobs: the panel reports 622 usable rows, the wrong-target detector's flags contain **no** `Rating = -1` row, no repair is proposed for any of the
  50 target-missing rows, and the Overview note says so.
* **S5 user-side check through the browser.** Upload DS jobs, `Rating` regression, calibrated, panel on: Overview, Baseline, Quality, Features,
  Repair plan, Review queue, Task impact, Outputs and Model panel are read for the new counts, notes and wording; no console errors; the Completeness
  status is whatever the rules give (it is reported, not tuned).
* **S6 reported, not a criterion.** Model panel on DS jobs after the change (default and aggressive arms). Cleaning is **not** claimed to improve
  models; the criteria above are about detection correctness and safety.

## Predictions (made now, scored afterwards)

P1 The 118 `Founded` outlier proposals disappear. P2 The wrong-target flags no longer contain any `Rating = -1` row, and the 15 legitimate ratings that were
flagged (7 of them 5.0) are mostly still flagged (the detector is unchanged). P3 The aggressive arm no longer collapses (Ridge better than -0.89,
the neural net better than -3.79), but is still not clearly better than the raw rows. P4 The default arm still reads "no change" (by construction). P5 No
file other than DS jobs changes (S0). P6 Bank Marketing `V14` is not flagged. P7 Calibration still declines or runs as before on DS jobs. I do not know whether P3 holds.

## Work, in order

K-A `_find_codes` (one shared function used by `_prepare` and `health_snapshot`; target handling in `prepare_target`'s caller) and unit tests (S2
including counter-cases). K-B hold-back extended in `_apply` (S3, one idempotent note per reason) and `held_back_summary`. K-C front-end text
and footnotes. K-D validation script `src/validate_sentinels.py` with `results/phase25_*.json` (S0, S1, S4, S6) and a test file `src/test_phase25.py`.
K-E the browser pass (S5); README and findings. Each step reports what it measured before the next starts. The DS jobs file is copied to
`data/new/` (gitignored, not committed: the licence of the scraped data is unknown).

## Not covered (stated now, so it is not a surprise later)

* **Disguised missing values that are not numeric codes:** `Unknown`, `Not specified`, `Unknown / Non-Applicable` (present in DS jobs `Size`
  and `Revenue`), `0`, `1900-01-01`, `99`, `999`. Only the user knows these.
* **A genuine value that looks like a code:** a real `-1` in a column of otherwise non-negative numbers with at least 3 occurrences and under 50%
  would be flagged. That is why the evidence-based file never applies it on its own, and the cost is a held-back proposal the user can ignore.
  There is no per-column opt-out in this phase.
* **Formatted text that carries a number:** `Salary Estimate` ("$137K-$171K (Glassdoor est.)") and `Company Name` with the rating appended
  ("Healthfirst\n3.1") are not parsed. Parsing them would be a different feature (text-to-number extraction).
* Excel files: a numeric code in a spreadsheet is a number, so R2 applies; no extra work expected, one workbook case is added to S2.


---

# Results (2026-11-17)

Code: `src/taskclean.py` (`_numeric_codes`, `_text_codes`, `_prepare`, `_apply`), `src/health_snapshot.py`, `src/regression.py` (Amendment A1 below),
`backend/main.py` (`held_back_summary`), `frontend/src` (wording, `heldBackReasons`). Tests: `src/test_phase25.py` (9 tests, mutation-checked: removing the majority cap
or the "others are non-negative" condition each makes the counter-case test fail). Validation: `src/validate_sentinels.py`, results in `results/phase25_*.json`.
"Before" numbers come from the committed code (HEAD c905148, run in a git worktree with the same script), not from memory.

| Criterion | Result |
|---|---|
| **S0** nothing else moves | **Pass.** All 13 test scripts pass (`test_taskclean`, `test_selfbench`, `test_phase14`-`23`, `test_phase25`). The five Phase 23 files re-run to the **same decisions and counts** (Titanic, Ames, Real Estate, Online Retail II, Concrete: every count, flag, applied cell, policy action and panel verdict equal; floats within 0.16%, the largest on Ames; the committed code shows the same wobble against its own saved results). The only text that changed is the limitations wording, on purpose. `demo_dirty_adult.csv`: the cleaned file, the aggressive file and the repair log are **byte-identical** (SHA-256) before and after. Bank Marketing `V14` (81.7% `-1`) is **not** flagged and gets the informational note (`results/phase25_S0_bank.json`). |
| **S1** mechanical, DS jobs | **Pass, exact.** Flagged: `Headquarters 31`, `Size 27`, `Founded 118`, `Type of ownership 27`, `Industry 71`, `Sector 71`, `Revenue 27` = **372 feature cells**, and `Rating 50` as the target. `Competitors` (501, 74.6%) not flagged, R5 note shown. Baseline: **422 of 10,080 missing cells (4.19%)**. |
| **S2** injected truth | **Pass.** 36 planted cases (Adult `age`, California `Population`; codes -1 and -999; 1%, 5%, 20%; 3 seeds): **recall 1.00, precision 1.00, no other column flagged**. One Excel workbook case: all 20 planted cells found. Unit counter-cases that must not fire all hold (genuine negatives with -1; 2 cells; 60% -1; another negative present; classification labels -1/1; ZIP-like text; all-code column). |
| **S3** policy | **Pass.** Unit test, and through the browser on Ames with 10 planted codes under calibration: 860 applied (A1), 6,969 held back for calibration scope, **10 held back for codes**; the evidence-based file leaves all 10 as `-1` (log shows original -1, nothing applied), the aggressive file fills all 10, one idempotent note, `held_back_summary` reports the two reasons without overlap. Shown on Overview (Data notes), Repair plan and Review queue. |
| **S4** target | **Pass.** DS jobs: 50 rows without a target, **622 usable rows** in the panel (497 train, 125 test), `Rating = -1` in none of the wrong-target flags, 0 proposals for those rows, `Rating` cells identical to the upload in both files. |
| **S5** user-side check | **Pass, one defect found and fixed (Amendment A1).** For DS jobs the upload step, Overview, Baseline, Calibration, Quality, Readiness, Task impact, Repair plan, Review queue, Outputs and Model panel were read; for the Ames file with planted codes, Overview, Repair plan and Review queue and the repair-log API. Features was not re-read in this phase. No failed requests on the pages checked (the console showed 404s for audit ids from before a server restart). The Baseline *target block* still said "rows without a target 0" and "min -1" while the audit had 50 missing targets: fixed, re-checked (50, 7.44%, min 2). |
| **S6** reported, not claimed | DS jobs panel, committed code vs this code (same file, `Rating`): see Findings. |

## Predictions scored

P1 **right**: the 118 `Founded` outlier proposals fell to 0 (52 remain, on genuine early founding years). P2 **half wrong**: no `Rating = -1` row is flagged (right), but wrong-target flags
fell from 35 to 9 and none are 5.0 (I expected the detector to behave as before; it changed because the model no longer trains on the coded rows). P3 **right for the three models measured**:
the aggressive arm no longer collapses (Ridge -0.89 to -0.02, random forest -0.11 to -0.01, gradient boosting -0.08 to -0.02, all "within noise"); the neural net cannot be compared (skipped, see F2).
P4 **right** (the default file changes nothing). P5 **right**. P6 **right**. P7 **wrong**: calibration did not "run as before"; it now declines (F1).

## Findings

* **F1 (a consequence, correct).** Calibration on DS jobs ran before the change and **declines now**: "only 544 clean-ish rows remain ... (need at least 600)". The old reference was "clean-ish" only because 422 coded cells were invisible to it. Without
  calibration the Task impact page shows "no regression evidence yet", which is the honest state for a 672-row file with 18% of its rows incomplete.
* **F2 (the earlier model score was an artifact).** Raw-rows Ridge R-squared was **0.54** with the `-1` ratings in the target and is **0.03** without them (random forest 0.20, gradient boosting 0.17). The `-1` ratings sat on the same rows as `-1` in
  `Founded`, `Industry` and other columns, so the models could "predict" them. The panel now says the target is barely predictable for Ridge. Dropping 50 rows also takes the training set to 497, under the panel's existing 500-row minimum, so the
  neural net is skipped (a disclosed rule, not a new one).
* **F3 (what this changes for the user).** Cleaning still does **not** improve any model on this file: every arm is "within noise" or "no change". What changed is that the report no longer calls the file complete, no longer proposes an invented founding year,
  and no longer scores a model on placeholder targets.
* **F4 (limit, unchanged).** A real `-1` in a column of non-negative numbers is flagged (held back, never applied). `Unknown`, `0`, `999`, `Salary Estimate` ranges and the rating appended to `Company Name` are not handled.

## Amendments

* **A1 (found by S5, 2026-11-17).** `target_profile_regression` (Baseline target block) did not apply the rule, so the Baseline said 0 rows without a target while the audit had 50. It now applies the same `_numeric_codes` rule
  (lazy import, because `taskclean` imports `regression`), and a unit test (`test_baseline_target_block_matches_the_audit`) pins the agreement. The spec listed the snapshot but not this block.
* **Wording, not behaviour.** S2 names the California column `Population` (its real name). The comparison tolerance in `validate_sentinels.py` (floats 2e-3, numbers in sentences to two decimals) was added after the first comparison flagged only run-to-run wobble;
  the first, stricter comparison reported differences such as 0.755613 vs 0.75564, and the committed code, run the same way, shows the same kind of difference against its own saved results.
