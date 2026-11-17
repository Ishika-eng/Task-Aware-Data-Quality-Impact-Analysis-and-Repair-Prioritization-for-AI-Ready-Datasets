# Phase 23: does the tool work on genuine datasets it has never seen? (validation, not new features)

Written **2026-11-17, before any download or run** (the datasets below were chosen after checking only that their download URLs
respond, and after listing which files the project already used). Fixed in advance; outcomes are reported as measured, failures kept as
findings. Anything that has to change afterwards is appended as a visible Amendment. **No product code changes in this phase** except
fixes that a finding forces, each recorded as an Amendment with the evidence that required it. Clustering is deferred to
[Phase 24](PHASE24_SPEC.md) and does not start until this phase is reported.

## Why

Everything built so far was developed against Adult, Bank Marketing, German Credit, Heart Disease, Telco churn, California Housing and
Diabetes. Detector settings, thresholds and one amendment (regression target detector, Phase 21 A1) were shaped by those files, so the
tool has never been tested out-of-sample on data nobody tuned it for. This phase is that test.

## Files (genuine, public; none used before)

| Id | File | Source | Why this one | Task / target |
|---|---|---|---|---|
| T | **Titanic** (891 rows, CSV, ~60 KB) | public GitHub CSV mirror | real, mixed types, real missing values (Age, Cabin, Embarked) | classification, `Survived` |
| A | **Ames Housing** (1,460 rows, ~80 columns) | OpenML `house_prices` | real, heavy missingness, skewed numbers, and the "NA means *no such feature*" convention that mimics placeholders | regression, `SalePrice` |
| R | **Real Estate Valuation** (414 rows, `.xlsx`) | UCI | a genuine small Excel file; under the 600-row calibration minimum, so it tests the decline path | regression, house price per unit area |
| O | **Online Retail II** (two sheets of about 0.5M rows, `.xlsx`, ~45 MB download) | UCI | a genuine large multi-sheet Excel file with missing customer ids, duplicates and returns (negative quantities) | no target (health report and feature checks), plus sheet switching |
| C | **Concrete Compressive Strength** (`.xls`) | UCI | a genuine legacy workbook: must be refused with the clear message | n/a |

Everything is downloaded into `data/new/` (not committed if large; sizes and checksums are recorded in the results file).

## Method

Each file goes through the product path (backend functions as the API calls them): upload, audit (calibration on where the file allows
it), model panel, outputs. Ground truth does not exist for genuine files, so criteria are of three kinds: **mechanical** (checkable
exactly), **injected-truth** (add known errors to the genuine file as the reference, then check detection against what was injected,
Phase 13 style), and **spot-checked** (a sample of what the tool flagged, judged by rules written below, labelled as the author's
judgement, not measured truth).

## Criteria (all fixed now)

* **V0 nothing broke.** All existing suites pass at the start and at the end (taskclean, selfbench, phases 14-22).
* **V1 completes, or fails clearly.** For T, A, R, O: load, health report, ranked issue list, cleaned file and repair log are produced, or
  the failure is a clear message (never a stack trace, hang or empty page). Wall time and peak memory are recorded. For O the load must
  complete within **5 minutes**, and if not it is reported as a limitation of the product with a measured size/time, not hidden. For C the
  legacy-`.xls` message must appear.
* **V2 mechanical honesty (T, A, R).** (a) every cell that differs between the raw file and `dataset_cleaned.csv` appears in `repair_log.csv`
  as applied; (b) with no calibration (R) or with no evidence, nothing but evidence-supported repairs is applied; (c) the rows/columns in
  the health report equal the file's; (d) regression pages contain no F1 and classification pages no R²; (e) no percentage is shown for a
  delta inside its noise floor.
* **V3 injected-truth detection (T, A).** Take the file's complete-case rows as the reference, inject 5% of each error type with the
  project's corruptors (missing cells, duplicates, outliers, wrong labels or wrong targets), run the detectors, and report precision and
  recall per type. **Hard checks:** missing-value and duplicate detection recall at least 0.95 (they are exact by construction; a miss is a
  bug). Outlier and wrong-label/target precision and recall are **reported with no threshold**, then compared with the Adult and Bank
  Marketing values already in the README, so that a large drop is visible.
* **V4 false alarms on the genuine file (T, A, R).** Report each detector's flag rate on the file as it is. A flag rate above **10%** of
  rows (cells for missing values and outliers) for any detector is recorded as a **concern** and spot-checked.
* **V5 spot check.** For each detector that flags anything, draw 20 flagged items at random (seed 0) and judge each by these rules:
  *duplicates*: mechanical (identical rows or not); *missing*: "defensible" if the cell is truly empty or a recognised placeholder,
  "false alarm" if the value is a genuine category in the file's own documentation (Ames `NA`); *outliers*: "defensible" if the value is
  outside 3 IQR of its column *and* physically or logically implausible or an extreme of a heavy-tailed column, "legitimate extreme" if
  the dataset documentation or domain makes it a real value; *wrong label/target*: "unclear" unless the row is plainly inconsistent.
  Counts per verdict are reported. This is judgement, not truth.
* **V6 panel on real data (T, A).** The model panel completes; every arm that changed no cells reads "no change"; every improved/worse
  verdict clears its noise floor with at least 80% seed agreement (the amended R5 rule); results are reported as measured. No claim is made
  that cleaning helped unless the panel says so.
* **V7 Excel intake on real workbooks (R, O).** Sheets are listed correctly (O: 2 sheets), the default sheet is the first with data, the
  sheet picker returns the other sheet with the same columns, the row count equals the count read independently by openpyxl, dates stay
  dates, and the audit of O (a sample, because of the row limit) reports its missing-customer-id and duplicate findings.
* **V8 decline paths are clear.** R (414 rows) calibration/panel decline with a reason that names the 600-row minimum; no crash.

## Predictions (recorded before any run; wrong ones are reported)

(1) Ames will be over-counted as missing for columns where `NA` means "none" (alley, pool, fence, fireplace quality), a known limitation
now seen on real data, so V4/V5 will record a concern there. (2) The regression target detector will flag more than 2% of Ames rows (its
2% limit was fitted on California Housing and checked on Diabetes only). (3) The outlier detector will flag a large share of the skewed
Ames area columns, mostly as legitimate extremes. (4) On Titanic the wrong-label detector will have low precision (survival is hard to
predict, so many "confident disagreements" are ordinary noise); the tool should then keep label repair in review, not apply it. (5) Titanic's
`Cabin` (77% missing) will be correctly flagged as a mostly-empty column rather than imputed. (6) Online Retail II will load slowly
(minutes) and use a lot of memory; it may break the 5-minute limit. (7) The panel will show "no change" for most arms on Titanic and
Ames, as in Phases 19-21, and that is the honest result. (8) At least one genuine bug or rough edge will surface (a poor message, a
wrong note, a slow path), because every earlier new file found one.

## Not claimed

That passing here proves the tool works on all data (five files); that spot-check verdicts are ground truth; that the injected errors
resemble real ones; or anything about clustering.

---

# Amendment A1 (2026-11-17, after the first runs of T, A, R, O, C; before any product code was changed)

**What was measured (`results/phase23_A.json`).** On Ames Housing with calibration on, the default policy auto-applied **7,829**
missing-value imputations. Of those, **6,608 (84%)** are in columns whose `NA` the dataset's own documentation defines as "feature absent"
(PoolQC, MiscFeature, Alley, Fence, FireplaceQu, Garage*, Bsmt*), and **864** more are the genuine category `None` in `MasVnrType` read as
a placeholder. Examples: all 1,453 `PoolQC` cells became `Gd` (a good-quality pool for houses with no pool), all 1,369 `Alley` cells
`Grvl`. Only about 280 of the 7,829 are plausibly missing measurements (`LotFrontage` 259 and a few others).
**V2(b) is therefore a failure**: repairs were applied that the calibration's evidence does not cover. Cause, read from the
calibration output: the self-benchmark reference **leaves out columns that are more than 30% missing** (here Alley, MasVnrType,
FireplaceQu, PoolQC, Fence, MiscFeature), so the evidence "imputation is safe" was never measured on them, yet the policy applied
imputation to the whole file. The calibration also only injects *random* blanks, so it can never say whether a file's real blanks mean
"absent"; that part stays a stated limitation.

**Change (written before the code).** An evidence-based auto-repair is applied **only to columns the calibration covered**. For any
column-level repair (missing values, outliers, inconsistent values) in a column the calibration left out of its reference, the proposal
stays `flagged_for_review` (not applied) and a note says how many were held back and why. A human override of the whole issue type still
applies everything, as before. Nothing else changes: no thresholds, no detectors, no token lists. With no calibration nothing is
auto-applied anyway, so uncalibrated runs are unchanged (K0-style check: all earlier suites must pass untouched).

**Expected effect (recorded now).** Ames applied imputations fall from 7,829 to roughly 860 (6,969 held back). About 590 of the remaining
~860 (Garage*/Bsmt* "no garage / no basement" cells, under 30% missing) are still structural and **will still be imputed**; this
amendment does not fix that, and it is reported as an open limitation, not claimed fixed.

**Not changed, reported as open limitations:** (i) the placeholder rule reads the genuine category `None` as missing (MasVnrType, 864
cells); (ii) numeric identifier columns with blanks (Online Retail `Customer ID`) and 77%-empty text identifiers (Titanic `Cabin`)
receive imputation *proposals* (median ID, most common cabin), which are review-only without calibration but would be auto-applied
under calibration if the column were inside the calibration scope.

---

# Results (2026-11-17; `src/validate_new_datasets.py`, `results/phase23_*.json`; each file run once, then A and T re-run after A1)

Reported as measured. Files: Titanic (T), Ames Housing (A), Real Estate Valuation `.xlsx` (R), Online Retail II `.xlsx` (O), Concrete `.xls` (C).
`results/phase23_A_before_A1.json` and `phase23_T_before_A1.json` keep the pre-amendment runs.

| Criterion | Outcome |
|---|---|
| **V0** nothing broke | **Pass.** All 12 suites (taskclean, selfbench, phases 14-23) pass after A1. |
| **V1** completes or fails clearly | **Pass.** T 18 s audit + 25 s panel; A 54 s + 65 s; R 3 s (panel and calibration decline with a reason); O (525,461 rows) **loaded in 11 s, peak 1.7 GB, audited in 7 s**; C refused with the exact message "legacy .xls workbooks are not supported: open the file in Excel and save it as .xlsx or CSV". No crash, stack trace or hang. |
| **V2a** every changed cell is in the repair log | **Pass** (T, A, R: 0 unexplained cells). |
| **V2b** only evidence-supported repairs applied | **Fail before A1, pass after.** See Amendment A1: 7,829 Ames imputations, 6,969 of them outside the calibration's evidence; after A1, 860 applied. |
| **V2c / V2d / V2e** report rows match the file; no wrong-task metric; no percentage inside noise | **Pass.** (My first F1 check matched the Ames column name `BsmtFinSF1`; a word-boundary check finds none. That was a flaw in the checker, not the product.) The check reads backend text only; page wording was last checked in a browser in Phase 21. |
| **V3** injected-truth detection | **Hard checks pass:** missing-value and duplicate detection precision and recall are **1.00** on both T and A at 5%. Reported, no threshold: outliers precision **0.21 (T), 0.09 (A)**, recall 1.00 / 0.95, against 0.36 / 1.00 on Adult, so outlier precision is much lower on both new files. Wrong labels (T): precision 0.29, recall 0.23 (Adult 0.45 / 0.20). Wrong targets (A): precision 0.89, recall 0.93. |
| **V4** false alarms on the genuine file | **Concerns recorded:** the outlier detector flags more than 10% of rows on T (**18.3%**), A (**38.3%**) and O (**17.7%**); R is just under (9.7%). On clean reference rows the outlier detector's own false-positive floor is 20.7% (T) and 34.8% (A). Target-error detector on A: **1.92%** (under the 2% limit); on R 1.7%. No missing-value or duplicate concern. |
| **V5** spot check (author's judgement) | **Outliers: 0 defensible errors in 60 drawn items**; all were legitimate extremes or false alarms: Titanic first-class fares, large families and an 80-year-old; Ames large lots, pools and `MSSubClass`/`OverallCond` (building-class code and 1-9 rating scale, which are categories stored as numbers: 7 of 20); Real Estate: **12 of 20 are latitude/longitude** (a city-sized range makes every coordinate look extreme), the rest real station distances; Online Retail: wholesale quantities (24-100) and high prices, and the negative quantities and prices are real returns and bad-debt entries. **Missing values:** Titanic 20/20 genuinely empty; Ames 95% false alarms by the dataset's own documentation (A1); Online Retail 107,927 blank `Customer ID` (guest checkouts) are genuinely empty. **Duplicates (O):** 6,865 of 6,866 flagged rows are exact repeats; the last differs only by a one-minute `InvoiceDate` (duplicates are judged without datetime columns). **Wrong label/target:** all unclear, none plainly wrong. **Inconsistent values (A):** 2 of 2 flagged are a genuine category (`SaleType` `Con`), false alarms, flag-only. For O the 20 outlier items were all reviewed by column and size; 12 were read in full. |
| **V6** panel on real data | **Pass.** T and A: 0 improved, 3 worse, 8 no change, 13-17 within noise. Every verdict has seed agreement 1.0; arms that changed nothing read "no change". The panel results did not move after A1. |
| **V7** Excel on real workbooks | **Pass.** R: 414 rows read, equals openpyxl's 414; a non-Latin sheet name loads. O: two sheets listed; default sheet is the first; the picker returns the other sheet (541,910 rows) with identical columns; row counts equal openpyxl's independent counts (**525,461 and 541,910**); `InvoiceDate` is a date column; the audit reports 107,927 blank customer ids and 6,866 duplicates. |
| **V8** decline paths | **Pass.** R calibration: "only 407 clean-ish rows remain ... (need at least 600)"; panel: "needs at least 600 rows with a numeric target; this file has 414". |

## Predictions scored

(1) Ames placeholder over-count: **right, and worse than predicted** (auto-applied, not just counted). (2) Ames target detector above 2%: **wrong** (1.92%). (3) Large outlier share on skewed Ames columns, mostly legitimate: **right** (38%, 0 defensible in the sample). (4) Titanic label detector low precision, kept in review: **right** (0.29; never auto-applied). (5) Cabin flagged, not imputed: **half right**: not applied, but it is *proposed* a replacement ("B96 B98" for 687 cells). (6) Online Retail slow, over 5 minutes: **wrong** (11 s). (7) Panel mostly "no change": **right**. (8) At least one genuine bug: **right** (A1).

## Findings

* **F1 (fixed, A1).** Calibrated default applied imputation outside the calibration's evidence (above).
* **F2 (open limitation).** The placeholder rule reads the genuine category `None` as missing (Ames `MasVnrType`, 864 cells), and about 590 of the 860 imputations still applied on Ames after A1 are structural ("no garage", "no basement", under the 30% line). The tool cannot tell a blank that means "absent" from a blank that means "missing"; only the user can. Under calibration these remain auto-applied.
* **F3 (open).** Imputation is *proposed* for identifier columns (a median `Customer ID`, the most common `Cabin`). Review-only without calibration.
* **F4 (open).** The outlier detector has low precision on all four real files (20% or less of what it flags was an injected error, and 0 of 60 sampled real flags were errors). It is never auto-applied, but the report should be read as "unusual values to look at", not "errors". Categories stored as numbers and geographic coordinates are the visible causes.
* **F5 (minor, disclosed).** Duplicate detection ignores datetime columns (one extra flag in 525,461 rows).

## Verdict

The tool completes on five genuine files it had never seen, refuses what it should, reads two real Excel workbooks (one of half a million rows per sheet) correctly, and its exact detectors (missing values, duplicates) and its regression target detector held up on new data. It also found one real defect in the product's own default behaviour (F1, fixed), and showed that the outlier detector is the weakest part. **Not shown:** that cleaning improves models on these files (the panel found no improvement on either), or that the tool is correct on data types not covered here.

## Front-end check and fixes (after validation)

The Ames audit was walked through the real UI (every page, downloads, mobile and light theme). Seven display problems were found and fixed; none changed a measured result.

1. "Run audit" was off-screen on a 1024x768 window: the button is now pinned at the bottom of the configure panel, whose height is capped to the viewport.
2. Regression pages used classification wording (Overview, Calibration, Features, the Model-panel toggle): now task-aware ("Wrong target values", R-squared, Ridge regression).
3. The A1 hold-back was explained only in Data notes: the Repair plan and Review queue now state how many proposals are held back, in which columns, and that the aggressive file applies them (new `held_back` field in the dashboard payload).
4. Task impact said "applied" next to "review before repairing automatically": an applied row now reads "Applied automatically: no harm was observed and repair precision met the bar".
5. The Baseline report said outlier counts are "same as the audit" while the snapshot counts the target column: a footnote now says why they can differ.
6. The Overview model card headlined the aggressive file: it now headlines the evidence-based (default) file and shows the aggressive result beside it.
7. The Quality table squeezed the Note column: it now has a minimum width and the table scrolls sideways on narrow windows (Note needs a scroll at 1024 px).

Verified in the browser on a fresh Ames audit (regression, calibrated, panel on). Not fixed: a "Duplicate row" row with nothing to repair still shows the generic "review before repairing automatically" text.
