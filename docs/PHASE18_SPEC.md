# Phase 18: the baseline health report

Written **2026-11-17, before any code**. Additive: no existing detector, repair or result file is changed. New code is
`src/health_snapshot.py`; new results are not produced (this is a product feature, not an experiment).

## What it is

A diagnostic snapshot of the **raw uploaded file**, captured the moment it is uploaded, before any cleaning, repair or
target choice, stored with the dataset and **never recomputed or edited afterwards**. The audit, the repair plan and the
cleaned files all come later and are measured against it. "Raw" means the file as parsed: the only processing is reading
it (encoding, delimiter, decimal comma, as `load_csv` already does) and reporting what that did.

## Contents (the requirements are the user's; the definitions are fixed here)

**1. Structural and volume vitals.** File size (MB), SHA-256 of the uploaded bytes, rows, columns, in-memory size (MB), how
the file was read. Column type breakdown: numeric, boolean, datetime, categorical (<= 50 distinct values), text or
high-cardinality (> 50), numbers stored as text. Numbers stored as text come in two kinds: *plain* (every value parses as a
number, which the audit already analyses as numeric) and *formatted* (currency symbols, thousands separators, percent signs, accounting
brackets, e.g. `"$1,200"`, `"12%"`). The audit does **not** parse formatted numbers; the report flags them and says the audit
treats them as categorical. That is a real gap, reported, not fixed here.

**2. Completeness and duplication.** Empty cells overall (real blanks plus placeholder tokens `? - -- n/a na n.a. #n/a null nan`
and empty text, the product's existing list, for text columns), split into blanks and placeholders, per column, rows with
at least one missing cell, columns entirely missing. Exact duplicate rows (same definition as the audit: whole record,
first occurrence kept). **Near-duplicate rows**: rows that become duplicates after trimming, collapsing internal spaces and
lower-casing text cells, counted as the rows *beyond* the exact duplicates. Near-duplicates are **new and report-only**:
not validated, never repaired.

**3. Statistical and anomaly warnings.** Outlier rows by the existing IQR detector (1.5 x IQR, with its percentile fallback
for zero-inflated columns; a row counts if any numeric cell is flagged), per-column cell counts. Outlier rows by |Z| > 3
(population standard deviation, NaN ignored, columns with zero spread skipped): new, report-only. Skewed numeric columns
with |skew| > 1.5 (sample skew, at least 8 non-missing values), with the note that this matters for linear models and neural
networks, not for tree models. Constant columns (at most one distinct non-missing value). Possible identifier columns
(every non-missing value distinct, and the column is text or integer-valued, with at least 20 non-missing values).
Inconsistent spellings: the existing detector's flagged cells, grouped per column with examples.

**4. Target block (on request, not part of the stored snapshot).** Because the target is chosen after upload: rows with no
target value, number of classes, class counts and shares, imbalance ratio, classes with fewer than 20 rows.

## What it deliberately leaves out

* **No health score.** The project's own test found an aggregate of detected rates unsupported (pooled r = -0.24), and a score
  rewards repairs the evidence says can hurt a model. Counts and rates only.
* **No severity labels** beyond the user's stated rules (skew, Z, IQR, constant, identifier). No invented thresholds.
* **Label errors and feature anomalies** need a target and a trained model: they stay with the full audit. The report says so.
* **No claim about how much any problem hurts a model.** That is the Impact page.

## Criteria (fixed now)

* **H1, agreement with the audit.** On any dataset, the snapshot's counts for exact duplicates, outlier rows (IQR),
  missing cells and inconsistent-value cells equal what `audit_dataset` reports on the same data.
* **H2, exactness.** On a planted dataset with known numbers of missing cells, placeholders, duplicates, near-duplicates,
  |Z| outliers, skewed, constant and identifier columns, formatted-number columns and inconsistent spellings, every count is
  exactly the planted number.
* **H3, immutability.** Building the snapshot does not change the dataframe (hash before = hash after), and the same input
  gives the same snapshot (apart from the capture timestamp).
* **H4, robustness.** No crash, and a sensible report, on: an all-missing column, a single-column file, one row, no
  duplicates, mixed types, a text-only file, a file with no usable feature column.
* **H5, speed.** Building it must take under 5 seconds on a 45,000-row file; if not, it runs in the background and the page
  shows a "profiling" state. Measured, reported.

---

# Results (2026-11-17)

Code `src/health_snapshot.py`; backend `store_dataset`, `GET /api/datasets/{id}/health`, audit payload and downloads in
`backend/main.py`; front end `HealthReport.tsx`, `Baseline.tsx` and a panel on the Start page; tests `src/test_phase18.py`
(9 tests). No experiment was run, so there are no results files.

| Criterion | Verdict | Evidence |
|---|---|---|
| H1 agreement with the audit | **PASS** | unit test on a planted frame (missing cells per column, exact duplicates, IQR outlier rows, spelling cells all equal `audit_dataset`'s); and on the real demo audit through the API: 4,416 missing cells, 300 duplicates, 2,058 outlier rows, identical in the snapshot and in the audit's quality report |
| H2 exactness | **PASS** | planted: 15 blanks + 10 placeholders = 25 missing cells in 25 rows; 12 exact and 5 near-duplicates; 6 IQR and 6 \|Z\| outliers in `income`; skew flags {income, spend} and not age; 8 + 5 spelling cells; formatted numbers, constant and identifier columns flagged, a continuous float column not flagged as an identifier |
| H3 immutability | **PASS** | dataframe hash before = after; same input gives the same snapshot; JSON-safe; and through the API the snapshot is byte-identical before and after applying repairs (6,300 rows in, 6,000 out) |
| H4 robustness | **PASS** | all-missing column, single column, one row, zero rows, no usable feature column, mixed types, datetime, boolean; a failing snapshot does not block the upload (recorded, reported as an error on request) |
| H5 speed | **PASS** | 1.0 s for 48,842 rows (limit 5 s); synchronous at upload |

**Found while building, reported not hidden.**
* The user's example `"$1,200"` is **not** parsed by the audit (`_looks_numeric` handles only plain numbers and decimal
  commas), so such columns are analysed as categories. The report flags them; the audit is unchanged.
* The IQR outlier share is large on columns where most values are identical: on the demo, `hours-per-week` accounts for
  1,679 of the 2,058 flagged rows (a tight fence around 40). This is the audit's own definition, shown as such, not a new problem.
* The first near-duplicate test data had an all-distinct text column that the identifier rule (all values distinct) correctly
  flagged; the test data was made realistic instead of weakening the rule.
* Not verified: behaviour on files far above 48,842 rows (the snapshot runs synchronously at upload; the 5-second criterion was
  measured only at that size).

---

# Real-data test on a second dataset, and the changes it caused (2026-11-17)

**Test.** IBM Telco customer churn (7,043 rows x 21 columns, never used in building the report; now `data/telco_churn_original.csv`,
SHA-256 `16320c9c...3055e91`). Every number in the report was compared with a separate script that uses only pandas/numpy (not
`health_snapshot`). 15 of 15 numbers matched at first: rows, columns, SHA-256, 11 placeholder cells (all in `TotalCharges`),
0 duplicates, `customerID` as an identifier, `TotalCharges` as numbers stored as text, 0 |Z| and 0 IQR outliers.

**Three real problems it found (the counts were right; the interpretation was not):**

| # | Problem | Cause | Change |
|---|---|---|---|
| 1 | Headline "682 inconsistent spellings": `MultipleLines` `No phone service` shown as a variant of `No` (the aggressive repair file would have applied all 682) | the Phase 14 abbreviation rule ran backwards when the short key was the more frequent one | **Fixed in the detector**: Phase 14 Amendment A1 (`docs/PHASE14_SPEC.md`). Telco now flags 0 |
| 2 | `SeniorCitizen` (0/1) listed as "heavily skewed" | skew was computed on every numeric column; a two-valued flag with a 16% minority has skew +1.8 by construction | skew is now reported only for columns with more than two distinct values (a three-valued column is still checked). The independent check's expectation was amended to the same rule |
| 3 | "1 columns" | the unit text was always plural | `1 column`, `1 row`, `1 cell` |

**Also changed (a consequence of #1).** The inconsistent-spelling count is now broken down by rule: *case/spacing variants*
(certain), *likely abbreviations* (a guess: verify they are not real categories), *ambiguous abbreviations* and *possible typos*
(flagged only). Per column and in the CSV as `cells by rule: <rule>`. The headline number is unchanged in meaning (it is still
the audit's own count) but no longer reads as one kind of error.

**Verification.** `src/test_phase18.py` 11 tests (2 new: the Telco regressions, the by-rule breakdown); all seven test suites
pass; `tsc` and `vite build` clean; the independent Telco check passes 16/16 after the amended skew expectation; in the browser,
a Telco-shaped upload shows `MultipleLines` unflagged, `SeniorCitizen` not skewed, "25 likely abbreviations, 20 case/spacing
variants" with the verify note, and no new console errors (the 404s are the sidebar's stale remembered audit id, unrelated).

**Still true / not done.** The `abbreviation` rule is a heuristic that can still be wrong on other data (see Amendment A1 scope).
Formatted numbers (`$1,200`) are reported but not parsed by the audit. The Streamlit app has no health report. Speed was
measured only at ~49k rows.
