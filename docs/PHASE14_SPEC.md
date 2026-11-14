# Phase 14: inconsistency repair and irrelevant-feature detection

Written **before** any code or validation run. Thresholds below are fixed in advance; the validation results do not
change them. Phases 1-11 and `results/phase1-13*` are untouched; everything here is additive.

## Why this phase exists

The product audited five issue types. Two data-quality categories were not covered properly:

| Category | State before | Gap |
|---|---|---|
| Inconsistent values | `detect_inconsistency` flagged case/whitespace variants only, was **report-only**, and was "not independently validated" | no repair, no evidence, misses abbreviations (`M` for `Male`) and typos |
| Irrelevant features | nothing | no constant / ID-like / redundant / no-signal column detection |

## A. Inconsistency detector and repair (`src/inconsistency.py`)

Applies to categorical columns with 2..500 distinct **keys** (see Amendment 3), skipping free text / identifiers
(keys > 50 **and** > 50% of non-null rows). Null cells and placeholder cells are ignored (missing-value territory).

Three rules, applied per column on the value counts:

1. **Format variants (tier `strict`, a proposed repair).** Two raw values with the same *key* are variants of one
   value. key = Unicode-NFKD, accents removed, case-folded, **decoration only ignored** (see Amendment 2: spacing,
   `_ - .` between alphanumerics, trailing `. , ; : ' "`). Canonical = the most frequent raw value in the group (ties: smallest string, so the result is deterministic).
2. **Abbreviations (tier `strict`).** A key `a` with at most 3 characters is a variant of key `b` when `b` starts with
   `a` or `a` equals the initials of `b` (two or more words). Merged only if **exactly one** `b` matches and the
   smaller of the two forms is at most **30%** of their combined count (so `No` vs `None`, both common, stay apart).
   More than one match: **flagged, not repaired** (ambiguous).
3. **Typos (tier `loose`, flag only, never repaired).** Both keys at least 5 characters, within one edit
   (insert, delete, substitute, adjacent swap), the differing character is not a digit, the rarer form is at most
   **5%** of the other, and the match is unique.

Row-level rate = share of rows with at least one flagged cell (strict or loose). Only cells that would *change*
are counted (the old detector also counted the canonical rows, so the two rates are not comparable).

**Repair policy:** `strict` proposals are repair candidates. Whether they are applied automatically is decided by the
**existing** gate, never by this module: measured harm risk 0, repair precision >= 0.95, status not negative, not
underpowered. Evidence comes from the Phase 14 validation below.

## B. Irrelevant-feature detector (`src/relevance.py`): report-only

Never changes the cleaned file and never drops a column; it recommends. Checks, first match wins:

1. **constant**: one distinct value, or the top value covers >= 99.5% of non-null rows.
2. **id_like**: categorical with unique ratio >= 0.95 (>= 50 rows); integer-like numeric, strictly monotonic with
   unique ratio >= 0.98; or a column named `id/uuid/guid/index/idx` with unique ratio >= 0.5.
3. **redundant**: numeric pair with |Pearson| >= 0.995, or a one-to-one value mapping between two columns
   (e.g. `education` and `education-num`). The **later** column is flagged and names the one it duplicates.
4. **no detectable relevance** (needs a target): random-probe test. Three random probe columns (Gaussian,
   integer-uniform, 20-level categorical) are added; a random forest (150 trees) is fit on a 70% split and
   permutation importance (5 repeats, balanced accuracy) is measured on the held-out 30%. A column is flagged only if
   **both** its permutation importance and its mutual information with the target are no larger than the **largest**
   probe's. Requiring both guards against flagging a feature whose importance is merely shared with a correlated one.
   Up to 6,000 rows, 100 columns. **(Probe count amended: see Amendment 1.)**

`no detectable relevance` is *not* proof a feature is useless: it can matter only in combination, and the test is
random-forest based.

## C. Validation design (Adult, `src/phase14_validation.py`)

Same discipline as Phases 7 and 9: seeds 42-46, paired difference against a clean baseline per seed, damage measured
on the untouched clean test set, corruption on the training split only, the real product path
(`audit_dataset` then `_apply`) used for detection and repair.

**Inconsistency:** inject into **categorical cells**, rates 5/10/20%, an equal mix of five kinds (case, whitespace,
punctuation, abbreviation = first 3 letters, one-character typo). The mix is an **assumption**, not a measurement of real
data. Report per kind. Measure detector precision/recall against the injection log, repair precision/recall
(a repaired cell is correct iff it equals the clean original), damage, recovery, and the same evidence-table columns
as Phase 10 (sign agreement 80%, noise floor 0.003, risk = P(harm) x harm magnitude). Also the **false-positive rate on
clean Adult**.

**Irrelevant features:** assess clean Adult; then add known-irrelevant columns (Gaussian, uniform int, 3-level and
30-level random categorical, constant, exact copy of `age`, linear copy of `age`, sequential row id, unique string id)
and measure recall per kind, plus what fraction of the 14 real columns are flagged. Utility: F1 with all columns vs after
dropping the flagged ones.

## D. What would count as a failure (reported, not hidden)

* Inconsistency false-positive rate on clean Adult above 1% of rows.
* Detector recall on the *detectable* kinds (case, whitespace, punctuation) below 90%.
* Repair precision below 0.95 (then the existing gate keeps the repair in human review: no special-casing).
* A real, informative Adult column flagged as irrelevant and dropping it lowers F1 by more than the 0.003 noise floor.
* Any irrelevant-column kind detected less than 80% of the time.

## E. Known limits, stated up front

* Single dataset; the injected mix of inconsistency kinds is the author's assumption.
* Synonyms and different spellings of the same thing (`USA` / `United States`, `Mr` / `Mister`) are not detectable
  without external knowledge. Not attempted.
* Irrelevance is model-conditional (random forest) and is a recommendation only.
* Calibration (`selfbench.py`) is **not** extended to these two types; on calibrated audits they keep using the Adult
  evidence and say so.

## Amendment 1 (made after unit tests on planted data, before any validation run)

Unit tests on a synthetic frame with known signal and noise columns exposed two defects in section B.4. No Adult
validation had been run, so this is a design correction, not tuning to results. Both are recorded here instead of
being changed silently.

1. **Bug: probe seed collision.** Probes were drawn from `default_rng(seed)`. A caller whose own data was generated
   from the same integer seed got probes *identical* to a real column (the test's `x1` equalled the Gaussian probe), which
   inflated the threshold and flagged a genuinely informative column. Probes now use `default_rng([seed, 14014])`.
2. **Design flaw: three probes are too few.** A truly irrelevant column is exchangeable with the probes, so it
   outranks all K of them with probability 1/(K+1). With K = 3 that is 25% per criterion, so up to about 50% of noise
   columns would be wrongly kept as "relevant" (observed: `noise_gauss` kept at seed 1). The probe set is now
   **K = 12** (four each of Gaussian, integer-uniform and 20-level categorical), giving about 8% per criterion and at
   most about 15% combined, i.e. an expected recall on pure-noise columns of roughly 85%. This is also why the
   failure criterion "any kind detected less than 80% of the time" is kept: it is now attainable but not guaranteed.

The threshold rule itself (flag only if **both** measures are no larger than the largest probe's) is unchanged.

## Amendment 2 (made after the independent real-data check, before the injected-error validation)

The real-data check (the detector run on the raw datasets in `data/`, results read by hand) found a correctness bug in
section A.1 as originally written. "Every run of non-alphanumerics replaced by one space" made `>50K` and `<=50K` both
become `50k`, so on the raw Adult file (no-target mode, so `class` is a feature) the detector flagged **11,687 cells**
and proposed rewriting `>50K` as `<=50K`. The same rule would merge `A+` with `A-`, or any pair whose only difference is a
symbol that carries meaning. That is a destructive false positive, not a tuning matter.

Fix: only *decoration* is ignored when building the key: whitespace, `_ - .` **between** alphanumerics, and trailing
`. , ; : ' "`. Every other symbol (`+ - < > = % $ # & /`, and a leading or trailing `-`) stays in the key. Regression tests
now assert that `>50K`/`<=50K`, `A+`/`A-`/`A`, and the eight blood types are never merged. The injector's "punctuation"
kind (`-` to space or `_`, or a trailing `.`) is consistent with the corrected rule and unchanged.

## Amendment 3 (made after the single-seed smoke run, before the 5-seed validation)

The smoke run reported ~93% recall on the *detectable* kinds (case, whitespace, punctuation) where the detector alone
measured 100% at 5% and 10% corruption and **87.9% at 20%**. A diagnostic found the cause: at 20% corruption the
`occupation` column grew from 14 real values to **691 distinct spellings**, which exceeded the 500-value guard, so the
whole column was skipped as "free text". The guard exists to ask whether a column has a stable vocabulary; counting every
typo as a new vocabulary item defeats that.

Fix: the guard now counts distinct **keys** (spelling variants that differ only in case, spacing or separators collapse)
instead of raw spellings. The thresholds (50, 500, 50%) are **unchanged**; a raw-string ceiling of 10,000 is kept as a
pure cost guard. Skipped columns are now also listed in the audit's data notes, so a skip is never silent. The
pre-fix result (87.9% at 20%, one of eight columns lost) is recorded here so the change is not read as tuning.

Other smoke-run observations, recorded and **not acted on** (they are not failure criteria, and changing the detector to
raise them would be tuning): abbreviation repair recall was 10.7% (most were flagged as ambiguous, because other injected
typo variants of the same value share the 3-letter prefix and break the "unique match" rule), and typo flag recall was
1.7% (typo variants of one word are each other's one-edit neighbours, so the unique-neighbour rule declines). Both are
amplified by the injector's dense, equal-mix design and are expected to be better on real data where typos are sparse;
the 5-seed run reports them as measured.

**Correction to Amendment 3 (written after the 5-seed validation).** The key-based guard did **not** fix the 20% case it
was written for. At 20% corruption `occupation` still has **625 distinct keys** (691 raw spellings): typo and
abbreviation variants are genuinely different keys, so counting keys instead of spellings barely helps. The column is
still skipped at 20%, detection recall on the detectable kinds is unchanged (0.715 overall at 20%), and the skip is
visible in the audit's data notes rather than silent. The 500-key ceiling was **not** raised: doing so after seeing the
numbers would be tuning, and a column whose vocabulary has exploded this far is a legitimate "do not trust this" case.
The key-based count is kept because it is the more faithful definition (and fixes the smaller case of case/spacing
variants inflating the count); this note only withdraws the claim that it resolved the 20% skip.

## Diagnostic: why is exact repair "negative"? (`src/phase14_diagnostic.py`, written after the 5-seed validation)

The product's inconsistency repair has precision 1.000 and no false repairs, yet measured F1 recovery was negative at
all three rates. To see whether the validation setup itself forces that sign, the injection log was used as an oracle to
restore chosen subsets of the injected cells exactly (5 seeds x 3 rates; `results/phase14_inconsistency_diagnostic.csv`).
Mean F1 recovery (std across seeds in brackets, seeds with negative recovery out of 5):

| Restored subset | share of injected cells | 5% | 10% | 20% |
|---|---|---|---|---|
| everything (oracle) | 100% | +0.0127 (0.006), 0/5 | +0.0104 (0.006), 0/5 | +0.0103 (0.003), 0/5 |
| only case/whitespace/punctuation | 62% | -0.0018 (0.005), 3/5 | -0.0035 (0.007), 3/5 | -0.0040 (0.007), 3/5 |
| only typo/abbreviation | 38% | +0.0088 (0.007), 0/5 | +0.0055 (0.004), 0/5 | +0.0121 (0.009), 0/5 |
| random subset, same size as the product's | 57-64% | +0.0021, 2/5 | -0.0004, 2/5 | -0.0058, 5/5 |

What this does and does not show:
- The setup can show recovery: restoring everything gives a consistent +0.010 to +0.013 (0 negative seeds).
- The sign comes from the subset, not from the repair being wrong. The kinds the product can repair with certainty
  (case, spacing, punctuation) give about zero to slightly negative recovery, with an effect no larger than the seed
  noise (|mean| < std). The kinds that carry the gain (typos, abbreviations) are the ones the product flags but does not
  repair, or repairs only rarely.
- Even a random subset of the same size is negative at 20%, so this is not specific to our repair rule.
- **Not explained:** why restoring only the easy kinds trends negative rather than zero. A hypothesis (partial
  consolidation leaves a column with a few clean categories plus stray variants, which a forest may split differently)
  is untested and is not claimed.
- Nothing was changed in response: the evidence rule's "negative" status stands, so these repairs stay in the review
  queue rather than being applied automatically.
