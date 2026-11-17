# Phase 24: clustering as a third ML task (impact-aware cleaning when there is no target)

Written **2026-11-17, before any code or run**. Fixed in advance; outcomes are reported whatever they are. Anything that has to change
afterwards is appended as a visible Amendment. Additive: classification and regression behaviour must stay byte-identical (K0), frozen
research files untouched, new results go to `results/phase24_*`. Status: **draft for the user's review**; no code is written until it is
accepted.

## Why

The project's objective is: the user picks the ML task (classification, regression or **clustering**); the system finds the problems,
ranks them by their impact **on that task**, repairs what it can, and compares model performance before and after. Classification
(phases 1-20), regression (21) and Excel intake (22) exist. Clustering is the last unmet part of the objective.

## The problem that makes clustering different

Every existing measurement of "damage" compares a model's score with the truth in a column: macro-F1 against labels, R² against a
numeric target. Clustering has **no truth column**, so "how much does this error hurt the task" has no obvious definition, and a
before/after comparison on an uploaded file has no obvious score. Three definitions are needed, and each must be honest about what it
can and cannot show:

1. **Damage of an error type (the impact ranking).** Defined *relative to the clustering the same algorithm finds on undamaged data*,
   which the self-benchmark can construct because it corrupts the user's own cleanest rows itself (so the undamaged version is known by
   construction, exactly as in phases 10 and 21). No external truth is needed.
2. **Before/after comparison on an uploaded file.** Cluster **stability** on held-out rows (below). This measures reproducibility, not
   correctness; the product says so on the page, and the proof (C6) measures how well it tracks correctness where the truth is known.
3. **Optional reference column.** If the user has a known grouping (segment, species, department) it can be supplied as a column that is
   *never used for clustering*, only to score. Then the before/after score is agreement with that column, which is real external
   validity. This is also exactly the design of the controlled proof.

## What changes for clustering

Feature-side detection (missing values, duplicates, outliers, inconsistent spellings, feature relevance) already runs with no target
(`audit_dataset(target=None)`, `health_snapshot`), and the "wrong label" detector is skipped. Everything target-dependent is replaced:

| Part | Classification / regression | Clustering (this phase) |
|---|---|---|
| Task input | target column | none; optional **k** (2-20, or auto) and optional **reference column** (scoring only, excluded from features) |
| Wrong-label / wrong-target detector | RF out-of-fold | not applicable; stated on the page, not silently dropped |
| Preprocessing | impute, encode | impute (median / most frequent), one-hot for text, **StandardScaler** on numerics (the usual practice; it is also how outliers do damage, and that is part of what is measured) |
| Algorithms | LR/RF/HistGB/MLP, or Ridge/RF/HistGB/MLP | **k-means, Gaussian mixture (diagonal covariance), Ward agglomerative**. Hierarchical has no `predict`, so held-out rows go to the nearest training-cluster centroid in the preprocessed space. Density methods (DBSCAN/HDBSCAN) are **excluded**: they label points as noise and have no stable `predict`, which breaks the paired design |
| k | n/a | auto = best silhouette for k = 2..10 on a prepared sample of at most 2,000 rows, **chosen once on the raw data and held fixed for every arm** so arms stay comparable |
| Damage metric | macro-F1 / R² drop | **1 - ARI** (adjusted Rand index) between the clustering from clean training rows and from damaged ones, both assigning the same untouched held-out rows. ARI is permutation-invariant and chance-adjusted. A companion "share of held-out rows that change cluster" (best label matching) is shown for readers, never used for a verdict |
| "Learnable" guard | beats chance by 0.05 | **structure guard**: silhouette of the real reference minus silhouette of the same data with every column independently shuffled must be at least `CLUSTER_MARGIN = 0.05`; otherwise the self-benchmark and panel **decline** ("no cluster structure beyond chance, so the effect of errors on it cannot be measured") |
| Noise floor | null experiment, min 0.003 | same null experiment (refit on a random 90% of training rows), **min `NOISE_FLOOR_MIN_CLUSTER = 0.02`** in ARI units |
| Evidence tables, safe-auto rule, harm rule | `selfbench._evidence_table`, `classify_repairability` | the **same functions** (they read only `damage`, `recovery`, `restored_exactly`, precision/recall), so clustering evidence is judged by the rule classification uses |
| Nothing auto-applied without evidence | yes | yes: no evidence borrowed from Adult or any other file; an uncalibrated clustering run applies nothing |

**Fixed parameters (set now, not tuned):** `K_RANGE = 2..10`, `CLUSTER_MARGIN = 0.05`, `NOISE_FLOOR_MIN_CLUSTER = 0.02`, tuned-selection
margin `0.02` (stability units), bootstrap stability `B = 10` resamples (45 pairs), k-means `n_init = 10`, `MIN_REFERENCE_ROWS = 600`
(unchanged), held-out fraction 0.25 (panel) / 0.30 (self-benchmark), `SIGN_AGREEMENT = 0.8`. `NOISE_FLOOR_MIN_CLUSTER` is a prior guess; if
the null experiment shows it is far from the measured noise it is changed by a visible Amendment, not silently.

## Stability, the before/after score on an uploaded file

For an algorithm and an arm (raw rows, default file, aggressive file, each repair alone): split first (held-out rows stay untouched, as in
phase 19), fit the repair and the preprocessing on the **training rows only**, draw `B = 10` bootstrap resamples of those training rows,
fit the algorithm on each, assign the **same held-out rows**, and report the mean pairwise ARI of those 10 assignments. Because every arm
is scored on identical held-out rows, arms are comparable; no reference clustering and no truth is needed. Alongside it the page shows the
smallest cluster's share and the number of non-empty clusters, because a degenerate clustering (almost everything in one group) can be
very stable and is not good. A "metric disagreement" flag (as F1 vs AUC in phase 10) is raised when stability and the fixed-space
silhouette move in opposite directions beyond their floors. Deltas inside the noise floor read "no change", never a percentage.

## Stages and acceptance criteria (all fixed now)

**Stage K-A, plumbing.** `task="clustering"` through `audit_dataset`, `_prepare` (no target; optional reference column excluded from
features and kept aside), auto-k, structure guard, policy and report wording ("Impact on clustering", "cluster stability", "agreement with
your reference column"), no F1/R² anywhere on a clustering page.

* **K0, no regression.** Every existing suite (taskclean, selfbench, phases 14-22) passes unchanged, and classification and regression
  audits, logs, cleaned files and summaries are identical for the same input and seed.
* **K1, inputs.** Clustering needs no target; an optional reference column is excluded from the features (a spy test shows it never
  reaches a model); k out of range or larger than the number of rows is a clear error; auto-k is reproducible for a seed.
* **K2, structure guard on planted data.** On generated blobs with known separated groups the guard passes; on the same data with every
  column independently shuffled it declines; a pure-noise file declines with the message above.
* **K7, determinism.** Same data, k and seed give identical assignments for all three algorithms.

**Stage K-B, evidence (self-benchmark).** Clustering self-benchmark: clean-ish reference from the user's own rows (complete, de-duplicated,
at most 6,000), train/test split, for each error type x rate (5, 10, 20%) x seed inject one error type into the training rows only,
measure damage = 1 - ARI against the clean-trained clustering on the untouched test rows, then run the real TaskClean path on the damaged
rows and measure recovery, repair precision/recall and harm. Error types: missing values, duplicates, outliers (the existing corruptors,
called with a dummy target). Inconsistent values are measured where the reference has text columns.

* **K3, split integrity.** Spy test as phase 19 U0: detectors, repairs, the scaler and the clustering see only training rows; the test rows
  are never modified and never fitted on.
* **K4, planted damage is recovered correctly.** On blobs with a planted structure, injected outliers measurably damage k-means (damage
  above the floor), the outlier repair recovers part of it, and the evidence table labels it as such; on a clean file the table shows no
  repair with `positive` status that changed no cells (a repair that changes nothing cannot be evidence of help).
* **K5, null file.** On a clean file the panel manufactures nothing: every improved/worse verdict belongs to an arm that actually changed
  cells, clears its floor with >= 80% seed agreement, and arms that changed nothing read "no change" (the amended R5 of phase 21, adopted
  from the start rather than rediscovered).

**Stage K-C, panel.** Three algorithms x arms, stability, optional reference-column score, verdicts as above.

* **K6, honest wording.** Stability is never described as accuracy or correctness; with no reference column the page states that the score
  measures reproducibility; with one, it reports agreement with that column (ARI and normalised mutual information) in addition.

**Stage K-D, controlled proof** (same shape as phases 20 and 21, with the truth hidden from the pipeline).

* Datasets with known groups: **Wine** (178 rows, 13 numeric features, 3 cultivars) and **Breast Cancer** (569 rows, 30 numeric
  features, 2 classes), both bundled with scikit-learn so nothing is downloaded. A third, **Penguins** (344 rows, 3 species, text columns
  that allow inconsistent-value corruption), is added **only if the user approves downloading it**; the proof is registered with two
  datasets and the third would be a separately labelled run.
* Design: clean held-out test rows with the group label as truth (never given to any detector, repair or selection step); errors injected
  into **training rows only** at 5% each (missing cells, outliers, duplicate rows, feature corruption; no label errors exist without a
  target); arms clean ceiling / dirty / default / aggressive / each repair alone / **tuned** (leak-free validation slice of the *dirty*
  training rows, selection by label-free stability, margin 0.02); **10 seeds** (the files are small); 3 algorithms; the score is ARI
  against the true groups on the clean held-out rows.
* **C0 evaluable.** A (dataset, algorithm) cell is evaluable if clean beats dirty on ARI-vs-truth beyond the noise floor. C0 passes if at
  least half of the cells are evaluable. If not, the run is reported "not evaluable" and no claim is made.
* **C1 complete.** Every arm for every algorithm and dataset is reported.
* **C2 positive result.** On evaluable cells, tuned cleaning improves ARI-vs-truth in at least 2 cells and is worse in none. Pass is
  called positive, fail is called negative, and the README says which.
* **C3 default file not worse than dirty** on any cell.
* **C4 the gap is stated.** Feature corruption is flagged, not repaired.
* **C5 tuned selection never reads** the truth labels, the clean rows or the test rows (spy test).
* **C6 does the label-free score track the truth?** Over all non-clean arms of evaluable cells where both moves exceed their floors, the
  sign of the stability change agrees with the sign of the ARI-vs-truth change in at least **75%** of comparisons. The figure is reported
  whatever it is. If it is below 75%, the README states that the before/after stability shown for uploaded files **must not be read as
  an improvement in cluster correctness**.

**Stage K-E, product.** Start-page selector gains **Clustering** (hides target, shows k and the optional reference column); backend
`task="clustering"`; Overview, Calibration, Task impact, Model panel and Proof pages use clustering wording; model panel shows stability,
smallest-cluster share and (if given) agreement with the reference column.

* **K8, end to end.** Uploading a file (CSV and Excel) through the API with clustering chosen yields: health report, ranked error list
  (by 1 - ARI damage), cleaned file, and the three-algorithm before/after table, with no F1 or R² on the page; classification and
  regression uploads are unchanged.

## Predictions (recorded before any run; wrong ones are reported)

(1) Outliers are the most damaging injected type for k-means and Ward, because the StandardScaler's mean and spread are dragged by them;
(2) 5% missing cells do little to Wine and Breast Cancer (median imputation keeps points near their group); (3) duplicates change ARI very
little (copies reweight, they do not move the structure), so most duplicate verdicts read "no change"; (4) Ward is the most outlier-sensitive
algorithm and the diagonal Gaussian mixture the least; (5) the aggressive file (outlier clipping) raises stability more often than it raises
ARI-vs-truth, so **C6 is a real risk and may fail**; (6) the default file equals dirty (no evidence, no auto repair); (7) with only two small
datasets, C2 may fail or be "not evaluable" on one of them, and that will be reported as the result.

## Not claimed

That stability is correctness; that k-means, a diagonal mixture and Ward cover density-based or other clustering families; that the auto-k is
the right k for the user's purpose or that the clusters are meaningful for their business question; that results on Wine, Breast Cancer
(and Penguins if run) transfer to other data, error mixes or rates; that injected errors are real-world errors; that damage measured as
1 - ARI is comparable to F1 or R² damage (different units, different floors).
