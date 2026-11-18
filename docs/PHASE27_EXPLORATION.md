# Phase 27 (exploratory, NOT pre-registered): can a stronger noisy-label filter make cleaning improve model performance?

Written **2026-11-18, after the results**, because the user asked to pursue the main objective (cleaning that improves ML performance) even though Phases 20-26
found no gain on real files and gains in only some controlled settings. This is **not** a pre-registered phase: the filter and the threshold were chosen by looking at results.
To keep that from inflating the answer, the protocol separates choosing from checking: the strategy and its one parameter were explored on **selection seeds** (42-44) only,
the choice was fixed (rule: the threshold with the best *worst* cell, so that no model gets worse), then run unchanged on **confirmation seeds** (100-104) and on a second dataset
(Bank Marketing) the choice had never seen, and finally the gated version on **fresh seeds** (200-204) with parameters carried over (threshold 0.4, margin 0.005 from Phase 20).
Everything tried is below, not only the best. Code: `src/explore_phase27.py`; data: `results/phase27_*`.

## Why this idea
In the controlled proof the conservative label detector flags 60 of 1,334 wrong labels at 20% label noise (recall 0.04), so label damage, the largest damage the project measured, is left in place.
The filter drops training rows whose own label gets less than a threshold probability from a random forest (out-of-fold, 5 folds), using only the dirty training rows. Setup: 8,000-row
samples, 20% test split kept clean, symmetric label flips at the stated rate only (no other error type), macro-F1 on the clean test rows, deltas are against training on the dirty rows.

## 1. Selection (Adult, seeds 42-44; ceiling = clean training rows)
| noise | model | τ=0.2 | τ=0.3 | τ=0.4 | τ=0.5 | ceiling |
|---|---|---|---|---|---|---|
| +0.1000 | gradient_boosting | +0.0008 | +0.0026 | +0.0042 | +0.0051 | +0.0160 |
| +0.1000 | logistic_regression | +0.0058 | +0.0083 | +0.0046 | +0.0007 | +0.0154 |
| +0.1000 | random_forest | +0.0060 | +0.0070 | +0.0120 | +0.0149 | +0.0121 |
| +0.2000 | gradient_boosting | -0.0124 | +0.0030 | +0.0067 | +0.0117 | +0.0369 |
| +0.2000 | logistic_regression | +0.0049 | +0.0105 | +0.0129 | +0.0131 | +0.0303 |
| +0.2000 | random_forest | +0.0111 | +0.0239 | +0.0337 | +0.0431 | +0.0538 |

Rule fixed from this table before anything else was run: the threshold with the best worst cell, which is **τ = 0.4** (τ = 0.5 helps most at 20% but is flat for logistic regression at 10%; τ = 0.2 makes gradient boosting worse at 20%).

## 2. Confirmation (τ = 0.4 unchanged; Adult seeds 100-104 and Bank Marketing, which the choice never saw)
| dataset | noise | model | always filter | ceiling |
|---|---|---|---|---|
| adult | +0.0000 | gradient_boosting | -0.0163 | +0.0000 |
| adult | +0.0000 | logistic_regression | -0.0109 | +0.0000 |
| adult | +0.0000 | neural_net | -0.0087 | +0.0000 |
| adult | +0.0000 | random_forest | -0.0012 | +0.0000 |
| adult | +0.0500 | gradient_boosting | -0.0053 | +0.0085 |
| adult | +0.0500 | logistic_regression | -0.0054 | +0.0061 |
| adult | +0.0500 | neural_net | -0.0047 | +0.0040 |
| adult | +0.0500 | random_forest | +0.0069 | +0.0066 |
| adult | +0.1000 | gradient_boosting | -0.0029 | +0.0131 |
| adult | +0.1000 | logistic_regression | +0.0007 | +0.0101 |
| adult | +0.1000 | neural_net | +0.0003 | +0.0092 |
| adult | +0.1000 | random_forest | +0.0118 | +0.0128 |
| adult | +0.2000 | gradient_boosting | +0.0105 | +0.0321 |
| adult | +0.2000 | logistic_regression | +0.0116 | +0.0202 |
| adult | +0.2000 | neural_net | +0.0041 | +0.0127 |
| adult | +0.2000 | random_forest | +0.0244 | +0.0277 |
| bank | +0.0000 | gradient_boosting | -0.0188 | +0.0000 |
| bank | +0.0000 | logistic_regression | -0.0147 | +0.0000 |
| bank | +0.0000 | neural_net | -0.0122 | +0.0000 |
| bank | +0.0000 | random_forest | -0.0187 | +0.0000 |
| bank | +0.0500 | gradient_boosting | -0.0149 | -0.0022 |
| bank | +0.0500 | logistic_regression | -0.0143 | +0.0037 |
| bank | +0.0500 | neural_net | -0.0253 | -0.0023 |
| bank | +0.0500 | random_forest | -0.0264 | -0.0041 |
| bank | +0.1000 | gradient_boosting | +0.0004 | +0.0086 |
| bank | +0.1000 | logistic_regression | -0.0094 | +0.0055 |
| bank | +0.1000 | neural_net | +0.0068 | +0.0237 |
| bank | +0.1000 | random_forest | -0.0172 | +0.0023 |
| bank | +0.2000 | gradient_boosting | -0.0127 | +0.0175 |
| bank | +0.2000 | logistic_regression | -0.0073 | +0.0136 |
| bank | +0.2000 | neural_net | +0.0441 | +0.0970 |
| bank | +0.2000 | random_forest | -0.0094 | +0.0187 |

**This failed as an always-on repair.** It helps at 20% label noise on Adult for all four models, but it **hurts on clean data (0% noise) and at 5% noise in most cells**, and on Bank it loses in most cells
(Bank has a 14% minority class; the filter drops the minority rows the model finds hard, which is a likely cause, not tested).

## 3. The filter as a gated candidate (fresh seeds 200-204; used only if it beats raw rows on a validation slice of the dirty data by more than 0.005)
| dataset | noise | model | always filter | gated filter | ceiling | filter used (share of seeds) |
|---|---|---|---|---|---|---|
| adult | +0.0000 | gradient_boosting | -0.0135 | +0.0000 | +0.0000 | +0.0000 |
| adult | +0.0000 | logistic_regression | -0.0108 | +0.0000 | +0.0000 | +0.0000 |
| adult | +0.0000 | neural_net | -0.0071 | -0.0028 | +0.0000 | +0.4000 |
| adult | +0.0000 | random_forest | -0.0034 | -0.0034 | +0.0000 | +0.4000 |
| adult | +0.0500 | gradient_boosting | -0.0063 | -0.0014 | +0.0076 | +0.2000 |
| adult | +0.0500 | logistic_regression | -0.0041 | +0.0000 | +0.0053 | +0.0000 |
| adult | +0.0500 | neural_net | +0.0002 | +0.0000 | +0.0135 | +0.0000 |
| adult | +0.0500 | random_forest | +0.0051 | +0.0005 | +0.0083 | +0.4000 |
| adult | +0.1000 | gradient_boosting | -0.0006 | +0.0000 | +0.0166 | +0.0000 |
| adult | +0.1000 | logistic_regression | +0.0036 | +0.0054 | +0.0141 | +0.6000 |
| adult | +0.1000 | neural_net | +0.0049 | +0.0000 | +0.0105 | +0.0000 |
| adult | +0.1000 | random_forest | +0.0189 | +0.0076 | +0.0204 | +0.4000 |
| adult | +0.2000 | gradient_boosting | +0.0169 | +0.0040 | +0.0385 | +0.4000 |
| adult | +0.2000 | logistic_regression | +0.0136 | +0.0045 | +0.0232 | +0.6000 |
| adult | +0.2000 | neural_net | +0.0053 | +0.0010 | +0.0184 | +0.2000 |
| adult | +0.2000 | random_forest | +0.0300 | +0.0143 | +0.0413 | +0.4000 |
| bank | +0.0000 | gradient_boosting | -0.0105 | +0.0000 | +0.0000 | +0.0000 |
| bank | +0.0000 | logistic_regression | -0.0160 | +0.0000 | +0.0000 | +0.0000 |
| bank | +0.0000 | neural_net | -0.0329 | -0.0117 | +0.0000 | +0.4000 |
| bank | +0.0000 | random_forest | -0.0119 | -0.0011 | +0.0000 | +0.2000 |
| bank | +0.0500 | gradient_boosting | -0.0155 | +0.0000 | +0.0041 | +0.0000 |
| bank | +0.0500 | logistic_regression | -0.0107 | +0.0000 | +0.0060 | +0.0000 |
| bank | +0.0500 | neural_net | +0.0397 | +0.0427 | +0.0693 | +0.4000 |
| bank | +0.0500 | random_forest | -0.0133 | -0.0011 | +0.0072 | +0.2000 |
| bank | +0.1000 | gradient_boosting | -0.0173 | -0.0088 | +0.0061 | +0.2000 |
| bank | +0.1000 | logistic_regression | -0.0130 | +0.0000 | +0.0117 | +0.0000 |
| bank | +0.1000 | neural_net | -0.0012 | +0.0023 | +0.0459 | +0.8000 |
| bank | +0.1000 | random_forest | -0.0289 | +0.0000 | +0.0008 | +0.0000 |
| bank | +0.2000 | gradient_boosting | -0.0056 | -0.0025 | +0.0334 | +0.2000 |
| bank | +0.2000 | logistic_regression | -0.0173 | +0.0000 | +0.0077 | +0.0000 |
| bank | +0.2000 | neural_net | +0.0168 | +0.0002 | +0.0575 | +0.2000 |
| bank | +0.2000 | random_forest | -0.0189 | +0.0000 | +0.0145 | +0.0000 |

## What this shows
* **Gating removes most of the harm and keeps part of the gain; it is not harm-free.** On Adult at 10-20% label noise the gated filter is positive for random forest (+0.008, +0.014), logistic regression (+0.005, +0.005) and gradient boosting at 20% (+0.004);
  that is between 10% (gradient boosting at 20%) and 38% (random forest at 10%, logistic regression at 10%) of the damage against the clean ceiling. It is not zero-harm: small losses remain on clean or lightly noisy data (Adult random forest -0.003 at 0%, Bank neural net -0.012 at 0%, Bank gradient boosting -0.009 at 10%).
* **It does not transfer cleanly to Bank Marketing.** Gains there are limited to the neural net (+0.043 at 5% noise, near zero elsewhere); the other three models are at zero or slightly negative.
* **So the honest statement is conditional:** cleaning label noise improves models when label noise is high (10-20%) and the data are balanced enough; below that, or on imbalanced data, a general filter can hurt, and the tool should decline to apply it, which the existing evidence gate already does.
* **Not integrated into the product.** The evidence is a modest, partly inconsistent effect from one exploration, not enough to put an automatic repair in front of users. It is a documented candidate, not a feature.

## Limits
Symmetric label flips on two datasets, 3-5 seeds per cell, one filter family, macro-F1, no other error types combined; the noise floors of Phase 20 were not recomputed for these cells, so differences of about 0.005 should be read as noise.
Bank was used twice (confirmation of the fixed filter, then the gated version on different seeds); the gated version's parameters were not changed between them.

## Next, if more is wanted
A class-balanced threshold (per-class) is the obvious fix for the Bank failure; it would have to be evaluated on new seeds and a third dataset, as everything here has already been looked at.


---

# Part 2: a real-file result (searched, then confirmed)

Requested by the user after Part 1: show, on at least one real file, that cleaning improved a model. Still **exploratory, not pre-registered**. The honest way to do this
is to search widely, report every result, then confirm the one candidate on data the search never used, with the claim written down before the confirmation ran.

## The search (`src/real_file_search.py`, `results/real_search_*.json`, `results/real_search_all.csv`)
The product's Model panel (repairs fitted on training rows only; models scored on untouched held-out rows of the user's own file; "improved" needs to clear the measured noise
floor with at least 80% of 5 seeds agreeing) was run on every real file in the project. Heart (303 rows) and Real Estate (414) are below the panel's minimum and were declined.
On the other eight files there were 217 model-by-repair comparisons: **3 improved, 11 got worse, 203 were within noise or unchanged.** The three improvements: Bank Marketing logistic
regression with outlier capping (+0.025 macro-F1, about 4 times the floor) and with the aggressive file (+0.024); Adult (original, dirty) gradient boosting with label fixes (+0.008, small).
The 11 "worse" include the same outlier capping on California: Ridge -0.33 and the neural net -0.28 R-squared. A scan like this finds some improvements by chance, so the best one needed confirming.

## The confirmation (`src/real_confirm_bank.py`, `results/real_confirm_bank*.{json,csv}`)
Claim fixed before the run: *the tool's tuned cleaning (Phase 20 rule: the repair is chosen per model on a validation slice of the training rows, kept only if it beats raw by more than 0.005;
nothing re-tuned) improves at least one model on real Bank Marketing rows in at least 4 of 5 seeds, and no model gets worse by more than its noise floor.* Fresh sample of 8,000 Bank rows
(sample seed 99, not the one used before), fresh seeds 100-104, 20% test rows never looked at until scoring. The test rows are raw rows of the file, as dirty as the file.

| Model | Mean change in macro-F1 | Seeds improved / worse | Chose a cleaning in | Noise floor |
|---|---|---|---|---|
| **Logistic regression** | **+0.030** (every seed: +0.014 to +0.041; raw 0.65-0.71) | **5 / 0** | 5 of 5 seeds | 0.006 |
| Neural net | +0.015 | 3 / 0 | 3 of 5 | 0.037 |
| Random forest | -0.002 | 1 / 2 | 3 of 5 | 0.014 |
| Gradient boosting | -0.008 (two seeds -0.019 and -0.021) | 1 / 2 | 3 of 5 | 0.016 |

**First part of the claim met:** logistic regression improved in 5 of 5 seeds by about five times its noise floor, on rows the search never saw, with the repair chosen by the tool from the training rows alone.
**Second part met on means, not on every seed:** the mean change of every other model is inside its noise floor, but gradient boosting was worse than its floor in two individual seeds (-0.0208, -0.0193 against 0.016).
Cleaning was selected (outlier capping, or the aggressive combination, for logistic regression) only where the validation slice supported it; when it did not, the raw rows were kept, which is why several cells are exactly 0.

## What this does and does not show
* **Shows:** on a real file, the tool's own impact-driven selection improved a real model's performance on held-out rows, reproducibly across fresh rows and seeds (the earlier panel runs on Bank
  also showed it: +0.024 and +0.034). The mechanism is plausible, not accidental: logistic regression is sensitive to extreme values in heavy-tailed numeric columns, which capping removes; the same capping badly hurts
  Ridge on California, which is exactly why the repair has to be chosen per model and per file, and not applied blindly.
* **Does not show:** (1) a gain for every model: random forest and gradient boosting did not gain, and gradient boosting lost a little; (2) a gain from the **default** file: the evidence-based default applies nothing on Bank, the gain came
  from the per-model tuned selection; (3) that the repairs were correct, because the held-out rows are as dirty as the file; (4) generalisation to other datasets: one dataset, a fresh sample of the same file, not an independent file;
  (5) anything pre-registered: the search and the choice of Bank came after seeing results, which is why the confirmation used fresh rows and a claim written in advance.
* **Selection caveat:** this is the best of 217 comparisons, confirmed once. The confirmation lowers, but does not remove, the chance that it is partly luck; the five per-seed gains were all positive and all at least twice the floor.
