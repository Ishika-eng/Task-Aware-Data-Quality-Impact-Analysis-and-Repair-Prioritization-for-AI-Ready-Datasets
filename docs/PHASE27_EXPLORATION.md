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
