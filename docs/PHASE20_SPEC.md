# Phase 20: a controlled proof on a clean held-out set (does the repaired data train better models?)

Written **2026-11-17, before any code or run**. Fixed in advance; the outcome is reported whatever it is. Anything that has to
change afterwards is appended as a visible Amendment. Additive: writes only `results/phase20_*`, frozen files untouched.

## Why

The Phase 19 model panel compares raw and repaired training rows on held-out rows that are **as dirty as the user's file**, so a
gain there cannot show that a repair was *right*. The fix is the one the earlier phases used: start from data known to be clean,
inject known errors, repair with the real TaskClean path, and score on a **clean** held-out set. Then "repaired beats dirty" means
the repair moved the data toward the truth. This phase puts that evaluation in the product, for all four model families, and
adds one new capability: a **tuned cleaning** arm that picks repairs from the user's own (dirty) validation data.

## Design (fixed now)

* **Data.** UCI Adult, `data/adult_clean.csv`, a fixed stratified sample of 8,000 rows (`random_state=7`), target `class`.
* **Per seed (42-46):** stratified 80/20 split of the clean rows. The 20% **test rows stay clean and are never touched.** All five
  error types (missing values, outliers, label errors, feature corruption, duplicates) are injected into the **training rows only**
  at 5% each with the Phase 3 corruptors (`build_combined_dirty`), the same recipe as the demo file.
* **Arms, all scored on the clean test rows** with the four panel models (settings as `src/model_panel.py`):
  1. **Clean ceiling**: trained on the clean training rows (what perfect cleaning could reach).
  2. **Dirty (control)**: trained on the dirty training rows.
  3. **Default file**: the real `audit_dataset` -> `_apply` with the evidence-based policy's auto-approved repairs.
  4. **Aggressive file**: every proposed repair except feature-anomaly repairs (as the product's second file).
  5. **Each repair type alone** (duplicates, missing values, outliers, label fixes).
  6. **Tuned cleaning (new).** Per model: split the *dirty* training rows 75/25 (train' / validation), apply each candidate arm
     (aggressive and each single repair, fitted on train' only), score on the dirty validation rows, and keep the best candidate
     **only if it beats the raw arm on validation by more than 0.005 macro-F1**; otherwise keep the raw data. The chosen repair
     set is then applied to the full dirty training rows, the model is retrained, and it is scored on the clean test rows. Selection
     uses no clean data and no test data. The 0.005 margin is fixed here, not tuned.
* **Metric:** macro-F1 on the clean test rows (primary); ROC-AUC reported.
* **Noise floor** per model from a null experiment (retrain on a random 90% of the clean training rows, 3 draws per seed); a change
  is *improved* / *worse* only if it exceeds the floor and >= 80% of seeds agree on its sign (Phase 10's rule), else *within noise*.
* **Reported:** damage (clean ceiling minus dirty), recovery (arm minus dirty) and the share of the damage recovered, per model and
  arm, **for every arm and model, favourable or not.**

## Criteria (fixed now)

* **C0, evaluable.** The corruption must damage the models: clean ceiling beats dirty beyond the noise floor for at least 3 of the
  4 models. Otherwise recovery cannot be read and the run is reported "not evaluable".
* **C1, completeness.** Every arm for every model appears in the saved results and in the page. Nothing is dropped.
* **C2, the headline question.** *Tuned cleaning* is **improved** (beyond noise, >= 80% of seeds) on at least 2 of the 4 models and
  **worse on none**. Passing is called a positive result; failing is called a negative result, and the README says which.
* **C3, safety of the default.** The default file is not *worse* than dirty for any model. (It is expected to equal dirty, since the
  evidence-based policy approves no automatic repair on Adult evidence; that is reported as "no change", not as a gain.)
* **C4, honesty about the gap.** Feature corruption is injected but not repaired by TaskClean (flag-only), so the clean ceiling is
  not expected to be reached. The share of damage recovered is reported as measured.
* **C5, selection is not leaking.** A unit test shows the tuned arm's selection never reads clean-test rows or clean-train rows.

## Predictions (recorded before the run; wrong ones are reported)

(1) the default file equals dirty; (2) the aggressive file helps logistic regression and hurts at least one tree model or the neural
net (as the panel's dirty-test run showed); (3) tuned cleaning is never worse than dirty by more than the selection margin on
average, because it falls back to raw when nothing clears the margin; (4) tuned cleaning recovers only a minority of the damage,
because feature corruption is left in.

## Not claimed

That this holds on other datasets, other corruption mixes or rates, or real-world errors (the corruptions are synthetic and random).
That repairs are correct cell by cell (the Phase 9 precision/recall numbers do that). That the result transfers to a user's file:
on an upload there is still no clean truth, and the Model panel remains the indicative view.

---

# Amendment A1 (2026-11-17, after a one-seed, two-model smoke test, before the registered run)

**What the smoke test showed.** It was a bug check, not the registered run (1 seed, logistic regression and random forest). For the
random forest, tuned cleaning chose the aggressive file because it scored higher on the dirty validation rows, but it scored lower on
the clean test rows. Cause: the dirty training rows contain exact duplicates, and a random validation slice of them holds rows that
also sit in train'. A model that keeps the duplicates has memorised those rows, so duplicate removal looks harmful on validation.
This is the same leakage the Phase 19 panel counts.

**Change.** The tuned arm now removes from the validation slice every row whose full record also appears in train' (uses only the dirty
data). **The original selection is kept and reported too** as `tuned_v0` ("raw validation", exactly as specified above), so the effect
of the change is visible and cannot be hidden. The headline arm `tuned` is the leak-free variant.

**Criteria.** C2 is evaluated for both variants and both are reported; the pass/fail called "the" result is that of `tuned`.
If `tuned` passes and `tuned_v0` fails, the README says so and says the leak-free validation was added after a smoke test.
C1 now expects 11 arms per model (the new `tuned_v0` included). Everything else is unchanged.

---

# Results (2026-11-17, registered run: 5 seeds, 8,000 Adult rows, 224 s; `results/phase20_proof*.{json,csv}`)

Reported as measured. Nothing was re-run, re-tuned or re-selected after seeing these numbers.

| Model | Clean ceiling | Dirty | Damage | Damage beyond noise? | Tuned vs dirty | Verdict |
|---|---|---|---|---|---|---|
| Logistic regression | 0.7951 | 0.7611 | +0.0340 | yes (floor 0.0033) | +0.0151, 5/5 seeds, 44% of the damage | **improved** |
| Random forest | 0.7848 | 0.7801 | +0.0047 | **no** (floor 0.0056) | -0.0001 | within noise |
| Gradient boosting | 0.8092 | 0.7993 | +0.0099 | yes (floor 0.0083) | +0.0025 | within noise |
| Neural net (MLP) | 0.7909 | 0.7769 | +0.0140 | yes (floor 0.0108) | +0.0049 | within noise |

**Criteria.** C0 evaluable: **pass** (3 of 4 models damaged beyond noise). C1 complete: **pass** (44 cells). **C2 tuned positive:
FAIL** (improved on 1 of 4 models, worse on none; same for `tuned_v0`, so the leak-free change of A1 did not change the verdict).
C3 default not worse: **pass**, trivially (the default file equals dirty: "no change" on every model). C4: feature corruption left
in, ceiling not reached, as expected (best share recovered 44%, logistic regression only). C5: `test_phase20.py`.

**Predictions.** (1) default equals dirty: right. (2) aggressive helps logistic regression and hurts a tree model or the net: half
right, it helped logistic regression (+0.0147, improved) but did not significantly hurt any model on clean test rows (the panel on
dirty test rows had shown harm on GB and MLP; that harm was not reproduced here, which is itself a finding). (3) tuned never worse
than dirty beyond the margin: right (no model is worse; random forest -0.0001). (4) recovers a minority of the damage: right (44%).

**What can be claimed.** On this setup, repairing the injected errors raises logistic regression's macro-F1 on clean held-out rows by
about 0.015 (consistent across all 5 seeds, 44% of what the errors cost it), duplicate removal alone accounts for nearly all of it
(+0.0144), and the tuned selector never made any model worse than leaving the data alone. **What cannot be claimed:** that the tool
improves model families other than logistic regression: for random forest, gradient boosting and the MLP the changes are positive
or zero in most arms but inside the noise floor, and for random forest the errors did not even damage the model beyond noise.
The pre-registered positive criterion (two or more families) was not met and the README says so.
