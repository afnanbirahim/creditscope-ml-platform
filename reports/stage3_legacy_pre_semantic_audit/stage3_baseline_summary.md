# Stage 3: holdout isolation and baseline modelling

## Scope and governance

Stage 3 represents application-time credit-risk assessment before a lending
decision. The target is `bad_credit_risk` (0 = good credit risk, 1 = bad credit
risk), with class 1 treated as positive. This is not an observed-default target.

The one stratified 80/20 split uses `random_state=42`: 800 development rows
(560 good, 240 bad) and 200 final-test rows (140 good, 60 bad). Its deterministic
source-position manifest SHA-256 is
`32fb4c9ac2cdb358e3bffd145cd97e1a393c14946888d46212d0b395523e99af`.
No final-test performance was calculated or inspected.

`personal_status_sex`, `age_years`, and `foreign_worker` are excluded from the
17 predictive features and retained as aligned audit attributes. This is a
conservative responsible-AI choice for this educational prototype, not a claim
about jurisdiction-specific legal requirements. Potential proxy variables remain
predictive candidates for later sensitivity and fairness analysis.

## Method

Both baselines used the same five-fold `StratifiedKFold` design with shuffling
and `random_state=42`, applied only to development data. Numeric inputs were
median-imputed defensively and standardized. The source currently has no missing
values. Officially ordered employment and job categories were ordinal-encoded;
other categorical features were one-hot encoded with unknown categories ignored.
All fitting occurred inside pipelines and within each training fold.

Logistic Regression used `C=1.0`, `max_iter=2000`, and no class weight. It was
not tuned. No resampling, feature selection, threshold search, or probability
calibration was performed.

## Development-only cross-validation results

| Model | ROC-AUC | Avg precision | Precision (bad) | Recall (bad) | F1 (bad) | Balanced accuracy | Brier | Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Dummy prior | 0.500 | 0.300 | 0.000 | 0.000 | 0.000 | 0.500 | 0.210 | 0.700 |
| Logistic Regression | 0.770 | 0.581 | 0.576 | 0.442 | 0.500 | 0.651 | 0.173 | 0.735 |

These aggregate values use aligned out-of-fold predictions across all 800
development rows. Logistic Regression's fold ROC-AUC mean was 0.771 (SD 0.046),
and its fold average-precision mean was 0.583 (SD 0.087). Complete fold results
and mean/SD tables are stored alongside this report.

The dummy prior model predicts every row as good at threshold 0.50, producing
70% accuracy while finding none of the bad-credit-risk cases. Accuracy alone is
therefore misleading.

## Fixed-threshold cost and calibration

The confusion-matrix orientation is actual rows `[good=0, bad=1]` by predicted
columns `[good=0, bad=1]`. A false negative is a bad risk predicted as good and
costs 5; a false positive is a good risk predicted as bad and costs 1.

At the unoptimized 0.50 threshold, the dummy model had 240 false negatives,
total cost 1,200, and average cost 1.500. Logistic Regression had 134 false
negatives and 78 false positives, total cost 748, and average cost 0.935. These
are development OOF diagnostics, not final-test estimates.

The Logistic Regression OOF Brier score was 0.173 versus 0.210 for the prior
baseline. The reliability plot is diagnostic only; no calibrator was fitted.

## Coefficients and limitations

The largest absolute development-fit coefficients include purpose A46
(education; positive bad-risk score), no checking account A14 (negative),
critical/other credit history A34 (negative), used-car purpose A41 (negative),
and unknown/no property A124 (positive). All categories are retained by one-hot
encoding, so categorical coefficients are regularized parameterization terms,
not contrasts against a dropped reference category. Numeric and ordinal
coefficients use standardized scales. These are associations, not causal effects,
and rare categories may yield unstable estimates in this small dataset.

The dataset's age, geography, sample size, incomplete timing documentation, and
proxy variables limit generalization. Variables are assumed available at
application time only because UCI does not identify them as post-outcome; this is
a limitation rather than certainty.

## Decisions deferred beyond Stage 3

- Whether proxy-risk variables should remain after sensitivity/fairness work.
- How candidate models should be compared without accessing the final holdout.
- Whether probability calibration is needed.
- How asymmetric cost should guide later threshold selection.

No model has been selected as final or described as production-ready.
