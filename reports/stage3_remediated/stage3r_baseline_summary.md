# Stage 3R semantically remediated baseline

## Scope

Stage 3R corrects only the preprocessing assumptions identified by the Stage 3A
audit. UCI dataset 144 remains the modelling data and UCI dataset 573 supplies
corrected semantic guidance. The locked split, development rows, CV folds, model
definitions, threshold, and cost matrix are unchanged. No final-test prediction
or performance calculation was made.

The original Stage 3 results are preserved under
`reports/stage3_legacy_pre_semantic_audit/` and are superseded for future model
comparison by this remediated baseline.

## Locked data boundary

- Development: 800 rows (560 good credit risk, 240 bad credit risk)
- Final test: 200 rows (140 good credit risk, 60 bad credit risk)
- Split manifest SHA-256:
  `32fb4c9ac2cdb358e3bffd145cd97e1a393c14946888d46212d0b395523e99af`
- Bad credit risk (1) remains the positive class.

The three audit-only attributes remain `personal_status_sex`, `age_years`, and
`foreign_worker`. They are retained with aligned row identifiers and excluded
from preprocessing and prediction.

## Remediated feature representation

Stage 3R numeric features are `duration_months` and `credit_amount`. Both use
median imputation for inference robustness and standard scaling. The source has
no missing values. Credit amount is DM-denominated but underwent an unknown
monotonic transformation, so its coefficient must not be read as a literal
effect per original DM.

The other 15 predictive fields use most-frequent imputation and
`OneHotEncoder(handle_unknown="ignore")`. This includes ordinal and discretized
variables, avoiding an unsupported equal-step linear-log-odds assumption.

## Development-only cross-validation

Both models use the original five-fold `StratifiedKFold` with shuffling and
`random_state=42` on the same 800 development rows.

| Model | ROC-AUC | Average precision | Precision (bad) | Recall (bad) | F1 (bad) | Balanced accuracy | Brier | Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Dummy prior | 0.500 | 0.300 | 0.000 | 0.000 | 0.000 | 0.500 | 0.210 | 0.700 |
| Logistic Regression | 0.763 | 0.571 | 0.575 | 0.446 | 0.502 | 0.652 | 0.176 | 0.735 |

Logistic Regression fold means and sample standard deviations were:

- ROC-AUC: 0.766 ± 0.050
- Average precision: 0.577 ± 0.071
- Precision for bad credit risk: 0.577 ± 0.086
- Recall for bad credit risk: 0.446 ± 0.084
- F1: 0.501 ± 0.078
- Balanced accuracy: 0.652 ± 0.047
- Brier score: 0.176 ± 0.021
- Accuracy: 0.735 ± 0.036

Accuracy remains secondary: the all-good dummy obtains 70% accuracy while
identifying no bad-credit-risk observations.

## Confusion matrix and fixed-threshold cost

At threshold 0.50, the remediated Logistic Regression OOF confusion matrix is:

|  | Predicted good (0) | Predicted bad (1) |
|---|---:|---:|
| Actual good (0) | 481 | 79 |
| Actual bad (1) | 133 | 107 |

The official cost is `5 × false negatives + 1 × false positives`, where a false
negative is an actual bad risk predicted as good. Logistic Regression therefore
has total cost `5 × 133 + 79 = 744`, or 0.930 per development observation. The
dummy has total cost 1,200, or 1.500 per observation. No threshold search occurred.

## Calibration

The remediated Logistic Regression OOF Brier score is 0.1762, compared with
0.2100 for the dummy prior. The reliability plot is diagnostic; no probability
calibration was fitted.

## Original versus remediated Logistic Regression

| Metric | Original Stage 3 | Stage 3R | Absolute difference |
|---|---:|---:|---:|
| ROC-AUC | 0.7701 | 0.7630 | 0.0072 |
| Average precision | 0.5806 | 0.5710 | 0.0096 |
| Precision | 0.5761 | 0.5753 | 0.0008 |
| Recall | 0.4417 | 0.4458 | 0.0042 |
| F1 | 0.5000 | 0.5023 | 0.0023 |
| Balanced accuracy | 0.6512 | 0.6524 | 0.0012 |
| Brier score | 0.1730 | 0.1762 | 0.0032 |
| Accuracy | 0.7350 | 0.7350 | 0.0000 |
| False positives | 78 | 79 | 1 |
| False negatives | 134 | 133 | 1 |
| True positives | 106 | 107 | 1 |
| True negatives | 482 | 481 | 1 |
| Total 5:1 cost | 748 | 744 | 4 |
| Average cost | 0.935 | 0.930 | 0.005 |

These small changes are consequences of a different, better-supported feature
representation. They are not independently evidence that one metric movement is
universally beneficial or harmful.

## Coefficients

The largest absolute coefficients include education purpose (`A46`, higher
bad-risk score), no checking account (`A14`, lower), critical/other credit
history (`A34`, lower), used-car purpose (`A41`, lower), and low savings (`A61`,
higher). One-hot coefficients are regularized conditional model parameters, not
causal effects or standalone category effects. Because all levels are retained,
they are not ordinary contrasts against a single dropped reference category.

No convergence warning occurred. This is an untuned educational baseline, not a
selected final or production model.
