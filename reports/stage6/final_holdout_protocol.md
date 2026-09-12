# Frozen one-time final-holdout protocol

> This document defines Stage 7 but does not execute it. No holdout values were inspected in Stage 6.

1. Verify the raw snapshot, split manifest, Stage 4 fold-membership, and frozen-policy SHA-256 values.
2. Load exactly the 17 frozen predictive features; keep the three audit-only attributes outside prediction.
3. Construct the unchanged Stage 3R tree preprocessing and fixed untuned Stage 4 XGBoost.
4. On the 800 development observations, fit `CalibratedClassifierCV(estimator=<fixed Stage 4 XGBoost pipeline>, method="sigmoid", cv=<exact persisted Stage 4 five-fold splits>, ensemble=True)`. `n_jobs` is not passed and therefore resolves to `None` under scikit-learn 1.4.2. This creates five fold-specific classifier/calibrator pairs; it does not train one XGBoost on all 800 rows.
5. Generate probabilities for the sealed 200-row holdout once. The final predictor averages the five calibrated probabilities produced by those fold-specific pairs.
6. Apply the already frozen threshold `0.16` without modification.
7. Report only the metrics predefined below. Do not alter the model, features, preprocessing, calibration, or threshold after seeing results.

## Predefined metrics

- `roc_auc`
- `average_precision`
- `precision_bad_credit_risk`
- `recall_bad_credit_risk`
- `f1_bad_credit_risk`
- `balanced_accuracy`
- `specificity`
- `brier_score`
- `log_loss`
- `accuracy`
- `true_negative`
- `false_positive`
- `false_negative`
- `true_positive`
- `total_cost`
- `average_cost`

## Confidence intervals

No confidence-interval procedure is authorized or predefined. Stage 7 must not add one without explicit authorization before holdout access.

The holdout is a one-time evaluation, not a new model-selection dataset.
