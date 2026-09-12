# Stage 7 one-time final holdout evaluation

The sealed 200-row holdout was accessed once after the Stage 6P policy freeze. No model, feature, preprocessing, calibration, or threshold change followed access.

## Frozen policy

- Fixed untuned Stage 4 XGBoost pipeline
- Sigmoid `CalibratedClassifierCV`, exact persisted five-fold development splits, `ensemble=True`, `n_jobs=None`
- Five fold-specific classifier/calibrator pairs; probabilities are averaged
- Threshold `0.16`
- Cost `5 * FN + FP`

## Final metrics and development comparison

| evidence                       |   roc_auc |   average_precision |   precision_bad_credit_risk |   recall_bad_credit_risk |   f1_bad_credit_risk |   balanced_accuracy |   specificity |   brier_score |   log_loss |   accuracy |   true_negative |   false_positive |   false_negative |   true_positive |   total_cost |   average_cost |
|:-------------------------------|----------:|--------------------:|----------------------------:|-------------------------:|---------------------:|--------------------:|--------------:|--------------:|-----------:|-----------:|----------------:|-----------------:|-----------------:|----------------:|-------------:|---------------:|
| Stage 6 nested development OOF |  0.778746 |            0.573177 |                    0.417154 |                 0.891667 |             0.568393 |            0.678869 |      0.466071 |      0.168334 |   0.507829 |   0.593750 |             261 |              299 |               26 |             214 |          429 |       0.536250 |
| Stage 7 final locked holdout   |  0.804643 |            0.679212 |                    0.398496 |                 0.883333 |             0.549223 |            0.655952 |      0.428571 |      0.152338 |   0.475878 |   0.565000 |              60 |               80 |                7 |              53 |          115 |       0.575000 |

ROC-AUC and Average Precision describe discrimination; Brier score and log loss describe probability quality; threshold metrics and 5:1 cost describe the frozen decision policy.

The 5:1 error cost is an educational/research modelling assumption, not a claim about real-world lending economics. Predictions are good-credit-risk or bad-credit-risk classes, not lending approvals or rejections.

No confidence intervals, alternative thresholds, alternative models, SHAP analysis, or post-hoc model development were performed.
