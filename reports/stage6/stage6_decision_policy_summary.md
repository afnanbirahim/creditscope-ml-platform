# Stage 6 calibration and threshold policy

Stage 5 tuning remains preserved negative evidence. Stage 6 uses only the fixed untuned Stage 4 XGBoost and the locked 800-row development population.

## Honest nested policy estimate

| policy                                               |   roc_auc |   average_precision |   precision_bad_credit_risk |   recall_bad_credit_risk |   f1_bad_credit_risk |   balanced_accuracy |   brier_score |   accuracy |   true_negative |   false_positive |   false_negative |   true_positive |   total_cost |   average_cost |
|:-----------------------------------------------------|----------:|--------------------:|----------------------------:|-------------------------:|---------------------:|--------------------:|--------------:|-----------:|----------------:|-----------------:|-----------------:|----------------:|-------------:|---------------:|
| Stage 4 XGBoost, threshold 0.50                      |  0.780982 |            0.581782 |                    0.612565 |                 0.487500 |             0.542923 |            0.677679 |      0.170874 |   0.753750 |             486 |               74 |              123 |             117 |          689 |       0.861250 |
| Nested calibration-and-threshold selection (Stage 6) |  0.778746 |            0.573177 |                    0.417154 |                 0.891667 |             0.568393 |            0.678869 |      0.168334 |   0.593750 |             261 |              299 |               26 |             214 |          429 |       0.536250 |

## Outer-fold policy choices

|   outer_fold | selected_calibration_method   |   selected_threshold |   inner_selected_brier_score |   inner_selected_log_loss |   inner_selected_roc_auc |   inner_selected_average_precision |   outer_brier_score |
|-------------:|:------------------------------|---------------------:|-----------------------------:|--------------------------:|-------------------------:|-----------------------------------:|--------------------:|
|            1 | isotonic                      |             0.200000 |                     0.173801 |                  0.574902 |                 0.749337 |                           0.576008 |            0.158442 |
|            2 | isotonic                      |             0.180000 |                     0.167248 |                  0.509257 |                 0.776908 |                           0.586506 |            0.158839 |
|            3 | isotonic                      |             0.150000 |                     0.173182 |                  0.567016 |                 0.765439 |                           0.566342 |            0.161019 |
|            4 | sigmoid                       |             0.160000 |                     0.170228 |                  0.514931 |                 0.769252 |                           0.602275 |            0.171032 |
|            5 | isotonic                      |             0.200000 |                     0.157949 |                  0.480333 |                 0.802258 |                           0.640970 |            0.192336 |

Selected thresholds ranged from 0.15 to 0.20 (mean 0.178, sample SD 0.023). The theoretical 1/6 threshold is a reference, not a forced choice.

## Frozen development-wide policy

- Calibration: `sigmoid`
- Threshold: `0.16`
- Model: fixed untuned Stage 4 XGBoost
- Cost: `5 × FN + FP`

## Key development thresholds

| threshold_label           |   threshold |   brier_score |   log_loss |   roc_auc |   average_precision |   precision_bad_credit_risk |   recall_bad_credit_risk |   f1_bad_credit_risk |   balanced_accuracy |   specificity |   accuracy |   true_negative |   false_positive |   false_negative |   true_positive |   total_cost |   average_cost |
|:--------------------------|------------:|--------------:|-----------:|----------:|--------------------:|----------------------------:|-------------------------:|---------------------:|--------------------:|--------------:|-----------:|----------------:|-----------------:|-----------------:|----------------:|-------------:|---------------:|
| Default 0.50              |    0.500000 |      0.167385 |   0.507536 |  0.784427 |            0.588089 |                    0.652778 |                 0.391667 |             0.489583 |            0.651190 |      0.910714 |   0.755000 |             510 |               50 |              146 |              94 |          780 |       0.975000 |
| Theoretical 1/6 reference |    0.166667 |      0.167385 |   0.507536 |  0.784427 |            0.588089 |                    0.415094 |                 0.916667 |             0.571429 |            0.681548 |      0.446429 |   0.587500 |             250 |              310 |               20 |             220 |          410 |       0.512500 |
| Empirically selected      |    0.160000 |      0.167385 |   0.507536 |  0.784427 |            0.588089 |                    0.411765 |                 0.933333 |             0.571429 |            0.680952 |      0.428571 |   0.580000 |             240 |              320 |               16 |             224 |          400 |       0.500000 |

No holdout prediction, metric, calibration fit, threshold selection, SHAP analysis, feature selection, or additional hyperparameter search occurred.
