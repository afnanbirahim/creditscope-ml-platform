# Stage 5 controlled XGBoost tuning

Nested cross-validation was performed only on the locked 800-row development partition. The final 200-row holdout was not scored, predicted, or used in preprocessing or search.

## Design

- The persisted Stage 4 folds are the five outer evaluation folds.
- Each outer-training partition used a four-fold stratified inner search.
- Every inner search used the frozen 40-draw search space and selected by Average Precision.
- XGBoost used one thread per fit while the search controlled parallelism.
- Threshold remained 0.50; 5:1 cost was monitored but not optimized.
- No weighting, resampling, calibration, SHAP, feature selection, or final-holdout evaluation occurred.

## Honest nested OOF comparison

| model                          |   roc_auc |   average_precision |   precision_bad_credit_risk |   recall_bad_credit_risk |   f1_bad_credit_risk |   balanced_accuracy |   brier_score |   accuracy |   false_positive |   false_negative |   total_cost |
|:-------------------------------|----------:|--------------------:|----------------------------:|-------------------------:|---------------------:|--------------------:|--------------:|-----------:|-----------------:|-----------------:|-------------:|
| Untuned XGBoost (Stage 4)      |    0.7810 |              0.5818 |                      0.6126 |                   0.4875 |               0.5429 |              0.6777 |        0.1709 |     0.7538 |               74 |              123 |          689 |
| Nested-tuned XGBoost (Stage 5) |    0.7736 |              0.5674 |                      0.5976 |                   0.4208 |               0.4939 |              0.6497 |        0.1711 |     0.7412 |               68 |              139 |          763 |

Nested tuning changed 56 classifications; 23 changed from wrong to correct and 33 changed from correct to wrong.

## Outer-fold variability

| metric                    |   outer_fold_mean |   outer_fold_sample_sd |
|:--------------------------|------------------:|-----------------------:|
| roc_auc                   |            0.7784 |                 0.0352 |
| average_precision         |            0.5748 |                 0.0802 |
| precision_bad_credit_risk |            0.5811 |                 0.1209 |
| recall_bad_credit_risk    |            0.4208 |                 0.1498 |
| f1_bad_credit_risk        |            0.4846 |                 0.1443 |
| balanced_accuracy         |            0.6497 |                 0.0771 |
| brier_score               |            0.1711 |                 0.0166 |
| accuracy                  |            0.7412 |                 0.0489 |
| total_cost                |          152.6000 |                36.3772 |
| average_cost              |            0.9537 |                 0.2274 |

## Hyperparameter stability

The five outer folds selected 4 distinct configurations. This is evidence about tuning stability, not a reason to force one retrospective configuration.

|   outer_fold |   best_inner_average_precision |   subsample |   reg_lambda |   reg_alpha |   n_estimators |   min_child_weight |   max_depth |   learning_rate |   gamma |   colsample_bytree |
|-------------:|-------------------------------:|------------:|-------------:|------------:|---------------:|-------------------:|------------:|----------------:|--------:|-------------------:|
|      1.00000 |                        0.57413 |     1.00000 |     10.00000 |     0.10000 |      700.00000 |            3.00000 |     5.00000 |         0.08000 | 0.25000 |            0.80000 |
|      2.00000 |                        0.59784 |     0.70000 |      5.00000 |     0.01000 |      700.00000 |            3.00000 |     3.00000 |         0.03000 | 0.00000 |            1.00000 |
|      3.00000 |                        0.59915 |     0.80000 |      5.00000 |     0.01000 |      250.00000 |            3.00000 |     2.00000 |         0.03000 | 0.00000 |            0.90000 |
|      4.00000 |                        0.61681 |     0.70000 |     10.00000 |     0.01000 |      350.00000 |            3.00000 |     5.00000 |         0.03000 | 1.00000 |            0.80000 |
|      5.00000 |                        0.66731 |     0.70000 |     10.00000 |     0.01000 |      350.00000 |            3.00000 |     5.00000 |         0.03000 | 1.00000 |            0.80000 |

## Train-versus-validation diagnostic

Nested-tuned mean ROC-AUC gap: 0.1447; Stage 4 untuned mean ROC-AUC gap: 0.1898.

## Final development-wide configuration search

The following configuration was selected by a separate 40-draw search on all 800 development rows:

```json
{
  "subsample": 0.7,
  "reg_lambda": 10.0,
  "reg_alpha": 0.01,
  "n_estimators": 350,
  "min_child_weight": 3,
  "max_depth": 5,
  "learning_rate": 0.03,
  "gamma": 1.0,
  "colsample_bytree": 0.8
}
```

Its internal CV score is selection evidence, not an unbiased performance estimate. The nested OOF result above is the honest development estimate of the tuning procedure.

## Interpretation

Average Precision changed from 0.5818 to 0.5674; total 5:1 cost at the unchanged threshold changed from 689 to 763. Small changes must be interpreted alongside fold variability and configuration instability.
