# Stage 4 untuned tree-model comparison

All results below use out-of-fold predictions on the locked 800-row development partition. The final 200-row holdout was not scored or inspected.

## Locked methodology

- Logistic Regression is the unchanged Stage 3R reference.
- Random Forest and XGBoost are single, untuned candidates.
- All models use the same five precomputed stratified folds.
- The class threshold is fixed at 0.50. No class weighting was used.
- Bad credit risk (1) is positive; cost is `5 * FN + 1 * FP`.
- Direct audit attributes are excluded; 2 numeric and 15 one-hot fields remain.

## Aggregate development OOF comparison

| model               |   roc_auc |   average_precision |   recall_bad_credit_risk |   balanced_accuracy |   brier_score |   accuracy |   false_positive |   false_negative |   total_cost |
|:--------------------|----------:|--------------------:|-------------------------:|--------------------:|--------------:|-----------:|-----------------:|-----------------:|-------------:|
| Logistic Regression |    0.7630 |              0.5710 |                   0.4458 |              0.6524 |        0.1762 |     0.7350 |               79 |              133 |          744 |
| Random Forest       |    0.7769 |              0.5816 |                   0.3792 |              0.6467 |        0.1689 |     0.7538 |               48 |              149 |          793 |
| XGBoost             |    0.7810 |              0.5818 |                   0.4875 |              0.6777 |        0.1709 |     0.7538 |               74 |              123 |          689 |

Accuracy is secondary: predicting every observation as good risk reaches 70% accuracy on this development population while missing every bad-risk case.

## Fold variability

| model               | metric            |   fold_mean |   fold_std |
|:--------------------|:------------------|------------:|-----------:|
| Logistic Regression | roc_auc           |      0.7655 |     0.0501 |
| Logistic Regression | average_precision |      0.5768 |     0.0713 |
| Logistic Regression | brier_score       |      0.1762 |     0.0214 |
| Random Forest       | roc_auc           |      0.7789 |     0.0241 |
| Random Forest       | average_precision |      0.5986 |     0.0728 |
| Random Forest       | brier_score       |      0.1689 |     0.0091 |
| XGBoost             | roc_auc           |      0.7836 |     0.0376 |
| XGBoost             | average_precision |      0.5961 |     0.0773 |
| XGBoost             | brier_score       |      0.1709 |     0.0186 |

## Complexity diagnostic

Random Forest mean train-minus-validation ROC-AUC gap: 0.2211.
XGBoost mean train-minus-validation ROC-AUC gap: 0.1898.
These gaps are diagnostics, not tuning criteria in this stage.

## Model agreement

All three candidates assign the same 0.50-threshold class to 649 of 800 development observations (81.1%).

## Candidate tradeoffs

- Strongest ROC-AUC: XGBoost.
- Strongest Average Precision: XGBoost.
- Strongest bad-risk recall at 0.50: XGBoost.
- Strongest bad-risk F1 at 0.50: XGBoost.
- Strongest balanced accuracy at 0.50: XGBoost.
- Lowest 5:1 cost at 0.50: XGBoost.
- Lowest Brier score: Random Forest.

These development-only results do not select a final or production model. Stage 5 must decide whether controlled tuning is justified and how to govern it.

## Importance limitations

Random Forest impurity importance and XGBoost gain-derived importance are model-specific diagnostics. Correlation and one-hot expansion can split or distort importance, and impurity measures may favor variables with more split opportunities. Importance is not causality and was not used for feature selection.

The leading encoded Random Forest importance was `numerical__credit_amount` (0.1122); the leading XGBoost importance was `categorical__checking_account_status_A14` (0.0696).

## Probability-quality diagnostic

Mean absolute gaps across ten quantile reliability bins were Logistic Regression: 0.0615, Random Forest: 0.0404, XGBoost: 0.0541. These descriptive bin gaps and Brier scores do not constitute probability calibration, and the small dataset limits interpretation.
