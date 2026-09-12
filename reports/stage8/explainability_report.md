# Stage 8 explainability report

## Frozen final model architecture

The frozen predictor is the Stage 3R tree preprocessor plus fixed untuned Stage 4 XGBoost inside a five-fold sigmoid-calibrated `CalibratedClassifierCV(ensemble=True)`, followed by threshold 0.16. The fitted Stage 7 Python object was not serialized; Stage 8 therefore deterministically reconstructs and refits that exact frozen architecture on the same 800 development rows solely to inspect its five fitted member pipelines. It does not fit an unrelated sixth model, refit for performance, or access holdout rows.

## Why calibrated-ensemble explanation is nontrivial

Each member has its own fitted preprocessing, XGBoost model, and sigmoid calibrator. Final probabilities average five calibrated member probabilities. Tree SHAP therefore does not exactly decompose the final probability.

## Exact SHAP methodology and scale

`shap.TreeExplainer(model_output="raw")` explains each underlying XGBoost member's raw tree margin. Encoded levels are aligned by canonical `ColumnTransformer` name. Statistics for an encoded level average only across members where that level exists; absence is reported separately and is not treated as zero effect.

## Global findings

The five leading original features by ensemble mean absolute raw-margin contribution were: purpose, credit_history, savings_account_status, checking_account_status, employment_duration. Rankings describe this frozen model, not causal importance.

|   rank | original_feature         |   mean_absolute_shap |   sd_member_mean_absolute_shap |
|-------:|:-------------------------|---------------------:|-------------------------------:|
|      1 | purpose                  |             1.76191  |                       0.575258 |
|      2 | credit_history           |             0.934977 |                       0.143247 |
|      3 | savings_account_status   |             0.893207 |                       0.14734  |
|      4 | checking_account_status  |             0.881027 |                       0.057061 |
|      5 | employment_duration      |             0.655545 |                       0.084248 |
|      6 | other_debtors_guarantors |             0.537202 |                       0.303037 |
|      7 | installment_rate_percent |             0.406364 |                       0.196791 |
|      8 | other_installment_plans  |             0.403083 |                       0.121506 |
|      9 | property                 |             0.356735 |                       0.156712 |
|     10 | credit_amount            |             0.30846  |                       0.025521 |

## Original-feature aggregation

For every member, absolute contributions across a source feature's one-hot levels were summed per observation and then averaged. These totals reconcile to encoded-level absolute totals within `1e-10`. This source-feature view is preferred because one-hot encoding fragments importance.

## Ensemble-member stability

Rankings varied across fold-specific members; the largest rank SD was for `property` (3.674). This is descriptive evidence of finite-sample/model instability.

## Local development cases

Cases were selected before inspecting SHAP patterns: minimum OOF probability, closest to 0.16, maximum probability, highest-probability false positive, and lowest-probability false negative, without row reuse. Contributors are ensemble means of underlying raw tree-score SHAP values, not calibrated-probability effects. The OOF probability comes from the honest Stage 6 nested-policy artifact and is re-thresholded at 0.16; fold-selected calibration methods varied, so it is not a row-level decomposition of the final full-development sigmoid ensemble.

| selection_rule       | source_row_id   |   bad_credit_risk |   stage6_oof_probability |   prediction_at_frozen_threshold_0_16 | outcome_type   | top_underlying_tree_score_contributors                                                                                                              |
|:---------------------|:----------------|------------------:|-------------------------:|--------------------------------------:|:---------------|:----------------------------------------------------------------------------------------------------------------------------------------------------|
| low_probability      | uci144-row-0009 |                 0 |                0         |                                     0 | true_negative  | savings_account_status=-0.466245; credit_amount=-0.449533; purpose=+0.440437; credit_history=+0.420654; checking_account_status=-0.320018           |
| closest_to_threshold | uci144-row-0701 |                 1 |                0.159944  |                                     0 | false_negative | savings_account_status=-0.472579; checking_account_status=-0.466242; purpose=+0.409103; credit_history=+0.402771; duration_months=-0.248354         |
| high_probability     | uci144-row-0877 |                 0 |                0.959264  |                                     1 | false_positive | savings_account_status=-0.503349; credit_history=+0.388846; checking_account_status=-0.378846; purpose=+0.354646; other_installment_plans=+0.202317 |
| false_positive       | uci144-row-0651 |                 0 |                0.878185  |                                     1 | false_positive | savings_account_status=-0.571891; duration_months=+0.377215; purpose=+0.292071; checking_account_status=-0.284263; credit_history=+0.228916         |
| false_negative       | uci144-row-0336 |                 1 |                0.0106383 |                                     0 | false_negative | duration_months=-0.771002; purpose=+0.491129; credit_history=+0.465590; savings_account_status=-0.454938; checking_account_status=-0.230876         |

## Semantic cautions

`credit_amount` is transformed historical DM-denominated information with an undocumented transformation. Employment duration, installment-rate category, residence duration, existing-credit count, job, and dependents are discretized/categorical concepts. No plotted pattern supports a literal continuous dose-response claim.

## Non-causal interpretation

All explanations are conditional associations learned from a small historical dataset. They do not establish causes, lending entitlements, or individual counterfactual outcomes.
