# Stage 2 data understanding and EDA summary

## 1. Dataset overview

The verified UCI source contains 1,000 observations and 20 candidate features. It contains no missing cells and no duplicate complete rows. Stage 2 created no model and no train/test split.

## 2. Target semantics

UCI defines source class 1 as good credit risk and class 2 as bad credit risk. The analysis view preserves `credit_risk_class` and adds `bad_credit_risk` (0 = good, 1 = bad). The target must not be interpreted as an observed loan-default event.

## 3. Main descriptive findings

There are 700 good-credit-risk records (70.0%) and 300 bad-credit-risk records (30.0%). The documented categorical schema contains 3 observed categories below the predeclared 2% rarity threshold. Detailed numerical ranges and category counts are in the generated CSV tables.

## 4. Important EDA observations

- For `checking_account_status`, observed bad-credit-risk rates ranged from 11.7% for A14 (no checking account, n=394) to 49.3% for A11 (... < 0 DM, n=274).
- For `savings_account_status`, observed bad-credit-risk rates ranged from 12.5% for A64 (.. >= 1000 DM, n=48) to 36.0% for A61 (... < 100 DM, n=603).
- For `credit_history`, observed bad-credit-risk rates ranged from 17.1% for A34 (critical account / other credits existing (not at this bank), n=293) to 62.5% for A30 (no credits taken / all credits paid back duly, n=40).
- For `purpose`, observed bad-credit-risk rates ranged from 11.1% for A48 (retraining, n=9) to 44.0% for A46 (education, n=50).
- Median duration was 18 months for good-credit-risk records and 24 months for bad-credit-risk records.
- Median credit amount was 2244 for good-credit-risk records and 2574 for bad-credit-risk records. UCI does not state a unit for this field in its current variable metadata.

These are sample associations, not causal effects. Small category counts make some rates unstable.

## 5. Class imbalance

The 70%/30% split is a moderate imbalance: both classes are represented, but an all-good classifier would already achieve 70% accuracy. Later evaluation therefore needs class-specific metrics and cost-sensitive analysis. No resampling or class weighting was applied.

## 6. Sensitive-feature considerations

`personal_status_sex`, `age_years`, and `foreign_worker` are explicitly demographic or sensitive in context. Employment, job, housing, property, savings, checking-account status, residence duration, dependents, and telephone may act as proxies. These labels do not assert jurisdiction-specific protected-class status. No feature was removed in Stage 2.

## 7. Potential leakage concerns

No feature is an obvious copy of the target or explicit post-outcome field. However, UCI does not fully document measurement timing. Credit history, existing credits at this bank, checking status, savings, other installment plans, and debtors/guarantors require confirmation that they were known at the decision point. The compound personal-status/sex field cannot cleanly separate its components. These are Stage 3 review flags, not reasons for automatic removal.

## 8. Cost-matrix implications

UCI assigns cost 5 to a bad credit risk classified as good and cost 1 to a good credit risk classified as bad. Later evaluation and threshold selection must reflect this asymmetry. No threshold was selected in Stage 2.

## 9. Dataset limitations

The dataset is small, donated in 1994, geographically specific, and sparsely documented on sampling and measurement timing. Several values are opaque codes, some categories are rare, and demographic/proxy variables create fairness concerns. It cannot establish causal effects or demonstrate suitability for contemporary lending decisions.

## 10. Decisions required before modelling

1. Define the intended decision-support use and evaluation population.
2. Choose whether sensitive fields are excluded from prediction, retained only for auditing, or used under a documented research rationale.
3. Decide how to treat likely proxy variables and the compound personal-status/sex field.
4. Confirm application-time availability and define a leakage-safe feature set.
5. Choose preprocessing for nominal, ordinal, special unknown/no-account, and rare categories.
6. Predefine validation design, metrics, cost-sensitive evaluation, and final holdout isolation.
