# Stage 8 responsible-AI audit

## Scope and population

This is a post-freeze, development-only audit using Stage 6 honest nested-policy OOF probabilities re-applied at frozen threshold 0.16. Those probabilities reflect calibration selected separately inside each outer training fold, rather than one global sigmoid OOF series; they remain the prespecified honest OOF evidence available for responsible-AI diagnostics. The audit was not used to optimize the model. Audit attributes were joined only after prediction and never entered the predictive feature matrix.

## Audit-only attributes and subgroup evidence

`personal_status_sex`, age, and foreign-worker status remain excluded from prediction. Tables report n before rates and retain undefined metrics as NA. Groups smaller than 50 are especially unstable. Small or absent groups observed: personal_status_sex/A91 — male: divorced/separated: n=41; personal_status_sex/A95 — female: single: n=0; foreign_worker/A202 — no: n=31.

| audit_attribute     | group                                    |   n |   bad_risk_n |   good_risk_n | bad_risk_prevalence   | mean_predicted_bad_risk_probability   | predicted_positive_rate_at_0_16   | precision   | recall_sensitivity   | specificity   | false_positive_rate   | false_negative_rate   | balanced_accuracy   | brier_score   |
|:--------------------|:-----------------------------------------|----:|-------------:|--------------:|:----------------------|:--------------------------------------|:----------------------------------|:------------|:---------------------|:--------------|:----------------------|:----------------------|:--------------------|:--------------|
| personal_status_sex | A91 — male: divorced/separated           |  41 |           17 |            24 | 0.4146                | 0.3213                                | 0.7561                            | 0.5161      | 0.9412               | 0.375         | 0.625                 | 0.0588                | 0.6581              | 0.2177        |
| personal_status_sex | A92 — female: divorced/separated/married | 249 |           89 |           160 | 0.3574                | 0.3192                                | 0.739                             | 0.4457      | 0.9213               | 0.3625        | 0.6375                | 0.0787                | 0.6419              | 0.1934        |
| personal_status_sex | A93 — male: single                       | 430 |          112 |           318 | 0.2605                | 0.2924                                | 0.6628                            | 0.3579      | 0.9107               | 0.4245        | 0.5755                | 0.0893                | 0.6676              | 0.1507        |
| personal_status_sex | A94 — male: married/widowed              |  80 |           22 |            58 | 0.275                 | 0.2765                                | 0.6875                            | 0.3455      | 0.8636               | 0.3793        | 0.6207                | 0.1364                | 0.6215              | 0.1598        |
| personal_status_sex | A95 — female: single                     |   0 |            0 |             0 | NA                    | NA                                    | NA                                | NA          | NA                   | NA            | NA                    | NA                    | NA                  | NA            |
| age_band            | under 25                                 | 110 |           42 |            68 | 0.3818                | 0.3636                                | 0.8273                            | 0.4396      | 0.9524               | 0.25          | 0.75                  | 0.0476                | 0.6012              | 0.2228        |
| age_band            | 25–34                                    | 333 |          109 |           224 | 0.3273                | 0.3102                                | 0.7417                            | 0.4089      | 0.9266               | 0.3482        | 0.6518                | 0.0734                | 0.6374              | 0.1753        |
| age_band            | 35–44                                    | 204 |           47 |           157 | 0.2304                | 0.2665                                | 0.6324                            | 0.3333      | 0.9149               | 0.4522        | 0.5478                | 0.0851                | 0.6836              | 0.1411        |
| age_band            | 45–54                                    |  95 |           22 |            73 | 0.2316                | 0.2627                                | 0.5368                            | 0.3725      | 0.8636               | 0.5616        | 0.4384                | 0.1364                | 0.7126              | 0.1283        |
| age_band            | 55+                                      |  58 |           20 |            38 | 0.3448                | 0.3085                                | 0.6379                            | 0.4324      | 0.8                  | 0.4474        | 0.5526                | 0.2                   | 0.6237              | 0.1864        |
| foreign_worker      | A201 — yes                               | 769 |          237 |           532 | 0.3082                | 0.3044                                | 0.7022                            | 0.4         | 0.9114               | 0.391         | 0.609                 | 0.0886                | 0.6512              | 0.1718        |
| foreign_worker      | A202 — no                                |  31 |            3 |            28 | 0.0968                | 0.2069                                | 0.4839                            | 0.2         | 1.0                  | 0.5714        | 0.4286                | 0.0                   | 0.7857              | 0.0829        |

## Compound personal-status/sex limitation

The source field combines historical personal-status and sex codes. Sex cannot be cleanly isolated, modern gender concepts cannot be inferred, and these descriptive categories cannot support clean gender-fairness conclusions.

## Age audit

Age bands were frozen before metric calculation: under 25, 25–34, 35–44, 45–54, and 55+. No alternative banding was searched.

## Foreign-worker audit

The source-documented yes/no categories are reported with sample sizes. The smaller category produces unstable rate estimates and should not support strong conclusions.

## Proxy-risk register

Seven predefined socioeconomic predictors are reviewed qualitatively. Direct demographic exclusion does not remove their potential to encode social or economic structure.

| feature                 | proxy_risk_category                 |   importance_rank |   mean_absolute_shap | governance_concern                                             |
|:------------------------|:------------------------------------|------------------:|---------------------:|:---------------------------------------------------------------|
| checking_account_status | high proxy potential                |                 4 |             0.881027 | Potential proxy remains despite direct demographic exclusions. |
| savings_account_status  | high proxy potential                |                 3 |             0.893207 | Potential proxy remains despite direct demographic exclusions. |
| employment_duration     | high proxy potential                |                 5 |             0.655545 | Potential proxy remains despite direct demographic exclusions. |
| housing                 | high proxy potential                |                13 |             0.2333   | Potential proxy remains despite direct demographic exclusions. |
| property                | high proxy potential                |                 9 |             0.356735 | Potential proxy remains despite direct demographic exclusions. |
| telephone               | contextual/moderate proxy potential |                16 |             0.090177 | Potential proxy remains despite direct demographic exclusions. |
| job                     | high proxy potential                |                15 |             0.173673 | Potential proxy remains despite direct demographic exclusions. |

## Limited audit-attribute associations

The three audit attributes are crossed only with the five leading frozen-model source features, using Spearman rho, correlation ratio, or Cramer's V as appropriate. These summaries identify plausible relationships; they neither prove proxy discrimination nor causality. The ten largest observed magnitudes are shown below.

| audit_attribute     | model_feature           | association_method    |   association_value |
|:--------------------|:------------------------|:----------------------|--------------------:|
| age_years           | employment_duration     | correlation_ratio_eta |              0.399  |
| foreign_worker      | purpose                 | cramers_v             |              0.1707 |
| age_years           | purpose                 | correlation_ratio_eta |              0.1706 |
| personal_status_sex | purpose                 | cramers_v             |              0.1628 |
| personal_status_sex | employment_duration     | cramers_v             |              0.1585 |
| age_years           | credit_history          | correlation_ratio_eta |              0.1557 |
| foreign_worker      | checking_account_status | cramers_v             |              0.1172 |
| age_years           | savings_account_status  | correlation_ratio_eta |              0.1101 |
| personal_status_sex | credit_history          | cramers_v             |              0.1028 |
| foreign_worker      | employment_duration     | cramers_v             |              0.0822 |

## Error analysis

At 0.16, development OOF error groups were: false_positive=336, true_negative=224, true_positive=219, false_negative=21. Numeric and selected categorical summaries are descriptive. This analysis occurred after policy freeze and was not used for feature engineering or optimization.

## Interpretation and governance recommendations

Subgroup performance differs, but fairness cannot be established from this dataset. Estimates are limited by small samples, historical compound coding, proxy predictors, 1970s West German context, oversampling, and uncertain transportability. Maintain human review, document context, monitor subgroup data quality and error rates in any new setting, and require jurisdiction-specific legal and ethical review before any real use. No subgroup-specific threshold is recommended.
