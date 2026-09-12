# CreditScope inference contract

## Scope

This contract covers local inference with `creditscope-model-1.0.0`. CreditScope
returns a good/bad credit-risk decision-support classification. It does not
return or imply approval, rejection, eligibility, or a lending decision.

## Strict input object

Each record must contain exactly these 17 fields:

| Field | Contract |
|---|---|
| `duration_months` | Required finite number; no undocumented bound is invented. |
| `credit_amount` | Required finite historical transformed quantity; not a literal contemporary monetary amount. |
| `checking_account_status` | One of `A11`, `A12`, `A13`, `A14`. |
| `credit_history` | One of `A30`, `A31`, `A32`, `A33`, `A34`. |
| `purpose` | One of `A40`, `A41`, `A42`, `A43`, `A44`, `A45`, `A46`, `A47`, `A48`, `A49`, `A410`. |
| `savings_account_status` | One of `A61`, `A62`, `A63`, `A64`, `A65`. |
| `employment_duration` | One of `A71`, `A72`, `A73`, `A74`, `A75`; discretized duration. |
| `installment_rate_percent` | Integer level `1`–`4`; not a literal percentage measurement. |
| `other_debtors_guarantors` | One of `A101`, `A102`, `A103`. |
| `residence_duration` | Integer level `1`–`4`; discretized duration. |
| `property` | One of `A121`, `A122`, `A123`, `A124`; no/unknown property is substantive. |
| `other_installment_plans` | One of `A141`, `A142`, `A143`. |
| `housing` | One of `A151`, `A152`, `A153`. |
| `existing_credits_count` | Integer level `1`–`4`; discretized count information. |
| `job` | One of `A171`, `A172`, `A173`, `A174`; represented categorically. |
| `dependents_count` | Integer level `1` or `2`; binary/discretized quantity. |
| `telephone` | `A191` or `A192`. |

The machine-readable source-code meanings and caveats are in
`artifacts/model/input_schema.json`.

Missing, duplicate, unexpected, null, NaN, infinite, wrongly typed, and invalid
categorical values are rejected. The audit-only fields `personal_status_sex`,
`age_years`, and `foreign_worker` are rejected rather than ignored.

## Output

Successful inference returns `probability_bad_risk`, `predicted_class`,
`predicted_label`, `threshold`, and `model_version`. Class `0` means
`good_credit_risk`; class `1` means `bad_credit_risk`. The sole decision rule is
bad credit risk when `probability_bad_risk >= 0.16`. Batch output preserves input
order.

## Failure and purity guarantees

The module exposes concise typed errors for contract, artifact, and compatibility
failures. It performs no fit, mutation, artifact write, external call, holdout
access, SHAP calculation, or threshold comparison. Core prediction has no SHAP
dependency.

