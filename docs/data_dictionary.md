# CreditScope data dictionary

Source used in Stages 1–3: [UCI Statlog (German Credit Data), dataset ID 144](https://archive.ics.uci.edu/dataset/144/statlog%2Bgerman%2Bcredit%2Bdata). Category wording below follows UCI's `german.doc`. Blank category cells identify numerical fields. “Not documented” means no unit was supplied by UCI.

> **Stage 3A warning:** UCI states that the older Statlog coding information has
> severe errors. Do not treat this document as the final semantic authority for
> modelling. See `docs/stage3a_semantics_audit.md` and the corrected
> [South German Credit documentation, dataset ID 573](https://archive.ics.uci.edu/dataset/573/south%2Bgerman%2Bcredit).

The original target is preserved as `credit_risk_class`: 1 = good credit risk and 2 = bad credit risk. The analysis-only `bad_credit_risk` maps these to 0 and 1.

The ID 144 category labels below remain recorded for reproducibility. Under the
ID 573 correction, `personal_status_sex` is a problematic compound code from
which sex cannot be cleanly recovered; it must not support standalone sex-group
inference. Stage 3R keeps the raw field only as a limited audit attribute.

| UCI field | Internal field | Official meaning | Official type | Analytical type | Unit |
|---|---|---|---|---|---|
| Attribute1 | `checking_account_status` | Status of existing checking account | Categorical | ordered categorical with a special no-account category | not documented |
| Attribute2 | `duration_months` | Duration | Integer | discrete numerical | months |
| Attribute3 | `credit_history` | Credit history | Categorical | nominal categorical | not documented |
| Attribute4 | `purpose` | Purpose | Categorical | nominal categorical | not documented |
| Attribute5 | `credit_amount` | Credit amount | Integer | discrete numerical | DM; corrected source says values underwent an unknown monotonic transformation |
| Attribute6 | `savings_account_status` | Savings account/bonds | Categorical | ordered categorical with a special unknown/no-account category | not documented |
| Attribute7 | `employment_duration` | Present employment since | Categorical | ordinal categorical | not documented |
| Attribute8 | `installment_rate_percent` | Installment rate in percentage of disposable income | Integer | ordinal, discretized quantitative | percentage bands of disposable income |
| Attribute9 | `personal_status_sex` | Personal status and sex | Categorical | nominal categorical (compound field) | not documented |
| Attribute10 | `other_debtors_guarantors` | Other debtors / guarantors | Categorical | nominal categorical | not documented |
| Attribute11 | `residence_duration` | Present residence since | Integer | ordinal, discretized quantitative | years, discretized |
| Attribute12 | `property` | Most valuable applicable property | Categorical | ordinal categorical | not documented |
| Attribute13 | `age_years` | Age | Integer | discrete numerical | years |
| Attribute14 | `other_installment_plans` | Other installment plans | Categorical | nominal categorical | not documented |
| Attribute15 | `housing` | Housing | Categorical | nominal categorical | not documented |
| Attribute16 | `existing_credits_count` | Number of credits including the current one that the debtor has or had at this bank | Integer | ordinal, discretized quantitative | credit-count bands; original values unavailable |
| Attribute17 | `job` | Job | Categorical | ordinal categorical | not documented |
| Attribute18 | `dependents_count` | Number of people financially dependent on the debtor | Integer | binary, discretized quantitative | people, discretized |
| Attribute19 | `telephone` | Telephone | Binary | binary categorical | not documented |
| Attribute20 | `foreign_worker` | Foreign worker | Binary | binary categorical | not documented |

## Official categorical codes

### Attribute1: `checking_account_status`

| Code | Official meaning |
|---|---|
| A11 | ... < 0 DM |
| A12 | 0 <= ... < 200 DM |
| A13 | ... >= 200 DM / salary assignments for at least 1 year |
| A14 | no checking account |

### Attribute3: `credit_history`

| Code | Official meaning |
|---|---|
| A30 | no credits taken / all credits paid back duly |
| A31 | all credits at this bank paid back duly |
| A32 | existing credits paid back duly till now |
| A33 | delay in paying off in the past |
| A34 | critical account / other credits existing (not at this bank) |

### Attribute4: `purpose`

| Code | Official meaning |
|---|---|
| A40 | car (new) |
| A41 | car (used) |
| A42 | furniture/equipment |
| A43 | radio/television |
| A44 | domestic appliances |
| A45 | repairs |
| A46 | education |
| A47 | (vacation - does not exist?) |
| A48 | retraining |
| A49 | business |
| A410 | others |

### Attribute6: `savings_account_status`

| Code | Official meaning |
|---|---|
| A61 | ... < 100 DM |
| A62 | 100 <= ... < 500 DM |
| A63 | 500 <= ... < 1000 DM |
| A64 | .. >= 1000 DM |
| A65 | unknown / no savings account |

### Attribute7: `employment_duration`

| Code | Official meaning |
|---|---|
| A71 | unemployed |
| A72 | ... < 1 year |
| A73 | 1 <= ... < 4 years |
| A74 | 4 <= ... < 7 years |
| A75 | .. >= 7 years |

### Attribute9: `personal_status_sex`

| Code | Official meaning |
|---|---|
| A91 | male: divorced/separated |
| A92 | female: divorced/separated/married |
| A93 | male: single |
| A94 | male: married/widowed |
| A95 | female: single |

### Attribute10: `other_debtors_guarantors`

| Code | Official meaning |
|---|---|
| A101 | none |
| A102 | co-applicant |
| A103 | guarantor |

### Attribute12: `property`

| Code | Official meaning |
|---|---|
| A121 | real estate |
| A122 | if not A121: building society savings agreement / life insurance |
| A123 | if not A121/A122: car or other, not in Attribute6 |
| A124 | unknown / no property |

### Attribute14: `other_installment_plans`

| Code | Official meaning |
|---|---|
| A141 | bank |
| A142 | stores |
| A143 | none |

### Attribute15: `housing`

| Code | Official meaning |
|---|---|
| A151 | rent |
| A152 | own |
| A153 | for free |

### Attribute17: `job`

| Code | Official meaning |
|---|---|
| A171 | unemployed / unskilled - non-resident |
| A172 | unskilled - resident |
| A173 | skilled employee / official |
| A174 | management / self-employed / highly qualified employee / officer |

### Attribute19: `telephone`

| Code | Official meaning |
|---|---|
| A191 | none |
| A192 | yes, registered under the customer's name |

### Attribute20: `foreign_worker`

| Code | Official meaning |
|---|---|
| A201 | yes |
| A202 | no |

## Target

| Source value | Source meaning | `bad_credit_risk` |
|---:|---|---:|
| 1 | good credit risk | 0 |
| 2 | bad credit risk | 1 |
