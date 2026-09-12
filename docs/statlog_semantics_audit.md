# Statlog German Credit semantic audit

Stage 3A audit date: 2026-09-11.

This audit changes no dataset, split, preprocessing implementation, model,
prediction, threshold, or Stage 3 result. It compares the existing CreditScope
representation with UCI's corrected semantic reference before Stage 4.

## Authoritative references and retrieval limits

- UCI [Statlog (German Credit Data), dataset 144](https://archive.ics.uci.edu/dataset/144/statloggermancreditdata),
  DOI [10.24432/C5NC77](https://doi.org/10.24432/C5NC77). Donated
  1994-11-16; page retrieved 2026-09-11.
- UCI [South German Credit, dataset 573](https://archive.ics.uci.edu/dataset/573/south%2Bgerman%2Bcredit%2Bupdate),
  DOI [10.24432/C5QG88](https://doi.org/10.24432/C5QG88). Donated
  2020-06-19; page retrieved 2026-09-11.

UCI describes ID 573 as a correction and background enhancement based on a
representation of the same German credit data. Its page warns that the older
Statlog coding information contained severe errors. The official pages and their
variable descriptions were retrieved. The ID 573 ZIP was reachable but could
not be decoded by the web retrieval service, while direct runtime network access
was blocked. Consequently, the accompanying `codetable.txt`, R reader, and ASCII
data were not independently parsed. Exact numeric code translation and row-level
equivalence are therefore not claimed.

## Feature-by-feature conclusion

The machine-readable 20-row comparison is
`reports/stage3a_semantics_comparison.csv`. It has one unique row per current
CreditScope feature, no blank fields, and preserves schema order.

| CreditScope feature | Corrected ID 573 meaning/type | Stage 3 type | Recommendation | Change? |
|---|---|---|---|---|
| `checking_account_status` | debtor's checking-account status; categorical | nominal | retain one-hot; no-account is substantive | no |
| `duration_months` | credit duration in months; quantitative | numerical | retain numeric scaling | no |
| `credit_history` | compliance with previous or concurrent credit contracts; categorical | nominal | retain one-hot; document clarified scope | no |
| `purpose` | purpose for which credit is needed; categorical | nominal | retain one-hot | no |
| `credit_amount` | DM amount after an unknown monotonic transformation; quantitative | numerical | retain numeric scaling; correct interpretation | no preprocessing change |
| `savings_account_status` | debtor's savings; categorical | nominal | retain one-hot; unknown/no-account is substantive | no |
| `employment_duration` | current-employer duration; ordinal discretization | ordinal | one-hot for baseline LR | yes |
| `installment_rate_percent` | installment-rate band; ordinal discretization | numerical | one-hot for baseline LR | yes |
| `personal_status_sex` | compound category from which sex cannot be recovered | audit-only nominal | retain raw only as limited audit attribute | no predictive change |
| `other_debtors_guarantors` | another debtor or guarantor; categorical | nominal | retain one-hot | no |
| `residence_duration` | years at present residence; ordinal discretization | numerical | one-hot for baseline LR | yes |
| `property` | most valuable property, highest applicable code; ordinal | nominal | retain conservative one-hot; fix documentation | no |
| `age_years` | age in years; quantitative | audit-only numeric | retain audit-only | no predictive change |
| `other_installment_plans` | plans from providers other than lending bank; categorical | nominal | retain one-hot | no |
| `housing` | housing type; categorical | nominal | retain one-hot | no |
| `existing_credits_count` | current/past bank-credit count including current credit; ordinal discretization | numerical | one-hot for baseline LR | yes |
| `job` | job quality; ordinal | ordinal | one-hot for baseline LR | yes |
| `dependents_count` | financially dependent persons; binary discretization | numerical | binary indicator or one-hot | yes |
| `telephone` | registered landline; binary and historically specific | nominal | retain one-hot; flag proxy/transportability risk | no |
| `foreign_worker` | foreign-worker indicator; binary | audit-only nominal | retain audit-only | no predictive change |

### Material semantic corrections

- `credit_amount` must not be presented as an untouched contractual amount. UCI
  says it is denominated in DM but was subjected to an unknown monotonic
  transformation.
- `employment_duration`, `installment_rate_percent`, `residence_duration`, and
  `existing_credits_count` are discretized ordered variables. Their stored codes
  do not establish equal distances.
- `existing_credits_count` includes the current credit and credits the debtor
  has or had at the bank; UCI says the original values are unavailable.
- `dependents_count` is binary/discretized rather than a general count.
- `property` is the most valuable applicable property, using the highest
  applicable code, rather than four unrelated ownership labels.
- `job` represents job quality. Order is supported, but equal step sizes are not.
- `personal_status_sex` does not permit reliable reconstruction of sex. UCI says
  one code combines male singles and female non-singles, and female widows are
  not cleanly covered by the female categories.
- `credit_history` concerns compliance with previous or concurrent contracts.
- ID 573 describes the target as whether the contract was complied with (good)
  or not (bad). CreditScope must retain good/bad credit-risk terminology rather
  than equating the target with an observed default event.

## Encoding implications

For Logistic Regression, using an ordinal integer as one column imposes a single
linear change in log odds for each adjacent code step. Order alone does not
justify equal spacing or a constant effect. One-hot encoding is the conservative
baseline for the six affected predictors because it allows level-specific
associations. It uses more parameters but avoids an unsupported functional form.

Tree models do not impose a linear coefficient, but an ordinal integer still
restricts splits to contiguous portions of the chosen order. One-hot encoding
permits more flexible groupings at the cost of dimensionality. Later tree work
may compare these encodings through development-only sensitivity analysis, but
no tree model is trained in this audit.

## Category-code findings and limitations

ID 144 supplies symbolic `Axx` labels. ID 573 uses corrected integer coding and
refers to its code table and R reader for detailed labels. Because those official
attachments could not be parsed here, this audit does not assert a complete
code-to-code crosswalk, reordered code values, or cell-level identity.

The accessible official descriptions nevertheless establish material differences:

- the Statlog `personal_status_sex` labels cannot be used to derive a reliable
  standalone sex attribute;
- property is an ordered maximum-property construction;
- employment duration, installment rate, residence duration, number of credits,
  and dependents are bands/discretizations rather than raw measurements; and
- no-account, unknown/no-savings, and no-property values remain substantive
  categories, not missing data.

## Dataset-equivalence assessment

Both UCI pages report 1,000 observations and a 700/300 good/bad balance. ID 573
describes a 1973–1975 stratified sample of actual credits with bad credits heavily
oversampled, and calls itself a corrected representation of the same data.
Target meaning is compatible at the good/bad level. The accessible page does not
establish the exact numeric target-code mapping needed for a cell comparison.

Matching dimensions, class counts, and UCI's “same data” description support
shared lineage, but do not prove row order or predictor equivalence. Because the
official ID 573 files could not be parsed, predictor equivalence after documented
recoding and any irreconcilable cell differences remain **not verified**.

## Project data policy

**Recommendation: option B.** Continue using ID 144 as the modelling dataset,
while treating ID 573 as the authoritative correction and clarification source
for feature semantics. Do not silently replace the raw data. Migration to ID 573
is not currently justified because exact code and row equivalence could not be
verified; it should be reconsidered only after retrieving and hashing the official
ID 573 files and constructing a defensible crosswalk.

## Effect on Stage 3

The split remains valid as a deterministic partition of the unchanged ID 144
rows. The Stage 3 implementation is reproducible, but its feature representation
is not the preferred corrected-semantics Logistic Regression baseline.

Stage 3R subsequently completed the authorized remediation without changing the
split. Original results are preserved under
`reports/stage3_legacy_pre_semantic_audit/`. Remediated metrics, coefficients,
figures, and the before/after comparison are under
`reports/stage3_remediated/`, which is authoritative for future comparison.
The executed `notebooks/03_baseline_modelling.ipynb` tells the Stage 3R story.

## Responsible-AI implications

The corrected source strengthens the project's limitations: the sample is old,
bad risks were oversampled, monetary values were transformed, and several inputs
are coarse socioeconomic bands. A 1970s registered landline is particularly weak
for modern transportability. Most importantly, `personal_status_sex` cannot
support a defensible sex-based fairness conclusion. It should remain excluded
from prediction and retained only as a problematic compound audit attribute with
that limitation made explicit.
