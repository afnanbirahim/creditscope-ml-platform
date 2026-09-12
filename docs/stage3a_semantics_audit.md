# Stage 3A data-semantics audit

> Historical audit status: the recommended remediation was completed in Stage
> 3R. See `reports/stage3_remediated/` for the authoritative baseline and
> `reports/stage3_legacy_pre_semantic_audit/` for preserved original results.

Retrieved 2026-09-11. This audit did not train a model, alter the locked split,
score the final test set, or optimize a threshold.

## Authoritative UCI references

- [Statlog (German Credit Data), ID 144](https://archive.ics.uci.edu/dataset/144/statlog%2Bgerman%2Bcredit%2Bdata),
  DOI [10.24432/C5NC77](https://doi.org/10.24432/C5NC77), donated 1994-11-16.
- [South German Credit, ID 573](https://archive.ics.uci.edu/dataset/573/south%2Bgerman%2Bcredit),
  DOI [10.24432/C5QG88](https://doi.org/10.24432/C5QG88), donated 2020-06-19.

UCI describes ID 573 as a correction and background enhancement of the same
German credit data and states that the older Statlog coding information had
severe errors as of November 2019. ID 573 identifies the observations as a
stratified sample of actual credits from 1973–1975, with bad credits heavily
oversampled. It reports 700 good and 300 bad credits.

The UCI pages and their displayed variable information were accessible. The
downloadable ID 573 archive, code table, R reader, and ASCII data were blocked
by this execution environment. Therefore exact South-German numeric code labels
and row-by-row value equivalence were not independently reproduced here. The
audit does not fill those gaps from third-party copies.

## Main semantic corrections

The complete 20-row comparison is in
`reports/stage3a_semantics_comparison.csv`. Material findings are:

- `credit_amount` is denominated in DM but is the result of an unknown monotonic
  transformation. It should not be described as the original contractual amount.
- `employment_duration`, `installment_rate_percent`, `residence_duration`, and
  `existing_credits_count` are ordinal, discretized quantitative variables.
  Their stored integers do not establish equal intervals.
- `existing_credits_count` includes the current credit and credits the debtor has
  or had at the bank. UCI says the original values are unavailable. It is not an
  exact count of currently existing credits.
- `dependents_count` is binary, discretized quantitative rather than a general
  count measurement.
- `property` is ordinal and records the debtor's most valuable property using the
  highest applicable code. The older description omits this construction rule.
- `job` is ordinal and means job quality. This supports an ordering but does not
  establish equal distances between categories.
- `personal_status_sex` cannot support a clean sex derivation. UCI says code 2
  combines male singles with female non-singles, while female widows are not
  clearly covered by the female categories. The current Statlog labels therefore
  must not be used to infer a reliable standalone sex attribute.
- `credit_history` concerns compliance with previous or concurrent credit
  contracts. This is clearer than the older generic label and reinforces the
  application-time timing assumption as a limitation that needs verification.
- The corrected target describes whether the credit contract was complied with
  (good) or not (bad). CreditScope should continue using good/bad credit-risk
  terminology and should not silently relabel the outcome as observed default.

## Variable-type decisions

| Feature | Corrected semantic type | Logistic Regression recommendation | Tree-model recommendation |
|---|---|---|---|
| `employment_duration` | ordinal, discretized quantitative | one-hot by default; ordered linear encoding only as a documented sensitivity | ordered category or one-hot |
| `installment_rate_percent` | ordinal, discretized quantitative | one-hot | ordered category or one-hot |
| `residence_duration` | ordinal, discretized quantitative | one-hot | ordered category or one-hot |
| `property` | ordinal categorical | retain one-hot | ordered category or one-hot |
| `existing_credits_count` | ordinal, discretized quantitative | one-hot | ordered category or one-hot |
| `job` | ordinal categorical | one-hot by default; ordered linear encoding only as a sensitivity | ordered category or one-hot |
| `dependents_count` | binary, discretized quantitative | binary indicator or one-hot | binary category or indicator |

For Logistic Regression, an integer or ordinal encoding estimates one linear
change in log odds per code step. That assumes equal spacing and a constant
direction across adjacent levels. The corrected documentation establishes order
for some fields but not equal spacing. One-hot encoding is the conservative
baseline because it permits a separate association for each level. It costs more
parameters but avoids an unsupported linear shape.

Tree models can make threshold splits on ordered codes without assuming a linear
effect size. Such encoding still constrains partitions to respect the supplied
order. One-hot encoding permits arbitrary category groupings indirectly but can
increase dimensionality. Both choices should be compared through development-only
sensitivity analysis when a later stage authorizes tree models.

## Category-code findings and limits

ID 144 uses symbolic `Axx` categories. The ID 573 page uses corrected integer
representations and directs users to its code table and R reader for exact labels.
Because those attachments could not be retrieved here, this audit does not claim
a complete code-to-code crosswalk.

The accessible official page does establish these category-level differences:

- `personal_status_sex` contains a collapsed code that prevents recovery of sex,
  contradicting any interpretation of the Statlog categories as clean sex groups.
- `property` represents the highest-valued applicable property rather than four
  unrelated nominal ownership labels.
- `existing_credits_count`, `dependents_count`, installment rate, residence time,
  and employment duration are recoded bands or discretizations, not raw amounts.
- Special categories such as no checking account or unknown/no savings remain
  substantive categories rather than missing values.

Exact claims about every corrected numeric code's label, direction, or one-to-one
relationship with an `Axx` code are deferred until the official attachments can
be retrieved and hashed.

## Do the data differ?

Both UCI pages report 1,000 observations and the same 700/300 good/bad balance.
UCI describes ID 573 as a corrected representation of the same data, not as a
new population sample. However, identical dimensions and class counts do not
prove row order or cell-level equivalence. The official ID 573 data file was not
available to this runtime, so this audit records row-level equivalence as **not
verified**. No changes were made to the ID 144 raw snapshot or split manifest.

## Effect on Stage 3

Current one-hot treatment of `property` remains conservative. Numeric treatment
of installment rate, residence duration, number of credits, and dependents is not
supported by corrected semantics. Ordinal encoding of employment duration and
job respects order but imposes an unverified equal-step linear effect in Logistic
Regression. The Stage 3 pipeline is reproducible, but its feature representation
is not the preferred corrected-semantics baseline.

Stage 3 metrics should be treated as superseded once a controlled remediation is
authorized. Regeneration should occur before Stage 4, using development-only CV,
the same locked memberships, and no final-test evaluation. The current results
must remain available as historical artifacts rather than being silently replaced.

## Dataset-lineage recommendation

Do not silently replace ID 144. Prefer a documented migration to ID 573 as the
canonical semantic and data source, conditional on retrieving the official files
and proving the row/value relationship to ID 144. If the records can be mapped
without ambiguity, preserve the locked development/test memberships. If not,
stop and obtain explicit approval before establishing a new data lineage or split.

Until that verification is possible, retain ID 144 as the actual project data,
mark its old codebook as unreliable, use ID 573 for corrected feature-level
semantics, and do not advance to Stage 4 modelling.

## Responsible-AI implications

The corrected documentation strengthens the project's caution. The sample is old,
bad risks were oversampled, amount values were transformed, and several variables
are coarse socioeconomic bands. Telephone reflects 1970s landline ownership.
Most importantly, `personal_status_sex` does not provide a valid isolated sex
attribute, so later fairness work must not report sex-group conclusions from it.
It may still be retained as a problematic compound audit attribute with explicit
limitations. Excluding it from prediction remains appropriate.
