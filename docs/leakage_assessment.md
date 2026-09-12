# Stage 2 leakage-risk assessment

No candidate feature is an obvious copy of the target, and none is explicitly
documented by UCI as post-outcome information. This is not sufficient to declare
the feature set leakage-free because the repository does not fully document when
each field was measured relative to the credit-risk classification or decision.

Fields needing an application-time availability decision in Stage 3 include
`credit_history`, `existing_credits_count`, `checking_account_status`,
`savings_account_status`, `other_installment_plans`, and
`other_debtors_guarantors`. Their names suggest information that may have been
available at assessment, but that timing is not confirmed by UCI and must not be
assumed.

Other concerns are semantic rather than direct leakage:

- `personal_status_sex` combines sex and personal/marital status in one code, so
  its components cannot be cleanly separated from this source.
- `checking_account_status`, `savings_account_status`, and `property` include
  unknown/no-account or unknown/no-property categories. These are substantive
  categories, not ordinary missing values.
- Several fields may be strongly associated with the target. Association alone
  is not evidence of leakage and is not a reason to remove a feature.

Stage 3 must define the decision point, confirm what information is available at
that point, and approve the candidate feature set before any model is trained.
