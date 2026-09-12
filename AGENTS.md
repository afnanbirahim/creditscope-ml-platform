# CreditScope working rules

- Work only on the stage explicitly authorized by the user; do not begin the
  next stage without permission.
- Treat this repository as an educational/research credit-risk decision-support
  prototype, never as an automated lending approval system.
- Use UCI Statlog German Credit Data dataset 144 as the current data source and
  UCI South German Credit dataset 573 as the corrected semantic reference. Do
  not silently replace datasets or invent data, meanings, findings, or metrics.
- Use “good credit risk,” “bad credit risk,” and “bad-credit-risk indicator.” Do
  not equate the target with an observed loan default.
- Preserve source target values 1 = good and 2 = bad. The internal binary target
  is `bad_credit_risk`: 0 = good, 1 = bad; class 1 is positive.
- Keep the raw source snapshot unchanged and ignored by Git. Validate it against
  the Stage 1 recorded SHA-256 before offline reuse.
- Put reusable logic under `src/creditscope`; notebooks should orchestrate and
  explain that logic, not contain the only implementation.
- Use `random_state=42` wherever supported. Execute code, tests, Ruff, established
  compilation checks, and relevant notebooks before reporting results.
- Never use the final test set for feature, preprocessing, threshold,
  hyperparameter, or model selection. Report explicitly when it remains untouched.
- Stage 3 locks the split in `reports/split_manifest.csv`: 80% development and
  20% final test, stratified with `random_state=42`.
- Stage 3R is the authoritative baseline after semantic remediation. Only
  `duration_months` and `credit_amount` use numeric scaling; the other 15
  predictive variables use one-hot encoding. Original Stage 3 results are
  preserved under `reports/stage3_legacy_pre_semantic_audit/`.
- Exclude `personal_status_sex`, `age_years`, and `foreign_worker` from predictive
  features, but retain them separately for audit analysis. Proxy-risk variables
  remain under review; consult `docs/modeling_governance.md`.
- Preserve UCI's asymmetric cost: a bad risk predicted good costs 5 and a good
  risk predicted bad costs 1. State confusion-matrix orientation when reporting it.
- Do not commit secrets, environments, caches, downloaded raw data, model
  artifacts, or unnecessary generated files. Preserve unrelated user changes.
- Document methodology, limitations, checks, and unresolved decisions at each
  stage. Exploratory and model associations are not causal conclusions.
