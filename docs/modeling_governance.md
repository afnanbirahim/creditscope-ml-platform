# Stage 3 modelling governance

> **Stage 3R status:** The semantic remediation is now the authoritative baseline.
> Original Stage 3 outputs are preserved under
> `reports/stage3_legacy_pre_semantic_audit/`. See
> `docs/statlog_semantics_audit.md`.

> **Stage 4 boundary:** Random Forest and XGBoost are untuned development-only
> candidates. They use the same 17 predictive fields and the same locked folds
> as Stage 3R. Direct audit attributes remain excluded. No Stage 4 result selects
> a final model, threshold, or production system.

> **Stage 5 boundary:** Only XGBoost is tuned, using the persisted Stage 4 folds
> as outer evaluation folds and Average Precision for four-fold inner selection.
> The threshold remains 0.50, and cost is monitored rather than optimized. The
> final holdout, direct audit attributes, calibration, SHAP, weighting, and
> resampling remain outside the authorized scope.

> **Stage 6 freeze:** Stage 5 tuning is rejected and retained as negative
> evidence. The fixed untuned Stage 4 XGBoost is the finalist. Calibration and
> threshold selection use only nested development evidence, with Brier score and
> `5 * FN + FP` as the respective objectives. The theoretical `1/6` threshold is
> a reference, not a forced decision. The resulting full policy is frozen before
> any one-time final-holdout access; direct audit attributes remain excluded.

## Decision-support boundary and prediction point

CreditScope remains an educational and research decision-support prototype. The
prediction point is application-time credit-risk assessment, before a lending
decision is made. The target is `bad_credit_risk` (0 = good credit risk, 1 = bad
credit risk); it is not treated as an observed default outcome.

## Direct sensitive and demographic attributes

The predictive feature set excludes `personal_status_sex`, `age_years`, and
`foreign_worker`. These columns remain in the analysis dataset and are retained
separately, aligned by deterministic source-row identifier, for later fairness
and performance auditing.

This is a conservative responsible-AI design choice for this educational
prototype. It is not presented as a jurisdiction-specific legal requirement.

## Potential proxy variables

The baseline retains employment duration, job, housing, property, savings
status, checking-account status, residence duration, dependents, and telephone.
These fields may proxy socioeconomic or demographic circumstances. Retention in
the baseline is not an endorsement for deployment; later sensitivity and
fairness analysis must examine their effects.

## Application-time availability assumption

UCI does not fully document measurement timing. Stage 3 treats the candidate
fields as information available at application time because the source does not
identify them as post-outcome variables. This is an explicit modelling
assumption and dataset limitation, not a claim of certainty. Credit history,
existing credits at this bank, checking and savings status, other installment
plans, and other debtors or guarantors require particular scrutiny.

## Locked Stage 3R baseline scope

- One stratified 80% development / 20% final-test split uses `random_state=42`.
- Final-test labels may be counted to verify stratification, but final-test
  predictive performance is not calculated or inspected in Stage 3.
- Direct sensitive attributes never enter preprocessing or model fitting.
- Potential proxy variables remain candidates pending later analysis.
- Only `duration_months` and `credit_amount` use median imputation as inference
  robustness and standard scaling. The verified source currently has no missing
  values. Credit amount is quantitative but underwent an unknown monotonic
  transformation, so coefficients must not receive literal monetary interpretation.
- All other predictive fields use most-frequent imputation and one-hot encoding
  with unknown categories ignored. This includes corrected ordinal and discretized
  quantities because the baseline must not impose equal-step linear log odds.
- Unknown/no-account/no-property categories are substantive categories, not
  missing values. Rare categories remain intact.
- Dummy and untuned Logistic Regression baselines use identical five-fold
  stratified development-only cross-validation.
- Logistic Regression uses default `C=1.0`, no class weighting, and a higher
  `max_iter` only to provide convergence headroom.
- All classifications use the fixed 0.50 threshold. No threshold optimization,
  resampling, feature selection, or hyperparameter search occurs.
- UCI cost is oriented with rows as actual class and columns as predicted class:
  false negative (actual bad/predicted good) costs 5; false positive (actual
  good/predicted bad) costs 1.

## Frozen Stage 6 policy

- Model: fixed untuned Stage 4 `XGBClassifier`.
- Predictors: the existing 17 governed fields; no feature selection.
- Preprocessing: Stage 3R tree preprocessing.
- Calibration: sigmoid, selected from development-only OOF Brier evidence.
- Calibration implementation: `CalibratedClassifierCV` with the fixed Stage 4
  pipeline, exact persisted five-fold development splits, `ensemble=True`, and
  omitted `n_jobs` (resolved `None`). The frozen predictor averages calibrated
  probabilities from five fold-specific classifier/calibrator pairs.
- Decision threshold: 0.16, selected on the fixed development-only grid.
- Cost: `5 * false_negative + false_positive`.
- Holdout rule: the sealed 200 rows may be evaluated once in Stage 7 under
  `reports/stage6/final_holdout_protocol.md`; results cannot trigger policy edits.

This freeze is methodological governance for the educational prototype, not a
claim that the policy is legally or operationally suitable for lending.

## Stage 7 final-evaluation status

The holdout was accessed once only after the Stage 6P policy and pre-access
manifest were frozen. The exact frozen five-member sigmoid-calibrated XGBoost
ensemble and threshold 0.16 were evaluated. No modelling change, alternative
threshold, alternative candidate, or calibration comparison followed access.

Final holdout metrics are recorded under `reports/stage7/`. They must not be used
to reopen model development or turn the holdout into another selection dataset.
The predefined result was ROC-AUC 0.804643, Average Precision 0.679212,
precision 0.398496, recall 0.883333, F1 0.549223, balanced accuracy 0.655952,
specificity 0.428571, Brier 0.152338, log loss 0.475878, accuracy 0.565000,
TN 60, FP 80, FN 7, TP 53, and assumed 5:1 cost 115 (0.575000 per row).
The 5:1 error cost remains an assumed educational-prototype cost structure, not
evidence about actual lending economics or a jurisdiction-specific requirement.

## Stage 8 post-hoc audit boundary

- Stage 8 cannot reopen model development or alter the final Stage 7 result.
- The nonserialized frozen ensemble is deterministically reconstructed on the 800
  development observations for explanation only. Its five member XGBoost models
  are explained with Tree SHAP on raw tree-margin output; SHAP does not exactly
  decompose sigmoid calibration, member averaging, or threshold 0.16.
- Primary importance, local cases, subgroup diagnostics, audit-feature
  associations, and error analysis use development data or honest Stage 6 OOF
  evidence. No new holdout subgroup or explanation analysis is permitted.
- Audit attributes remain excluded from prediction. Their post-prediction join is
  permitted solely for descriptive governance analysis.
- The personal-status/sex source field remains compound historical coding and may
  not be presented as a clean gender variable. Age bands are fixed as under 25,
  25–34, 35–44, 45–54, and 55+. No subgroup-specific thresholds are permitted.
- Socioeconomic proxy potential remains documented for checking and savings
  status, employment duration, housing, property, telephone, and job. No Stage 8
  importance or audit result authorizes feature removal or selection.
- Subgroup differences do not establish fairness, absence of discrimination, or
  legal compliance. Explanations and associations are non-causal.

## Stage 9 packaging controls

- `creditscope-model-1.0.0` packages the unchanged Stage 6P policy. Only the 800
  locked development IDs participate in fitting; holdout evidence is checked by
  hash without generating or evaluating holdout predictions.
- The joblib artifact contains all five sigmoid-calibrated XGBoost pipeline
  members. Production inference cannot refit, select, calibrate, or alter them.
- Input is strictly limited to the 17 governed predictors. Audit-only and unknown
  fields are rejected. The only class boundary is probability `>= 0.16` for bad
  credit risk.
- Outputs describe good or bad credit risk and never approval, rejection,
  eligibility, or a lending decision.
- Strict loading verifies SHA-256 before joblib deserialization. Because pickle
  loading is unsafe, artifacts must come only from a trusted internal build.
- Explanation is not part of core inference and SHAP is not imported there.

## Stage 10 API controls

- The FastAPI service is an HTTP adapter for the existing Stage 9 inference
  layer. It may not train, refit, recalibrate, select features, alter parameters,
  or expose policy mutation routes.
- Startup performs strict artifact integrity checks and verifies
  `creditscope-model-1.0.0`, the exact 17 predictors, sigmoid calibration,
  `ensemble=True`, five members, and threshold 0.16 before readiness.
- The model is loaded once and treated as read-only. Health checks do not invoke
  prediction; request handlers do not reload the artifact.
- HTTP input is strict and rejects audit-only fields, unexpected fields, invalid
  source categories, numeric strings, nulls, NaN, infinity, empty batches, and
  oversized batches. A malformed batch is rejected as a whole.
- API outputs use only good/bad credit-risk terminology. There are no approval,
  rejection, eligibility, default, training, holdout, SHAP, or audit-data routes.
- CORS is disabled by default, payloads are not persisted or application-logged,
  and clients cannot provide artifact paths or upload serialized objects.
- The service has no runtime dependency on Stage 7 holdout rows, predictions, or
  metrics. Stage 10 does not reopen model development or evaluation.

## Stage 11 frontend controls

- Streamlit may communicate only with the Stage 10 service through HTTP. It may
  not load joblib, import estimator prediction functions, reproduce the frozen
  threshold, or access training/holdout evidence.
- The form is restricted to the 17 governed predictors. Audit-only attributes,
  threshold controls, feature-selection controls, and model controls are absent.
- Canonical category codes are submitted unchanged; human-readable text comes
  from the documented repository source semantics. FastAPI remains the final
  validation authority.
- The UI obtains readiness, model version, family, calibration, threshold,
  predictor count, and target semantics from API responses. It cannot override
  those values.
- Prediction payloads are not persisted, logged, or sent anywhere except the
  configured CreditScope API. The frontend does not fall back to local model
  loading when the API is unavailable.
- UI language remains good/bad credit risk and explicitly rejects lending
  approval, rejection, eligibility, financial-advice, causal, or fairness claims.

## Stage 12 infrastructure controls

- Containerization and CI may package and verify the existing services but may
  not retrain, rebuild, reserialize, tune, recalibrate, rethreshold, or rescore
  the frozen policy.
- The API runtime may deserialize only the trusted bundled Stage 9 artifact after
  strict hash verification. No user-controlled pickle path or upload exists.
- The UI runtime contains no model artifact, joblib, XGBoost, SHAP, inference
  module, training data, or holdout evidence; it communicates only over HTTP.
- Raw, development, holdout, subgroup-audit, and performance-evidence files are
  excluded from runtime image contexts. Model-policy values remain immutable.
- CI and local container checks are engineering parity tests using committed
  development-derived golden fixtures. They are not new performance evaluation.
- No container registry publication, cloud deployment, credentials, privileged
  mode, Docker socket mount, or Stage 13 work is authorized.

## Stage 8 post-evaluation governance

- Stage 7 performance and the frozen policy are permanent; explainability or
  subgroup findings cannot reopen model selection.
- SHAP is restricted to the five fitted XGBoost members in the reconstructed
  frozen ensemble architecture and to development observations. It explains raw
  tree scores, not calibrated ensemble probabilities or causal effects.
- `personal_status_sex`, `age_years`, and `foreign_worker` remain audit-only.
  Their subgroup metrics use honest development OOF evidence and may not support
  subgroup-specific thresholds.
- The personal-status/sex field is a compound historical code. Sex cannot be
  cleanly isolated and modern gender concepts cannot be inferred.
- Checking and savings status, employment duration, housing, property, telephone,
  and job remain on the proxy-risk register. Their inclusion is not evidence that
  the system is fair or suitable for real lending.
- Fairness, absence of discrimination, and transportability cannot be established
  from this small historical dataset. Any real use would require contemporary
  representative data, jurisdiction-specific review, human oversight, impact
  assessment, monitoring, and recourse.
