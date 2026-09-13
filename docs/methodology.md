# Methodology decisions

## Stage 1: source and integrity

- **Canonical source:** UCI Machine Learning Repository dataset ID 144, fetched
  through the repository's documented `ucimlrepo` client.
- **Representation:** the original mixed categorical/integer feature table is
  retained. The alternate all-numeric file is not selected implicitly.
- **Local data policy:** a raw CSV snapshot may be generated for reproducible
  inspection, but is ignored by Git. Its SHA-256 digest is recorded in the small
  verification report.
- **Verification contract:** the fetch must identify dataset ID 144 and produce
  1,000 rows, 20 features, one target, no missing values, and target labels 1 and
  2. These are source-integrity checks based on UCI's documentation, not model
  results.
- **Target semantics:** UCI documents class 1 as “good” and class 2 as “bad.” We
  preserve the source encoding in Stage 1 and defer any remapping to a later,
  explicitly documented preprocessing stage.
- **Asymmetric error costs:** UCI supplies a cost matrix in which predicting a
  bad risk as good is five times as costly as predicting a good risk as bad.
  This must inform later metric and threshold choices; it does not justify a
  business deployment claim.
- **Responsible-use boundary:** the data are old, small, geographically specific,
  and include demographic attributes. Any later analysis must examine fairness,
  limitations, and possible proxy effects. CreditScope remains an educational
  decision-support prototype.
- **Holdout policy:** no split is made in Stage 1. A future final test set must be
  created once and kept out of model selection.

## Stage 2: semantics and exploratory analysis

- Clean feature names are a one-to-one presentation layer. Raw UCI fields and
  values remain unchanged in the ignored source snapshot.
- The source target is retained as `credit_risk_class` (1 = good credit risk,
  2 = bad credit risk). `bad_credit_risk` is an analysis-only binary view where
  0 = good and 1 = bad. Neither field denotes an observed default event.
- Official categorical codes are validated against UCI's `german.doc`, including
  documented codes that do not occur in the 1,000-row source table.
- “Rare” is defined before inspection as an observed category containing fewer
  than 2% of records. This flag supports later encoding decisions; categories are
  not combined or removed in Stage 2.
- EDA reports observed distributions and bad-credit-risk rates. It does not test
  causal hypotheses or justify feature selection.
- Mechanical plausibility checks cover missingness, duplicates, documented code
  domains, and nonpositive duration, amount, or age. UCI does not provide full
  permissible numerical ranges, so passing these checks is not a comprehensive
  real-world validity assessment.
- Sensitive/proxy and timing decisions are deliberately deferred until Stage 3,
  before modelling or creation of a final holdout split.

## Stage 3: holdout isolation and baselines

- The prediction point is application time, before a lending decision.
- One 80/20 stratified split is locked with `random_state=42`. Only the 80%
  development partition is used for preprocessing fit, model fit, and five-fold
  cross-validation; the 20% final-test partition is not evaluated.
- `personal_status_sex`, `age_years`, and `foreign_worker` are excluded from
  prediction and retained in aligned audit tables. See
  `docs/modeling_governance.md` for the rationale and proxy review.
- Numeric, ordinal, and nominal transformations are fitted inside a
  `ColumnTransformer`/`Pipeline`. Special source categories such as no-account
  or unknown/no-property remain substantive categories.
- Baselines are a prior `DummyClassifier` and one untuned, unweighted Logistic
  Regression. Five-fold stratified CV uses shuffling and `random_state=42`.
- Bad credit risk (1) is positive. The threshold is fixed at 0.50. UCI cost is
  `5 * false_negative + 1 * false_positive`, where a false negative is a bad
  risk predicted as good.
- Stage 3 includes OOF discrimination, classification, cost, and calibration
  diagnostics plus development-fit coefficients. It includes no model selection,
  threshold optimization, resampling, probability calibration, or causal claim.

## Stage 3A: corrected-semantics audit

The complete audit is documented in `docs/statlog_semantics_audit.md`, with the
20-feature machine-readable comparison in
`reports/stage3a_semantics_comparison.csv`.

- UCI South German Credit dataset 573 is the corrected/background-enhanced
  semantic reference for the same German credit data. UCI warns that the older
  Statlog coding information has severe errors.
- No dataset, split membership, model, prediction, or threshold changed in this
  audit.
- Corrected types show that installment rate, residence duration, and number of
  credits are ordinal discretizations rather than ordinary numeric measurements;
  dependents is a binary discretization.
- Employment duration and job have genuine order, but Stage 3's ordinal encoding
  imposes equal-step linearity in Logistic Regression. One-hot encoding is the
  recommended conservative Logistic Regression baseline representation.
- Stage 3 should be regenerated in an explicitly authorized remediation step
  before Stage 4. Preserve the locked memberships and do not use final-test
  performance.
- Migration to dataset 573 is recommended only after official-file retrieval and
  row/value equivalence checks. Do not silently replace dataset 144.

## Stage 3R: semantic remediation

- Dataset 144 remains the modelling data; dataset 573 supplies corrected semantic
  guidance. No data migration occurred.
- The original Stage 3 outputs are preserved under
  `reports/stage3_legacy_pre_semantic_audit/` and are superseded for future model
  comparison by `reports/stage3_remediated/`.
- The locked split and five-fold CV memberships remain unchanged.
- Only `duration_months` and `credit_amount` use median imputation and standard
  scaling. Credit amount remains quantitative but its unknown monotonic
  transformation precludes literal monetary-effect interpretation.
- The other 15 predictive fields use most-frequent imputation and one-hot
  encoding. This avoids unsupported equal-step linear-log-odds assumptions for
  ordinal and discretized fields.
- The model set, default hyperparameters, class weighting, threshold 0.50, UCI
  5:1 cost orientation, and development-only evaluation design are unchanged.
- No final-test predictions or performance estimates were produced, no threshold
  was optimized, and no model was selected as a production model.

## Stage 4: untuned tree-model comparison

- Stage 3R reproducibility is reconciled under the typed policy documented in
  `docs/stage3r_reproducibility_reconciliation.md`: structural and discrete
  evidence must match exactly, while reproduced floating metrics use `rtol=0`
  and `atol=1e-10`. The tolerance is not a general allowance for methodological
  changes.
- Stage 3R Logistic Regression remains unchanged as the reference candidate.
- One Random Forest and one XGBoost configuration are fixed before comparative
  inspection. No parameter search, manual iteration, early stopping, class
  weighting, sample weighting, or resampling is permitted.
- All three candidates use the same precomputed five stratified development
  folds with shuffling and `random_state=42`. Every development row must receive
  exactly one OOF prediction.
- Tree preprocessing passes `duration_months` and `credit_amount` through median
  imputation without scaling. The 15 categorical/discretized predictors remain
  one-hot encoded with unknown-category handling inside each training fold.
- Evaluation remains development-only. The threshold stays 0.50 and UCI cost is
  `5 * false_negative + 1 * false_positive`, with bad credit risk as class 1.
- Brier and reliability diagnostics assess probability quality; probabilities
  are not calibrated. Intrinsic feature importance is diagnostic only and is not
  used for feature selection.
- Fold training-versus-validation comparisons diagnose obvious overfitting but
  do not authorize parameter changes in Stage 4.
- The locked final holdout is not scored, and no final model is selected. Stage 5
  must decide whether a controlled tuning protocol is warranted.
- Stage 4 persists development-only fold membership and aligned row-level OOF
  probabilities/predictions for all three models, with SHA-256 hashes recorded
  in run metadata. No final-holdout predictions are included.

## Stage 5: controlled XGBoost tuning

- XGBoost alone is tuned. Stage 4 did not justify a Random Forest search because
  its untuned candidate had lower bad-risk recall, higher 5:1 cost, and a larger
  train-versus-validation gap.
- The persisted Stage 4 five-fold membership is reused as the outer evaluation
  structure. Each outer-validation fold remains isolated while a four-fold
  stratified `RandomizedSearchCV` runs on its 640 outer-training observations.
- Every inner search uses the same predefined space, 40 draws,
  `random_state=42`, and Average Precision as the selection objective. ROC-AUC,
  Brier score, and recall at 0.50 are recorded as diagnostics.
- Preprocessing remains inside the pipeline. XGBoost uses one thread per fit and
  search-level parallelism to avoid nested thread-pool oversubscription.
- The five outer-validation predictions form the honest nested OOF estimate of
  the tuning procedure. Threshold 0.50 and `5 * FN + FP` are evaluated but not
  optimized.
- After nested evaluation, a separate 40-draw search uses all 800 development
  rows and the locked five-fold structure to select configuration metadata. Its
  best CV score is selection evidence, not an unbiased performance estimate.
  `refit=False` prevents creation of a final all-development estimator.
- Stage 5 performs no feature selection, weighting, resampling, probability
  calibration, SHAP analysis, threshold search, or final-holdout evaluation.

## Stage 6: development-only calibration and threshold governance

- The nested-tuned Stage 5 candidate is rejected because its honest nested OOF
  estimate worsened Average Precision, recall, F1, balanced accuracy, Brier score,
  and 5:1 cost. Those Stage 5 results remain preserved as negative evidence.
- The only Stage 6 model is the fixed untuned Stage 4 XGBoost with its original
  hyperparameters, 17 governed predictors, and Stage 3R tree preprocessing.
- The exact persisted Stage 4 folds form the outer evaluation layer. Within each
  outer-training partition, four-fold OOF evidence compares no calibration,
  sigmoid, and isotonic strategies. Calibrated estimators fit calibration only
  within their training data.
- Calibration selection minimizes inner OOF Brier score. Differences no greater
  than `1e-4` invoke the fixed parsimony order: none, sigmoid, isotonic.
- The selected strategy's inner OOF probabilities support a fixed threshold grid
  from 0.05 through 0.50 by 0.01. Selection minimizes `5 * FN + FP`, then prefers
  fewer false negatives, higher balanced accuracy, and proximity to `1/6`.
- `1 / (1 + 5) = 1/6` is the theoretical decision threshold only when predicted
  probabilities are perfectly calibrated and the stated error costs are the
  complete decision costs. It is a reference, not a forced threshold.
- Aggregate outer-validation predictions provide the honest development estimate
  of the complete calibration-choice and threshold-selection procedure.
- Only after nested evaluation, all 800 development rows and the persisted folds
  select one calibration method and threshold. These are selection evidence, not
  an unbiased performance estimate.
- Stage 6 freezes the model, features, preprocessing, calibration method,
  threshold, and cost function before holdout access. Further model-development
  changes would invalidate the planned one-time Stage 7 evaluation.
- Source inspection confirms every calibrated Stage 6 path explicitly used
  `CalibratedClassifierCV(estimator=<fixed Stage 4 XGBoost pipeline>,
  method=<sigmoid or isotonic>, cv=<training-only splits>, ensemble=True)`.
  `n_jobs` was omitted and resolves to `None` in scikit-learn 1.4.2. The frozen
  sigmoid policy preserves this architecture: Stage 7 fits five fold-specific
  classifier/calibrator pairs using the exact persisted Stage 4 development
  folds and averages their calibrated probabilities. It does not fit one
  XGBoost classifier on all 800 development rows.
- No additional hyperparameter tuning, feature selection, SHAP analysis, holdout
  inspection, holdout scoring, or Stage 7 metric calculation occurs.

## Stage 7: one-time final holdout evaluation

- Before access, the exact Python/scientific environment, eight locked evidence
  hashes, split membership, frozen policy, source implementation, complete tests,
  Ruff, compilation, and dependency consistency were verified. The immutable
  pre-access manifest was then written and hashed.
- The final predictor is the frozen sigmoid `CalibratedClassifierCV` ensemble:
  five fold-specific Stage 3R-preprocessed, fixed Stage 4 XGBoost classifiers and
  calibrators using the exact persisted development folds. Their calibrated
  probabilities are averaged; no single XGBoost was fit on all 800 rows.
- The sealed 200-row holdout was accessed once. Exactly one probability was
  generated per row and threshold 0.16 was applied without comparison to any
  alternative threshold, calibration method, or model.
- Final locked-holdout results were ROC-AUC 0.804643, Average Precision 0.679212,
  precision 0.398496, recall 0.883333, F1 0.549223, balanced accuracy 0.655952,
  specificity 0.428571, Brier score 0.152338, log loss 0.475878, and accuracy
  0.565000. The confusion counts were TN 60, FP 80, FN 7, and TP 53; assumed
  5:1 total cost was 115 (average 0.575000).
- The results were documented without changing model, features, preprocessing,
  calibration, or threshold. No confidence intervals, SHAP analysis, deployment,
  or post-hoc development occurred.
- The Stage 6 honest nested estimate remains the development reference. Final
  results are broadly consistent with it: recall and policy tradeoffs are close,
  discrimination and probability quality are somewhat stronger on the holdout,
  while balanced accuracy and average cost are modestly worse. With only 200
  observations, ordinary sampling variation is expected.
- The `5 * FN + FP` cost is an educational/research modelling assumption and is
  not presented as real-world lending economics.

## Stage 8: post-hoc explainability and responsible-AI audit

- The model-development lifecycle remains closed. Stage 8 verifies the locked
  Stage 6P/7 hashes, does not generate new holdout predictions, and makes no
  model, feature, calibration, or threshold change.
- Because the fitted Stage 7 Python object was intentionally not serialized,
  Stage 8 deterministically refits the exact frozen `CalibratedClassifierCV`
  architecture on the same 800 development rows solely for explanation. Its five
  fitted base pipelines—not a separate sixth XGBoost—are inspected.
- `shap.TreeExplainer(model_output="raw")` explains each member's underlying
  XGBoost margin. It does not exactly decompose sigmoid calibration, arithmetic
  averaging across members, or the final threshold decision.
- Encoded features are aligned by canonical preprocessing output name. A level
  absent from member training would be reported as absent rather than assigned a
  zero effect. In this run, all five members contained the same 64 encoded names.
- Source-feature importance sums absolute SHAP across all encoded levels per
  observation and then averages. Each member's encoded and source-feature totals
  reconcile within absolute tolerance `1e-10`.
- Local cases follow a probability-only rule fixed before SHAP inspection. The
  Stage 6 nested-policy OOF probabilities are re-thresholded at the frozen 0.16;
  they support development audit, not exact final-ensemble probability
  decomposition.
- Age bands were fixed before subgroup calculation: under 25, 25–34, 35–44,
  45–54, and 55+. Undefined subgroup metrics remain NA.
- Audit attributes are joined after prediction only. Limited proxy association
  summaries use just the five leading frozen-model source features. Subgroup,
  proxy, and error diagnostics are descriptive and post-freeze; none supports
  fairness, causality, legal compliance, or model optimization claims.

## Stage 9: frozen-model packaging and inference

- Production reconstruction verifies locked hashes and fits the exact frozen
  five-member sigmoid-calibrated XGBoost ensemble using only the 800 development
  IDs. This is packaging of a closed policy, not model selection or evaluation.
- The complete estimator is serialized with joblib as
  `creditscope-model-1.0.0`. Its manifest records provenance, environment,
  configuration, feature boundary, audit exclusions, calibration, threshold,
  artifact hash, and development-only training population.
- Golden fixtures use a fixed development-only rule: minimum fitted probability,
  closest to 0.16, closest to predefined moderate reference 0.35, maximum
  probability, and maximum minimum categorical Hamming distance from the first
  four selections. They test serialization rather than model performance.
- Reloaded fixture probabilities must agree within `1e-12`; classes and labels
  must be exact. Joblib hashes are version-sensitive and loading untrusted pickle
  artifacts is unsafe.
- Inference strictly validates 17 raw predictors and rejects audit-only,
  unexpected, missing, invalid, NaN, and infinite inputs. The sole threshold rule
  is bad credit risk when probability is greater than or equal to 0.16.
- Core inference does not import SHAP, refit any component, write artifacts,
  contact external services, or access holdout data.

## Stage 10: FastAPI inference service

- Stage 10 is a transport layer over the authoritative Stage 9 inference module;
  it does not reconstruct validation, prediction, or threshold logic.
- `creditscope-api-1.0.0` loads the canonical `creditscope-model-1.0.0` joblib
  artifact once during application startup with strict SHA-256 verification.
  Startup additionally checks the model version, 17-predictor boundary, sigmoid
  calibration, five-member ensemble, and threshold 0.16.
- Pydantic provides a strict HTTP schema, while Stage 9 remains the semantic
  validation and inference authority. Single and ordered batch inference return
  only calibrated bad-risk probability and frozen good/bad credit-risk labels.
- The initial batch maximum is 100; empty, oversized, partially invalid, or
  semantically invalid requests are rejected rather than truncated or partially
  scored. CORS is disabled by default.
- The process treats the startup-loaded estimator as read-only. Request handlers
  never fit, mutate, write payloads, import SHAP, access holdout data, or expose
  training/model-update operations.
- The API requires only the Stage 9 model, manifest/schema, and inference code.
  Stage 7 predictions and metrics remain immutable evidence and are not runtime
  dependencies.

## Stage 11: Streamlit frontend

- The frontend is an HTTP client of the Stage 10 FastAPI service. It does not
  deserialize or import the estimator, reproduce validation or threshold logic,
  or access training/holdout evidence.
- A reusable HTTPX client handles health, model metadata, single prediction, and
  optional batch transport. The initial UI exposes only single-record inference.
- The page verifies service readiness and the API-reported 17-feature order,
  then presents documented source-category descriptions while submitting the
  unchanged canonical codes.
- Results use the API's probability, class, label, threshold, and model version.
  Percentage formatting is presentational and does not change the underlying
  response or calculate a local decision.
- Network failures and HTTP 400/413/422/500/503 responses become sanitized
  messages. Payloads are not persisted or application-logged.
- The limitations panel preserves the historical-data, proxy-risk, fairness,
  non-causal, non-lending-decision, and assumed-5:1-cost cautions. No live SHAP,
  batch upload, deployment, Docker, or CI/CD work occurs.

## Stage 12: containerization and CI

- Stage 12 is infrastructure-only. The API container loads the unchanged
  canonical Stage 9 joblib artifact; neither image fits, rebuilds, recalibrates,
  tunes, or evaluates a model.
- FastAPI and Streamlit run as separate non-root services. Streamlit reaches the
  API over a private user-defined Compose bridge and contains no model artifact
  or direct inference dependency. Published host ports remain loopback-only.
- The Docker context is allow-listed. Raw, development, holdout, audit, report,
  notebook, environment, cache, and Git content is excluded from runtime images.
- Linux verification must pass the model hash, manifest, architecture, 17-field
  contract, and development-only golden fixtures at `1e-12` probability
  tolerance. A cross-platform failure stops validation; it does not authorize a
  replacement Linux model.
- CI validates quality, frozen evidence, inference, API, UI, Docker builds, and
  Compose parity. It has no secrets, publishing, deployment, holdout analysis,
  or model-development path.
- Docker is unavailable inside the Codex sandbox. Runtime and Compose validation
  was therefore executed externally and then confirmed by the green hosted Linux
  workflow before Stage 13.

## Stage 13: portfolio release candidate

- Model development remains closed. Stage 13 changes presentation, navigation,
  public-repository hygiene, release documentation, and reproducibility guidance
  only.
- Public claims trace to existing frozen artifacts. Stage 7 rows are not rescored
  and no metric, SHAP, subgroup, threshold, calibration, or model analysis is
  added.
- Repository release `v1.0.0`, API version, and model version are separate
  concepts. No release tag or GitHub Release is created in this stage.
- A fresh-clone rehearsal, screenshots, final release commit, and final hosted CI
  run remain manual release gates.
