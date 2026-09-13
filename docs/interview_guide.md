# CreditScope interview guide

## 30-second explanation

CreditScope is an educational credit-risk decision-support project that shows a
governed ML lifecycle end to end. I verified and semantically audited UCI German
Credit data, isolated one final holdout, compared transparent and nonlinear
models, rejected tuning when nested CV did not improve honest evidence, selected
calibration and a cost-sensitive threshold using development data only, froze the
policy, and evaluated the holdout once. I then packaged the exact five-member
calibrated ensemble behind strict inference, FastAPI, Streamlit, Docker Compose,
golden tests, explainability, responsible-AI documentation, and Linux CI.

## Two-minute explanation

The project deliberately treats reproducibility and governance as part of model
quality. UCI ID 144 supplies the 1,000 modelling records; corrected ID 573
documentation clarified problematic feature semantics. Three compound or
demographic variables are audit-only, while 17 governed predictors enter the
pipeline. A fixed 80/20 split sealed 200 records before model selection.

On the 800-row development set, Logistic Regression, Random Forest, and XGBoost
used identical folds. Fixed untuned XGBoost was strongest overall. A controlled
nested search subsequently worsened honest Average Precision and key policy
metrics, so I retained the untuned model rather than reporting the optimistic
inner-search winner. Nested development evidence then selected sigmoid
calibration and threshold 0.16 under an explicit `5 * FN + FP` demonstration
cost. The frozen policy achieved holdout ROC-AUC 0.804643 and recall 0.883333,
with specificity 0.428571—an intentional trade-off under that assumed cost.

The deployed artifact is not one estimator trained on all rows. It is a
five-member `CalibratedClassifierCV(ensemble=True)` whose probabilities are
averaged. The API verifies the artifact hash and strict 17-field contract; the
UI calls the API only. Golden fixtures proved identical semantics after
serialization and after moving the Windows artifact into Linux containers.

## Five-minute technical walkthrough

1. **Data and semantics:** official UCI retrieval, raw SHA-256, explicit target
   semantics, and correction of misleading legacy code definitions.
2. **Governance:** audit-only attributes, proxy-risk register, application-time
   assumptions, fixed split, and deterministic fold membership.
3. **Selection:** dummy/LR baseline, paired LR/RF/XGB comparison, then nested CV
   for XGBoost tuning. The negative tuning result is preserved rather than hidden.
4. **Policy:** nested calibration choice by Brier score and threshold choice by
   `5 * FN + FP`, with all choices confined to development data.
5. **Evaluation:** policy frozen first; 200-row holdout scored once; no post-hoc
   policy change or confidence-interval invention.
6. **Explanation and audit:** Tree SHAP on each underlying XGBoost raw margin,
   original-feature aggregation, development-OOF subgroup diagnostics, and
   explicit proxy and transportability limitations.
7. **Engineering:** trusted joblib artifact, manifest/schema, golden fixtures,
   strict API, HTTP-only UI, separate non-root containers, and CI parity on Linux.

## Likely questions and concise answers

### What problem does CreditScope solve?

It demonstrates how to produce a governed good/bad credit-risk classification
and calibrated probability. It does not approve loans or determine eligibility.

### Why use the German Credit dataset, and what are its limitations?

It is a compact public benchmark with mixed feature types and an explicit error
cost, making the full lifecycle reproducible. It has only 1,000 historical West
German records, unclear timing for some variables, oversampling, transformed
credit amount, compound demographic coding, and poor modern transportability.

### Why say good/bad credit risk rather than default?

That is the official target meaning. The source does not establish that the
label is an observed loan-default event, so renaming it would overclaim.

### Why exclude some variables?

`personal_status_sex`, `age_years`, and `foreign_worker` were conservatively
excluded from prediction and retained for audit. Exclusion is not a fairness
guarantee because remaining socioeconomic fields can act as proxies.

### Why XGBoost?

The fixed XGBoost candidate gave the strongest development discrimination among
the three tested families while accommodating nonlinearities and interactions.
It remained a candidate only until the later policy freeze.

### Why retain the untuned XGBoost? Why did nested tuning not win?

The nested-tuned procedure worsened honest OOF Average Precision, recall, F1,
balanced accuracy, Brier score, and 5:1 cost. Small data and search variance can
make added tuning complexity unhelpful. Keeping the simpler fixed candidate was
the evidence-based choice.

### Why nested CV?

Hyperparameters chosen and evaluated on ordinary CV folds produce optimistic
selection bias. Nested CV keeps each outer validation fold outside its inner
search and estimates the complete tuning procedure honestly.

### Why calibration, sigmoid, and not isotonic?

Policy decisions use probabilities, so probability quality matters. Development
OOF Brier score selected sigmoid under a predefined parsimony rule. Isotonic is
more flexible and can be unstable on small calibration samples.

### Why threshold 0.16, high recall, and modest specificity?

The project assigns cost 5 to a false negative and 1 to a false positive. The
development-selected threshold therefore prioritizes detecting bad-risk cases.
It produced holdout recall 0.883333 and specificity 0.428571. This is an explicit
educational trade-off, not a universal business optimum.

### What is the 5:1 cost assumption?

It follows the source benchmark's asymmetric structure and is used as a
demonstration governance rule. It is not empirically calibrated lending economics.

### How did you avoid holdout leakage? What does one-time holdout mean?

The 200 IDs were locked before model selection. Features, hyperparameters,
calibration, and threshold used only development data. Stage 7 then generated one
set of probabilities for one frozen policy; the result did not trigger revisions.

### Why five calibrated members?

The frozen `CalibratedClassifierCV` uses the exact five development folds with
`ensemble=True`. Each member fits on four folds and has its own sigmoid
calibrator; prediction averages their calibrated probabilities.

### What does SHAP explain—and not explain?

Tree SHAP explains each underlying XGBoost member's raw margin. It does not
directly decompose sigmoid calibration, averaging across five members, or the
final calibrated probability and threshold decision. It is associative, not causal.

### Why separate UI from API?

It keeps model loading, validation, and policy logic in one authority. Streamlit
cannot deserialize the model or apply a divergent threshold; it is only an HTTP
presentation client.

### Why Docker, golden fixtures, and frozen hashes?

Docker tests a reproducible Linux runtime. Golden fixtures catch semantic or
numerical drift through packaging layers. Hashes make changes to model, policy,
split, predictions, and metrics detectable rather than relying on filenames.

### What did cross-platform validation prove?

The exact Windows-created artifact loaded on Linux, retained five calibrated
members and the 17-field contract, and reproduced development-derived golden
predictions within `1e-12`. It does not prove business validity.

### What did CI catch?

First, Linux checkout exposed byte-normalization drift in frozen policy evidence;
canonical bytes were then preserved explicitly with `.gitattributes`. Later,
GitHub Linux Compose exposed a host-port behavior difference from Docker Desktop
when the bridge was marked `internal`. Removing only that network flag restored
host smoke portability while preserving loopback ports and model semantics.

### What would have to change for real lending?

Contemporary representative data, external and temporal validation,
jurisdiction-specific legal review, impact and fairness assessment, authenticated
and monitored infrastructure, business-calibrated costs, human review and
recourse, privacy controls, security testing, and ongoing drift governance.

## Three-to-five-minute demo

1. **README:** say, “The key point is the governed evidence chain, not merely an
   XGBoost score.”
2. **Architecture:** show `docs/architecture.md`; explain the UI/API/model boundary.
3. **Streamlit:** open the clean form; emphasize the 17 governed inputs and absent
   audit-only fields.
4. **Prediction:** submit `examples/api_single_request.json` values; explain that
   probability and thresholded class are distinct and neither is a lending decision.
5. **API:** show `/docs` and `/model-info`; point out version, threshold, member
   count, and artifact hash rather than internal estimator objects.
6. **Responsible AI:** open the model card; acknowledge proxy risks, small subgroup
   counts, and the inability to establish fairness.
7. **Docker and CI:** show separate services and the green workflow; mention the
   two portability bugs that CI revealed without altering inference semantics.
8. **Frozen evidence:** finish with the release provenance hashes and one-time
   holdout boundary.

## Manual screenshot checkpoint

Use only the committed development-derived sample input—never a holdout row.

1. Start FastAPI and Streamlit using the README commands.
2. Capture the untouched form and save it as
   `docs/assets/creditscope-input.png`.
3. Enter the values from `examples/api_single_request.json`, submit once, capture
   the result panel, and save `docs/assets/creditscope-result.png`.
4. Optionally open `http://127.0.0.1:8000/docs`, capture the route overview, and
   save `docs/assets/creditscope-api-docs.png`.
5. Check that no local paths, browser profile details, terminal windows, personal
   data, or holdout evidence appear in the images.

Screenshots remain manual release assets; no placeholder or fabricated image is
part of the repository.

