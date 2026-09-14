# Stage A0 assurance scope and independence statement

## Assessed object

The assessed object is the frozen CreditScope v1.0.0 release identified by the
annotated Git tag `v1.0.0` and commit
`51db046543c2d95c35058469067ba9f988a6133b`. Its relevant components are:

- `creditscope-model-1.0.0`, the serialized calibrated model artifact;
- the frozen model policy, including Stage 3R preprocessing, 17 predictors,
  fixed untuned XGBoost, sigmoid calibration, five calibrated members,
  `ensemble=True`, and threshold `0.16`;
- target semantics `0 = good credit risk` and `1 = bad credit risk`;
- the final holdout prediction and metric evidence created by the released
  one-time evaluation; and
- the input contract, artifact verifier, API metadata, golden fixtures,
  container controls, and CI controls relevant to runtime identity.

This definition identifies what will be assessed. It does not conclude that
the object is suitable, effective, fair, compliant, or independently
validated.

## Assurance objectives and current-stage limit

The completed assurance case may eventually assess intended use, model
methodology, data and provenance, performance, calibration, threshold policy,
explainability, fairness and proxy risk, failure modes, human oversight,
monitoring, model risk, controls, and a final assurance opinion.

Stage A0/A1 only defines the assessment boundary and establishes the evidence
register, evidence map, chain of custody, and integrity verification. It makes
no substantive conclusion or score in the future assurance domains listed
above.

## Independence limitation

CreditScope AI Assurance Case is a post-development technical assurance
assessment of the frozen CreditScope v1.0.0 portfolio model. The assessment is
structurally separated from model development but is not organizationally
independent because the model developer and assessor are the same person.

Accordingly, this work may be described as independent-style re-performance,
model challenge, or evidence-based technical assurance. It must not be
represented as an external independent audit.

## Non-modification rule

Assurance work must not:

- retrain or tune a model;
- recalibrate the model;
- change features, preprocessing, target semantics, or the 17-field contract;
- change or replace threshold `0.16`;
- regenerate or modify final holdout predictions;
- use holdout or assurance findings to optimize or redesign the released
  policy; or
- alter the bytes of canonical frozen evidence.

Any future recommended control or remediation must be prospective and must not
be presented as part of the already-released v1.0.0 result.

## Holdout governance

The final holdout was accessed during the released Stage 7 one-time evaluation.
Future assurance stages may examine the already-frozen holdout predictions and
may independently re-perform calculations from that evidence. Such work is
diagnostic assurance or model challenge only. It cannot be used for post-holdout
feature engineering, tuning, calibration, threshold selection, policy
replacement, or any other redesign of v1.0.0.

## Assurance use

The assurance case is intended to organize evidence, test the integrity of the
released technical record, challenge claims and assumptions, surface evidence
gaps, and support transparent portfolio governance. It does not authorize the
system for a real lending workflow and does not replace qualified legal,
compliance, risk, ethics, security, or domain review.

## Limitations

- This is not a regulatory audit or legal/compliance certification.
- The assessment is not organizationally independent.
- No new external validation is performed in Stage A0/A1.
- No conclusion about real-world lending suitability is made in Stage A0/A1.
- The release evidence was produced within the same portfolio project; source
  independence and evidential weight must be considered in later stages.

## Chain-of-custody boundary

The annotated tag `v1.0.0` resolves to release commit
`51db046543c2d95c35058469067ba9f988a6133b`. Stage A0/A1 begins on branch
`assurance-case-v1.1.0` with that release commit as its starting HEAD. Assurance
work therefore follows the release on a separate branch. It may add assurance
records and verification controls, but it does not move the tag, redefine the
tagged tree, or redefine v1.0.0. The four canonical files and their expected
SHA-256 values are enumerated in `evidence_register.csv` and independently
checked by `scripts/verify_assurance_evidence.py`.
