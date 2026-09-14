# Stage A1 evidence map and chain of custody

This map connects future assurance questions to identified release evidence. It
does not answer those questions or assign assurance ratings. Evidence IDs refer
to `evidence_register.csv`.

## Evidence classes

- **CANONICAL_FROZEN**: the four byte-identical artifacts explicitly designated
  as canonical for the released model, policy, holdout predictions, and holdout
  metrics.
- **SUPPORTING_FROZEN**: machine-readable evidence whose current release bytes
  support reconstruction or challenge but which is not one of the four
  canonical artifacts.
- **SUPPORTING**: documentation, source, or control definitions relevant to an
  assurance question. Their existence does not independently prove execution.
- **GENERATED_ASSURANCE**: evidence created by the assurance process. No such
  item is treated as source evidence in the initial register; the register is
  not self-hashed.

## Assurance-question navigation

### What identifies the released production model and policy?

Authoritative navigation: EV-MODEL-001, EV-POLICY-001, EV-META-001,
EV-INTEGRITY-001, EV-INTEGRITY-002.

### What evidence supports dataset provenance and semantic interpretation?

Authoritative navigation: EV-DATA-001, EV-DATA-002, EV-DATA-003. The raw source
snapshot is intentionally absent from the released Git tree; EV-DATA-001 records
its source metadata and hash. ID 573 supplies semantic correction rather than a
replacement modelling dataset.

### What preprocessing and predictor contract were released?

Authoritative navigation: EV-PREPROC-001, EV-FEATURE-001, EV-SCHEMA-001,
EV-CONTRACT-001, EV-POLICY-001.

### What fixes development/holdout and cross-validation membership?

Authoritative navigation: EV-SPLIT-001 and EV-CV-001.

### What records model-family selection before the final holdout?

Authoritative navigation: EV-SELECT-001, EV-SELECT-002, EV-SELECT-003,
EV-METHOD-001.

### What records nested tuning and the decision not to carry it forward?

Authoritative navigation: EV-TUNE-001, EV-TUNE-002, EV-TUNE-003,
EV-POLICY-001. The tuned result is preserved as evidence rather than erased.

### What records calibration selection and the calibrated architecture?

Authoritative navigation: EV-CAL-001, EV-CAL-002, EV-CAL-003,
EV-POLICY-001, EV-META-001.

### What records threshold selection and the cost policy?

Authoritative navigation: EV-THRESH-001, EV-CAL-002, EV-CAL-003,
EV-POLICY-001.

### What establishes the pre-access plan and one-time final holdout record?

Authoritative navigation: EV-HOLDOUT-003, EV-HOLDOUT-004, EV-HOLDOUT-005,
EV-HOLDOUT-001, EV-HOLDOUT-002, EV-HOLDOUT-006. Later assurance analysis may
re-perform calculations from the frozen files but may not optimize the release.

### What evidence exists for post-hoc explainability?

Authoritative navigation: EV-XAI-001 and EV-XAI-002. These records distinguish
underlying tree-margin explanations from final calibrated probabilities.

### What evidence exists for subgroup and proxy-risk review?

Authoritative navigation: EV-RAI-001, EV-RAI-002, EV-RAI-003, EV-RAI-004,
and EV-FEATURE-001. These are descriptive project-generated records and are not
fairness certification.

### What proves artifact integrity and golden inference parity controls exist?

Authoritative navigation: EV-MODEL-001, EV-META-001, EV-GOLDEN-001,
EV-INTEGRITY-001, EV-INTEGRITY-002, EV-RUNTIME-001, EV-CONTAINER-001.
Configuration and verifier source establish controls; execution claims require
corroborating run evidence.

### What defines the API contract and released runtime boundary?

Authoritative navigation: EV-SCHEMA-001, EV-CONTRACT-001, EV-API-001,
EV-RUNTIME-001, EV-META-001.

### What evidence supports container reproducibility?

Authoritative navigation: EV-CONTAINER-001, EV-CONTAINER-002,
EV-CONTAINER-003, EV-CI-001, EV-RELEASE-001. The repository contains the
configuration and verifier, while EV-RELEASE-001 narrates completed validation;
the initial register does not contain a separately signed external container
attestation.

### What records development governance and methodology?

Authoritative navigation: EV-METHOD-001, EV-GOV-001, EV-DATA-002,
EV-RAI-004. These are claims and controls to be challenged in later stages, not
assurance conclusions in A0/A1.

### What establishes release provenance?

The annotated Git tag `v1.0.0` is the primary repository custody marker and
resolves to commit `51db046543c2d95c35058469067ba9f988a6133b`. Supporting
navigation: EV-RELEASE-001, EV-INTEGRITY-002, EV-META-001. Assurance work begins
after that release on a separate branch and does not redefine the tagged tree.

## Chain of custody

1. The released repository tree is fixed by annotated tag `v1.0.0`.
2. The tag resolves to release commit
   `51db046543c2d95c35058469067ba9f988a6133b`.
3. The four canonical paths and expected raw-byte SHA-256 values are fixed in
   the evidence register and in the independent assurance verifier.
4. The verifier hashes bytes before any possible joblib deserialization and
   fails on a missing or mismatched canonical artifact.
5. Supporting release evidence is registered with its current byte hash so
   later changes or substitutions are detectable.
6. Stage A0/A1 begins on `assurance-case-v1.1.0` at the release commit.
   Assurance additions therefore occur after release on a separate branch.
   They may describe or challenge v1.0.0 but cannot change the tag or the
   released evidence chain.

## Known evidence limitations and ambiguities

- `docs/release_provenance.md` was written before tagging and still labels the
  tag as pending. Git now contains the annotated tag at the stated release
  commit. The historical file is preserved; Git is used as the stronger current
  custody fact.
- The raw UCI ID 144 snapshot is intentionally Git-ignored. Its hash and source
  metadata are recorded in EV-DATA-001, but its bytes are not part of the public
  release tree.
- The semantic audit records that official ID 573 attachments were not
  independently parsed; exact row-level equivalence was not established.
- Runtime, CI, and release validation records are project-authored. No signed
  external validation attestation is registered.
- The assessor and model developer are the same person. Structural separation
  does not create organizational independence.

These points are evidence-quality boundaries for later assurance work. They are
not resolved or scored in Stage A0/A1.
