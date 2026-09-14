# CreditScope AI Assurance Case

CreditScope AI Assurance Case is a post-development technical assurance
assessment of the frozen CreditScope v1.0.0 portfolio model. The assessment is
structurally separated from model development but is not organizationally
independent because the model developer and assessor are the same person.

## Assessed release

The assessed object is the repository release identified by annotated tag
`v1.0.0` and release commit
`51db046543c2d95c35058469067ba9f988a6133b`, including
`creditscope-model-1.0.0`, its frozen decision policy, the one-time final
holdout evidence, and relevant runtime integrity controls. CreditScope is an
educational/research credit-risk decision-support prototype, not a lending
approval system.

## Separation from model development

Model development selected and evaluated the released policy. Assurance work
begins after that release and may inspect, hash, re-perform calculations from,
and challenge frozen evidence. It must not retrain, tune, recalibrate, change
features or preprocessing, change the threshold, regenerate holdout
predictions, or use assurance findings to redesign v1.0.0. Assurance findings
do not change the released model.

## Assurance stages

- A0 defines scope, boundaries, the assessed object, permitted assurance use,
  and the independence limitation.
- A1 establishes the evidence register, evidence map, chain of custody, and
  byte-level integrity controls.
- Later stages may address intended use, methodology, data, performance,
  calibration, threshold policy, explainability, responsible AI, failure
  modes, monitoring, model risk, and a final assurance opinion. Those matters
  are not assessed in A0/A1.

The [scope and independence statement](scope_and_independence.md) controls the
assessment boundary. The [evidence register](evidence_register.csv) records
identified release evidence, and the [evidence map](evidence_map.md) provides
navigation from future assurance questions to that evidence.

The assurance case is neither regulatory certification nor an external
independent audit. It makes no legal, compliance, fairness-certification, or
real-world lending-suitability claim.
