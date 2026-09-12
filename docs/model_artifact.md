# CreditScope frozen model artifact

Model version `creditscope-model-1.0.0` is a joblib serialization of the complete
frozen predictor: Stage 3R preprocessing, fixed untuned Stage 4 XGBoost, and
sigmoid `CalibratedClassifierCV` using the exact persisted five development folds
with `ensemble=True`. It contains five classifier/calibrator pairs and averages
their calibrated probabilities. Threshold `0.16` is enforced by inference code.

The build fits only the 800 development observations (560 good credit risks and
240 bad credit risks). It does not select, fit, or score holdout rows. Stage 7
files are checked by cryptographic hash only as immutable evidence.

## Artifacts

- `creditscope_frozen_model.joblib`
- `model_manifest.json`
- `input_schema.json`
- `golden_inference_fixtures.json`
- `artifact_checksums.sha256`

These are under `artifacts/model/`; no raw training dataset is stored there.

## Build and infer

```powershell
.\.venv\Scripts\python.exe -m creditscope.build_frozen_model
.\.venv\Scripts\python.exe -m creditscope.inference examples/sample_prediction_input.json
```

The build verifies frozen hashes and configuration, serializes the estimator,
reloads it, and requires five development-derived golden fixture probabilities
to agree within `1e-12`, with exact classes and labels.

## Security and compatibility

Joblib relies on pickle and loading can execute arbitrary code. Never load an
untrusted artifact. Strict loading checks SHA-256 against the trusted manifest
before deserialization. Use the scientific versions recorded in the manifest.
The binary hash is version-sensitive and is not guaranteed across environment
rebuilds.

SHAP is not imported by production inference. Explanation is optional post-hoc
analysis and remains operationally separate from prediction.

