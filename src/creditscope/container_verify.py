"""Verify the canonical frozen artifact and golden fixtures in a runtime image."""

from __future__ import annotations

import argparse
import json
import platform
from importlib.metadata import version
from pathlib import Path
from typing import Any

from creditscope.inference import (
    FROZEN_THRESHOLD,
    MODEL_VERSION,
    file_sha256,
    load_frozen_model,
    predict_batch,
    predict_one,
)
from creditscope.modeling import PREDICTIVE_FEATURES, XGBOOST_PARAMETERS

EXPECTED_ARTIFACT_SHA256 = (
    "c92e062e1cdde4965a7130be244cba8d01bc95e2b0d0cbb8a7feda43fbc3bcb7"
)
PROBABILITY_ATOL = 1e-12


def verify_container_runtime(model_dir: Path) -> dict[str, Any]:
    """Load the trusted artifact and validate architecture and golden parity."""
    artifact_path = model_dir / "creditscope_frozen_model.joblib"
    manifest_path = model_dir / "model_manifest.json"
    schema_path = model_dir / "input_schema.json"
    fixture_path = model_dir / "golden_inference_fixtures.json"

    observed_hash = file_sha256(artifact_path)
    if observed_hash != EXPECTED_ARTIFACT_SHA256:
        raise RuntimeError("Canonical model artifact SHA-256 changed.")
    loaded = load_frozen_model(artifact_path, manifest_path, strict_integrity=True)
    manifest = loaded.manifest
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    fixtures = json.loads(fixture_path.read_text(encoding="utf-8"))

    if manifest["model_version"] != MODEL_VERSION:
        raise RuntimeError("Model version changed.")
    if manifest["decision_threshold"] != FROZEN_THRESHOLD:
        raise RuntimeError("Frozen threshold changed.")
    if tuple(manifest["predictor_names"]) != PREDICTIVE_FEATURES:
        raise RuntimeError("Manifest predictor boundary changed.")
    if tuple(schema["field_order"]) != PREDICTIVE_FEATURES:
        raise RuntimeError("Input-schema predictor boundary changed.")
    estimator = loaded.estimator
    if estimator.method != "sigmoid" or estimator.ensemble is not True:
        raise RuntimeError("Calibration architecture changed.")
    if len(estimator.calibrated_classifiers_) != 5:
        raise RuntimeError("Calibrated ensemble must contain five members.")
    for member in estimator.calibrated_classifiers_:
        classifier = member.estimator.named_steps["classifier"]
        parameters = classifier.get_params()
        if any(
            parameters.get(name) != expected
            for name, expected in XGBOOST_PARAMETERS.items()
        ):
            raise RuntimeError("A fitted XGBoost member configuration changed.")

    records = [fixture["input"] for fixture in fixtures]
    batch_results = predict_batch(loaded, records)
    maximum_difference = 0.0
    for fixture, batch_result in zip(fixtures, batch_results, strict=True):
        expected = fixture["expected"]
        single_result = predict_one(loaded, fixture["input"])
        difference = abs(
            single_result["probability_bad_risk"]
            - expected["probability_bad_risk"]
        )
        maximum_difference = max(maximum_difference, difference)
        if difference > PROBABILITY_ATOL:
            raise RuntimeError("Golden probability exceeded the 1e-12 tolerance.")
        for field in ("predicted_class", "predicted_label", "threshold", "model_version"):
            if single_result[field] != expected[field]:
                raise RuntimeError(f"Golden {field} did not match.")
        batch_difference = abs(
            batch_result["probability_bad_risk"]
            - single_result["probability_bad_risk"]
        )
        if batch_difference > PROBABILITY_ATOL:
            raise RuntimeError("Single and batch probabilities differ.")
        if batch_result != single_result:
            raise RuntimeError("Single and batch prediction semantics differ.")

    return {
        "status": "pass",
        "artifact_sha256": observed_hash,
        "model_version": MODEL_VERSION,
        "threshold": FROZEN_THRESHOLD,
        "predictor_count": len(PREDICTIVE_FEATURES),
        "calibration_method": estimator.method,
        "ensemble": estimator.ensemble,
        "calibrated_members": len(estimator.calibrated_classifiers_),
        "golden_fixture_count": len(fixtures),
        "probability_atol": PROBABILITY_ATOL,
        "maximum_absolute_probability_difference": maximum_difference,
        "single_batch_consistency": True,
        "environment": {
            "python": platform.python_version(),
            "numpy": version("numpy"),
            "scipy": version("scipy"),
            "pandas": version("pandas"),
            "scikit_learn": version("scikit-learn"),
            "xgboost": version("xgboost"),
            "joblib": version("joblib"),
        },
    }


def main() -> None:
    """CLI entry point used locally, in Docker, and in CI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("artifacts") / "model",
    )
    args = parser.parse_args()
    print(json.dumps(verify_container_runtime(args.model_dir), indent=2))


if __name__ == "__main__":
    main()
