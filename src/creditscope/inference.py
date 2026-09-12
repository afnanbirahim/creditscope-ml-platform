"""Strict, pure inference for the serialized CreditScope frozen policy."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV

from creditscope.modeling import (
    AUDIT_ATTRIBUTES,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    PREDICTIVE_FEATURES,
)
from creditscope.schema import CATEGORY_LABELS

MODEL_VERSION = "creditscope-model-1.0.0"
FROZEN_THRESHOLD = 0.16
CLASS_LABELS = {0: "good_credit_risk", 1: "bad_credit_risk"}

CATEGORY_DOMAINS: dict[str, tuple[Any, ...]] = {
    name: tuple(CATEGORY_LABELS[name])
    for name in CATEGORICAL_FEATURES
    if name in CATEGORY_LABELS
}
CATEGORY_DOMAINS.update(
    {
        "installment_rate_percent": (1, 2, 3, 4),
        "residence_duration": (1, 2, 3, 4),
        "existing_credits_count": (1, 2, 3, 4),
        "dependents_count": (1, 2),
    }
)
if set(CATEGORY_DOMAINS) != set(CATEGORICAL_FEATURES):
    raise RuntimeError("Production categorical contract is incomplete.")


class InferenceError(ValueError):
    """Base class for safe, user-facing inference failures."""


class MissingFeatureError(InferenceError):
    """A required predictor is absent."""


class UnexpectedFeatureError(InferenceError):
    """An undeclared or audit-only field was submitted."""


class InvalidValueError(InferenceError):
    """A predictor value violates its frozen contract."""


class ArtifactIntegrityError(InferenceError):
    """The model artifact does not match trusted metadata."""


class ModelCompatibilityError(InferenceError):
    """The manifest or loaded estimator is not the supported frozen version."""


class DuplicateFieldError(InferenceError):
    """A JSON object contains the same field more than once."""


@dataclass(frozen=True)
class LoadedFrozenModel:
    """Trusted estimator plus safe manifest metadata."""

    estimator: CalibratedClassifierCV
    manifest: dict[str, Any]


def file_sha256(path: Path) -> str:
    """Return the lowercase SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def classify_probability(probability_bad_risk: float) -> tuple[int, str]:
    """Apply the one authoritative frozen decision-support threshold rule."""
    if not math.isfinite(probability_bad_risk) or not 0 <= probability_bad_risk <= 1:
        raise InvalidValueError("Probability must be finite and within [0, 1].")
    predicted_class = int(probability_bad_risk >= FROZEN_THRESHOLD)
    return predicted_class, CLASS_LABELS[predicted_class]


def validate_raw_record(record: Mapping[str, Any]) -> pd.DataFrame:
    """Validate one strict 17-field payload and return model-column order."""
    if not isinstance(record, Mapping):
        raise InferenceError("Prediction input must be a JSON-style object.")
    if not all(isinstance(name, str) for name in record):
        raise InferenceError("Every prediction field name must be a string.")
    received = set(record)
    required = set(PREDICTIVE_FEATURES)
    missing = sorted(required - received)
    unexpected = sorted(received - required)
    if missing:
        raise MissingFeatureError(
            f"Missing required predictor(s): {', '.join(missing)}"
        )
    if unexpected:
        audit = sorted(set(unexpected) & set(AUDIT_ATTRIBUTES))
        if audit:
            raise UnexpectedFeatureError(
                "Audit-only field(s) are not accepted for prediction: "
                + ", ".join(audit)
            )
        raise UnexpectedFeatureError(f"Unexpected field(s): {', '.join(unexpected)}")

    validated: dict[str, Any] = {}
    for name in PREDICTIVE_FEATURES:
        value = record[name]
        if value is None or (not isinstance(value, str) and pd.isna(value)):
            raise InvalidValueError(f"{name} must not be null or NaN.")
        if name in NUMERICAL_FEATURES:
            if isinstance(value, bool):
                raise InvalidValueError(f"{name} must be a finite numeric value.")
            try:
                numeric = float(value)
            except (TypeError, ValueError) as error:
                raise InvalidValueError(
                    f"{name} must be a finite numeric value."
                ) from error
            if not math.isfinite(numeric):
                raise InvalidValueError(f"{name} must not be infinite or NaN.")
            validated[name] = numeric
        else:
            if isinstance(value, bool) or value not in CATEGORY_DOMAINS[name]:
                accepted = ", ".join(str(item) for item in CATEGORY_DOMAINS[name])
                raise InvalidValueError(
                    f"Invalid category for {name}: {value!r}. Accepted: {accepted}"
                )
            validated[name] = value
    return pd.DataFrame([validated], columns=list(PREDICTIVE_FEATURES))


def load_frozen_model(
    artifact_path: Path,
    manifest_path: Path,
    *,
    strict_integrity: bool = True,
) -> LoadedFrozenModel:
    """Load only a trusted artifact, optionally checking SHA-256 first.

    Joblib uses pickle semantics and can execute code during loading. Never pass
    an artifact from an untrusted source, even when strict integrity is disabled.
    """
    if not artifact_path.is_file():
        raise FileNotFoundError(f"Frozen model artifact not found: {artifact_path}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Model manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("model_version") != MODEL_VERSION:
        raise ModelCompatibilityError("Unsupported model version in manifest.")
    recorded_environment = manifest.get("environment", {})
    current_environment = {
        "python": platform.python_version(),
        "numpy": version("numpy"),
        "scipy": version("scipy"),
        "pandas": version("pandas"),
        "scikit_learn": version("scikit-learn"),
        "xgboost": version("xgboost"),
        "joblib": version("joblib"),
    }
    mismatches = {
        name: {"recorded": recorded_environment.get(name), "current": current}
        for name, current in current_environment.items()
        if recorded_environment.get(name) != current
    }
    if mismatches:
        raise ModelCompatibilityError(
            "Scientific environment differs from the model manifest: "
            + json.dumps(mismatches, sort_keys=True)
        )
    if strict_integrity:
        expected = manifest.get("artifact_sha256")
        observed = file_sha256(artifact_path)
        if not isinstance(expected, str) or observed != expected:
            raise ArtifactIntegrityError(
                "Frozen model SHA-256 does not match the trusted manifest."
            )
    estimator = joblib.load(artifact_path)
    if not isinstance(estimator, CalibratedClassifierCV):
        raise ModelCompatibilityError("Artifact is not CalibratedClassifierCV.")
    if estimator.method != "sigmoid" or estimator.ensemble is not True:
        raise ModelCompatibilityError("Frozen calibration architecture changed.")
    if len(estimator.calibrated_classifiers_) != 5:
        raise ModelCompatibilityError("Frozen ensemble must contain five members.")
    return LoadedFrozenModel(estimator=estimator, manifest=manifest)


def predict_batch(
    model: LoadedFrozenModel, records: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Validate and predict a non-empty ordered batch without fitting or writes."""
    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise InferenceError("Batch input must be a sequence of record objects.")
    if not records:
        raise InferenceError("Batch input must contain at least one record.")
    frames = [validate_raw_record(record) for record in records]
    frame = pd.concat(frames, ignore_index=True)
    probabilities = model.estimator.predict_proba(frame)[:, 1]
    output: list[dict[str, Any]] = []
    for probability in probabilities:
        predicted_class, predicted_label = classify_probability(float(probability))
        output.append(
            {
                "probability_bad_risk": float(probability),
                "predicted_class": predicted_class,
                "predicted_label": predicted_label,
                "threshold": FROZEN_THRESHOLD,
                "model_version": MODEL_VERSION,
            }
        )
    return output


def predict_one(model: LoadedFrozenModel, record: Mapping[str, Any]) -> dict[str, Any]:
    """Predict one validated record using the frozen calibrated ensemble."""
    return predict_batch(model, [record])[0]


def safe_model_metadata(manifest_path: Path) -> dict[str, Any]:
    """Return public metadata without loading private estimator internals."""
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Model manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    keys = (
        "model_version",
        "model_family",
        "calibration",
        "decision_threshold",
        "target_semantics",
        "positive_class",
        "predictor_count",
        "predictor_names",
        "training_population",
        "environment",
        "artifact_sha256",
    )
    return {key: manifest[key] for key in keys}


def reject_duplicate_fields(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject duplicate JSON keys before a parser can silently discard them."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DuplicateFieldError(f"Duplicate JSON field: {key}")
        result[key] = value
    return result


def load_json_payload(path: Path) -> dict[str, Any]:
    """Load a single-record JSON payload while rejecting duplicate fields."""
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicate_fields
        )
    except json.JSONDecodeError as error:
        raise InferenceError("Input is not valid JSON.") from error
    if not isinstance(payload, dict):
        raise InferenceError("Input JSON must contain one object.")
    return payload


def main() -> None:
    """Load, validate, and predict one local JSON payload."""
    parser = argparse.ArgumentParser(description="Run local CreditScope inference.")
    parser.add_argument("input_json", type=Path)
    parser.add_argument("--model-dir", type=Path, default=Path("artifacts") / "model")
    args = parser.parse_args()
    try:
        model = load_frozen_model(
            args.model_dir / "creditscope_frozen_model.joblib",
            args.model_dir / "model_manifest.json",
        )
        result = predict_one(model, load_json_payload(args.input_json))
    except (InferenceError, FileNotFoundError) as error:
        print(json.dumps({"error": str(error)}), file=sys.stderr)
        raise SystemExit(2) from None
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
