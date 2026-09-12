"""Stage 9 packaging and pure-inference contract tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from creditscope.build_frozen_model import (
    build_frozen_model_artifacts,
    create_input_schema,
    load_development_training_data,
    validate_frozen_architecture,
)
from creditscope.inference import (
    FROZEN_THRESHOLD,
    ArtifactIntegrityError,
    DuplicateFieldError,
    InvalidValueError,
    MissingFeatureError,
    UnexpectedFeatureError,
    classify_probability,
    file_sha256,
    load_frozen_model,
    load_json_payload,
    predict_batch,
    predict_one,
    safe_model_metadata,
    validate_raw_record,
)
from creditscope.modeling import (
    AUDIT_ATTRIBUTES,
    PREDICTIVE_FEATURES,
    XGBOOST_PARAMETERS,
)
from creditscope.splits import ROW_ID_NAME, TEST_SPLIT


@pytest.fixture(scope="session")
def packaged_model(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    """Build once in pytest-owned storage, never the canonical artifact directory."""
    root = Path(__file__).resolve().parents[1]
    output = tmp_path_factory.mktemp("stage9-model")
    result = build_frozen_model_artifacts(root, output_dir=output)
    return {"root": root, "output": output, "result": result}


@pytest.fixture
def sample_record(packaged_model: dict[str, object]) -> dict[str, object]:
    output = packaged_model["output"]
    assert isinstance(output, Path)
    fixtures = json.loads(
        (output / "golden_inference_fixtures.json").read_text(encoding="utf-8")
    )
    return fixtures[2]["input"]


def _load(packaged_model: dict[str, object]):
    output = packaged_model["output"]
    assert isinstance(output, Path)
    return load_frozen_model(
        output / "creditscope_frozen_model.joblib",
        output / "model_manifest.json",
    )


def test_packaged_architecture_is_exact(packaged_model: dict[str, object]) -> None:
    loaded = _load(packaged_model)
    validate_frozen_architecture(loaded.estimator)
    assert loaded.estimator.method == "sigmoid"
    assert loaded.estimator.ensemble is True
    assert loaded.estimator.n_jobs is None
    assert len(loaded.estimator.calibrated_classifiers_) == 5
    for member in loaded.estimator.calibrated_classifiers_:
        classifier = member.estimator.named_steps["classifier"]
        assert {
            key: classifier.get_params()[key] for key in XGBOOST_PARAMETERS
        } == XGBOOST_PARAMETERS


def test_threshold_boundary_is_inclusive() -> None:
    assert FROZEN_THRESHOLD == 0.16
    assert classify_probability(0.159999999999)[0] == 0
    assert classify_probability(0.16) == (1, "bad_credit_risk")


def test_schema_has_exact_governed_contract() -> None:
    schema = create_input_schema()
    assert schema["field_order"] == list(PREDICTIVE_FEATURES)
    assert len(schema["fields"]) == 17
    assert schema["audit_only_fields_rejected"] == list(AUDIT_ATTRIBUTES)
    assert schema["additional_fields_allowed"] is False


def test_input_validation_failures(sample_record: dict[str, object]) -> None:
    missing = dict(sample_record)
    missing.pop("duration_months")
    with pytest.raises(MissingFeatureError, match="duration_months"):
        validate_raw_record(missing)
    unexpected = {**sample_record, "unknown": 1}
    with pytest.raises(UnexpectedFeatureError, match="unknown"):
        validate_raw_record(unexpected)
    audit = {**sample_record, "age_years": 35}
    with pytest.raises(UnexpectedFeatureError, match="Audit-only"):
        validate_raw_record(audit)
    invalid_category = {**sample_record, "purpose": "A999"}
    with pytest.raises(InvalidValueError, match="purpose"):
        validate_raw_record(invalid_category)
    for invalid in ("not-numeric", np.nan, np.inf, True):
        record = {**sample_record, "credit_amount": invalid}
        with pytest.raises(InvalidValueError, match="credit_amount"):
            validate_raw_record(record)


def test_duplicate_json_field_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text('{"duration_months": 12, "duration_months": 18}', encoding="utf-8")
    with pytest.raises(DuplicateFieldError, match="duration_months"):
        load_json_payload(path)


def test_single_batch_and_repeat_predictions_match(
    packaged_model: dict[str, object], sample_record: dict[str, object]
) -> None:
    loaded = _load(packaged_model)
    single = predict_one(loaded, sample_record)
    repeated = predict_one(loaded, sample_record)
    batch = predict_batch(loaded, [sample_record, sample_record])
    assert single == repeated == batch[0] == batch[1]
    assert 0 <= single["probability_bad_risk"] <= 1
    assert single["predicted_label"] in {"good_credit_risk", "bad_credit_risk"}


def test_inference_never_refits(
    packaged_model: dict[str, object],
    sample_record: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded = _load(packaged_model)

    def forbidden_fit(*args: object, **kwargs: object) -> None:
        raise AssertionError("fit must never run during inference")

    monkeypatch.setattr(loaded.estimator, "fit", forbidden_fit)
    assert 0 <= predict_one(loaded, sample_record)["probability_bad_risk"] <= 1


def test_serialization_manifest_and_round_trip(
    packaged_model: dict[str, object],
) -> None:
    output = packaged_model["output"]
    result = packaged_model["result"]
    assert isinstance(output, Path) and isinstance(result, dict)
    artifact = output / "creditscope_frozen_model.joblib"
    manifest_path = output / "model_manifest.json"
    assert artifact.is_file() and manifest_path.is_file()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["artifact_sha256"] == file_sha256(artifact)
    assert len(manifest["artifact_sha256"]) == 64
    assert manifest["round_trip_validation"] == {
        "fixture_count": 5,
        "probability_atol": 1e-12,
        "maximum_absolute_probability_difference": 0.0,
        "predictions_exact": True,
        "labels_exact": True,
        "five_members_preserved": True,
    }
    assert result["holdout_rows_used"] == 0


def test_tampered_artifact_is_rejected(
    packaged_model: dict[str, object], tmp_path: Path
) -> None:
    output = packaged_model["output"]
    assert isinstance(output, Path)
    artifact = tmp_path / "model.joblib"
    manifest = tmp_path / "manifest.json"
    artifact.write_bytes(
        (output / "creditscope_frozen_model.joblib").read_bytes() + b"tampered"
    )
    manifest.write_bytes((output / "model_manifest.json").read_bytes())
    with pytest.raises(ArtifactIntegrityError, match="SHA-256"):
        load_frozen_model(artifact, manifest)


def test_safe_metadata_fields(packaged_model: dict[str, object]) -> None:
    output = packaged_model["output"]
    assert isinstance(output, Path)
    metadata = safe_model_metadata(output / "model_manifest.json")
    assert metadata["model_version"] == "creditscope-model-1.0.0"
    assert metadata["decision_threshold"] == 0.16
    assert metadata["predictor_count"] == 17
    assert metadata["training_population"]["rows"] == 800


def test_build_population_is_development_only() -> None:
    root = Path(__file__).resolve().parents[1]
    X, y, development_ids, _ = load_development_training_data(root)
    split = pd.read_csv(root / "reports" / "split_manifest.csv")
    holdout_ids = set(split.loc[split["split"] == TEST_SPLIT, ROW_ID_NAME])
    assert len(X) == len(y) == len(development_ids) == 800
    assert set(development_ids).isdisjoint(holdout_ids)
    assert set(X.columns) == set(PREDICTIVE_FEATURES)
    assert set(X.columns).isdisjoint(AUDIT_ATTRIBUTES)
