"""Stage 10 FastAPI transport and frozen-policy parity tests."""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import creditscope.api as api_module
from creditscope.api import (
    API_VERSION,
    DEFAULT_BATCH_LIMIT,
    EXPECTED_ARTIFACT_SHA256,
    CreditRiskInput,
    create_app,
)
from creditscope.inference import (
    FROZEN_THRESHOLD,
    MODEL_VERSION,
    ArtifactIntegrityError,
    ModelCompatibilityError,
    load_frozen_model,
    predict_one,
)
from creditscope.modeling import (
    AUDIT_ATTRIBUTES,
    PREDICTIVE_FEATURES,
    XGBOOST_PARAMETERS,
)


@pytest.fixture(scope="module")
def root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def golden_records(root: Path) -> list[dict[str, object]]:
    fixtures = json.loads(
        (root / "artifacts/model/golden_inference_fixtures.json").read_text(
            encoding="utf-8"
        )
    )
    return [item["input"] for item in fixtures]


@pytest.fixture(scope="module")
def api_client(root: Path) -> Iterator[TestClient]:
    app = create_app(
        model_path=root / "artifacts/model/creditscope_frozen_model.joblib",
        manifest_path=root / "artifacts/model/model_manifest.json",
    )
    with TestClient(app) as client:
        yield client


def _copy_model_files(root: Path, destination: Path) -> tuple[Path, Path]:
    artifact = destination / "model.joblib"
    manifest = destination / "manifest.json"
    shutil.copyfile(root / "artifacts/model/creditscope_frozen_model.joblib", artifact)
    shutil.copyfile(root / "artifacts/model/model_manifest.json", manifest)
    return artifact, manifest


def _rewrite_manifest(path: Path, updates: dict[str, object]) -> None:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest.update(updates)
    path.write_text(json.dumps(manifest), encoding="utf-8")


def test_api_dependency_surface_and_versions() -> None:
    assert API_VERSION == "creditscope-api-1.0.0"
    assert MODEL_VERSION == "creditscope-model-1.0.0"
    assert FROZEN_THRESHOLD == 0.16
    assert DEFAULT_BATCH_LIMIT == 100
    assert tuple(CreditRiskInput.model_fields) == PREDICTIVE_FEATURES
    assert set(CreditRiskInput.model_fields).isdisjoint(AUDIT_ATTRIBUTES)


def test_startup_loads_verified_model_exactly_once(
    root: Path, golden_records: list[dict[str, object]], monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = 0
    real_loader = api_module.load_frozen_model

    def counted_loader(*args: object, **kwargs: object):
        nonlocal calls
        calls += 1
        return real_loader(*args, **kwargs)

    monkeypatch.setattr(api_module, "load_frozen_model", counted_loader)
    app = create_app(
        model_path=root / "artifacts/model/creditscope_frozen_model.joblib",
        manifest_path=root / "artifacts/model/model_manifest.json",
    )
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        assert client.post("/predict", json=golden_records[0]).status_code == 200
        assert client.post("/predict", json=golden_records[1]).status_code == 200
    assert calls == 1


def test_startup_rejects_missing_artifact(root: Path, tmp_path: Path) -> None:
    app = create_app(
        model_path=tmp_path / "missing.joblib",
        manifest_path=root / "artifacts/model/model_manifest.json",
    )
    with pytest.raises(
        FileNotFoundError, match="Frozen model artifact not found"
    ), TestClient(app):
        pass


def test_startup_rejects_missing_manifest(root: Path, tmp_path: Path) -> None:
    app = create_app(
        model_path=root / "artifacts/model/creditscope_frozen_model.joblib",
        manifest_path=tmp_path / "missing.json",
    )
    with pytest.raises(
        FileNotFoundError, match="Model manifest not found"
    ), TestClient(app):
        pass


def test_startup_rejects_tampered_artifact(root: Path, tmp_path: Path) -> None:
    artifact, manifest = _copy_model_files(root, tmp_path)
    artifact.write_bytes(artifact.read_bytes() + b"tampered")
    app = create_app(model_path=artifact, manifest_path=manifest)
    with pytest.raises(ArtifactIntegrityError, match="SHA-256"), TestClient(app):
        pass


def test_startup_rejects_manifest_hash_mismatch(root: Path, tmp_path: Path) -> None:
    artifact, manifest = _copy_model_files(root, tmp_path)
    _rewrite_manifest(manifest, {"artifact_sha256": "0" * 64})
    app = create_app(model_path=artifact, manifest_path=manifest)
    with pytest.raises(ArtifactIntegrityError, match="SHA-256"), TestClient(app):
        pass


def test_startup_rejects_wrong_model_version(root: Path, tmp_path: Path) -> None:
    artifact, manifest = _copy_model_files(root, tmp_path)
    _rewrite_manifest(manifest, {"model_version": "wrong-model"})
    app = create_app(model_path=artifact, manifest_path=manifest)
    with pytest.raises(
        ModelCompatibilityError, match="model version"
    ), TestClient(app):
        pass


def test_startup_rejects_threshold_mismatch(root: Path, tmp_path: Path) -> None:
    artifact, manifest = _copy_model_files(root, tmp_path)
    _rewrite_manifest(manifest, {"decision_threshold": 0.17})
    app = create_app(model_path=artifact, manifest_path=manifest)
    with pytest.raises(ModelCompatibilityError, match="threshold"), TestClient(app):
        pass


def test_core_routes_and_safe_responses(api_client: TestClient) -> None:
    root_response = api_client.get("/")
    assert root_response.status_code == 200
    assert root_response.json() == {
        "service_name": "CreditScope Inference API",
        "service_version": API_VERSION,
        "model_version": MODEL_VERSION,
        "purpose": "Educational/research credit-risk decision-support prototype.",
        "documentation": "/docs",
    }
    assert api_client.get("/health").json() == {
        "status": "ok",
        "model_loaded": True,
        "model_version": MODEL_VERSION,
        "artifact_verified": True,
    }
    info = api_client.get("/model-info")
    assert info.status_code == 200
    assert info.json()["artifact_sha256"] == EXPECTED_ARTIFACT_SHA256
    assert info.json()["predictor_names"] == list(PREDICTIVE_FEATURES)
    assert info.json()["calibrated_members"] == 5
    assert "estimator" not in info.text and "artifact_filename" not in info.text
    estimator = api_client.app.state.creditscope.model.estimator
    assert estimator.method == "sigmoid"
    assert estimator.ensemble is True
    assert estimator.n_jobs is None
    assert len(estimator.calibrated_classifiers_) == 5
    for member in estimator.calibrated_classifiers_:
        classifier = member.estimator.named_steps["classifier"]
        assert {
            name: classifier.get_params()[name] for name in XGBOOST_PARAMETERS
        } == XGBOOST_PARAMETERS


def test_openapi_exposes_only_core_business_routes(api_client: TestClient) -> None:
    paths = set(api_client.get("/openapi.json").json()["paths"])
    assert paths == {"/", "/health", "/model-info", "/predict", "/predict-batch"}
    assert all(
        forbidden not in path
        for path in paths
        for forbidden in ("train", "fit", "update", "threshold", "holdout", "shap")
    )


def test_api_predictions_match_direct_stage9_inference(
    root: Path,
    api_client: TestClient,
    golden_records: list[dict[str, object]],
) -> None:
    model = load_frozen_model(
        root / "artifacts/model/creditscope_frozen_model.joblib",
        root / "artifacts/model/model_manifest.json",
    )
    for record in golden_records:
        direct = predict_one(model, record)
        response = api_client.post("/predict", json=record)
        assert response.status_code == 200
        actual = response.json()
        assert actual["probability_bad_risk"] == pytest.approx(
            direct["probability_bad_risk"], rel=0, abs=1e-12
        )
        assert actual["predicted_class"] == direct["predicted_class"]
        assert actual["predicted_label"] == direct["predicted_label"]
        assert actual["threshold"] == direct["threshold"] == 0.16
        assert actual["model_version"] == direct["model_version"] == MODEL_VERSION


def test_batch_preserves_order_and_matches_single(
    api_client: TestClient, golden_records: list[dict[str, object]]
) -> None:
    batch = api_client.post("/predict-batch", json=golden_records)
    singles = [api_client.post("/predict", json=record).json() for record in golden_records]
    assert batch.status_code == 200
    assert batch.json() == singles


def test_empty_and_oversized_batches_are_rejected(
    api_client: TestClient, golden_records: list[dict[str, object]]
) -> None:
    empty = api_client.post("/predict-batch", json=[])
    assert empty.status_code == 422
    oversized = api_client.post(
        "/predict-batch", json=[golden_records[0]] * (DEFAULT_BATCH_LIMIT + 1)
    )
    assert oversized.status_code == 413
    assert "100-record limit" in oversized.json()["detail"]


@pytest.mark.parametrize(
    ("mutation", "value"),
    [
        ("missing", None),
        ("unexpected", 1),
        ("audit", 35),
        ("invalid_category", "A999"),
        ("numeric_string", "12"),
        ("null", None),
    ],
)
def test_invalid_inputs_return_safe_422(
    api_client: TestClient,
    golden_records: list[dict[str, object]],
    mutation: str,
    value: object,
) -> None:
    record = dict(golden_records[0])
    if mutation == "missing":
        record.pop("duration_months")
    elif mutation == "unexpected":
        record["unexpected"] = value
    elif mutation == "audit":
        record[AUDIT_ATTRIBUTES[1]] = value
    elif mutation == "invalid_category":
        record["purpose"] = value
    elif mutation == "numeric_string":
        record["duration_months"] = value
    else:
        record["duration_months"] = value
    response = api_client.post("/predict", json=record)
    assert response.status_code == 422
    assert "Traceback" not in response.text
    assert str(Path.cwd()) not in response.text


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_numeric_values_are_rejected(
    api_client: TestClient,
    golden_records: list[dict[str, object]],
    value: float,
) -> None:
    record = dict(golden_records[0])
    record["duration_months"] = value
    record_json = json.dumps(record)
    response = api_client.post(
        "/predict", content=record_json, headers={"content-type": "application/json"}
    )
    assert response.status_code == 422
    assert "Traceback" not in response.text


def test_malformed_duplicate_and_invalid_batch_member_are_atomic(
    api_client: TestClient, golden_records: list[dict[str, object]]
) -> None:
    malformed = api_client.post(
        "/predict", content="{", headers={"content-type": "application/json"}
    )
    assert malformed.status_code == 422
    duplicate = api_client.post(
        "/predict",
        content='{"duration_months": 12, "duration_months": 18}',
        headers={"content-type": "application/json"},
    )
    assert duplicate.status_code == 400
    invalid = dict(golden_records[1])
    invalid["housing"] = "A999"
    batch = api_client.post("/predict-batch", json=[golden_records[0], invalid])
    assert batch.status_code == 422
    assert "predictions" not in batch.json()


def test_health_does_not_predict_and_requests_do_not_fit(
    api_client: TestClient,
    golden_records: list[dict[str, object]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    estimator = api_client.app.state.creditscope.model.estimator

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("health must not predict and inference must not fit")

    monkeypatch.setattr(estimator, "fit", forbidden)
    original_predict = estimator.predict_proba
    monkeypatch.setattr(estimator, "predict_proba", forbidden)
    assert api_client.get("/health").status_code == 200
    monkeypatch.setattr(estimator, "predict_proba", original_predict)
    assert api_client.post("/predict", json=golden_records[0]).status_code == 200


def test_api_source_has_no_holdout_shap_or_model_development_path(root: Path) -> None:
    source = (root / "src/creditscope/api.py").read_text(encoding="utf-8").lower()
    assert "import shap" not in source
    assert "final_holdout" not in source
    assert ".fit(" not in source
    assert "allow_origins" not in source
