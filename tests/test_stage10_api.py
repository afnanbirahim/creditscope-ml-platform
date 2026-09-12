"""Stage 10 API tests: transport parity, fail-closed startup, and boundaries."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import creditscope.api as api_module
from creditscope.api import API_VERSION, create_app
from creditscope.inference import (
    FROZEN_THRESHOLD,
    ArtifactIntegrityError,
    load_frozen_model,
    predict_one,
)
from creditscope.modeling import AUDIT_ATTRIBUTES, PREDICTIVE_FEATURES


@pytest.fixture(scope="module")
def root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def artifact_paths(root: Path) -> tuple[Path, Path]:
    directory = root / "artifacts" / "model"
    return directory / "creditscope_frozen_model.joblib", directory / "model_manifest.json"


@pytest.fixture(scope="module")
def golden(root: Path) -> list[dict[str, object]]:
    return json.loads(
        (root / "artifacts" / "model" / "golden_inference_fixtures.json").read_text(
            encoding="utf-8"
        )
    )


@pytest.fixture(scope="module")
def client(artifact_paths: tuple[Path, Path]) -> TestClient:
    artifact, manifest = artifact_paths
    with TestClient(create_app(model_path=artifact, manifest_path=manifest)) as test_client:
        yield test_client


def test_app_starts_and_loads_model_once(
    artifact_paths: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact, manifest = artifact_paths
    calls = 0
    original = api_module.load_frozen_model

    def counted_load(*args: object, **kwargs: object):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(api_module, "load_frozen_model", counted_load)
    with TestClient(create_app(model_path=artifact, manifest_path=manifest)) as local:
        assert local.get("/health").status_code == 200
        assert local.get("/health").status_code == 200
        assert calls == 1


def test_startup_rejects_missing_or_tampered_artifact(
    artifact_paths: tuple[Path, Path], tmp_path: Path
) -> None:
    artifact, manifest = artifact_paths
    app = create_app(model_path=tmp_path / "missing.joblib", manifest_path=manifest)
    with pytest.raises(FileNotFoundError), TestClient(app):
        pass

    tampered = tmp_path / "tampered.joblib"
    tampered.write_bytes(artifact.read_bytes() + b"tamper")
    app = create_app(model_path=tampered, manifest_path=manifest)
    with pytest.raises(ArtifactIntegrityError), TestClient(app):
        pass


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("artifact_sha256", "0" * 64, "SHA-256"),
        ("model_version", "creditscope-model-wrong", "model version"),
        ("decision_threshold", 0.5, "decision threshold"),
    ],
)
def test_startup_rejects_manifest_policy_mismatch(
    artifact_paths: tuple[Path, Path],
    tmp_path: Path,
    field: str,
    value: object,
    match: str,
) -> None:
    artifact, manifest_path = artifact_paths
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest[field] = value
    altered = tmp_path / f"{field}.json"
    altered.write_text(json.dumps(manifest), encoding="utf-8")
    app = create_app(model_path=artifact, manifest_path=altered)
    with (
        pytest.raises((ArtifactIntegrityError, RuntimeError, ValueError), match=match),
        TestClient(app),
    ):
        pass


def test_core_routes_and_safe_metadata(client: TestClient) -> None:
    root_response = client.get("/")
    assert root_response.status_code == 200
    assert root_response.json()["service_version"] == API_VERSION
    assert "educational/research" in root_response.json()["purpose"].lower()

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json() == {
        "status": "ok",
        "model_loaded": True,
        "model_version": "creditscope-model-1.0.0",
        "artifact_verified": True,
    }

    info = client.get("/model-info")
    assert info.status_code == 200
    body = info.json()
    assert body["threshold"] == 0.16
    assert body["calibration_method"] == "sigmoid"
    assert body["ensemble"] is True
    assert body["calibrated_members"] == 5
    assert body["predictor_names"] == list(PREDICTIVE_FEATURES)
    assert not (set(body["predictor_names"]) & set(AUDIT_ATTRIBUTES))
    assert "environment" not in body


def test_api_matches_direct_stage9_golden_inference(
    client: TestClient,
    artifact_paths: tuple[Path, Path],
    golden: list[dict[str, object]],
) -> None:
    artifact, manifest = artifact_paths
    direct_model = load_frozen_model(artifact, manifest)
    for fixture in golden:
        record = fixture["input"]
        assert isinstance(record, dict)
        api_response = client.post("/predict", json=record)
        assert api_response.status_code == 200
        actual = api_response.json()
        expected = predict_one(direct_model, record)
        assert actual["probability_bad_risk"] == pytest.approx(
            expected["probability_bad_risk"], abs=1e-12, rel=0
        )
        for field in ("predicted_class", "predicted_label", "threshold", "model_version"):
            assert actual[field] == expected[field]


def test_batch_preserves_order_and_matches_single(
    client: TestClient, golden: list[dict[str, object]]
) -> None:
    records = [fixture["input"] for fixture in golden]
    response = client.post("/predict-batch", json=records)
    assert response.status_code == 200
    body = response.json()
    singles = [client.post("/predict", json=record).json() for record in records]
    assert body == singles


def test_batch_boundaries(client: TestClient, golden: list[dict[str, object]]) -> None:
    record = golden[0]["input"]
    assert client.post("/predict-batch", json=[]).status_code == 422
    too_large = client.post("/predict-batch", json=[record] * 101)
    assert too_large.status_code == 413
    mixed = [record, {**record, "purpose": "A999"}]
    invalid = client.post("/predict-batch", json=mixed)
    assert invalid.status_code == 422
    assert not isinstance(invalid.json(), list)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda record: {key: value for key, value in record.items() if key != "purpose"},
        lambda record: {**record, "unexpected": "value"},
        lambda record: {**record, "age_years": 30},
        lambda record: {**record, "purpose": "A999"},
        lambda record: {**record, "duration_months": "12"},
        lambda record: {**record, "duration_months": None},
    ],
)
def test_invalid_inputs_are_rejected_without_tracebacks(
    client: TestClient,
    golden: list[dict[str, object]],
    mutation: object,
) -> None:
    record = golden[0]["input"]
    assert isinstance(record, dict) and callable(mutation)
    response = client.post("/predict", json=mutation(record))
    assert response.status_code == 422
    assert "traceback" not in response.text.lower()


@pytest.mark.parametrize("invalid_token", ["NaN", "Infinity", "-Infinity"])
def test_nonfinite_json_numbers_are_rejected(
    client: TestClient,
    golden: list[dict[str, object]],
    invalid_token: str,
) -> None:
    record = dict(golden[0]["input"])
    record["duration_months"] = 12
    encoded = json.dumps(record).replace('"duration_months": 12', f'"duration_months": {invalid_token}')
    response = client.post(
        "/predict", content=encoded, headers={"content-type": "application/json"}
    )
    assert response.status_code == 422
    assert "traceback" not in response.text.lower()


def test_malformed_payloads_are_rejected(client: TestClient) -> None:
    malformed_json = client.post(
        "/predict", content=b'{"duration_months":', headers={"content-type": "application/json"}
    )
    assert malformed_json.status_code == 422
    assert client.post("/predict", json=[1, 2, 3]).status_code == 422


def test_unexpected_failure_returns_sanitized_500(
    artifact_paths: tuple[Path, Path],
    golden: list[dict[str, object]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected(*args: object, **kwargs: object) -> None:
        raise RuntimeError("private local path C:/secret/model")

    monkeypatch.setattr(api_module, "predict_one", unexpected)
    artifact, manifest = artifact_paths
    app = create_app(model_path=artifact, manifest_path=manifest)
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post("/predict", json=golden[0]["input"])
    assert response.status_code == 500
    assert response.json() == {"detail": "Unexpected internal service error."}
    assert "secret" not in response.text.lower()


def test_openapi_and_route_boundary(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()
    documented = set(schema["paths"])
    assert documented == {"/", "/health", "/model-info", "/predict", "/predict-batch"}
    assert "lending decisions" in schema["info"]["description"]
    for prohibited in ("/train", "/fit", "/threshold", "/model-update", "/shap", "/holdout"):
        assert client.post(prohibited).status_code == 404


def test_request_cannot_override_frozen_policy(
    client: TestClient, golden: list[dict[str, object]]
) -> None:
    record = golden[0]["input"]
    response = client.post("/predict?threshold=0.50", json=record)
    assert response.status_code == 200
    assert response.json()["threshold"] == 0.16
    for field in ("threshold", "model_parameters", "calibration", "predictors"):
        assert client.post("/predict", json={**record, field: "changed"}).status_code == 422


def test_api_module_has_no_shap_or_holdout_runtime_dependency() -> None:
    source = Path(api_module.__file__).read_text(encoding="utf-8")
    assert "import shap" not in source
    assert "final_holdout" not in source
    assert ".fit(" not in source
    assert FROZEN_THRESHOLD == 0.16
