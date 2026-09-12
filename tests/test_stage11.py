"""Stage 11 HTTP-client, frontend-schema, and presentation tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from creditscope.api import create_app
from creditscope.api_client import APIClientError, CreditScopeAPIClient
from creditscope.frontend import (
    UI_FIELD_NAMES,
    UI_FIELDS,
    build_payload,
    display_label,
    format_probability,
    presentation_values,
)

AUDIT_ONLY = {"personal_status_sex", "age_years", "foreign_worker"}


@pytest.fixture(scope="module")
def root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def sample_request(root: Path) -> dict[str, Any]:
    return json.loads((root / "examples/api_single_request.json").read_text())


def _client_for_handler(handler: Any) -> CreditScopeAPIClient:
    transport = httpx.MockTransport(handler)
    return CreditScopeAPIClient(
        "http://testserver",
        http_client=httpx.Client(base_url="http://testserver", transport=transport),
    )


def test_frontend_exposes_exact_predictor_boundary(
    sample_request: dict[str, Any],
) -> None:
    assert len(UI_FIELD_NAMES) == 17
    assert UI_FIELD_NAMES == tuple(sample_request)
    assert set(UI_FIELD_NAMES).isdisjoint(AUDIT_ONLY)
    assert "threshold" not in UI_FIELD_NAMES
    assert all(field.label != field.name for field in UI_FIELDS)


def test_category_labels_preserve_canonical_codes() -> None:
    for field in UI_FIELDS:
        if field.kind != "category":
            continue
        assert field.options
        assert field.default in field.options
        assert field.option_labels is not None
        assert set(field.option_labels) == set(field.options)
        for code, label in field.option_labels.items():
            assert f"({code})" in label


def test_payload_builder_preserves_values_and_rejects_boundary_changes(
    sample_request: dict[str, Any],
) -> None:
    built = build_payload(dict(reversed(list(sample_request.items()))))
    assert tuple(built) == UI_FIELD_NAMES
    assert built == sample_request
    with pytest.raises(ValueError, match="missing"):
        build_payload({key: value for key, value in built.items() if key != "purpose"})
    with pytest.raises(ValueError, match="unexpected"):
        build_payload({**built, "age_years": 30})


def test_healthy_api_client_and_metadata() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok", "model_loaded": True})
        return httpx.Response(200, json={"model_version": "creditscope-model-1.0.0"})

    client = _client_for_handler(handler)
    assert client.health()["status"] == "ok"
    assert client.model_info()["model_version"] == "creditscope-model-1.0.0"


def test_api_client_sample_prediction_matches_stage10(
    root: Path, sample_request: dict[str, Any]
) -> None:
    backend_app = create_app(
        model_path=root / "artifacts/model/creditscope_frozen_model.joblib",
        manifest_path=root / "artifacts/model/model_manifest.json",
    )
    with TestClient(backend_app) as backend:
        direct = backend.post("/predict", json=sample_request)

        def handler(request: httpx.Request) -> httpx.Response:
            response = backend.request(
                request.method,
                request.url.path,
                content=request.content,
                headers={"content-type": "application/json"},
            )
            return httpx.Response(
                response.status_code,
                content=response.content,
                headers={"content-type": "application/json"},
            )

        client = _client_for_handler(handler)
        frontend = client.predict(sample_request)

    assert direct.status_code == 200
    expected = direct.json()
    assert frontend["probability_bad_risk"] == pytest.approx(
        expected["probability_bad_risk"], rel=0, abs=1e-12
    )
    for field in ("predicted_class", "predicted_label", "threshold", "model_version"):
        assert frontend[field] == expected[field]


@pytest.mark.parametrize(
    ("status_code", "expected_text"),
    [
        (400, "rejected"),
        (413, "maximum size"),
        (422, "input values"),
        (500, "internal error"),
        (503, "unavailable"),
    ],
)
def test_api_status_failures_are_sanitized(
    status_code: int, expected_text: str
) -> None:
    client = _client_for_handler(
        lambda _request: httpx.Response(
            status_code, text="private traceback C:/secret/model"
        )
    )
    with pytest.raises(APIClientError) as captured:
        client.predict({})
    assert captured.value.status_code == status_code
    assert expected_text in captured.value.user_message
    assert "secret" not in captured.value.user_message


@pytest.mark.parametrize(
    ("exception", "expected_text"),
    [
        (httpx.ConnectError("refused"), "unavailable"),
        (httpx.ReadTimeout("slow"), "timed out"),
    ],
)
def test_network_failures_are_sanitized(
    exception: httpx.RequestError, expected_text: str
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        exception.request = request
        raise exception

    client = _client_for_handler(handler)
    with pytest.raises(APIClientError) as captured:
        client.health()
    assert expected_text in captured.value.user_message
    assert "traceback" not in captured.value.user_message.lower()


def test_batch_client_preserves_response_order() -> None:
    expected = [
        {"predicted_class": 0, "predicted_label": "good_credit_risk"},
        {"predicted_class": 1, "predicted_label": "bad_credit_risk"},
    ]
    client = _client_for_handler(lambda _request: httpx.Response(200, json=expected))
    assert client.predict_batch([{}, {}]) == expected


def test_response_presentation_is_neutral_and_non_mutating() -> None:
    prediction = {
        "probability_bad_risk": 0.349589290168932,
        "predicted_class": 1,
        "predicted_label": "bad_credit_risk",
        "threshold": 0.16,
        "model_version": "creditscope-model-1.0.0",
    }
    displayed = presentation_values(prediction)
    assert displayed["label"] == "Bad credit risk"
    assert displayed["probability"] == prediction["probability_bad_risk"]
    assert displayed["probability_display"] == "35.0%"
    assert displayed["threshold_display"] == "16.0%"
    assert prediction["probability_bad_risk"] == 0.349589290168932
    assert display_label("good_credit_risk") == "Good credit risk"
    assert format_probability(0.16) == "16.0%"


def test_frontend_has_no_model_bypass_or_holdout_dependency(root: Path) -> None:
    sources = "\n".join(
        (root / "src" / "creditscope" / name).read_text(encoding="utf-8").lower()
        for name in ("api_client.py", "frontend.py", "ui.py")
    )
    for forbidden in (
        "import joblib",
        "import xgboost",
        "import shap",
        "from creditscope.inference",
        "creditscope_frozen_model",
        "final_holdout",
        ".fit(",
    ):
        assert forbidden not in sources

