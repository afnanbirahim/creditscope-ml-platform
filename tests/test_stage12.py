"""Static Stage 12 container/CI boundaries and canonical-package verification."""

from __future__ import annotations

from pathlib import Path

from creditscope.container_verify import (
    EXPECTED_ARTIFACT_SHA256,
    PROBABILITY_ATOL,
    verify_container_runtime,
)


def test_runtime_verifier_checks_canonical_package() -> None:
    root = Path(__file__).resolve().parents[1]
    result = verify_container_runtime(root / "artifacts" / "model")
    assert result["status"] == "pass"
    assert result["artifact_sha256"] == EXPECTED_ARTIFACT_SHA256
    assert result["probability_atol"] == PROBABILITY_ATOL == 1e-12
    assert result["maximum_absolute_probability_difference"] <= PROBABILITY_ATOL
    assert result["model_version"] == "creditscope-model-1.0.0"
    assert result["threshold"] == 0.16
    assert result["predictor_count"] == 17
    assert result["calibration_method"] == "sigmoid"
    assert result["ensemble"] is True
    assert result["calibrated_members"] == 5
    assert result["golden_fixture_count"] == 5
    assert result["single_batch_consistency"] is True


def test_api_image_is_non_root_lean_and_immutable() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "Dockerfile.api").read_text(encoding="utf-8")
    lowered = source.lower()
    assert source.startswith("FROM python:3.12.0-slim-bookworm")
    assert "USER creditscope" in source
    assert "creditscope_frozen_model.joblib" in source
    assert "container_verify.py" in source
    assert 'CMD ["python", "-m", "uvicorn"' in source
    assert "health" in lowered
    for forbidden in (
        "data/raw",
        "data/processed",
        "reports/stage7",
        "final_holdout",
        "notebooks",
        "shap",
        "pytest",
    ):
        assert forbidden not in lowered


def test_ui_image_has_no_model_or_inference_package() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "Dockerfile.ui").read_text(encoding="utf-8")
    lowered = source.lower()
    assert source.startswith("FROM python:3.12.0-slim-bookworm")
    assert "USER creditscope" in source
    assert "CREDITSCOPE_API_URL" not in source
    assert 'CMD ["python", "-m", "streamlit"' in source
    for forbidden in (
        "artifacts/",
        "joblib",
        "xgboost",
        "inference.py",
        "api.py",
        "data/",
        "reports/",
        "shap",
    ):
        assert forbidden not in lowered


def test_runtime_dependencies_are_explicitly_pinned_and_split() -> None:
    root = Path(__file__).resolve().parents[1]
    api = (root / "docker" / "requirements-api.txt").read_text()
    ui = (root / "docker" / "requirements-ui.txt").read_text()
    expected_api = {
        "numpy==2.5.3",
        "scipy==1.18.1",
        "pandas==2.3.3",
        "scikit-learn==1.4.2",
        "xgboost==3.4.1",
        "joblib==1.6.0",
        "fastapi==0.141.1",
        "pydantic==2.13.5",
        "uvicorn==0.52.4",
    }
    assert set(api.splitlines()) == expected_api
    assert set(ui.splitlines()) == {"httpx==0.28.1", "streamlit==1.63.0"}
    assert "shap" not in api.lower()
    assert "xgboost" not in ui.lower() and "joblib" not in ui.lower()


def test_docker_context_is_allow_listed_without_excluding_model() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / ".dockerignore").read_text(encoding="utf-8")
    lines = {line.strip() for line in source.splitlines()}
    assert "*" in lines
    assert "!artifacts/model/creditscope_frozen_model.joblib" in lines
    assert "!artifacts/model/model_manifest.json" in lines
    assert "!artifacts/model/input_schema.json" in lines
    assert "!artifacts/model/golden_inference_fixtures.json" in lines
    assert not any(line.startswith("!reports") for line in lines)
    assert not any(line.startswith("!data") for line in lines)


def test_compose_is_two_service_health_gated_and_hardened() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "compose.yaml").read_text(encoding="utf-8")
    assert "  api:\n" in source and "  ui:\n" in source
    assert '"127.0.0.1:8000:8000"' in source
    assert '"127.0.0.1:8501:8501"' in source
    assert "CREDITSCOPE_API_URL: http://api:8000" in source
    assert "condition: service_healthy" in source
    assert source.count("read_only: true") == 2
    assert source.count("no-new-privileges:true") == 2
    assert "networks:\n  creditscope-private:\n    driver: bridge" in source
    assert source.count("      - creditscope-private") == 2
    assert "internal: true" not in source
    lowered = source.lower()
    assert "privileged:" not in lowered
    assert "docker.sock" not in lowered
    assert "secret" not in lowered
    assert "0.16" not in source


def test_ci_has_no_deployment_secrets_or_sensitive_uploads() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / ".github" / "workflows" / "ci.yml").read_text(
        encoding="utf-8"
    )
    lowered = source.lower()
    assert "push:" in source and "pull_request:" in source
    assert "permissions:\n  contents: read" in source
    assert "python -m pytest" in source
    assert "python -m ruff check ." in source
    assert "sha256sum --check scripts/frozen_evidence.sha256" in source
    assert "python -m creditscope.container_verify" in source
    assert "docker build --file Dockerfile.api" in source
    assert "docker build --file Dockerfile.ui" in source
    assert "docker compose up --detach --wait --no-build" in source
    assert "python scripts/smoke_compose.py" in source
    for forbidden in (
        "secrets.",
        "upload-artifact",
        "docker push",
        "deploy",
        "aws",
        "azure",
        "gcloud",
    ):
        assert forbidden not in lowered


def test_compose_smoke_script_is_transport_only() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "scripts" / "smoke_compose.py").read_text(encoding="utf-8")
    lowered = source.lower()
    assert "httpx" in lowered
    assert "api_single_request.json" in source
    assert "api_single_response.json" in source
    for forbidden in (
        "joblib",
        "xgboost",
        "creditscope.inference",
        "final_holdout",
        ".fit(",
    ):
        assert forbidden not in lowered
