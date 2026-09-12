"""FastAPI transport layer for the immutable Stage 9 inference package."""

from __future__ import annotations

import json
import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Annotated, Any, Literal

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, RootModel, model_validator

from creditscope.inference import (
    FROZEN_THRESHOLD,
    MODEL_VERSION,
    DuplicateFieldError,
    InferenceError,
    LoadedFrozenModel,
    ModelCompatibilityError,
    file_sha256,
    load_frozen_model,
    predict_batch,
    predict_one,
    reject_duplicate_fields,
    safe_model_metadata,
    validate_raw_record,
)
from creditscope.modeling import PREDICTIVE_FEATURES, XGBOOST_PARAMETERS

API_VERSION = "creditscope-api-1.0.0"
SERVICE_NAME = "CreditScope Inference API"
DEFAULT_BATCH_LIMIT = 100
EXPECTED_ARTIFACT_SHA256 = (
    "c92e062e1cdde4965a7130be244cba8d01bc95e2b0d0cbb8a7feda43fbc3bcb7"
)
EXPECTED_XGBOOST_PARAMETERS = XGBOOST_PARAMETERS
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "artifacts" / "model" / (
    "creditscope_frozen_model.joblib"
)
DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "artifacts" / "model" / "model_manifest.json"

logger = logging.getLogger(__name__)


class StrictAPIModel(BaseModel):
    """Base HTTP schema with no coercion or undeclared fields."""

    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class CreditRiskInput(StrictAPIModel):
    """Exactly the 17 governed Stage 9 prediction inputs."""

    duration_months: Annotated[
        float,
        Field(strict=True, allow_inf_nan=False, description="Credit duration in months."),
    ]
    credit_amount: Annotated[
        float,
        Field(
            strict=True,
            allow_inf_nan=False,
            description=(
                "Historical transformed credit-amount quantity; not a literal "
                "contemporary monetary amount."
            ),
        ),
    ]
    checking_account_status: Annotated[
        Literal["A11", "A12", "A13", "A14"],
        Field(description="Documented status-of-checking-account source code."),
    ]
    credit_history: Annotated[
        Literal["A30", "A31", "A32", "A33", "A34"],
        Field(description="Documented credit-history source code."),
    ]
    purpose: Annotated[
        Literal[
            "A40",
            "A41",
            "A42",
            "A43",
            "A44",
            "A45",
            "A46",
            "A47",
            "A48",
            "A49",
            "A410",
        ],
        Field(description="Documented credit-purpose source code."),
    ]
    savings_account_status: Annotated[
        Literal["A61", "A62", "A63", "A64", "A65"],
        Field(description="Documented savings-account/bonds source code."),
    ]
    employment_duration: Annotated[
        Literal["A71", "A72", "A73", "A74", "A75"],
        Field(description="Discretized present-employment-duration source code."),
    ]
    installment_rate_percent: Annotated[
        Literal[1, 2, 3, 4],
        Field(description="Discretized installment-rate level, not a literal percentage."),
    ]
    other_debtors_guarantors: Annotated[
        Literal["A101", "A102", "A103"],
        Field(description="Documented other-debtors/guarantors source code."),
    ]
    residence_duration: Annotated[
        Literal[1, 2, 3, 4],
        Field(description="Discretized residence-duration source level."),
    ]
    property: Annotated[
        Literal["A121", "A122", "A123", "A124"],
        Field(description="Documented property-category source code."),
    ]
    other_installment_plans: Annotated[
        Literal["A141", "A142", "A143"],
        Field(description="Documented other-installment-plans source code."),
    ]
    housing: Annotated[
        Literal["A151", "A152", "A153"],
        Field(description="Documented housing-status source code."),
    ]
    existing_credits_count: Annotated[
        Literal[1, 2, 3, 4],
        Field(description="Discretized existing-credit-count source level."),
    ]
    job: Annotated[
        Literal["A171", "A172", "A173", "A174"],
        Field(description="Documented job-category source code."),
    ]
    dependents_count: Annotated[
        Literal[1, 2],
        Field(description="Binary/discretized dependents-count source level."),
    ]
    telephone: Annotated[
        Literal["A191", "A192"],
        Field(description="Documented telephone-status source code."),
    ]

    @model_validator(mode="after")
    def enforce_stage9_contract(self) -> CreditRiskInput:
        """Delegate final semantic validation to the Stage 9 authority."""
        try:
            validate_raw_record(self.model_dump())
        except InferenceError as error:
            raise ValueError(str(error)) from error
        return self


class CreditRiskPrediction(StrictAPIModel):
    """Frozen decision-support prediction response."""

    probability_bad_risk: Annotated[
        float, Field(ge=0.0, le=1.0, description="Calibrated probability of bad credit risk.")
    ]
    predicted_class: Literal[0, 1]
    predicted_label: Literal["good_credit_risk", "bad_credit_risk"]
    threshold: Literal[0.16]
    model_version: Literal["creditscope-model-1.0.0"]


class BatchPredictionRequest(RootModel[list[CreditRiskInput]]):
    """A JSON array of prediction records; size is enforced by the app."""

    @model_validator(mode="after")
    def reject_empty_batch(self) -> BatchPredictionRequest:
        if not self.root:
            raise ValueError("Batch must contain at least one record.")
        return self


class BatchPredictionResponse(RootModel[list[CreditRiskPrediction]]):
    """Ordered all-or-nothing batch response."""


class HealthResponse(StrictAPIModel):
    """Lightweight readiness state established during startup."""

    status: Literal["ok"]
    model_loaded: Literal[True]
    model_version: Literal["creditscope-model-1.0.0"]
    artifact_verified: Literal[True]


class ModelInfoResponse(StrictAPIModel):
    """Safe public metadata without estimator internals or training rows."""

    service_version: Literal["creditscope-api-1.0.0"]
    model_version: Literal["creditscope-model-1.0.0"]
    model_family: str
    calibration_method: Literal["sigmoid"]
    ensemble: Literal[True]
    calibrated_members: Literal[5]
    threshold: Literal[0.16]
    target_semantics: dict[str, str]
    positive_class: Literal[1]
    predictor_count: Literal[17]
    predictor_names: list[str]
    training_population: dict[str, Any]
    artifact_sha256: str


class RootResponse(StrictAPIModel):
    """Service discovery response."""

    service_name: str
    service_version: Literal["creditscope-api-1.0.0"]
    model_version: Literal["creditscope-model-1.0.0"]
    purpose: str
    documentation: Literal["/docs"]


def _configured_path(explicit: Path | None, environment_name: str, default: Path) -> Path:
    if explicit is not None:
        return explicit
    configured = os.getenv(environment_name)
    return Path(configured) if configured else default


def _configured_batch_limit(explicit: int | None) -> int:
    raw: int | str = explicit if explicit is not None else os.getenv(
        "CREDITSCOPE_BATCH_LIMIT", str(DEFAULT_BATCH_LIMIT)
    )
    try:
        limit = int(raw)
    except (TypeError, ValueError) as error:
        raise ValueError("CREDITSCOPE_BATCH_LIMIT must be a positive integer.") from error
    if limit < 1:
        raise ValueError("CREDITSCOPE_BATCH_LIMIT must be a positive integer.")
    return limit


def _verify_startup_contract(model: LoadedFrozenModel, artifact_path: Path) -> None:
    """Fail closed when API metadata differs from the frozen Stage 9 policy."""
    manifest = model.manifest
    calibration = manifest.get("calibration", {})
    failures: list[str] = []
    if manifest.get("artifact_sha256") != EXPECTED_ARTIFACT_SHA256:
        failures.append("authoritative artifact SHA-256")
    if file_sha256(artifact_path) != EXPECTED_ARTIFACT_SHA256:
        failures.append("model artifact SHA-256")
    if manifest.get("model_version") != MODEL_VERSION:
        failures.append("model version")
    if manifest.get("decision_threshold") != FROZEN_THRESHOLD:
        failures.append("decision threshold")
    if manifest.get("predictor_count") != len(PREDICTIVE_FEATURES):
        failures.append("predictor count")
    if tuple(manifest.get("predictor_names", ())) != tuple(PREDICTIVE_FEATURES):
        failures.append("predictor boundary/order")
    if calibration.get("method") != "sigmoid":
        failures.append("calibration method")
    if calibration.get("ensemble") is not True:
        failures.append("calibration ensemble setting")
    if calibration.get("calibrated_members") != 5:
        failures.append("calibrated member count")
    if manifest.get("model_hyperparameters") != EXPECTED_XGBOOST_PARAMETERS:
        failures.append("model hyperparameters")
    for member in model.estimator.calibrated_classifiers_:
        try:
            classifier = member.estimator.named_steps["classifier"]
            parameters = classifier.get_params()
        except (AttributeError, KeyError):
            failures.append("fitted estimator architecture")
            break
        if any(
            parameters.get(name) != expected
            for name, expected in EXPECTED_XGBOOST_PARAMETERS.items()
        ):
            failures.append("fitted XGBoost parameters")
            break
    if failures:
        raise ModelCompatibilityError(
            "Frozen model startup contract mismatch: " + ", ".join(failures)
        )


def _safe_model_info(metadata: dict[str, Any]) -> dict[str, Any]:
    calibration = metadata["calibration"]
    return {
        "service_version": API_VERSION,
        "model_version": metadata["model_version"],
        "model_family": metadata["model_family"],
        "calibration_method": calibration["method"],
        "ensemble": calibration["ensemble"],
        "calibrated_members": calibration["calibrated_members"],
        "threshold": metadata["decision_threshold"],
        "target_semantics": metadata["target_semantics"],
        "positive_class": metadata["positive_class"],
        "predictor_count": metadata["predictor_count"],
        "predictor_names": metadata["predictor_names"],
        "training_population": metadata["training_population"],
        "artifact_sha256": metadata["artifact_sha256"],
    }


def _model_from_state(request: Request) -> LoadedFrozenModel:
    state = getattr(request.app.state, "creditscope", None)
    if state is None or state.model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Verified model is unavailable.",
        )
    return state.model


def create_app(
    *,
    model_path: Path | None = None,
    manifest_path: Path | None = None,
    batch_limit: int | None = None,
) -> FastAPI:
    """Create a testable API whose model is loaded once during lifespan startup."""
    resolved_model_path = _configured_path(
        model_path, "CREDITSCOPE_MODEL_PATH", DEFAULT_MODEL_PATH
    )
    resolved_manifest_path = _configured_path(
        manifest_path, "CREDITSCOPE_MANIFEST_PATH", DEFAULT_MANIFEST_PATH
    )
    resolved_batch_limit = _configured_batch_limit(batch_limit)

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        try:
            loaded = load_frozen_model(
                resolved_model_path,
                resolved_manifest_path,
                strict_integrity=True,
            )
            _verify_startup_contract(loaded, resolved_model_path)
            public_metadata = safe_model_metadata(resolved_manifest_path)
        except Exception:
            logger.exception("CreditScope API startup model verification failed.")
            raise
        application.state.creditscope = SimpleNamespace(
            model=loaded,
            metadata=public_metadata,
            artifact_verified=True,
            batch_limit=resolved_batch_limit,
        )
        logger.info("CreditScope API started with verified model %s.", MODEL_VERSION)
        yield

    application = FastAPI(
        title=SERVICE_NAME,
        description=(
            "Educational/research credit-risk decision-support prototype. "
            "It estimates good/bad credit-risk classification and does not make "
            "lending decisions."
        ),
        version=API_VERSION,
        lifespan=lifespan,
    )

    @application.middleware("http")
    async def reject_duplicate_json_keys(request: Request, call_next: Any):
        """Apply the Stage 9 duplicate-key rule before JSON decoding loses keys."""
        if request.method == "POST" and request.url.path in {"/predict", "/predict-batch"}:
            try:
                json.loads(await request.body(), object_pairs_hook=reject_duplicate_fields)
            except DuplicateFieldError as error:
                return JSONResponse(status_code=400, content={"detail": str(error)})
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass
        return await call_next(request)

    @application.exception_handler(RequestValidationError)
    async def validation_error_handler(
        _request: Request, error: RequestValidationError
    ) -> JSONResponse:
        details = [
            {"loc": item.get("loc"), "msg": item.get("msg"), "type": item.get("type")}
            for item in error.errors()
        ]
        return JSONResponse(status_code=422, content={"detail": details})

    @application.exception_handler(InferenceError)
    async def inference_error_handler(
        _request: Request, error: InferenceError
    ) -> JSONResponse:
        return JSONResponse(status_code=400, content={"detail": str(error)})

    @application.exception_handler(Exception)
    async def unexpected_error_handler(
        _request: Request, _error: Exception
    ) -> JSONResponse:
        logger.exception("Unexpected CreditScope API request failure.")
        return JSONResponse(
            status_code=500,
            content={"detail": "Unexpected internal service error."},
        )

    @application.get(
        "/",
        response_model=RootResponse,
        summary="Describe the CreditScope inference service",
    )
    def root() -> dict[str, Any]:
        return {
            "service_name": SERVICE_NAME,
            "service_version": API_VERSION,
            "model_version": MODEL_VERSION,
            "purpose": "Educational/research credit-risk decision-support prototype.",
            "documentation": "/docs",
        }

    @application.get(
        "/health",
        response_model=HealthResponse,
        summary="Report lightweight model readiness",
    )
    def health(request: Request) -> dict[str, Any]:
        _model_from_state(request)
        return {
            "status": "ok",
            "model_loaded": True,
            "model_version": MODEL_VERSION,
            "artifact_verified": request.app.state.creditscope.artifact_verified,
        }

    @application.get(
        "/model-info",
        response_model=ModelInfoResponse,
        summary="Return safe frozen-model metadata",
    )
    def model_info(request: Request) -> dict[str, Any]:
        _model_from_state(request)
        return _safe_model_info(request.app.state.creditscope.metadata)

    @application.post(
        "/predict",
        response_model=CreditRiskPrediction,
        summary="Predict calibrated bad-credit-risk probability for one record",
    )
    def predict(payload: CreditRiskInput, request: Request) -> dict[str, Any]:
        return predict_one(_model_from_state(request), payload.model_dump())

    @application.post(
        "/predict-batch",
        response_model=BatchPredictionResponse,
        summary="Predict an ordered batch of up to the configured limit",
    )
    def predict_many(
        payload: BatchPredictionRequest, request: Request
    ) -> dict[str, Any]:
        limit = request.app.state.creditscope.batch_limit
        if len(payload.root) > limit:
            raise HTTPException(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                detail=f"Batch exceeds the {limit}-record limit.",
            )
        records = [record.model_dump() for record in payload.root]
        predictions = predict_batch(_model_from_state(request), records)
        return predictions

    return application


app = create_app()
