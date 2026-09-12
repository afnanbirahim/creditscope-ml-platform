"""HTTP-only client for the CreditScope Stage 10 service."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from typing import Any

import httpx

DEFAULT_API_URL = "http://127.0.0.1:8000"
DEFAULT_TIMEOUT_SECONDS = 5.0


class APIClientError(RuntimeError):
    """A sanitized, user-presentable API communication failure."""

    def __init__(self, user_message: str, *, status_code: int | None = None) -> None:
        super().__init__(user_message)
        self.user_message = user_message
        self.status_code = status_code


class CreditScopeAPIClient:
    """Small synchronous client with no model or decision-policy logic."""

    def __init__(
        self,
        base_url: str | None = None,
        *,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        http_client: httpx.Client | None = None,
    ) -> None:
        configured_url = base_url or os.getenv("CREDITSCOPE_API_URL", DEFAULT_API_URL)
        self.base_url = configured_url.rstrip("/")
        self._client = http_client or httpx.Client(
            base_url=self.base_url,
            timeout=timeout_seconds,
        )
        self._owns_client = http_client is None

    def close(self) -> None:
        """Close an internally created HTTP connection pool."""
        if self._owns_client:
            self._client.close()

    def health(self) -> dict[str, Any]:
        """Return service readiness established by API startup."""
        return self._request("GET", "/health")

    def model_info(self) -> dict[str, Any]:
        """Return safe public model metadata from the API."""
        return self._request("GET", "/model-info")

    def predict(self, record: Mapping[str, Any]) -> dict[str, Any]:
        """Submit one record to the authoritative API prediction route."""
        return self._request("POST", "/predict", json=dict(record))

    def predict_batch(
        self, records: Sequence[Mapping[str, Any]]
    ) -> list[dict[str, Any]]:
        """Submit an ordered batch without local validation or scoring."""
        result = self._request(
            "POST", "/predict-batch", json=[dict(record) for record in records]
        )
        if not isinstance(result, list):
            raise APIClientError("CreditScope returned an unexpected batch response.")
        return result

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        try:
            response = self._client.request(method, path, **kwargs)
        except httpx.TimeoutException as error:
            raise APIClientError(
                "CreditScope prediction service timed out. Please try again."
            ) from error
        except httpx.RequestError as error:
            raise APIClientError(
                "CreditScope prediction service is unavailable. Start the FastAPI "
                "service first."
            ) from error

        if response.is_success:
            try:
                return response.json()
            except ValueError as error:
                raise APIClientError(
                    "CreditScope returned an unreadable response."
                ) from error

        messages = {
            400: "The prediction request was rejected by the service.",
            413: "The batch exceeds the service's maximum size.",
            422: "Some input values were rejected by the service.",
            500: "The prediction service encountered an internal error.",
            503: "CreditScope prediction service is unavailable. Start the FastAPI service first.",
        }
        message = messages.get(
            response.status_code,
            f"CreditScope request failed with HTTP status {response.status_code}.",
        )
        raise APIClientError(message, status_code=response.status_code)
