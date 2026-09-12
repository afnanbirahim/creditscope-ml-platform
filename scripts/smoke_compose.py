"""Engineering smoke test for a running local CreditScope Compose stack."""

from __future__ import annotations

import json
import os
from pathlib import Path

import httpx

PROBABILITY_ATOL = 1e-12


def main() -> None:
    """Validate API/UI health and committed single/batch sample parity."""
    root = Path(__file__).resolve().parents[1]
    api_url = os.getenv("CREDITSCOPE_SMOKE_API_URL", "http://127.0.0.1:8000")
    ui_url = os.getenv("CREDITSCOPE_SMOKE_UI_URL", "http://127.0.0.1:8501")
    request = json.loads((root / "examples/api_single_request.json").read_text())
    expected = json.loads((root / "examples/api_single_response.json").read_text())
    batch_request = json.loads(
        (root / "examples/api_batch_request.json").read_text()
    )
    batch_expected = json.loads(
        (root / "examples/api_batch_response.json").read_text()
    )

    with httpx.Client(timeout=15.0) as client:
        health = client.get(f"{api_url}/health")
        health.raise_for_status()
        if health.json().get("status") != "ok":
            raise RuntimeError("API health response is not ready.")
        model_info = client.get(f"{api_url}/model-info")
        model_info.raise_for_status()
        if model_info.json().get("model_version") != "creditscope-model-1.0.0":
            raise RuntimeError("API model version changed.")
        actual = client.post(f"{api_url}/predict", json=request)
        actual.raise_for_status()
        actual_payload = actual.json()
        if abs(
            actual_payload["probability_bad_risk"]
            - expected["probability_bad_risk"]
        ) > PROBABILITY_ATOL:
            raise RuntimeError("API golden probability exceeded tolerance.")
        for field in ("predicted_class", "predicted_label", "threshold", "model_version"):
            if actual_payload[field] != expected[field]:
                raise RuntimeError(f"API golden {field} changed.")
        batch = client.post(f"{api_url}/predict-batch", json=batch_request)
        batch.raise_for_status()
        actual_predictions = batch.json()
        expected_predictions = batch_expected
        if len(actual_predictions) != len(expected_predictions):
            raise RuntimeError("API batch response length changed.")
        for actual_item, expected_item in zip(
            actual_predictions, expected_predictions, strict=True
        ):
            if abs(
                actual_item["probability_bad_risk"]
                - expected_item["probability_bad_risk"]
            ) > PROBABILITY_ATOL:
                raise RuntimeError("API batch probability exceeded tolerance.")
            for field in (
                "predicted_class",
                "predicted_label",
                "threshold",
                "model_version",
            ):
                if actual_item[field] != expected_item[field]:
                    raise RuntimeError(f"API batch golden {field} changed.")
        ui_health = client.get(f"{ui_url}/_stcore/health")
        ui_health.raise_for_status()
        if ui_health.text.strip() != "ok":
            raise RuntimeError("Streamlit health response is not ready.")

    print("Compose smoke test passed: API/UI health and golden endpoint parity.")


if __name__ == "__main__":
    main()
