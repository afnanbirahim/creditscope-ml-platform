"""Streamlit frontend for the existing CreditScope FastAPI service."""

from __future__ import annotations

import os
from typing import Any

import streamlit as st

from creditscope.api_client import DEFAULT_API_URL, APIClientError, CreditScopeAPIClient
from creditscope.frontend import (
    UI_FIELD_NAMES,
    UI_FIELDS,
    build_payload,
    format_probability,
    presentation_values,
)


@st.cache_resource
def get_api_client(base_url: str) -> CreditScopeAPIClient:
    """Reuse one HTTP connection pool for the Streamlit process."""
    return CreditScopeAPIClient(base_url)


def _render_limitations() -> None:
    with st.expander("Model limitations and responsible use"):
        st.markdown(
            """
- The source is a small historical German credit-risk dataset and may not
  represent contemporary populations or lending contexts.
- Socioeconomic proxy concerns remain for several predictors. The audit-only
  attributes `personal_status_sex`, `age_years`, and `foreign_worker` are not
  requested and do not enter prediction.
- Subgroup diagnostics cannot establish fairness. Model associations and
  post-hoc explanations are not causal.
- This prototype does not approve or reject loans, determine eligibility,
  replace human review, or provide financial advice.

The repository model card is at `reports/stage8/model_card.md`. Its SHAP audit
explains underlying XGBoost raw scores, not exact calibrated probabilities.
            """
        )


def _render_input(field: Any) -> float | int | str:
    if field.kind == "number":
        return st.number_input(
            field.label,
            value=float(field.default),
            step=1.0,
            help=field.help_text,
            key=field.name,
        )
    labels = field.option_labels or {}
    return st.selectbox(
        field.label,
        options=field.options,
        index=field.options.index(field.default),
        format_func=lambda value: labels.get(value, str(value)),
        help=field.help_text,
        key=field.name,
    )


def _render_result(prediction: dict[str, Any]) -> None:
    presented = presentation_values(prediction)
    with st.container(border=True):
        st.subheader("Model result")
        st.write(f"**Model classification:** {presented['label']}")
        st.write(
            "**Estimated bad-credit-risk probability:** "
            f"{presented['probability_display']}"
        )
        st.write(
            "**Frozen decision-support threshold:** "
            f"{presented['threshold_display']}"
        )
        st.caption(f"Model version: {presented['model_version']}")
        st.caption("This probability is a model estimate, not statistical certainty.")


def main() -> None:
    """Render the HTTP-only CreditScope portfolio interface."""
    st.set_page_config(page_title="CreditScope", page_icon="🔎", layout="wide")
    st.title("CreditScope")
    st.markdown(
        "Educational/research credit-risk decision-support prototype. "
        "It does not approve or reject loans, determine eligibility, replace "
        "human review, or provide financial advice."
    )

    api_url = os.getenv("CREDITSCOPE_API_URL", DEFAULT_API_URL)
    client = get_api_client(api_url)
    try:
        health = client.health()
        model_info = client.model_info()
    except APIClientError as error:
        st.error(error.user_message)
        st.info(
            "Start the FastAPI service first: `python -m uvicorn "
            "creditscope.api:app --host 127.0.0.1 --port 8000`."
        )
        _render_limitations()
        return

    status_column, model_column = st.columns(2)
    status_column.metric("Service", "Available")
    model_column.metric("Model", str(health["model_version"]))

    with st.expander("Verified model information"):
        st.write(f"**Model family:** {model_info['model_family']}")
        st.write(f"**Calibration:** {model_info['calibration_method']}")
        st.write(f"**Threshold:** {format_probability(float(model_info['threshold']))}")
        st.write(f"**Predictors:** {model_info['predictor_count']}")
        st.write(f"**Target semantics:** {model_info['target_semantics']}")

    api_predictors = tuple(model_info["predictor_names"])
    if api_predictors != UI_FIELD_NAMES:
        st.error("Frontend field contract does not match the active API model.")
        _render_limitations()
        return

    st.header("Credit-risk classification input")
    st.caption("All fields use the historical UCI source definitions shown in the controls.")
    with st.form("credit-risk-input", clear_on_submit=False):
        left, right = st.columns(2)
        values: dict[str, Any] = {}
        for index, field in enumerate(UI_FIELDS):
            with (left if index % 2 == 0 else right):
                values[field.name] = _render_input(field)
        submitted = st.form_submit_button("Estimate credit risk", type="primary")

    if submitted:
        try:
            prediction = client.predict(build_payload(values))
            _render_result(prediction)
        except (APIClientError, ValueError, KeyError, TypeError) as error:
            message = (
                error.user_message
                if isinstance(error, APIClientError)
                else "The service returned an unexpected prediction response."
            )
            st.error(message)

    st.info(
        "The frozen threshold reflects an assumed 5:1 relative cost of missing "
        "bad-risk cases versus flagging good-risk cases. This is an educational "
        "modelling assumption, not established real-world lending economics."
    )
    _render_limitations()


if __name__ == "__main__":
    main()
