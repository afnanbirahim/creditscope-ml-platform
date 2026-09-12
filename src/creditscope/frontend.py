"""Presentation-only schema and formatting helpers for the Streamlit UI."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

from creditscope.schema import CATEGORY_LABELS


@dataclass(frozen=True)
class FrontendField:
    """Human-readable UI metadata for one canonical API field."""

    name: str
    label: str
    kind: Literal["number", "category"]
    help_text: str
    default: float | int | str
    options: tuple[float | int | str, ...] = ()
    option_labels: dict[float | int | str, str] | None = None


def _coded_options(name: str) -> tuple[tuple[str, ...], dict[str, str]]:
    labels = dict(CATEGORY_LABELS[name])
    return tuple(labels), {code: f"{meaning} ({code})" for code, meaning in labels.items()}


def _levels(maximum: int) -> tuple[tuple[int, ...], dict[int, str]]:
    values = tuple(range(1, maximum + 1))
    return values, {value: f"Source level {value} ({value})" for value in values}


_CHECKING_OPTIONS, _CHECKING_LABELS = _coded_options("checking_account_status")
_HISTORY_OPTIONS, _HISTORY_LABELS = _coded_options("credit_history")
_PURPOSE_OPTIONS, _PURPOSE_LABELS = _coded_options("purpose")
_SAVINGS_OPTIONS, _SAVINGS_LABELS = _coded_options("savings_account_status")
_EMPLOYMENT_OPTIONS, _EMPLOYMENT_LABELS = _coded_options("employment_duration")
_DEBTOR_OPTIONS, _DEBTOR_LABELS = _coded_options("other_debtors_guarantors")
_PROPERTY_OPTIONS, _PROPERTY_LABELS = _coded_options("property")
_PLAN_OPTIONS, _PLAN_LABELS = _coded_options("other_installment_plans")
_HOUSING_OPTIONS, _HOUSING_LABELS = _coded_options("housing")
_JOB_OPTIONS, _JOB_LABELS = _coded_options("job")
_PHONE_OPTIONS, _PHONE_LABELS = _coded_options("telephone")
_FOUR_LEVELS, _FOUR_LEVEL_LABELS = _levels(4)
_TWO_LEVELS, _TWO_LEVEL_LABELS = _levels(2)

UI_FIELDS = (
    FrontendField(
        "duration_months",
        "Duration (months)",
        "number",
        "Credit duration in months under the historical source definition.",
        12.0,
    ),
    FrontendField(
        "credit_amount",
        "Credit amount (historical transformed quantity)",
        "number",
        "Not a literal contemporary currency amount; the source transformation is undocumented.",
        1000.0,
    ),
    FrontendField("checking_account_status", "Checking account status", "category", "Historical source category.", "A11", _CHECKING_OPTIONS, _CHECKING_LABELS),
    FrontendField("credit_history", "Credit history", "category", "Historical source category.", "A32", _HISTORY_OPTIONS, _HISTORY_LABELS),
    FrontendField("purpose", "Credit purpose", "category", "Historical source category.", "A40", _PURPOSE_OPTIONS, _PURPOSE_LABELS),
    FrontendField("savings_account_status", "Savings account status", "category", "Historical source category.", "A61", _SAVINGS_OPTIONS, _SAVINGS_LABELS),
    FrontendField("employment_duration", "Employment duration", "category", "Discretized duration category; codes are not numeric magnitudes.", "A73", _EMPLOYMENT_OPTIONS, _EMPLOYMENT_LABELS),
    FrontendField("installment_rate_percent", "Installment-rate category", "category", "Historical discretized category, not a literal percentage.", 2, _FOUR_LEVELS, _FOUR_LEVEL_LABELS),
    FrontendField("other_debtors_guarantors", "Other debtors or guarantors", "category", "Historical source category.", "A101", _DEBTOR_OPTIONS, _DEBTOR_LABELS),
    FrontendField("residence_duration", "Residence-duration category", "category", "Historical discretized duration level.", 2, _FOUR_LEVELS, _FOUR_LEVEL_LABELS),
    FrontendField("property", "Property category", "category", "Historical source category; unknown/no property is substantive.", "A121", _PROPERTY_OPTIONS, _PROPERTY_LABELS),
    FrontendField("other_installment_plans", "Other installment plans", "category", "Historical source category.", "A143", _PLAN_OPTIONS, _PLAN_LABELS),
    FrontendField("housing", "Housing status", "category", "Historical source category.", "A152", _HOUSING_OPTIONS, _HOUSING_LABELS),
    FrontendField("existing_credits_count", "Existing-credit-count category", "category", "Discretized source level, not an unrestricted count.", 1, _FOUR_LEVELS, _FOUR_LEVEL_LABELS),
    FrontendField("job", "Job category", "category", "Ordinal source concept represented categorically by the model.", "A173", _JOB_OPTIONS, _JOB_LABELS),
    FrontendField("dependents_count", "Dependents-count category", "category", "Binary/discretized source level, not an unrestricted count.", 1, _TWO_LEVELS, _TWO_LEVEL_LABELS),
    FrontendField("telephone", "Telephone status", "category", "Historical source category.", "A191", _PHONE_OPTIONS, _PHONE_LABELS),
)

UI_FIELD_NAMES = tuple(field.name for field in UI_FIELDS)


def build_payload(values: dict[str, Any]) -> dict[str, Any]:
    """Order an exact UI payload without adding policy or audit fields."""
    missing = [name for name in UI_FIELD_NAMES if name not in values]
    unexpected = sorted(set(values) - set(UI_FIELD_NAMES))
    if missing or unexpected:
        raise ValueError(
            f"Frontend field mismatch; missing={missing}, unexpected={unexpected}."
        )
    return {name: values[name] for name in UI_FIELD_NAMES}


def format_probability(probability: float) -> str:
    """Format an API probability for display without changing the source value."""
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("Probability must be finite and within [0, 1].")
    return f"{probability:.1%}"


def display_label(api_label: str) -> str:
    """Translate only the two API labels into neutral readable text."""
    labels = {
        "good_credit_risk": "Good credit risk",
        "bad_credit_risk": "Bad credit risk",
    }
    if api_label not in labels:
        raise ValueError("Unexpected prediction label returned by the API.")
    return labels[api_label]


def presentation_values(prediction: dict[str, Any]) -> dict[str, Any]:
    """Build display strings while retaining the unchanged API probability."""
    probability = float(prediction["probability_bad_risk"])
    threshold = float(prediction["threshold"])
    return {
        "label": display_label(str(prediction["predicted_label"])),
        "probability": probability,
        "probability_display": format_probability(probability),
        "threshold_display": format_probability(threshold),
        "model_version": str(prediction["model_version"]),
    }
