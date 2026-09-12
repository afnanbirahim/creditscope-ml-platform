"""Tests for Stage 8 explainability and responsible-AI audit helpers."""

from pathlib import Path

import numpy as np
import pandas as pd

from creditscope.modeling import AUDIT_ATTRIBUTES, PREDICTIVE_FEATURES
from creditscope.stage8 import (
    AGE_BINS,
    AGE_LABELS,
    EXPECTED_HASHES,
    FROZEN_THRESHOLD,
    aggregate_original_importance,
    original_feature_name,
    select_local_cases,
    subgroup_metrics,
    verify_locked_evidence,
)


def test_frozen_stage8_boundaries() -> None:
    """Stage 8 keeps the governed feature and threshold boundaries."""
    assert len(PREDICTIVE_FEATURES) == 17
    assert set(PREDICTIVE_FEATURES).isdisjoint(AUDIT_ATTRIBUTES)
    assert FROZEN_THRESHOLD == 0.16
    assert AGE_BINS == (-np.inf, 24, 34, 44, 54, np.inf)
    assert AGE_LABELS == ("under 25", "25–34", "35–44", "45–54", "55+")


def test_locked_artifacts_remain_exact() -> None:
    """Stage 8 refuses changed Stage 6P/7 evidence."""
    root = Path(__file__).resolve().parents[1]
    observed = verify_locked_evidence(root)
    assert observed == {
        name: expected for name, (_, expected) in EXPECTED_HASHES.items()
    }


def test_encoded_feature_mapping() -> None:
    """Numeric and one-hot names map to the governed original variables."""
    assert original_feature_name("numerical__credit_amount") == "credit_amount"
    assert (
        original_feature_name("categorical__checking_account_status_A11")
        == "checking_account_status"
    )


def test_original_importance_aggregation_preserves_all_features() -> None:
    """Member summaries retain every original feature and a deterministic rank."""
    rows = [
        {
            "member": member,
            "original_feature": feature,
            "mean_absolute_shap": float(position + member),
            "signed_mean_shap": 0.0,
            "encoded_level_count": 1,
        }
        for member in range(1, 6)
        for position, feature in enumerate(PREDICTIVE_FEATURES)
    ]
    result = aggregate_original_importance(pd.DataFrame(rows))
    assert set(result["original_feature"]) == set(PREDICTIVE_FEATURES)
    assert result["rank"].tolist() == list(range(1, 18))


def test_subgroup_metrics_keep_undefined_values_as_na() -> None:
    """Groups without both classes do not receive invented zero rates."""
    index = pd.Index(["r1", "r2", "r3", "r4"])
    audit = pd.DataFrame(
        {
            "personal_status_sex": ["A91", "A91", "A92", "A92"],
            "age_years": [22, 26, 36, 56],
            "foreign_worker": ["A201", "A201", "A202", "A202"],
        },
        index=index,
    )
    y = pd.Series([0, 0, 1, 1], index=index)
    probabilities = pd.Series([0.1, 0.2, 0.3, 0.4], index=index)
    metrics, _ = subgroup_metrics(audit, y, probabilities)
    first = metrics[
        (metrics["audit_attribute"] == "personal_status_sex")
        & (metrics["group"] == "A91 — male: divorced/separated")
    ].iloc[0]
    assert np.isnan(first["recall_sensitivity"])
    assert np.isnan(first["false_negative_rate"])


def test_local_case_selection_is_probability_based_and_unique() -> None:
    """Case selection is fixed before SHAP inspection and does not reuse rows."""
    oof = pd.DataFrame(
        {
            "source_row_id": [f"r{i}" for i in range(1, 8)],
            "bad_credit_risk": [0, 0, 1, 0, 1, 0, 1],
            "probability": [0.01, 0.159, 0.95, 0.8, 0.05, 0.4, 0.3],
        }
    )
    selected = select_local_cases(oof)
    assert selected["source_row_id"].is_unique
    assert set(selected["selection_rule"]) == {
        "low_probability",
        "closest_to_threshold",
        "high_probability",
        "false_positive",
        "false_negative",
    }
