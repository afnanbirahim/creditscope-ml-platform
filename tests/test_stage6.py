"""Tests for Stage 6 calibration and threshold governance."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from creditscope.modeling import (
    AUDIT_ATTRIBUTES,
    PREDICTIVE_FEATURES,
    XGBOOST_PARAMETERS,
)
from creditscope.stage5 import STAGE4_FOLD_SHA256, load_outer_folds
from creditscope.stage6 import (
    CALIBRATION_METHODS,
    CALIBRATION_PARSIMONY_TOLERANCE,
    LOCKED_SPLIT_SHA256,
    THEORETICAL_THRESHOLD,
    THRESHOLD_GRID,
    build_probability_estimator,
    decision_metrics,
    select_calibration_method,
    select_threshold,
    validate_training_validation_isolation,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_fixed_xgboost_and_calibration_candidate_set() -> None:
    assert CALIBRATION_METHODS == ("none", "sigmoid", "isotonic")
    uncalibrated = build_probability_estimator("none")
    for key, expected in XGBOOST_PARAMETERS.items():
        assert uncalibrated.named_steps["classifier"].get_params()[key] == expected
    sigmoid = build_probability_estimator("sigmoid")
    isotonic = build_probability_estimator("isotonic")
    assert sigmoid.method == "sigmoid"
    assert isotonic.method == "isotonic"
    assert sigmoid.ensemble is True
    assert isotonic.ensemble is True
    assert sigmoid.n_jobs is None
    assert isotonic.n_jobs is None
    with pytest.raises(ValueError, match="Unknown calibration"):
        build_probability_estimator("temperature")


def test_stage4_fold_and_split_evidence_are_exact() -> None:
    fold_path = PROJECT_ROOT / "reports" / "stage4" / "fold_membership.csv"
    split_path = PROJECT_ROOT / "reports" / "split_manifest.csv"
    assert hashlib.sha256(fold_path.read_bytes()).hexdigest() == STAGE4_FOLD_SHA256
    assert hashlib.sha256(split_path.read_bytes()).hexdigest() == LOCKED_SPLIT_SHA256

    manifest = pd.read_csv(split_path)
    development = manifest.loc[manifest["split"] == "development", "source_row_id"]
    membership, folds = load_outer_folds(PROJECT_ROOT, pd.Index(development))
    assert len(membership) == 800
    assert len(folds) == 5
    assert membership.groupby("fold").size().eq(160).all()


def test_training_validation_overlap_is_rejected() -> None:
    validate_training_validation_isolation(pd.Index([1, 2]), pd.Index([3, 4]))
    with pytest.raises(ValueError, match="overlap"):
        validate_training_validation_isolation(pd.Index([1, 2]), pd.Index([2, 3]))


def test_threshold_grid_cost_and_theoretical_reference() -> None:
    assert THRESHOLD_GRID == tuple(
        float(value) for value in np.round(np.arange(0.05, 0.51, 0.01), 2)
    )
    assert len(THRESHOLD_GRID) == 46
    assert THEORETICAL_THRESHOLD == pytest.approx(1 / 6)

    y = pd.Series([0, 0, 1, 1])
    probabilities = pd.Series([0.1, 0.8, 0.2, 0.9])
    result = decision_metrics(y, probabilities, 0.5)
    assert result["false_positive"] == 1
    assert result["false_negative"] == 1
    assert result["total_cost"] == 6


def test_calibration_selection_uses_brier_and_parsimony() -> None:
    comparison = pd.DataFrame(
        {
            "calibration_method": list(CALIBRATION_METHODS),
            "brier_score": [0.17000, 0.16995, 0.18000],
        }
    )
    assert CALIBRATION_PARSIMONY_TOLERANCE == 1e-4
    assert select_calibration_method(comparison) == "none"
    comparison.loc[1, "brier_score"] = 0.16980
    assert select_calibration_method(comparison) == "sigmoid"


def test_threshold_selection_obeys_predefined_ties() -> None:
    search = pd.DataFrame(
        {
            "threshold": THRESHOLD_GRID,
            "total_cost": 100,
            "false_negative": 10,
            "balanced_accuracy": 0.5,
        }
    )
    assert select_threshold(search) == 0.17
    search.loc[search["threshold"] == 0.22, "total_cost"] = 99
    assert select_threshold(search) == 0.22


def test_frozen_policy_contains_governed_fields_after_stage6() -> None:
    path = PROJECT_ROOT / "reports" / "stage6" / "frozen_model_policy.json"
    if not path.exists():
        pytest.skip("Stage 6 artifacts have not been generated yet.")
    policy = json.loads(path.read_text(encoding="utf-8"))
    assert policy["policy_status"] == "frozen_before_final_holdout_access"
    assert policy["model"]["hyperparameters"] == XGBOOST_PARAMETERS
    assert policy["decision_threshold"] in THRESHOLD_GRID
    assert policy["calibration"]["method"] in CALIBRATION_METHODS
    assert policy["calibration"]["ensemble"] is True
    assert policy["calibration"]["n_jobs"] is None
    assert policy["calibration"]["n_jobs_explicitly_set"] is False
    assert "five fold-specific classifier/calibrator pairs" in policy[
        "calibration"
    ]["prediction_aggregation"]
    assert policy["feature_governance"]["predictive_features"] == list(
        PREDICTIVE_FEATURES
    )
    assert set(policy["feature_governance"]["audit_only_excluded"]) == set(
        AUDIT_ATTRIBUTES
    )
    assert not set(AUDIT_ATTRIBUTES) & set(PREDICTIVE_FEATURES)


def test_stage6_oof_contains_development_rows_only_after_generation() -> None:
    path = PROJECT_ROOT / "reports" / "stage6" / "nested_policy_oof_predictions.csv"
    if not path.exists():
        pytest.skip("Stage 6 artifacts have not been generated yet.")
    oof = pd.read_csv(path)
    manifest = pd.read_csv(PROJECT_ROOT / "reports" / "split_manifest.csv")
    development = set(
        manifest.loc[manifest["split"] == "development", "source_row_id"]
    )
    holdout = set(manifest.loc[manifest["split"] == "test", "source_row_id"])
    assert len(oof) == 800
    assert oof["source_row_id"].nunique() == 800
    assert set(oof["source_row_id"]) == development
    assert set(oof["source_row_id"]).isdisjoint(holdout)
    assert oof.groupby("outer_fold").size().eq(160).all()
    assert oof["probability"].between(0.0, 1.0).all()
