"""Pre-access tests for the frozen Stage 7 holdout evaluator."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV

from creditscope import stage7
from creditscope.modeling import (
    AUDIT_ATTRIBUTES,
    PREDICTIVE_FEATURES,
    XGBOOST_PARAMETERS,
)
from creditscope.stage7 import (
    FINAL_METRIC_NAMES,
    FROZEN_THRESHOLD,
    LOCKED_ARTIFACTS,
    PREDICTION_COLUMNS,
    build_final_estimator,
    final_metric_record,
    make_prediction_artifact,
    membership_summary,
    validate_frozen_policy,
    validate_prediction_artifact,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_frozen_policy_hash_and_contents_are_exact() -> None:
    policy_path = PROJECT_ROOT / LOCKED_ARTIFACTS["frozen_policy"][0]
    assert hashlib.sha256(policy_path.read_bytes()).hexdigest() == LOCKED_ARTIFACTS[
        "frozen_policy"
    ][1]
    policy = validate_frozen_policy(PROJECT_ROOT)
    assert policy["model"]["hyperparameters"] == XGBOOST_PARAMETERS
    assert policy["calibration"]["method"] == "sigmoid"
    assert policy["calibration"]["ensemble"] is True
    assert policy["calibration"]["n_jobs"] is None
    assert policy["decision_threshold"] == 0.16


def test_final_estimator_is_exact_five_fold_sigmoid_ensemble() -> None:
    folds = tuple(
        (
            np.array([position for position in range(10) if position % 5 != fold]),
            np.array([position for position in range(10) if position % 5 == fold]),
        )
        for fold in range(5)
    )
    estimator = build_final_estimator(folds)
    assert isinstance(estimator, CalibratedClassifierCV)
    assert estimator.method == "sigmoid"
    assert estimator.cv is folds
    assert estimator.ensemble is True
    assert estimator.n_jobs is None
    classifier = estimator.estimator.named_steps["classifier"]
    for name, value in XGBOOST_PARAMETERS.items():
        assert classifier.get_params()[name] == value


def test_metric_set_and_cost_orientation_are_frozen() -> None:
    y = pd.Series([0, 0, 1, 1])
    probabilities = pd.Series([0.1, 0.8, 0.2, 0.9])
    predictions = pd.Series([0, 1, 0, 1])
    metrics = final_metric_record(y, probabilities, predictions)
    assert tuple(metrics) == FINAL_METRIC_NAMES
    assert metrics["false_positive"] == 1
    assert metrics["false_negative"] == 1
    assert metrics["total_cost"] == 6


def test_prediction_schema_threshold_and_probability_validation() -> None:
    holdout_ids = pd.Index([f"holdout-{index:03d}" for index in range(200)])
    development_ids = pd.Index([f"development-{index:03d}" for index in range(800)])
    y = pd.Series(np.tile([0, 1], 100), index=holdout_ids, dtype="int8")
    probabilities = np.linspace(0.0, 1.0, 200)
    artifact = make_prediction_artifact(holdout_ids, y, probabilities)
    assert tuple(artifact.columns) == PREDICTION_COLUMNS
    assert artifact["frozen_threshold"].eq(FROZEN_THRESHOLD).all()
    assert set(artifact["final_prediction"]) <= {0, 1}
    validate_prediction_artifact(artifact, development_ids, holdout_ids)


def test_locked_membership_is_unique_and_disjoint() -> None:
    summary = membership_summary(PROJECT_ROOT)
    assert summary["development_rows"] == 800
    assert summary["holdout_rows"] == 200
    assert summary["unique_development_ids"] == 800
    assert summary["unique_holdout_ids"] == 200
    assert summary["overlap"] == 0


def test_no_alternative_policy_evaluation_path_exists() -> None:
    assert FROZEN_THRESHOLD == 0.16
    assert not hasattr(stage7, "threshold_search_table")
    assert not hasattr(stage7, "build_logistic_pipeline")
    assert not hasattr(stage7, "build_random_forest_pipeline")
    assert not hasattr(stage7, "build_tuning_pipeline")
    assert not set(AUDIT_ATTRIBUTES) & set(PREDICTIVE_FEATURES)
