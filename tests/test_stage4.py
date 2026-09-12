"""Tests for Stage 4 tree pipelines and paired development evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from creditscope.evaluation import evaluate_cross_validation, make_cv_splits
from creditscope.modeling import (
    AUDIT_ATTRIBUTES,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    PREDICTIVE_FEATURES,
    RANDOM_FOREST_PARAMETERS,
    XGBOOST_PARAMETERS,
    build_random_forest_pipeline,
    build_tree_preprocessor,
    build_xgboost_pipeline,
)
from creditscope.schema import FEATURE_SPECS
from creditscope.stage4 import (
    FLOAT_METRIC_ATOL,
    feature_importance_table,
    fold_membership_table,
    oof_prediction_table,
    verify_logistic_reference,
)


def make_model_data(rows: int = 200) -> tuple[pd.DataFrame, pd.Series]:
    """Build deterministic schema-shaped predictors for isolated tests."""
    values: dict[str, list[object]] = {}
    for offset, spec in enumerate(FEATURE_SPECS, start=1):
        if spec.name not in PREDICTIVE_FEATURES:
            continue
        if spec.categories:
            codes = [code for code, _ in spec.categories]
            values[spec.name] = [codes[index % len(codes)] for index in range(rows)]
        else:
            values[spec.name] = [offset + index % 11 for index in range(rows)]
    X = pd.DataFrame(values)[list(PREDICTIVE_FEATURES)]
    y = pd.Series([0] * 140 + [1] * 60, name="bad_credit_risk", dtype="int8")
    return X, y


def test_tree_preprocessor_uses_corrected_semantic_routes_without_scaling() -> None:
    preprocessor = build_tree_preprocessor()
    columns = {name: tuple(fields) for name, _, fields in preprocessor.transformers}
    assert columns["numerical"] == NUMERICAL_FEATURES
    assert columns["categorical"] == CATEGORICAL_FEATURES
    assert not set(AUDIT_ATTRIBUTES) & set(PREDICTIVE_FEATURES)

    numeric = preprocessor.transformers[0][1]
    categorical = preprocessor.transformers[1][1]
    assert not any(isinstance(step, StandardScaler) for step in numeric.named_steps.values())
    assert isinstance(categorical.named_steps["encoder"], OneHotEncoder)
    assert categorical.named_steps["encoder"].handle_unknown == "ignore"


def test_tree_preprocessor_handles_unknown_categories() -> None:
    X, _ = make_model_data()
    fitted = build_tree_preprocessor().fit(X)
    unknown = X.iloc[[0]].copy()
    unknown.loc[unknown.index[0], CATEGORICAL_FEATURES[0]] = "UNSEEN"
    transformed = fitted.transform(unknown)
    assert transformed.shape[0] == 1
    values = transformed.toarray() if hasattr(transformed, "toarray") else transformed
    assert np.isfinite(values).all()


def test_random_forest_configuration_is_fixed_and_unweighted() -> None:
    pipeline = build_random_forest_pipeline()
    classifier = pipeline.named_steps["classifier"]
    for parameter, value in RANDOM_FOREST_PARAMETERS.items():
        assert classifier.get_params()[parameter] == value
    assert classifier.class_weight is None


def test_xgboost_configuration_is_fixed_and_unweighted() -> None:
    pytest.importorskip("xgboost")
    pipeline = build_xgboost_pipeline()
    parameters = pipeline.named_steps["classifier"].get_params()
    for parameter, value in XGBOOST_PARAMETERS.items():
        assert parameters[parameter] == value
    assert parameters.get("scale_pos_weight") in (None, 1)
    assert parameters.get("early_stopping_rounds") is None


def test_paired_folds_cover_each_row_once_and_oof_probabilities_are_valid() -> None:
    X, y = make_model_data()
    first = make_cv_splits(X, y)
    second = make_cv_splits(X, y)
    for (first_train, first_validation), (second_train, second_validation) in zip(
        first, second, strict=True
    ):
        np.testing.assert_array_equal(first_train, second_train)
        np.testing.assert_array_equal(first_validation, second_validation)

    validation_positions = np.concatenate([validation for _, validation in first])
    assert sorted(validation_positions.tolist()) == list(range(len(X)))

    pipeline = build_random_forest_pipeline()
    pipeline.named_steps["classifier"].set_params(n_estimators=10, n_jobs=1)
    result = evaluate_cross_validation(
        "Random Forest test fixture",
        pipeline,
        X,
        y,
        cv_splits=first,
    )
    assert result.oof_probabilities.notna().all()
    assert result.oof_probabilities.between(0, 1).all()
    assert result.oof_predictions.index.equals(y.index)


def test_tree_feature_names_align_with_importance_values() -> None:
    X, y = make_model_data()
    pipeline = build_random_forest_pipeline()
    pipeline.named_steps["classifier"].set_params(n_estimators=10, n_jobs=1)
    fitted = pipeline.fit(X, y)
    importance = feature_importance_table(fitted)
    assert len(importance) == len(
        fitted.named_steps["preprocessor"].get_feature_names_out()
    )
    assert set(importance["source_feature"]) <= set(PREDICTIVE_FEATURES)
    assert importance["importance"].between(0, 1).all()
    assert importance["importance_rank"].tolist() == list(
        range(1, len(importance) + 1)
    )


def test_reconciliation_policy_accepts_small_float_drift_and_requires_exact_counts(
    tmp_path,
) -> None:
    columns = {
        "model": ["Logistic Regression"],
        "roc_auc": [0.75],
        "average_precision": [0.55],
        "precision_bad_credit_risk": [0.5],
        "recall_bad_credit_risk": [0.4],
        "f1_bad_credit_risk": [0.44],
        "balanced_accuracy": [0.61],
        "brier_score": [0.18],
        "accuracy": [0.72],
        "true_negative": [480],
        "false_positive": [80],
        "false_negative": [144],
        "true_positive": [96],
        "total_cost": [800],
        "average_cost": [1.0],
    }
    report_dir = tmp_path / "reports" / "stage3_remediated"
    report_dir.mkdir(parents=True)
    pd.DataFrame(columns).to_csv(report_dir / "baseline_comparison.csv", index=False)
    fold = pd.DataFrame(
        {
            "model": ["Logistic Regression"],
            "fold": [1],
            "training_rows": [640],
            "validation_rows": [160],
            "validation_good_credit_risk": [112],
            "validation_bad_credit_risk": [48],
            "roc_auc": [0.75],
            "average_precision": [0.55],
            "precision_bad_credit_risk": [0.5],
            "recall_bad_credit_risk": [0.4],
            "f1_bad_credit_risk": [0.44],
            "balanced_accuracy": [0.61],
            "brier_score": [0.18],
            "accuracy": [0.72],
            "default_threshold_total_cost": [160],
            "default_threshold_average_cost": [1.0],
        }
    )
    fold.to_csv(report_dir / "fold_metrics.csv", index=False)
    observed = pd.DataFrame(columns)
    observed.loc[0, "brier_score"] += FLOAT_METRIC_ATOL / 2
    observed_fold = fold.copy()
    observed_fold.loc[0, "brier_score"] += FLOAT_METRIC_ATOL / 2
    assert verify_logistic_reference(tmp_path, observed, observed_fold)[
        "floating_metrics_match"
    ]

    observed.loc[0, "brier_score"] = 0.18 + FLOAT_METRIC_ATOL * 2
    with pytest.raises(ValueError):
        verify_logistic_reference(tmp_path, observed, observed_fold)

    observed = pd.DataFrame(columns)
    observed.loc[0, "false_negative"] += 1
    with pytest.raises(ValueError):
        verify_logistic_reference(tmp_path, observed, fold)


def test_fold_and_oof_evidence_are_complete_and_development_only() -> None:
    X, y = make_model_data()
    folds = make_cv_splits(X, y)
    membership = fold_membership_table(X.index, folds)
    pipeline = build_random_forest_pipeline()
    pipeline.named_steps["classifier"].set_params(n_estimators=10, n_jobs=1)
    result = evaluate_cross_validation(
        "Random Forest", pipeline, X, y, cv_splits=folds
    )
    result = type(result)(
        model="Logistic Regression",
        fold_metrics=result.fold_metrics,
        oof_probabilities=result.oof_probabilities,
        oof_predictions=result.oof_predictions,
    )
    evidence = oof_prediction_table(y, membership, [result])
    assert len(membership) == len(X)
    assert membership["fold"].between(1, 5).all()
    assert len(evidence) == len(X)
    assert evidence["source_row_id"].is_unique
    assert evidence["logistic_regression_probability"].between(0, 1).all()
