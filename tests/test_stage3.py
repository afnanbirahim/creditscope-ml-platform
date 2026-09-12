"""Tests for governed Stage 3 splitting, preprocessing, and evaluation."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from creditscope.evaluation import asymmetric_cost, threshold_predictions
from creditscope.modeling import (
    AUDIT_ATTRIBUTES,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    PREDICTIVE_FEATURES,
    build_logistic_pipeline,
    build_preprocessor,
    partition_model_data,
    validate_feature_specification,
)
from creditscope.schema import FEATURE_SPECS, TARGET_BINARY_NAME, TARGET_ORIGINAL_NAME
from creditscope.splits import (
    DEVELOPMENT_SPLIT,
    ROW_ID_NAME,
    TEST_SPLIT,
    create_split_manifest,
    write_locked_split,
)


def make_analysis_data(rows: int = 1_000) -> pd.DataFrame:
    """Create a schema-valid deterministic table without fetching UCI."""
    values: dict[str, list[object]] = {}
    for offset, spec in enumerate(FEATURE_SPECS, start=1):
        if spec.categories:
            codes = [code for code, _ in spec.categories]
            values[spec.name] = [codes[index % len(codes)] for index in range(rows)]
        else:
            values[spec.name] = [offset + index % 9 for index in range(rows)]
    target = np.array([0] * 700 + [1] * 300, dtype=np.int8)
    values[TARGET_ORIGINAL_NAME] = (target + 1).tolist()
    values[TARGET_BINARY_NAME] = target.tolist()
    return pd.DataFrame(values)


def test_locked_split_is_reproducible_complete_and_disjoint() -> None:
    data = make_analysis_data()
    first = create_split_manifest(data)
    second = create_split_manifest(data)
    pd.testing.assert_frame_equal(first, second)

    development = set(
        first.loc[first["split"] == DEVELOPMENT_SPLIT, ROW_ID_NAME]
    )
    test = set(first.loc[first["split"] == TEST_SPLIT, ROW_ID_NAME])
    assert len(development) == 800
    assert len(test) == 200
    assert development.isdisjoint(test)
    assert development | test == set(first[ROW_ID_NAME])
    assert set(first[TARGET_BINARY_NAME]) == {0, 1}


def test_locked_split_can_be_verified_from_disk(tmp_path) -> None:
    data = make_analysis_data()
    first, first_summary = write_locked_split(data, tmp_path)
    second, second_summary = write_locked_split(data, tmp_path)
    pd.testing.assert_frame_equal(first, second)
    assert first_summary["manifest_sha256"] == second_summary["manifest_sha256"]


def test_feature_boundary_and_audit_attributes_are_preserved() -> None:
    data = make_analysis_data()
    partitions = partition_model_data(data, create_split_manifest(data))
    validate_feature_specification()

    assert len(PREDICTIVE_FEATURES) == 17
    assert set(partitions.X_development) == set(PREDICTIVE_FEATURES)
    assert not set(AUDIT_ATTRIBUTES) & set(partitions.X_development)
    assert list(partitions.audit_development) == list(AUDIT_ATTRIBUTES)
    assert len(partitions.audit_development) == 800
    assert len(partitions.audit_test) == 200


def test_feature_typing_is_explicit_and_exhaustive() -> None:
    typed = set(NUMERICAL_FEATURES) | set(CATEGORICAL_FEATURES)
    assert typed == set(PREDICTIVE_FEATURES)
    assert NUMERICAL_FEATURES == ("duration_months", "credit_amount")
    assert not set(NUMERICAL_FEATURES) & set(CATEGORICAL_FEATURES)
    assert {
        "employment_duration",
        "installment_rate_percent",
        "residence_duration",
        "property",
        "existing_credits_count",
        "job",
        "dependents_count",
    } <= set(CATEGORICAL_FEATURES)


def test_corrected_preprocessor_routes_features_as_governed() -> None:
    preprocessor = build_preprocessor()
    columns_by_transformer = {
        name: tuple(columns) for name, _, columns in preprocessor.transformers
    }
    assert columns_by_transformer["numerical"] == NUMERICAL_FEATURES
    assert columns_by_transformer["categorical"] == CATEGORICAL_FEATURES
    assert set(columns_by_transformer) == {"numerical", "categorical"}
    categorical_pipeline = preprocessor.transformers[1][1]
    encoder = categorical_pipeline.named_steps["encoder"]
    assert isinstance(encoder, OneHotEncoder)
    assert encoder.handle_unknown == "ignore"


def test_preprocessor_is_deterministic_and_handles_unknown_categories() -> None:
    data = make_analysis_data()
    partitions = partition_model_data(data, create_split_manifest(data))
    first = build_preprocessor().fit(partitions.X_development)
    second = build_preprocessor().fit(partitions.X_development)
    transformed_first = first.transform(partitions.X_development.iloc[:10])
    transformed_second = second.transform(partitions.X_development.iloc[:10])
    np.testing.assert_allclose(
        transformed_first.toarray()
        if hasattr(transformed_first, "toarray")
        else transformed_first,
        transformed_second.toarray()
        if hasattr(transformed_second, "toarray")
        else transformed_second,
    )

    unknown = partitions.X_test.iloc[[0]].copy()
    unknown.loc[:, CATEGORICAL_FEATURES[0]] = "UNSEEN_AT_TRAINING"
    assert first.transform(unknown).shape[0] == 1


def test_preprocessor_fit_statistics_come_only_from_development_rows() -> None:
    data = make_analysis_data()
    manifest = create_split_manifest(data)
    test_positions = manifest.index[manifest["split"] == TEST_SPLIT]
    data.loc[test_positions, NUMERICAL_FEATURES[0]] = 1_000_000
    partitions = partition_model_data(data, manifest)
    fitted = build_preprocessor().fit(partitions.X_development)
    scaler = fitted.named_transformers_["numerical"].named_steps["scaler"]
    development_mean = partitions.X_development[NUMERICAL_FEATURES[0]].mean()
    assert scaler.mean_[0] == development_mean
    assert scaler.mean_[0] != data[NUMERICAL_FEATURES[0]].mean()


def test_target_orientation_cost_and_default_threshold() -> None:
    probabilities = pd.Series([0.49, 0.50, 0.90, 0.10])
    assert threshold_predictions(probabilities).tolist() == [0, 1, 1, 0]

    cost = asymmetric_cost(
        pd.Series([0, 0, 1, 1]),
        pd.Series([0, 1, 0, 1]),
    )
    assert cost["false_positive"] == 1
    assert cost["false_negative"] == 1
    assert cost["total_cost"] == 6
    assert cost["average_cost"] == 1.5


def test_logistic_pipeline_executes_without_class_weight() -> None:
    data = make_analysis_data()
    partitions = partition_model_data(data, create_split_manifest(data))
    pipeline = build_logistic_pipeline().fit(
        partitions.X_development, partitions.y_development
    )
    classifier = pipeline.named_steps["classifier"]
    assert classifier.class_weight is None
    probabilities = pipeline.predict_proba(partitions.X_development.iloc[:4])[:, 1]
    assert probabilities.shape == (4,)
    assert np.all((0 <= probabilities) & (probabilities <= 1))


def test_logistic_pipeline_is_deterministic() -> None:
    data = make_analysis_data()
    partitions = partition_model_data(data, create_split_manifest(data))
    first = build_logistic_pipeline().fit(
        partitions.X_development, partitions.y_development
    )
    second = build_logistic_pipeline().fit(
        partitions.X_development, partitions.y_development
    )
    np.testing.assert_allclose(
        first.predict_proba(partitions.X_development)[:, 1],
        second.predict_proba(partitions.X_development)[:, 1],
    )


def test_repository_split_manifest_hash_and_membership_are_locked() -> None:
    project_root = Path(__file__).resolve().parents[1]
    manifest_path = project_root / "reports" / "split_manifest.csv"
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    assert digest == "32fb4c9ac2cdb358e3bffd145cd97e1a393c14946888d46212d0b395523e99af"

    manifest = pd.read_csv(manifest_path)
    counts = manifest.groupby(["split", TARGET_BINARY_NAME]).size()
    assert counts.to_dict() == {
        (DEVELOPMENT_SPLIT, 0): 560,
        (DEVELOPMENT_SPLIT, 1): 240,
        (TEST_SPLIT, 0): 140,
        (TEST_SPLIT, 1): 60,
    }
    assert manifest[ROW_ID_NAME].is_unique
