"""Tests for Stage 2 schema and analysis-dataset construction."""

from types import SimpleNamespace

import pandas as pd
import pytest

from creditscope.analysis import build_analysis_dataset, validate_analysis_dataset
from creditscope.eda import data_dictionary_frame
from creditscope.schema import (
    FEATURE_NAME_MAP,
    FEATURE_SPECS,
    TARGET_BINARY_NAME,
    TARGET_ORIGINAL_NAME,
)


def make_full_schema_dataset(*, rows: int = 1_000, bad_code: bool = False):
    values: dict[str, list[object]] = {}
    for index, spec in enumerate(FEATURE_SPECS, start=1):
        if spec.categories:
            value = spec.categories[0][0]
            values[spec.uci_name] = [value] * rows
        else:
            values[spec.uci_name] = [index] * rows
    if bad_code:
        values["Attribute1"][0] = "NOT_DOCUMENTED"
    features = pd.DataFrame(values)
    targets = pd.DataFrame({"class": [1] * (rows - 1) + [2]})
    return SimpleNamespace(
        metadata={"uci_id": 144},
        data=SimpleNamespace(features=features, targets=targets),
    )


def test_clean_feature_mapping_is_complete_unique_and_snake_case() -> None:
    assert len(FEATURE_NAME_MAP) == 20
    assert len(set(FEATURE_NAME_MAP.values())) == 20
    assert list(FEATURE_NAME_MAP) == [f"Attribute{number}" for number in range(1, 21)]
    assert all(
        name.isidentifier() and name == name.lower()
        for name in FEATURE_NAME_MAP.values()
    )


def test_analysis_builder_preserves_source_and_converts_target() -> None:
    dataset = make_full_schema_dataset()
    source_before = dataset.data.features.copy(deep=True)
    analysis = build_analysis_dataset(dataset)

    pd.testing.assert_frame_equal(dataset.data.features, source_before)
    assert analysis.shape == (1_000, 22)
    assert analysis.loc[0, TARGET_ORIGINAL_NAME] == 1
    assert analysis.loc[0, TARGET_BINARY_NAME] == 0
    assert analysis.loc[999, TARGET_ORIGINAL_NAME] == 2
    assert analysis.loc[999, TARGET_BINARY_NAME] == 1


def test_analysis_builder_rejects_unofficial_categorical_code() -> None:
    with pytest.raises(
        ValueError, match="Unexpected values for checking_account_status"
    ):
        build_analysis_dataset(make_full_schema_dataset(bad_code=True))


def test_analysis_builder_enforces_row_count() -> None:
    with pytest.raises(ValueError, match="Expected feature shape"):
        build_analysis_dataset(make_full_schema_dataset(rows=999))


def test_analysis_validation_rejects_inconsistent_binary_target() -> None:
    analysis = build_analysis_dataset(make_full_schema_dataset())
    analysis.loc[0, TARGET_BINARY_NAME] = 1
    with pytest.raises(ValueError, match="Binary target is inconsistent"):
        validate_analysis_dataset(analysis)


def test_data_dictionary_includes_target_semantics() -> None:
    dictionary = data_dictionary_frame()
    target_rows = dictionary[dictionary["uci_field"] == "class"]
    assert target_rows["category_code"].tolist() == [1, 2]
    assert target_rows["official_category_meaning"].tolist() == [
        "good credit risk",
        "bad credit risk",
    ]
