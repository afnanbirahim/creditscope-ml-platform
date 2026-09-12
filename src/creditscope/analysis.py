"""Reusable construction and validation of the Stage 2 analysis dataset."""

from __future__ import annotations

from typing import Any

import pandas as pd

from creditscope.data import (
    EXPECTED_FEATURES,
    EXPECTED_ROWS,
    fetch_official_dataset,
    validate_dataset,
)
from creditscope.schema import (
    CATEGORICAL_SPECS,
    FEATURE_NAME_MAP,
    TARGET_BINARY_MAP,
    TARGET_BINARY_NAME,
    TARGET_ORIGINAL_NAME,
)


def validate_categorical_domains(data: pd.DataFrame) -> None:
    """Reject categorical values not documented in UCI's official codebook."""
    for spec in CATEGORICAL_SPECS:
        allowed = {code for code, _ in spec.categories}
        observed = set(data[spec.name].dropna().astype(str).unique())
        unexpected = observed - allowed
        if unexpected:
            raise ValueError(
                f"Unexpected values for {spec.name}: {sorted(unexpected)}; "
                f"allowed values are {sorted(allowed)}."
            )


def validate_analysis_dataset(
    data: pd.DataFrame,
    *,
    expected_rows: int = EXPECTED_ROWS,
) -> None:
    """Validate the cleaned feature schema and both target representations."""
    expected_feature_names = list(FEATURE_NAME_MAP.values())
    expected_columns = expected_feature_names + [
        TARGET_ORIGINAL_NAME,
        TARGET_BINARY_NAME,
    ]
    if data.shape != (expected_rows, EXPECTED_FEATURES + 2):
        raise ValueError(
            f"Expected analysis shape {(expected_rows, EXPECTED_FEATURES + 2)}, "
            f"received {data.shape}."
        )
    if list(data.columns) != expected_columns:
        raise ValueError("Analysis columns do not match the documented schema.")
    if set(data[TARGET_ORIGINAL_NAME].unique()) != {1, 2}:
        raise ValueError(
            "credit_risk_class must contain exactly source values 1 and 2."
        )
    if set(data[TARGET_BINARY_NAME].unique()) != {0, 1}:
        raise ValueError("bad_credit_risk must contain exactly binary values 0 and 1.")
    expected_binary = data[TARGET_ORIGINAL_NAME].map(TARGET_BINARY_MAP)
    if not expected_binary.equals(data[TARGET_BINARY_NAME].astype("int64")):
        raise ValueError("Binary target is inconsistent with the source target.")
    if data.isna().any().any():
        raise ValueError("Analysis dataset contains missing values.")
    validate_categorical_domains(data)


def build_analysis_dataset(
    dataset: Any | None = None,
    *,
    expected_rows: int = EXPECTED_ROWS,
) -> pd.DataFrame:
    """Build a non-mutating analysis view from the verified official source."""
    source = dataset if dataset is not None else fetch_official_dataset()
    features, targets = validate_dataset(source, expected_rows=expected_rows)

    if set(features.columns) != set(FEATURE_NAME_MAP):
        raise ValueError("Source feature names do not match the UCI 144 schema.")

    analysis = features.rename(columns=FEATURE_NAME_MAP).copy(deep=True)
    source_target = targets.iloc[:, 0].astype("int64")
    analysis[TARGET_ORIGINAL_NAME] = source_target.to_numpy()
    analysis[TARGET_BINARY_NAME] = (
        source_target.map(TARGET_BINARY_MAP).astype("int8").to_numpy()
    )
    validate_analysis_dataset(analysis, expected_rows=expected_rows)
    return analysis
