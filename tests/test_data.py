"""Unit tests for Stage 1 data-integrity behavior."""

import hashlib
import json
from types import SimpleNamespace

import pandas as pd
import pytest

from creditscope.data import (
    combine_tables,
    load_verified_raw_snapshot,
    validate_dataset,
)


def make_dataset(*, dataset_id: int = 144, include_missing: bool = False):
    features = pd.DataFrame(
        {
            "feature_a": ["A", "B", "A", "B"],
            "feature_b": [1, 2, 3, 4],
        }
    )
    if include_missing:
        features.loc[0, "feature_a"] = None
    targets = pd.DataFrame({"class": [1, 1, 2, 2]})
    return SimpleNamespace(
        metadata={"uci_id": dataset_id},
        data=SimpleNamespace(features=features, targets=targets),
    )


def test_validate_dataset_accepts_valid_contract() -> None:
    features, targets = validate_dataset(
        make_dataset(), expected_rows=4, expected_features=2
    )
    assert features.shape == (4, 2)
    assert targets.shape == (4, 1)


def test_validate_dataset_rejects_wrong_source() -> None:
    with pytest.raises(ValueError, match="Expected UCI dataset 144"):
        validate_dataset(
            make_dataset(dataset_id=999), expected_rows=4, expected_features=2
        )


def test_validate_dataset_rejects_missing_values() -> None:
    with pytest.raises(ValueError, match="Expected no missing values"):
        validate_dataset(
            make_dataset(include_missing=True), expected_rows=4, expected_features=2
        )


def test_combine_tables_rejects_misaligned_indices() -> None:
    features = pd.DataFrame({"x": [1, 2]}, index=[0, 1])
    targets = pd.DataFrame({"class": [1, 2]}, index=[1, 2])
    with pytest.raises(ValueError, match="indices do not align"):
        combine_tables(features, targets)


def test_raw_snapshot_hash_must_match_recorded_hash_exactly(tmp_path) -> None:
    """Reject even a one-byte change to the Stage 1 snapshot."""
    raw_dir = tmp_path / "data" / "raw"
    report_dir = tmp_path / "reports"
    raw_dir.mkdir(parents=True)
    report_dir.mkdir()
    raw_path = raw_dir / "german_credit.csv"
    raw_path.write_text("feature,class\nA,1\nB,2\n", encoding="utf-8")
    recorded = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    (report_dir / "data_verification.json").write_text(
        json.dumps({"raw_snapshot": {"sha256": recorded}}),
        encoding="utf-8",
    )

    loaded = load_verified_raw_snapshot(tmp_path)
    assert loaded.data.targets["class"].tolist() == [1, 2]

    raw_path.write_text("feature,class\nA,1\nB,1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 does not match"):
        load_verified_raw_snapshot(tmp_path)
