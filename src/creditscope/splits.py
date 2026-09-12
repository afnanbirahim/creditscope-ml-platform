"""Deterministic final-holdout isolation for CreditScope."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from creditscope.schema import TARGET_BINARY_NAME

RANDOM_STATE = 42
TEST_SIZE = 0.20
ROW_ID_NAME = "source_row_id"
DEVELOPMENT_SPLIT = "development"
TEST_SPLIT = "test"


def source_row_ids(data: pd.DataFrame) -> pd.Series:
    """Create stable identifiers from immutable UCI source-row positions."""
    return pd.Series(
        [f"uci144-row-{position + 1:04d}" for position in range(len(data))],
        index=data.index,
        name=ROW_ID_NAME,
    )


def dataset_fingerprint(data: pd.DataFrame) -> str:
    """Hash the complete ordered analysis table without writing another copy."""
    payload = data.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def manifest_fingerprint(manifest: pd.DataFrame) -> str:
    """Hash canonical split membership metadata."""
    payload = manifest.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def create_split_manifest(data: pd.DataFrame) -> pd.DataFrame:
    """Create the single 80/20 stratified split with random_state=42."""
    if set(data[TARGET_BINARY_NAME].unique()) != {0, 1}:
        raise ValueError("bad_credit_risk must contain exactly 0 and 1.")
    row_ids = source_row_ids(data)
    development_ids, _ = train_test_split(
        row_ids,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=data[TARGET_BINARY_NAME],
    )
    development_set = set(development_ids)
    manifest = pd.DataFrame(
        {
            ROW_ID_NAME: row_ids.to_numpy(),
            "split": [
                DEVELOPMENT_SPLIT if row_id in development_set else TEST_SPLIT
                for row_id in row_ids
            ],
            TARGET_BINARY_NAME: data[TARGET_BINARY_NAME].astype("int8").to_numpy(),
        }
    )
    validate_split_manifest(data, manifest)
    return manifest


def validate_split_manifest(data: pd.DataFrame, manifest: pd.DataFrame) -> None:
    """Validate exclusivity, completeness, target integrity, and exact sizes."""
    expected_columns = [ROW_ID_NAME, "split", TARGET_BINARY_NAME]
    if list(manifest.columns) != expected_columns:
        raise ValueError("Split manifest columns do not match the locked schema.")
    if len(manifest) != len(data):
        raise ValueError("Split manifest does not cover the complete dataset.")
    if manifest[ROW_ID_NAME].duplicated().any():
        raise ValueError("Split manifest contains overlapping/duplicate row identifiers.")
    if set(manifest[ROW_ID_NAME]) != set(source_row_ids(data)):
        raise ValueError("Split manifest union does not equal the complete dataset.")
    if set(manifest["split"].unique()) != {DEVELOPMENT_SPLIT, TEST_SPLIT}:
        raise ValueError("Split manifest must contain development and test memberships.")
    expected_development = round(len(data) * (1 - TEST_SIZE))
    counts = manifest["split"].value_counts()
    if int(counts[DEVELOPMENT_SPLIT]) != expected_development:
        raise ValueError("Development split has an unexpected row count.")
    if int(counts[TEST_SPLIT]) != len(data) - expected_development:
        raise ValueError("Test split has an unexpected row count.")
    if set(manifest[TARGET_BINARY_NAME].unique()) != {0, 1}:
        raise ValueError("Manifest target values must be 0 and 1.")
    if not (
        manifest[TARGET_BINARY_NAME].astype("int8").reset_index(drop=True)
        == data[TARGET_BINARY_NAME].astype("int8").reset_index(drop=True)
    ).all():
        raise ValueError("Manifest targets do not align with source rows.")


def write_locked_split(data: pd.DataFrame, reports_dir: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    """Create or verify the immutable deterministic manifest and its summary."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = reports_dir / "split_manifest.csv"
    expected = create_split_manifest(data)
    if manifest_path.exists():
        existing = pd.read_csv(manifest_path, dtype={TARGET_BINARY_NAME: "int8"})
        validate_split_manifest(data, existing)
        if not existing.equals(expected):
            raise ValueError("Existing split manifest differs from the locked deterministic split.")
        manifest = existing
    else:
        manifest = expected
        manifest.to_csv(manifest_path, index=False, lineterminator="\n")

    target_counts = (
        manifest.groupby(["split", TARGET_BINARY_NAME]).size().unstack(fill_value=0)
    )
    summary: dict[str, object] = {
        "total_rows": len(manifest),
        "development_rows": int((manifest["split"] == DEVELOPMENT_SPLIT).sum()),
        "test_rows": int((manifest["split"] == TEST_SPLIT).sum()),
        "target_counts": {
            split: {
                "good_credit_risk_0": int(target_counts.loc[split, 0]),
                "bad_credit_risk_1": int(target_counts.loc[split, 1]),
            }
            for split in (DEVELOPMENT_SPLIT, TEST_SPLIT)
        },
        "random_state": RANDOM_STATE,
        "test_fraction": TEST_SIZE,
        "stratified": True,
        "method": "sklearn.model_selection.train_test_split",
        "row_identifier": "one-based immutable UCI source-row position",
        "dataset_sha256": dataset_fingerprint(data),
        "manifest_sha256": manifest_fingerprint(manifest),
        "final_test_performance_evaluated": False,
    }
    (reports_dir / "split_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return manifest, summary
