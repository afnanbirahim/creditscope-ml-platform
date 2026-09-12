"""Fetch and verify the official UCI Statlog German Credit dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd

UCI_DATASET_ID = 144
UCI_DATASET_URL = (
    "https://archive.ics.uci.edu/dataset/144/statlog%2Bgerman%2Bcredit%2Bdata"
)
UCI_DATASET_DOI = "10.24432/C5NC77"
EXPECTED_ROWS = 1_000
EXPECTED_FEATURES = 20
EXPECTED_TARGET_LABELS = {1, 2}


def fetch_official_dataset() -> Any:
    """Fetch UCI dataset 144 using UCI's maintained Python client."""
    from ucimlrepo import fetch_ucirepo

    return fetch_ucirepo(id=UCI_DATASET_ID)


def load_verified_raw_snapshot(project_root: Path) -> Any:
    """Load Stage 1's source snapshot after verifying its recorded SHA-256."""
    raw_path = project_root / "data" / "raw" / "german_credit.csv"
    report_path = project_root / "reports" / "data_verification.json"
    if not raw_path.is_file() or not report_path.is_file():
        raise FileNotFoundError(
            "The verified Stage 1 raw snapshot and verification report are required."
        )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    recorded_hash = report.get("raw_snapshot", {}).get("sha256")
    observed_hash = _sha256(raw_path)
    if not recorded_hash or observed_hash != recorded_hash:
        raise ValueError("Raw snapshot SHA-256 does not match Stage 1 verification.")

    combined = pd.read_csv(raw_path)
    if "class" not in combined.columns:
        raise ValueError("Raw snapshot does not contain the documented class target.")
    return SimpleNamespace(
        metadata={"uci_id": UCI_DATASET_ID, "name": "Statlog (German Credit Data)"},
        data=SimpleNamespace(
            features=combined.drop(columns="class"),
            targets=combined[["class"]],
        ),
    )


def combine_tables(features: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    """Join source features and target without changing values or encoding."""
    if features.index.equals(targets.index) is False:
        raise ValueError("Feature and target indices do not align.")
    return pd.concat([features, targets], axis="columns")


def validate_dataset(
    dataset: Any,
    *,
    expected_rows: int = EXPECTED_ROWS,
    expected_features: int = EXPECTED_FEATURES,
    expected_target_labels: set[int] = EXPECTED_TARGET_LABELS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate source identity, dimensions, missingness, and target domain."""
    metadata = dataset.metadata
    dataset_id = int(metadata["uci_id"])
    if dataset_id != UCI_DATASET_ID:
        raise ValueError(f"Expected UCI dataset 144, received {dataset_id}.")

    features = dataset.data.features
    targets = dataset.data.targets
    if not isinstance(features, pd.DataFrame) or not isinstance(targets, pd.DataFrame):
        raise TypeError("UCI client did not return pandas DataFrames.")
    if features.shape != (expected_rows, expected_features):
        raise ValueError(
            f"Expected feature shape {(expected_rows, expected_features)}, "
            f"received {features.shape}."
        )
    if targets.shape != (expected_rows, 1):
        raise ValueError(
            f"Expected target shape {(expected_rows, 1)}, received {targets.shape}."
        )

    combined = combine_tables(features, targets)
    missing_cells = int(combined.isna().sum().sum())
    if missing_cells:
        raise ValueError(f"Expected no missing values, found {missing_cells} cells.")

    labels = {int(value) for value in targets.iloc[:, 0].unique()}
    if labels != expected_target_labels:
        raise ValueError(
            f"Expected target labels {sorted(expected_target_labels)}, "
            f"received {sorted(labels)}."
        )
    return features, targets


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_report(
    dataset: Any,
    features: pd.DataFrame,
    targets: pd.DataFrame,
    raw_path: Path | None,
    raw_display_path: str | None = None,
) -> dict[str, Any]:
    """Build a JSON-serializable verification summary from observed data."""
    combined = combine_tables(features, targets)
    target_name = str(targets.columns[0])
    class_counts = targets.iloc[:, 0].value_counts().sort_index()
    report: dict[str, Any] = {
        "source": {
            "repository": "UCI Machine Learning Repository",
            "dataset_id": int(dataset.metadata["uci_id"]),
            "name": str(dataset.metadata["name"]),
            "url": UCI_DATASET_URL,
            "doi": UCI_DATASET_DOI,
            "license": "CC BY 4.0",
        },
        "observed": {
            "rows": int(features.shape[0]),
            "features": int(features.shape[1]),
            "targets": int(targets.shape[1]),
            "target_name": target_name,
            "target_class_counts": {
                str(int(label)): int(count) for label, count in class_counts.items()
            },
            "missing_cells": int(combined.isna().sum().sum()),
            "duplicate_rows_including_target": int(combined.duplicated().sum()),
            "feature_columns": [str(column) for column in features.columns],
        },
        "checks": {
            "official_dataset_id": True,
            "documented_dimensions": True,
            "no_missing_values": True,
            "target_labels_are_1_and_2": True,
            "modeling_performed": False,
        },
    }
    if raw_path is not None:
        report["raw_snapshot"] = {
            "path": raw_display_path or raw_path.name,
            "sha256": _sha256(raw_path),
        }
    return report


def verify_and_write(project_root: Path, save_raw: bool) -> dict[str, Any]:
    """Run the live UCI verification and write reproducible evidence."""
    dataset = fetch_official_dataset()
    features, targets = validate_dataset(dataset)
    combined = combine_tables(features, targets)

    raw_path: Path | None = None
    if save_raw:
        raw_path = project_root / "data" / "raw" / "german_credit.csv"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(raw_path, index=False, lineterminator="\n")

    report = build_report(
        dataset,
        features,
        targets,
        raw_path,
        raw_display_path="data/raw/german_credit.csv" if raw_path else None,
    )
    report_path = project_root / "reports" / "data_verification.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--save-raw",
        action="store_true",
        help="write an ignored CSV snapshot under data/raw",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]
    report = verify_and_write(project_root, save_raw=args.save_raw)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
