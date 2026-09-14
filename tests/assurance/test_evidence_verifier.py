"""Focused Stage A1 evidence-integrity controls."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import pytest

from scripts.verify_assurance_evidence import (
    CANONICAL_EXPECTED_HASHES,
    REGISTER_COLUMNS,
    EvidenceVerificationError,
    read_evidence_register,
    validate_evidence_register,
    verify_expected_hashes,
)

ROOT = Path(__file__).resolve().parents[2]
REGISTER = ROOT / "assurance" / "evidence_register.csv"


def test_canonical_expected_hash_definitions_are_exact() -> None:
    assert CANONICAL_EXPECTED_HASHES == {
        "artifacts/model/creditscope_frozen_model.joblib": (
            "c92e062e1cdde4965a7130be244cba8d01bc95e2b0d0cbb8a7feda43fbc3bcb7"
        ),
        "reports/stage6/frozen_model_policy.json": (
            "afee2e4ea82d438fa0ba638a8b7bdadaecbce9a4ada7da1f72d66866e4313afc"
        ),
        "reports/stage7/final_holdout_predictions.csv": (
            "dc43c8f81bfd993ec6735253be3fdc572ba44deda2743d4c7b9f8dbff41dce18"
        ),
        "reports/stage7/final_holdout_metrics.csv": (
            "c676b0f0ea0a02a4bfe8b681cc29903831aef83631def31c38eeb599879b124d"
        ),
    }


def test_current_canonical_artifacts_and_register_verify() -> None:
    assert verify_expected_hashes(ROOT, CANONICAL_EXPECTED_HASHES) == (
        CANONICAL_EXPECTED_HASHES
    )
    rows = read_evidence_register(REGISTER)
    calculated = validate_evidence_register(ROOT, rows)
    assert len(calculated) == len(rows)


def test_tampered_temporary_evidence_fails(tmp_path: Path) -> None:
    relative_path = "evidence.bin"
    evidence = tmp_path / relative_path
    evidence.write_bytes(b"original")
    original_hash = hashlib.sha256(b"original").hexdigest()
    expected = verify_expected_hashes(
        tmp_path,
        {relative_path: original_hash},
    )
    assert expected == {relative_path: original_hash}
    evidence.write_bytes(b"altered")
    with pytest.raises(EvidenceVerificationError, match="SHA-256 mismatch"):
        verify_expected_hashes(tmp_path, expected)


def test_missing_temporary_evidence_fails(tmp_path: Path) -> None:
    with pytest.raises(EvidenceVerificationError, match="Missing required evidence"):
        verify_expected_hashes(tmp_path, {"missing.bin": "0" * 64})


def test_register_schema_is_exact() -> None:
    with REGISTER.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert tuple(reader.fieldnames or ()) == REGISTER_COLUMNS


def test_duplicate_evidence_ids_are_detected(tmp_path: Path) -> None:
    rows = read_evidence_register(REGISTER)
    duplicated = rows + [dict(rows[0], path="duplicate/path.bin")]
    with pytest.raises(EvidenceVerificationError, match="Duplicate evidence ID"):
        validate_evidence_register(ROOT, duplicated)


def test_canonical_register_rows_use_expected_hashes() -> None:
    rows = read_evidence_register(REGISTER)
    canonical = {
        row["path"]: row["sha256"]
        for row in rows
        if row["frozen_status"] == "CANONICAL_FROZEN"
    }
    assert canonical == CANONICAL_EXPECTED_HASHES
