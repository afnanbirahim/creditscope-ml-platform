"""Focused Stage A1 evidence-integrity controls."""

from __future__ import annotations

import csv
import hashlib
import subprocess
from pathlib import Path

import pytest

from scripts.verify_assurance_evidence import (
    CANONICAL_EXPECTED_HASHES,
    REGISTER_COLUMNS,
    RELEASE_COMMIT,
    RELEASE_TAG,
    EvidenceVerificationError,
    read_evidence_register,
    validate_evidence_register,
    validate_register_structure,
    verify_expected_hashes,
    verify_git_anchored_file,
    verify_release_custody,
)

ROOT = Path(__file__).resolve().parents[2]
REGISTER = ROOT / "assurance" / "evidence_register.csv"


def _git(repository: Path, *arguments: str) -> None:
    subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=True,
        capture_output=True,
    )


def _make_text_release_repository(tmp_path: Path) -> tuple[Path, str]:
    repository = tmp_path / "release-repository"
    repository.mkdir()
    _git(repository, "init", "--quiet")
    _git(repository, "config", "user.name", "Assurance Test")
    _git(repository, "config", "user.email", "assurance-test@example.invalid")
    (repository / ".gitattributes").write_bytes(b"*.txt text\n")
    evidence = repository / "evidence.txt"
    evidence.write_bytes(b"line one\nline two\n")
    _git(repository, "add", ".gitattributes", "evidence.txt")
    _git(repository, "commit", "--quiet", "-m", "test release")
    _git(repository, "tag", "-a", RELEASE_TAG, "-m", "test release")
    release_hash = hashlib.sha256(b"line one\nline two\n").hexdigest()
    return repository, release_hash


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
    verify_release_custody(ROOT)
    assert verify_expected_hashes(ROOT, CANONICAL_EXPECTED_HASHES) == (
        CANONICAL_EXPECTED_HASHES
    )
    rows = read_evidence_register(REGISTER)
    calculated = validate_evidence_register(ROOT, rows)
    assert len(calculated) == len(rows)


def test_canonical_raw_byte_tamper_fails(tmp_path: Path) -> None:
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


@pytest.mark.parametrize(
    "checkout_bytes",
    [b"line one\nline two\n", b"line one\r\nline two\r\n"],
    ids=["linux-lf", "windows-crlf"],
)
def test_git_anchored_text_accepts_git_defined_checkout_eol(
    tmp_path: Path, checkout_bytes: bytes
) -> None:
    repository, release_hash = _make_text_release_repository(tmp_path)
    (repository / "evidence.txt").write_bytes(checkout_bytes)
    assert (
        verify_git_anchored_file(repository, "evidence.txt", release_hash)
        == release_hash
    )


def test_git_anchored_text_rejects_genuine_content_change(tmp_path: Path) -> None:
    repository, release_hash = _make_text_release_repository(tmp_path)
    (repository / "evidence.txt").write_bytes(b"line one\r\nchanged\r\n")
    with pytest.raises(EvidenceVerificationError, match="Working-tree content"):
        verify_git_anchored_file(repository, "evidence.txt", release_hash)


def test_explicit_evolving_control_remains_anchored_to_release(tmp_path: Path) -> None:
    repository, release_hash = _make_text_release_repository(tmp_path)
    (repository / "evidence.txt").write_bytes(b"post-release control update\n")
    assert (
        verify_git_anchored_file(
            repository,
            "evidence.txt",
            release_hash,
            enforce_current_checkout=False,
        )
        == release_hash
    )


def test_git_anchored_text_requires_working_file(tmp_path: Path) -> None:
    repository, release_hash = _make_text_release_repository(tmp_path)
    (repository / "evidence.txt").unlink()
    with pytest.raises(EvidenceVerificationError, match="is missing"):
        verify_git_anchored_file(repository, "evidence.txt", release_hash)


def test_register_schema_is_exact() -> None:
    with REGISTER.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert tuple(reader.fieldnames or ()) == REGISTER_COLUMNS


def test_duplicate_evidence_ids_are_detected_independently() -> None:
    rows = read_evidence_register(REGISTER)
    duplicated = rows + [dict(rows[0], path="duplicate/path.bin")]
    with pytest.raises(EvidenceVerificationError, match="Duplicate evidence ID"):
        validate_register_structure(duplicated)


def test_duplicate_evidence_paths_are_detected_independently() -> None:
    rows = read_evidence_register(REGISTER)
    duplicated = rows + [dict(rows[0], evidence_id="EV-TEST-999")]
    with pytest.raises(EvidenceVerificationError, match="Duplicate evidence path"):
        validate_register_structure(duplicated)


def test_malformed_sha256_is_rejected_structurally() -> None:
    rows = read_evidence_register(REGISTER)
    malformed = [dict(rows[0], sha256="not-a-sha256"), *rows[1:]]
    with pytest.raises(EvidenceVerificationError, match="Malformed registered SHA-256"):
        validate_register_structure(malformed)


def test_invalid_evidence_status_is_rejected_structurally() -> None:
    rows = read_evidence_register(REGISTER)
    malformed = [dict(rows[0], frozen_status="UNCONTROLLED"), *rows[1:]]
    with pytest.raises(EvidenceVerificationError, match="Unknown frozen status"):
        validate_register_structure(malformed)


def test_repository_relative_path_is_enforced_structurally() -> None:
    rows = read_evidence_register(REGISTER)
    malformed = [dict(rows[0], path="../outside.bin"), *rows[1:]]
    with pytest.raises(EvidenceVerificationError, match="escapes the repository"):
        validate_register_structure(malformed)


def test_canonical_register_rows_use_expected_hashes() -> None:
    rows = read_evidence_register(REGISTER)
    canonical = {
        row["path"]: row["sha256"]
        for row in rows
        if row["frozen_status"] == "CANONICAL_FROZEN"
    }
    assert canonical == CANONICAL_EXPECTED_HASHES


def test_release_constants_are_frozen() -> None:
    assert RELEASE_TAG == "v1.0.0"
    assert RELEASE_COMMIT == "51db046543c2d95c35058469067ba9f988a6133b"


def test_quality_ci_checkout_fetches_release_tag_history() -> None:
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(
        encoding="utf-8"
    )
    quality_job, containers_job = workflow.split("\n  containers:", maxsplit=1)
    assert "uses: actions/checkout@v4" in quality_job
    assert "fetch-depth: 0" in quality_job
    assert "fetch-depth: 0" not in containers_job
    assert "permissions:\n  contents: read" in workflow
