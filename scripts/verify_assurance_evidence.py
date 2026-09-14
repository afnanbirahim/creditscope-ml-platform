"""Verify CreditScope assurance evidence without deserializing model artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import subprocess
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path, PurePosixPath

CANONICAL_EXPECTED_HASHES: dict[str, str] = {
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
RELEASE_TAG = "v1.0.0"
RELEASE_COMMIT = "51db046543c2d95c35058469067ba9f988a6133b"

REGISTER_COLUMNS = (
    "evidence_id",
    "category",
    "path",
    "description",
    "source_stage",
    "sha256",
    "frozen_status",
    "release_relationship",
    "assurance_purpose",
    "integrity_status",
    "notes",
)
ALLOWED_FROZEN_STATUSES = {
    "CANONICAL_FROZEN",
    "SUPPORTING_FROZEN",
    "SUPPORTING",
    "GENERATED_ASSURANCE",
}
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
EVIDENCE_ID_PATTERN = re.compile(r"^EV-[A-Z0-9]+-[0-9]{3}$")


class EvidenceVerificationError(RuntimeError):
    """Raised when an assurance evidence control fails."""


def sha256_file(path: Path) -> str:
    """Return the SHA-256 of raw file bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_repository_relative_path(value: str) -> None:
    if not value or "\\" in value or re.match(r"^[A-Za-z]:", value):
        raise EvidenceVerificationError(
            f"Evidence path must be a non-empty POSIX repository-relative path: {value!r}"
        )
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        raise EvidenceVerificationError(
            f"Evidence path escapes the repository boundary: {value!r}"
        )


def _git(repository_root: Path, *arguments: str) -> bytes:
    command = [
        "git",
        "-c",
        f"safe.directory={repository_root.as_posix()}",
        *arguments,
    ]
    result = subprocess.run(
        command,
        cwd=repository_root,
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise EvidenceVerificationError(
            f"Git evidence query failed ({' '.join(arguments)}): {detail}"
        )
    return result.stdout


def verify_release_custody(repository_root: Path) -> None:
    """Require the annotated release tag to resolve to the frozen release commit."""
    tag_type = _git(repository_root, "cat-file", "-t", RELEASE_TAG).decode().strip()
    if tag_type != "tag":
        raise EvidenceVerificationError(
            f"{RELEASE_TAG} must be an annotated tag, got {tag_type!r}"
        )
    resolved = (
        _git(repository_root, "rev-list", "-n", "1", RELEASE_TAG).decode().strip()
    )
    if resolved != RELEASE_COMMIT:
        raise EvidenceVerificationError(
            f"{RELEASE_TAG} resolves to {resolved}, expected {RELEASE_COMMIT}"
        )


def git_blob_bytes(repository_root: Path, revision: str, relative_path: str) -> bytes:
    """Read exact version-controlled blob bytes without checkout conversion."""
    _validate_repository_relative_path(relative_path)
    return _git(repository_root, "show", f"{revision}:{relative_path}")


def verify_git_anchored_file(
    repository_root: Path,
    relative_path: str,
    expected_release_sha256: str,
    release_ref: str = RELEASE_TAG,
    enforce_current_checkout: bool = True,
) -> str:
    """Verify release blob, current HEAD, and checked-out Git content identity."""
    _validate_repository_relative_path(relative_path)
    if not SHA256_PATTERN.fullmatch(expected_release_sha256):
        raise EvidenceVerificationError(
            f"Malformed registered SHA-256 for {relative_path}"
        )

    working_path = repository_root / PurePosixPath(relative_path)
    if not working_path.is_file():
        raise EvidenceVerificationError(f"Registered evidence is missing: {relative_path}")

    release_bytes = git_blob_bytes(repository_root, release_ref, relative_path)
    release_sha256 = hashlib.sha256(release_bytes).hexdigest()
    if release_sha256 != expected_release_sha256:
        raise EvidenceVerificationError(
            f"Release Git-blob SHA-256 mismatch for {relative_path}: "
            f"expected {expected_release_sha256}, got {release_sha256}"
        )

    if not enforce_current_checkout:
        _git(repository_root, "ls-files", "--error-unmatch", relative_path)
        return release_sha256

    head_bytes = git_blob_bytes(repository_root, "HEAD", relative_path)
    if head_bytes != release_bytes:
        raise EvidenceVerificationError(
            f"Current HEAD redefines registered v1.0.0 evidence: {relative_path}"
        )
    release_oid = (
        _git(repository_root, "rev-parse", f"{release_ref}:{relative_path}")
        .decode()
        .strip()
    )
    working_oid = (
        _git(
            repository_root,
            "hash-object",
            f"--path={relative_path}",
            relative_path,
        )
        .decode()
        .strip()
    )
    if working_oid != release_oid:
        raise EvidenceVerificationError(
            f"Working-tree content differs from released evidence: {relative_path}"
        )
    return release_sha256


def verify_expected_hashes(
    repository_root: Path, expected_hashes: Mapping[str, str]
) -> dict[str, str]:
    """Verify expected hashes and return calculated values, or fail as a group."""
    calculated: dict[str, str] = {}
    failures: list[str] = []
    for relative_path, expected_hash in expected_hashes.items():
        _validate_repository_relative_path(relative_path)
        if not SHA256_PATTERN.fullmatch(expected_hash):
            failures.append(f"Malformed expected SHA-256 for {relative_path}")
            continue
        path = repository_root / PurePosixPath(relative_path)
        if not path.is_file():
            failures.append(f"Missing required evidence: {relative_path}")
            continue
        actual_hash = sha256_file(path)
        calculated[relative_path] = actual_hash
        if actual_hash != expected_hash:
            failures.append(
                f"SHA-256 mismatch for {relative_path}: "
                f"expected {expected_hash}, got {actual_hash}"
            )
    if failures:
        raise EvidenceVerificationError("\n".join(failures))
    return calculated


def read_evidence_register(register_path: Path) -> list[dict[str, str]]:
    """Load and structurally validate the evidence-register CSV."""
    with register_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != REGISTER_COLUMNS:
            raise EvidenceVerificationError(
                "Evidence register columns differ from the required schema"
            )
        rows = list(reader)
    if not rows:
        raise EvidenceVerificationError("Evidence register contains no evidence rows")
    return rows


def validate_register_structure(
    rows: Iterable[Mapping[str, str]],
) -> list[Mapping[str, str]]:
    """Validate register structure without consulting unrelated live evidence."""
    materialized = list(rows)
    observed_ids: set[str] = set()
    observed_paths: set[str] = set()
    canonical_rows: dict[str, Mapping[str, str]] = {}

    for line_number, row in enumerate(materialized, start=2):
        if set(row) != set(REGISTER_COLUMNS):
            raise EvidenceVerificationError(
                f"Malformed register row at CSV line {line_number}"
            )
        if any(row[column] == "" for column in REGISTER_COLUMNS[:-1]):
            raise EvidenceVerificationError(
                f"Blank required value at CSV line {line_number}"
            )

        evidence_id = row["evidence_id"]
        if not EVIDENCE_ID_PATTERN.fullmatch(evidence_id):
            raise EvidenceVerificationError(f"Malformed evidence ID: {evidence_id!r}")
        if evidence_id in observed_ids:
            raise EvidenceVerificationError(f"Duplicate evidence ID: {evidence_id}")
        observed_ids.add(evidence_id)

        relative_path = row["path"]
        _validate_repository_relative_path(relative_path)
        if relative_path in observed_paths:
            raise EvidenceVerificationError(
                f"Duplicate evidence path: {relative_path}"
            )
        observed_paths.add(relative_path)

        registered_hash = row["sha256"]
        if not SHA256_PATTERN.fullmatch(registered_hash):
            raise EvidenceVerificationError(
                f"Malformed registered SHA-256 for {relative_path}"
            )
        if row["frozen_status"] not in ALLOWED_FROZEN_STATUSES:
            raise EvidenceVerificationError(
                f"Unknown frozen status for {relative_path}: {row['frozen_status']}"
            )

        if row["frozen_status"] == "CANONICAL_FROZEN":
            canonical_rows[relative_path] = row

    if set(canonical_rows) != set(CANONICAL_EXPECTED_HASHES):
        missing = sorted(set(CANONICAL_EXPECTED_HASHES) - set(canonical_rows))
        unexpected = sorted(set(canonical_rows) - set(CANONICAL_EXPECTED_HASHES))
        raise EvidenceVerificationError(
            f"Canonical register membership differs; missing={missing}, "
            f"unexpected={unexpected}"
        )
    for relative_path, expected_hash in CANONICAL_EXPECTED_HASHES.items():
        if canonical_rows[relative_path]["sha256"] != expected_hash:
            raise EvidenceVerificationError(
                f"Canonical register hash differs for {relative_path}"
            )
    return materialized


def validate_evidence_register(
    repository_root: Path, rows: Iterable[Mapping[str, str]]
) -> dict[str, str]:
    """Verify canonical raw bytes and other evidence against v1.0.0 Git blobs."""
    structured_rows = validate_register_structure(rows)
    verify_release_custody(repository_root)
    calculated: dict[str, str] = {}

    for row in structured_rows:
        relative_path = row["path"]
        registered_hash = row["sha256"]
        if row["frozen_status"] == "CANONICAL_FROZEN":
            raw_hash = verify_expected_hashes(
                repository_root, {relative_path: registered_hash}
            )[relative_path]
            release_hash = hashlib.sha256(
                git_blob_bytes(repository_root, RELEASE_TAG, relative_path)
            ).hexdigest()
            if release_hash != registered_hash:
                raise EvidenceVerificationError(
                    f"Canonical release blob differs for {relative_path}: "
                    f"expected {registered_hash}, got {release_hash}"
                )
            head_hash = hashlib.sha256(
                git_blob_bytes(repository_root, "HEAD", relative_path)
            ).hexdigest()
            if head_hash != registered_hash:
                raise EvidenceVerificationError(
                    f"Current HEAD redefines canonical evidence: {relative_path}"
                )
            calculated[relative_path] = raw_hash
        elif row["frozen_status"] in {"SUPPORTING_FROZEN", "SUPPORTING"}:
            calculated[relative_path] = verify_git_anchored_file(
                repository_root,
                relative_path,
                registered_hash,
                enforce_current_checkout=(
                    row["release_relationship"] != "POST_RELEASE_EVOLVING_CONTROL"
                ),
            )
        else:
            calculated[relative_path] = verify_expected_hashes(
                repository_root, {relative_path: registered_hash}
            )[relative_path]
    return calculated


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify canonical and registered CreditScope assurance evidence."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root (defaults to the script's parent repository)",
    )
    parser.add_argument(
        "--register",
        type=Path,
        default=None,
        help="Evidence register path (defaults to assurance/evidence_register.csv)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.root.resolve()
    register_path = args.register or root / "assurance" / "evidence_register.csv"
    try:
        canonical = verify_expected_hashes(root, CANONICAL_EXPECTED_HASHES)
        for relative_path, actual_hash in canonical.items():
            print(f"PASS canonical {relative_path} {actual_hash}")
        rows = read_evidence_register(register_path)
        registered = validate_evidence_register(root, rows)
        print(
            f"PASS evidence register {register_path.name}: "
            f"{len(rows)} unique items, {len(registered)} authoritative hashes verified"
        )
    except (EvidenceVerificationError, OSError, csv.Error) as exc:
        print(f"FAIL assurance evidence verification: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
