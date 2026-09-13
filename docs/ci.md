# Continuous integration

## Scope

`.github/workflows/ci.yml` runs on every push and pull request. It validates the
committed application and canonical frozen model; it does not train, rebuild,
tune, deploy, or upload data. The workflow requires no secrets and grants only
read access to repository contents.

The hosted workflow was green at Stage 13 entry commit `b8dd5bd`. Linux CI
previously exposed two portability issues—frozen-file byte normalization and a
Compose bridge marked `internal`—which were corrected without changing inference
semantics.

## Quality job

The Ubuntu 24.04 quality job:

1. installs Python 3.12.0;
2. installs the development, API, and UI dependency groups under
   `docker/constraints-ci.txt`;
3. runs `pip check`;
4. verifies the committed frozen-evidence hashes;
5. retrieves the ignored official UCI snapshot needed by legacy tests and
   verifies it through the Stage 1 loader;
6. runs Ruff and Python compilation;
7. runs the complete pytest suite; and
8. runs `python -m creditscope.container_verify` on Linux.

The UCI fetch is input reconstruction from the authoritative source, not model
development. The raw file is not uploaded as a CI artifact.

## Container job

After quality succeeds, a separate Ubuntu job:

- builds the API and UI images;
- runs the canonical-model verifier inside the Linux API image;
- starts the two-service Compose stack and waits for health;
- checks the UI-to-API user-defined bridge-network path;
- runs the end-to-end golden response test; and
- always tears down Compose.

Logs are printed only on failure. No image is pushed to a registry and no cloud
deployment occurs.

## Frozen evidence

CI checks `scripts/frozen_evidence.sha256`, the Stage 9 manifest, model version,
threshold 0.16, 17-feature boundary, calibration architecture, and golden
fixtures. It never regenerates the canonical joblib model.

## Local equivalents

Python checks:

```powershell
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m compileall -q -f src tests scripts
.\.venv\Scripts\python.exe -m pytest --basetemp=.pytest_tmp
.\.venv\Scripts\python.exe -m creditscope.container_verify
```

Docker checks:

```powershell
.\scripts\manual_docker_validation.ps1
```

## Deliberate exclusions

CI does not use secrets, deploy services, publish images, upload raw or holdout
data, expose sample payloads as artifacts, rerun holdout evaluation, calculate
performance metrics, or execute model-development entry points.
