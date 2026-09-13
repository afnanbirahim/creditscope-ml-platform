# Changelog

## v1.0.0 — 2026-09-13

### Added

- Verified UCI dataset provenance, integrity checks, and corrected semantic audit.
- Reproducible EDA, governed feature boundary, locked holdout, and baseline models.
- Paired model-family comparison and nested XGBoost tuning evidence.
- Development-only calibration and asymmetric-cost threshold governance.
- Frozen one-time holdout evaluation and immutable evidence hashes.
- Post-hoc SHAP explanation, subgroup diagnostics, proxy-risk register, and model card.
- Strict serialized inference package with manifest, schema, and golden fixtures.
- FastAPI prediction service and HTTP-only Streamlit interface.
- Separate hardened Docker images, Compose orchestration, and Linux artifact parity.
- GitHub Actions quality, frozen-evidence, API/UI, container, and end-to-end checks.
- Release-candidate architecture, interview, provenance, and release documentation.

### Governance

- Model development is closed for `creditscope-model-1.0.0`.
- The 5:1 false-negative/false-positive cost ratio is an educational assumption,
  not an industry standard or claim of real lending economics.
- CreditScope remains an educational/research decision-support prototype and is
  not suitable for automated lending decisions.
