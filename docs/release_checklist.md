# v1.0.0 release checklist

`PASS` means direct evidence exists. `PENDING` requires manual or future release
work. No release tag or GitHub Release exists yet.

| Gate | Status | Evidence/action |
|---|---|---|
| Clean working tree at Stage 13 entry | PASS | Entry audit at commit `b8dd5bd` |
| Frozen model/policy/Stage 7 hashes unchanged | PASS | Stage 13 entry hash audit |
| Model version and five-member architecture unchanged | PASS | Production artifact verifier |
| Feature contract remains 17 predictors | PASS | Manifest, schema, verifier, tests |
| Threshold remains 0.16 | PASS | Frozen policy and verifier |
| Full pytest suite | PASS | 139 passed after Stage 13 implementation |
| Ruff | PASS | `ruff check .` |
| Python compilation | PASS | `compileall` over source, tests, and scripts |
| `pip check` | PASS | No broken requirements |
| Golden fixtures and container verifier | PASS | Five fixtures; maximum local probability difference 0.0 |
| API contract and negative paths | PASS | Stage 9–11 tests passed without behavior changes |
| Public path/secret audit | PASS | No secret/email findings; local Stage 4 paths sanitized |
| Docker API and UI builds | PASS | Stage 12 external validation |
| Compose configuration and health | PASS | Stage 12 external validation and hosted CI |
| UI-to-API and host-side smoke | PASS | Stage 12 external validation and hosted CI |
| Linux canonical-artifact parity | PASS | Stage 12 container verifier and hosted CI |
| GitHub Actions on release-candidate commit | PASS | Hosted workflow green at `0acd122` |
| Documentation navigation and relative links | PASS | Automated local-target scan found zero broken links |
| Genuine application screenshots | PASS | Captured and committed from the running application |
| Local Streamlit inspection | PASS | Manually inspected |
| Local FastAPI verification | PASS | Manually verified |
| Docker release-candidate validation | PASS | API/UI images and Compose workflow passed |
| Canonical artifact in Linux | PASS | Strict load and integrity verification passed |
| API `/health` | PASS | Manual release validation |
| API `/model-info` | PASS | Manual release validation |
| Golden `/predict` request | PASS | Manual release validation |
| UI-to-API container connectivity | PASS | Manual release validation |
| Compose golden smoke test | PASS | Manual release validation |
| Compose teardown | PASS | Containers and network removed cleanly |
| Public GitHub rendering | PASS | Manually reviewed |
| README rendering | PASS | Manually reviewed on GitHub |
| Mermaid rendering | PASS | Manually reviewed on GitHub |
| Release-candidate commit pushed | PASS | `0acd122f10aec77467fc9204bb93a3abce5c77a3` |
| Fresh-clone repository state | PASS | Public clone was clean and contained no `.venv` or `.python` |
| Fresh-clone frozen hashes | PASS | All four canonical hashes matched |
| Fresh-clone Docker build | PASS | Completed successfully |
| Fresh-clone Linux artifact verification | PASS | Canonical artifact and golden evidence passed |
| Fresh-clone Compose startup | PASS | Services reached healthy state |
| Fresh-clone `/health` and `/model-info` | PASS | Both endpoints passed |
| Fresh-clone golden prediction | PASS | Frozen endpoint result passed |
| Fresh-clone UI-to-API connectivity | PASS | Service-name HTTP path passed |
| Fresh-clone teardown and Git state | PASS | Clean teardown; working tree remained clean |
| Software license decision | PASS | No software LICENSE for v1.0.0; explicitly deferred owner decision |
| Citation metadata decision | PASS | `CITATION.cff` intentionally deferred; no metadata invented |
| Final Stage 13 closeout commit | PENDING | User-managed Git mutation after this documentation closeout |
| Final hosted CI for closeout commit | PENDING | Must be green after push |
| Release notes reviewed | PENDING | Review dated `CHANGELOG.md` entry before tagging |
| `v1.0.0` tag and GitHub Release | PENDING | Explicit later authorization required |
| Post-release hash verification | PENDING | Verify after tag/release |

The absence of a software LICENSE and `CITATION.cff` is deliberate for v1.0.0,
not accidental. Dataset attribution and third-party terms remain documented
separately. Public visibility must not be interpreted as an open-source software
license. ORCID, DOI, affiliation, and preferred citation metadata are not guessed.

## Fresh-clone rehearsal

Run from a parent directory that is not this working tree:

```powershell
git clone https://github.com/afnanbirahim/creditscope-ml-platform.git creditscope-release-rehearsal
Set-Location -LiteralPath .\creditscope-release-rehearsal
git status --short
Get-FileHash -Algorithm SHA256 artifacts/model/creditscope_frozen_model.joblib
docker compose build
docker compose up --detach --wait --no-build
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-RestMethod http://127.0.0.1:8000/model-info
$body = Get-Content examples/api_single_request.json -Raw
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/predict -ContentType application/json -Body $body
docker compose exec -T ui python -c "from creditscope.api_client import CreditScopeAPIClient; c=CreditScopeAPIClient(); assert c.health()['status']=='ok'; c.close(); print('UI-to-API passed')"
docker compose down --volumes --remove-orphans
```

The rehearsal must not reuse the original `.venv`, project-local Python, Docker
containers, or uncommitted files.
