# Release-candidate provenance

This record describes the Stage 13 entry state for the planned repository release
`v1.0.0`. It does not create a tag or GitHub Release.

| Item | Frozen value |
|---|---|
| Stage 13 starting commit | `b8dd5bd36ab910fef450da63145d534acaf4a4c7` |
| Stage 13 release-candidate commit | `0acd122f10aec77467fc9204bb93a3abce5c77a3` |
| Branch | `main` |
| Stage 9 model SHA-256 | `c92e062e1cdde4965a7130be244cba8d01bc95e2b0d0cbb8a7feda43fbc3bcb7` |
| Stage 6P policy SHA-256 | `afee2e4ea82d438fa0ba638a8b7bdadaecbce9a4ada7da1f72d66866e4313afc` |
| Stage 7 prediction SHA-256 | `dc43c8f81bfd993ec6735253be3fdc572ba44deda2743d4c7b9f8dbff41dce18` |
| Stage 7 metric SHA-256 | `c676b0f0ea0a02a4bfe8b681cc29903831aef83631def31c38eeb599879b124d` |
| Model version | `creditscope-model-1.0.0` |
| API version | `creditscope-api-1.0.0` |
| Decision threshold | `0.16` |
| Predictive features | 17 |
| Calibration | Sigmoid, `ensemble=True`, five calibrated members |
| Hosted CI for release candidate | Green |
| Definitive v1.0.0 release commit | Commit referenced by annotated tag `v1.0.0` — **PENDING** |
| Final hosted CI for closeout commit | **PENDING** |
| `v1.0.0` tag | **PENDING** |
| GitHub Release | **PENDING** |

The project release version describes the repository as a whole. The model and
API have their own explicit versions because their compatibility and governance
lifecycles are distinct.

## Completed release-candidate validation

The `0acd122` candidate passed local FastAPI and Streamlit inspection, genuine
application screenshot capture, Docker release validation, strict loading of the
canonical artifact on Linux, golden endpoint parity, public GitHub README and
Mermaid rendering review, hosted GitHub Actions, and a public fresh-clone
rehearsal. The fresh clone contained no local Python environment, retained all
four frozen hashes, built both images, verified the Linux artifact, started a
healthy Compose stack, passed `/health`, `/model-info`, golden prediction, and
UI-to-API connectivity checks, then shut down cleanly without changing Git state.

No holdout analysis or model-policy change occurred during release validation.

The definitive v1.0.0 release commit is the commit referenced by the annotated
Git tag `v1.0.0`. The tag itself remains pending and is not created by Stage 13.

## Dataset provenance

The modelling dataset is UCI Statlog (German Credit Data), dataset ID 144, DOI
`10.24432/C5NC77`. UCI South German Credit, dataset ID 573, DOI
`10.24432/C5QG88`, is used only as the corrected semantic reference. The project
does not silently replace or combine the modelling dataset.
