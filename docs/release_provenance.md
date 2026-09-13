# Release-candidate provenance

This record describes the Stage 13 entry state for the planned repository release
`v1.0.0`. It does not create a tag or GitHub Release.

| Item | Frozen value |
|---|---|
| Stage 13 starting commit | `b8dd5bd36ab910fef450da63145d534acaf4a4c7` |
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
| Hosted CI at entry | Green |
| Final release commit | **PENDING** |
| Final release CI run | **PENDING** |
| `v1.0.0` tag | **PENDING** |
| GitHub Release | **PENDING** |

The project release version describes the repository as a whole. The model and
API have their own explicit versions because their compatibility and governance
lifecycles are distinct.

## Dataset provenance

The modelling dataset is UCI Statlog (German Credit Data), dataset ID 144, DOI
`10.24432/C5NC77`. UCI South German Credit, dataset ID 573, DOI
`10.24432/C5QG88`, is used only as the corrected semantic reference. The project
does not silently replace or combine the modelling dataset.

