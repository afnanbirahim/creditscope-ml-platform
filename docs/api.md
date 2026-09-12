# CreditScope FastAPI inference service

## Purpose and boundary

`creditscope-api-1.0.0` is a local HTTP adapter for the authoritative Stage 9
inference layer and `creditscope-model-1.0.0`. CreditScope is an
educational/research credit-risk decision-support prototype. It estimates good
or bad credit risk; it does not approve, reject, price, or determine eligibility
for lending.

The service cannot train, refit, recalibrate, select features, change the fixed
threshold, load user-supplied artifacts, expose holdout data, or run SHAP. The
model is loaded once during application startup and treated as read-only.

## Run locally

From the repository root on PowerShell, after installing the `api` dependency
group and creating the Stage 9 artifact:

```powershell
.\.venv\Scripts\python.exe -m uvicorn creditscope.api:app --host 127.0.0.1 --port 8000
```

Swagger UI is at `http://127.0.0.1:8000/docs`; the OpenAPI document is at
`http://127.0.0.1:8000/openapi.json`. CORS is disabled. Localhost is the
recommended Stage 10 binding.

Optional process configuration is limited to:

- `CREDITSCOPE_MODEL_PATH`: trusted model artifact path; defaults to the
  canonical Stage 9 artifact.
- `CREDITSCOPE_MANIFEST_PATH`: matching trusted manifest path.
- `CREDITSCOPE_BATCH_LIMIT`: positive maximum batch size; defaults to 100.

Threshold, feature set, preprocessing, model parameters, and calibration are
not configurable.

## Routes

| Method | Route | Purpose |
|---|---|---|
| `GET` | `/` | Service, API version, model version, purpose, and docs link. |
| `GET` | `/health` | Lightweight startup/readiness state; does not predict. |
| `GET` | `/model-info` | Safe frozen-policy metadata. |
| `POST` | `/predict` | One strict 17-field prediction. |
| `POST` | `/predict-batch` | Ordered, all-or-nothing prediction for 1–100 records. |

There are no administrative, model-update, threshold-update, training,
explanation, audit-data, or holdout routes.

## Request contract

`POST /predict` accepts exactly the following 17 fields. The canonical category
domains and semantic descriptions remain in `artifacts/model/input_schema.json`
and `docs/inference_contract.md`.

| Field | HTTP type |
|---|---|
| `duration_months` | finite number |
| `credit_amount` | finite number |
| `checking_account_status` | documented source code string |
| `credit_history` | documented source code string |
| `purpose` | documented source code string |
| `savings_account_status` | documented source code string |
| `employment_duration` | documented source code string |
| `installment_rate_percent` | integer source level 1–4 |
| `other_debtors_guarantors` | documented source code string |
| `residence_duration` | integer source level 1–4 |
| `property` | documented source code string |
| `other_installment_plans` | documented source code string |
| `housing` | documented source code string |
| `existing_credits_count` | integer source level 1–4 |
| `job` | documented source code string |
| `dependents_count` | integer source level 1–2 |
| `telephone` | documented source code string |

The Pydantic transport model is strict: numeric strings are rejected rather
than coerced. The request is then passed to the Stage 9 validator, which remains
the semantic authority. Missing, additional, audit-only, null, non-finite, and
invalid-category values are rejected. In particular, `personal_status_sex`,
`age_years`, and `foreign_worker` are not prediction inputs.

`POST /predict-batch` accepts a top-level JSON array of the same objects. Empty
batches fail validation. A batch over 100 records receives `413`; a batch with
one invalid member is rejected in full and nothing is scored or truncated.

## Response contract

Successful single prediction:

```json
{
  "probability_bad_risk": 0.349589290168932,
  "predicted_class": 1,
  "predicted_label": "bad_credit_risk",
  "threshold": 0.16,
  "model_version": "creditscope-model-1.0.0"
}
```

Class 0 means `good_credit_risk`; class 1 means `bad_credit_risk`. The sole
decision-support rule is class 1 when calibrated probability is greater than or
equal to 0.16. A batch response is an ordered JSON array using the same
prediction schema. See all four files named `api_*` under `examples/`.

PowerShell example:

```powershell
$body = Get-Content examples/api_single_request.json -Raw
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/predict `
  -ContentType 'application/json' -Body $body
```

## Error contract

- `422`: malformed JSON or request-schema/semantic validation failure.
- `400`: a Stage 9 semantic inference error that passes the HTTP schema.
- `413`: batch exceeds the configured limit.
- `500`: unexpected internal service failure; no stack trace is returned.
- `503`: verified model is unavailable after startup state is lost.

Startup itself fails closed if the model or manifest is absent, the model hash
does not match both the manifest and authoritative artifact hash, the scientific
environment is incompatible, or model version, predictor boundary, calibration,
member count, or threshold differs from the frozen policy.

## Security, privacy, and concurrency

Joblib has pickle semantics and is unsafe for untrusted files. Only the internal
hash-verified artifact is loaded, from process configuration—not a request.
There is no upload or arbitrary-path endpoint. Request payloads are not written
to disk or logged by application code. Audit-only attributes cannot enter the
contract. Unexpected internal failures are logged without application-level
payload logging.

The estimator is loaded once and inference is read-only. FastAPI may handle
concurrent requests, but handlers do not refit or mutate model state. Stage 10
does not add queues, distributed workers, or asynchronous model execution. This
is an engineering prototype rather than a complete production security posture;
authentication, authorization, rate limiting, TLS termination, observability,
and formal load testing remain outside this stage.
