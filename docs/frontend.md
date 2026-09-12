# CreditScope Streamlit frontend

## Architecture and scope

The Stage 11 frontend is a presentation layer over the existing Stage 10 API:

```text
Streamlit UI
    -> HTTP
FastAPI service
    -> Stage 9 inference layer
    -> frozen Stage 9 model artifact
```

`src/creditscope/ui.py` renders the page, while
`src/creditscope/api_client.py` owns HTTP transport and sanitized failure
handling. `src/creditscope/frontend.py` contains presentation-only field labels,
canonical category display mappings, payload ordering, and output formatting.

The frontend never loads joblib, imports XGBoost or SHAP, applies a threshold,
or accesses training or holdout data. FastAPI remains the prediction and
validation authority.

CreditScope is an educational/research credit-risk decision-support prototype.
It does not approve or reject loans, determine eligibility, replace human
review, or provide financial advice.

## Prerequisites and local workflow

Install the API, UI, and development dependency groups in the existing virtual
environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[api,ui,dev]"
```

Use two PowerShell terminals from the repository root.

Terminal 1 — verified FastAPI service:

```powershell
.\.venv\Scripts\python.exe -m uvicorn creditscope.api:app --host 127.0.0.1 --port 8000
```

Terminal 2 — Streamlit UI:

```powershell
.\.venv\Scripts\python.exe -m streamlit run src/creditscope/ui.py --server.address 127.0.0.1 --server.port 8501
```

Open `http://127.0.0.1:8501`. FastAPI documentation remains at
`http://127.0.0.1:8000/docs`. Public network exposure is outside Stage 11.

The frontend uses `CREDITSCOPE_API_URL` when set and otherwise calls
`http://127.0.0.1:8000`. No model-policy setting is configurable through the UI
or environment.

## Form and source semantics

The form exposes exactly the 17 governed predictors listed in
`artifacts/model/input_schema.json`. It does not request
`personal_status_sex`, `age_years`, or `foreign_worker`. Human-readable labels
and descriptions accompany canonical category codes; the code, rather than the
label, is sent to FastAPI.

The category display mappings come from the repository's documented UCI source
semantics. Important cautions appear as control help text: credit amount is a
historical transformed quantity rather than a literal contemporary monetary
amount, and installment rate is a discretized category rather than a literal
percentage.

Frontend constraints improve usability but do not replace backend validation.
The UI compares its 17-field order with `/model-info` before enabling the form.

## Result semantics

On form submission the UI sends exactly one `POST /predict` request and displays
the response's:

- good/bad credit-risk classification;
- calibrated estimated bad-credit-risk probability, rounded to one decimal
  percentage point for presentation only;
- frozen threshold returned by the API; and
- model version.

The original API probability remains unchanged in memory. The frontend does not
recalculate the class or threshold. It explains that the assumed 5:1 cost is an
educational modelling assumption, not established real-world lending economics.

Batch upload is intentionally deferred; Stage 11 focuses on the safer
single-record workflow.

## Failures and troubleshooting

The HTTP client uses a five-second timeout and turns connection failures,
timeouts, and HTTP 400/413/422/500/503 responses into concise messages. Raw
HTTPX exceptions, response bodies, stack traces, and local paths are not shown.

If the UI says the prediction service is unavailable:

1. Start FastAPI in Terminal 1.
2. Check `http://127.0.0.1:8000/health`.
3. Confirm `CREDITSCOPE_API_URL` points to that service if overridden.
4. Confirm the Stage 9 artifact and manifest pass FastAPI startup integrity.

The UI never falls back to local model loading.

## Privacy and responsible use

The UI does not persist session inputs, write prediction history, add tracking,
or log payloads. Inputs are transmitted only to the configured CreditScope API.
Page refresh or session termination may discard the result.

The source data are small, historical, and geographically/contextually limited.
Potential socioeconomic proxies remain among the predictors; excluding three
audit attributes does not establish fairness. Outputs and explanations are
associative rather than causal. The concise UI limitations panel references the
full Stage 8 model card.

## Screenshots

Screenshots are intentionally optional for this code-focused stage and can be
added later without changing inference behavior.
