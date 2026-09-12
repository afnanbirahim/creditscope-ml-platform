# CreditScope

CreditScope is a production-minded, explainable credit-risk decision-support
prototype for education, research, and professional portfolio demonstration. It
is **not** an automated lending approval system and must not be used to make
real credit decisions.

## Completed scope: Stages 1–11; Stage 12 implemented

Stage 1 initialized the project and verified the source data. Stage 2 adds the
official data dictionary, stable internal names, a validated analysis view,
descriptive tables, ten focused EDA figures, a reproducible notebook, and feature
governance and leakage reviews. Stage 3 locks a stratified 80/20 holdout,
separates predictive features from audit attributes, and evaluates a dummy model
and one untuned Logistic Regression baseline using development-only CV. The final
test set remains untouched for predictive performance evaluation.

Stage 3A compares the existing semantics with UCI South German Credit dataset
573, which UCI describes as a corrected/background-enhanced representation of
the same data. The audit found material type and coding concerns. No model or
split was changed. See `docs/stage3a_semantics_audit.md`.

Stage 3R remediates those concerns without changing the data or locked split.
Only duration and transformed credit amount are scaled as quantitative features;
the other 15 predictive fields are one-hot encoded. Original Stage 3 results are
preserved under `reports/stage3_legacy_pre_semantic_audit/`. The authoritative
baseline is now under `reports/stage3_remediated/`.

Stage 4 compares the unchanged Stage 3R Logistic Regression with one fixed,
untuned Random Forest and one fixed, untuned XGBoost candidate. All three use the
same five development-only folds. The stage records aggregate and fold metrics,
0.50-threshold confusion matrices and 5:1 costs, probability-quality diagnostics,
train-versus-validation gaps, intrinsic feature importances, model agreement, and
hashed row-level OOF evidence under `reports/stage4/`. The final holdout remains
unscored, and no final model has been selected.

Stage 5 performs controlled XGBoost-only tuning with nested cross-validation.
The persisted Stage 4 folds are the outer evaluation folds; each outer-training
partition receives the same 40-draw, four-fold inner search selected by Average
Precision. Nested OOF predictions provide the honest development estimate. A
separate full-development search selects configuration metadata only and is not
reported as unbiased performance.

Stage 6 rejects the tuned Stage 5 candidate based on that nested evidence and
retains the fixed untuned Stage 4 XGBoost. Using development data only, nested
policy evaluation selects among no calibration, sigmoid, and isotonic calibration
by Brier score, then selects a threshold by the official `5 * FN + FP` cost. The
full model, feature, preprocessing, calibration, and threshold policy is frozen
under `reports/stage6/` before any final-holdout access.

Stage 7 performs the preregistered one-time final evaluation. After all hashes,
software checks, and the pre-access manifest were verified, the frozen five-fold
sigmoid-calibrated XGBoost ensemble generated exactly one probability and policy
decision for each of the 200 locked holdout rows at threshold 0.16. No policy
change followed access. Final discrimination was ROC-AUC 0.804643 and Average
Precision 0.679212; the frozen policy produced recall 0.883333, specificity
0.428571, Brier score 0.152338, and total assumed 5:1 cost 115. The remaining
predefined results were precision 0.398496, F1 0.549223, balanced accuracy
0.655952, log loss 0.475878, accuracy 0.565000, TN 60, FP 80, FN 7, TP 53,
and average cost 0.575000.

Stage 8 adds post-hoc explainability and responsible-AI auditing without reopening
model development. It explains the five actual frozen-architecture XGBoost
members on their raw tree-margin scale, aggregates one-hot contributions to the
17 governed source features, documents member stability and reproducibly selected
development cases, and audits honest development OOF outcomes across the three
audit-only attributes. It also records proxy risks, limited audit-feature
associations, error-group summaries, an explainability report, a responsible-AI
report, and a model card under `reports/stage8/`. SHAP values do not decompose the
final calibrated ensemble probability and are not causal effects.

Stage 9 packages the unchanged policy as `creditscope-model-1.0.0`. The complete
five-member sigmoid-calibrated estimator is fit using only the 800 development
rows and serialized under `artifacts/model/`. It adds a strict 17-field contract,
hash-verified loader, deterministic single/batch inference, development-derived
golden fixtures, schema, manifest, and local JSON CLI. This is production
engineering, not new model development; no holdout prediction or analysis occurs.

Stage 10 exposes that unchanged Stage 9 inference layer through FastAPI as
`creditscope-api-1.0.0`. The service loads and verifies the canonical model once
at startup, provides health and safe metadata routes, and supports strict single
and ordered batch prediction. It adds no modelling behavior, does not import SHAP
for prediction, and has no holdout, training, model-update, or threshold-update
route. See `docs/api.md`.

Stage 11 adds a Streamlit portfolio interface that communicates exclusively
with the Stage 10 FastAPI service over HTTP. It renders the governed 17-field
form with documented source-category labels, displays neutral good/bad
credit-risk results, and handles unavailable or rejected API requests without
raw traces. It does not load the model, reproduce threshold logic, request audit
attributes, retain payloads, or access holdout evidence. See `docs/frontend.md`.

Stage 12 adds separate, non-root API and UI container definitions, a private
two-service Compose topology, strict build-context controls, frozen-model Linux
verification, and push/pull-request CI. The Codex sandbox cannot access Docker;
therefore container runtime, Linux artifact compatibility, and Compose parity
remain an explicit external-validation checkpoint until the documented
PowerShell workflow is run and its output reviewed. See `docs/containers.md` and
`docs/ci.md`.

The data source is the official UCI Machine Learning Repository entry
[Statlog (German Credit Data), dataset ID 144](https://archive.ics.uci.edu/dataset/144/statlog%2Bgerman%2Bcredit%2Bdata),
donated by Hans Hofmann. UCI assigns DOI
[10.24432/C5NC77](https://doi.org/10.24432/C5NC77) and distributes the dataset
under CC BY 4.0. The project fetches it with UCI's documented `ucimlrepo`
interface rather than committing a copied dataset.

The verified source has 1,000 observations, 20 candidate features, no missing
values, and source target `class`: 1 = good credit risk and 2 = bad credit risk.
The analysis view preserves that source target and adds `bad_credit_risk` with
0 = good and 1 = bad. This target is not described as an observed default event.

See `docs/data_dictionary.md` for official feature and category semantics,
`reports/data_dictionary.csv` for the machine-readable dictionary, and
`reports/stage2_eda_summary.md` for the Stage 2 findings.

## Project structure

```text
CreditScope/
├── artifacts/          # generated model/ML artifacts (ignored)
├── configs/            # future configuration files
├── data/
│   ├── processed/      # future derived data (ignored)
│   └── raw/            # downloaded source snapshot (ignored)
├── docs/               # methodology and governance notes
├── notebooks/          # reproducible EDA notebook
├── reports/            # reproducible verification, EDA, and baseline reports
├── src/creditscope/    # installable application package
├── tests/              # automated tests
├── .gitignore
├── pyproject.toml
└── README.md
```

## Reproduce the project and run the API

From the repository root on PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev,api,ui]"
creditscope-verify-data --save-raw
creditscope-eda
creditscope-baseline
creditscope-tree-comparison
creditscope-tune-xgboost
creditscope-freeze-policy
creditscope-explainability-audit
creditscope-build-frozen-model
pytest
ruff check .
```

Run the verified API locally after the canonical Stage 9 artifact exists:

```powershell
.\.venv\Scripts\python.exe -m uvicorn creditscope.api:app --host 127.0.0.1 --port 8000
```

Swagger UI is available at `http://127.0.0.1:8000/docs`. A single request can be
sent with:

```powershell
$body = Get-Content examples/api_single_request.json -Raw
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/predict `
  -ContentType 'application/json' -Body $body
```

The API and model versions are `creditscope-api-1.0.0` and
`creditscope-model-1.0.0`. The strict field/category contract remains defined by
`artifacts/model/input_schema.json` and explained in `docs/inference_contract.md`.

With FastAPI still running in the first terminal, start the Streamlit frontend
in a second terminal:

```powershell
.\.venv\Scripts\python.exe -m streamlit run src/creditscope/ui.py --server.address 127.0.0.1 --server.port 8501
```

Open `http://127.0.0.1:8501`. The frontend defaults to the local API at
`http://127.0.0.1:8000`; `CREDITSCOPE_API_URL` may point it to another trusted
CreditScope API. Model policy values are not frontend configuration. See
`docs/frontend.md` for architecture, validation, privacy, and troubleshooting.

Run the complete two-service demo with Docker Desktop:

```powershell
docker compose build
docker compose up --detach --wait --no-build
docker compose ps
```

Open Streamlit at `http://127.0.0.1:8501`, FastAPI at
`http://127.0.0.1:8000`, or Swagger at `http://127.0.0.1:8000/docs`. The full
cross-platform model, endpoint, networking, and golden-parity validation is:

```powershell
.\scripts\manual_docker_validation.ps1
```

Stop the stack with `docker compose down --volumes --remove-orphans`.

The verification command fetches UCI dataset ID 144, validates its identity and
documented dimensions, checks missing values and target labels, writes the
ignored source snapshot to `data/raw/german_credit.csv`, and writes a compact
evidence report to `reports/data_verification.json`.

Internet access is required for the UCI fetch. Re-running the command overwrites
only those two generated files.

`creditscope-eda` refetches and validates the official source through the Stage 1
loader, then recreates the Stage 2 dictionaries, summary tables, findings report,
and ten figures. It verifies that the raw CSV hash is unchanged.

The notebook `notebooks/02_exploratory_data_analysis.ipynb` calls the same reusable
package functions and can be executed from the repository root.

`creditscope-baseline` verifies the unchanged Stage 1 snapshot against its
recorded SHA-256, verifies `reports/split_manifest.csv`, and regenerates the
authoritative development-only baseline under `reports/stage3_remediated/`. See
`reports/stage3_remediated/stage3r_baseline_summary.md` and
`notebooks/03_baseline_modelling.ipynb`.

`creditscope-tree-comparison` preserves the split and Stage 3R reference, then
compares Logistic Regression, Random Forest, and XGBoost with the same five
development folds. It writes only development-set artifacts under
`reports/stage4/`; it never computes final-holdout predictions or metrics.

`creditscope-tune-xgboost` verifies the locked split and Stage 4 fold/OOF hashes,
runs the predefined nested search, writes development-only evidence under
`reports/stage5/`, and performs the separate full-development configuration
search without refitting a final estimator or using the holdout.

`creditscope-freeze-policy` preserves all prior evidence, evaluates calibration
and threshold selection through the fixed Stage 4 outer folds, then performs the
separate full-development policy selection. It writes development-only evidence,
the frozen policy, and the predefined one-time holdout protocol under
`reports/stage6/` without reading or scoring holdout observations.

`creditscope-explainability-audit` verifies every locked Stage 6P/7 hash and then
uses only the 800 development observations. It deterministically reconstructs
the frozen five-member ensemble architecture for post-hoc inspection and writes
Stage 8 evidence. It does not rerun the irreversible Stage 7 holdout evaluation.

`creditscope-build-frozen-model` verifies the freeze, selects exactly the locked
development IDs, fits and serializes the frozen ensemble, and performs a golden
reload check. Local inference is:

```powershell
.\.venv\Scripts\python.exe -m creditscope.inference examples/sample_prediction_input.json
```

See `docs/inference_contract.md` and `docs/model_artifact.md`. Joblib uses pickle
semantics; only trusted, hash-verified internal artifacts may be loaded.

## Responsible-use and data limitations

The dataset is small, represents 1973–1975 credits, is geographically specific,
and does not fully document measurement timing. Bad credits were heavily
oversampled, and credit amount underwent an unknown monotonic transformation. It
includes age, a compound personal-status/sex field, foreign-worker status, and
several potential proxy variables. Jurisdiction-specific protected-class claims
are intentionally not made. The baseline excludes `personal_status_sex`,
`age_years`, and `foreign_worker` from prediction while retaining them for later
audit analysis.
Stage 3A establishes that the compound personal-status/sex field cannot recover
sex cleanly, so it must not support standalone sex-group fairness claims. Proxy
risks remain under review. CreditScope remains an educational/research
decision-support prototype.

## Reproducibility policy

Stochastic work uses `random_state=42` where supported. The locked final test set
contains 200 rows and is excluded from Stage 3R performance reporting, feature or
preprocessing decisions, model selection, and threshold selection. Stage 3R used
threshold 0.50; no tuning occurred and no final model was selected in that stage.

Stage 4 keeps the threshold at 0.50 and adds no class weighting, resampling,
probability calibration, SHAP analysis, or hyperparameter search. Random Forest
and XGBoost are candidates rather than selected or production models.

Stage 5 tunes XGBoost only. Random Forest is not tuned because Stage 4 did not
justify expanding its search. Average Precision is the inner selection objective.
The threshold remains 0.50 and the 5:1 cost is monitored rather than optimized.
No class/sample weighting, resampling, calibration, SHAP, or holdout evaluation
occurs.

Stage 5 tuning is retained as negative evidence: its honest nested estimate was
worse than the fixed Stage 4 XGBoost on the primary and key operational metrics.
Stage 6 therefore retains the untuned model. Calibration and threshold selection
are development-only; Brier score selects calibration and the official 5:1 cost
selects threshold. The theoretical `1/6` threshold is a reference rather than a
forced choice. Stage 6 freezes the complete policy before the one-time holdout
evaluation defined for Stage 7.

Stage 7 accessed the sealed holdout once after the policy freeze. The evaluated
policy is the fixed Stage 4 XGBoost pipeline inside sigmoid
`CalibratedClassifierCV` with the exact persisted five development folds,
`ensemble=True`, and threshold 0.16. No alternative model, calibration, or
threshold was evaluated afterward. The 5:1 error cost remains an assumed
educational-prototype structure, not a claim about real lending economics.

Stage 8 is post-hoc only. The principal explanation is original-feature aggregate
absolute SHAP on the underlying raw XGBoost margin. It is not an exact explanation
of sigmoid calibration, five-member probability averaging, or threshold 0.16.
Audit-only attributes remain excluded from prediction and are joined only to
development OOF evidence. Subgroup differences cannot establish fairness, and
the compound personal-status/sex field cannot support clean sex or modern-gender
comparisons.

Stage 9 keeps prediction separate from explanation: core inference does not
import SHAP. It accepts exactly 17 governed fields, rejects audit-only and
unexpected fields, and emits only calibrated bad-risk probability, good/bad
credit-risk classification, threshold, and model version. It never emits lending
approval, rejection, or eligibility.

Stage 10 and Stage 11 preserve that separation. FastAPI owns the verified model
process; Streamlit calls FastAPI only and contains no model loading, fitting,
thresholding, XGBoost, SHAP, training-data, or holdout-data path.

Stage 12 changes only infrastructure. Its API image loads the unchanged,
hash-verified Stage 9 artifact; its UI image contains no model or inference
stack. Container startup cannot fit or rebuild the model. CI validates evidence,
quality, services, and images but performs no deployment or model development.

The Stage 3R reconciliation policy requires exact equality for hashes, membership,
counts, classifications, integer costs, feature boundaries, and fixed model
configuration. Reproduced floating-point metrics use `rtol=0` and `atol=1e-10`;
see `docs/stage3r_reproducibility_reconciliation.md`. Stage 4 records hashed,
development-only fold membership and row-level OOF predictions for future checks.
