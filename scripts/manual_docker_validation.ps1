$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot

$apiImage = "creditscope-api:1.0.0"
$uiImage = "creditscope-ui:1.0.0"
$apiContainer = "creditscope-api-stage12-check"
$expectedModelHash = "c92e062e1cdde4965a7130be244cba8d01bc95e2b0d0cbb8a7feda43fbc3bcb7"

function Wait-HttpOk {
    param(
        [Parameter(Mandatory = $true)][string]$Uri,
        [int]$Attempts = 60
    )
    for ($attempt = 1; $attempt -le $Attempts; $attempt++) {
        try {
            $response = Invoke-WebRequest -UseBasicParsing -Uri $Uri -TimeoutSec 2
            if ($response.StatusCode -eq 200) { return }
        }
        catch {
            if ($attempt -eq $Attempts) { throw }
        }
        Start-Sleep -Seconds 2
    }
    throw "Timed out waiting for $Uri"
}

Write-Host "[1/10] Docker prerequisite and daemon checks"
docker --version
docker compose version
docker context show
docker info --format "Server={{.ServerVersion}} OS={{.OperatingSystem}} Arch={{.Architecture}}"
docker run --rm hello-world | Out-Host

Write-Host "[2/10] Frozen model hash"
$actualModelHash = (Get-FileHash -Algorithm SHA256 -LiteralPath "artifacts/model/creditscope_frozen_model.joblib").Hash.ToLowerInvariant()
if ($actualModelHash -ne $expectedModelHash) {
    throw "Canonical model hash mismatch: $actualModelHash"
}

Write-Host "[3/10] Build separate API and UI images"
docker build --file Dockerfile.api --tag $apiImage .
docker build --file Dockerfile.ui --tag $uiImage .

Write-Host "[4/10] Verify non-root image users and image separation"
$apiUser = docker image inspect $apiImage --format "{{.Config.User}}"
$uiUser = docker image inspect $uiImage --format "{{.Config.User}}"
if ([string]::IsNullOrWhiteSpace($apiUser) -or $apiUser -eq "0" -or $apiUser -eq "root") { throw "API image is not configured with a non-root user." }
if ([string]::IsNullOrWhiteSpace($uiUser) -or $uiUser -eq "0" -or $uiUser -eq "root") { throw "UI image is not configured with a non-root user." }
docker run --rm --entrypoint python $uiImage -c "from pathlib import Path; import importlib.util; assert not Path('/app/artifacts/model/creditscope_frozen_model.joblib').exists(); assert importlib.util.find_spec('xgboost') is None; assert importlib.util.find_spec('joblib') is None; assert importlib.util.find_spec('shap') is None; print('UI image isolation passed')"

Write-Host "[5/10] Linux canonical-model and golden-fixture verification"
docker run --rm $apiImage python -m creditscope.container_verify

Write-Host "[6/10] Standalone API smoke"
docker rm --force $apiContainer 2>$null | Out-Null
try {
    docker run --detach --name $apiContainer --publish 127.0.0.1:8000:8000 $apiImage | Out-Host
    Wait-HttpOk -Uri "http://127.0.0.1:8000/health"
    Invoke-RestMethod -Uri "http://127.0.0.1:8000/health" | ConvertTo-Json -Depth 10
    Invoke-RestMethod -Uri "http://127.0.0.1:8000/model-info" | ConvertTo-Json -Depth 10
    $singleBody = Get-Content -Raw -LiteralPath "examples/api_single_request.json"
    Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/predict" -ContentType "application/json" -Body $singleBody | ConvertTo-Json -Depth 10
    $batchBody = Get-Content -Raw -LiteralPath "examples/api_batch_request.json"
    Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/predict-batch" -ContentType "application/json" -Body $batchBody | ConvertTo-Json -Depth 10
}
finally {
    docker logs $apiContainer 2>$null | Out-Host
    docker rm --force $apiContainer 2>$null | Out-Null
}

Write-Host "[7/10] Build and start Compose stack"
docker compose build
try {
    docker compose up --detach --wait --no-build
    docker compose ps

    Write-Host "[8/10] Compose API, UI, and service-network checks"
    Wait-HttpOk -Uri "http://127.0.0.1:8000/health"
    Wait-HttpOk -Uri "http://127.0.0.1:8501/_stcore/health"
    Invoke-RestMethod -Uri "http://127.0.0.1:8000/health" | ConvertTo-Json -Depth 10
    Invoke-RestMethod -Uri "http://127.0.0.1:8000/model-info" | ConvertTo-Json -Depth 10
    Invoke-WebRequest -UseBasicParsing -Uri "http://127.0.0.1:8501" | Select-Object StatusCode
    docker compose exec -T ui python -c "from creditscope.api_client import CreditScopeAPIClient; c=CreditScopeAPIClient(); assert c.health()['status']=='ok'; assert c.model_info()['model_version']=='creditscope-model-1.0.0'; c.close(); print('UI-to-API connectivity passed')"

    Write-Host "[9/10] End-to-end golden smoke"
    .\.venv\Scripts\python.exe scripts\smoke_compose.py
}
finally {
    Write-Host "Compose logs follow for auditability"
    docker compose logs --no-color --tail 200
    docker compose down --volumes --remove-orphans
}

Write-Host "[10/10] Final Docker status"
docker compose ps --all
docker image inspect $apiImage $uiImage --format "{{.RepoTags}} user={{.Config.User}} size={{.Size}}"
Write-Host "STAGE 12 EXTERNAL DOCKER VALIDATION PASSED"
