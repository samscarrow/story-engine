param(
  [switch]$Live,
  [string]$LmEndpoint,
  [string]$LmModel
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Write-Header([string]$Title) {
  Write-Host "";
  Write-Host ("=== {0} ===" -f $Title) -ForegroundColor Cyan
}

function Invoke-Pytest([string[]]$Args) {
  & pytest @Args
  if ($LASTEXITCODE -ne 0) {
    throw "Pytest failed with exit code $LASTEXITCODE for: pytest $($Args -join ' ')"
  }
}

function With-TempEnv($Pairs, [scriptblock]$Body) {
  $backup = @{}
  foreach ($k in $Pairs.Keys) { $backup[$k] = [System.Environment]::GetEnvironmentVariable($k, 'Process') }
  try {
    foreach ($k in $Pairs.Keys) { [System.Environment]::SetEnvironmentVariable($k, $Pairs[$k], 'Process') }
    & $Body
  }
  finally {
    foreach ($k in $Pairs.Keys) { [System.Environment]::SetEnvironmentVariable($k, $backup[$k], 'Process') }
  }
}

# -------------------- Granular (non-live) --------------------
Write-Header "Granular • Orchestrator Loader"
Invoke-Pytest @(' -q ', 'tests/test_orchestrator_loader.py::test_loader_registers_providers')

Write-Header "Granular • POML Adapter"
Invoke-Pytest @(' -q ', 'tests/test_poml_adapter.py')

Write-Header "Granular • POML Extended + Config Flag"
Invoke-Pytest @(' -q ', 'tests/test_poml_adapter_extended.py', 'tests/test_poml_config_flag.py')

Write-Header "Granular • Pipeline + Orchestrated Integration"
Invoke-Pytest @(' -q ', 'tests/test_pipeline_smoke.py', 'tests/test_orchestrated_poml_integration.py::test_orchestrated_scene_and_dialogue_with_poml')

# -------------------- Live (optional) --------------------
if ($Live -or ($env:LM_ENDPOINT -and $env:LMSTUDIO_MODEL)) {
  $ep = if ($LmEndpoint) { $LmEndpoint } else { $env:LM_ENDPOINT }
  $model = if ($LmModel) { $LmModel } else { $env:LMSTUDIO_MODEL }
  if (-not $ep -or -not $model) {
    throw "Live requested but LM_ENDPOINT or LMSTUDIO_MODEL is missing. Provide -LmEndpoint and -LmModel or set env vars."
  }

  $envPairs = @{ 'LM_ENDPOINT' = $ep; 'LMSTUDIO_MODEL' = $model; 'STORY_ENGINE_LIVE' = '1' }

  With-TempEnv $envPairs {
    Write-Header "Live • Single Flow (Pilate POML)"
    Invoke-Pytest @(' -q ', 'tests/test_live_pilate_poml.py::test_live_poml_pilate_flow_minimal')

    Write-Header "Live • Full Suite"
    Invoke-Pytest @(' -q ')
  }
} else {
  Write-Host "(Live tests skipped — pass -Live and provide -LmEndpoint/-LmModel or set LM_ENDPOINT/LMSTUDIO_MODEL)" -ForegroundColor Yellow
}

Write-Host "`nAll requested tests completed successfully." -ForegroundColor Green

