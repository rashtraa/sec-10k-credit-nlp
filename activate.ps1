# Run this INSTEAD of .venv\Scripts\Activate.ps1, every session.
#
# It activates the X: environment and redirects every cache that would
# otherwise write to C:. Sourcing it is what keeps your C: drive from
# filling up during stage 3, when the embedding model downloads.
#
#   cd X:\Projects\edgar-credit-signal
#   .\activate.ps1

$ErrorActionPreference = "Stop"

$PROJ = "X:\Projects\edgar-credit-signal"

if (-not (Test-Path $PROJ)) {
    Write-Host "Project folder not found: $PROJ" -ForegroundColor Red
    Write-Host "Unzip the archive there first." -ForegroundColor Red
    exit 1
}

# If an old environment is active in this shell, shed it. A stale
# VIRTUAL_ENV is the usual reason `python` keeps resolving to the C: venv
# even after you cd elsewhere.
if ($env:VIRTUAL_ENV) {
    Write-Host "shedding active environment: $env:VIRTUAL_ENV" -ForegroundColor Yellow
    if (Get-Command deactivate -ErrorAction SilentlyContinue) { deactivate }
    Remove-Item Env:\VIRTUAL_ENV -ErrorAction SilentlyContinue
}

Set-Location $PROJ

if (-not (Test-Path "$PROJ\.venv\Scripts\python.exe")) {
    Write-Host "No .venv here yet. Create it with:" -ForegroundColor Red
    Write-Host "  py -3.12 -m venv .venv" -ForegroundColor Red
    exit 1
}

& "$PROJ\.venv\Scripts\Activate.ps1"

# --- redirect every cache that defaults to C:\Users\<you>\ ----------------
$CACHE = "$PROJ\.cache"
New-Item -ItemType Directory -Force -Path `
    "$CACHE\pip", "$CACHE\hf", "$CACHE\torch", "$CACHE\tmp" | Out-Null

$env:EDGAR_DATA_ROOT = "$PROJ\data"   # our own outputs
$env:PIP_CACHE_DIR   = "$CACHE\pip"   # wheel cache, several GB over time
$env:HF_HOME         = "$CACHE\hf"    # sentence-transformers models (~1 GB)
$env:TRANSFORMERS_CACHE = "$CACHE\hf"
$env:TORCH_HOME      = "$CACHE\torch"
$env:TMP             = "$CACHE\tmp"
$env:TEMP            = "$CACHE\tmp"

# ---------------------------------------------------------------- .env
# The SEC requires a contact User-Agent on every request, and that is
# personal data, so it is NOT hardcoded in config.py. Put it in .env
# (which is gitignored) as a single line:
#
#     SEC_USER_AGENT=Your Name your.email@domain.com
#
# Any other KEY=VALUE lines in .env are loaded the same way.
$envFile = "$PROJ\.env"
if (Test-Path $envFile) {
    Get-Content $envFile | ForEach-Object {
        $line = $_.Trim()
        if ($line -and -not $line.StartsWith('#') -and $line.Contains('=')) {
            $parts = $line.Split('=', 2)
            [Environment]::SetEnvironmentVariable($parts[0].Trim(), $parts[1].Trim(), 'Process')
        }
    }
}
if (-not $env:SEC_USER_AGENT) {
    Write-Host "WARNING: SEC_USER_AGENT is not set - the SEC will reject requests." -ForegroundColor Yellow
    Write-Host "         Add it to .env, then run this script again." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "environment active on X:" -ForegroundColor Green
Write-Host "  python : $((Get-Command python).Source)"
Write-Host "  data   : $env:EDGAR_DATA_ROOT"
Write-Host "  caches : $CACHE"
Write-Host "  sec UA : $env:SEC_USER_AGENT"
Write-Host ""
Write-Host "Verify with:  python env_check.py" -ForegroundColor Cyan
