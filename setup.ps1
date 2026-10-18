# One-shot setup. Run this once, from the project folder, in PowerShell.
#
#   cd X:\Projects\edgar-credit-signal
#   .\setup.ps1
#
# It creates the virtual environment, points every cache at this folder so
# nothing lands on C:, installs pinned dependencies, and runs the offline
# tests. It stops at the first thing that goes wrong and tells you what.

$ErrorActionPreference = "Stop"

function Step($n, $msg) { Write-Host "`n[$n] $msg" -ForegroundColor Cyan }
function Ok($msg)       { Write-Host "    ok   $msg" -ForegroundColor Green }
function Die($msg)      { Write-Host "`nSTOPPED: $msg" -ForegroundColor Red; exit 1 }

$PROJ = $PSScriptRoot
Write-Host "`n=== edgar-credit-signal setup ===" -ForegroundColor White
Write-Host "project: $PROJ"

# ---------------------------------------------------------------- 1. sanity
Step 1 "Checking location and files"

if ($PROJ -like "C:\*") {
    Write-Host "    WARNING: project is on C:. You said C: is short on space." -ForegroundColor Yellow
    Write-Host "    Expected somewhere like X:\Projects\edgar-credit-signal" -ForegroundColor Yellow
    $ans = Read-Host "    Continue anyway? (y/N)"
    if ($ans -ne "y") { Die "Move the folder to X: and rerun." }
}
if ($PROJ -like "*OneDrive*") {
    Die "This folder is inside OneDrive. Sync locks files mid-install, which is what broke the first attempt. Move it to X:\Projects\edgar-credit-signal and rerun."
}

$needed = @("config.py", "sec_client.py", "build_universe.py",
            "download_filings.py", "sections.py", "env_check.py",
            "test_parsing.py", "test_sections.py",
            "requirements-stage1.txt", "activate.ps1")
$missing = $needed | Where-Object { -not (Test-Path (Join-Path $PROJ $_)) }
if ($missing) { Die "Missing files: $($missing -join ', ')`n  The zip did not extract here, or extracted into a subfolder." }
Ok "all $($needed.Count) expected files present"

$drive = (Get-Item $PROJ).PSDrive.Name
$free = [math]::Round((Get-PSDrive $drive).Free / 1GB, 1)
if ($free -lt 3) { Die "Only $free GB free on ${drive}:. Need about 3 GB." }
Ok "$free GB free on ${drive}:"

# ---------------------------------------------------------------- 2. python
Step 2 "Locating Python 3.12"

$py = $null
foreach ($c in @("py -3.12", "py -3.11", "python")) {
    try {
        $parts = $c.Split(" ")
        $v = & $parts[0] $parts[1..($parts.Length-1)] -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null
        if ($v -match "^3\.(11|12)$") { $py = $c; Ok "using '$c' (Python $v)"; break }
    } catch { }
}
if (-not $py) { Die "No Python 3.11 or 3.12 found. Install 3.12 from python.org, tick 'Add to PATH', reopen PowerShell." }

# ---------------------------------------------------------------- 3. venv
Step 3 "Creating a fresh virtual environment"

$venv = Join-Path $PROJ ".venv"
if (Test-Path $venv) {
    Write-Host "    removing the previous .venv"
    try { Remove-Item -Recurse -Force $venv }
    catch { Die "Could not delete .venv - something is using it. Close every terminal, VS Code, and Jupyter kernel, then rerun." }
}
$parts = $py.Split(" ")
& $parts[0] $parts[1..($parts.Length-1)] -m venv $venv
if (-not (Test-Path "$venv\Scripts\python.exe")) { Die "venv creation failed." }
Ok "created $venv"

$PY = "$venv\Scripts\python.exe"

# ---------------------------------------------------------------- 4. caches
Step 4 "Redirecting caches away from C:"

$CACHE = Join-Path $PROJ ".cache"
foreach ($d in @("pip", "hf", "torch", "tmp")) {
    New-Item -ItemType Directory -Force -Path (Join-Path $CACHE $d) | Out-Null
}
$env:EDGAR_DATA_ROOT      = Join-Path $PROJ "data"
$env:PIP_CACHE_DIR        = Join-Path $CACHE "pip"
$env:HF_HOME              = Join-Path $CACHE "hf"
$env:TRANSFORMERS_CACHE   = Join-Path $CACHE "hf"
$env:TORCH_HOME           = Join-Path $CACHE "torch"
$env:TMP                  = Join-Path $CACHE "tmp"
$env:TEMP                 = Join-Path $CACHE "tmp"
Ok "pip, huggingface, torch and temp now write to $CACHE"

# ---------------------------------------------------------------- 5. install
Step 5 "Installing pinned dependencies (1-2 minutes)"

& $PY -m pip install --upgrade pip --quiet
& $PY -m pip install -r (Join-Path $PROJ "requirements-stage1.txt") --quiet
if ($LASTEXITCODE -ne 0) { Die "pip install failed - scroll up for the reason." }

$vers = & $PY -c "import pandas, numpy; print(pandas.__version__, numpy.__version__)"
if ($vers -notmatch "^2\.2\.\d+ 1\.26\.\d+$") {
    Write-Host "    got '$vers', expected '2.2.3 1.26.4' - forcing reinstall" -ForegroundColor Yellow
    & $PY -m pip install -r (Join-Path $PROJ "requirements-stage1.txt") --force-reinstall --quiet
    $vers = & $PY -c "import pandas, numpy; print(pandas.__version__, numpy.__version__)"
}
Ok "pandas/numpy $vers"

# ---------------------------------------------------------------- 6. verify
Step 6 "Verifying nothing writes to C:"
& $PY (Join-Path $PROJ "env_check.py")
if ($LASTEXITCODE -ne 0) { Die "env_check failed - see the FAIL lines above." }

Step 7 "Running offline tests (no network)"
foreach ($t in @("test_parsing.py", "test_sections.py")) {
    $out = & $PY (Join-Path $PROJ $t) 2>&1 | Out-String
    if ($out -notmatch "ALL PASS") { Write-Host $out; Die "$t failed." }
    Ok "$t"
}

# ---------------------------------------------------------------- done
Write-Host "`n=== SETUP COMPLETE ===`n" -ForegroundColor Green
Write-Host "Next, in this same window:`n"
Write-Host "    .\activate.ps1"           -ForegroundColor White
Write-Host "    python build_universe.py    # ~5 min"   -ForegroundColor White
Write-Host "    python download_filings.py  # ~45-90 min, resumable" -ForegroundColor White
Write-Host "`nEvery future session: cd here, then .\activate.ps1`n"
