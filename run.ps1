# Run any project command with the project's virtualenv — from any terminal,
# activated or not.
#
#   .\run.ps1 python spark_features.py --check
#   .\run.ps1 pytest -q
#   .\run.ps1 uvicorn api:app --port 8000
#
# WHY THIS EXISTS
# A venv is per-terminal state. Every new PowerShell window starts without it,
# and the failure is never "the venv is not active" — it is
# `ModuleNotFoundError: No module named tqdm`, or `No module named pyspark`, or
# `java is not recognized`, each of which reads like something is broken rather
# than like you are in the wrong terminal.
#
# If this wrapper ever misbehaves, the thing it is a shortcut for always works:
#
#   .\.venv\Scripts\python.exe spark_features.py --check

[CmdletBinding()]
param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Command)

$ErrorActionPreference = "Stop"

# @(...) forces an array even when PowerShell hands back a single scalar.
# Without it, `$Command[1..0]` returns a bare string, and splatting a string
# passes it one CHARACTER at a time — which is how `spark_doctor.py` reached
# Python as the filename `s`.
$all = @($Command)

if ($all.Count -eq 0) {
    Write-Host "usage: .\run.ps1 <command> [args...]" -ForegroundColor Yellow
    Write-Host "   e.g. .\run.ps1 python spark_features.py --check"
    exit 2
}

$scripts = Join-Path $PSScriptRoot ".venv\Scripts"
$python  = Join-Path $scripts "python.exe"

if (-not (Test-Path $python)) {
    Write-Host "No virtualenv at $scripts" -ForegroundColor Red
    Write-Host "Create it with:  python -m venv .venv"
    exit 1
}

# Put the venv first on PATH for the child process only. This is what
# `activate` does, minus the persistence and minus the ceremony.
$env:PATH        = "$scripts;$env:PATH"
$env:VIRTUAL_ENV = Join-Path $PSScriptRoot ".venv"

$exe  = [string]$all[0]
$rest = @($all | Select-Object -Skip 1)

# Resolve the executable inside the venv when one exists there, so `python`,
# `pytest` and `uvicorn` all reach the project's copies rather than whatever
# Anaconda or the Microsoft Store alias would supply.
if ($exe -eq "python" -or $exe -eq "python.exe") {
    $exe = $python
} elseif (Test-Path (Join-Path $scripts "$exe.exe")) {
    $exe = Join-Path $scripts "$exe.exe"
}

Write-Verbose "exec: $exe $($rest -join ' ')"

if ($rest.Count -eq 0) { & $exe } else { & $exe @rest }
exit $LASTEXITCODE
