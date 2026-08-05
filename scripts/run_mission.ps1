$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    throw ".venv not found. Run scripts\first_setup.ps1 once."
}

& .\.venv\Scripts\python.exe examples\run_mission.py
