$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

if (-not (Test-Path ".git")) {
    throw "This folder is not a Git clone. Clone the GitHub repository once before using update_project.ps1."
}
if (-not (Test-Path ".venv\Scripts\python.exe")) {
    throw ".venv not found. Run scripts\first_setup.ps1 once."
}

Write-Host "Pulling the newest code..."
git pull --ff-only
& .\.venv\Scripts\python.exe -m pip install -e ".[dev]"
& .\.venv\Scripts\python.exe -m pytest
Write-Host "Update complete. Your existing .venv was kept."
