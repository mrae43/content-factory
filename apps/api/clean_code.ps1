# Ensure Ruff is installed
if (!(Get-Command ruff -ErrorAction SilentlyContinue)) {
    Write-Host "Ruff not found. Installing..." -ForegroundColor Yellow
    pip install ruff
}

Write-Host "--- Gravity-Defying Indentation Fix Incoming ---" -ForegroundColor Cyan

# 'ruff format' handles the PEP-8 indentation and spacing
ruff format .

# 'ruff check --fix' handles the logic, unused imports, and linting
ruff check . --fix

Write-Host "--- Your codebase is now PEP-8 compliant! ---" -ForegroundColor Green