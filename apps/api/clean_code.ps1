Write-Host "--- Gravity-Defying Indentation Fix Incoming ---" -ForegroundColor Cyan

# 'ruff format' handles the PEP-8 indentation and spacing
uv run ruff format .

# 'ruff check --fix' handles the logic, unused imports, and linting
uv run ruff check . --fix

Write-Host "--- Your codebase is now PEP-8 compliant! ---" -ForegroundColor Green