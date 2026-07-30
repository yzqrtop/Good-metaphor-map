# GoodM Framework PowerShell Start Script
# Usage: .\start.ps1 [options]

param(
    [switch]$Install,
    [switch]$Test,
    [switch]$Example,
    [int]$RQ,
    [switch]$Help
)

if ($Help) {
    Write-Host "GoodM Framework Start Script"
    Write-Host "Usage: .\start.ps1 [-Install] [-Test] [-Example] [-RQ 1-6] [-Help]"
    Write-Host ""
    Write-Host "Options:"
    Write-Host "  -Install    Install dependencies"
    Write-Host "  -Test       Run tests"
    Write-Host "  -Example    Run basic usage example"
    Write-Host "  -RQ [1-6]   Run specific RQ experiment"
    Write-Host "  -Help       Show help information"
    return
}

if ($Install) {
    Write-Host "Installing GoodM framework..."
    pip install -e .
    pip install -e ".[dev]"
    Write-Host "Installation complete!"
} elseif ($Test) {
    Write-Host "Running tests..."
    python -m pytest tests/ -v
} elseif ($Example) {
    Write-Host "Running basic usage example..."
    python examples/basic_usage.py
} elseif ($RQ) {
    Write-Host "Running RQ$RQ experiment..."
    python run_experiments.py --rq $RQ
} else {
    Write-Host "Error: Please specify -Install, -Test, -Example, or -RQ [1-6]"
    Write-Host "Use -Help for more information"
    exit 1
}