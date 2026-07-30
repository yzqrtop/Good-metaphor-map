# GoodM Framework PowerShell Evaluation Script
# Usage: .\evaluate.ps1 [options]

param(
    [int]$RQ,
    [switch]$All,
    [string]$Model,
    [string]$Dataset,
    [string]$Output,
    [switch]$Help
)

if ($Help) {
    Write-Host "GoodM Framework Evaluation Script"
    Write-Host "Usage: .\evaluate.ps1 [-RQ 1-6] [-All] [-Model name] [-Dataset path] [-Output path]"
    Write-Host ""
    Write-Host "Options:"
    Write-Host "  -RQ [1-6]       Run specific RQ experiment"
    Write-Host "  -All             Run all RQ experiments"
    Write-Host "  -Model [name]    Specify LLM model (e.g., gpt-4, qwen-plus)"
    Write-Host "  -Dataset [path]  Path to dataset file"
    Write-Host "  -Output [path]   Path to save results"
    Write-Host "  -Help            Show help information"
    return
}

# Build command
$cmdArgs = @()

if ($All) {
    $cmdArgs += "--all"
} elseif ($RQ) {
    $cmdArgs += "--rq", $RQ
} else {
    Write-Host "Error: Please specify -RQ [1-6] or -All"
    exit 1
}

if ($Model) {
    $cmdArgs += "--model", $Model
}

if ($Dataset) {
    $cmdArgs += "--dataset", $Dataset
}

if ($Output) {
    $cmdArgs += "--output", $Output
}

Write-Host "Running: python run_experiments.py $cmdArgs"
python run_experiments.py @cmdArgs