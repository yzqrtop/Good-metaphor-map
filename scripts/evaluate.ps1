#!/usr/bin/env pwsh

# GoodM project evaluation script (PowerShell version)

# Show help information
function Show-Help {
    Write-Host "GoodM project evaluation script"
    Write-Host "Usage: .\evaluate.ps1 [options]"
    Write-Host ""
    Write-Host "Options:"
    Write-Host "  -Generator        Evaluate seven-tuple generation quality"
    Write-Host "  -Evaluator        Evaluate evaluator performance"
    Write-Host "  -Search           Evaluate search service performance"
    Write-Host "  -Xcultural        Evaluate cross-cultural labeling accuracy"
    Write-Host "  -All              Evaluate all components"
    Write-Host "  -Help             Show help information"
}

# Evaluate seven-tuple generation quality
function Evaluate-Generator {
    Write-Host "Evaluating seven-tuple generation quality..."
    
    # Check if generation result file exists
    if (-not (Test-Path "data/gen/goodm_raw.jsonl")) {
        Write-Host "Error: Generation result file does not exist, please run generator first"
        return
    }
    
    # Count generation results
    $totalRecords = (Get-Content "data/gen/goodm_raw.jsonl" | Measure-Object -Line).Lines
    Write-Host "Total generated records: $totalRecords"
    
    # Check generation result format
    Write-Host "Checking generation result format..."
    $validRecords = 0
    $invalidRecords = 0
    
    Get-Content "data/gen/goodm_raw.jsonl" | ForEach-Object {
        try {
            $record = $_.Trim() | ConvertFrom-Json
            if ($record.E -and $record.V -and $record.EE -and $record.VE -and $record.C -and $record.Sc -and $record.Sy) {
                $validRecords++
            } else {
                $invalidRecords++
            }
        } catch {
            $invalidRecords++
        }
    }
    
    Write-Host "Valid records: $validRecords"
    Write-Host "Invalid records: $invalidRecords"
    if ($totalRecords -gt 0) {
        $validRate = [math]::Round(($validRecords * 100.0) / $totalRecords, 2)
        Write-Host "Validity rate: $validRate%"
    }
}

# Evaluate evaluator performance
function Evaluate-Evaluator {
    Write-Host "Evaluating evaluator performance..."
    
    # Check if evaluation result file exists
    if (-not (Test-Path "data/scored/goodm_scored.jsonl")) {
        Write-Host "Error: Evaluation result file does not exist, please run evaluator first"
        return
    }
    
    # Count evaluation results
    $totalRecords = (Get-Content "data/scored/goodm_scored.jsonl" | Measure-Object -Line).Lines
    Write-Host "Total evaluated records: $totalRecords"
    
    # Calculate average MEF score
    Write-Host "Calculating average MEF score..."
    $totalMef = 0
    $recordCount = 0
    
    Get-Content "data/scored/goodm_scored.jsonl" | ForEach-Object {
        try {
            $record = $_.Trim() | ConvertFrom-Json
            $totalMef += $record.MEF
            $recordCount++
        } catch {
            # Ignore invalid records
        }
    }
    
    if ($recordCount -gt 0) {
        $avgMef = [math]::Round($totalMef / $recordCount, 4)
        Write-Host "Average MEF score: $avgMef"
    }
    
    # Calculate average scores for each dimension
    Write-Host "Calculating average scores for each dimension..."
    foreach ($dim in @("f1", "f2", "f3", "f4", "f5", "f6", "f7")) {
        $totalDim = 0
        $dimCount = 0
        
        Get-Content "data/scored/goodm_scored.jsonl" | ForEach-Object {
            try {
                $record = $_.Trim() | ConvertFrom-Json
                $totalDim += $record.$dim
                $dimCount++
            } catch {
                # Ignore invalid records
            }
        }
        
        if ($dimCount -gt 0) {
            $avgDim = [math]::Round($totalDim / $dimCount, 4)
            Write-Host "Average $dim score: $avgDim"
        }
    }
}

# Evaluate search service performance
function Evaluate-Search {
    Write-Host "Evaluating search service performance..."
    
    # Check if search service is running
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:8000/health" -ErrorAction Stop
    } catch {
        Write-Host "Warning: Search service is not running, please start search service first"
        return
    }
    
    # Test search functionality
    Write-Host "Testing search functionality..."
    $testQueries = @("井底之蛙", "守株待兔", "画蛇添足", "掩耳盗铃", "亡羊补牢")
    
    foreach ($query in $testQueries) {
        Write-Host "Test query: $query"
        try {
            $body = @{ "query" = $query; "k" = 3 } | ConvertTo-Json
            $response = Invoke-WebRequest -Uri "http://localhost:8000/search" -Method POST -Body $body -ContentType "application/json" -ErrorAction Stop
            $results = $response.Content | ConvertFrom-Json
            Write-Host "Search successful, returned results: $($results.Length)"
        } catch {
            Write-Host "Search failed: $($_.Exception.Message)"
        }
    }
}

# Evaluate cross-cultural labeling accuracy
function Evaluate-Xcultural {
    Write-Host "Evaluating cross-cultural labeling accuracy..."
    
    # Check if cross-cultural labeling result file exists
    if (-not (Test-Path "data/xcultural/goodm_xc.jsonl")) {
        Write-Host "Error: Cross-cultural labeling result file does not exist, please run cross-cultural labeling module first"
        return
    }
    
    # Count label distribution
    Write-Host "Counting label distribution..."
    $universalCount = 0
    $specificCount = 0
    $adaptableCount = 0
    $unknownCount = 0
    
    Get-Content "data/xcultural/goodm_xc.jsonl" | ForEach-Object {
        try {
            $record = $_.Trim() | ConvertFrom-Json
            switch ($record.culture_label) {
                "universal" { $universalCount++ }
                "culture-specific" { $specificCount++ }
                "adaptable" { $adaptableCount++ }
                default { $unknownCount++ }
            }
        } catch {
            $unknownCount++
        }
    }
    
    $totalRecords = $universalCount + $specificCount + $adaptableCount + $unknownCount
    
    Write-Host "Label distribution:"
    if ($totalRecords -gt 0) {
        $universalRate = [math]::Round(($universalCount * 100.0) / $totalRecords, 2)
        $specificRate = [math]::Round(($specificCount * 100.0) / $totalRecords, 2)
        $adaptableRate = [math]::Round(($adaptableCount * 100.0) / $totalRecords, 2)
        $unknownRate = [math]::Round(($unknownCount * 100.0) / $totalRecords, 2)
        
        Write-Host "universal: $universalCount ($universalRate%)"
        Write-Host "culture-specific: $specificCount ($specificRate%)"
        Write-Host "adaptable: $adaptableCount ($adaptableRate%)"
        Write-Host "unknown: $unknownCount ($unknownRate%)"
    } else {
        Write-Host "No valid records"
    }
}

# Evaluate all components
function Evaluate-All {
    Write-Host "Evaluating all components..."
    Evaluate-Generator
    Write-Host ""
    Evaluate-Evaluator
    Write-Host ""
    Evaluate-Search
    Write-Host ""
    Evaluate-Xcultural
}

# Parse command line arguments
if ($args.Length -eq 0) {
    Show-Help
    return
}

foreach ($arg in $args) {
    switch ($arg) {
        "-Generator" {
            Evaluate-Generator
        }
        "-Evaluator" {
            Evaluate-Evaluator
        }
        "-Search" {
            Evaluate-Search
        }
        "-Xcultural" {
            Evaluate-Xcultural
        }
        "-All" {
            Evaluate-All
        }
        "-Help" {
            Show-Help
        }
        default {
            Write-Host "Unknown option: $arg"
            Show-Help
            return
        }
    }
}