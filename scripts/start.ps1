#!/usr/bin/env pwsh

# GoodM project startup script (PowerShell version)

# Show help information
function Show-Help {
    Write-Host "GoodM project startup script"
    Write-Host "Usage: .\start.ps1 [options]"
    Write-Host ""
    Write-Host "Options:"
    Write-Host "  -Crawler          Start data crawling"
    Write-Host "  -Generator        Start seven-tuple generation"
    Write-Host "  -Evaluator        Start evaluator"
    Write-Host "  -Merge            Start gold dataset construction"
    Write-Host "  -Train            Start model training"
    Write-Host "  -Search           Start search service"
    Write-Host "  -Xcultural        Start cross-cultural labeling"
    Write-Host "  -All              Start complete workflow"
    Write-Host "  -Help             Show help information"
}

# Start data crawling
function Start-Crawler {
    Write-Host "Starting data crawling..."
    python -m src.crawler.crawl_cxd --out data/raw/cxd_idioms.jsonl
}

# Start seven-tuple generation
function Start-Generator {
    Write-Host "Starting seven-tuple generation..."
    python -m src.generate.batch_gen --in data/raw/cxd_idioms.jsonl `
                                    --out data/gen/goodm_raw.jsonl `
                                    --model qwen-plus --workers 32
}

# Start evaluator
function Start-Evaluator {
    Write-Host "Starting evaluator..."
    python -m src.minieval.score --in data/gen/goodm_raw.jsonl `
                                --out data/scored/goodm_scored.jsonl
}

# Start gold dataset construction
function Start-Merge {
    Write-Host "Starting gold dataset construction..."
    python -m src.merge.merge_best --in data/scored/goodm_scored.jsonl `
                                 --out data/gold/goodm_gold.jsonl
}

# Start model training
function Start-Train {
    Write-Host "Starting model training..."
    python src/train/train.py --config-name intent_cls
}

# Start search service
function Start-Search {
    Write-Host "Starting search service..."
    # First build index
    python -m src.search.index --in data/gold/goodm_gold.jsonl
    # Start API service
    uvicorn src.search.api:app --reload --host 0.0.0.0 --port 8000
}

# Start cross-cultural labeling
function Start-Xcultural {
    Write-Host "Starting cross-cultural labeling..."
    python -m src.xcultural.label --in data/gold/goodm_gold.jsonl `
                                 --out data/xcultural/goodm_xc.jsonl
}

# Start complete workflow
function Start-All {
    Write-Host "Starting complete workflow..."
    Start-Crawler
    Start-Generator
    Start-Evaluator
    Start-Merge
    Start-Train
    Start-Search
    Start-Xcultural
}

# Parse command line arguments
if ($args.Length -eq 0) {
    Show-Help
    return
}

foreach ($arg in $args) {
    switch ($arg) {
        "-Crawler" {
            Start-Crawler
        }
        "-Generator" {
            Start-Generator
        }
        "-Evaluator" {
            Start-Evaluator
        }
        "-Merge" {
            Start-Merge
        }
        "-Train" {
            Start-Train
        }
        "-Search" {
            Start-Search
        }
        "-Xcultural" {
            Start-Xcultural
        }
        "-All" {
            Start-All
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