#!/bin/bash

# GoodM Framework Evaluation Script
# Usage: ./evaluate.sh [options]

show_help() {
    echo "GoodM Framework Evaluation Script"
    echo "Usage: ./evaluate.sh [options]"
    echo ""
    echo "Options:"
    echo "  --rq [1-6]      Run specific RQ experiment"
    echo "  --all            Run all RQ experiments"
    echo "  --model [name]   Specify LLM model (e.g., gpt-4, qwen-plus)"
    echo "  --dataset [path] Path to dataset file"
    echo "  --output [path]  Path to save results"
    echo "  --help           Show help information"
}

# Parse command line arguments
if [ $# -eq 0 ]; then
    show_help
    exit 1
fi

# Build command
CMD="python run_experiments.py"

while [[ $# -gt 0 ]]; do
    case $1 in
        --rq)
            CMD="$CMD --rq $2"
            shift 2
            ;;
        --all)
            CMD="$CMD --all"
            shift
            ;;
        --model)
            CMD="$CMD --model $2"
            shift 2
            ;;
        --dataset)
            CMD="$CMD --dataset $2"
            shift 2
            ;;
        --output)
            CMD="$CMD --output $2"
            shift 2
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

echo "Running: $CMD"
eval $CMD