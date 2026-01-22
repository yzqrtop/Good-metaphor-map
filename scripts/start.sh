#!/bin/bash

# GoodM project startup script

# Show help information
show_help() {
    echo "GoodM project startup script"
    echo "Usage: ./start.sh [options]"
    echo ""
    echo "Options:"
    echo "  --crawler          Start data crawling"
    echo "  --generator        Start seven-tuple generation"
    echo "  --evaluator        Start evaluator"
    echo "  --merge            Start gold dataset construction"
    echo "  --train            Start model training"
    echo "  --search           Start search service"
    echo "  --xcultural        Start cross-cultural labeling"
    echo "  --all              Start complete workflow"
    echo "  --help             Show help information"
}

# Start data crawling
start_crawler() {
    echo "Starting data crawling..."
    python -m src.crawler.crawl_cxd --out data/raw/cxd_idioms.jsonl
}

# Start seven-tuple generation
start_generator() {
    echo "Starting seven-tuple generation..."
    python -m src.generate.batch_gen --in data/raw/cxd_idioms.jsonl \
                                    --out data/gen/goodm_raw.jsonl \
                                    --model qwen-plus --workers 32
}

# Start evaluator
start_evaluator() {
    echo "Starting evaluator..."
    python -m src.minieval.score --in data/gen/goodm_raw.jsonl \
                                --out data/scored/goodm_scored.jsonl
}

# Start gold dataset construction
start_merge() {
    echo "Starting gold dataset construction..."
    python -m src.merge.merge_best --in data/scored/goodm_scored.jsonl \
                                 --out data/gold/goodm_gold.jsonl
}

# Start model training
start_train() {
    echo "Starting model training..."
    python src/train/train.py --config-name intent_cls
}

# Start search service
start_search() {
    echo "Starting search service..."
    # First build index
    python -m src.search.index --in data/gold/goodm_gold.jsonl
    # Start API service
    uvicorn src.search.api:app --reload --host 0.0.0.0 --port 8000
}

# Start cross-cultural labeling
start_xcultural() {
    echo "Starting cross-cultural labeling..."
    python -m src.xcultural.label --in data/gold/goodm_gold.jsonl \
                                 --out data/xcultural/goodm_xc.jsonl
}

# Start complete workflow
start_all() {
    echo "Starting complete workflow..."
    start_crawler
    start_generator
    start_evaluator
    start_merge
    start_train
    start_search
    start_xcultural
}

# Parse command line arguments
if [ $# -eq 0 ]; then
    show_help
    exit 1
fi

for arg in "$@"; do
    case $arg in
        --crawler)
            start_crawler
            ;;
        --generator)
            start_generator
            ;;
        --evaluator)
            start_evaluator
            ;;
        --merge)
            start_merge
            ;;
        --train)
            start_train
            ;;
        --search)
            start_search
            ;;
        --xcultural)
            start_xcultural
            ;;
        --all)
            start_all
            ;;
        --help)
            show_help
            ;;
        *)
            echo "Unknown option: $arg"
            show_help
            exit 1
            ;;
    esac
done