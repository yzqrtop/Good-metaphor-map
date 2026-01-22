#!/bin/bash

# GoodM project evaluation script

# Show help information
show_help() {
    echo "GoodM project evaluation script"
    echo "Usage: ./evaluate.sh [options]"
    echo ""
    echo "Options:"
    echo "  --generator        Evaluate seven-tuple generation quality"
    echo "  --evaluator        Evaluate evaluator performance"
    echo "  --search           Evaluate search service performance"
    echo "  --xcultural        Evaluate cross-cultural labeling accuracy"
    echo "  --all              Evaluate all components"
    echo "  --help             Show help information"
}

# Evaluate seven-tuple generation quality
evaluate_generator() {
    echo "Evaluating seven-tuple generation quality..."
    
    # Check if generation result file exists
    if [ ! -f "data/gen/goodm_raw.jsonl" ]; then
        echo "Error: Generation result file does not exist, please run generator first"
        return 1
    fi
    
    # Count generation results
    total_records=$(wc -l < data/gen/goodm_raw.jsonl)
    echo "Total generated records: $total_records"
    
    # Check generation result format
    echo "Checking generation result format..."
    valid_records=0
    invalid_records=0
    
    while IFS= read -r line; do
        if echo "$line" | jq -e '.E and .V and .EE and .VE and .C and .Sc and .Sy' > /dev/null 2>&1; then
            valid_records=$((valid_records + 1))
        else
            invalid_records=$((invalid_records + 1))
        fi
    done < data/gen/goodm_raw.jsonl
    
    echo "Valid records: $valid_records"
    echo "Invalid records: $invalid_records"
    echo "Validity rate: $((valid_records * 100 / total_records))%"
}

# Evaluate evaluator performance
evaluate_evaluator() {
    echo "Evaluating evaluator performance..."
    
    # Check if evaluation result file exists
    if [ ! -f "data/scored/goodm_scored.jsonl" ]; then
        echo "Error: Evaluation result file does not exist, please run evaluator first"
        return 1
    fi
    
    # Count evaluation results
    total_records=$(wc -l < data/scored/goodm_scored.jsonl)
    echo "Total evaluated records: $total_records"
    
    # Calculate average MEF score
    echo "Calculating average MEF score..."
    avg_mef=$(jq -r '[.MEF] | add / length' data/scored/goodm_scored.jsonl)
    echo "Average MEF score: $avg_mef"
    
    # Calculate average scores for each dimension
    echo "Calculating average scores for each dimension..."
    for dim in f1 f2 f3 f4 f5 f6 f7; do
        avg_dim=$(jq -r "[.$dim] | add / length" data/scored/goodm_scored.jsonl)
        echo "Average $dim score: $avg_dim"
    done
}

# Evaluate search service performance
evaluate_search() {
    echo "Evaluating search service performance..."
    
    # Check if search service is running
    if ! curl -s http://localhost:8000/health > /dev/null; then
        echo "Warning: Search service is not running, please start search service first"
        return 1
    fi
    
    # Test search functionality
    echo "Testing search functionality..."
    test_queries=("井底之蛙" "守株待兔" "画蛇添足" "掩耳盗铃" "亡羊补牢")
    
    for query in "${test_queries[@]}"; do
        echo "Test query: $query"
        response=$(curl -s -X POST http://localhost:8000/search \
                      -H "Content-Type: application/json" \
                      -d '{"query": "'"$query"'", "k": 3}')
        
        if echo "$response" | jq -e '.' > /dev/null 2>&1; then
            echo "Search successful, returned results: $(echo "$response" | jq length)"
        else
            echo "Search failed: $response"
        fi
    done
}

# Evaluate cross-cultural labeling accuracy
evaluate_xcultural() {
    echo "Evaluating cross-cultural labeling accuracy..."
    
    # Check if cross-cultural labeling result file exists
    if [ ! -f "data/xcultural/goodm_xc.jsonl" ]; then
        echo "Error: Cross-cultural labeling result file does not exist, please run cross-cultural labeling module first"
        return 1
    fi
    
    # Count label distribution
    echo "Counting label distribution..."
    universal_count=$(grep -c '"culture_label":"universal"' data/xcultural/goodm_xc.jsonl)
    specific_count=$(grep -c '"culture_label":"culture-specific"' data/xcultural/goodm_xc.jsonl)
    adaptable_count=$(grep -c '"culture_label":"adaptable"' data/xcultural/goodm_xc.jsonl)
    unknown_count=$(grep -c '"culture_label":"unknown"' data/xcultural/goodm_xc.jsonl)
    
    total_records=$((universal_count + specific_count + adaptable_count + unknown_count))
    
    echo "Label distribution:"
    echo "universal: $universal_count ($((universal_count * 100 / total_records))%)"
    echo "culture-specific: $specific_count ($((specific_count * 100 / total_records))%)"
    echo "adaptable: $adaptable_count ($((adaptable_count * 100 / total_records))%)"
    echo "unknown: $unknown_count ($((unknown_count * 100 / total_records))%)"
}

# Evaluate all components
evaluate_all() {
    echo "Evaluating all components..."
    evaluate_generator
    echo ""
    evaluate_evaluator
    echo ""
    evaluate_search
    echo ""
    evaluate_xcultural
}

# Parse command line arguments
if [ $# -eq 0 ]; then
    show_help
    exit 1
fi

for arg in "$@"; do
    case $arg in
        --generator)
            evaluate_generator
            ;;
        --evaluator)
            evaluate_evaluator
            ;;
        --search)
            evaluate_search
            ;;
        --xcultural)
            evaluate_xcultural
            ;;
        --all)
            evaluate_all
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