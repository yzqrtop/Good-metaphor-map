# GoodM Framework Start Script
# Usage: ./start.sh [options]

show_help() {
    echo "GoodM Framework Start Script"
    echo "Usage: ./start.sh [options]"
    echo ""
    echo "Options:"
    echo "  --install        Install dependencies"
    echo "  --test           Run tests"
    echo "  --example        Run basic usage example"
    echo "  --rq [1-6]       Run specific RQ experiment"
    echo "  --help           Show help information"
}

# Parse command line arguments
if [ $# -eq 0 ]; then
    show_help
    exit 1
fi

case $1 in
    --install)
        echo "Installing GoodM framework..."
        pip install -e .
        pip install -e ".[dev]"
        echo "Installation complete!"
        ;;
    --test)
        echo "Running tests..."
        python -m pytest tests/ -v
        ;;
    --example)
        echo "Running basic usage example..."
        python examples/basic_usage.py
        ;;
    --rq)
        if [ -z "$2" ]; then
            echo "Error: Please specify RQ number (1-6)"
            exit 1
        fi
        echo "Running RQ$2 experiment..."
        python run_experiments.py --rq $2
        ;;
    --help)
        show_help
        ;;
    *)
        echo "Unknown option: $1"
        show_help
        exit 1
        ;;
esac