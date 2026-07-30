# Makefile for GoodM Framework

# Install dependencies
install:
	pip install -e .
	pip install -e ".[dev]"

# Run tests
test:
	python -m pytest tests/ -v

# Run basic example
example:
	python examples/basic_usage.py

# Run specific RQ experiment
rq1:
	python run_experiments.py --rq 1

rq2:
	python run_experiments.py --rq 2

rq3:
	python run_experiments.py --rq 3

rq4:
	python run_experiments.py --rq 4

rq5:
	python run_experiments.py --rq 5

rq6:
	python run_experiments.py --rq 6

# Run all experiments
all:
	python run_experiments.py --all

# Run experiments with specific model
rq1-gpt4:
	python run_experiments.py --rq 1 --model gpt-4

rq5-qwen:
	python run_experiments.py --rq 5 --model qwen-plus

# Clean generated files
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete