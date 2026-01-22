# Makefile for GoodM metaphor project

# Install dependencies
install:
	pip install -e .
	pip install pre-commit
	pre-commit install

# Generate data pipeline (0-4 stages)
data:
	python -m src.crawler.crawl_cxd --out data/raw/cxd_idioms.jsonl
	python -m src.generate.batch_gen --in data/raw/cxd_idioms.jsonl \
	                                    --out data/gen/goodm_raw.jsonl \
	                                    --model qwen-plus --workers 32
	python -m src.minieval.score --in data/gen/goodm_raw.jsonl \
	                                --out data/scored/goodm_scored.jsonl
	python -m src.merge.merge_best --in data/scored/goodm_scored.jsonl \
	                                 --out data/gold/goodm_gold.jsonl

# Train intent classifier
train:
	python src/train/train.py --config-name intent_cls

# Build search index
index:
	python -m src.search.index --in data/gold/goodm_gold.jsonl

# Run API service
api:
	uvicorn src.search.api:app --reload --host 0.0.0.0 --port 8000

# Add cross-cultural labels
xcultural:
	python -m src.xcultural.label --in data/gold/goodm_gold.jsonl \
	                                 --out data/xcultural/goodm_xc.jsonl

# Run tests
test:
	pytest tests/ -v

# Run pre-commit hooks
lint:
	pre-commit run -a

# Clean generated files
clean:
	rm -rf data/gen/* data/scored/* data/gold/* data/xcultural/*
	rm -rf .pytest_cache/ __pycache__/

# Docker commands
docker-up:
	docker-compose up -d milvus
	docker-compose up -d api

docker-down:
	docker-compose down