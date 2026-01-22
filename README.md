# GoodM Metaphor Analysis System

## Project Introduction

GoodM is a modular Python project for metaphor generation, evaluation, and analysis, designed to support the implementation of the "Method + Experiment" section in papers. The project is divided into multiple functionally distinct modules, covering the complete workflow from data crawling to cross-cultural labeling.

## Project Structure

```
GoodM_master/
├─ data/                 # Raw corpus & intermediate results
│  ├─ raw/               # Raw crawled data
│  ├─ gen/               # Generated seven-tuple data
│  ├─ scored/            # Scoring results
│  ├─ gold/              # Gold dataset
│  └─ xcultural/         # Cross-cultural labeling data
├─ src/
│  ├─ crawler/           # Data crawling module
│  ├─ generate/          # Seven-tuple generation module
│  ├─ minieval/          # Evaluator module
│  ├─ merge/             # Gold dataset construction module
│  ├─ train/             # Model training module
│  ├─ search/            # Vector search module
│  └─ xcultural/         # Cross-cultural labeling module
├─ tests/                # Unit tests
├─ configs/              # Hydra configurations
├─ scripts/              # Startup and evaluation scripts
│  ├─ start.sh           # Startup script (bash)
│  ├─ evaluate.sh        # Evaluation script (bash)
│  ├─ start.ps1          # Startup script (PowerShell)
│  └─ evaluate.ps1       # Evaluation script (PowerShell)
├─ pyproject.toml        # Project dependencies and configuration
├─ Makefile              # Project commands and workflow
├─ docker-compose.yml    # Docker deployment configuration
└─ Dockerfile.api        # FastAPI service Dockerfile
```

## Module Description

### 1. Data Acquisition (`crawler/`)
- **Function**: Crawl CXD idiom data
- **Input**: None
- **Output**: `data/raw/cxd_idioms.jsonl`
- **Usage**:
  ```bash
  python -m src.crawler.crawl_cxd --out data/raw/cxd_idioms.jsonl
  ```

### 2. Automatic Seven-tuple Generation (`generate/`)
- **Function**: Generate seven-tuples (E, V, EE, VE, C, Sc, Sy) using LLM
- **Input**: `data/raw/cxd_idioms.jsonl`
- **Output**: `data/gen/goodm_raw.jsonl`
- **Usage**:
  ```bash
  python -m src.generate.batch_gen --in data/raw/cxd_idioms.jsonl \
                                    --out data/gen/goodm_raw.jsonl \
                                    --model qwen-plus --workers 32
  ```

### 3. MiniEvalFeature Scoring (`minieval/`)
- **Function**: Score generated seven-tuples
- **Input**: `data/gen/goodm_raw.jsonl`
- **Output**: `data/scored/goodm_scored.jsonl`
- **Usage**:
  ```bash
  python -m src.minieval.score --in data/gen/goodm_raw.jsonl \
                                --out data/scored/goodm_scored.jsonl
  ```

### 4. Gold Dataset Construction (`merge/`)
- **Function**: Select highest-scoring records for each (E, V) pair from scoring results to construct gold dataset
- **Input**: `data/scored/goodm_scored.jsonl`
- **Output**: `data/gold/goodm_gold.jsonl`
- **Usage**:
  ```bash
  python -m src.merge.merge_best --in data/scored/goodm_scored.jsonl \
                                 --out data/gold/goodm_gold.jsonl
  ```

### 5. Fine-tuning & Intent Classification (`train/`)
- **Function**: Fine-tune models and perform intent classification training
- **Input**: `data/gold/goodm_gold.jsonl` + 2000 manually annotated intent labels
- **Output**: `ckpt/intent_cls/{model_name}/best.ckpt`
- **Usage**:
  ```bash
  python src/train/train.py --config-name intent_cls
  ```

### 6. Vector Search Service (`search/`)
- **Function**: Build vector index and provide FastAPI search service
- **Input**: `data/gold/goodm_gold.jsonl`
- **Output**: Milvus index + FastAPI service
- **Usage**:
  ```bash
  # Build index
  python -m src.search.index --in data/gold/goodm_gold.jsonl
  
  # Start API service
  make api
  
  # Or deploy with Docker
  docker-compose up -d
  ```

### 7. Cross-cultural Labeling (`xcultural/`)
- **Function**: Add cross-cultural labels to metaphors
- **Input**: `data/gold/goodm_gold.jsonl`
- **Output**: `data/xcultural/goodm_xc.jsonl`
- **Usage**:
  ```bash
  python -m src.xcultural.label --in data/gold/goodm_gold.jsonl \
                                 --out data/xcultural/goodm_xc.jsonl
  ```

## Startup and Evaluation Scripts

### Startup Scripts

The project provides convenient startup scripts that support starting individual modules or the complete workflow:

#### Bash Version (`scripts/start.sh`)

```bash
# Start individual modules
./scripts/start.sh --crawler    # Start data crawling
./scripts/start.sh --generator  # Start seven-tuple generation
./scripts/start.sh --evaluator  # Start evaluator
./scripts/start.sh --merge      # Start gold dataset construction
./scripts/start.sh --train      # Start model training
./scripts/start.sh --search     # Start search service
./scripts/start.sh --xcultural  # Start cross-cultural labeling

# Start complete workflow
./scripts/start.sh --all
```

#### PowerShell Version (`scripts/start.ps1`)

```powershell
# Start individual modules
./scripts/start.ps1 -Crawler    # Start data crawling
./scripts/start.ps1 -Generator  # Start seven-tuple generation
./scripts/start.ps1 -Evaluator  # Start evaluator
./scripts/start.ps1 -Merge      # Start gold dataset construction
./scripts/start.ps1 -Train      # Start model training
./scripts/start.ps1 -Search     # Start search service
./scripts/start.ps1 -Xcultural  # Start cross-cultural labeling

# Start complete workflow
./scripts/start.ps1 -All
```

### Evaluation Scripts

The project provides evaluation scripts for assessing the performance and result quality of each module:

#### Bash Version (`scripts/evaluate.sh`)

```bash
# Evaluate individual modules
./scripts/evaluate.sh --generator  # Evaluate seven-tuple generation quality
./scripts/evaluate.sh --evaluator  # Evaluate evaluator performance
./scripts/evaluate.sh --search     # Evaluate search service performance
./scripts/evaluate.sh --xcultural  # Evaluate cross-cultural labeling accuracy

# Evaluate all modules
./scripts/evaluate.sh --all
```

#### PowerShell Version (`scripts/evaluate.ps1`)

```powershell
# Evaluate individual modules
./scripts/evaluate.ps1 -Generator  # Evaluate seven-tuple generation quality
./scripts/evaluate.ps1 -Evaluator  # Evaluate evaluator performance
./scripts/evaluate.ps1 -Search     # Evaluate search service performance
./scripts/evaluate.ps1 -Xcultural  # Evaluate cross-cultural labeling accuracy

# Evaluate all modules
./scripts/evaluate.ps1 -All
```

## Environment Configuration

### Install Dependencies

```bash
make install
```

Or install manually:

```bash
pip install -e .
pip install pre-commit
pre-commit install
```

### Docker Deployment

```bash
# Start Milvus and API services
docker-compose up -d

# Stop services
docker-compose down
```

## End-to-end One-click Script

```bash
# Install dependencies
make install

# Generate data (stages 0-4)
make data

# Train intent classification model
make train

# Build vector index
make index

# Start search service
make api

# Add cross-cultural labels
make xcultural

# Run tests
make test

# Run code quality checks
make lint
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific module tests
pytest tests/test_crawler.py -v
pytest tests/test_generator.py -v
pytest tests/test_evaluator.py -v
pytest tests/test_labeler.py -v
```

## Code Quality

```bash
# Run pre-commit hooks
pre-commit run -a

# Run individual check tools
black src/ tests/
isort src/ tests/
flake8 src/ tests/
mypy src/
```

## Notes

1. **Dependencies**: The project depends on multiple external libraries, including transformers, sentence-transformers, bert-score, etc. These libraries may require significant storage space and computational resources.

2. **LLM API**: The generation module requires access to LLM APIs (such as OpenAI), which requires configuring appropriate API keys.

3. **Milvus**: The vector search service depends on Milvus database, so ensure the Milvus service is running properly.

4. **Computational Resources**: Fine-tuning models requires significant computational resources, and it is recommended to run in a GPU environment.

5. **Data Format**: The project assumes input files are in JSONL format, with each record containing necessary fields. Actual usage may require adjustments based on input data format.

## License

[MIT License](LICENSE)