# GoodM - Metaphor Analysis System

GoodM (Good Metaphor) is a modular Python project for metaphor generation, evaluation, and analysis, supporting the implementation of the "Method + Experiments" section in the paper.

## Quick Start

### Installation

```bash
pip install -e .
```

### Basic Usage

```python
from goodm import GoodMTuple, MEF

# Create seven-tuple
tuple_obj = GoodMTuple(
    E="竹篮",
    V="一场空",
    EE="用竹子编织的篮子，质地疏松，无法盛水",
    VE="什么都没有，完全落空的状态",
    C="无法保持或留住",
    Sc="努力付出但没有任何收获的场景",
    Sy="徒劳无功，白费力气"
)

# Validate
print(tuple_obj.validate())  # True

# Evaluate
mef = MEF()
scores = mef.evaluate(tuple_obj)
print(f"MEF Score: {scores['MEF']}")
```

### Generate Seven-tuple

```python
import asyncio
from goodm import GenerationPipeline
from goodm.models.llm_wrappers import QwenWrapper

async def generate():
    # Initialize LLM
    llm = QwenWrapper(model_name="qwen-plus")
    
    # Create pipeline
    pipeline = GenerationPipeline(
        llm_client=llm,
        temperature=0.7,
        consensus_rounds=3
    )
    
    # Generate
    result = await pipeline.generate(E="竹篮", V="一场空")
    print(result.to_prefix())

asyncio.run(generate())
```

### Ablation Study

```python
from goodm import MEF, AblationStudy

# Prepare test data
test_tuples = [...]  # List[GoodMTuple]

# Run ablation experiments
mef = MEF()
ablation = AblationStudy(mef)

# Leave-one-out
results = ablation.leave_one_out(test_tuples)

# All experiments
all_results = ablation.run_all_experiments(
    test_tuples,
    output_path="results/ablation.json"
)
```

## Examples

Run examples from the `examples/` directory:

```bash
# Basic usage example
python examples/basic_usage.py

# Ablation study example
python examples/ablation_study.py
```

## License

MIT License