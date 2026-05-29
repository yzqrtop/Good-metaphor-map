"""
GoodM - Metaphor Generation, Evaluation, and Analysis System

GoodM is a modular Python project for metaphor generation, evaluation, and analysis,
supporting the implementation of the "Method + Experiments" section in the paper.

Core Features:
- Seven-tuple generation (E, V, EE, VE, C, Sc, Sy)
- MiniEvalFeature evaluation
- Downstream tasks (intent prediction, etc.)
- Ablation experiments

Project Structure:
    ./
    ├── core/               # Core data structures
    │   └── schema.py       # GoodMTuple seven-tuple definition
    ├── src/
    │   ├── generate/       # Data generation pipeline
    │   │   └── pipeline.py # Four-stage generation + multi-round consensus
    │   ├── models/         # Model wrappers
    │   │   └── llm_wrappers/  # LLM wrappers (Qwen, GPT-4)
    │   ├── minieval/       # MiniEvalFeature evaluation
    │   │   └── mef.py      # 7-metric weighted aggregation
    │   └── tasks/          # Downstream tasks
    │       └── intent.py   # Intent prediction
    ├── experiments/        # Experiment scripts
    │   └── ablation.py     # Ablation experiments
    └── data/prompts/       # Generation prompt templates

Usage Examples:
    >>> from core.schema import GoodMTuple
    >>> from src.generation.pipeline import GenerationPipeline
    >>> from src.minieval.mef import MEF

    >>> # Create seven-tuple
    >>> tuple_obj = GoodMTuple(E="竹篮", V="一场空")

    >>> # Use pipeline to generate complete seven-tuple
    >>> pipeline = GenerationPipeline(llm_client)
    >>> result = await pipeline.generate("竹篮", "一场空")

    >>> # Evaluate
    >>> mef = MEF()
    >>> scores = mef.evaluate(result)
"""

__version__ = "0.1.0"

__all__ = [
    'GoodMTuple',
    'GenerationPipeline',
    'MEF',
    'IntentPredictor',
    'AblationStudy'
]