# GoodM - Metaphor Analysis System

GoodM (Good Metaphor) is a modular Python framework for metaphor generation, evaluation, and analysis. It implements the seven-tuple representation and five processing operators as described in the associated paper.

## Framework Overview

GoodM represents metaphors using a seven-tuple structure:

- **E** (Essence): Riddle-like expression
- **V** (Vehicle): Literal explanation
- **EE** (Essence Explanation): Detailed expansion of E
- **VE** (Vehicle Explanation): Detailed expansion of V
- **C** (Commonality): Cross-domain alignment between EE and VE
- **Sc** (Scene): Situational grounding
- **Sy** (Synthetics): Pragmatic consequence

## Five Processing Operators

Each operator is a constrained generation module:

| Operator | Function | Description |
|----------|----------|-------------|
| Φ_EE: E → L | Entity Expansion | Expands E via semantic role labeling + HowNet retrieval; Qwen2.5-1.5B for fluency only (temp=0.1) |
| Φ_VE: V → L | Vehicle Expansion | Expands V via dictionary definition + HowNet sememe retrieval; fluency refinement only |
| Φ_C: L×L → C | Commonality Extraction | TF-IDF keyword extraction + Jaccard intersection + HowNet expansion; embedding similarity verifies overlap |
| Φ_Sc: L×L → Sc | Scene Generalization | Template-based candidate generation (3-5 scenes) + LLM selection (Qwen2.5-7B, selection only) + embedding verification |
| Φ_Sy: Sc×C → Sy | Synthesis | Rule-based matching (200-300 rules) + embedding similarity scoring + LLM validation; fallback if confidence < 0.7 |

## Verifiable Constraints

Four constraints with calibrated thresholds (calibrated on 500 CTRC dev entries, validation F1 > 0.85):

| Constraint | Formula | Threshold |
|------------|---------|-----------|
| Faithfulness | Sim(EE,E) ≥ τ_faith ∧ Sim(VE,V) ≥ τ_faith | 0.78 |
| Commonality | Sim(C,EE) ≥ τ_common ∧ Sim(C,VE) ≥ τ_common | 0.76 |
| Synthesis | Sim(Sy,C) ≥ τ_synth ∧ Sim(Sy,Sc) ≥ τ_synth | 0.74 |
| Non-degeneracy | Sim(E,V) < τ_nondeg | 0.82 |

*Sim(·,·) = cosine similarity with Jina-embeddings-v3*

## Datasets

- **GoodM-Structured Dataset**: 18,103 TAS entries with full seven-tuple annotations. Constructed via generation-verification pipeline; human QA on 500 samples (κ=0.81).
- **CTRC** (existing benchmark): 75,296 paragraphs, 4 candidates per instance. Used **exclusively for evaluation** (no re-annotation).
- **Golden OOD Set**: 200 entries from Supplemental Volume; Sy labels by 3 experts (κ=0.79); multi-layer decontamination against C4 (<3% flagged).
- **Intent Subset**: 2,000 CTRC instances with educate/persuade/warn labels (κ=0.76); for RQ2.

## Research Questions & Evaluation Protocol

| RQ | Task | Dataset | Key Design | Metrics |
|----|------|---------|------------|---------|
| RQ1 | TAS Cloze | CTRC Test (7,796) | Generate 7-tuple → select answer | Acc, Macro-F1, BERTScore, CPR |
| RQ2 | Intent Prediction | CTRC-Intent (2,000) | Generate Sy → infer intent | Per-class Acc, Sy-Intent Consistency |
| RQ3 | Constraint Verification | Hard Subset (342) | Generate → verify → revise → answer | Acc, Error Interception Rate |
| RQ4 | Hard & Frequency | Hard Subset / CTRC | Stratified by confidence & E-V frequency | Hard Acc, Gap Reduction, Freq Gain |
| RQ5 | Ablation | CTRC Dev (7,500) | Selective component generation | ΔAcc Drop, CPR |
| RQ6 | Zero-shot OOD | OOD Set (200) | Generate 7-tuple → explain | BERTScore, Human Sy Agreement |

## Models & Implementation

**Supported Models**: GPT-4, GPT-5, Claude-3.5-Sonnet, DeepSeek-V3, Qwen-Plus, LLaMA-3.1-8B, Mistral-7B, Gemma-7B, Yi-34B-Chat, BERT-base, DeepMet-S, Jina-embeddings-v3.

**Inference**: 32-shot ICL for all LLMs (no fine-tuning). Base = context + candidates only. GoodM = each demonstration includes full seven-tuple generation. Semantic matching: Base uses E embedding; GoodM uses C embedding.

All results tested via paired bootstrap (1,000 iterations, α=0.05). OOD: 95% CI (B=10,000). Constraint Pass Rate (CPR) and Human Score (1-5) reported for GoodM conditions.

## Key Results Summary

- **RQ1**: GoodM improves all LLMs (+1.58–2.42% Acc) and semantic models (+2.50–2.67 BERTScore). BERT-base gains +3.70%.
- **RQ2**: Intent prediction +9.6–12.3%, largest on "persuade" (+13.5 pp for GPT-4). Sy-Intent Consistency 85.3–91.2%.
- **RQ3**: Verification improves hard subset by +5.3–6.3%; error interception 55.2–62.4%.
- **RQ4**: Hard subset gains +5.3–5.9% (vs. full-set +1.6–2.4%); gap reduced from -11.8 to -8.2 pp (GPT-4).
- **RQ5**: C and Sy most critical (drops -4.1 and -4.4 pp when ablated).
- **RQ6**: OOD BERTScore +2.7–3.3; Human Sy Agreement +10.5–12.0%.

## Quick Start

### Installation

```bash
pip install -e .
```

### Basic Usage

```python
from core.schema import GoodMTuple

# Create seven-tuple with real Chinese xielingyu data
tuple_obj = GoodMTuple(
    E="王婆卖瓜",
    V="自买自夸",
    EE="王婆在集市上卖瓜，比喻自我吹嘘",
    VE="自己买自己的瓜，自己夸奖自己的货物",
    C="对自己的东西进行过度夸奖，自我标榜",
    Sc="市场交易场景中卖方过度宣传商品",
    Sy="讽刺自吹自擂的行为，提醒要客观看待事物"
)

# Validate
print(tuple_obj.validate())

# Generate prefix for LLM
print(tuple_obj.to_prefix())
```

### Run Experiments

```bash
# Run specific RQ experiment
python run_experiments.py --rq 1 --model gpt-4

# Run all experiments
python run_experiments.py --all --model qwen-plus

# Run with specific dataset
python run_experiments.py --rq 5 --dataset data/ctrc.json --output results/rq5.json
```

### Using Operators

```python
from operators import (
    EntityExpansion,
    VehicleExpansion,
    CommonalityExtraction,
    SceneGeneralization,
    Synthesis
)
from embeddings import JinaEmbedder

# Initialize components
embedder = JinaEmbedder()
entity_expander = EntityExpansion(embedder=embedder)
vehicle_expander = VehicleExpansion(embedder=embedder)

# Expand entities (real xielingyu)
EE = entity_expander.expand("王婆卖瓜")
VE = vehicle_expander.expand("自买自夸")

# Extract commonality
extractor = CommonalityExtraction(embedder=embedder)
C = extractor.extract(EE, VE)

# Generate scene
scene_gen = SceneGeneralization(embedder=embedder)
Sc = scene_gen.generalize(EE, VE)

# Synthesize
synthesizer = Synthesis(embedder=embedder)
Sy = synthesizer.synthesize(Sc, C)
```

### Verify Constraints

```python
from constraints import ConstraintVerifier
from core.schema import GoodMTuple

# Create verifier
verifier = ConstraintVerifier()

# Verify tuple with real xielingyu data
tuple_obj = GoodMTuple(
    E="王婆卖瓜",
    V="自买自夸",
    EE="比喻自我吹嘘、自我标榜",
    VE="自己夸奖自己的货物",
    C="自我标榜、自我吹嘘",
    Sc="市场交易场景",
    Sy="讽刺自吹自擂的行为"
)

results = verifier.verify_all(tuple_obj)
print(f"All constraints passed: {results['overall']['passed']}")
print(f"CPR: {results['overall']['pass_rate']}")
```

### Load Datasets

```python
from datasets import GoodMDataset, CTRCDataset, OODDataset, IntentSubset

# Load GoodM-Structured Dataset
goodm_dataset = GoodMDataset(path="data/goodm_structured.json")
goodm_data = goodm_dataset.load()

# Load CTRC benchmark
ctrc_dataset = CTRCDataset(path="data/ctrc.json")
test_set = ctrc_dataset.get_test_set()

# Load OOD dataset
ood_dataset = OODDataset(path="data/ood_set.json")
ood_data = ood_dataset.to_tuples_with_labels()

# Load intent subset
intent_dataset = IntentSubset(path="data/intent_subset.json")
intent_data = intent_dataset.load()
```

## Project Structure

```
efs_program_master_yzq/
├── core/                          # Core data structures
│   └── schema.py                  # GoodMTuple seven-tuple definition
├── operators/                     # Five processing operators
│   ├── entity_expansion.py        # Φ_EE: E → L
│   ├── vehicle_expansion.py       # Φ_VE: V → L
│   ├── commonality_extraction.py  # Φ_C: L×L → C
│   ├── scene_generalization.py    # Φ_Sc: L×L → Sc
│   └── synthesis.py               # Φ_Sy: Sc×C → Sy
├── constraints/                   # Constraint verification
│   └── verifier.py                # 4 constraint verifier with thresholds
├── embeddings/                    # Embedding integration
│   └── jina_embedder.py           # Jina-embeddings-v3 integration
├── datasets/                      # Dataset loaders
│   ├── goodm_dataset.py           # GoodM-Structured Dataset (18,103 entries)
│   ├── ctrc_dataset.py            # CTRC Benchmark (75,296 paragraphs)
│   ├── ood_dataset.py             # Golden OOD Set (200 entries)
│   └── intent_subset.py           # Intent Subset (2,000 instances)
├── models/                        # Model wrappers
│   ├── llm_wrappers/              # LLM wrappers
│   │   ├── base.py                # BaseLLM class
│   │   ├── qwen.py                # Qwen wrapper
│   │   ├── gpt4.py                # GPT-4 wrapper
│   │   ├── claude.py              # Claude wrapper
│   │   ├── deepseek.py            # DeepSeek wrapper
│   │   └── opensource.py          # Open-source model wrapper
│   └── semantic_models.py         # BERT-base & DeepMet-S
├── experiments/                   # RQ experiment scripts
│   ├── rq1_tas_cloze.py           # RQ1: TAS Cloze Evaluation
│   ├── rq2_intent_prediction.py   # RQ2: Intent Prediction
│   ├── rq3_constraint_verification.py  # RQ3: Constraint Verification
│   ├── rq4_hard_frequency.py      # RQ4: Hard & Frequency Analysis
│   ├── rq5_ablation_study.py      # RQ5: Ablation Study
│   └── rq6_zeroshot_ood.py        # RQ6: Zero-shot OOD Evaluation
├── run_experiments.py             # Main experiment runner
└── README.md
```

## API Reference

### GoodMTuple

```python
from core.schema import GoodMTuple

# Create tuple
t = GoodMTuple(E="...", V="...", EE="...", VE="...", C="...", Sc="...", Sy="...")

# Methods
t.validate()                     # Check all fields non-empty
t.validate_with_details()        # Detailed validation report
t.to_prefix()                    # LLM input format
t.to_embedding_input()           # List of field strings
t.to_dict()                      # Convert to dictionary
t.to_json()                      # Convert to JSON string
```

### ConstraintVerifier

```python
from constraints import ConstraintVerifier

verifier = ConstraintVerifier(thresholds={
    'faithfulness': 0.78,
    'commonality': 0.76,
    'synthesis': 0.74,
    'non_degeneracy': 0.82
})

# Verify all constraints
results = verifier.verify_all(tuple_obj)

# Compute CPR
cpr = verifier.compute_cpr(tuple_list)
```

### Model Wrappers

```python
from models.llm_wrappers import create_model

# Create model by name
llm = create_model("gpt-4", api_key="your-api-key")

# Generate text
response = llm.generate("Your prompt here")
```

## License

MIT License