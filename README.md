# 🀄 GoodM

A seven-tuple construction framework for Chinese two-part allegorical sayings.

![Python 3.9+](assets/badges/python.svg) ![SRL: LTP](assets/badges/ltp.svg) ![Lexicon: HowNet + Cilin](assets/badges/lexicon.svg) ![Embedding: Jina-v3](assets/badges/embedding.svg) ![Metrics: BERTScore | DeepMet-S](assets/badges/metrics.svg) ![License: MIT](assets/badges/license.svg)

`⟨E, V, EE, VE, C, Sc, Sy⟩`

---

## 📖 Overview

GoodM decomposes an allegorical saying into a **source domain E** and a **target domain V**, then generates the full seven-tuple with five operators:

| Operator | Pipeline | Output |
|:---|:---|:---|
| Φ_EE | LTP semantic roles + HowNet sememes + Cilin synonyms | Source explanation EE |
| Φ_VE | Dictionary gloss + sememe features | Target explanation VE |
| Φ_C | TF-IDF + Jaccard + sememe abstraction | Commonality C |
| Φ_Sc | Template candidates + model selection + τ check | Scene Sc |
| Φ_Sy | 216-rule retrieval + confidence fallback | Consequence Sy |

Every output must pass four constraints: faithfulness 0.78, commonality 0.76, synthesis 0.74, non-degeneracy 0.82. The LLM only assembles and selects — it never generates freely.

## ⚙️ Installation

```bash
git clone <repo-url> && cd efs_program_master_yzq
pip install -e .                      # numpy / requests / ltp / OpenHowNet

export JINA_API_KEY=...               # required: embedding backend
export DASHSCOPE_API_KEY=...          # optional: Qwen models
# optional: export HOWNET_JSON_URL=https://.../hownet.json

pip install -e ".[metrics]"           # optional: BERTScore / DeepMet-S
```

Cilin and HowNet are downloaded at run time and cached under `data/cache/`; no lexicon is bundled with the repository.

## 🚀 Usage

```bash
# single run
python run_experiments.py --rq 1 --ctrc data/ctrc.json

# all experiments (RQ1–RQ6)
python run_experiments.py --all \
    --ctrc data/ctrc.json --intent data/intent.json --ood data/ood.json \
    --model qwen-plus --output results/all.json

# RQ6 with BERTScore and DeepMet-S
python run_experiments.py --rq 6 --ood data/ood.json \
    --bertscore bert-base-chinese --deepmets /path/to/deepmets
```

As a library:

```python
from core.pipeline import GoodMPipeline

pipeline = GoodMPipeline()
print(pipeline.build("王婆卖瓜", "自我推销和自夸").to_json())
```

## 📁 Layout

```text
core/           schema · tokenizer · online lexicons · SRL · pipeline · metrics
operators/      Φ_EE · Φ_VE · Φ_C · Φ_Sc · Φ_Sy
constraints/    four-constraint verifier
embeddings/     Jina-v3 · DeepMet-S · BERTScore
datasets/       CTRC · Intent · OOD loaders
experiments/    RQ1–RQ6 protocols
llm_wrappers/   Qwen2.5 / GPT-4 wrappers
data/prompts/   Appendix-F prompt templates
data/rule_base/ rule seeds (expanded to 216 triples at load time)
assets/badges/  local badge SVGs used by this README
```

---

*MIT License · Dataset and lexicon rights belong to their original publishers.*
