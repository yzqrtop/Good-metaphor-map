"""
Φ_VE Operator: Vehicle Explanation (V → VE)

Pipeline (paper, Appendix E, Operator 2):
    1. Dictionary lookup    → standard interpretation of V (Xinhua glosses)
    2. HowNet sememe fetch  → explicit semantic features of V
    3. Rule-based assembly  → literal VE composed from definition + features
    4. Fluency refinement   → LLM at temperature=0.1, max_tokens=1024

VE must stay literal: it interprets V "literally for comparison" with EE and
must not import the figurative content of E.
"""

import os
from typing import Any, Dict, List, Optional

from core.knowledge_base import HowNetKB
from core.tokenizer import ChineseTokenizer
from embeddings.jina_embedder import JinaEmbedder
from llm_wrappers import BaseLLM, GenerationConfig

PROMPT_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "prompts", "vehicle_expansion.txt",
)

# Sememe → explicit behavioural consequence, used to make VE self-contained.
SEMEME_CONSEQUENCE = {
    "夸大": "夸大其词而不顾客观现实",
    "失真": "与客观情况不相符",
    "自夸": "反复肯定自己",
    "称赞": "对人或事物给予肯定评价",
    "推销": "向他人推介以促成接受",
    "不信任": "对他人心存疑虑",
    "警告": "提醒他人防备后患",
}


class VehicleExpansion:
    """Vehicle Explanation Operator (Φ_VE: V → L)."""

    def __init__(
        self,
        llm_client: Optional[BaseLLM] = None,
        embedder: Optional[JinaEmbedder] = None,
        kb: Optional[HowNetKB] = None,
    ):
        self.llm_client = llm_client
        self.embedder = embedder
        self.kb = kb or HowNetKB()
        self.tokenizer: ChineseTokenizer = self.kb.build_tokenizer()
        self.prompt_template = self._load_prompt_template()

    def _load_prompt_template(self) -> str:
        if os.path.exists(PROMPT_FILE):
            with open(PROMPT_FILE, "r", encoding="utf-8") as f:
                return f.read()
        raise FileNotFoundError(f"Prompt template not found: {PROMPT_FILE}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def expand(self, V: str) -> str:
        definition = self._lookup_dictionary_definition(V)
        features = self._retrieve_hownet_sememes(V)
        components = {
            "vehicle": V,
            "definition": definition,
            "features": "、".join(features),
        }
        if self.llm_client:
            prompt = self.prompt_template.format(
                definition=components["definition"],
                semantic_features=components["features"],
            )
            refined = self._refine_fluency(prompt)
            if refined:
                return refined
        return self._template_assemble(components)

    # ------------------------------------------------------------------
    # Step 1 — dictionary lookup
    # ------------------------------------------------------------------
    def _lookup_dictionary_definition(self, V: str) -> str:
        tokens = [t for t in self.tokenizer.tokenize(V) if t not in self.kb.stopwords]
        glosses: List[str] = []
        for token in tokens:
            gloss = self.kb.gloss(token)
            if gloss and gloss not in glosses:
                glosses.append(gloss)

        if not glosses:
            return f"字面意思为“{V}”"
        # In Chinese coordinate compounds the semantic head is the final item
        # ("自我推销和自夸" → 自夸), so its gloss defines V.
        return glosses[-1] if len(glosses) > 1 else glosses[0]

    # ------------------------------------------------------------------
    # Step 2 — HowNet semantic features
    # ------------------------------------------------------------------
    def _retrieve_hownet_sememes(self, V: str) -> List[str]:
        tokens = [t for t in self.tokenizer.tokenize(V) if t not in self.kb.stopwords]
        return self.kb.sememes_of(tokens)

    # ------------------------------------------------------------------
    # Step 3 — assembly
    # ------------------------------------------------------------------
    def _template_assemble(self, components: Dict[str, Any]) -> str:
        definition = components["definition"]
        clause = f"{components['vehicle']}，指{definition}"
        consequences = [
            text for sememe, text in SEMEME_CONSEQUENCE.items()
            if sememe in components["features"] and text not in clause
        ]
        if consequences:
            clause += "，" + consequences[0]
        return clause.rstrip("。") + "。"

    def _refine_fluency(self, prompt: str) -> str:
        try:
            response = self.llm_client.generate(
                prompt, config=GenerationConfig(temperature=0.1, max_tokens=1024)
            )
            return (response or "").strip()
        except Exception:
            return ""
