"""
Φ_EE Operator: Essence Explanation (E → EE)

Pipeline (paper, Appendix E, Operator 1):
    1. Chinese SRL (LTP)            → subject / action / object / scene
    2. Attribute retrieval          → HowNet sememes + Chinese Synonym Forest
    3. Structured assembly          → fluent EE (no new semantic content)
    4. Fluency refinement           → LLM at temperature=0.1, max_tokens=1024

The LLM never invents content: it receives the extracted slots and may only
rephrase them. Without an LLM, ``_template_assemble`` builds the same EE
deterministically from those slots.
"""

import os
from typing import Any, Dict, List, Optional

from core.knowledge_base import HowNetKB
from core.srl import SemanticRoleLabeler
from core.tokenizer import ChineseTokenizer
from embeddings.jina_embedder import JinaEmbedder
from llm_wrappers import BaseLLM, GenerationConfig

PROMPT_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "prompts", "entity_expansion.txt",
)

# Sememe → human-readable role description (used only for the subject slot).
ROLE_BY_SEMEME = {
    "商贩": "摊贩",
    "买方": "顾客",
    "管理者": "经理",
    "听众": "听众",
    "接收者": "旁观者",
    "政治": "政客",
    "竞选": "候选人",
    "学习": "学生",
    "宗教": "修行者",
    "残疾": "残障者",
    "说话": "说话者",
}


class EntityExpansion:
    """Essence Explanation Operator (Φ_EE: E → L)."""

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
        self.srl = SemanticRoleLabeler(self.kb)
        self.prompt_template = self._load_prompt_template()

    def _load_prompt_template(self) -> str:
        if os.path.exists(PROMPT_FILE):
            with open(PROMPT_FILE, "r", encoding="utf-8") as f:
                return f.read()
        raise FileNotFoundError(f"Prompt template not found: {PROMPT_FILE}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def expand(self, E: str) -> str:
        components = self.srl.analyze(E)
        components["attributes"] = self._retrieve_attributes(components["tokens"])
        if self.llm_client:
            refined = self._refine_fluency(self._assemble_prompt(components))
            if refined:
                return refined
        return self._template_assemble(components)

    # ------------------------------------------------------------------
    # Step 2 — HowNet + Synonym Forest attribute retrieval
    # ------------------------------------------------------------------
    def _retrieve_attributes(self, tokens: List[str]) -> List[str]:
        attributes: List[str] = []
        for sememe in self.kb.sememes_of(tokens):
            if sememe not in attributes:
                attributes.append(sememe)
        for synonym in self.kb.expand(tokens, top_k=3):
            label = f"近义于“{synonym}”"
            if label not in attributes:
                attributes.append(label)
        return attributes[:12]

    # ------------------------------------------------------------------
    # Step 3 — assembly
    # ------------------------------------------------------------------
    def _assemble_prompt(self, components: Dict[str, Any]) -> str:
        return self.prompt_template.format(
            subject=components["subject"] or "（未识别）",
            action=components["action"] or "（未识别）",
            object=components["object"] or "（未识别）",
            scene=components["scene"],
            attrs="、".join(components["attributes"]),
        )

    def _template_assemble(self, components: Dict[str, Any]) -> str:
        subject = components["subject"]
        action = components["action"]
        obj = components["object"]
        scene = components["scene"]

        # SRL could not recover an action-object frame: fall back to E itself
        # rather than fabricating a scene that E does not contain.
        if not action or not obj:
            return self._source_essence(components)

        description = self._describe_subject(subject)
        if not subject:
            clause = "某人"
        elif description:
            clause = f"{subject}是{description}"
        else:
            clause = subject
        location = scene if scene.endswith("上") else f"{scene}上"
        return f"{clause}，在{location}{action}{obj}。"

    def _source_essence(self, components: Dict[str, Any]) -> str:
        tokens = components.get("tokens", [])
        return "".join(tokens) if tokens else components.get("subject", "")

    def _describe_subject(self, subject: str) -> str:
        if not subject:
            return ""
        sememes = set(self.kb.sememes(subject))
        gender = "女" if "女性" in sememes else ("男" if "男性" in sememes else "")
        age = "年长的" if "年长" in sememes else ("年幼的" if "年幼" in sememes else "")
        role = next((ROLE_BY_SEMEME[s] for s in ROLE_BY_SEMEME if s in sememes), "")
        if role:
            return f"{age}{gender}{role}"
        if age or gender:
            return f"{age}{gender}人"
        return ""

    def _refine_fluency(self, prompt: str) -> str:
        try:
            response = self.llm_client.generate(
                prompt, config=GenerationConfig(temperature=0.1, max_tokens=1024)
            )
            return (response or "").strip()
        except Exception:
            return ""
