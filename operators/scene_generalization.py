"""
Φ_Sc Operator: Scene Generalization (EE × VE → Sc)

Pipeline (paper, Appendix E, Operator 4):
    1. Candidate generation — template instantiations of
       "A [subject] in [situation] [acts], leading to [consequence]."
       filled from the EE/VE slots (no LLM generation)
    2. Selection — Qwen2.5-7B selects one candidate, selection only
       (temperature=0.7, max_tokens=1024)
    3. Verification — the selected scene is compared with EE and VE against
       τ_common; if it fails, the next-best candidate is selected

Scenes stay *concrete but domain-general*: the specific market stall is lifted
into a class of situations while keeping the audience reaction explicit, which
is what Sy later consumes.
"""

import hashlib
import os
import re
from typing import Any, Dict, List, Optional

from constraints.verifier import ConstraintVerifier
from core.knowledge_base import HowNetKB
from core.tokenizer import ChineseTokenizer
from embeddings.jina_embedder import JinaEmbedder
from llm_wrappers import BaseLLM, GenerationConfig

PROMPT_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "prompts", "scene_generalization.txt",
)

# Audience reaction per domain: commercial scenes address 顾客, public speech 听众.
AUDIENCE_BY_DOMAIN = {
    "commercial": ("顾客", ["怀疑", "不信任", "犹豫", "观望"]),
    "public": ("听众", ["不信任", "质疑", "反感", "保留"]),
    "workplace": ("同事", ["质疑", "保留", "观望", "不满"]),
}

TEMPLATES = [
    "{role}在{place}不断{action}自己的{item}，结果引起{reaction}。",
    "一位{role}在{place}上{action}自家{item}，最终导致{reaction}。",
    "某位{role}在{place}中{action}自己的{item}，使得{reaction}。",
    "{role}在{place}向{audience}{action}自己的{item}，随后出现{reaction}。",
    "一名{role}在{place}把{item}{action}得过头，换来{reaction}。",
]


class SceneGeneralization:
    """Scene Generalization Operator (Φ_Sc: L × L → Sc)."""

    def __init__(
        self,
        llm_client: Optional[BaseLLM] = None,
        embedder: Optional[JinaEmbedder] = None,
        kb: Optional[HowNetKB] = None,
        num_candidates: int = 3,
    ):
        self.llm_client = llm_client
        self.embedder = embedder
        self.kb = kb or HowNetKB()
        self.tokenizer: ChineseTokenizer = self.kb.build_tokenizer()
        self.num_candidates = max(3, min(num_candidates, 5))
        self.prompt_template = self._load_prompt_template()
        self.last_selection: Dict[str, Any] = {}

    def _load_prompt_template(self) -> str:
        if os.path.exists(PROMPT_FILE):
            with open(PROMPT_FILE, "r", encoding="utf-8") as f:
                return f.read()
        raise FileNotFoundError(f"Prompt template not found: {PROMPT_FILE}")

    # ------------------------------------------------------------------
    # Step 1 — slot filling and candidate generation
    # ------------------------------------------------------------------
    def _slot_values(self, EE: str, VE: str) -> Dict[str, str]:
        tokens = [
            token for token in self.tokenizer.tokenize(f"{EE} {VE}")
            if token not in self.kb.stopwords
        ]
        tokens = list(dict.fromkeys(tokens))
        sememes = set(self.kb.sememes_of(tokens))

        commercial = bool({"交易", "市场", "商品", "商贩"} & sememes)
        public = bool({"政治", "竞选", "演讲", "听众"} & sememes)

        if public:
            role, place, domain = "政客", "辩论台", "public"
        elif commercial:
            role, place, domain = "销售人员", "集市", "commercial"
        else:
            role, place, domain = "说话者", "公开场合", "workplace"

        if "自夸" in sememes or "夸耀" in sememes:
            action = "吹嘘"
        elif "推销" in sememes:
            action = "推销"
        else:
            action = "宣传"

        item = "产品" if commercial else "自身能力"
        audience, reactions = AUDIENCE_BY_DOMAIN[domain]
        reaction = f"{audience}的{reactions[0]}"

        return {
            "role": role, "place": place, "action": action, "item": item,
            "audience": audience, "reaction": reaction, "domain": domain,
        }

    def _generate_candidates(self, EE: str, VE: str) -> List[str]:
        slots = self._slot_values(EE, VE)
        seed = int(hashlib.md5(f"{EE}|{VE}".encode("utf-8")).hexdigest()[:8], 16)
        candidates: List[str] = []
        for offset in range(self.num_candidates):
            template = TEMPLATES[(seed + offset) % len(TEMPLATES)]
            candidate = template.format(**slots)
            if candidate not in candidates:
                candidates.append(candidate)
        return candidates

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generalize(self, EE: str, VE: str, C: Optional[str] = None) -> str:
        candidates = self._generate_candidates(EE, VE)
        ranked = self._rank_candidates(EE, VE, C or "", candidates)

        preferred: List[str] = []
        if self.llm_client:
            chosen = self._select_with_llm(EE, VE, C or "", candidates)
            if chosen:
                preferred.append(chosen)

        # LLM choice first, then embedding order; skip candidates that fail
        # the τ_common check against EE and VE (Appendix E, step 3).
        order = preferred + [c for c in ranked if c not in preferred]
        threshold = self._tau_common()
        for candidate in order:
            check = self._verify(candidate, EE, VE, threshold)
            if check["passed"]:
                self.last_selection = {"scene": candidate, "verified": True, **check}
                return candidate

        selected = order[0] if order else ""
        self.last_selection = {
            "scene": selected,
            "verified": False,
            "threshold": threshold,
            "reason": "no candidate passed tau_common; returned top-ranked",
        }
        return selected

    def _tau_common(self) -> Optional[float]:
        if not self.embedder:
            return None
        return ConstraintVerifier(self.embedder).thresholds["commonality"]

    def _verify(
        self, candidate: str, EE: str, VE: str, threshold: Optional[float]
    ) -> Dict[str, Any]:
        if not self.embedder or threshold is None:
            return {"passed": True, "sim_Sc_EE": None, "sim_Sc_VE": None}
        sim_ee = self.embedder.cosine_similarity(candidate, EE)
        sim_ve = self.embedder.cosine_similarity(candidate, VE)
        return {
            "passed": sim_ee >= threshold and sim_ve >= threshold,
            "sim_Sc_EE": sim_ee,
            "sim_Sc_VE": sim_ve,
            "threshold": threshold,
        }

    def _rank_candidates(
        self, EE: str, VE: str, C: str, candidates: List[str]
    ) -> List[str]:
        if not self.embedder or not candidates:
            return list(candidates)
        contexts = [text for text in (C, EE, VE) if text]
        scored = []
        for candidate in candidates:
            score = sum(self.embedder.cosine_similarity(candidate, ctx) for ctx in contexts)
            scored.append((score / max(len(contexts), 1), candidate))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [candidate for _, candidate in scored]

    # ------------------------------------------------------------------
    # Step 2 — LLM selection only
    # ------------------------------------------------------------------
    def _select_with_llm(
        self, EE: str, VE: str, C: str, candidates: List[str]
    ) -> str:
        padded = candidates + [""] * (3 - len(candidates))
        prompt = self.prompt_template.format(
            EE=EE,
            VE=VE,
            c1=padded[0],
            c2=padded[1],
            c3=padded[2],
        )
        try:
            response = (self.llm_client.generate(
                prompt, config=GenerationConfig(temperature=0.7, max_tokens=1024)
            ) or "").strip()
        except Exception:
            return ""

        for candidate in candidates:
            if candidate and candidate in response:
                return candidate
        digit_match = re.search(r"(?:^|[\s：:])([1-9])(?:$|[\s。.,，])", response)
        if digit_match and 1 <= int(digit_match.group(1)) <= len(candidates):
            return candidates[int(digit_match.group(1)) - 1]
        letter_match = re.search(r"(?:^|[\s：:])([ABC])(?:$|[\s。.,，])", response)
        if letter_match:
            index = "ABC".index(letter_match.group(1))
            if index < len(candidates):
                return candidates[index]
        return ""
