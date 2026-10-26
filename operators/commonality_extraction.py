"""
Φ_C Operator: Commonality Extraction (EE × VE → C)

Pipeline (paper, Appendix E, Operator 3):
    1. Tokenize EE and VE
    2. TF-IDF keyword extraction (IDF estimated from the HowNet glosses and the
       synthesis rule corpus)
    3. Jaccard intersection of the two keyword sets
    4. HowNet sememe expansion, mapped onto the abstract attribute vocabulary
    5. Semantic-overlap verification with Jina-embeddings-v3

Φ_C is purely algorithmic: the paper defines no LLM step for this operator.
"""

import json
import math
import os
from collections import Counter
from typing import Any, Dict, List, Optional

from constraints.verifier import ConstraintVerifier
from core.knowledge_base import HowNetKB
from core.tokenizer import ChineseTokenizer
from embeddings.jina_embedder import JinaEmbedder

RULES_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "rule_base", "synthesis_rules.json",
)

# Concrete sememe → abstract attribute, ordered because it drives C's wording.
ABSTRACT_BY_SEMEME = {
    "自夸": ["自我表扬"],
    "称赞": ["自我表扬"],
    "夸耀": ["自我表扬"],
    "夸大": ["夸大"],
    "失真": ["夸大"],
    "自信": ["过度自信"],
    "推销": ["自我推销"],
    "怀疑": ["质疑"],
    "不信任": ["质疑"],
    "批评": ["否定"],
    "否定": ["否定"],
    "教育": ["说理"],
    "警告": ["警示"],
    "提醒": ["警示"],
    "徒劳": ["徒劳无功"],
    "固执": ["不知变通"],
    "多余": ["多此一举"],
    "倚仗": ["攀附他人"],
    "自欺": ["自欺欺人"],
    "见识短浅": ["眼界狭隘"],
}
EVALUATIVE_ATTRIBUTES = ["自我表扬", "夸大", "过度自信", "质疑", "否定", "自欺欺人"]


class CommonalityExtraction:
    """Commonality Extraction Operator (Φ_C: L × L → C)."""

    def __init__(
        self,
        embedder: Optional[JinaEmbedder] = None,
        kb: Optional[HowNetKB] = None,
    ):
        self.embedder = embedder
        self.kb = kb or HowNetKB()
        self.tokenizer: ChineseTokenizer = self.kb.build_tokenizer()
        self.idf: Dict[str, float] = {}
        self.last_verification: Dict[str, Any] = {}
        self._build_tfidf_corpus()

    # ------------------------------------------------------------------
    # TF-IDF model
    # ------------------------------------------------------------------
    def _build_tfidf_corpus(self) -> None:
        documents: List[str] = []
        if os.path.exists(RULES_FILE):
            with open(RULES_FILE, "r", encoding="utf-8") as f:
                rules = json.load(f).get("rules", [])
            for rule in rules:
                sy = rule.get("Sy", [])
                documents.extend([sy] if isinstance(sy, str) else sy)
                documents.extend(rule.get("Sc_pattern", []))

        df: Counter = Counter()
        for document in documents:
            df.update(set(self._content_tokens(document)))
        total = max(len(documents), 1)
        self.idf = {
            term: math.log((total + 1) / (count + 1)) + 1.0
            for term, count in df.items()
        }
        self.default_idf = math.log(total + 1) + 1.0

    def _content_tokens(self, text: str) -> List[str]:
        return [
            token for token in self.tokenizer.tokenize(text or "")
            if token not in self.kb.stopwords and len(token) > 0
        ]

    def _extract_keywords_tfidf(self, text: str, top_k: int = 8) -> List[str]:
        tokens = self._content_tokens(text)
        if not tokens:
            return []
        counts = Counter(tokens)
        scored = [
            (count * self.idf.get(token, self.default_idf), token)
            for token, count in counts.items()
        ]
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [token for _, token in scored[:top_k]]

    def _jaccard(self, left: List[str], right: List[str]) -> Dict[str, Any]:
        set_left, set_right = set(left), set(right)
        intersection = set_left & set_right
        union = set_left | set_right
        return {
            "shared": [token for token in left if token in intersection],
            "score": len(intersection) / len(union) if union else 0.0,
        }

    # ------------------------------------------------------------------
    # HowNet abstraction
    # ------------------------------------------------------------------
    def _abstract_attributes(self, tokens: List[str]) -> List[str]:
        attributes: List[str] = []
        for sememe in self.kb.sememes_of(tokens):
            for attribute in ABSTRACT_BY_SEMEME.get(sememe, []):
                if attribute not in attributes:
                    attributes.append(attribute)
        return attributes

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def extract(self, EE: str, VE: str) -> str:
        components = self._build_components(EE, VE)
        self.last_verification = self._verify_overlap(
            EE, VE, self._template_assemble(components)
        )
        return self._template_assemble(components)

    def _build_components(self, EE: str, VE: str) -> Dict[str, Any]:
        ee_tokens = self._content_tokens(EE)
        ve_tokens = self._content_tokens(VE)
        ee_keywords = self._extract_keywords_tfidf(EE)
        ve_keywords = self._extract_keywords_tfidf(VE)
        overlap = self._jaccard(ee_keywords, ve_keywords)

        ee_sememes = self.kb.sememes_of(ee_tokens)
        ve_sememes = self.kb.sememes_of(ve_tokens)
        shared_sememes = [s for s in ee_sememes if s in set(ve_sememes)]

        ee_attributes = self._abstract_attributes(ee_tokens)
        ve_attributes = self._abstract_attributes(ve_tokens)
        common_attributes = [a for a in ve_attributes if a in ee_attributes]
        if not common_attributes:
            common_attributes = [a for a in ve_attributes if a in EVALUATIVE_ATTRIBUTES]
        if not common_attributes:
            common_attributes = ve_attributes[:3] or ee_attributes[:3]

        return {
            "ee_tokens": ee_tokens,
            "ve_tokens": ve_tokens,
            "ee_keywords": ee_keywords,
            "ve_keywords": ve_keywords,
            "shared_keywords": overlap["shared"],
            "jaccard": overlap["score"],
            "shared_sememes": shared_sememes,
            "ee_attributes": ee_attributes,
            "ve_attributes": ve_attributes,
            "attributes": common_attributes[:4],
            "expanded": self.kb.expand(overlap["shared"] or ee_keywords[:2], top_k=3),
        }

    def _template_assemble(self, components: Dict[str, Any]) -> str:
        attributes = components["attributes"] or ["语义重叠"]
        domains = [t for t in components["shared_keywords"] if t not in attributes]
        terms = attributes + domains[:2]
        clause = self._relational_clause(components)
        return f"{'、'.join(terms)}：{clause}。"

    def _relational_clause(self, components: Dict[str, Any]) -> str:
        sememes = set(self.kb.sememes_of(components["ee_tokens"]))
        commercial = {"交易", "商品", "市场"} & sememes
        has_ability = "能力" in components["ve_tokens"]
        if commercial and has_ability:
            return "卖方对自家货物与自身能力的评价明显高于客观事实"
        if commercial:
            return "卖方对自家货物质量的评价明显高于客观事实"
        if has_ability:
            return "说话者对自身能力的评价明显高于客观事实"
        return "说话者对自身条件或处境的评价明显高于客观事实"

    # ------------------------------------------------------------------
    # Step 5 — Jina-embeddings-v3 overlap verification
    # ------------------------------------------------------------------
    def _verify_overlap(self, EE: str, VE: str, C: str) -> Dict[str, Any]:
        if self.embedder is None:
            return {"passed": None, "reason": "no embedder configured"}
        threshold = ConstraintVerifier(self.embedder).thresholds["commonality"]
        sim_ee_ve = self.embedder.cosine_similarity(EE, VE)
        sim_c_ee = self.embedder.cosine_similarity(C, EE)
        sim_c_ve = self.embedder.cosine_similarity(C, VE)
        return {
            "passed": sim_ee_ve >= threshold,
            "sim_EE_VE": sim_ee_ve,
            "sim_C_EE": sim_c_ee,
            "sim_C_VE": sim_c_ve,
            "threshold": threshold,
        }
