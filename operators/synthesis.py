"""
Φ_Sy Operator: Synthesis (Sc × C → Sy)

Pipeline (paper, Appendix E, Operator 5):
    1. Keyword extraction from Sc and C
    2. Matching against the curated <C_pattern, Sc_pattern, Sy> rule base
       (200–300 rules) using embedding similarity
    3. Top-3 candidates are kept
    4. The LLM validates/selects the most plausible candidate and returns a
       confidence score; below 0.7 the embedding-ranked candidate is used

The LLM is a SELECTOR + SCORER, not a GENERATOR: every Sy string comes from
the rule base.
"""

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from embeddings.jina_embedder import JinaEmbedder
from llm_wrappers import BaseLLM, GenerationConfig

PROMPT_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "prompts", "synthesis.txt",
)


class Synthesis:
    """Synthesis Operator (Φ_Sy: Sc × C → Sy)."""

    RULE_BASE_PATH = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "rule_base", "synthesis_rules.json",
    )

    def __init__(
        self,
        llm_client: Optional[BaseLLM] = None,
        embedder: Optional[JinaEmbedder] = None,
        rules_path: Optional[str] = None,
        confidence_threshold: float = 0.7,
    ):
        self.llm_client = llm_client
        self.embedder = embedder
        self.rules_path = rules_path or self.RULE_BASE_PATH
        self.confidence_threshold = confidence_threshold
        self.rule_base = self._load_rule_base()
        self.num_rules = len(self.rule_base)
        self.prompt_template = self._load_prompt_template()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def _load_rule_base(self) -> List[Dict[str, Any]]:
        """
        Expand each category entry into <C_pattern, Sc_pattern, Sy> triples.

        A seed entry holds one C_pattern, several prototypical scenes and several
        canonical Sy formulations; every (scene, Sy) pair inside a category is a
        valid rule instance, which yields the 200–300-rule inventory of the paper.
        """
        if not (self.rules_path and os.path.exists(self.rules_path)):
            raise FileNotFoundError(
                f"Synthesis rule base not found: {self.rules_path}. "
                "Phi_Sy cannot run without its retrieval corpus."
            )
        with open(self.rules_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        seeds = data.get("rules", data) if isinstance(data, dict) else data
        if not isinstance(seeds, list) or not seeds:
            raise ValueError(f"Malformed rule base: {self.rules_path}")

        rules: List[Dict[str, Any]] = []
        for seed in seeds:
            sy_field = seed.get("Sy", [])
            sy_list = [sy_field] if isinstance(sy_field, str) else list(sy_field or [])
            for scene in seed.get("Sc_pattern", []):
                for sy in sy_list:
                    if not sy:
                        continue
                    rules.append({
                        "id": seed.get("id"),
                        "category": seed.get("category"),
                        "C_pattern": list(seed.get("C_pattern", [])),
                        "Sc_pattern": scene,
                        "Sy": sy,
                    })
        return rules

    def _load_prompt_template(self) -> str:
        if os.path.exists(PROMPT_FILE):
            with open(PROMPT_FILE, "r", encoding="utf-8") as f:
                return f.read()
        raise FileNotFoundError(f"Prompt template not found: {PROMPT_FILE}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def synthesize(self, Sc: str, C: str) -> str:
        candidates = self._rule_based_candidates(Sc, C, top_k=3)
        fallback_sy = self._embedding_top_candidate(candidates, Sc, C)

        if self.llm_client and candidates:
            llm_sy, confidence = self._llm_select_and_score(candidates, Sc, C)
            if confidence >= self.confidence_threshold and llm_sy:
                return llm_sy
        return fallback_sy

    # ------------------------------------------------------------------
    # Steps 1–3 — rule-base retrieval
    # ------------------------------------------------------------------
    def _rule_based_candidates(
        self, Sc: str, C: str, top_k: int = 3
    ) -> List[str]:
        scored: List[Tuple[float, str]] = []
        matched: List[Tuple[float, str]] = []

        for rule in self.rule_base:
            c_patterns = rule.get("C_pattern", [])
            c_text = "、".join(c_patterns)
            c_score = (
                self.embedder.cosine_similarity(C, c_text)
                if self.embedder and c_text else 0.0
            )

            sc_pattern = rule.get("Sc_pattern", "")
            sc_score = (
                self.embedder.cosine_similarity(Sc, sc_pattern)
                if self.embedder and sc_pattern else 0.0
            )

            sy_text = rule.get("Sy", "")
            if not sy_text:
                continue
            sy_score = 0.0
            if self.embedder:
                sy_score = 0.5 * self.embedder.cosine_similarity(sy_text, Sc) \
                    + 0.5 * self.embedder.cosine_similarity(sy_text, C)

            score = 0.4 * c_score + 0.2 * sc_score + 0.4 * sy_score
            scored.append((score, sy_text))
            if any(term and term in C for term in c_patterns):
                matched.append((score, sy_text))

        # Rules whose C_pattern actually occurs in C are the structurally valid
        # matches; within that pool (or across all rules when none matched) the
        # candidates are ordered by embedding similarity, as the paper specifies.
        pool = matched or scored
        pool.sort(key=lambda item: item[0], reverse=True)

        selected: List[str] = []
        for _, text in pool:
            if text and text not in selected:
                selected.append(text)
            if len(selected) >= top_k:
                break
        if not selected:
            raise LookupError("No Sy candidate retrieved from the rule base.")
        return selected

    def _embedding_top_candidate(
        self, candidates: List[str], Sc: str, C: str
    ) -> str:
        if not self.embedder:
            return candidates[0]
        best_score, best = -1.0, candidates[0]
        for candidate in candidates:
            score = 0.5 * self.embedder.cosine_similarity(candidate, Sc) \
                + 0.5 * self.embedder.cosine_similarity(candidate, C)
            if score > best_score:
                best_score, best = score, candidate
        return best

    # ------------------------------------------------------------------
    # Step 4 — LLM selection + confidence
    # ------------------------------------------------------------------
    def _llm_select_and_score(
        self, candidates: List[str], Sc: str, C: str
    ) -> Tuple[str, float]:
        padded = candidates + [""] * (3 - len(candidates))
        prompt = self.prompt_template.format(
            Sc=Sc,
            C=C,
            c1=padded[0],
            c2=padded[1],
            c3=padded[2],
        )
        try:
            response = self.llm_client.generate(
                prompt, config=GenerationConfig(temperature=0.7, max_tokens=1024)
            )
            index, confidence = self._parse_sy_output(response or "")
            if 1 <= index <= len(candidates):
                return candidates[index - 1], confidence
        except Exception:
            pass
        return candidates[0], 0.0

    @staticmethod
    def _parse_sy_output(response: str) -> Tuple[int, float]:
        """Parse ``Candidate 2 / Confidence: 0.88`` (English or Chinese)."""
        index = -1
        confidence = 0.5

        cand_match = re.search(r"Candidate\s*(\d+)", response, flags=re.IGNORECASE)
        if cand_match:
            index = int(cand_match.group(1))
        else:
            cn_match = re.search(r"候选\s*(\d+)", response)
            if cn_match:
                index = int(cn_match.group(1))
            else:
                digits = re.findall(r"\d+", response)
                if digits:
                    index = int(digits[0])

        conf_match = re.search(
            r"(?:Confidence|Score|score|置信度)[:\s]*([01](?:\.\d+)?)",
            response,
            flags=re.IGNORECASE,
        )
        if conf_match:
            try:
                confidence = float(conf_match.group(1))
            except ValueError:
                confidence = 0.5
        else:
            bracketed = re.findall(
                r"\[(?:Confidence|置信度)[:\s]*([01](?:\.\d+)?)\]",
                response,
                flags=re.IGNORECASE,
            )
            if bracketed:
                try:
                    confidence = float(bracketed[-1])
                except ValueError:
                    confidence = 0.5
        return index, confidence
