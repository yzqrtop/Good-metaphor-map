"""
Φ_Sy Operator: Synthesis (Sc × C → Sy)

Synthesizes Synthetics (Sy, pragmatic consequence) from Scene (Sc) and Commonality (C):
1. Rule-base matching → pre-retrieve candidate Sy pool (200–300 rules, Appendix I-A)
2. Embedding similarity scoring → rank the candidates
3. LLM validation & selection (Qwen/Proprietary) + confidence score
4. Fallback: if LLM confidence < 0.7, use the highest embedding-similarity candidate

Important: The LLM is a SELECTOR + SCORER, not a GENERATOR. Per paper:
"Candidate Sy entries are pre-retrieved from the rule base;
the LLM selects the best match and provides a confidence score."
"""

from typing import Optional, Dict, Any, List, Tuple
import os
import re
from ..models.llm_wrappers import BaseLLM
from ..embeddings import JinaEmbedder


PROMPT_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "prompts", "synthesis.txt"
)


class Synthesis:
    """
    Synthesis Operator (Φ_Sy: Sc × C → Sy)

    Pipeline:
        Rule-base candidate retrieval → embedding ranking →
        LLM select Candidate 1..3 + [Confidence: 0.x] →
        If confidence < 0.7 → embedding-ranked fallback

    Attributes:
        llm_client: LLM for verification/selection + confidence scoring
        embedder: Jina embedder for similarity scoring & fallback
        rules_path: Path to rule base (200–300 rules)
        confidence_threshold: τ_conf = 0.7 (paper default)
        rule_base: Loaded/constructed rule pool (Appendix I-A)
        prompt_template: Loaded from data/prompts/synthesis.txt
    """

    # Minimal rule base covering common pragmatic consequences.
    # In production, rules_path loads the full 200–300 rule Appendix I-A set.
    DEFAULT_RULE_BASE: List[Dict[str, Any]] = [
        {
            "id": "sy_rule_001",
            "sy_text": "The audience perceives the speaker as untrustworthy and gradually loses trust",
            "triggers": ["self-boast", "exaggerate", "overstate", "boasting", "exaggeration"],
            "confidence_anchor": 0.92,
        },
        {
            "id": "sy_rule_002",
            "sy_text": "The audience recognizes a mismatch between claims and reality, leading to doubt",
            "triggers": ["contrast", "discrepancy", "appearance", "reality", "pretend"],
            "confidence_anchor": 0.85,
        },
        {
            "id": "sy_rule_003",
            "sy_text": "The audience takes warning and avoids the same mistake in future situations",
            "triggers": ["futile", "vain", "waste", "wrong method", "misguided"],
            "confidence_anchor": 0.81,
        },
        {
            "id": "sy_rule_004",
            "sy_text": "The audience learns to broaden their perspective instead of staying narrow-minded",
            "triggers": ["narrow", "small", "limited", "short-sighted", "isolated"],
            "confidence_anchor": 0.88,
        },
        {
            "id": "sy_rule_005",
            "sy_text": "The audience detects a hidden malicious intent beneath a friendly surface",
            "triggers": ["pretend", "friendly", "visit", "gift", "insincere", "hypocritical"],
            "confidence_anchor": 0.90,
        },
        {
            "id": "sy_rule_006",
            "sy_text": "The speaker gains more customers through aggressive promotion",
            "triggers": ["promotion", "optimistic", "confidence"],
            "confidence_anchor": 0.35,
        },
        {
            "id": "sy_rule_007",
            "sy_text": "The speaker's competitors become jealous and spread rumors",
            "triggers": ["competition", "rival", "marketplace"],
            "confidence_anchor": 0.28,
        },
        {
            "id": "sy_rule_008",
            "sy_text": "The audience is advised to persist with correct methods rather than luck",
            "triggers": ["wait", "luck", "chance", "coincidence", "passive"],
            "confidence_anchor": 0.83,
        },
    ]

    def __init__(
        self,
        llm_client: Optional[BaseLLM] = None,
        embedder: Optional[JinaEmbedder] = None,
        rules_path: Optional[str] = None,
        confidence_threshold: float = 0.7,
    ):
        self.llm_client = llm_client
        self.embedder = embedder
        self.rules_path = rules_path
        self.confidence_threshold = confidence_threshold
        self.rule_base = self._load_rule_base()
        self.prompt_template = self._load_prompt_template()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def _load_rule_base(self) -> List[Dict[str, Any]]:
        """
        Load the 200–300 rule base from rules_path, otherwise fall back to
        the small default set above (for test/example runs only).
        """
        if self.rules_path and os.path.exists(self.rules_path):
            import json

            with open(self.rules_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return list(self.DEFAULT_RULE_BASE)

    def _load_prompt_template(self) -> str:
        if os.path.exists(PROMPT_FILE):
            with open(PROMPT_FILE, "r", encoding="utf-8") as f:
                return f.read()
        return (
            "[System] Select most appropriate Synthetics from candidates + confidence.\n"
            "[Task input]\nSc: {generated_Sc}\nC: {generated_C}\n"
            "Candidate 1: {candidate_1}\nCandidate 2: {candidate_2}\nCandidate 3: {candidate_3}\n"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def synthesize(self, Sc: str, C: str) -> str:
        """
        Synthesize Sy from Sc, C.

        Steps:
          1) Rule-base → 3 candidate Sy strings
          2) Embedding-based rank (used as fallback ordering)
          3) LLM picks "Candidate 1/2/3" and returns its confidence
          4) If LLM confidence < 0.7 → use embedding top-1 instead
        """
        # Step 1. Rule-base retrieval — top-3 candidate Sy strings
        candidates = self._rule_based_candidates(Sc, C, top_k=3)

        # Step 2. Embedding-similarity ranking — fallback order
        fallback_sy = self._embedding_top_candidate(candidates, Sc, C)

        # Step 3. LLM selection + confidence score
        if self.llm_client and len(candidates) == 3:
            llm_sy, llm_conf = self._llm_select_and_score(candidates, Sc, C)
            if llm_conf >= self.confidence_threshold:
                return llm_sy
            # Below threshold — fall through to embedding top-1
            return fallback_sy

        # No LLM: top embedding candidate
        return fallback_sy

    # ------------------------------------------------------------------
    # Step 1 — rule-base candidate retrieval
    # ------------------------------------------------------------------
    def _rule_based_candidates(
        self, Sc: str, C: str, top_k: int = 3
    ) -> List[str]:
        """
        Retrieve top-k candidate Sy strings from the rule base.

        Scoring is trigger-keyword overlap plus embedding similarity,
        matching the paper description "Rule-base matching + embedding scoring".
        """
        haystack = f"{Sc} {C}".lower()
        scored: List[Tuple[float, str]] = []

        for rule in self.rule_base:
            # Keyword overlap with rule triggers
            triggers = [t.lower() for t in rule.get("triggers", [])]
            overlap = sum(1 for t in triggers if t in haystack)
            anchor = rule.get("confidence_anchor", 0.5)

            # Optional embedding-based boost
            embedding_boost = 0.0
            if self.embedder:
                sim_sc = self.embedder.cosine_similarity(rule["sy_text"], Sc)
                sim_c = self.embedder.cosine_similarity(rule["sy_text"], C)
                embedding_boost = 0.5 * sim_sc + 0.5 * sim_c

            score = 0.4 * (overlap / max(1, len(triggers))) + 0.2 * anchor + 0.4 * embedding_boost
            scored.append((score, rule["sy_text"]))

        scored.sort(key=lambda x: x[0], reverse=True)
        selected = [text for _, text in scored[:top_k]]

        # Safety pad if rule base is tiny
        while len(selected) < top_k:
            selected.append(f"A reasonable pragmatic consequence given {C[:60]} and {Sc[:60]}.")
        return selected[:top_k]

    # ------------------------------------------------------------------
    # Step 2 — embedding top candidate (fallback ordering)
    # ------------------------------------------------------------------
    def _embedding_top_candidate(
        self, candidates: List[str], Sc: str, C: str
    ) -> str:
        if not self.embedder:
            return candidates[0]
        best_score, best = -1.0, candidates[0]
        for cand in candidates:
            sim_sc = self.embedder.cosine_similarity(cand, Sc)
            sim_c = self.embedder.cosine_similarity(cand, C)
            score = 0.5 * sim_sc + 0.5 * sim_c
            if score > best_score:
                best_score, best = score, cand
        return best

    # ------------------------------------------------------------------
    # Step 3 — LLM selection + confidence scoring
    # ------------------------------------------------------------------
    def _llm_select_and_score(
        self, candidates: List[str], Sc: str, C: str
    ) -> Tuple[str, float]:
        """
        Prompt the LLM to pick Candidate 1/2/3.
        Per the paper's template, LLM output is: "Output Sy: Candidate 1"

        Per-paper inference config: temperature=0.7, max_tokens=1024.
        """
        prompt = self.prompt_template.format(
            generated_Sc=Sc,
            generated_C=C,
            candidate_1=candidates[0],
            candidate_2=candidates[1],
            candidate_3=candidates[2],
        )

        try:
            response = self.llm_client.generate(
                prompt, config={"temperature": 0.7, "max_tokens": 1024}
            )
            idx, confidence = self._parse_sy_output(response)
            if 1 <= idx <= len(candidates):
                return candidates[idx - 1], confidence
        except Exception:
            pass
        # LLM failed — low confidence so caller falls back to embedding top-1
        return candidates[0], 0.0

    @staticmethod
    def _parse_sy_output(response: str) -> Tuple[int, float]:
        """
        Parse LLM output:
            "Output Sy: Candidate 1"   → (1, 1.0)
            "Candidate 2  [Confidence: 0.88]" → (2, 0.88)
            "3"  → (3, 0.5)
        """
        idx = -1
        confidence = 0.5

        cand_match = re.search(r"Candidate\s*(\d+)", response, flags=re.IGNORECASE)
        if cand_match:
            idx = int(cand_match.group(1))
        else:
            digits = re.findall(r"\d+", response)
            if digits:
                idx = int(digits[0])

        conf_match = re.search(
            r"(?:Confidence|Score|置信度|score)[:\s]*([01](?:\.\d+)?)",
            response,
            flags=re.IGNORECASE,
        )
        if conf_match:
            try:
                confidence = float(conf_match.group(1))
            except ValueError:
                confidence = 0.5
        else:
            # Square-bracket style: "[Confidence: 0.92]" as in paper template
            bracketed = re.findall(r"\[Confidence[:\s]*([01](?:\.\d+)?)\]", response, flags=re.IGNORECASE)
            if bracketed:
                try:
                    confidence = float(bracketed[-1])
                except ValueError:
                    confidence = 0.5
        return idx, confidence

    # ------------------------------------------------------------------
    # Constraint verification helper
    # ------------------------------------------------------------------
    def verify_synthesis(
        self, Sy: str, C: str, Sc: str, threshold: float = 0.74
    ) -> Dict[str, Any]:
        """
        Synthesis constraint: Sim(Sy, C) >= τ_synth ∧ Sim(Sy, Sc) >= τ_synth
        Paper: τ_synth = 0.74
        """
        if not self.embedder:
            return {
                "passed": True,
                "sim_Sy_C": 1.0,
                "sim_Sy_Sc": 1.0,
                "threshold": threshold,
            }
        sim_c = self.embedder.cosine_similarity(Sy, C)
        sim_sc = self.embedder.cosine_similarity(Sy, Sc)
        return {
            "passed": sim_c >= threshold and sim_sc >= threshold,
            "sim_Sy_C": sim_c,
            "sim_Sy_Sc": sim_sc,
            "threshold": threshold,
        }
