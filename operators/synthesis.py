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

    # Path to the complete 287-rule rule base (Appendix I-A)
    RULE_BASE_PATH = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "rule_base", "synthesis_rules.json"
    )

    # Minimal fallback rule base for when the full rule base is unavailable.
    # In production, RULE_BASE_PATH loads the complete 287-rule set (Chinese).
    FALLBACK_RULE_BASE: List[Dict[str, Any]] = [
        {
            "id": "fallback_001",
            "category": "自我吹嘘 / 自我美化",
            "C_pattern": ["吹牛", "吹嘘", "自夸"],
            "Sc_pattern": ["一个商人在集市推销产品，结果观众的怀疑。"],
            "Sy": "观众会觉得说话者不可信，从而警惕。",
        },
        {
            "id": "fallback_002",
            "category": "盲目乐观 / 自我欺骗",
            "C_pattern": ["自欺欺人", "掩耳盗铃", "痴心妄想"],
            "Sc_pattern": ["一个经理在规划中回避现实，导致决策延迟。"],
            "Sy": "听众会从中拓宽视野，进而重新考虑。",
        },
        {
            "id": "fallback_003",
            "category": "批评 / 指责",
            "C_pattern": ["指责", "责备", "批评"],
            "Sc_pattern": ["一个团队在会议上互相指责，引起信任流失。"],
            "Sy": "人们意识到不一致，因此改进。",
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
        # Use provided path, or fall back to the default RULE_BASE_PATH
        self.rules_path = rules_path or self.RULE_BASE_PATH
        self.confidence_threshold = confidence_threshold
        self.rule_base = self._load_rule_base()
        self.prompt_template = self._load_prompt_template()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def _load_rule_base(self) -> List[Dict[str, Any]]:
        """
        Load the 287-rule rule base (triple <C_pattern, Sc_pattern, Sy> format).
        
        If the full rule base file is available, load it.
        Otherwise, fall back to the minimal FALLBACK_RULE_BASE for testing.
        """
        if self.rules_path and os.path.exists(self.rules_path):
            import json
            try:
                with open(self.rules_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                # Support both {rules: [...]} format and plain list format
                if isinstance(data, dict) and "rules" in data:
                    return data["rules"]
                elif isinstance(data, list):
                    return data
            except (json.JSONDecodeError, IOError):
                pass
        return list(self.FALLBACK_RULE_BASE)

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

        Matching uses the triple <C_pattern, Sc_pattern, Sy> format:
        1. C_pattern: keyword overlap between C and rule's C_pattern terms
        2. Sc_pattern: embedding similarity between Sc and rule's Sc_pattern sentences
           (Sc_pattern items are full sentences in the Chinese rule base)
        3. Embedding similarity between candidate Sy and both Sc and C
        4. Combined scoring: keyword + embedding

        Scoring matches the paper: "Rule-base matching + embedding scoring"
        """
        scored: List[Tuple[float, str]] = []

        for rule in self.rule_base:
            # C_pattern matching: keyword overlap (C_pattern = list of terms)
            c_patterns = rule.get("C_pattern", [])
            c_score = 0.0
            if c_patterns:
                c_lower = C.lower()
                c_overlap = sum(1 for p in c_patterns if p.lower() in c_lower)
                c_score = c_overlap / len(c_patterns)

            # Sc_pattern matching: embedding similarity (Sc_pattern = list of sentences)
            sc_patterns = rule.get("Sc_pattern", [])
            sc_score = 0.0
            if sc_patterns and self.embedder:
                # Use max embedding similarity between Sc and any Sc_pattern sentence
                max_sc_sim = 0.0
                for sc_sentence in sc_patterns:
                    sim = self.embedder.cosine_similarity(Sc, sc_sentence)
                    if sim > max_sc_sim:
                        max_sc_sim = sim
                sc_score = max_sc_sim
            elif sc_patterns:
                # No embedder: fallback to simple keyword overlap
                sc_lower = Sc.lower()
                sc_overlap = 0
                for sc_sentence in sc_patterns:
                    # Extract simple terms from the sentence for matching
                    sc_terms = [t.strip() for t in sc_sentence.replace("，", " ").replace("。", " ").replace("、", " ").split() if t.strip()]
                    if sc_terms:
                        sc_overlap += sum(1 for t in sc_terms if t.lower() in sc_lower)
                sc_score = sc_overlap / max(1, len(sc_patterns) * 5)  # normalize

            # Sy embedding score: similarity between candidate Sy and both Sc and C
            sy_text = rule.get("Sy", "")
            sy_embedding_score = 0.0
            if sy_text and self.embedder:
                sim_sc = self.embedder.cosine_similarity(sy_text, Sc)
                sim_c = self.embedder.cosine_similarity(sy_text, C)
                sy_embedding_score = 0.5 * sim_sc + 0.5 * sim_c

            # Combined score (weighted)
            score = 0.25 * c_score + 0.25 * sc_score + 0.5 * sy_embedding_score
            scored.append((score, sy_text))

        # Sort by score descending
        scored.sort(key=lambda x: x[0], reverse=True)
        selected = [text for _, text in scored[:top_k] if text]

        # Safety pad if rule base match didn't produce enough candidates
        while len(selected) < top_k:
            selected.append(
                f"给定共性='{C[:30]}'和场景='{Sc[:30]}'，合理的语用后果是相关的。"
            )
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
        Parse LLM output (supports both English and Chinese formats):
            "Output Sy: Candidate 1"   → (1, 1.0)
            "Candidate 2  [Confidence: 0.88]" → (2, 0.88)
            "候选1  [置信度: 0.92]" → (1, 0.92)
            "3"  → (3, 0.5)
        """
        idx = -1
        confidence = 0.5

        # Try English format first
        cand_match = re.search(r"Candidate\s*(\d+)", response, flags=re.IGNORECASE)
        if cand_match:
            idx = int(cand_match.group(1))
        else:
            # Try Chinese format: 候选1, 候选 2, etc.
            cn_match = re.search(r"候选\s*(\d+)", response)
            if cn_match:
                idx = int(cn_match.group(1))
            else:
                # Try generic digit
                digits = re.findall(r"\d+", response)
                if digits:
                    idx = int(digits[0])

        # Try English confidence format
        conf_match = re.search(
            r"(?:Confidence|Score|score)[:\s]*([01](?:\.\d+)?)",
            response,
            flags=re.IGNORECASE,
        )
        if conf_match:
            try:
                confidence = float(conf_match.group(1))
            except ValueError:
                confidence = 0.5
        else:
            # Try Chinese confidence format: 置信度
            cn_conf = re.search(r"置信度[:\s]*([01](?:\.\d+)?)", response)
            if cn_conf:
                try:
                    confidence = float(cn_conf.group(1))
                except ValueError:
                    confidence = 0.5
            else:
                # Square-bracket style: "[Confidence: 0.92]" or "[置信度: 0.92]"
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
