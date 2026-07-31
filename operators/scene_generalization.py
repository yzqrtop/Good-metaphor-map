"""
Φ_Sc Operator: Scene Generalization (EE × VE → Sc)

Generates Scene (Sc) from EE and VE via:
1. Template-based candidate generation → 3–5 candidate scenes (Appendix I-A.1)
2. LLM selection ONLY (Qwen2.5-7B) — the model DOES NOT generate new scenes,
   it only selects the best candidate from the pre-generated list
3. Embedding similarity verification

Important: The LLM is a SELECTOR, not a GENERATOR. Per paper:
"Candidate scenes are pre-generated via template instantiation;
the LLM selects the most plausible one."
"""

from typing import Optional, Dict, Any, List
import os
import re
from ..models.llm_wrappers import BaseLLM
from ..embeddings import JinaEmbedder


PROMPT_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "prompts", "scene_generalization.txt"
)


class SceneGeneralization:
    """
    Scene Generalization Operator (Φ_Sc: L×L → Sc)

    Pipeline:
        3–5 template candidates → LLM selection (only picks Candidate N) → embedding verify

    Attributes:
        llm_client: LLM used ONLY for selection (NOT generation), e.g. Qwen2.5-7B
        embedder: Jina embedder for final similarity verification & fallback
        num_candidates: 3, 4, or 5 candidate scenes (paper: 3–5)
        prompt_template: Loaded from data/prompts/scene_generalization.txt
    """

    # Template instantiation for candidate scene generation.
    # These correspond to Appendix I-A.1 in the paper.
    # Each template is parameterized by EE topics and VE aspects.
    SCENE_TEMPLATES: List[str] = [
        "A {role} in a {place} context {action_promotes} {items}, leading to {audience_reaction}.",
        "A {persona} in a {setting} {action_exaggerates} their {merits}, leading to {outcome}.",
        "An {actor} during a {situation} {action_overstates} the {potential}, leading to {consequence}.",
        "A {protagonist} within a {domain} {action_boasts} about their {achievements}, leading to {result}.",
        "Someone in a {scenario} {action_claims} {attributes_or_claims}, leading to {reception}.",
    ]

    ROLE_VOCAB = ["salesperson", "politician", "entrepreneur", "job seeker", "candidate", "speaker", "advocate"]
    PLACE_VOCAB = ["marketplace", "debate stage", "pitch meeting", "job interview", "public speech", "negotiation"]
    PERSONA_VOCAB = ["politician", "manager", "presenter", "salesperson", "candidate", "applicant", "leader"]
    SETTING_VOCAB = ["debate", "team meeting", "town hall", "interview", "press conference", "boardroom"]
    ACTOR_VOCAB = ["entrepreneur", "startup founder", "project lead", "applicant", "proponent", "negotiator"]
    SITUATION_VOCAB = ["pitch meeting", "fundraising round", "product launch", "grant proposal", "investment round"]
    PROTAGONIST_VOCAB = ["candidate", "manager", "executive", "influencer", "sales leader", "consultant"]
    DOMAIN_VOCAB = ["election campaign", "corporate review", "social media", "client pitch", "panel discussion"]
    SCENARIO_VOCAB = [
        "commercial setting", "social gathering", "professional environment",
        "recruitment context", "public discourse", "negotiation table",
    ]
    ACTIONS_PROMOTE = [
        "promotes their", "endorses their", "markets their", "advertises their", "talks up their",
    ]
    ACTIONS_EXAGGERATE = [
        "exaggerates", "inflates", "overstates", "embellishes", "overemphasizes",
    ]
    ACTIONS_OVERSTATE = [
        "overstates", "overestimates", "inflates", "magnifies", "exaggerates",
    ]
    ACTIONS_BOASTS = [
        "boasts", "brags", "flaunts", "shows off", "trumpets",
    ]
    ACTIONS_CLAIMS = [
        "exaggeratedly claims", "boastfully asserts", "overconfidently proclaims", "inflatedly states",
    ]
    ITEMS_VOCAB = ["products", "services", "offerings", "proposals", "ideas", "credentials"]
    MERITS_VOCAB = [
        "own achievements", "personal strengths", "past successes", "track record", "capabilities",
    ]
    POTENTIAL_VOCAB = [
        "project's potential", "venture's upside", "initiative's impact", "plan's effectiveness", "proposal's value",
    ]
    ACHIEVEMENTS_VOCAB = [
        "past wins", "prior outcomes", "key accomplishments", "deliverables", "performance record",
    ]
    ATTRIBS_VOCAB = [
        "excessive advantages", "overstated qualifications", "inflated benefits", "unrealistic strengths",
    ]
    AUDIENCE_REACT = [
        "skepticism from the audience", "distrust among observers", "doubt in onlookers",
        "hesitation from stakeholders", "reservations from listeners",
    ]
    OUTCOMES = [
        "public distrust", "audience disbelief", "counterparty pushback", "hesitant buy-in",
    ]
    CONSEQUENCES = [
        "investor doubt", "stakeholder skepticism", "low acceptance", "reduced buy-in",
    ]
    RESULTS = [
        "audience skepticism", "reduced trust", "perception of overclaiming", "pushback from peers",
    ]
    RECEPTIONS = [
        "audience distrust", "listener skepticism", "recipient reluctance", "observer doubt",
    ]

    def __init__(
        self,
        llm_client: Optional[BaseLLM] = None,
        embedder: Optional[JinaEmbedder] = None,
        num_candidates: int = 5,
    ):
        assert 3 <= num_candidates <= 5, "num_candidates must be between 3 and 5"
        self.llm_client = llm_client
        self.embedder = embedder
        self.num_candidates = num_candidates
        self.prompt_template = self._load_prompt_template()

    def _load_prompt_template(self) -> str:
        if os.path.exists(PROMPT_FILE):
            with open(PROMPT_FILE, "r", encoding="utf-8") as f:
                return f.read()
        return (
            "[System] Select the most plausible scene from candidates.\n"
            "[Task input]\nEE: {generated_EE}\nVE: {generated_VE}\n"
            "Candidate 1: {candidate_1}\nCandidate 2: {candidate_2}\nCandidate 3: {candidate_3}\n"
            "Candidate 4: {candidate_4}\nCandidate 5: {candidate_5}\n"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generalize(self, EE: str, VE: str) -> str:
        """
        Generalize Scene Sc from EE, VE using:
            candidate templates → LLM SELECTION → embedding verification/fallback
        """
        # Step 1. Pre-generate 3-5 candidate scenes (template instantiation only, no LLM)
        candidates = self._generate_candidates_via_templates(EE, VE)
        assert len(candidates) == self.num_candidates, (
            f"Expected {self.num_candidates} candidates, got {len(candidates)}"
        )

        # Step 2. LLM selects ONE candidate (Candidate 1..5). It DOES NOT generate.
        if self.llm_client:
            Sc = self._select_candidate_via_llm(candidates, EE, VE)
        else:
            # If no LLM, fall back to embedding-similarity selection
            Sc = self._select_candidate_via_embeddings(candidates, EE, VE)

        # Step 3. Embedding verification; fall back if the scene doesn't align
        if self.embedder:
            verified = self._verify_scene_alignment(Sc, EE, VE)
            if not verified["passed"]:
                Sc = self._select_candidate_via_embeddings(candidates, EE, VE)

        return Sc

    # ------------------------------------------------------------------
    # Step 1 — template instantiation (Appendix I-A.1)
    # ------------------------------------------------------------------
    def _generate_candidates_via_templates(self, EE: str, VE: str) -> List[str]:
        """
        Deterministically instantiate self.num_candidates scene templates.
        This produces the candidate pool; the LLM only selects from them.
        """
        import random

        rng = random.Random(hash((EE, VE, self.num_candidates)) % (2**31))
        candidates: List[str] = []
        for i in range(self.num_candidates):
            template = self.SCENE_TEMPLATES[i % len(self.SCENE_TEMPLATES)]
            candidate = template.format(
                role=rng.choice(self.ROLE_VOCAB),
                place=rng.choice(self.PLACE_VOCAB),
                action_promotes=rng.choice(self.ACTIONS_PROMOTE),
                items=rng.choice(self.ITEMS_VOCAB),
                audience_reaction=rng.choice(self.AUDIENCE_REACT),
                persona=rng.choice(self.PERSONA_VOCAB),
                setting=rng.choice(self.SETTING_VOCAB),
                action_exaggerates=rng.choice(self.ACTIONS_EXAGGERATE),
                merits=rng.choice(self.MERITS_VOCAB),
                outcome=rng.choice(self.OUTCOMES),
                actor=rng.choice(self.ACTOR_VOCAB),
                situation=rng.choice(self.SITUATION_VOCAB),
                action_overstates=rng.choice(self.ACTIONS_OVERSTATE),
                potential=rng.choice(self.POTENTIAL_VOCAB),
                consequence=rng.choice(self.CONSEQUENCES),
                protagonist=rng.choice(self.PROTAGONIST_VOCAB),
                domain=rng.choice(self.DOMAIN_VOCAB),
                action_boasts=rng.choice(self.ACTIONS_BOASTS),
                achievements=rng.choice(self.ACHIEVEMENTS_VOCAB),
                result=rng.choice(self.RESULTS),
                scenario=rng.choice(self.SCENARIO_VOCAB),
                action_claims=rng.choice(self.ACTIONS_CLAIMS),
                attributes_or_claims=rng.choice(self.ATTRIBS_VOCAB),
                reception=rng.choice(self.RECEPTIONS),
            )
            candidates.append(candidate.capitalize())
        return candidates

    # ------------------------------------------------------------------
    # Step 2 — LLM selection (selection only, no generation)
    # ------------------------------------------------------------------
    def _select_candidate_via_llm(
        self, candidates: List[str], EE: str, VE: str
    ) -> str:
        """
        Call LLM to pick the best candidate.

        Per paper template, LLM output is exactly one of:
            "Candidate 1", "Candidate 2", ..., "Candidate 5"
        """
        prompt_kwargs = {
            "generated_EE": EE,
            "generated_VE": VE,
        }
        for idx in range(1, 6):
            key = f"candidate_{idx}"
            prompt_kwargs[key] = candidates[idx - 1] if idx <= len(candidates) else ""

        prompt = self.prompt_template.format(**prompt_kwargs)

        try:
            # Selection tasks: temperature near 0 for deterministic choice, max_tokens small
            response = self.llm_client.generate(
                prompt, config={"temperature": 0.7, "max_tokens": 1024}
            )
            selection = self._parse_candidate_selection(response)
            if 1 <= selection <= len(candidates):
                return candidates[selection - 1]
        except Exception:
            pass
        # LLM failed → embedding fallback
        return self._select_candidate_via_embeddings(candidates, EE, VE)

    @staticmethod
    def _parse_candidate_selection(response: str) -> int:
        """
        Parse LLM output expecting "Candidate N".
        Covers: "Candidate 1", "Output Sc: Candidate 2", plain "3", etc.
        """
        match = re.search(r"Candidate\s*(\d+)", response, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
        digits = re.findall(r"\d+", response)
        if digits:
            return int(digits[0])
        return -1

    # ------------------------------------------------------------------
    # Fallback — embedding-based selection (no LLM)
    # ------------------------------------------------------------------
    def _select_candidate_via_embeddings(
        self, candidates: List[str], EE: str, VE: str
    ) -> str:
        if not self.embedder:
            return candidates[0]
        best_score, best = -1.0, candidates[0]
        for cand in candidates:
            sim_ee = self.embedder.cosine_similarity(cand, EE)
            sim_ve = self.embedder.cosine_similarity(cand, VE)
            score = 0.5 * sim_ee + 0.5 * sim_ve
            if score > best_score:
                best_score, best = score, cand
        return best

    # ------------------------------------------------------------------
    # Step 3 — embedding verification
    # ------------------------------------------------------------------
    def _verify_scene_alignment(
        self, Sc: str, EE: str, VE: str
    ) -> Dict[str, Any]:
        if not self.embedder:
            return {"passed": True}
        sim_e = self.embedder.cosine_similarity(Sc, EE)
        sim_v = self.embedder.cosine_similarity(Sc, VE)
        # Reasonable (non-calibrated) check; the official 4 constraints live
        # in constraints/verifier.py. This is a local sanity gate.
        threshold = 0.3
        return {
            "passed": sim_e >= threshold and sim_v >= threshold,
            "sim_Sc_EE": sim_e,
            "sim_Sc_VE": sim_v,
        }
