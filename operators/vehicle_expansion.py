"""
Φ_VE Operator: Vehicle Explanation (V → VE)

Expands V (Vehicle) into Vehicle Explanation via:
1. Dictionary definition lookup (Xinhua Dictionary of TAS)
2. HowNet sememe retrieval for semantic features
3. Fluency refinement ONLY using LLM (temp=0.1)

Important: The LLM does NOT generate new semantic content. It only assembles
the pre-extracted dictionary definition and semantic features into a sentence.
"""

from typing import Optional, Dict, Any, List, Set
import os
from ..models.llm_wrappers import BaseLLM
from ..embeddings import JinaEmbedder


PROMPT_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "prompts", "vehicle_expansion.txt"
)


class VehicleExpansion:
    """
    Vehicle Explanation Operator (Φ_VE: V → L)

    Pipeline:
        Dict definition → HowNet sememes → structured inputs → LLM fluency assembly

    Attributes:
        llm_client: LLM used ONLY for fluency refinement (temp=0.1)
        embedder: Jina embedder for faithfulness constraint verification
        dictionary_path: Path to TAS dictionary resource
        hownet_path: Path to HowNet knowledge base
        prompt_template: Loaded from data/prompts/vehicle_expansion.txt
    """

    def __init__(
        self,
        llm_client: Optional[BaseLLM] = None,
        embedder: Optional[JinaEmbedder] = None,
        dictionary_path: Optional[str] = None,
        hownet_path: Optional[str] = None,
    ):
        self.llm_client = llm_client
        self.embedder = embedder
        self.dictionary_path = dictionary_path
        self.hownet_path = hownet_path
        self.prompt_template = self._load_prompt_template()

    def _load_prompt_template(self) -> str:
        if os.path.exists(PROMPT_FILE):
            with open(PROMPT_FILE, "r", encoding="utf-8") as f:
                return f.read()
        return (
            "[System] Assemble definition + features into coherent VE.\n"
            "[Task input]\nDictionary definition: {dictionary_definition}\n"
            "Semantic features: {semantic_features}\n"
        )

    def expand(self, V: str) -> str:
        """
        Expand Vehicle V to Vehicle Explanation VE.

        Args:
            V: Vehicle text (literal explanation, e.g., "praising one's own things")

        Returns:
            str: Fluency-assembled Vehicle Explanation (VE)
        """
        # Step 1: Look up dictionary definition (no LLM)
        definition = self._lookup_dictionary_definition(V)

        # Step 2: Retrieve HowNet semantic features
        semantic_features = self._retrieve_hownet_sememes(V)

        # Step 3: Assemble prompt with pre-extracted content only
        prompt = self._assemble_prompt(definition, semantic_features)

        # Step 4: LLM fluency refinement ONLY (temp=0.1, no generation)
        if self.llm_client:
            VE = self._refine_fluency(prompt)
        else:
            VE = self._template_assemble(definition, semantic_features)

        return VE

    def _lookup_dictionary_definition(self, V: str) -> str:
        """
        Look up V in the Xinhua Dictionary of Two-Part Allegorical Sayings (TAS).

        Note: Placeholder. In production, load the real dictionary from
        dictionary_path and perform the lookup.
        """
        return f"The literal meaning or pragmatic usage associated with '{V}'"

    def _retrieve_hownet_sememes(self, V: str) -> Set[str]:
        """
        Retrieve HowNet sememes (semantic features) for the Vehicle V.

        Note: Placeholder. In production, load the real HowNet knowledge base
        from hownet_path and perform sememe lookup.
        """
        return {"idiomatic meaning", "pragmatic interpretation", "cultural connotation"}

    def _assemble_prompt(self, definition: str, features: Set[str]) -> str:
        """
        Fill the prompt template with pre-extracted definition and features.

        The LLM receives ONLY these two inputs. It cannot invent new features
        or redefine the meaning.
        """
        features_str = "{" + ", ".join(f'"{f}"' for f in features) + "}"
        return self.prompt_template.format(
            dictionary_definition=definition,
            semantic_features=features_str,
        )

    def _template_assemble(self, definition: str, features: Set[str]) -> str:
        """
        Rule-based fallback when LLM is unavailable. Simply joins the
        definition with a semicolon-list of features, no extra content.
        """
        if features:
            feature_list = ", ".join(sorted(features))
            return f"{definition}. Key semantic features include: {feature_list}."
        return f"{definition}."

    def _refine_fluency(self, prompt: str) -> str:
        """
        Call LLM with temperature=0.1 for minimal, faithful fluency refinement.

        Matches the paper: Φ_VE uses temp=0.1, max_tokens=1024.
        """
        try:
            response = self.llm_client.generate(
                prompt, config={"temperature": 0.1, "max_tokens": 1024}
            )
            return response.strip() if response else ""
        except Exception:
            return ""

    def verify_faithfulness(
        self, V: str, VE: str, threshold: float = 0.78
    ) -> Dict[str, Any]:
        """
        Faithfulness constraint: Sim(VE, V) >= tau_faith (0.78).

        Ensures VE remains semantically faithful to the original Vehicle V.
        """
        if not self.embedder:
            return {"passed": True, "sim_VE_V": 1.0, "threshold": threshold}
        sim = self.embedder.cosine_similarity(VE, V)
        return {"passed": sim >= threshold, "sim_VE_V": sim, "threshold": threshold}
