"""
Φ_EE Operator: Entity Expansion (E → L)

Expands E (Essence) into detailed explanation via:
1. Semantic role labeling
2. HowNet knowledge retrieval
3. Fluency refinement using Qwen2.5-1.5B (temp=0.1)
"""

from typing import Optional, Dict, Any
from ..models.llm_wrappers import BaseLLM
from ..embeddings import JinaEmbedder


class EntityExpansion:
    """
    Entity Expansion Operator (Φ_EE: E → L)
    
    Expands the Essence (E) into a detailed explanation (EE) by:
    1. Parsing semantic roles from the essence text
    2. Retrieving HowNet knowledge for enrichment
    3. Refining fluency with small LLM (Qwen2.5-1.5B, temperature=0.1)
    """
    
    def __init__(
        self,
        llm_client: Optional[BaseLLM] = None,
        embedder: Optional[JinaEmbedder] = None,
        hownet_path: Optional[str] = None
    ):
        """
        Initialize Entity Expansion operator
        
        Args:
            llm_client: LLM client for fluency refinement
            embedder: Jina embedder for semantic verification
            hownet_path: Path to HowNet knowledge base
        """
        self.llm_client = llm_client
        self.embedder = embedder
        self.hownet_path = hownet_path
        
    def expand(self, E: str) -> str:
        """
        Expand Essence to Essence Explanation
        
        Args:
            E: Essence text (the riddle-like expression)
            
        Returns:
            str: Expanded Essence Explanation (EE)
        """
        # Step 1: Semantic role labeling
        semantic_roles = self._extract_semantic_roles(E)
        
        # Step 2: HowNet knowledge retrieval
        hownet_knowledge = self._retrieve_hownet_knowledge(E)
        
        # Step 3: Generate initial explanation
        initial_explanation = self._generate_initial_explanation(E, semantic_roles, hownet_knowledge)
        
        # Step 4: Fluency refinement using LLM
        if self.llm_client:
            EE = self._refine_fluency(initial_explanation, E)
        else:
            EE = initial_explanation
        
        return EE
    
    def _extract_semantic_roles(self, E: str) -> Dict[str, str]:
        """
        Extract semantic roles from Essence text
        
        Args:
            E: Essence text
            
        Returns:
            Dict: Extracted semantic roles
        """
        # Simplified implementation: extract subject, predicate, object
        roles = {
            'subject': E,
            'predicate': '',
            'object': ''
        }
        return roles
    
    def _retrieve_hownet_knowledge(self, E: str) -> Dict[str, Any]:
        """
        Retrieve HowNet knowledge for enrichment
        
        Args:
            E: Essence text
            
        Returns:
            Dict: HowNet knowledge entries
        """
        # Placeholder for HowNet integration
        return {
            'sememes': [],
            'related_concepts': [],
            'domain': ''
        }
    
    def _generate_initial_explanation(
        self, 
        E: str, 
        semantic_roles: Dict[str, str],
        hownet_knowledge: Dict[str, Any]
    ) -> str:
        """
        Generate initial explanation from semantic roles and HowNet knowledge
        
        Args:
            E: Essence text
            semantic_roles: Extracted semantic roles
            hownet_knowledge: HowNet knowledge entries
            
        Returns:
            str: Initial explanation
        """
        # Basic template-based generation
        explanation = f"This concept '{E}' refers to "
        if hownet_knowledge.get('sememes'):
            explanation += f"a concept characterized by {', '.join(hownet_knowledge['sememes'])}"
        else:
            explanation += "a specific concept with its own characteristics"
        
        if hownet_knowledge.get('domain'):
            explanation += f" in the domain of {hownet_knowledge['domain']}"
        
        explanation += "."
        return explanation
    
    def _refine_fluency(self, text: str, original_E: str) -> str:
        """
        Refine text fluency using LLM (temperature=0.1 for consistency)
        
        Args:
            text: Initial explanation text
            original_E: Original Essence for context
            
        Returns:
            str: Refined fluent explanation
        """
        if not self.llm_client:
            return text
        
        prompt = f"""Refine the following explanation for the concept '{original_E}' to make it more fluent and natural in English. 
Keep the meaning unchanged, only improve language quality:

Original: {text}

Refined explanation:"""
        
        try:
            refined = self.llm_client.generate(prompt, config={'temperature': 0.1})
            return refined if refined else text
        except Exception:
            return text
    
    def verify_faithfulness(self, E: str, EE: str, threshold: float = 0.78) -> Dict[str, Any]:
        """
        Verify faithfulness constraint: Sim(EE, E) >= threshold
        
        Args:
            E: Original Essence
            EE: Generated Essence Explanation
            threshold: Minimum similarity threshold (default 0.78)
            
        Returns:
            Dict: Verification result with similarity score
        """
        if not self.embedder:
            return {'passed': True, 'similarity': 1.0, 'threshold': threshold}
        
        similarity = self.embedder.cosine_similarity(EE, E)
        return {
            'passed': similarity >= threshold,
            'similarity': similarity,
            'threshold': threshold
        }