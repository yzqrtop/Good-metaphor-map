"""
Φ_Sc Operator: Scene Generalization (L×L → Sc)

Generates Scene (Sc) from EE and VE via:
1. Template-based candidate generation (3-5 scenes)
2. LLM selection (Qwen2.5-7B, selection only)
3. Embedding verification
"""

from typing import Optional, Dict, Any, List
from ..models.llm_wrappers import BaseLLM
from ..embeddings import JinaEmbedder


class SceneGeneralization:
    """
    Scene Generalization Operator (Φ_Sc: L×L → Sc)
    
    Generates a generalized Scene (Sc) from Essence Explanation (EE) 
    and Vehicle Explanation (VE) by:
    1. Generating 3-5 candidate scenes using templates
    2. Selecting the best scene using LLM (Qwen2.5-7B, selection only)
    3. Verifying with embedding similarity
    """
    
    # Scene templates for candidate generation
    SCENE_TEMPLATES = [
        "A situation where {topic} manifests in daily life, involving {aspect1} and {aspect2}.",
        "In the context of {domain}, {concept} often leads to {outcome}.",
        "When {condition}, the relationship between {entity} and {phenomenon} becomes apparent.",
        "This scenario typically occurs when {trigger}, resulting in {consequence}.",
        "A common situation involving {subject}, where {action} leads to {result}."
    ]
    
    def __init__(
        self,
        llm_client: Optional[BaseLLM] = None,
        embedder: Optional[JinaEmbedder] = None,
        num_candidates: int = 4
    ):
        """
        Initialize Scene Generalization operator
        
        Args:
            llm_client: LLM client for scene selection
            embedder: Jina embedder for verification
            num_candidates: Number of candidate scenes to generate (3-5)
        """
        self.llm_client = llm_client
        self.embedder = embedder
        self.num_candidates = min(max(num_candidates, 3), 5)
        
    def generalize(self, EE: str, VE: str) -> str:
        """
        Generalize Scene from Essence Explanation and Vehicle Explanation
        
        Args:
            EE: Essence Explanation text
            VE: Vehicle Explanation text
            
        Returns:
            str: Generated Scene (Sc)
        """
        # Step 1: Template-based candidate generation
        candidates = self._generate_candidates(EE, VE)
        
        if len(candidates) == 0:
            candidates = [self._generate_default_scene(EE, VE)]
        
        # Step 2: LLM selection
        if self.llm_client and len(candidates) > 1:
            Sc = self._select_best_scene(candidates, EE, VE)
        else:
            Sc = candidates[0]
        
        # Step 3: Embedding verification (if embedder available)
        if self.embedder:
            verification = self._verify_scene(Sc, EE, VE)
            if not verification['passed']:
                # Fallback to best matching candidate
                Sc = self._find_best_matching_candidate(candidates, EE, VE)
        
        return Sc
    
    def _generate_candidates(self, EE: str, VE: str) -> List[str]:
        """
        Generate candidate scenes using templates
        
        Args:
            EE: Essence Explanation text
            VE: Vehicle Explanation text
            
        Returns:
            List[str]: List of candidate scenes
        """
        candidates = []
        ee_words = EE.split()[:3] if EE.split() else [EE]
        ve_words = VE.split()[:3] if VE.split() else [VE]
        
        # Generate candidates using templates
        for i in range(min(self.num_candidates, len(self.SCENE_TEMPLATES))):
            template = self.SCENE_TEMPLATES[i]
            try:
                candidate = template.format(
                    topic=ee_words[0] if ee_words else 'this topic',
                    aspect1=ve_words[0] if ve_words else 'aspects',
                    aspect2='various elements',
                    domain='this domain',
                    concept=EE[:50],
                    outcome='specific outcomes',
                    condition='certain conditions',
                    entity='entities',
                    phenomenon='phenomena',
                    trigger='specific triggers',
                    consequence='consequences',
                    subject='a subject',
                    action='an action',
                    result='a result'
                )
                candidates.append(candidate)
            except (KeyError, IndexError):
                # Fallback if template formatting fails
                candidates.append(self._generate_default_scene(EE, VE))
        
        return candidates
    
    def _generate_default_scene(self, EE: str, VE: str) -> str:
        """
        Generate a default scene description
        
        Args:
            EE: Essence Explanation text
            VE: Vehicle Explanation text
            
        Returns:
            str: Default scene
        """
        return (
            f"A situation where the concepts expressed in '{EE}' and '{VE}' "
            f"interact, revealing their underlying relationship and shared characteristics."
        )
    
    def _select_best_scene(
        self, 
        candidates: List[str], 
        EE: str, 
        VE: str
    ) -> str:
        """
        Select the best scene using LLM (selection only)
        
        Args:
            candidates: List of candidate scenes
            EE: Essence Explanation text
            VE: Vehicle Explanation text
            
        Returns:
            str: Best selected scene
        """
        candidates_text = "\n".join(
            [f"{i+1}. {c}" for i, c in enumerate(candidates)]
        )
        
        prompt = f"""Select the best scene description that best encompasses 
the concepts from both explanations below:

Essence Explanation: {EE}
Vehicle Explanation: {VE}

Candidate scenes:
{candidates_text}

Reply with ONLYHING but the number of the best scene (1, 2, 3, etc.):"""
        
        try:
            response = self.llm_client.generate(prompt, config={'temperature': 0.0})
            # Extract number from response
            number = int(response.strip()[0])
            if 1 <= number <= len(candidates):
                return candidates[number - 1]
        except (ValueError, IndexError, Exception):
            pass
        
        # Default to first candidate
        return candidates[0]
    
    def _verify_scene(self, Sc: str, EE: str, VE: str) -> Dict[str, Any]:
        """
        Verify scene relevance using embedding similarity
        
        Args:
            Sc: Scene text
            EE: Essence Explanation text
            VE: Vehicle Explanation text
            
        Returns:
            Dict: Verification result
        """
        if not self.embedder:
            return {'passed': True}
        
        sim_Sc_EE = self.embedder.cosine_similarity(Sc, EE)
        sim_Sc_VE = self.embedder.cosine_similarity(Sc, VE)
        
        # Scene should be relevant to both explanations
        threshold = 0.5
        return {
            'passed': sim_Sc_EE >= threshold and sim_Sc_VE >= threshold,
            'sim_Sc_EE': sim_Sc_EE,
            'sim_Sc_VE': sim_Sc_VE
        }
    
    def _find_best_matching_candidate(
        self, 
        candidates: List[str], 
        EE: str, 
        VE: str
    ) -> str:
        """
        Find the best matching candidate based on embedding similarity
        
        Args:
            candidates: List of candidate scenes
            EE: Essence Explanation text
            VE: Vehicle Explanation text
            
        Returns:
            str: Best matching candidate
        """
        if not self.embedder or len(candidates) == 0:
            return candidates[0] if candidates else self._generate_default_scene(EE, VE)
        
        best_score = -1
        best_candidate = candidates[0]
        
        for candidate in candidates:
            sim_EE = self.embedder.cosine_similarity(candidate, EE)
            sim_VE = self.embedder.cosine_similarity(candidate, VE)
            score = (sim_EE + sim_VE) / 2
            
            if score > best_score:
                best_score = score
                best_candidate = candidate
        
        return best_candidate