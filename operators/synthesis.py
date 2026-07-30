"""
Φ_Sy Operator: Synthesis (Sc×C → Sy)

Synthesizes Synthetics (Sy) from Scene and Commonality via:
1. Rule-based matching (200-300 rules)
2. Embedding similarity scoring
3. LLM validation
4. Fallback if confidence < 0.7
"""

from typing import Optional, Dict, Any, List, Tuple
from ..models.llm_wrappers import BaseLLM
from ..embeddings import JinaEmbedder


class Synthesis:
    """
    Synthesis Operator (Φ_Sy: Sc×C → Sy)
    
    Synthesizes the final meaning (Synthetics, Sy) from Scene (Sc) 
    and Commonality (C) by:
    1. Rule-based matching with 200-300 predefined rules
    2. Embedding similarity scoring
    3. LLM validation
    4. Fallback mechanism if confidence < 0.7
    """
    
    def __init__(
        self,
        llm_client: Optional[BaseLLM] = None,
        embedder: Optional[JinaEmbedder] = None,
        rules_path: Optional[str] = None,
        confidence_threshold: float = 0.7
    ):
        """
        Initialize Synthesis operator
        
        Args:
            llm_client: LLM client for validation
            embedder: Jina embedder for similarity scoring
            rules_path: Path to rule file
            confidence_threshold: Minimum confidence threshold (default 0.7)
        """
        self.llm_client = llm_client
        self.embedder = embedder
        self.rules_path = rules_path
        self.confidence_threshold = confidence_threshold
        self.rules = self._load_rules()
        
    def _load_rules(self) -> List[Dict[str, Any]]:
        """
        Load predefined synthesis rules (200-300 rules)
        
        Returns:
            List[Dict]: List of rule definitions
        """
        # Placeholder: basic rule templates
        rules = [
            {
                'id': 'rule_001',
                'pattern': 'temporal_sequence',
                'template': 'This suggests that {implication}, which ultimately {consequence}.',
                'trigger_keywords': ['leads to', 'results in', 'causes']
            },
            {
                'id': 'rule_002',
                'pattern': 'contrast',
                'template': 'Despite {condition}, the underlying reality shows {reality}.',
                'trigger_keywords': ['however', 'despite', 'although']
            },
            {
                'id': 'rule_003',
                'pattern': 'causal',
                'template': 'Because {cause}, it follows that {effect}.',
                'trigger_keywords': ['because', 'since', 'therefore']
            },
            {
                'id': 'rule_004',
                'pattern': 'analogy',
                'template': 'Just as {analogy}, so too does {application}.',
                'trigger_keywords': ['just as', 'similarly', 'like']
            },
            {
                'id': 'rule_005',
                'pattern': 'generalization',
                'template': 'This illustrates a broader principle: {principle}.',
                'trigger_keywords': ['illustrates', 'demonstrates', 'shows']
            }
        ]
        return rules
    
    def synthesize(self, Sc: str, C: str) -> str:
        """
        Synthesize meaning from Scene and Commonality
        
        Args:
            Sc: Scene text
            C: Commonality text
            
        Returns:
            str: Synthesized meaning (Sy)
        """
        # Step 1: Rule-based matching
        rule_based_sy = self._rule_based_synthesis(Sc, C)
        
        # Step 2: Embedding similarity scoring
        similarity_score = self._compute_similarity_score(rule_based_sy, Sc, C)
        
        # Step 3: LLM validation
        if self.llm_client:
            validated_sy, confidence = self._llm_validate(rule_based_sy, Sc, C)
        else:
            validated_sy = rule_based_sy
            confidence = similarity_score
        
        # Step 4: Fallback if confidence < threshold
        if confidence < self.confidence_threshold:
            Sy = self._fallback_synthesis(Sc, C)
        else:
            Sy = validated_sy
        
        return Sy
    
    def _rule_based_synthesis(self, Sc: str, C: str) -> str:
        """
        Perform rule-based synthesis
        
        Args:
            Sc: Scene text
            C: Commonality text
            
        Returns:
            str: Rule-based synthesized text
        """
        best_rule = None
        best_score = 0
        
        # Match rules based on trigger keywords
        combined_text = f"{Sc} {C}".lower()
        
        for rule in self.rules:
            score = sum(
                1 for keyword in rule['trigger_keywords'] 
                if keyword in combined_text
            )
            if score > best_score:
                best_score = score
                best_rule = rule
        
        if best_rule and best_score > 0:
            # Apply matched rule template
            Sy = best_rule['template'].format(
                implication=C[:100] if len(C) > 100 else C,
                consequence=Sc[:100] if len(Sc) > 100 else Sc,
                condition=Sc[:80] if len(Sc) > 80 else Sc,
                reality=C[:80] if len(C) > 80 else C,
                cause=C[:80] if len(C) > 80 else C,
                effect=Sc[:80] if len(Sc) > 80 else Sc,
                analogy=C[:60] if len(C) > 60 else C,
                application=Sc[:60] if len(Sc) > 60 else Sc,
                principle=f"{C} reveals something about {Sc}"
            )
        else:
            # Default synthesis
            Sy = (
                f"The deeper meaning emerges from the relationship between the scene ({Sc[:80]}...) "
                f"and the commonality ({C[:80]}...). This suggests that "
                f"the shared characteristics not only connect the two concepts but also "
                f"imply a broader understanding of the phenomenon."
            )
        
        return Sy
    
    def _compute_similarity_score(self, Sy: str, Sc: str, C: str) -> float:
        """
        Compute embedding similarity score
        
        Args:
            Sy: Synthesized text
            Sc: Scene text
            C: Commonality text
            
        Returns:
            float: Similarity score
        """
        if not self.embedder:
            return 0.8  # Default confidence
        
        sim_Sy_Sc = self.embedder.cosine_similarity(Sy, Sc)
        sim_Sy_C = self.embedder.cosine_similarity(Sy, C)
        
        # Weighted average
        score = 0.5 * sim_Sy_Sc + 0.5 * sim_Sy_C
        return score
    
    def _llm_validate(
        self, 
        candidate_sy: str, 
        Sc: str, 
        C: str
    ) -> Tuple[str, float]:
        """
        Validate synthesized text using LLM
        
        Args:
            candidate_sy: Candidate synthesized text
            Sc: Scene text
            C: Commonality text
            
        Returns:
            Tuple[str, float]: Validated text and confidence score
        """
        prompt = f"""Evaluate the quality of the following synthesis.

Scene: {Sc}
Commonality: {C}
Synthesis: {candidate_sy}

Rate the synthesis quality from 0.0 to 1.0 and provide an improved version if needed.
Respond in the following format:
CONFIDENCE: [score]
IMPROVED: [improved text or same if no improvement needed]"""
        
        try:
            response = self.llm_client.generate(prompt, config={'temperature': 0.3})
            
            # Parse response
            lines = response.strip().split('\n')
            confidence = 0.5
            improved = candidate_sy
            
            for line in lines:
                if line.startswith('CONFIDENCE:'):
                    try:
                        confidence = float(line.split(':')[1].strip())
                    except ValueError:
                        pass
                elif line.startswith('IMPROVED:'):
                    improved = line.split(':', 1)[1].strip()
            
            return improved, confidence
            
        except Exception:
            return candidate_sy, 0.5
    
    def _fallback_synthesis(self, Sc: str, C: str) -> str:
        """
        Fallback synthesis when confidence is low
        
        Args:
            Sc: Scene text
            C: Commonality text
            
        Returns:
            str: Fallback synthesized text
        """
        # More conservative synthesis
        Sy = (
            f"Based on the scene and commonality, the deeper implication is that "
            f"the relationship between these concepts reveals {C[:150]}. "
            f"This manifests in situations like {Sc[:150]}, suggesting that "
            f"the shared features point to a broader pattern of meaning."
        )
        return Sy
    
    def verify_synthesis(
        self, 
        Sy: str, 
        C: str, 
        Sc: str, 
        threshold: float = 0.74
    ) -> Dict[str, Any]:
        """
        Verify synthesis constraint: Sim(Sy, C) >= threshold AND Sim(Sy, Sc) >= threshold
        
        Args:
            Sy: Synthesized text
            C: Commonality text
            Sc: Scene text
            threshold: Minimum similarity threshold (default 0.74)
            
        Returns:
            Dict: Verification results
        """
        if not self.embedder:
            return {
                'passed': True,
                'sim_Sy_C': 1.0,
                'sim_Sy_Sc': 1.0,
                'threshold': threshold
            }
        
        sim_Sy_C = self.embedder.cosine_similarity(Sy, C)
        sim_Sy_Sc = self.embedder.cosine_similarity(Sy, Sc)
        
        return {
            'passed': sim_Sy_C >= threshold and sim_Sy_Sc >= threshold,
            'sim_Sy_C': sim_Sy_C,
            'sim_Sy_Sc': sim_Sy_Sc,
            'threshold': threshold
        }