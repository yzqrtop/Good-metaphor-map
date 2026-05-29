"""
MiniEvalFeature (MEF) Evaluator
Implements 7-metric weighted aggregation evaluation
"""

from typing import Dict, Any, List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer

from core.schema import GoodMTuple


class MEF:
    """
    MiniEvalFeature Evaluator
    
    Evaluation metrics:
    - f1: Cognitive Accuracy
    - f2: Feature Match
    - f3: Composite Alignment
    - f4: Conceptual Depth
    - f5: Scene Relevance
    - f6: Symbolic Resonance
    - f7: Overall Coherence
    
    Default weights: [0.20, 0.15, 0.15, 0.15, 0.15, 0.10, 0.10]
    """
    
    DEFAULT_WEIGHTS = [0.20, 0.15, 0.15, 0.15, 0.15, 0.10, 0.10]
    
    def __init__(
        self,
        embedding_model: str = "jinaai/jina-embeddings-v3",
        weights: Optional[List[float]] = None,
        similarity_threshold: float = 0.6
    ):
        """
        Initialize MEF evaluator
        
        Args:
            embedding_model: Embedding model for semantic similarity calculation
            weights: List of weights for 7 metrics, defaults to DEFAULT_WEIGHTS
            similarity_threshold: Semantic similarity threshold
        """
        self.sim = SentenceTransformer(embedding_model)
        self.weights = weights or self.DEFAULT_WEIGHTS
        self.similarity_threshold = similarity_threshold
        
        # Validate weights
        if len(self.weights) != 7:
            raise ValueError("Weights must have exactly 7 elements")
        if abs(sum(self.weights) - 1.0) > 1e-6:
            raise ValueError("Weights must sum to 1.0")
    
    def evaluate(self, tuple_obj: GoodMTuple) -> Dict[str, float]:
        """
        Evaluate a single seven-tuple
        
        Args:
            tuple_obj: GoodMTuple to be evaluated
            
        Returns:
            Dict[str, float]: Dictionary containing f1-f7 and MEF total score
        """
        # Calculate each metric
        f1 = self._f1_cognitive_accuracy(tuple_obj)
        f2 = self._f2_feature_match(tuple_obj)
        f3 = self._f3_composite_alignment(tuple_obj)
        f4 = self._f4_conceptual_depth(tuple_obj)
        f5 = self._f5_scene_relevance(tuple_obj)
        f6 = self._f6_symbolic_resonance(tuple_obj)
        f7 = self._f7_overall_coherence(tuple_obj)
        
        # Calculate weighted MEF total score
        scores = [f1, f2, f3, f4, f5, f6, f7]
        mef = sum(w * s for w, s in zip(self.weights, scores))
        
        return {
            'MEF': float(mef),
            'f1': float(f1),
            'f2': float(f2),
            'f3': float(f3),
            'f4': float(f4),
            'f5': float(f5),
            'f6': float(f6),
            'f7': float(f7)
        }
    
    def _calc_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts"""
        emb1 = self.sim.encode(text1)
        emb2 = self.sim.encode(text2)
        # Cosine similarity
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        return float(similarity)
    
    def _f1_cognitive_accuracy(self, t: GoodMTuple) -> float:
        """
        f1: Cognitive Accuracy
        Check if the semantic similarity between EE and E, and between VE and V exceeds the threshold
        """
        sim_ee_e = self._calc_similarity(t.EE, t.E)
        sim_ve_v = self._calc_similarity(t.VE, t.V)
        
        # If both similarities exceed the threshold, score 1, otherwise score proportionally
        score = 0.0
        if sim_ee_e > self.similarity_threshold:
            score += 0.5
        if sim_ve_v > self.similarity_threshold:
            score += 0.5
        
        return score
    
    def _f2_feature_match(self, t: GoodMTuple) -> float:
        """
        f2: Feature Match
        Calculate semantic similarity between EE and VE
        """
        return self._calc_similarity(t.EE, t.VE)
    
    def _f3_composite_alignment(self, t: GoodMTuple) -> float:
        """
        f3: Composite Alignment
        Calculate overall alignment between Sy and the {EE, VE} set
        """
        sim_sy_ee = self._calc_similarity(t.Sy, t.EE)
        sim_sy_ve = self._calc_similarity(t.Sy, t.VE)
        return (sim_sy_ee + sim_sy_ve) / 2
    
    def _f4_conceptual_depth(self, t: GoodMTuple) -> float:
        """
        f4: Conceptual Depth
        Evaluate based on length and complexity of C
        """
        c = t.C
        # Simple evaluation based on length
        length_score = min(len(c) / 100, 1.0)  # Maximum score at 100 characters
        
        # Can add more complexity metrics, such as keyword count, etc.
        return length_score * 0.5 + 0.5  # Base score 0.5, length bonus up to 0.5
    
    def _f5_scene_relevance(self, t: GoodMTuple) -> float:
        """
        f5: Scene Relevance
        Calculate relevance between Sc and {EE, VE}
        """
        sim_sc_ee = self._calc_similarity(t.Sc, t.EE)
        sim_sc_ve = self._calc_similarity(t.Sc, t.VE)
        return (sim_sc_ee + sim_sc_ve) / 2
    
    def _f6_symbolic_resonance(self, t: GoodMTuple) -> float:
        """
        f6: Symbolic Resonance
        Calculate semantic similarity between Sy and C
        """
        return self._calc_similarity(t.Sy, t.C)
    
    def _f7_overall_coherence(self, t: GoodMTuple) -> float:
        """
        f7: Overall Coherence
        Evaluate based on field completeness and internal consistency
        """
        # Check field completeness
        validation = t.validate_with_details()
        completeness = validation['field_count'] / 7
        
        # Check internal consistency: Sy should be related to both Sc and C
        sim_sy_sc = self._calc_similarity(t.Sy, t.Sc)
        
        return (completeness + sim_sy_sc) / 2
    
    def evaluate_batch(
        self,
        tuples: List[GoodMTuple]
    ) -> List[Dict[str, float]]:
        """
        Batch evaluation
        
        Args:
            tuples: List of GoodMTuple
            
        Returns:
            List[Dict[str, float]]: List of evaluation results
        """
        return [self.evaluate(t) for t in tuples]