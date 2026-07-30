"""
RQ3: Constraint Verification

Task: Generate → verify → revise → answer
Dataset: Hard Subset (342 entries)
Metrics: Accuracy, Error Interception Rate
"""

from typing import Dict, Any, List, Optional
from ..core.schema import GoodMTuple
from ..operators import (
    EntityExpansion,
    VehicleExpansion,
    CommonalityExtraction,
    SceneGeneralization,
    Synthesis
)
from ..constraints import ConstraintVerifier
from ..embeddings import JinaEmbedder
from ..datasets import CTRCDataset
from ..models.llm_wrappers import BaseLLM


class RQ3ConstraintVerification:
    """
    RQ3: Constraint Verification
    
    Evaluates the effectiveness of constraint verification in improving 
    generation quality and answer accuracy on hard subset.
    """
    
    def __init__(
        self,
        llm_client: Optional[BaseLLM] = None,
        embedder: Optional[JinaEmbedder] = None,
        dataset_path: Optional[str] = None
    ):
        """
        Initialize RQ3 experiment
        
        Args:
            llm_client: LLM client for generation
            embedder: Jina embedder for similarity
            dataset_path: Path to CTRC dataset
        """
        self.llm_client = llm_client
        self.embedder = embedder
        
        # Initialize operators
        self.entity_expander = EntityExpansion(llm_client, embedder)
        self.vehicle_expander = VehicleExpansion(llm_client, embedder)
        self.commonality_extractor = CommonalityExtraction(embedder)
        self.scene_generator = SceneGeneralization(llm_client, embedder)
        self.synthesizer = Synthesis(llm_client, embedder)
        
        # Initialize verifier
        self.verifier = ConstraintVerifier(embedder)
        
        # Load dataset
        self.dataset = CTRCDataset(dataset_path)
        
    def run(self) -> Dict[str, Any]:
        """
        Run RQ3 experiment
        
        Returns:
            Dict: Complete evaluation results
        """
        # Load hard subset
        hard_subset = self.dataset.get_hard_subset()
        
        results = {
            'rq': 'RQ3',
            'task': 'Constraint Verification',
            'dataset': 'Hard Subset',
            'num_instances': len(hard_subset),
            'results': self._evaluate(hard_subset)
        }
        
        return results
    
    def _evaluate(self, instances: List[Dict]) -> Dict[str, Any]:
        """
        Evaluate with constraint verification
        
        Args:
            instances: List of hard instances
            
        Returns:
            Dict: Evaluation metrics
        """
        correct_verified = 0
        correct_unverified = 0
        intercepted_errors = 0
        total = len(instances)
        
        for instance in instances:
            paragraph = instance['paragraph']
            candidates = instance['candidates']
            gold_answer = instance['answer']
            
            # Generate without verification
            tuple_unverified = self._generate_seven_tuple(paragraph)
            pred_unverified = self._select_answer(tuple_unverified, candidates)
            
            # Generate with verification and revision
            tuple_verified = self._generate_and_verify(paragraph)
            pred_verified = self._select_answer(tuple_verified, candidates)
            
            # Track results
            if pred_unverified == gold_answer:
                correct_unverified += 1
            if pred_verified == gold_answer:
                correct_verified += 1
            
            # Check if verification intercepted an error
            verification_results = self.verifier.verify_all(tuple_unverified)
            if not verification_results['overall']['passed'] and pred_unverified != gold_answer:
                intercepted_errors += 1
        
        return {
            'accuracy_without_verification': correct_unverified / total if total > 0 else 0.0,
            'accuracy_with_verification': correct_verified / total if total > 0 else 0.0,
            'improvement': (correct_verified - correct_unverified) / total if total > 0 else 0.0,
            'error_interception_rate': intercepted_errors / total if total > 0 else 0.0,
            'correct_verified': correct_verified,
            'correct_unverified': correct_unverified,
            'intercepted_errors': intercepted_errors,
            'total': total
        }
    
    def _generate_seven_tuple(self, paragraph: str) -> GoodMTuple:
        """
        Generate complete 7-tuple without verification
        
        Args:
            paragraph: Input paragraph
            
        Returns:
            GoodMTuple: Generated seven-tuple
        """
        E, V = self._parse_essence_vehicle(paragraph)
        
        EE = self.entity_expander.expand(E)
        VE = self.vehicle_expander.expand(V)
        C = self.commonality_extractor.extract(EE, VE)
        Sc = self.scene_generator.generalize(EE, VE)
        Sy = self.synthesizer.synthesize(Sc, C)
        
        return GoodMTuple(E=E, V=V, EE=EE, VE=VE, C=C, Sc=Sc, Sy=Sy)
    
    def _generate_and_verify(self, paragraph: str) -> GoodMTuple:
        """
        Generate with constraint verification and revision
        
        Args:
            paragraph: Input paragraph
            
        Returns:
            GoodMTuple: Verified and revised seven-tuple
        """
        tuple_obj = self._generate_seven_tuple(paragraph)
        
        # Verify constraints
        verification_results = self.verifier.verify_all(tuple_obj)
        
        # Revise if constraints fail
        if not verification_results['overall']['passed']:
            tuple_obj = self._revise_tuple(tuple_obj, verification_results)
        
        return tuple_obj
    
    def _revise_tuple(
        self, 
        tuple_obj: GoodMTuple, 
        verification_results: Dict[str, Any]
    ) -> GoodMTuple:
        """
        Revise tuple based on constraint violations
        
        Args:
            tuple_obj: Original tuple
            verification_results: Verification results
            
        Returns:
            GoodMTuple: Revised tuple
        """
        revised = tuple_obj
        
        # Revise based on failed constraints
        if not verification_results['faithfulness']['passed']:
            # Re-expand EE and VE
            revised.EE = self.entity_expander.expand(revised.E)
            revised.VE = self.vehicle_expander.expand(revised.V)
        
        if not verification_results['commonality']['passed']:
            # Re-extract commonality
            revised.C = self.commonality_extractor.extract(revised.EE, revised.VE)
        
        if not verification_results['synthesis']['passed']:
            # Re-synthesize
            revised.Sc = self.scene_generator.generalize(revised.EE, revised.VE)
            revised.Sy = self.synthesizer.synthesize(revised.Sc, revised.C)
        
        return revised
    
    def _parse_essence_vehicle(self, paragraph: str) -> tuple:
        """Parse Essence and Vehicle from paragraph"""
        if ' - ' in paragraph:
            parts = paragraph.split(' - ', 1)
            return parts[0], parts[1]
        elif '：' in paragraph:
            parts = paragraph.split('：', 1)
            return parts[0], parts[1]
        else:
            return paragraph, paragraph
    
    def _select_answer(self, tuple_obj: GoodMTuple, candidates: List[str]) -> int:
        """Select best answer candidate"""
        if not self.embedder:
            return 0
        
        best_score = -1
        best_idx = 0
        
        for i, candidate in enumerate(candidates):
            score = self.embedder.cosine_similarity(candidate, tuple_obj.Sy)
            score += self.embedder.cosine_similarity(candidate, tuple_obj.C)
            if score > best_score:
                best_score = score
                best_idx = i
        
        return best_idx