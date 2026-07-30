"""
RQ5: Ablation Study

Task: Selective component generation to measure contribution
Dataset: CTRC Dev (7,500 entries)
Metrics: ΔAcc Drop, CPR (Constraint Pass Rate)
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


class RQ5AblationStudy:
    """
    RQ5: Ablation Study
    
    Measures the contribution of each GoodM component by:
    1. Ablating one component at a time
    2. Measuring accuracy drop
    3. Computing CPR changes
    """
    
    COMPONENT_NAMES = ['E', 'V', 'EE', 'VE', 'C', 'Sc', 'Sy']
    
    def __init__(
        self,
        llm_client: Optional[BaseLLM] = None,
        embedder: Optional[JinaEmbedder] = None,
        dataset_path: Optional[str] = None
    ):
        """
        Initialize RQ5 experiment
        
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
        Run RQ5 experiment
        
        Returns:
            Dict: Complete evaluation results
        """
        # Load dev set
        dev_set = self.dataset.get_dev_set()
        
        results = {
            'rq': 'RQ5',
            'task': 'Ablation Study',
            'dataset': 'CTRC Dev',
            'num_instances': len(dev_set),
            'results': {
                'baseline': self._run_baseline(dev_set),
                'ablations': self._run_ablations(dev_set)
            }
        }
        
        return results
    
    def _run_baseline(self, instances: List[Dict]) -> Dict[str, Any]:
        """
        Run baseline with complete seven-tuple
        
        Args:
            instances: Dev instances
            
        Returns:
            Dict: Baseline results
        """
        correct = 0
        total_cpr = 0
        count = 0
        
        for instance in instances:
            paragraph = instance['paragraph']
            candidates = instance['candidates']
            gold_answer = instance['answer']
            
            # Generate complete tuple
            tuple_obj = self._generate_seven_tuple(paragraph)
            
            # Predict
            pred = self._select_answer(tuple_obj, candidates)
            if pred == gold_answer:
                correct += 1
            
            # Compute CPR
            verification = self.verifier.verify_all(tuple_obj)
            if verification['overall']['passed']:
                total_cpr += 1
            count += 1
        
        return {
            'accuracy': correct / count if count > 0 else 0.0,
            'cpr': total_cpr / count if count > 0 else 0.0,
            'correct': correct,
            'total': count
        }
    
    def _run_ablations(self, instances: List[Dict]) -> Dict[str, Any]:
        """
        Run ablation for each component
        
        Args:
            instances: Dev instances
            
        Returns:
            Dict: Ablation results for each component
        """
        ablation_results = {}
        
        for component in self.COMPONENT_NAMES:
            result = self._ablate_component(instances, component)
            ablation_results[component] = result
        
        return ablation_results
    
    def _ablate_component(
        self, 
        instances: List[Dict], 
        component: str
    ) -> Dict[str, Any]:
        """
        Ablate a specific component and measure impact
        
        Args:
            instances: Dev instances
            component: Component to ablate
            
        Returns:
            Dict: Ablation results
        """
        correct = 0
        total_cpr = 0
        count = 0
        
        for instance in instances:
            paragraph = instance['paragraph']
            candidates = instance['candidates']
            gold_answer = instance['answer']
            
            # Generate tuple with component ablated
            tuple_obj = self._generate_ablated_tuple(paragraph, component)
            
            # Predict
            pred = self._select_answer(tuple_obj, candidates)
            if pred == gold_answer:
                correct += 1
            
            # Compute CPR
            verification = self.verifier.verify_all(tuple_obj)
            if verification['overall']['passed']:
                total_cpr += 1
            count += 1
        
        accuracy = correct / count if count > 0 else 0.0
        cpr = total_cpr / count if count > 0 else 0.0
        
        return {
            'accuracy': accuracy,
            'cpr': cpr,
            'accuracy_drop': 0.0,  # Will be computed after baseline
            'cpr_drop': 0.0,
            'correct': correct,
            'total': count
        }
    
    def _generate_ablated_tuple(
        self, 
        paragraph: str, 
        ablate_component: str
    ) -> GoodMTuple:
        """
        Generate tuple with a specific component ablated
        
        Args:
            paragraph: Input paragraph
            ablate_component: Component to ablate
            
        Returns:
            GoodMTuple: Tuple with ablated component
        """
        E, V = self._parse_essence_vehicle(paragraph)
        
        # Generate each component, but ablate the specified one
        EE = "" if ablate_component == 'EE' else self.entity_expander.expand(E)
        VE = "" if ablate_component == 'VE' else self.vehicle_expander.expand(V)
        C = "" if ablate_component == 'C' else self.commonality_extractor.extract(EE, VE)
        Sc = "" if ablate_component == 'Sc' else self.scene_generator.generalize(EE, VE)
        Sy = "" if ablate_component == 'Sy' else self.synthesizer.synthesize(Sc, C)
        
        # If ablating E or V, set them to empty
        if ablate_component == 'E':
            E = ""
        if ablate_component == 'V':
            V = ""
        
        return GoodMTuple(E=E, V=V, EE=EE, VE=VE, C=C, Sc=Sc, Sy=Sy)
    
    def _generate_seven_tuple(self, paragraph: str) -> GoodMTuple:
        """Generate complete 7-tuple"""
        E, V = self._parse_essence_vehicle(paragraph)
        EE = self.entity_expander.expand(E)
        VE = self.vehicle_expander.expand(V)
        C = self.commonality_extractor.extract(EE, VE)
        Sc = self.scene_generator.generalize(EE, VE)
        Sy = self.synthesizer.synthesize(Sc, C)
        return GoodMTuple(E=E, V=V, EE=EE, VE=VE, C=C, Sc=Sc, Sy=Sy)
    
    def _parse_essence_vehicle(self, paragraph: str) -> tuple:
        """Parse Essence and Vehicle"""
        if ' - ' in paragraph:
            parts = paragraph.split(' - ', 1)
            return parts[0], parts[1]
        return paragraph, paragraph
    
    def _select_answer(self, tuple_obj: GoodMTuple, candidates: List[str]) -> int:
        """Select best answer"""
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