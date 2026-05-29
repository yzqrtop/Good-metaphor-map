"""
Ablation Study Module
Used to evaluate the contribution of each component in the GoodM seven-tuple
"""

from typing import List, Dict, Any, Set, Tuple
from itertools import combinations
import json
from pathlib import Path

from core.schema import GoodMTuple
from src.minieval.mef import MEF


class AblationStudy:
    """
    Ablation Study
    
    Experiment types:
    1. Leave-one-out: Iterate through 7 components, remove 1 at a time
    2. Pairwise combination: Iterate through C(7,2)=21 pairwise combinations
    3. Remove key components: Simultaneously remove {C, Sc, Sy}
    """
    
    COMPONENT_NAMES = ['E', 'V', 'EE', 'VE', 'C', 'Sc', 'Sy']
    
    def __init__(self, mef_evaluator: MEF):
        """
        Initialize ablation study
        
        Args:
            mef_evaluator: MEF evaluator instance
        """
        self.mef = mef_evaluator
    
    def leave_one_out(
        self,
        test_tuples: List[GoodMTuple]
    ) -> Dict[str, Any]:
        """
        Leave-one-out ablation experiment
        
        Remove one component at a time, evaluate performance degradation with remaining 6 components
        
        Args:
            test_tuples: List of test seven-tuples
            
        Returns:
            Dict: Ablation results for each component
        """
        # Baseline performance (complete seven-tuple)
        baseline_scores = [self.mef.evaluate(t)['MEF'] for t in test_tuples]
        baseline_avg = sum(baseline_scores) / len(baseline_scores)
        
        results = {
            'baseline': baseline_avg,
            'ablations': {}
        }
        
        # Perform ablation for each component
        for component in self.COMPONENT_NAMES:
            # Create test data with that component removed
            ablated_tuples = self._remove_component(test_tuples, component)
            
            # Evaluate performance after ablation
            ablated_scores = [self.mef.evaluate(t)['MEF'] for t in ablated_tuples]
            ablated_avg = sum(ablated_scores) / len(ablated_scores)
            
            # Calculate performance drop
            performance_drop = baseline_avg - ablated_avg
            
            results['ablations'][component] = {
                'score_without': ablated_avg,
                'performance_drop': performance_drop,
                'drop_percentage': (performance_drop / baseline_avg) * 100 if baseline_avg > 0 else 0
            }
        
        return results
    
    def pairwise_combination(
        self,
        test_tuples: List[GoodMTuple]
    ) -> Dict[str, Any]:
        """
        Pairwise combination ablation experiment
        
        Iterate through C(7,2)=21 pairwise combinations, evaluate using only 2 components
        Find the minimal sufficient set
        
        Args:
            test_tuples: List of test seven-tuples
            
        Returns:
            Dict: Performance results for each combination
        """
        results = {
            'combinations': {},
            'best_combination': None,
            'best_score': 0.0
        }
        
        # Generate all pairwise combinations
        for comp1, comp2 in combinations(self.COMPONENT_NAMES, 2):
            combo_name = f"{comp1}+{comp2}"
            
            # Create test data containing only these two components
            pairwise_tuples = self._keep_only_components(
                test_tuples, {comp1, comp2}
            )
            
            # Evaluate
            scores = [self.mef.evaluate(t)['MEF'] for t in pairwise_tuples]
            avg_score = sum(scores) / len(scores)
            
            results['combinations'][combo_name] = {
                'components': [comp1, comp2],
                'score': avg_score
            }
            
            # Update best combination
            if avg_score > results['best_score']:
                results['best_score'] = avg_score
                results['best_combination'] = [comp1, comp2]
        
        return results
    
    def remove_key_components(
        self,
        test_tuples: List[GoodMTuple]
    ) -> Dict[str, Any]:
        """
        Remove key components experiment
        
        Simultaneously remove {C, Sc, Sy}, compare with complete seven-tuple performance
        
        Args:
            test_tuples: List of test seven-tuples
            
        Returns:
            Dict: Experiment results
        """
        # Baseline performance
        baseline_scores = [self.mef.evaluate(t)['MEF'] for t in test_tuples]
        baseline_avg = sum(baseline_scores) / len(baseline_scores)
        
        # Remove key components
        key_components = {'C', 'Sc', 'Sy'}
        removed_tuples = self._remove_components(test_tuples, key_components)
        
        # Evaluate
        removed_scores = [self.mef.evaluate(t)['MEF'] for t in removed_tuples]
        removed_avg = sum(removed_scores) / len(removed_scores)
        
        performance_drop = baseline_avg - removed_avg
        
        return {
            'baseline': baseline_avg,
            'without_key_components': removed_avg,
            'removed_components': list(key_components),
            'performance_drop': performance_drop,
            'drop_percentage': (performance_drop / baseline_avg) * 100 if baseline_avg > 0 else 0
        }
    
    def _remove_component(
        self,
        tuples: List[GoodMTuple],
        component: str
    ) -> List[GoodMTuple]:
        """Remove specified component"""
        result = []
        for t in tuples:
            data = t.to_dict()
            data[component] = ""  # Set to empty
            result.append(GoodMTuple.from_dict(data))
        return result
    
    def _remove_components(
        self,
        tuples: List[GoodMTuple],
        components: Set[str]
    ) -> List[GoodMTuple]:
        """Remove multiple components"""
        result = []
        for t in tuples:
            data = t.to_dict()
            for comp in components:
                data[comp] = ""  # Set to empty
            result.append(GoodMTuple.from_dict(data))
        return result
    
    def _keep_only_components(
        self,
        tuples: List[GoodMTuple],
        components: Set[str]
    ) -> List[GoodMTuple]:
        """Keep only specified components"""
        result = []
        for t in tuples:
            data = t.to_dict()
            for comp in self.COMPONENT_NAMES:
                if comp not in components:
                    data[comp] = ""  # Set other components to empty
            result.append(GoodMTuple.from_dict(data))
        return result
    
    def run_all_experiments(
        self,
        test_tuples: List[GoodMTuple],
        output_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Run all ablation experiments
        
        Args:
            test_tuples: List of test seven-tuples
            output_path: Result save path (optional)
            
        Returns:
            Dict: All experiment results
        """
        print("Running Leave-one-out experiment...")
        leave_one_out_results = self.leave_one_out(test_tuples)
        
        print("Running Pairwise combination experiment...")
        pairwise_results = self.pairwise_combination(test_tuples)
        
        print("Running Remove key components experiment...")
        key_components_results = self.remove_key_components(test_tuples)
        
        all_results = {
            'leave_one_out': leave_one_out_results,
            'pairwise_combination': pairwise_results,
            'remove_key_components': key_components_results
        }
        
        # Save results
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, ensure_ascii=False, indent=2)
            print(f"Results saved to {output_path}")
        
        return all_results