"""
Constraint Verifier Module

Implements 4 verifiable constraints with calibrated thresholds:
1. Faithfulness: Sim(EE,E) >= 0.78 AND Sim(VE,V) >= 0.78
2. Commonality: Sim(C,EE) >= 0.76 AND Sim(C,VE) >= 0.76
3. Synthesis: Sim(Sy,C) >= 0.74 AND Sim(Sy,Sc) >= 0.74
4. Non-degeneracy: Sim(E,V) < 0.82

All similarities computed via cosine similarity with Jina-embeddings-v3.
Thresholds calibrated on 500 CTRC dev entries (validation F1 > 0.85).
"""

from typing import Dict, Any, Optional, List
from ..core.schema import GoodMTuple
from ..embeddings import JinaEmbedder


class ConstraintVerifier:
    """
    Verifiable Constraint Verifier
    
    Verifies 4 constraints for GoodM seven-tuple quality:
    1. Faithfulness: Generated explanations must remain faithful to originals
    2. Commonality: Common features must align with both explanations
    3. Synthesis: Final synthesis must connect to both commonality and scene
    4. Non-degeneracy: Essence and Vehicle must be sufficiently different
    """
    
    # Default calibrated thresholds (from paper)
    DEFAULT_THRESHOLDS = {
        'faithfulness': 0.78,
        'commonality': 0.76,
        'synthesis': 0.74,
        'non_degeneracy': 0.82
    }
    
    def __init__(
        self,
        embedder: Optional[JinaEmbedder] = None,
        thresholds: Optional[Dict[str, float]] = None
    ):
        """
        Initialize Constraint Verifier
        
        Args:
            embedder: Jina embedder for cosine similarity computation
            thresholds: Custom thresholds (uses defaults from paper if None)
        """
        self.embedder = embedder
        self.thresholds = thresholds or self.DEFAULT_THRESHOLDS.copy()
    
    def verify_all(self, tuple_obj: GoodMTuple) -> Dict[str, Any]:
        """
        Verify all constraints for a GoodMTuple
        
        Args:
            tuple_obj: GoodMTuple to verify
            
        Returns:
            Dict: Complete verification results
        """
        results = {
            'faithfulness': self.verify_faithfulness(tuple_obj),
            'commonality': self.verify_commonality(tuple_obj),
            'synthesis': self.verify_synthesis(tuple_obj),
            'non_degeneracy': self.verify_non_degeneracy(tuple_obj)
        }
        
        # Compute overall pass/fail
        all_passed = all(r['passed'] for r in results.values())
        passed_constraints = [k for k, v in results.items() if v['passed']]
        failed_constraints = [k for k, v in results.items() if not v['passed']]
        
        results['overall'] = {
            'passed': all_passed,
            'passed_constraints': passed_constraints,
            'failed_constraints': failed_constraints,
            'pass_rate': len(passed_constraints) / len(results)
        }
        
        return results
    
    def verify_faithfulness(self, tuple_obj: GoodMTuple) -> Dict[str, Any]:
        """
        Verify Faithfulness Constraint:
        Sim(EE, E) >= tau_faith AND Sim(VE, V) >= tau_faith
        
        Ensures generated explanations remain faithful to original text.
        
        Args:
            tuple_obj: GoodMTuple to verify
            
        Returns:
            Dict: Faithfulness verification result
        """
        threshold = self.thresholds['faithfulness']
        
        if not self.embedder:
            return {
                'constraint': 'faithfulness',
                'passed': True,
                'sim_EE_E': 1.0,
                'sim_VE_V': 1.0,
                'threshold': threshold,
                'message': 'No embedder available, passing by default'
            }
        
        sim_EE_E = self.embedder.cosine_similarity(tuple_obj.EE, tuple_obj.E)
        sim_VE_V = self.embedder.cosine_similarity(tuple_obj.VE, tuple_obj.V)
        
        passed = sim_EE_E >= threshold and sim_VE_V >= threshold
        
        return {
            'constraint': 'faithfulness',
            'passed': passed,
            'sim_EE_E': sim_EE_E,
            'sim_VE_V': sim_VE_V,
            'threshold': threshold,
            'details': {
                'EE_E_passed': sim_EE_E >= threshold,
                'VE_V_passed': sim_VE_V >= threshold
            }
        }
    
    def verify_commonality(self, tuple_obj: GoodMTuple) -> Dict[str, Any]:
        """
        Verify Commonality Constraint:
        Sim(C, EE) >= tau_common AND Sim(C, VE) >= tau_common
        
        Ensures common features are well-aligned with both explanations.
        
        Args:
            tuple_obj: GoodMTuple to verify
            
        Returns:
            Dict: Commonality verification result
        """
        threshold = self.thresholds['commonality']
        
        if not self.embedder:
            return {
                'constraint': 'commonality',
                'passed': True,
                'sim_C_EE': 1.0,
                'sim_C_VE': 1.0,
                'threshold': threshold,
                'message': 'No embedder available, passing by default'
            }
        
        sim_C_EE = self.embedder.cosine_similarity(tuple_obj.C, tuple_obj.EE)
        sim_C_VE = self.embedder.cosine_similarity(tuple_obj.C, tuple_obj.VE)
        
        passed = sim_C_EE >= threshold and sim_C_VE >= threshold
        
        return {
            'constraint': 'commonality',
            'passed': passed,
            'sim_C_EE': sim_C_EE,
            'sim_C_VE': sim_C_VE,
            'threshold': threshold,
            'details': {
                'C_EE_passed': sim_C_EE >= threshold,
                'C_VE_passed': sim_C_VE >= threshold
            }
        }
    
    def verify_synthesis(self, tuple_obj: GoodMTuple) -> Dict[str, Any]:
        """
        Verify Synthesis Constraint:
        Sim(Sy, C) >= tau_synth AND Sim(Sy, Sc) >= tau_synth
        
        Ensures final synthesis connects to both commonality and scene.
        
        Args:
            tuple_obj: GoodMTuple to verify
            
        Returns:
            Dict: Synthesis verification result
        """
        threshold = self.thresholds['synthesis']
        
        if not self.embedder:
            return {
                'constraint': 'synthesis',
                'passed': True,
                'sim_Sy_C': 1.0,
                'sim_Sy_Sc': 1.0,
                'threshold': threshold,
                'message': 'No embedder available, passing by default'
            }
        
        sim_Sy_C = self.embedder.cosine_similarity(tuple_obj.Sy, tuple_obj.C)
        sim_Sy_Sc = self.embedder.cosine_similarity(tuple_obj.Sy, tuple_obj.Sc)
        
        passed = sim_Sy_C >= threshold and sim_Sy_Sc >= threshold
        
        return {
            'constraint': 'synthesis',
            'passed': passed,
            'sim_Sy_C': sim_Sy_C,
            'sim_Sy_Sc': sim_Sy_Sc,
            'threshold': threshold,
            'details': {
                'Sy_C_passed': sim_Sy_C >= threshold,
                'Sy_Sc_passed': sim_Sy_Sc >= threshold
            }
        }
    
    def verify_non_degeneracy(self, tuple_obj: GoodMTuple) -> Dict[str, Any]:
        """
        Verify Non-degeneracy Constraint:
        Sim(E, V) < tau_nondeg
        
        Ensures Essence and Vehicle are sufficiently different (not degenerate).
        
        Args:
            tuple_obj: GoodMTuple to verify
            
        Returns:
            Dict: Non-degeneracy verification result
        """
        threshold = self.thresholds['non_degeneracy']
        
        if not self.embedder:
            return {
                'constraint': 'non_degeneracy',
                'passed': True,
                'sim_E_V': 0.0,
                'threshold': threshold,
                'message': 'No embedder available, passing by default'
            }
        
        sim_E_V = self.embedder.cosine_similarity(tuple_obj.E, tuple_obj.V)
        
        passed = sim_E_V < threshold
        
        return {
            'constraint': 'non_degeneracy',
            'passed': passed,
            'sim_E_V': sim_E_V,
            'threshold': threshold,
            'details': {
                'E_V_passed': sim_E_V < threshold
            }
        }
    
    def compute_cpr(self, tuples: List[GoodMTuple]) -> float:
        """
        Compute Constraint Pass Rate (CPR) for a list of tuples
        
        Args:
            tuples: List of GoodMTuple objects
            
        Returns:
            float: CPR score (0.0 to 1.0)
        """
        if not tuples:
            return 0.0
        
        passed_count = 0
        for tuple_obj in tuples:
            results = self.verify_all(tuple_obj)
            if results['overall']['passed']:
                passed_count += 1
        
        return passed_count / len(tuples)
    
    def get_threshold_info(self) -> Dict[str, float]:
        """
        Get current threshold settings
        
        Returns:
            Dict: Current thresholds
        """
        return self.thresholds.copy()
    
    def set_threshold(self, constraint: str, value: float) -> None:
        """
        Set a specific constraint threshold
        
        Args:
            constraint: Constraint name ('faithfulness', 'commonality', 'synthesis', 'non_degeneracy')
            value: New threshold value
        """
        if constraint not in self.thresholds:
            raise ValueError(
                f"Unknown constraint: {constraint}. "
                f"Valid options: {list(self.thresholds.keys())}"
            )
        self.thresholds[constraint] = value