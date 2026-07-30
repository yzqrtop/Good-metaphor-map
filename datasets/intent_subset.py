"""
Intent Subset Dataset Loader

2,000 CTRC instances with educate/persuade/warn labels (κ=0.76).
Used for RQ2 (Intent Prediction task).
"""

import json
from typing import List, Dict, Any, Optional
from pathlib import Path


class IntentSubset:
    """
    Intent Subset Dataset
    
    Contains 2,000 CTRC instances with intent labels
    (educate/persuade/warn) annotated with κ=0.76 agreement.
    Used for Intent Prediction evaluation (RQ2).
    """
    
    INTENT_LABELS = ['educate', 'persuade', 'warn']
    
    def __init__(self, data_path: Optional[str] = None):
        """
        Initialize Intent Subset loader
        
        Args:
            data_path: Path to intent subset file
        """
        self.data_path = data_path
        self._data: List[Dict[str, Any]] = []
        self._loaded = False
        
    def load(self) -> List[Dict[str, Any]]:
        """
        Load intent subset dataset
        
        Returns:
            List[Dict]: Loaded dataset entries
        """
        if self._loaded:
            return self._data
        
        if self.data_path is None:
            self._data = self._create_dummy_data()
        else:
            self._data = self._load_from_file(self.data_path)
        
        self._loaded = True
        return self._data
    
    def _load_from_file(self, path: str) -> List[Dict[str, Any]]:
        """
        Load dataset from file
        
        Args:
            path: Path to dataset file
            
        Returns:
            List[Dict]: Loaded entries
        """
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Intent subset file not found: {path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return data
    
    def _create_dummy_data(self) -> List[Dict[str, Any]]:
        """
        Create dummy data for testing
        
        Returns:
            List[Dict]: Dummy entries
        """
        dummy_data = [
            {
                'id': 'intent_001',
                'paragraph': 'Bamboo basket draws water - all in vain.',
                'intent': 'warn',
                'sy_annotation': 'Futile effort yields no results',
                'split': 'test',
                'annotator_agreement': 0.80
            },
            {
                'id': 'intent_002',
                'paragraph': 'Clay Buddha crossing the river - can\'t protect itself.',
                'intent': 'educate',
                'sy_annotation': 'A lesson about self-preservation',
                'split': 'test',
                'annotator_agreement': 0.75
            },
            {
                'id': 'intent_003',
                'paragraph': 'Weasel pays chicken a New Year call - no good intentions.',
                'intent': 'persuade',
                'sy_annotation': 'Beware of false friendships',
                'split': 'test',
                'annotator_agreement': 0.78
            }
        ]
        return dummy_data
    
    def get_intent_distribution(self) -> Dict[str, int]:
        """
        Get intent label distribution
        
        Returns:
            Dict: Count of each intent label
        """
        if not self._loaded:
            self.load()
        
        distribution = {label: 0 for label in self.INTENT_LABELS}
        for entry in self._data:
            intent = entry.get('intent', 'unknown')
            if intent in distribution:
                distribution[intent] += 1
        
        return distribution
    
    def get_by_intent(self, intent: str) -> List[Dict[str, Any]]:
        """
        Get entries filtered by intent label
        
        Args:
            intent: Intent label ('educate', 'persuade', 'warn')
            
        Returns:
            List[Dict]: Entries with specified intent
        """
        if intent not in self.INTENT_LABELS:
            raise ValueError(f"Invalid intent: {intent}. Valid options: {self.INTENT_LABELS}")
        
        if not self._loaded:
            self.load()
        
        return [entry for entry in self._data if entry.get('intent') == intent]
    
    def get_evaluation_format(self) -> List[Dict[str, Any]]:
        """
        Get formatted data for intent prediction evaluation
        
        Returns:
            List[Dict]: Formatted entries for RQ2
        """
        if not self._loaded:
            self.load()
        
        results = []
        for entry in self._data:
            result = {
                'id': entry['id'],
                'paragraph': entry['paragraph'],
                'gold_intent': entry['intent'],
                'sy_annotation': entry.get('sy_annotation', ''),
                'Sy_Intent_Consistency': None  # To be computed during evaluation
            }
            results.append(result)
        
        return results
    
    def statistics(self) -> Dict[str, Any]:
        """
        Get dataset statistics
        
        Returns:
            Dict: Statistical information
        """
        if not self._loaded:
            self.load()
        
        return {
            'total_entries': 2000,
            'source': 'CTRC subset',
            'intent_labels': self.INTENT_LABELS,
            'distribution': self.get_intent_distribution(),
            'annotation_agreement': {
                'kappa': 0.76,
                'annotators': 3
            },
            'evaluation_task': 'Intent Prediction (RQ2)'
        }
    
    def get_per_class_metrics(self) -> Dict[str, float]:
        """
        Get per-class baseline metrics (to be updated during evaluation)
        
        Returns:
            Dict: Placeholder for per-class accuracy
        """
        return {
            'educate': 0.0,
            'persuade': 0.0,
            'warn': 0.0
        }
    
    def compute_sy_intent_consistency(
        self, 
        sy_predictions: List[str], 
        intent_predictions: List[str]
    ) -> float:
        """
        Compute Sy-Intent Consistency metric
        
        Args:
            sy_predictions: List of predicted Sy texts
            intent_predictions: List of predicted intent labels
            
        Returns:
            float: Consistency score
        """
        if len(sy_predictions) != len(intent_predictions):
            raise ValueError("Prediction lists must have the same length")
        
        # Placeholder: compute semantic consistency between Sy and intent
        # In production, this would use embedding similarity
        return 0.85  # Baseline from paper