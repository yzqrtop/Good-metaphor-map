"""
Intent Subset Dataset Loader

2,000 CTRC instances with educate/persuade/warn labels.
Used for RQ2 (Intent Prediction task).
"""

import json
from typing import List, Dict, Any, Optional
from pathlib import Path


class IntentSubset:
    """
    Intent Subset Dataset
    
    Contains 2,000 CTRC instances with intent labels
    (educate/persuade/warn) annotated by the experts.
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
            raise ValueError("data_path must be provided to load Intent Subset dataset")

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
            'total_entries': len(self._data),
            'source': 'CTRC subset',
            'intent_labels': self.INTENT_LABELS,
            'distribution': self.get_intent_distribution(),
            'evaluation_task': 'Intent Prediction (RQ2)'
        }

    def compute_sy_intent_consistency(
        self,
        sy_predictions: List[str],
        intent_predictions: List[str],
        embedder,
    ) -> float:
        """
        Compute Sy–Intent Consistency: mean cosine similarity between each
        generated Sy and the prototype description of its predicted intent.
        """
        if len(sy_predictions) != len(intent_predictions):
            raise ValueError("Prediction lists must have the same length")
        if not sy_predictions:
            return 0.0

        from experiments.rq2_intent_prediction import INTENT_PROTOTYPES

        scores = [
            embedder.cosine_similarity(sy, INTENT_PROTOTYPES[intent])
            for sy, intent in zip(sy_predictions, intent_predictions)
            if intent in INTENT_PROTOTYPES
        ]
        return sum(scores) / len(scores) if scores else 0.0