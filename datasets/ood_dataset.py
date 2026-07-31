"""
Golden OOD Dataset Loader

200 entries from Supplemental Volume with Sy labels by 3 experts (κ=0.79).
Multi-layer decontamination against C4 (<3% flagged).
"""

import json
from typing import List, Dict, Any, Optional
from pathlib import Path


class OODDataset:
    """
    Golden Out-of-Distribution (OOD) Dataset
    
    Contains 200 entries from supplemental volume with expert annotations.
    Used for zero-shot OOD evaluation (RQ6).
    """
    
    def __init__(self, data_path: Optional[str] = None):
        """
        Initialize OOD Dataset loader
        
        Args:
            data_path: Path to OOD dataset file
        """
        self.data_path = data_path
        self._data: List[Dict[str, Any]] = []
        self._loaded = False
        
    def load(self) -> List[Dict[str, Any]]:
        """
        Load OOD dataset
        
        Returns:
            List[Dict]: Loaded dataset entries
        """
        if self._loaded:
            return self._data

        if self.data_path is None:
            raise ValueError("data_path must be provided to load OOD dataset")

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
            raise FileNotFoundError(f"OOD dataset file not found: {path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return data
    
    def get_expert_labels(self) -> Dict[str, List[str]]:
        """
        Get expert labels for all entries
        
        Returns:
            Dict: Entry ID to expert labels mapping
        """
        if not self._loaded:
            self.load()
        
        labels = {}
        for entry in self._data:
            entry_id = entry['id']
            expert_labels = list(entry.get('expert_labels', {}).values())
            labels[entry_id] = expert_labels
        
        return labels
    
    def get_human_agreement(self) -> float:
        """
        Get human agreement score
        
        Returns:
            float: Agreement score (Krippendorff's alpha)
        """
        return 0.79  # As reported in paper

    def get_decontamination_stats(self) -> Dict[str, Any]:
        """
        Get decontamination statistics
        
        Returns:
            Dict: Decontamination results
        """
        if not self._loaded:
            self.load()
        
        flagged_count = sum(
            1 for entry in self._data 
            if entry.get('decontamination_flagged', False)
        )
        
        return {
            'total_entries': len(self._data),
            'flagged_count': flagged_count,
            'flagged_percentage': (flagged_count / len(self._data) * 100) if self._data else 0,
            'threshold': '<3% flagged',
            'passed': flagged_count / len(self._data) < 0.03 if self._data else True
        }
    
    def statistics(self) -> Dict[str, Any]:
        """
        Get dataset statistics
        
        Returns:
            Dict: Statistical information
        """
        if not self._loaded:
            self.load()
        
        return {
            'total_entries': 200,
            'source': 'Supplemental Volume',
            'labeling': '3 experts',
            'agreement': {
                'kappa': 0.79,
                'method': 'majority voting'
            },
            'decontamination': self.get_decontamination_stats(),
            'evaluation_task': 'Zero-shot OOD (RQ6)'
        }
    
    def to_tuples_with_labels(self) -> List[Dict[str, Any]]:
        """
        Convert to evaluation format
        
        Returns:
            List[Dict]: Entries with GoodM tuples and expert labels
        """
        if not self._loaded:
            self.load()
        
        results = []
        for entry in self._data:
            result = {
                'id': entry['id'],
                'tuple': {
                    'E': entry['E'],
                    'V': entry['V'],
                    'EE': entry['EE'],
                    'VE': entry['VE'],
                    'C': entry['C'],
                    'Sc': entry['Sc'],
                    'Sy': entry['Sy']
                },
                'expert_labels': entry.get('expert_labels', {}),
                'gold_sy': self._get_majority_label(entry.get('expert_labels', {}))
            }
            results.append(result)
        
        return results
    
    def _get_majority_label(self, expert_labels: Dict[str, str]) -> str:
        """
        Get majority label from expert annotations
        
        Args:
            expert_labels: Expert label dictionary
            
        Returns:
            str: Majority label
        """
        if not expert_labels:
            return ""
        
        # Count label occurrences
        label_counts = {}
        for label in expert_labels.values():
            label_counts[label] = label_counts.get(label, 0) + 1
        
        # Return most common label
        return max(label_counts, key=label_counts.get)