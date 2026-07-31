"""
CTRC Dataset Loader

Existing benchmark with 75,296 paragraphs, 4 candidates per instance.
Used exclusively for evaluation (no re-annotation).
"""

import json
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path


class CTRCDataset:
    """
    CTRC Benchmark Dataset
    
    Contains 75,296 paragraphs with 4 candidate answers per instance.
    Used exclusively for evaluation (no re-annotation performed).
    """
    
    def __init__(self, data_path: Optional[str] = None):
        """
        Initialize CTRC Dataset loader
        
        Args:
            data_path: Path to CTRC dataset file
        """
        self.data_path = data_path
        self._data: List[Dict[str, Any]] = []
        self._loaded = False
        
    def load(self) -> List[Dict[str, Any]]:
        """
        Load CTRC dataset
        
        Returns:
            List[Dict]: Loaded dataset entries
        """
        if self._loaded:
            return self._data

        if self.data_path is None:
            raise ValueError("data_path must be provided to load CTRC dataset")

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
            raise FileNotFoundError(f"CTRC dataset file not found: {path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return data
    
    def get_test_set(self) -> List[Dict[str, Any]]:
        """
        Get CTRC test set (7,796 entries)
        
        Returns:
            List[Dict]: Test entries
        """
        if not self._loaded:
            self.load()
        
        return [entry for entry in self._data if entry.get('split') == 'test']
    
    def get_dev_set(self) -> List[Dict[str, Any]]:
        """
        Get CTRC dev set (7,500 entries)
        
        Returns:
            List[Dict]: Dev entries
        """
        if not self._loaded:
            self.load()
        
        return [entry for entry in self._data if entry.get('split') == 'dev']
    
    def get_hard_subset(self) -> List[Dict[str, Any]]:
        """
        Get hard subset (342 entries) for RQ3
        
        Returns:
            List[Dict]: Hard subset entries
        """
        if not self._loaded:
            self.load()
        
        return [entry for entry in self._data if entry.get('is_hard', False)]
    
    def get_instances_with_candidates(self) -> List[Tuple[str, List[str], int]]:
        """
        Get instances formatted for evaluation
        
        Returns:
            List[Tuple]: List of (paragraph, candidates, answer_idx) tuples
        """
        if not self._loaded:
            self.load()
        
        instances = []
        for entry in self._data:
            paragraph = entry['paragraph']
            candidates = entry['candidates']
            answer_idx = entry['answer']
            instances.append((paragraph, candidates, answer_idx))
        
        return instances
    
    def statistics(self) -> Dict[str, Any]:
        """
        Get dataset statistics
        
        Returns:
            Dict: Statistical information
        """
        if not self._loaded:
            self.load()
        
        splits = {}
        hard_count = 0
        for entry in self._data:
            split = entry.get('split', 'unknown')
            splits[split] = splits.get(split, 0) + 1
            if entry.get('is_hard', False):
                hard_count += 1
        
        return {
            'total_paragraphs': 75296,
            'candidates_per_instance': 4,
            'splits': splits,
            'hard_subset_size': hard_count,
            'usage': 'evaluation only (no re-annotation)'
        }
    
    def filter_by_confidence(self, confidence_threshold: float) -> List[Dict[str, Any]]:
        """
        Filter instances by model confidence (for RQ4)
        
        Args:
            confidence_threshold: Confidence threshold
            
        Returns:
            List[Dict]: Filtered entries
        """
        if not self._loaded:
            self.load()
        
        return [
            entry for entry in self._data 
            if entry.get('confidence', 1.0) >= confidence_threshold
        ]