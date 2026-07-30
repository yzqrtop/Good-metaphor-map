"""
GoodM-Structured Dataset Loader

18,103 TAS entries with full seven-tuple annotations.
Constructed via generation-verification pipeline with human QA on 500 samples (κ=0.81).
"""

import json
from typing import List, Dict, Any, Optional
from pathlib import Path
from ..core.schema import GoodMTuple


class GoodMDataset:
    """
    GoodM-Structured Dataset
    
    Contains 18,103 entries with complete GoodM seven-tuple annotations.
    Constructed through generation-verification pipeline with quality assurance.
    """
    
    def __init__(self, data_path: Optional[str] = None):
        """
        Initialize GoodM Dataset loader
        
        Args:
            data_path: Path to dataset file (JSON format)
        """
        self.data_path = data_path
        self._data: List[Dict[str, Any]] = []
        self._loaded = False
        
    def load(self) -> List[Dict[str, Any]]:
        """
        Load dataset from file
        
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
        Load dataset from JSON file
        
        Args:
            path: Path to JSON file
            
        Returns:
            List[Dict]: Loaded entries
        """
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return data
    
    def _create_dummy_data(self) -> List[Dict[str, Any]]:
        """
        Create dummy data for testing
        
        Returns:
            List[Dict]: Dummy dataset entries
        """
        dummy_data = [
            {
                'id': 'goodm_001',
                'E': 'bamboo basket',
                'V': 'all in vain',
                'EE': 'A basket woven from bamboo, with loose structure that cannot hold water',
                'VE': 'Completely empty, achieving nothing',
                'C': 'Unable to retain or preserve',
                'Sc': 'A situation where effort yields no results',
                'Sy': 'Futile effort, wasted labor',
                'split': 'train',
                'source': 'goodm_structured'
            },
            {
                'id': 'goodm_002',
                'E': 'clay Buddha',
                'V': 'cannot cross the river',
                'EE': 'A Buddha statue made of clay, which dissolves when encountering water',
                'VE': 'Unable to protect oneself, let alone others',
                'C': 'Fragile, incapable of self-preservation',
                'Sc': 'Being in danger and unable to help others',
                'Sy': 'Too preoccupied with self-preservation to help others',
                'split': 'train',
                'source': 'goodm_structured'
            }
        ]
        return dummy_data
    
    def to_tuples(self) -> List[GoodMTuple]:
        """
        Convert dataset entries to GoodMTuple objects
        
        Returns:
            List[GoodMTuple]: List of GoodMTuple objects
        """
        if not self._loaded:
            self.load()
        
        tuples = []
        for entry in self._data:
            tuple_obj = GoodMTuple.from_dict(entry)
            tuples.append(tuple_obj)
        
        return tuples
    
    def get_train_split(self) -> List[Dict[str, Any]]:
        """
        Get training split of dataset
        
        Returns:
            List[Dict]: Training entries
        """
        if not self._loaded:
            self.load()
        
        return [entry for entry in self._data if entry.get('split') == 'train']
    
    def get_dev_split(self) -> List[Dict[str, Any]]:
        """
        Get development split of dataset
        
        Returns:
            List[Dict]: Development entries
        """
        if not self._loaded:
            self.load()
        
        return [entry for entry in self._data if entry.get('split') == 'dev']
    
    def get_test_split(self) -> List[Dict[str, Any]]:
        """
        Get test split of dataset
        
        Returns:
            List[Dict]: Test entries
        """
        if not self._loaded:
            self.load()
        
        return [entry for entry in self._data if entry.get('split') == 'test']
    
    def statistics(self) -> Dict[str, Any]:
        """
        Get dataset statistics
        
        Returns:
            Dict: Statistical information
        """
        if not self._loaded:
            self.load()
        
        splits = {}
        for entry in self._data:
            split = entry.get('split', 'unknown')
            splits[split] = splits.get(split, 0) + 1
        
        return {
            'total_entries': len(self._data),
            'splits': splits,
            'has_annotations': True,
            'annotation_quality': {
                'human_QA_samples': 500,
                'kappa_score': 0.81
            }
        }
    
    def save(self, path: str) -> None:
        """
        Save dataset to file
        
        Args:
            path: Output file path
        """
        if not self._loaded:
            raise RuntimeError("No data to save. Load or create data first.")
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)