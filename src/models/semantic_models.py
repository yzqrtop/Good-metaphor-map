"""
Semantic Model Wrappers
Supports BERT-base and DeepMet-S for semantic similarity tasks
"""

from typing import Optional, List, np.ndarray
import numpy as np


class SemanticModel:
    """
    Base class for semantic models
    
    Used for semantic matching and embedding generation
    """
    
    def __init__(self, model_name: str = "bert-base-chinese"):
        """
        Initialize semantic model
        
        Args:
            model_name: Model name or path
        """
        self.model_name = model_name
        self._model = None
        self._tokenizer = None
        
    def load(self) -> None:
        """Load model (to be implemented by subclasses)"""
        raise NotImplementedError
    
    def encode(self, text: str) -> np.ndarray:
        """Encode text to embedding"""
        raise NotImplementedError
    
    def cosine_similarity(self, text1: str, text2: str) -> float:
        """Compute cosine similarity"""
        emb1 = self.encode(text1)
        emb2 = self.encode(text2)
        
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(np.dot(emb1, emb2) / (norm1 * norm2))


class BERTBase(SemanticModel):
    """
    BERT-base Model for semantic similarity
    
    Used as baseline semantic model for comparison
    """
    
    def __init__(self, model_name: str = "bert-base-chinese"):
        super().__init__(model_name)
        
    def load(self) -> None:
        """Load BERT model"""
        try:
            from transformers import AutoTokenizer, AutoModel
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModel.from_pretrained(self.model_name)
            self._model.eval()
        except ImportError:
            raise ImportError("transformers library is required for BERT model")
    
    def encode(self, text: str) -> np.ndarray:
        """Encode text using BERT"""
        if self._model is None:
            self.load()
        
        import torch
        
        inputs = self._tokenizer(
            text,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        )
        
        with torch.no_grad():
            outputs = self._model(**inputs)
            # Use CLS token embedding
            embedding = outputs.last_hidden_state[:, 0, :].numpy()
        
        return embedding.flatten()


class DeepMetS(SemanticModel):
    """
    DeepMet-S Model for metaphor understanding
    
    Specialized model for metaphor processing
    """
    
    def __init__(self, model_name: str = "DeepMet-S"):
        super().__init__(model_name)
        
    def load(self) -> None:
        """Load DeepMet-S model"""
        # Placeholder: DeepMet-S would be loaded here
        pass
    
    def encode(self, text: str) -> np.ndarray:
        """Encode text using DeepMet-S"""
        # Placeholder: return random embedding for demonstration
        embedding = np.random.randn(768)
        return embedding / np.linalg.norm(embedding)


def create_model(model_type: str, **kwargs) -> SemanticModel:
    """
    Factory function to create semantic models
    
    Args:
        model_type: Type of model ('bert' or 'deepmet')
        **kwargs: Additional model parameters
        
    Returns:
        SemanticModel: Created model instance
    """
    if model_type == 'bert':
        return BERTBase(**kwargs)
    elif model_type == 'deepmet':
        return DeepMetS(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Valid options: 'bert', 'deepmet'")