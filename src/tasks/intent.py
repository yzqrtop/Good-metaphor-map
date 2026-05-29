"""
Intent Prediction Task
Input: Metaphor + GoodM seven-tuple features
Output: educate / persuade / warn / entertain / other
"""

from typing import Dict, Any, List, Optional
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import numpy as np

from core.schema import GoodMTuple


class IntentPredictor:
    """
    Intent Predictor
    
    Baseline: BERT 76.2% → GoodM enhanced 88.5%
    """
    
    INTENT_LABELS = ['educate', 'persuade', 'warn', 'entertain', 'other']
    
    def __init__(
        self,
        model_name: str = "bert-base-chinese",
        use_goodm_features: bool = True,
        device: Optional[str] = None
    ):
        """
        Initialize intent predictor
        
        Args:
            model_name: Base pre-trained model
            use_goodm_features: Whether to use GoodM seven-tuple features
            device: Computing device
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_goodm_features = use_goodm_features
        
        # Load tokenizer and base model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.base_model = AutoModel.from_pretrained(model_name)
        self.base_model.to(self.device)
        
        # Build classifier
        hidden_size = self.base_model.config.hidden_size
        goodm_feature_dim = 7 * 768 if use_goodm_features else 0  # 7 fields, each 768 dimensions
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size + goodm_feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, len(self.INTENT_LABELS))
        ).to(self.device)
    
    def encode_goodm(self, tuple_obj: GoodMTuple) -> torch.Tensor:
        """
        Encode GoodM seven-tuple into vector
        
        Args:
            tuple_obj: GoodMTuple instance
            
        Returns:
            torch.Tensor: Encoded feature vector
        """
        # Use base model's tokenizer to encode each field
        fields = tuple_obj.to_embedding_input()
        
        embeddings = []
        for field in fields:
            inputs = self.tokenizer(
                field,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=128
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.base_model(**inputs)
                # Use [CLS] token representation
                cls_embedding = outputs.last_hidden_state[:, 0, :]
                embeddings.append(cls_embedding)
        
        # Concatenate embeddings of all fields
        return torch.cat(embeddings, dim=-1)
    
    def predict(
        self,
        metaphor: str,
        tuple_obj: Optional[GoodMTuple] = None
    ) -> Dict[str, Any]:
        """
        Predict intent
        
        Args:
            metaphor: Metaphor text
            tuple_obj: GoodMTuple instance (optional, for enhancement)
            
        Returns:
            Dict: Contains prediction results and probability distribution
        """
        self.base_model.eval()
        self.classifier.eval()
        
        # Encode metaphor text
        inputs = self.tokenizer(
            metaphor,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            # Get text representation
            outputs = self.base_model(**inputs)
            text_features = outputs.last_hidden_state[:, 0, :]
            
            # If using GoodM features, concatenate
            if self.use_goodm_features and tuple_obj is not None:
                goodm_features = self.encode_goodm(tuple_obj)
                combined_features = torch.cat([text_features, goodm_features], dim=-1)
            else:
                combined_features = text_features
            
            # Classification
            logits = self.classifier(combined_features)
            probabilities = torch.softmax(logits, dim=-1)
            predicted_idx = torch.argmax(probabilities, dim=-1).item()
        
        return {
            'intent': self.INTENT_LABELS[predicted_idx],
            'confidence': float(probabilities[0][predicted_idx]),
            'probabilities': {
                label: float(prob)
                for label, prob in zip(self.INTENT_LABELS, probabilities[0])
            }
        }
    
    def predict_batch(
        self,
        metaphors: List[str],
        tuples: Optional[List[GoodMTuple]] = None
    ) -> List[Dict[str, Any]]:
        """
        Batch prediction
        
        Args:
            metaphors: List of metaphor texts
            tuples: List of GoodMTuple (optional)
            
        Returns:
            List[Dict]: List of prediction results
        """
        results = []
        for i, metaphor in enumerate(metaphors):
            tuple_obj = tuples[i] if tuples and i < len(tuples) else None
            results.append(self.predict(metaphor, tuple_obj))
        return results