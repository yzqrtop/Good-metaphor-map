"""
Jina Embedding Module

Provides integration with Jina-embeddings-v3 for:
- Cosine similarity computation
- Text embedding generation
- Semantic matching
"""

from typing import List, Optional
import numpy as np


class JinaEmbedder:
    """
    Jina Embeddings-v3 Integration
    
    Provides text embedding and cosine similarity computation
    using Jina's embedding model for semantic verification.
    """
    
    def __init__(
        self,
        model_name: str = "jina-embeddings-v3",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        device: Optional[str] = None
    ):
        """
        Initialize Jina Embedder
        
        Args:
            model_name: Model name (default: jina-embeddings-v3)
            api_key: API key for Jina API
            base_url: Base URL for API
            device: Computing device
        """
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url
        self.device = device
        self._model = None
        self._initialized = False
        
    def initialize(self) -> None:
        """
        Initialize the embedding model (lazy loading)
        """
        if self._initialized:
            return
        
        try:
            # Try to import and initialize the model
            # This is a placeholder for actual Jina model loading
            self._initialized = True
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Jina embedder: {e}")
    
    def encode(self, text: str) -> np.ndarray:
        """
        Encode text to embedding vector
        
        Args:
            text: Input text
            
        Returns:
            np.ndarray: Embedding vector
        """
        self._ensure_initialized()
        
        # Placeholder: generate random embedding for demonstration
        # In production, this would call the actual Jina API/model
        embedding = np.random.randn(768)
        embedding = embedding / np.linalg.norm(embedding)  # Normalize
        return embedding
    
    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """
        Encode multiple texts to embedding vectors
        
        Args:
            texts: List of input texts
            
        Returns:
            np.ndarray: Matrix of embedding vectors (n x 768)
        """
        self._ensure_initialized()
        
        embeddings = []
        for text in texts:
            embeddings.append(self.encode(text))
        
        return np.array(embeddings)
    
    def cosine_similarity(self, text1: str, text2: str) -> float:
        """
        Compute cosine similarity between two texts
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            float: Cosine similarity score (0.0 to 1.0)
        """
        emb1 = self.encode(text1)
        emb2 = self.encode(text2)
        
        return self._cosine_similarity_from_embeddings(emb1, emb2)
    
    def batch_cosine_similarity(
        self, 
        text1: str, 
        texts: List[str]
    ) -> List[float]:
        """
        Compute cosine similarity between one text and multiple texts
        
        Args:
            text1: Reference text
            texts: List of texts to compare
            
        Returns:
            List[float]: List of cosine similarity scores
        """
        emb1 = self.encode(text1)
        embeddings = self.encode_batch(texts)
        
        similarities = []
        for i in range(len(texts)):
            sim = self._cosine_similarity_from_embeddings(emb1, embeddings[i])
            similarities.append(sim)
        
        return similarities
    
    def _cosine_similarity_from_embeddings(
        self, 
        emb1: np.ndarray, 
        emb2: np.ndarray
    ) -> float:
        """
        Compute cosine similarity between two embedding vectors
        
        Args:
            emb1: First embedding vector
            emb2: Second embedding vector
            
        Returns:
            float: Cosine similarity score
        """
        # Ensure normalized vectors
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # Compute cosine similarity
        similarity = np.dot(emb1, emb2) / (norm1 * norm2)
        
        # Clamp to [0, 1] range (absolute cosine similarity)
        return float(max(0.0, min(1.0, similarity)))
    
    def _ensure_initialized(self) -> None:
        """
        Ensure the model is initialized
        """
        if not self._initialized:
            self.initialize()
    
    def get_embedding_dimension(self) -> int:
        """
        Get the embedding dimension
        
        Returns:
            int: Embedding dimension (768 for Jina-embeddings-v3)
        """
        return 768
    
    def semantic_match(
        self, 
        query: str, 
        candidates: List[str], 
        top_k: int = 1
    ) -> List[tuple]:
        """
        Find top-k semantically matching candidates
        
        Args:
            query: Query text
            candidates: List of candidate texts
            top_k: Number of top matches to return
            
        Returns:
            List[tuple]: List of (candidate, score) tuples
        """
        similarities = self.batch_cosine_similarity(query, candidates)
        
        # Sort by similarity score
        ranked = sorted(
            zip(candidates, similarities), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        return ranked[:top_k]