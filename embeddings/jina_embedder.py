"""
Jina-embeddings-v3 client.

This is the embedding backend used for every similarity in the paper: the four
constraint checks, Φ_C's overlap verification, Φ_Sc's candidate ranking and
Φ_Sy's rule retrieval. It is an online service — ``JINA_API_KEY`` must be set
(or an ``api_key`` passed in) and the call fails loudly otherwise.
"""

import os
from typing import Dict, List, Optional

import numpy as np

JINA_ENDPOINT = "https://api.jina.ai/v1/embeddings"


class JinaEmbedder:
    """Jina-embeddings-v3 compatibility layer (768-d, text-matching task)."""

    def __init__(
        self,
        model_name: str = "jina-embeddings-v3",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        dimension: int = 768,
    ):
        self.model_name = model_name
        self.api_key = api_key or os.getenv("JINA_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "JINA_API_KEY is required: Jina-embeddings-v3 is the embedding "
                "backend for all constraint checks."
            )
        self.base_url = base_url or JINA_ENDPOINT
        self.dimension = dimension
        self._cache: Dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------
    def encode(self, text: str) -> np.ndarray:
        text = text or ""
        if text in self._cache:
            return self._cache[text]
        vector = self._encode_api(text)
        self._cache[text] = vector
        return vector

    def encode_batch(self, texts: List[str]) -> np.ndarray:
        return (
            np.vstack([self.encode(t) for t in texts])
            if texts else np.zeros((0, self.dimension))
        )

    def _encode_api(self, text: str) -> np.ndarray:
        import requests

        response = requests.post(
            self.base_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model_name,
                "input": [text],
                "task": "text-matching",
                "dimensions": self.dimension,
            },
            timeout=30,
        )
        response.raise_for_status()
        vector = np.asarray(response.json()["data"][0]["embedding"], dtype=float)
        return self._normalize(vector)

    @staticmethod
    def _normalize(vector: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm

    # ------------------------------------------------------------------
    # Similarity
    # ------------------------------------------------------------------
    def cosine_similarity(self, text1: str, text2: str) -> float:
        return self._cosine(self.encode(text1), self.encode(text2))

    def batch_cosine_similarity(self, text1: str, texts: List[str]) -> List[float]:
        emb1 = self.encode(text1)
        return [self._cosine(emb1, self.encode(t)) for t in texts]

    @staticmethod
    def _cosine(emb1: np.ndarray, emb2: np.ndarray) -> float:
        n1, n2 = np.linalg.norm(emb1), np.linalg.norm(emb2)
        if n1 == 0 or n2 == 0:
            return 0.0
        return float(np.clip(np.dot(emb1, emb2) / (n1 * n2), 0.0, 1.0))

    def semantic_match(self, query: str, candidates: List[str], top_k: int = 1):
        scores = self.batch_cosine_similarity(query, candidates)
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    def get_embedding_dimension(self) -> int:
        return self.dimension
