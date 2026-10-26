"""
DeepMet-S sentence encoder.

DeepMet-S is used as an alternative similarity metric for the generated text
(RQ6 Sy agreement). It is a HuggingFace encoder fine-tuned for metaphor
semantics, loaded from a local checkpoint directory — set ``DEEPMETS_PATH`` or
pass ``checkpoint``. The checkpoint is not bundled, so the loader fails loudly
when it is missing.
"""

import os
from typing import Dict, List, Optional

import numpy as np


class DeepMetS:
    """DeepMet-S encoder with mean-pooled, L2-normalized sentence vectors."""

    def __init__(
        self,
        checkpoint: Optional[str] = None,
        device: Optional[str] = None,
        max_length: int = 128,
    ):
        path = checkpoint or os.getenv("DEEPMETS_PATH")
        if not path or not os.path.isdir(path):
            raise FileNotFoundError(
                "DeepMet-S checkpoint directory not found. Pass `checkpoint=` "
                "or set DEEPMETS_PATH to the local DeepMet-S model directory."
            )
        try:
            from transformers import AutoModel, AutoTokenizer  # type: ignore
        except Exception as error:
            raise ImportError(
                "DeepMet-S requires torch + transformers. Install them with "
                "`pip install -e \".[metrics]\"`."
            ) from error

        import torch

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModel.from_pretrained(path).to(self.device).eval()
        self._cache: Dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    def encode(self, text: str) -> np.ndarray:
        text = text or ""
        if text in self._cache:
            return self._cache[text]
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=self.max_length
        ).to(self.device)
        with self._torch.no_grad():
            hidden = self.model(**inputs).last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
        vector = pooled[0].cpu().numpy().astype(float)
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        self._cache[text] = vector
        return vector

    def cosine_similarity(self, text1: str, text2: str) -> float:
        return self._cosine(self.encode(text1), self.encode(text2))

    def batch_cosine_similarity(self, text1: str, texts: List[str]) -> List[float]:
        emb1 = self.encode(text1)
        return [self._cosine(emb1, self.encode(text)) for text in texts]

    @staticmethod
    def _cosine(emb1: np.ndarray, emb2: np.ndarray) -> float:
        n1, n2 = np.linalg.norm(emb1), np.linalg.norm(emb2)
        if n1 == 0 or n2 == 0:
            return 0.0
        return float(np.clip(np.dot(emb1, emb2) / (n1 * n2), 0.0, 1.0))
