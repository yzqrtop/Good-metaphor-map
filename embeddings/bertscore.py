"""
Reference-based BERTScore for Chinese generation quality.

Wraps the ``bert-score`` package (``pip install bert-score``) with a Chinese
encoder, so the reported numbers match the BERTScore column of the paper
instead of a character-overlap proxy.
"""

from typing import Dict, List, Optional, Sequence


class BERTScorer:
    """BERTScore wrapper; ``score`` returns mean precision / recall / F1."""

    def __init__(
        self,
        model_type: str = "bert-base-chinese",
        num_layers: Optional[int] = None,
        device: Optional[str] = None,
        rescale_with_baseline: bool = False,
    ):
        try:
            from bert_score import BERTScorer as _BERTScorer  # type: ignore
        except Exception as error:
            raise ImportError(
                "bert-score is required for the BERTScore metric. Install it "
                "with `pip install -e \".[metrics]\"`."
            ) from error

        self.scorer = _BERTScorer(
            lang="zh",
            model_type=model_type,
            num_layers=num_layers,
            device=device,
            rescale_with_baseline=rescale_with_baseline,
        )
        self.model_type = model_type

    # ------------------------------------------------------------------
    def score(
        self, predictions: Sequence[str], references: Sequence[str]
    ) -> Dict[str, float]:
        pairs = [(p, r) for p, r in zip(predictions, references) if p and r]
        if not pairs:
            return {"bertscore_precision": 0.0, "bertscore_recall": 0.0,
                    "bertscore_f1": 0.0, "n": 0}
        hypotheses = [p for p, _ in pairs]
        targets = [r for _, r in pairs]
        precision, recall, f1 = self.scorer.score(hypotheses, targets)
        return {
            "bertscore_precision": float(precision.mean()),
            "bertscore_recall": float(recall.mean()),
            "bertscore_f1": float(f1.mean()),
            "n": len(pairs),
        }

    def f1(self, prediction: str, reference: str) -> float:
        return self.score([prediction], [reference])["bertscore_f1"]
