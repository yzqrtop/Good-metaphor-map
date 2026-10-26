"""
Evaluation metrics used by RQ1–RQ6.

Only metrics that can be computed from the predictions themselves are
implemented here — no constant or simulated scores.
"""

from collections import Counter
from typing import Dict, List, Optional, Sequence

CJK_PUNCT = set("，。！？、；：“”‘’（）…— ")


def accuracy(predictions: Sequence, golds: Sequence) -> float:
    if not golds:
        return 0.0
    return sum(1 for p, g in zip(predictions, golds) if p == g) / len(golds)


def macro_f1(
    predictions: Sequence, golds: Sequence, labels: Optional[Sequence] = None
) -> float:
    """Macro-averaged F1 without sklearn, computed over the observed label set."""
    if not golds:
        return 0.0
    label_set = list(labels) if labels else sorted({*predictions, *golds})
    scores = []
    for label in label_set:
        tp = sum(1 for p, g in zip(predictions, golds) if p == label and g == label)
        fp = sum(1 for p, g in zip(predictions, golds) if p == label and g != label)
        fn = sum(1 for p, g in zip(predictions, golds) if p != label and g == label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )
    return sum(scores) / len(scores)


def per_class_accuracy(
    predictions: Sequence, golds: Sequence, labels: Sequence
) -> Dict[str, float]:
    result = {}
    for label in labels:
        gold_indices = [i for i, g in enumerate(golds) if g == label]
        if not gold_indices:
            result[label] = 0.0
            continue
        correct = sum(1 for i in gold_indices if predictions[i] == label)
        result[label] = correct / len(gold_indices)
    return result


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def content_chars(text: str) -> List[str]:
    return [c for c in text or "" if c not in CJK_PUNCT and not c.isspace()]


def token_f1(prediction: str, reference: str) -> float:
    """Character-level F1 for Chinese text (surface overlap, no model needed)."""
    pred_chars = content_chars(prediction)
    ref_chars = content_chars(reference)
    if not pred_chars or not ref_chars:
        return 0.0
    overlap = Counter(pred_chars) & Counter(ref_chars)
    common = sum(overlap.values())
    if common == 0:
        return 0.0
    precision = common / len(pred_chars)
    recall = common / len(ref_chars)
    return 2 * precision * recall / (precision + recall)


def embedding_similarity(prediction: str, reference: str, embedder) -> float:
    """Cosine similarity in the active embedding space (reported as SemSim)."""
    return embedder.cosine_similarity(prediction, reference)


def generation_quality(
    predictions: List[str],
    references: List[str],
    embedder,
    scorer=None,
) -> Dict[str, float]:
    """Aggregate reference-based scores for generated text."""
    if not references:
        return {"sem_sim": 0.0, "char_f1": 0.0}
    sem = [
        embedding_similarity(p, r, embedder)
        for p, r in zip(predictions, references)
        if p and r
    ]
    f1 = [
        token_f1(p, r) for p, r in zip(predictions, references) if p and r
    ]
    result = {
        "sem_sim": mean(sem),
        "char_f1": mean(f1),
        "n": len(references),
    }
    if scorer is not None:
        extra = scorer.score(predictions, references)
        extra.pop("n", None)
        result.update(extra)
    return result
