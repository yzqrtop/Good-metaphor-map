"""
Shared helpers for the RQ scripts (cloze parsing and candidate scoring).
"""

from typing import List, Sequence, Tuple

from core.schema import GoodMTuple


def parse_essence_vehicle(entry: dict) -> Tuple[str, str]:
    """Read E and V from a CTRC-style entry."""
    if entry.get("E") and entry.get("V"):
        return entry["E"], entry["V"]

    paragraph = entry.get("paragraph", "")
    for sep in ("——", "—", "：", ":", " - "):
        if sep in paragraph:
            left, right = paragraph.split(sep, 1)
            return left.strip(), right.strip()
    raise ValueError(
        "Cannot recover E/V from entry: provide 'E'/'V' fields or a paragraph "
        "with an explicit separator."
    )


def tuple_score(tuple_obj: GoodMTuple, candidate: str, embedder,
                weights: Sequence[float] = (0.4, 0.3, 0.2, 0.1)) -> float:
    """Score a candidate answer against the generated seven-tuple."""
    parts = [tuple_obj.Sy, tuple_obj.C, tuple_obj.EE, tuple_obj.VE]
    return sum(w * embedder.cosine_similarity(candidate, p)
               for w, p in zip(weights, parts))


def select_candidate(tuple_obj: GoodMTuple, candidates: List[str], embedder,
                     weights: Sequence[float] = (0.4, 0.3, 0.2, 0.1)) -> int:
    scores = [tuple_score(tuple_obj, c, embedder, weights) for c in candidates]
    return max(range(len(scores)), key=lambda i: scores[i]) if scores else 0
