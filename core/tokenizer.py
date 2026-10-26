"""
Chinese tokenizer.

Segmentation uses a maximum-matching (MM) algorithm over the curated
HowNet-zh lexicon, with a character-level fallback for out-of-vocabulary
spans. Keeping the tokenizer in-repo avoids a runtime dependency on jieba/LTP.
"""

import re
from typing import Iterable, List, Set

# Split on anything that is not a CJK character or a latin letter/digit.
_SEGMENT_RE = re.compile(r"[^\u4e00-\u9fffA-Za-z0-9]+")


class ChineseTokenizer:
    """Maximum-matching tokenizer driven by the lexicon surface forms."""

    def __init__(self, lexicon: Iterable[str], max_word_len: int = 6):
        self.lexicon: Set[str] = set(lexicon)
        max_known = max((len(w) for w in self.lexicon), default=1)
        self.max_word_len = min(max_word_len, max_known)

    def tokenize(self, text: str) -> List[str]:
        """Segment text into tokens; punctuation and spaces are dropped."""
        tokens: List[str] = []
        for segment in _SEGMENT_RE.split(text or ""):
            if not segment:
                continue
            tokens.extend(self._mm(segment))
        return tokens

    def _mm(self, segment: str) -> List[str]:
        tokens: List[str] = []
        i, n = 0, len(segment)
        while i < n:
            matched = None
            for length in range(min(self.max_word_len, n - i), 1, -1):
                candidate = segment[i:i + length]
                if candidate in self.lexicon:
                    matched = candidate
                    break
            if matched is None:
                matched = segment[i]
            tokens.append(matched)
            i += len(matched)
        return tokens

    @staticmethod
    def char_ngrams(text: str, n: int = 2) -> List[str]:
        """Character n-grams over CJK spans."""
        chars = list(_SEGMENT_RE.sub("", text or ""))
        if len(chars) < n:
            return chars
        return ["".join(chars[i:i + n]) for i in range(len(chars) - n + 1)]
