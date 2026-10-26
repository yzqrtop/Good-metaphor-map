"""
Lexical knowledge access for the five operators.

HowNet supplies sememes (Φ_EE / Φ_VE / Φ_C); the Chinese Synonym Forest supplies
near-synonyms for attribute retrieval (Φ_EE) and the segmentation lexicon. Both
are online resources — there is no bundled copy. HowNet is reached through the
official ``OpenHowNet`` package, or through a hosted JSON export when
``HOWNET_JSON_URL`` is set.
"""

import os
from typing import Dict, List, Optional, Sequence, Set

from core.knowledge_sources import CilinKB, HowNetJsonSource, OpenHowNetSource
from core.tokenizer import ChineseTokenizer

STOPWORDS: Set[str] = {
    "的", "了", "和", "与", "在", "是", "把", "被", "对", "就", "都", "也",
    "而", "及", "或", "一个", "一位", "一名", "这种", "这个", "那个", "自己",
    "不断", "其", "之", "等", "将", "并", "很", "更", "还", "又", "地", "得",
}


class HowNetKB:
    """HowNet-zh sememe knowledge base plus its Cilin companion."""

    def __init__(
        self,
        cilin: Optional[CilinKB] = None,
        hownet: Optional[object] = None,
    ):
        self.hownet = hownet if hownet is not None else self._create_hownet()
        self.cilin = cilin if cilin is not None else CilinKB()
        self.stopwords: Set[str] = set(STOPWORDS)

    @staticmethod
    def _create_hownet():
        if os.getenv("HOWNET_JSON_URL"):
            return HowNetJsonSource()
        return OpenHowNetSource()

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------
    def sememes(self, word: str) -> List[str]:
        return self.hownet.sememes(word)

    def gloss(self, word: str) -> str:
        getter = getattr(self.hownet, "gloss", None)
        return getter(word) if getter else ""

    def pos(self, word: str) -> str:
        getter = getattr(self.hownet, "pos", None)
        return getter(word) if getter else ""

    def scene_hint(self, word: str) -> str:
        getter = getattr(self.hownet, "scene_hint", None)
        return getter(word) if getter else ""

    def known(self, word: str) -> bool:
        return bool(self.sememes(word)) or self.cilin.word_to_code.get(word) is not None

    # ------------------------------------------------------------------
    # Sememe / synonym expansion
    # ------------------------------------------------------------------
    def sememes_of(self, tokens: Sequence[str]) -> List[str]:
        """Union of sememes for a token sequence, de-duplicated and ordered."""
        seen: List[str] = []
        for token in tokens:
            for sememe in self.sememes(token):
                if sememe not in seen:
                    seen.append(sememe)
        return seen

    def synonyms(self, word: str) -> List[str]:
        return self.cilin.synonyms(word)

    def expand(self, tokens: Sequence[str], top_k: int = 6) -> List[str]:
        """Near-synonyms from the Chinese Synonym Forest, ranked by token hits."""
        return self.cilin.expand(tokens, top_k=top_k)

    def words_with_sememe(self, sememe: str) -> List[str]:
        words = getattr(self.hownet, "data", {}).get("words", {})
        return [w for w, e in words.items() if sememe in e.get("sememes", [])]

    def build_tokenizer(self, max_word_len: int = 6) -> ChineseTokenizer:
        return ChineseTokenizer(self.cilin.lexicon, max_word_len=max_word_len)

    # ------------------------------------------------------------------
    def sources(self) -> Dict[str, str]:
        """Which online backend actually served each knowledge layer."""
        return {
            "hownet": type(self.hownet).__name__,
            "cilin": f"online ({len(self.cilin.lexicon)} words)",
        }
