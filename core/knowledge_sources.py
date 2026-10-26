"""
Online lexical resources used by the five operators.

The paper (Appendix E) requires two external lexical resources, both fetched
over the network at run time:

* Chinese Synonym Forest (同义词词林扩展版) — attribute retrieval for Φ_EE and the
  segmentation lexicon; downloaded from a public mirror on first use.
* OpenHowNet (知网) — sememe retrieval for Φ_EE / Φ_VE / Φ_C, either through the
  official ``OpenHowNet`` package or an exported HowNet JSON endpoint
  (``HOWNET_JSON_URL``).

There is no offline copy of either resource: if a source cannot be reached, the
loader fails loudly instead of silently degrading.
"""

import os
from typing import Dict, List, Optional, Sequence

CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "cache"
)

# Public mirrors of the expanded Chinese Synonym Forest (GBK encoded).
CILIN_URLS = (
    "https://raw.githubusercontent.com/ashengtx/CilinSimilarity/master/data/cilin.txt",
    "https://cdn.jsdelivr.net/gh/ashengtx/CilinSimilarity@master/data/cilin.txt",
)
CILIN_CACHE = os.path.join(CACHE_DIR, "cilin.txt")

HOWNET_CACHE = os.path.join(CACHE_DIR, "hownet.json")


def download(url: str, dest: str, timeout=(10, 120)) -> None:
    """Download ``url`` to ``dest`` atomically, raising on failure."""
    import requests

    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = f"{dest}.part"
    try:
        with requests.get(url, stream=True, timeout=timeout) as response:
            response.raise_for_status()
            with open(tmp, "wb") as handle:
                for chunk in response.iter_content(chunk_size=1 << 16):
                    if chunk:
                        handle.write(chunk)
        os.replace(tmp, dest)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


class CilinKB:
    """
    Chinese Synonym Forest (词林扩展版) access layer.

    Each line has the form ``Aa01A01= 词 词 词``; words sharing a code are
    synonyms (code ending in ``=``), same-category words (``#``) or unrelated
    (``@``). Synonym groups drive Φ_EE attribute retrieval, and the full word
    inventory provides the maximum-matching lexicon for Φ_C / Φ_Sc.
    """

    def __init__(
        self,
        urls: Sequence[str] = CILIN_URLS,
        cache_path: str = CILIN_CACHE,
        max_synonyms: int = 8,
    ):
        self.urls = tuple(urls)
        self.cache_path = cache_path
        self.max_synonyms = max_synonyms
        self.groups: Dict[str, List[str]] = {}
        self.word_to_code: Dict[str, str] = {}
        self.lexicon: List[str] = []
        self._load()

    # ------------------------------------------------------------------
    def _load(self) -> None:
        if not os.path.exists(self.cache_path):
            env_url = os.getenv("CILIN_URL")
            urls = ((env_url,) if env_url else ()) + self.urls
            errors = []
            for url in urls:
                try:
                    download(url, self.cache_path)
                    break
                except Exception as error:  # try the next mirror
                    errors.append(f"{url}: {error}")
            else:
                raise RuntimeError(
                    "Could not download the Chinese Synonym Forest. Provide a "
                    "reachable mirror via CILIN_URL. Attempts: " + "; ".join(errors)
                )

        with open(self.cache_path, "r", encoding="gbk", errors="ignore") as handle:
            self._parse(handle)
        if not self.word_to_code:
            raise RuntimeError(
                f"Chinese Synonym Forest at {self.cache_path} contains no "
                "synonym groups; the mirror may be serving a different file."
            )

    def _parse(self, handle) -> None:
        lexicon: Dict[str, None] = {}
        for line in handle:
            line = line.rstrip("\n").strip()
            if not line:
                continue
            parts = line.split(" ", 1)
            if len(parts) != 2:
                continue
            code, payload = parts[0], parts[1]
            words = [word for word in payload.split() if word]
            for word in words:
                lexicon.setdefault(word, None)
            if not code.endswith("=") or len(words) < 2:
                continue
            self.groups[code] = words
            for word in words:
                self.word_to_code.setdefault(word, code)
        self.lexicon = sorted(lexicon)

    # ------------------------------------------------------------------
    def synonyms(self, word: str) -> List[str]:
        """Synonyms of ``word`` in the same Cilin group (excluding itself)."""
        code = self.word_to_code.get(word)
        if not code:
            return []
        group = self.groups.get(code, [])
        return [w for w in group if w != word][: self.max_synonyms]

    def expand(self, tokens: Sequence[str], top_k: int = 6) -> List[str]:
        """Near-synonyms of a token sequence, ranked by how many tokens hit."""
        ranking: Dict[str, int] = {}
        for token in tokens:
            for synonym in self.synonyms(token):
                ranking[synonym] = ranking.get(synonym, 0) + 1
        ordered = sorted(ranking, key=lambda w: (-ranking[w], len(w), w))
        excluded = set(tokens)
        return [w for w in ordered if w not in excluded][:top_k]

    def is_synonym(self, left: str, right: str) -> bool:
        code = self.word_to_code.get(left)
        return bool(code) and self.word_to_code.get(right) == code


class OpenHowNetSource:
    """
    Adapter over the official ``OpenHowNet`` package (``pip install OpenHowNet``).

    The package downloads HowNet's core data on first use. Sememe labels are
    requested in Chinese so they line up with the rest of the pipeline.
    """

    def __init__(self):
        try:
            import OpenHowNet  # type: ignore
        except Exception as error:
            raise ImportError(
                "OpenHowNet is required for sememe retrieval. Install it with "
                "`pip install OpenHowNet`, or set HOWNET_JSON_URL to a HowNet "
                "JSON endpoint."
            ) from error

        try:
            self._dict = OpenHowNet.HowNetDict()
        except Exception:
            OpenHowNet.download()
            self._dict = OpenHowNet.HowNetDict()

    def sememes(self, word: str) -> List[str]:
        try:
            senses = self._dict.get_sememes_by_word(
                word, structured=False, lang="zh", merge=False
            )
        except Exception:
            return []
        labels: List[str] = []
        for sense in senses or []:
            raw: List[str] = []
            if isinstance(sense, dict):
                value = sense.get("sememes") or sense.get("DefiningText") or ""
                raw = value if isinstance(value, list) else str(value).split(",")
            elif isinstance(sense, str):
                raw = sense.split(",")
            for item in raw:
                for label in str(item).replace("|", ",").split(","):
                    label = label.strip()
                    if label and label not in labels and self._is_chinese(label):
                        labels.append(label)
        return labels

    @staticmethod
    def _is_chinese(text: str) -> bool:
        return any("\u4e00" <= char <= "\u9fff" for char in text)


class HowNetJsonSource:
    """HowNet JSON export hosted at ``HOWNET_JSON_URL`` (downloaded each run)."""

    def __init__(self, url: Optional[str] = None):
        url = url or os.getenv("HOWNET_JSON_URL")
        if not url:
            raise RuntimeError(
                "HOWNET_JSON_URL is empty; either set it to a hosted HowNet JSON "
                "export or install the OpenHowNet package."
            )
        self.url = url
        self.data = self._fetch(url)

    @staticmethod
    def _fetch(url: str) -> Dict:
        import json

        try:
            download(url, HOWNET_CACHE)
        except Exception as error:
            raise RuntimeError(
                f"Could not fetch the HowNet JSON export from {url}."
            ) from error
        with open(HOWNET_CACHE, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def sememes(self, word: str) -> List[str]:
        entry = self.data.get("words", {}).get(word, {})
        return list(entry.get("sememes", []))

    def gloss(self, word: str) -> str:
        return self.data.get("words", {}).get(word, {}).get("gloss", "")

    def pos(self, word: str) -> str:
        return self.data.get("words", {}).get(word, {}).get("pos", "")

    def scene_hint(self, word: str) -> str:
        return self.data.get("words", {}).get(word, {}).get("scene", "")
