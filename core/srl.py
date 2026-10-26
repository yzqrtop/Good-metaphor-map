"""
Semantic role labeling for Φ_EE.

The paper (Appendix E, Operator 1) specifies the LTP toolkit for Chinese SRL.
LTP is therefore a hard requirement: ``pip install ltp``. If it is missing the
operator fails loudly rather than substituting a different analysis.
"""

from typing import Any, Dict, List, Sequence

from core.knowledge_base import HowNetKB

DEFAULT_SCENE = "日常"


class SemanticRoleLabeler:
    """Subject / action / object / scene extraction with LTP."""

    def __init__(self, kb: HowNetKB, ltp_model: str = "LTP/small"):
        self.kb = kb
        try:
            from ltp import LTP  # type: ignore
        except Exception as error:
            raise ImportError(
                "LTP is required for the SRL step of Φ_EE. Install it with "
                "`pip install ltp`."
            ) from error
        self._ltp = LTP(ltp_model)

    # ------------------------------------------------------------------
    def analyze(self, text: str) -> Dict[str, Any]:
        output = self._ltp.pipeline([text], tasks=["cws", "pos", "srl"])
        words: List[str] = list(output.cws[0])
        frames = output.srl[0]
        frame = frames[0] if isinstance(frames, list) and frames else None
        if not isinstance(frame, dict):
            raise ValueError(f"LTP returned no SRL frame for: {text}")

        arguments: Sequence[Dict] = frame.get("arguments", [])
        subject = self._span_text(words, arguments, "A0")
        obj = self._span_text(words, arguments, "A1")
        action = frame.get("predicate", "") or ""

        scene = (
            self.kb.scene_hint(obj)
            or self.kb.scene_hint(action)
            or self.kb.scene_hint(subject)
            or DEFAULT_SCENE
        )
        return {
            "tokens": words,
            "subject": subject,
            "action": action,
            "object": obj,
            "scene": scene,
        }

    @staticmethod
    def _span_text(words: Sequence[str], arguments: Sequence[Dict], role: str) -> str:
        for argument in arguments or []:
            if argument.get("role") != role:
                continue
            span = argument.get("span") or []
            if len(span) != 2:
                continue
            start, end = span
            return "".join(words[start:end])
        return ""
