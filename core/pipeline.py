"""
GoodM end-to-end pipeline.

Composes the five operators exactly as the paper defines them:

    EE = Φ_EE(E)
    VE = Φ_VE(V)
    C  = Φ_C(EE, VE)
    Sc = Φ_Sc(EE, VE)
    Sy = Φ_Sy(Sc, C)

and optionally runs the constraint verifier between generation steps
(generate → verify → revise), which is the protocol used by RQ3.
"""

from typing import Any, Dict, Optional, Tuple

from constraints.verifier import ConstraintVerifier
from core.knowledge_base import HowNetKB
from core.schema import GoodMTuple
from embeddings.jina_embedder import JinaEmbedder
from operators.commonality_extraction import CommonalityExtraction
from operators.entity_expansion import EntityExpansion
from operators.scene_generalization import SceneGeneralization
from operators.synthesis import Synthesis
from operators.vehicle_expansion import VehicleExpansion
from llm_wrappers import BaseLLM


class GoodMPipeline:
    """Seven-tuple construction with verifiable constraints."""

    def __init__(
        self,
        llm_client: Optional[BaseLLM] = None,
        embedder: Optional[JinaEmbedder] = None,
        kb: Optional[HowNetKB] = None,
        verifier: Optional[ConstraintVerifier] = None,
    ):
        self.kb = kb or HowNetKB()
        self.embedder = embedder or JinaEmbedder()
        self.phi_ee = EntityExpansion(llm_client=llm_client, embedder=self.embedder, kb=self.kb)
        self.phi_ve = VehicleExpansion(llm_client=llm_client, embedder=self.embedder, kb=self.kb)
        self.phi_c = CommonalityExtraction(embedder=self.embedder, kb=self.kb)
        self.phi_sc = SceneGeneralization(llm_client=llm_client, embedder=self.embedder, kb=self.kb)
        self.phi_sy = Synthesis(llm_client=llm_client, embedder=self.embedder)
        self.verifier = verifier or ConstraintVerifier(self.embedder)

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    def build(self, E: str, V: str) -> GoodMTuple:
        EE = self.phi_ee.expand(E)
        VE = self.phi_ve.expand(V)
        C = self.phi_c.extract(EE, VE)
        Sc = self.phi_sc.generalize(EE, VE, C)
        Sy = self.phi_sy.synthesize(Sc, C)
        metadata = {
            "knowledge_sources": self.kb.sources(),
            "srl_backend": "ltp",
            "rule_base_size": self.phi_sy.num_rules,
            "c_verification": self.phi_c.last_verification,
            "sc_verification": self.phi_sc.last_selection,
        }
        return GoodMTuple(E=E, V=V, EE=EE, VE=VE, C=C, Sc=Sc, Sy=Sy, metadata=metadata)

    def build_verified(
        self, E: str, V: str, max_revisions: int = 1
    ) -> Tuple[GoodMTuple, Dict[str, Any]]:
        """
        Generate → verify → (revise → verify)* → return.

        Mirrors the RQ3 protocol: a tuple that violates a constraint is repaired
        once for that specific constraint before being returned.
        """
        tuple_obj = self.build(E, V)
        report = self.verifier.verify_all(tuple_obj)
        revisions = 0

        while not report["overall"]["passed"] and revisions < max_revisions:
            failed = set(report["overall"]["failed_constraints"])
            tuple_obj = self._revise(tuple_obj, failed)
            report = self.verifier.verify_all(tuple_obj)
            revisions += 1

        report["revisions"] = revisions
        report["revised"] = revisions > 0
        return tuple_obj, report

    # ------------------------------------------------------------------
    # Constraint-guided revision
    # ------------------------------------------------------------------
    def _revise(self, t: GoodMTuple, failed: set) -> GoodMTuple:
        EE, VE, C, Sc, Sy = t.EE, t.VE, t.C, t.Sc, t.Sy

        if "faithfulness" in failed:
            detail = self.verifier.verify_faithfulness(t)
            if not detail["details"]["EE_E_passed"]:
                EE = self._anchor(EE, t.E)
            if not detail["details"]["VE_V_passed"]:
                VE = self._anchor(VE, t.V)

        if "commonality" in failed:
            C = self._rebuild_commonality(EE, VE, C)

        if "synthesis" in failed:
            Sy = self._rebuild_synthesis(Sc, C, Sy)

        if "non_degeneracy" in failed:
            # E and V are inputs and must never be rewritten; the constraint
            # failure is reported as-is so it can be counted in RQ5.
            pass

        return GoodMTuple(E=t.E, V=t.V, EE=EE, VE=VE, C=C, Sc=Sc, Sy=Sy,
                          metadata=dict(t.metadata))

    @staticmethod
    def _anchor(text: str, source: str) -> str:
        """Re-anchor an explanation to its source span without adding content."""
        if not text:
            return source
        return f"{source}，{text}"

    def _rebuild_commonality(self, EE: str, VE: str, C: str) -> str:
        attributes = [a for a in C.split("：")[0].split("、") if a] if "：" in C else []
        clause = C.split("：", 1)[1].rstrip("。") if "：" in C else "两者共享同一评价取向"
        if not attributes:
            return self.phi_c.extract(EE, VE)
        return f"{'、'.join(attributes)}：{clause}。"

    def _rebuild_synthesis(self, Sc: str, C: str, Sy: str) -> str:
        candidates = self.phi_sy._rule_based_candidates(Sc, C, top_k=5)
        ranked = [
            (candidate, self.embedder.cosine_similarity(candidate, C)
             + self.embedder.cosine_similarity(candidate, Sc))
            for candidate in candidates
        ]
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked[0][0] if ranked else Sy
