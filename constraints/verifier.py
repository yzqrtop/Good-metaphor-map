"""
Constraint Verifier Module

Implements the four verifiable constraints of GoodM:
1. Faithfulness:    Sim(EE,E) >= τ_faith  AND Sim(VE,V) >= τ_faith
2. Alignment:       Sim(C,EE) >= τ_common AND Sim(C,VE) >= τ_common
3. Synthesis:       Sim(Sy,C) >= τ_synth AND Sim(Sy,Sc) >= τ_synth
4. Non-degeneracy:  Sim(E,V) < τ_nondeg

All similarities are cosine similarities between embeddings.

Thresholds are the paper's values, calibrated on 500 CTRC development entries
with Jina-embeddings-v3.
"""

from typing import Any, Dict, List, Optional

from core.schema import GoodMTuple
from embeddings.jina_embedder import JinaEmbedder


class ConstraintVerifier:
    """Verifies the four constraints for GoodM seven-tuples."""

    # Paper values (Jina-embeddings-v3, calibrated on 500 CTRC dev entries).
    DEFAULT_THRESHOLDS = {
        "faithfulness": 0.78,
        "commonality": 0.76,
        "synthesis": 0.74,
        "non_degeneracy": 0.82,
    }

    def __init__(
        self,
        embedder: JinaEmbedder,
        thresholds: Optional[Dict[str, float]] = None,
    ):
        if embedder is None:
            raise ValueError("ConstraintVerifier requires an embedder.")
        self.embedder = embedder
        if thresholds is not None:
            self.thresholds = dict(thresholds)
        else:
            self.thresholds = dict(self.DEFAULT_THRESHOLDS)

    # ------------------------------------------------------------------
    def verify_all(self, tuple_obj: GoodMTuple) -> Dict[str, Any]:
        results = {
            "faithfulness": self.verify_faithfulness(tuple_obj),
            "commonality": self.verify_commonality(tuple_obj),
            "synthesis": self.verify_synthesis(tuple_obj),
            "non_degeneracy": self.verify_non_degeneracy(tuple_obj),
        }
        constraints = ["faithfulness", "commonality", "synthesis", "non_degeneracy"]
        passed = [k for k in constraints if results[k]["passed"]]
        failed = [k for k in constraints if not results[k]["passed"]]
        results["overall"] = {
            "passed": not failed,
            "passed_constraints": passed,
            "failed_constraints": failed,
            "pass_rate": len(passed) / len(constraints),
        }
        return results

    def verify_faithfulness(self, tuple_obj: GoodMTuple) -> Dict[str, Any]:
        threshold = self.thresholds["faithfulness"]
        sim_ee = self.embedder.cosine_similarity(tuple_obj.EE, tuple_obj.E)
        sim_ve = self.embedder.cosine_similarity(tuple_obj.VE, tuple_obj.V)
        return {
            "constraint": "faithfulness",
            "passed": sim_ee >= threshold and sim_ve >= threshold,
            "sim_EE_E": sim_ee,
            "sim_VE_V": sim_ve,
            "threshold": threshold,
            "details": {
                "EE_E_passed": sim_ee >= threshold,
                "VE_V_passed": sim_ve >= threshold,
            },
        }

    def verify_commonality(self, tuple_obj: GoodMTuple) -> Dict[str, Any]:
        threshold = self.thresholds["commonality"]
        sim_ee = self.embedder.cosine_similarity(tuple_obj.C, tuple_obj.EE)
        sim_ve = self.embedder.cosine_similarity(tuple_obj.C, tuple_obj.VE)
        return {
            "constraint": "commonality",
            "passed": sim_ee >= threshold and sim_ve >= threshold,
            "sim_C_EE": sim_ee,
            "sim_C_VE": sim_ve,
            "threshold": threshold,
            "details": {
                "C_EE_passed": sim_ee >= threshold,
                "C_VE_passed": sim_ve >= threshold,
            },
        }

    def verify_synthesis(self, tuple_obj: GoodMTuple) -> Dict[str, Any]:
        threshold = self.thresholds["synthesis"]
        sim_c = self.embedder.cosine_similarity(tuple_obj.Sy, tuple_obj.C)
        sim_sc = self.embedder.cosine_similarity(tuple_obj.Sy, tuple_obj.Sc)
        return {
            "constraint": "synthesis",
            "passed": sim_c >= threshold and sim_sc >= threshold,
            "sim_Sy_C": sim_c,
            "sim_Sy_Sc": sim_sc,
            "threshold": threshold,
            "details": {
                "Sy_C_passed": sim_c >= threshold,
                "Sy_Sc_passed": sim_sc >= threshold,
            },
        }

    def verify_non_degeneracy(self, tuple_obj: GoodMTuple) -> Dict[str, Any]:
        threshold = self.thresholds["non_degeneracy"]
        sim = self.embedder.cosine_similarity(tuple_obj.E, tuple_obj.V)
        return {
            "constraint": "non_degeneracy",
            "passed": sim < threshold,
            "sim_E_V": sim,
            "threshold": threshold,
            "details": {"E_V_passed": sim < threshold},
        }

    # ------------------------------------------------------------------
    def compute_cpr(self, tuples: List[GoodMTuple]) -> float:
        """Constraint Pass Rate over a list of tuples."""
        if not tuples:
            return 0.0
        passed = sum(1 for t in tuples if self.verify_all(t)["overall"]["passed"])
        return passed / len(tuples)

    def get_threshold_info(self) -> Dict[str, float]:
        return dict(self.thresholds)

    def set_threshold(self, constraint: str, value: float) -> None:
        if constraint not in self.thresholds:
            raise ValueError(
                f"Unknown constraint: {constraint}. "
                f"Valid options: {list(self.thresholds.keys())}"
            )
        self.thresholds[constraint] = value
