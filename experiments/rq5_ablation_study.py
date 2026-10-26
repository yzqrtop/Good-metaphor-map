"""
RQ5 — Ablation Study

Protocol (paper, Table 6): on the 7,500-entry CTRC development set, selectively
remove one generated component (C, Sc or Sy) and re-run answer selection.
Reports the accuracy drop (ΔAcc) and the constraint pass rate for each condition.
"""

from typing import Any, Dict, List, Optional

from constraints.verifier import ConstraintVerifier
from core.metrics import accuracy
from core.pipeline import GoodMPipeline
from datasets.ctrc_dataset import CTRCDataset
from embeddings.jina_embedder import JinaEmbedder
from experiments.common import parse_essence_vehicle, select_candidate
from llm_wrappers import BaseLLM

CONDITIONS = ("full", "w_o_C", "w_o_Sc", "w_o_Sy")


class RQ5AblationStudy:
    """RQ5: component ablation on the development set."""

    def __init__(
        self,
        llm_client: Optional[BaseLLM] = None,
        embedder: Optional[JinaEmbedder] = None,
        dataset_path: Optional[str] = None,
        limit: Optional[int] = None,
    ):
        self.embedder = embedder or JinaEmbedder()
        self.pipeline = GoodMPipeline(llm_client=llm_client, embedder=self.embedder)
        self.verifier = ConstraintVerifier(self.embedder)
        self.dataset = CTRCDataset(dataset_path)
        self.limit = limit

    def run(self) -> Dict[str, Any]:
        dev = self.dataset.get_dev_set()
        if self.limit:
            dev = dev[: self.limit]
        results = {condition: self._evaluate(dev, condition) for condition in CONDITIONS}
        full_acc = results["full"]["accuracy"]
        for condition, metrics in results.items():
            metrics["delta_acc"] = full_acc - metrics["accuracy"]
        return {
            "rq": "RQ5",
            "task": "Ablation",
            "dataset": self.dataset.data_path,
            "num_instances": len(dev),
            "results": results,
        }

    def _evaluate(self, instances: List[Dict], condition: str) -> Dict[str, Any]:
        predictions, golds, cpr = [], [], []
        for entry in instances:
            E, V = parse_essence_vehicle(entry)
            tuple_obj = self._build_condition(E, V, condition)
            weight_map = {
                "full": (0.4, 0.3, 0.2, 0.1),
                "w_o_C": (0.5, 0.0, 0.3, 0.2),
                "w_o_Sc": (0.4, 0.3, 0.2, 0.1),
                "w_o_Sy": (0.0, 0.5, 0.3, 0.2),
            }[condition]
            predictions.append(
                select_candidate(tuple_obj, entry["candidates"], self.embedder, weight_map)
            )
            golds.append(entry["answer"])
            if condition == "full":
                cpr.append(self.verifier.verify_all(tuple_obj)["overall"]["pass_rate"])
        return {
            "accuracy": accuracy(predictions, golds),
            "constraint_pass_rate": (sum(cpr) / len(cpr)) if cpr else None,
            "total": len(instances),
        }

    def _build_condition(self, E: str, V: str, condition: str):
        from core.schema import GoodMTuple

        EE = self.pipeline.phi_ee.expand(E)
        VE = self.pipeline.phi_ve.expand(V)
        C = self.pipeline.phi_c.extract(EE, VE)
        Sc = self.pipeline.phi_sc.generalize(EE, VE, C)

        # The ablated component is passed to Φ_Sy as an empty string, so its
        # contribution to Sy generation and to answer scoring drops out.
        if condition == "w_o_C":
            C = ""
            Sy = self.pipeline.phi_sy.synthesize(Sc, C)
        elif condition == "w_o_Sc":
            Sc = ""
            Sy = self.pipeline.phi_sy.synthesize(Sc, C)
        elif condition == "w_o_Sy":
            Sy = ""
        else:
            Sy = self.pipeline.phi_sy.synthesize(Sc, C)

        return GoodMTuple(E=E, V=V, EE=EE, VE=VE, C=C, Sc=Sc, Sy=Sy)
