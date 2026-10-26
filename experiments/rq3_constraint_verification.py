"""
RQ3 — Constraint Verification

Protocol (paper, Table 6): generate → verify → revise → answer on the 342-entry
hard subset. Reports answer accuracy, the share of tuples that initially violate
a constraint, and the error interception rate (violations repaired by revision).
"""

from typing import Any, Dict, List, Optional

from constraints.verifier import ConstraintVerifier
from core.metrics import accuracy
from core.pipeline import GoodMPipeline
from datasets.ctrc_dataset import CTRCDataset
from embeddings.jina_embedder import JinaEmbedder
from experiments.common import parse_essence_vehicle, select_candidate
from llm_wrappers import BaseLLM


class RQ3ConstraintVerification:
    """RQ3: generate → verify → revise → answer, evaluated on hard instances."""

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
        hard = self.dataset.get_hard_subset()
        if self.limit:
            hard = hard[: self.limit]
        return {
            "rq": "RQ3",
            "task": "Constraint Verification",
            "dataset": self.dataset.data_path,
            "num_instances": len(hard),
            "results": self.evaluate(hard),
        }

    def evaluate(self, instances: List[Dict]) -> Dict[str, Any]:
        plain_pred, revised_pred, golds = [], [], []
        initial_failures = 0
        repaired = 0
        records: List[Dict[str, Any]] = []

        for entry in instances:
            E, V = parse_essence_vehicle(entry)

            # Path A: answer from the unverified tuple
            plain = self.pipeline.build(E, V)
            plain_pred.append(select_candidate(plain, entry["candidates"], self.embedder))

            # Path B: generate → verify → revise → answer
            revised, report = self.pipeline.build_verified(E, V)
            revised_pred.append(select_candidate(revised, entry["candidates"], self.embedder))

            golds.append(entry["answer"])
            # A violation exists if the first pass failed; revision success is
            # recorded separately so interception is not conflated with accuracy.
            first_pass = self.verifier.verify_all(plain)
            if not first_pass["overall"]["passed"]:
                initial_failures += 1
                if report["overall"]["passed"]:
                    repaired += 1

            records.append({
                "id": entry.get("id"),
                "plain_pred": plain_pred[-1],
                "revised_pred": revised_pred[-1],
                "gold": entry["answer"],
                "revised": report["revised"],
                "final_passed": report["overall"]["passed"],
                "failed": first_pass["overall"]["failed_constraints"],
            })

        total = len(instances)
        return {
            "accuracy": accuracy(revised_pred, golds),
            "accuracy_without_verification": accuracy(plain_pred, golds),
            "constraint_violation_rate": initial_failures / total if total else 0.0,
            "error_interception_rate": repaired / initial_failures if initial_failures else 0.0,
            "total": total,
            "per_instance": records,
        }
