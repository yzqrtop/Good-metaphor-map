"""
RQ1 — TAS Cloze

Protocol (paper, Table 6): generate the seven-tuple from a CTRC test instance,
then select the answer whose semantics best match the generated tuple.
Metrics: Top-1 Accuracy, Macro-F1, semantic similarity to the gold explanation,
and Constraint Pass Rate.
"""

from typing import Any, Dict, List, Optional

from constraints.verifier import ConstraintVerifier
from core.metrics import accuracy, macro_f1, mean
from core.pipeline import GoodMPipeline
from datasets.ctrc_dataset import CTRCDataset
from embeddings.jina_embedder import JinaEmbedder
from experiments.common import parse_essence_vehicle, select_candidate
from llm_wrappers import BaseLLM


class RQ1TASCloze:
    """RQ1: TAS cloze evaluation."""

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
        test_set = self.dataset.get_test_set()
        if self.limit:
            test_set = test_set[: self.limit]
        return {
            "rq": "RQ1",
            "task": "TAS Cloze",
            "dataset": self.dataset.data_path,
            "num_instances": len(test_set),
            "results": self.evaluate(test_set),
        }

    def evaluate(self, instances: List[Dict]) -> Dict[str, Any]:
        predictions: List[int] = []
        golds: List[int] = []
        tuple_sims: List[float] = []
        passed = 0
        records: List[Dict[str, Any]] = []

        for entry in instances:
            E, V = parse_essence_vehicle(entry)
            tuple_obj = self.pipeline.build(E, V)
            candidates = entry["candidates"]
            gold = entry["answer"]
            predicted = select_candidate(tuple_obj, candidates, self.embedder)

            predictions.append(predicted)
            golds.append(gold)
            tuple_sims.append(
                self.embedder.cosine_similarity(tuple_obj.Sy, candidates[gold])
            )
            verification = self.verifier.verify_all(tuple_obj)
            passed += int(verification["overall"]["passed"])
            records.append({
                "id": entry.get("id"),
                "predicted": predicted,
                "gold": gold,
                "cpr": verification["overall"]["pass_rate"],
            })

        if not instances:
            return {"accuracy": 0.0, "macro_f1": 0.0, "sem_sim": 0.0,
                    "cpr": 0.0, "total": 0}

        return {
            "accuracy": accuracy(predictions, golds),
            "macro_f1": macro_f1(predictions, golds),
            "sem_sim": mean(tuple_sims),
            "cpr": passed / len(instances),
            "correct": sum(1 for p, g in zip(predictions, golds) if p == g),
            "total": len(instances),
            "per_instance": records,
        }
