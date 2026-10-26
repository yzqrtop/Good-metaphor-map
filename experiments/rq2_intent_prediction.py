"""
RQ2 — Intent Prediction

Protocol (paper, Table 6): generate Sy for each CTRC-Intent instance, then infer
the communicative intent (educate / persuade / warn) from Sy.
Metrics: per-class accuracy, overall accuracy, macro-F1, and Sy–Intent
consistency (similarity between generated Sy and the expert Sy annotation).
"""

from typing import Any, Dict, List, Optional

from core.metrics import accuracy, macro_f1, mean, per_class_accuracy
from core.pipeline import GoodMPipeline
from datasets.intent_subset import IntentSubset
from embeddings.jina_embedder import JinaEmbedder
from experiments.common import parse_essence_vehicle
from llm_wrappers import BaseLLM

# Intent prototypes used to read the intent off the generated Sy.
INTENT_PROTOTYPES = {
    "educate": "讲明一个道理，使听者明白事理、增长见识",
    "persuade": "劝说听者接受某个观点或改变行为",
    "warn": "提醒听者防备风险、警惕后果",
}


class RQ2IntentPrediction:
    """RQ2: intent prediction from generated Sy."""

    def __init__(
        self,
        llm_client: Optional[BaseLLM] = None,
        embedder: Optional[JinaEmbedder] = None,
        dataset_path: Optional[str] = None,
        limit: Optional[int] = None,
    ):
        self.embedder = embedder or JinaEmbedder()
        self.pipeline = GoodMPipeline(llm_client=llm_client, embedder=self.embedder)
        self.dataset = IntentSubset(dataset_path)
        self.limit = limit

    def run(self) -> Dict[str, Any]:
        instances = self.dataset.load()
        if self.limit:
            instances = instances[: self.limit]
        return {
            "rq": "RQ2",
            "task": "Intent Prediction",
            "dataset": self.dataset.data_path,
            "num_instances": len(instances),
            "results": self.evaluate(instances),
        }

    def evaluate(self, instances: List[Dict]) -> Dict[str, Any]:
        predictions: List[str] = []
        golds: List[str] = []
        consistencies: List[float] = []
        records: List[Dict[str, Any]] = []

        for entry in instances:
            E, V = parse_essence_vehicle(entry)
            tuple_obj = self.pipeline.build(E, V)
            predicted = self._infer_intent(tuple_obj.Sy)
            reference = entry.get("sy_annotation", "")

            predictions.append(predicted)
            golds.append(entry["intent"])
            if reference:
                consistencies.append(
                    self.embedder.cosine_similarity(tuple_obj.Sy, reference)
                )
            records.append({
                "id": entry.get("id"),
                "predicted": predicted,
                "gold": entry["intent"],
                "Sy": tuple_obj.Sy,
            })

        labels = IntentSubset.INTENT_LABELS
        return {
            "accuracy": accuracy(predictions, golds),
            "macro_f1": macro_f1(predictions, golds, labels),
            "per_class_accuracy": per_class_accuracy(predictions, golds, labels),
            "sy_intent_consistency": mean(consistencies),
            "total": len(instances),
            "per_instance": records,
        }

    def _infer_intent(self, Sy: str) -> str:
        scores = {
            label: self.embedder.cosine_similarity(Sy, prototype)
            for label, prototype in INTENT_PROTOTYPES.items()
        }
        return max(scores, key=scores.get)
