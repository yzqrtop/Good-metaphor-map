"""
RQ4 — Hard & Frequency Analysis

Protocol (paper, Table 6): stratify RQ1 results by model confidence and by
essence frequency, and compare the hard subset against the full test set.
Reports hard-subset accuracy, the overall↔hard accuracy gap, and the
high- vs low-frequency accuracy gain.
"""

from typing import Any, Dict, List, Optional

from core.metrics import accuracy, mean
from core.pipeline import GoodMPipeline
from datasets.ctrc_dataset import CTRCDataset
from embeddings.jina_embedder import JinaEmbedder
from experiments.common import parse_essence_vehicle, select_candidate
from llm_wrappers import BaseLLM


class RQ4HardFrequency:
    """RQ4: hard-subset and frequency-stratified analysis."""

    def __init__(
        self,
        llm_client: Optional[BaseLLM] = None,
        embedder: Optional[JinaEmbedder] = None,
        dataset_path: Optional[str] = None,
        limit: Optional[int] = None,
    ):
        self.embedder = embedder or JinaEmbedder()
        self.pipeline = GoodMPipeline(llm_client=llm_client, embedder=self.embedder)
        self.dataset = CTRCDataset(dataset_path)
        self.limit = limit

    def run(self) -> Dict[str, Any]:
        all_entries = self.dataset.load()
        frequency = self._essence_frequency(all_entries)
        test_set = self.dataset.get_test_set()
        hard_set = self.dataset.get_hard_subset()
        if self.limit:
            test_set, hard_set = test_set[: self.limit], hard_set[: self.limit]

        overall = self._run_instances(test_set)
        hard = self._run_instances(hard_set)

        median = sorted(frequency.values())[len(frequency) // 2] if frequency else 0
        high = [r for r in overall["records"] if frequency.get(r["E"], 0) > median]
        low = [r for r in overall["records"] if frequency.get(r["E"], 0) <= median]

        return {
            "rq": "RQ4",
            "task": "Hard & Frequency Analysis",
            "dataset": self.dataset.data_path,
            "results": {
                "overall_acc": overall["accuracy"],
                "hard_acc": hard["accuracy"],
                "acc_gap": overall["accuracy"] - hard["accuracy"],
                "high_freq_acc": accuracy([r["predicted"] for r in high],
                                          [r["gold"] for r in high]),
                "low_freq_acc": accuracy([r["predicted"] for r in low],
                                         [r["gold"] for r in low]),
                "freq_gain": (
                    accuracy([r["predicted"] for r in high], [r["gold"] for r in high])
                    - accuracy([r["predicted"] for r in low], [r["gold"] for r in low])
                ),
                "mean_confidence": mean([r["confidence"] for r in overall["records"]]),
                "num_test": len(test_set),
                "num_hard": len(hard_set),
                "frequency_bins": {
                    "median": median,
                    "high_n": len(high),
                    "low_n": len(low),
                },
            },
        }

    def _essence_frequency(self, entries: List[Dict]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for entry in entries:
            try:
                E, _ = parse_essence_vehicle(entry)
            except ValueError:
                continue
            counts[E] = counts.get(E, 0) + 1
        return counts

    def _run_instances(self, instances: List[Dict]) -> Dict[str, Any]:
        predictions, golds, records = [], [], []
        for entry in instances:
            E, V = parse_essence_vehicle(entry)
            tuple_obj = self.pipeline.build(E, V)
            predicted = select_candidate(tuple_obj, entry["candidates"], self.embedder)
            confidence = self.embedder.cosine_similarity(
                tuple_obj.Sy, entry["candidates"][predicted]
            )
            predictions.append(predicted)
            golds.append(entry["answer"])
            records.append({
                "E": E,
                "predicted": predicted,
                "gold": entry["answer"],
                "confidence": confidence,
            })
        return {
            "accuracy": accuracy(predictions, golds),
            "records": records,
        }
