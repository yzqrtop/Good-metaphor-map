"""
RQ6 — Zero-shot Out-of-Distribution

Protocol (paper, Table 6): generate the seven-tuple for 200 unseen supplemental
entries and compare the generated Sy with the expert Sy annotation.
Metrics: semantic similarity, character-level F1, BERTScore, DeepMet-S
similarity and human Sy agreement.
"""

from typing import Any, Dict, List, Optional

from constraints.verifier import ConstraintVerifier
from core.metrics import generation_quality, mean
from core.pipeline import GoodMPipeline
from datasets.ood_dataset import OODDataset
from embeddings.bertscore import BERTScorer
from embeddings.deepmets import DeepMetS
from embeddings.jina_embedder import JinaEmbedder
from llm_wrappers import BaseLLM


class RQ6ZeroShotOOD:
    """RQ6: zero-shot generalization on the golden OOD set."""

    def __init__(
        self,
        llm_client: Optional[BaseLLM] = None,
        embedder: Optional[JinaEmbedder] = None,
        dataset_path: Optional[str] = None,
        limit: Optional[int] = None,
        bertscore: Optional[BERTScorer] = None,
        deepmets: Optional[DeepMetS] = None,
    ):
        self.embedder = embedder or JinaEmbedder()
        self.pipeline = GoodMPipeline(llm_client=llm_client, embedder=self.embedder)
        self.verifier = ConstraintVerifier(self.embedder)
        self.dataset = OODDataset(dataset_path)
        self.limit = limit
        self.bertscore = bertscore
        self.deepmets = deepmets

    def run(self) -> Dict[str, Any]:
        entries = self.dataset.load()
        if self.limit:
            entries = entries[: self.limit]
        return {
            "rq": "RQ6",
            "task": "Zero-shot OOD",
            "dataset": self.dataset.data_path,
            "num_instances": len(entries),
            "results": self.evaluate(entries),
        }

    def evaluate(self, entries: List[Dict]) -> Dict[str, Any]:
        generated, references, passed = [], [], []
        records: List[Dict[str, Any]] = []

        for entry in entries:
            tuple_obj = self.pipeline.build(entry["E"], entry["V"])
            reference = entry.get("Sy", "")
            verification = self.verifier.verify_all(tuple_obj)

            generated.append(tuple_obj.Sy)
            references.append(reference)
            passed.append(verification["overall"]["pass_rate"])
            records.append({
                "id": entry.get("id"),
                "E": entry["E"],
                "V": entry["V"],
                "Sy": tuple_obj.Sy,
                "reference_Sy": reference,
                "cpr": verification["overall"]["pass_rate"],
            })

        quality = generation_quality(
            generated, references, self.embedder, scorer=self.bertscore
        )
        result = {
            "sem_sim": quality["sem_sim"],
            "char_f1": quality["char_f1"],
            "sy_agreement": mean([
                self.embedder.cosine_similarity(g, r) if r else 0.0
                for g, r in zip(generated, references)
            ]),
            "cpr": mean(passed),
            "total": len(entries),
            "per_instance": records,
        }
        if self.bertscore:
            result["bertscore_f1"] = quality["bertscore_f1"]
        if self.deepmets:
            result["deepmets_sim"] = mean([
                self.deepmets.cosine_similarity(g, r) if r else 0.0
                for g, r in zip(generated, references)
            ])
        return result
