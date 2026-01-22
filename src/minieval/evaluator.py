from typing import Dict, Any
from sentence_transformers import SentenceTransformer
from bert_score import BERTScore


class MiniEval:
    def __init__(self):
        self.sim = SentenceTransformer("jinaai/jina-embeddings-v3")
        self.bertscorer = BERTScore(lang="zh")

    def f1_cognitive_acc(self, ee: str, e: str) -> float:
        """Calculate cognitive accuracy score"""
        # In actual projects, should use more complex calculation method
        # Currently using simple similarity calculation
        emb_ee = self.sim.encode(ee)
        emb_e = self.sim.encode(e)
        score = float(self.sim.similarity([emb_ee], [emb_e])[0][0])
        return score

    def f2_feature_match(self, ee: str, ve: str) -> float:
        """Calculate feature matching score"""
        # In actual projects, should use more complex calculation method
        # Currently using simple similarity calculation
        emb_ee = self.sim.encode(ee)
        emb_ve = self.sim.encode(ve)
        score = float(self.sim.similarity([emb_ee], [emb_ve])[0][0])
        return score

    def f3_composite_align(self, sy: str, ee: str, ve: str) -> float:
        """Calculate composite alignment score"""
        # In actual projects, should use more complex calculation method
        # Currently using simple similarity calculation
        emb_sy = self.sim.encode(sy)
        emb_ee = self.sim.encode(ee)
        emb_ve = self.sim.encode(ve)
        
        sim_sy_ee = float(self.sim.similarity([emb_sy], [emb_ee])[0][0])
        sim_sy_ve = float(self.sim.similarity([emb_sy], [emb_ve])[0][0])
        score = (sim_sy_ee + sim_sy_ve) / 2
        return score

    def f4_conceptual_depth(self, c: str) -> float:
        """Calculate conceptual depth score"""
        # In actual projects, should use more complex calculation method
        # Currently return a default value
        return 0.7  # Example value

    def f5_scene_relevance(self, sc: str, ee: str, ve: str) -> float:
        """Calculate scene relevance score"""
        # In actual projects, should use more complex calculation method
        # Currently using simple similarity calculation
        emb_sc = self.sim.encode(sc)
        emb_ee = self.sim.encode(ee)
        emb_ve = self.sim.encode(ve)
        
        sim_sc_ee = float(self.sim.similarity([emb_sc], [emb_ee])[0][0])
        sim_sc_ve = float(self.sim.similarity([emb_sc], [emb_ve])[0][0])
        score = (sim_sc_ee + sim_sc_ve) / 2
        return score

    def f6_symbolic_resonance(self, sy: str, c: str) -> float:
        """Calculate symbolic resonance score"""
        # In actual projects, should use more complex calculation method
        # Currently using simple similarity calculation
        emb_sy = self.sim.encode(sy)
        emb_c = self.sim.encode(c)
        score = float(self.sim.similarity([emb_sy], [emb_c])[0][0])
        return score

    def f7_overall_coherence(self, record: Dict[str, Any]) -> float:
        """Calculate overall coherence score"""
        # In actual projects, should use more complex calculation method
        # Currently return a default value
        return 0.8  # Example value

    def score_one(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Score single record"""
        try:
            ee = record.get("EE", "")
            e = record.get("E", "")
            ve = record.get("VE", "")
            sy = record.get("Sy", "")
            c = record.get("C", "")
            sc = record.get("Sc", "")

            f1 = self.f1_cognitive_acc(ee, e)
            f2 = self.f2_feature_match(ee, ve)
            f3 = self.f3_composite_align(sy, ee, ve)
            f4 = self.f4_conceptual_depth(c)
            f5 = self.f5_scene_relevance(sc, ee, ve)
            f6 = self.f6_symbolic_resonance(sy, c)
            f7 = self.f7_overall_coherence(record)

            # Calculate MEF total score
            mef = (f1 + f2 + f3 + f4 + f5 + f6 + f7) / 7

            return {
                **record,
                "MEF": float(mef),
                "f1": float(f1),
                "f2": float(f2),
                "f3": float(f3),
                "f4": float(f4),
                "f5": float(f5),
                "f6": float(f6),
                "f7": float(f7)
            }
        except Exception as e:
            print(f"Error scoring: {e}")
            return {**record, "MEF": 0.0, "f1": 0.0, "f2": 0.0, "f3": 0.0, "f4": 0.0, "f5": 0.0, "f6": 0.0, "f7": 0.0}