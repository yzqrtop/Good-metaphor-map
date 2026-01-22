import pytest

from src.minieval.evaluator import MiniEval


def test_minieval():
    """Test MiniEval class"""
    evaluator = MiniEval()
    
    # Test f1_cognitive_acc method
    ee = "Detailed explanation of井底之蛙"
    e = "井底之蛙"
    f1 = evaluator.f1_cognitive_acc(ee, e)
    assert isinstance(f1, float)
    assert 0 <= f1 <= 1
    
    # Test f2_feature_match method
    ve = "Detailed explanation of related metaphor vehicle"
    f2 = evaluator.f2_feature_match(ee, ve)
    assert isinstance(f2, float)
    assert 0 <= f2 <= 1
    
    # Test f3_composite_align method
    sy = "Symbolic meaning"
    f3 = evaluator.f3_composite_align(sy, ee, ve)
    assert isinstance(f3, float)
    assert 0 <= f3 <= 1
    
    # Test f4_conceptual_depth method
    c = "Common feature"
    f4 = evaluator.f4_conceptual_depth(c)
    assert isinstance(f4, float)
    assert 0 <= f4 <= 1
    
    # Test f5_scene_relevance method
    sc = "Related scenario"
    f5 = evaluator.f5_scene_relevance(sc, ee, ve)
    assert isinstance(f5, float)
    assert 0 <= f5 <= 1
    
    # Test f6_symbolic_resonance method
    f6 = evaluator.f6_symbolic_resonance(sy, c)
    assert isinstance(f6, float)
    assert 0 <= f6 <= 1
    
    # Test f7_overall_coherence method
    record = {
        "E": e,
        "V": "Related metaphor vehicle",
        "EE": ee,
        "VE": ve,
        "C": c,
        "Sc": sc,
        "Sy": sy
    }
    f7 = evaluator.f7_overall_coherence(record)
    assert isinstance(f7, float)
    assert 0 <= f7 <= 1
    
    # Test score_one method
    scored_record = evaluator.score_one(record)
    assert isinstance(scored_record, dict)
    assert "MEF" in scored_record
    assert "f1" in scored_record
    assert "f2" in scored_record
    assert "f3" in scored_record
    assert "f4" in scored_record
    assert "f5" in scored_record
    assert "f6" in scored_record
    assert "f7" in scored_record