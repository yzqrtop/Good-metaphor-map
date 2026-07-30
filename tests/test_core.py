"""
Tests for GoodMTuple core data structure
"""

import pytest
from core.schema import GoodMTuple


class TestGoodMTuple:
    """Test GoodMTuple class"""
    
    def test_create_empty_tuple(self):
        """Test creating an empty tuple"""
        tuple_obj = GoodMTuple()
        assert tuple_obj.E == ""
        assert tuple_obj.V == ""
        assert tuple_obj.EE == ""
        assert tuple_obj.VE == ""
        assert tuple_obj.C == ""
        assert tuple_obj.Sc == ""
        assert tuple_obj.Sy == ""
    
    def test_create_complete_tuple(self):
        """Test creating a complete tuple"""
        tuple_obj = GoodMTuple(
            E="bamboo basket",
            V="all in vain",
            EE="A basket woven from bamboo",
            VE="Completely empty",
            C="Unable to retain",
            Sc="Effort without results",
            Sy="Futile effort"
        )
        assert tuple_obj.E == "bamboo basket"
        assert tuple_obj.Sy == "Futile effort"
    
    def test_validate_complete_tuple(self):
        """Test validation of complete tuple"""
        tuple_obj = GoodMTuple(
            E="bamboo basket",
            V="all in vain",
            EE="A basket woven from bamboo",
            VE="Completely empty",
            C="Unable to retain",
            Sc="Effort without results",
            Sy="Futile effort"
        )
        assert tuple_obj.validate() is True
    
    def test_validate_incomplete_tuple(self):
        """Test validation of incomplete tuple"""
        tuple_obj = GoodMTuple(
            E="bamboo basket",
            V="all in vain"
        )
        assert tuple_obj.validate() is False
    
    def test_validate_with_details(self):
        """Test detailed validation"""
        tuple_obj = GoodMTuple(
            E="bamboo basket",
            V="all in vain",
            EE="A basket woven from bamboo",
            VE="Completely empty"
        )
        result = tuple_obj.validate_with_details()
        assert result['is_valid'] is False
        assert 'C' in result['missing_fields']
        assert 'Sc' in result['missing_fields']
        assert 'Sy' in result['missing_fields']
    
    def test_to_prefix(self):
        """Test prefix format"""
        tuple_obj = GoodMTuple(
            E="bamboo basket",
            V="all in vain",
            EE="A basket woven from bamboo",
            VE="Completely empty",
            C="Unable to retain",
            Sc="Effort without results",
            Sy="Futile effort"
        )
        prefix = tuple_obj.to_prefix()
        assert "Essence: bamboo basket" in prefix
        assert "Synthetics: Futile effort" in prefix
    
    def test_to_embedding_input(self):
        """Test embedding input format"""
        tuple_obj = GoodMTuple(
            E="bamboo basket",
            V="all in vain",
            EE="A basket woven from bamboo",
            VE="Completely empty",
            C="Unable to retain",
            Sc="Effort without results",
            Sy="Futile effort"
        )
        embeddings = tuple_obj.to_embedding_input()
        assert len(embeddings) == 7
        assert embeddings[0] == "bamboo basket"
    
    def test_to_dict(self):
        """Test dictionary conversion"""
        tuple_obj = GoodMTuple(
            E="bamboo basket",
            V="all in vain"
        )
        d = tuple_obj.to_dict()
        assert d['E'] == "bamboo basket"
        assert d['V'] == "all in vain"
    
    def test_from_dict(self):
        """Test creating from dictionary"""
        data = {
            'E': 'bamboo basket',
            'V': 'all in vain',
            'EE': 'explanation1',
            'VE': 'explanation2',
            'C': 'commonality',
            'Sc': 'scene',
            'Sy': 'synthetics'
        }
        tuple_obj = GoodMTuple.from_dict(data)
        assert tuple_obj.E == "bamboo basket"
        assert tuple_obj.Sy == "synthetics"
    
    def test_to_json(self):
        """Test JSON conversion"""
        tuple_obj = GoodMTuple(
            E="bamboo basket",
            V="all in vain"
        )
        json_str = tuple_obj.to_json()
        assert '"E": "bamboo basket"' in json_str
    
    def test_from_json(self):
        """Test creating from JSON"""
        import json
        data = {
            'E': 'bamboo basket',
            'V': 'all in vain'
        }
        json_str = json.dumps(data)
        tuple_obj = GoodMTuple.from_json(json_str)
        assert tuple_obj.E == "bamboo basket"
    
    def test_get_core_fields(self):
        """Test getting core fields"""
        tuple_obj = GoodMTuple(
            E="bamboo basket",
            V="all in vain",
            EE="explanation1",
            VE="explanation2",
            C="commonality",
            Sc="scene",
            Sy="synthetics"
        )
        core = tuple_obj.get_core_fields()
        assert len(core) == 7
        assert 'E' in core
        assert 'Sy' in core
    
    def test_equality(self):
        """Test equality comparison"""
        t1 = GoodMTuple(E="test", V="test")
        t2 = GoodMTuple(E="test", V="test")
        t3 = GoodMTuple(E="different", V="test")
        assert t1 == t2
        assert t1 != t3
    
    def test_hash(self):
        """Test hash functionality"""
        t1 = GoodMTuple(E="test", V="test")
        t2 = GoodMTuple(E="test", V="test")
        assert hash(t1) == hash(t2)
        s = {t1, t2}
        assert len(s) == 1


class TestConstraintVerifier:
    """Test ConstraintVerifier class"""
    
    def test_verifier_initialization(self):
        """Test verifier initialization"""
        from constraints import ConstraintVerifier
        verifier = ConstraintVerifier()
        assert verifier.thresholds['faithfulness'] == 0.78
        assert verifier.thresholds['commonality'] == 0.76
    
    def test_verify_all_without_embedder(self):
        """Test verification without embedder (passes by default)"""
        from constraints import ConstraintVerifier
        verifier = ConstraintVerifier()
        
        tuple_obj = GoodMTuple(
            E="bamboo basket",
            V="all in vain",
            EE="explanation1",
            VE="explanation2",
            C="commonality",
            Sc="scene",
            Sy="synthetics"
        )
        results = verifier.verify_all(tuple_obj)
        # Without embedder, all constraints should pass
        assert results['faithfulness']['passed'] is True
        assert results['overall']['passed'] is True
    
    def test_compute_cpr(self):
        """Test CPR computation"""
        from constraints import ConstraintVerifier
        verifier = ConstraintVerifier()
        
        tuples = [
            GoodMTuple(E="test1", V="test1", EE="e1", VE="v1", C="c1", Sc="s1", Sy="sy1"),
            GoodMTuple(E="test2", V="test2", EE="e2", VE="v2", C="c2", Sc="s2", Sy="sy2"),
        ]
        cpr = verifier.compute_cpr(tuples)
        # Without embedder, all should pass
        assert cpr == 1.0
    
    def test_custom_thresholds(self):
        """Test custom thresholds"""
        from constraints import ConstraintVerifier
        verifier = ConstraintVerifier(thresholds={
            'faithfulness': 0.9,
            'commonality': 0.85,
            'synthesis': 0.8,
            'non_degeneracy': 0.7
        })
        assert verifier.thresholds['faithfulness'] == 0.9
    
    def test_set_threshold(self):
        """Test setting individual threshold"""
        from constraints import ConstraintVerifier
        verifier = ConstraintVerifier()
        verifier.set_threshold('faithfulness', 0.85)
        assert verifier.thresholds['faithfulness'] == 0.85


class TestOperators:
    """Test processing operators"""
    
    def test_entity_expansion(self):
        """Test entity expansion operator"""
        from operators import EntityExpansion
        
        expander = EntityExpansion()
        result = expander.expand("bamboo basket")
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_vehicle_expansion(self):
        """Test vehicle expansion operator"""
        from operators import VehicleExpansion
        
        expander = VehicleExpansion()
        result = expander.expand("all in vain")
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_commonality_extraction(self):
        """Test commonality extraction operator"""
        from operators import CommonalityExtraction
        
        extractor = CommonalityExtraction()
        result = extractor.extract(
            "A basket woven from bamboo",
            "Completely empty, achieving nothing"
        )
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_scene_generalization(self):
        """Test scene generalization operator"""
        from operators import SceneGeneralization
        
        generator = SceneGeneralization()
        result = generator.generalize(
            "A basket woven from bamboo",
            "Completely empty, achieving nothing"
        )
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_synthesis(self):
        """Test synthesis operator"""
        from operators import Synthesis
        
        synthesizer = Synthesis()
        result = synthesizer.synthesize(
            "A situation where effort produces no results",
            "Unable to retain or preserve"
        )
        assert isinstance(result, str)
        assert len(result) > 0


class TestJinaEmbedder:
    """Test Jina Embedding integration"""
    
    def test_embedder_initialization(self):
        """Test embedder initialization"""
        from embeddings import JinaEmbedder
        
        embedder = JinaEmbedder()
        assert embedder.model_name == "jina-embeddings-v3"
    
    def test_encode(self):
        """Test encoding text"""
        from embeddings import JinaEmbedder
        import numpy as np
        
        embedder = JinaEmbedder()
        embedding = embedder.encode("test text")
        assert isinstance(embedding, np.ndarray)
        assert len(embedding) == 768
    
    def test_cosine_similarity(self):
        """Test cosine similarity computation"""
        from embeddings import JinaEmbedder
        
        embedder = JinaEmbedder()
        similarity = embedder.cosine_similarity("test", "test")
        assert 0.0 <= similarity <= 1.0
    
    def test_batch_encode(self):
        """Test batch encoding"""
        from embeddings import JinaEmbedder
        import numpy as np
        
        embedder = JinaEmbedder()
        embeddings = embedder.encode_batch(["text1", "text2"])
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape[0] == 2