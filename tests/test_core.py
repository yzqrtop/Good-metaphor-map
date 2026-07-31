"""
Tests for GoodMTuple core data structure
Uses real Chinese xielingyu (allegorical saying) data
"""

import pytest
from core.schema import GoodMTuple


class TestGoodMTuple:
    """Test GoodMTuple class with real xielingyu data"""
    
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
        """Test creating a complete tuple with real xielingyu data"""
        tuple_obj = GoodMTuple(
            E="王婆卖瓜",
            V="自买自夸",
            EE="王婆在集市上卖瓜，比喻自我吹嘘",
            VE="自己买自己的瓜，自己夸奖自己的货物",
            C="自我吹嘘、自我标榜",
            Sc="市场交易场景中的过度宣传",
            Sy="讽刺自吹自擂的行为"
        )
        assert tuple_obj.E == "王婆卖瓜"
        assert tuple_obj.Sy == "讽刺自吹自擂的行为"
    
    def test_validate_complete_tuple(self):
        """Test validation of complete tuple"""
        tuple_obj = GoodMTuple(
            E="井底之蛙",
            V="没见过大世面",
            EE="井底的青蛙，比喻见识短浅的人",
            VE="没有见过大的世面，形容见识狭隘",
            C="见识短浅、视野狭隘",
            Sc="局限于狭小空间或环境中",
            Sy="告诫人们不要固步自封"
        )
        assert tuple_obj.validate() is True
    
    def test_validate_incomplete_tuple(self):
        """Test validation of incomplete tuple"""
        tuple_obj = GoodMTuple(
            E="竹篮打水",
            V="一场空"
        )
        assert tuple_obj.validate() is False
    
    def test_validate_with_details(self):
        """Test detailed validation"""
        tuple_obj = GoodMTuple(
            E="守株待兔",
            V="妄想",
            EE="守着树桩等待兔子",
            VE="不切实际的幻想"
        )
        result = tuple_obj.validate_with_details()
        assert result['is_valid'] is False
        assert 'C' in result['missing_fields']
        assert 'Sc' in result['missing_fields']
        assert 'Sy' in result['missing_fields']
    
    def test_to_prefix(self):
        """Test prefix format"""
        tuple_obj = GoodMTuple(
            E="王婆卖瓜",
            V="自买自夸",
            EE="比喻自我吹嘘",
            VE="自己夸奖自己的货物",
            C="自我标榜",
            Sc="市场交易场景",
            Sy="讽刺自吹自擂"
        )
        prefix = tuple_obj.to_prefix()
        assert "Essence: 王婆卖瓜" in prefix
        assert "Synthetics: 讽刺自吹自擂" in prefix
    
    def test_to_embedding_input(self):
        """Test embedding input format"""
        tuple_obj = GoodMTuple(
            E="井底之蛙",
            V="没见过大世面",
            EE="井底的青蛙",
            VE="见识狭隘",
            C="见识短浅",
            Sc="局限环境",
            Sy="开阔眼界"
        )
        embeddings = tuple_obj.to_embedding_input()
        assert len(embeddings) == 7
        assert embeddings[0] == "井底之蛙"
    
    def test_to_dict(self):
        """Test dictionary conversion"""
        tuple_obj = GoodMTuple(
            E="画蛇添足",
            V="多此一举"
        )
        d = tuple_obj.to_dict()
        assert d['E'] == "画蛇添足"
        assert d['V'] == "多此一举"
    
    def test_from_dict(self):
        """Test creating from dictionary"""
        data = {
            'E': '掩耳盗铃',
            'V': '自欺欺人',
            'EE': '捂住耳朵偷铃铛',
            'VE': '欺骗自己和他人',
            'C': '自欺欺人',
            'Sc': '欺骗场景',
            'Sy': '告诫诚实'
        }
        tuple_obj = GoodMTuple.from_dict(data)
        assert tuple_obj.E == "掩耳盗铃"
        assert tuple_obj.Sy == "告诫诚实"
    
    def test_to_json(self):
        """Test JSON conversion"""
        tuple_obj = GoodMTuple(
            E="亡羊补牢",
            V="为时未晚"
        )
        json_str = tuple_obj.to_json()
        assert '"E": "亡羊补牢"' in json_str
    
    def test_from_json(self):
        """Test creating from JSON"""
        import json
        data = {
            'E': '刻舟求剑',
            'V': '拘泥不知变通'
        }
        json_str = json.dumps(data)
        tuple_obj = GoodMTuple.from_json(json_str)
        assert tuple_obj.E == "刻舟求剑"
    
    def test_get_core_fields(self):
        """Test getting core fields"""
        tuple_obj = GoodMTuple(
            E="塞翁失马",
            V="焉知非福",
            EE="边塞老人丢失了马",
            VE="怎么知道不是福气",
            C="祸福相依",
            Sc="人生际遇",
            Sy="辩证看待得失"
        )
        core = tuple_obj.get_core_fields()
        assert len(core) == 7
        assert 'E' in core
        assert 'Sy' in core
    
    def test_equality(self):
        """Test equality comparison"""
        t1 = GoodMTuple(E="对牛弹琴", V="白费口舌")
        t2 = GoodMTuple(E="对牛弹琴", V="白费口舌")
        t3 = GoodMTuple(E="缘木求鱼", V="白费力气")
        assert t1 == t2
        assert t1 != t3
    
    def test_hash(self):
        """Test hash functionality"""
        t1 = GoodMTuple(E="一石二鸟", V="一举两得")
        t2 = GoodMTuple(E="一石二鸟", V="一举两得")
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
            E="王婆卖瓜",
            V="自买自夸",
            EE="比喻自我吹嘘",
            VE="自己夸奖自己的货物",
            C="自我标榜",
            Sc="市场交易场景",
            Sy="讽刺自吹自擂"
        )
        results = verifier.verify_all(tuple_obj)
        # Without embedder, all constraints should pass
        assert results['faithfulness']['passed'] is True
        assert results['overall']['passed'] is True
    
    def test_compute_cpr(self):
        """Test CPR computation with real xielingyu"""
        from constraints import ConstraintVerifier
        verifier = ConstraintVerifier()
        
        tuples = [
            GoodMTuple(E="井底之蛙", V="没见过大世面", EE="e1", VE="v1", C="c1", Sc="s1", Sy="sy1"),
            GoodMTuple(E="竹篮打水", V="一场空", EE="e2", VE="v2", C="c2", Sc="s2", Sy="sy2"),
            GoodMTuple(E="守株待兔", V="妄想", EE="e3", VE="v3", C="c3", Sc="s3", Sy="sy3"),
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
    """Test processing operators with real xielingyu"""
    
    def test_entity_expansion(self):
        """Test entity expansion operator"""
        from operators import EntityExpansion
        
        expander = EntityExpansion()
        result = expander.expand("王婆卖瓜")
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_vehicle_expansion(self):
        """Test vehicle expansion operator"""
        from operators import VehicleExpansion
        
        expander = VehicleExpansion()
        result = expander.expand("自买自夸")
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_commonality_extraction(self):
        """Test commonality extraction operator"""
        from operators import CommonalityExtraction
        
        extractor = CommonalityExtraction()
        result = extractor.extract(
            "比喻自我吹嘘、自我标榜",
            "自己夸奖自己的货物"
        )
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_scene_generalization(self):
        """Test scene generalization operator"""
        from operators import SceneGeneralization
        
        generator = SceneGeneralization()
        result = generator.generalize(
            "比喻自我吹嘘、自我标榜",
            "自己夸奖自己的货物"
        )
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_synthesis(self):
        """Test synthesis operator"""
        from operators import Synthesis
        
        synthesizer = Synthesis()
        result = synthesizer.synthesize(
            "市场交易场景中的过度宣传",
            "自我吹嘘、自我标榜"
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
        embedding = embedder.encode("测试文本")
        assert isinstance(embedding, np.ndarray)
        assert len(embedding) == 768
    
    def test_cosine_similarity(self):
        """Test cosine similarity computation"""
        from embeddings import JinaEmbedder
        
        embedder = JinaEmbedder()
        similarity = embedder.cosine_similarity("王婆卖瓜", "王婆卖瓜")
        assert 0.0 <= similarity <= 1.0
    
    def test_batch_encode(self):
        """Test batch encoding"""
        from embeddings import JinaEmbedder
        import numpy as np
        
        embedder = JinaEmbedder()
        embeddings = embedder.encode_batch(["井底之蛙", "竹篮打水"])
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape[0] == 2