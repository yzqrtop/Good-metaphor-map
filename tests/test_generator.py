import pytest
import asyncio

from src.generate.generator import GoodMGenerator


@pytest.mark.asyncio
async def test_goodm_generator():
    """Test GoodMGenerator class"""
    generator = GoodMGenerator(model_name="test-model")
    
    # Test ae method
    e = "井底之蛙"
    ee = await generator.ae(e)
    assert isinstance(ee, str)
    assert ee != ""
    
    # Test ve method
    v = "Related metaphor vehicle"
    ve = await generator.ve(v)
    assert isinstance(ve, str)
    assert ve != ""
    
    # Test c method
    c = await generator.c(ee, ve)
    assert isinstance(c, str)
    assert c != ""
    
    # Test sc method
    sc = await generator.sc(ee, ve)
    assert isinstance(sc, str)
    assert sc != ""
    
    # Test sy method
    sy = await generator.sy(sc, c)
    assert isinstance(sy, str)
    assert sy != ""
    
    # Test build_one method
    result = await generator.build_one(e, v)
    assert isinstance(result, dict)
    assert "E" in result
    assert "V" in result
    assert "EE" in result
    assert "VE" in result
    assert "C" in result
    assert "Sc" in result
    assert "Sy" in result