import pytest

from src.xcultural.labeler import culture_label


def test_culture_label():
    """Test cross-cultural labeling function"""
    # Test universal label
    ee_universal = "This is an explanation of a universal concept"
    ve_universal = "This is an explanation of a universal concept vehicle"
    label_universal = culture_label(ee_universal, ve_universal)
    assert label_universal in ["universal", "adaptable"]
    
    # Test culture-specific label
    ee_specific = "This is an explanation of a concept with Chinese cultural characteristics, containing many unique Chinese elements"
    ve_specific = "This is a completely different vehicle explanation with no commonalities with the original concept"
    label_specific = culture_label(ee_specific, ve_specific)
    assert label_specific in ["culture-specific", "adaptable"]
    
    # Test adaptable label
    ee_adaptable = "This is an explanation of a concept with some cultural characteristics"
    ve_adaptable = "This is a vehicle explanation with some overlap with the original concept"
    label_adaptable = culture_label(ee_adaptable, ve_adaptable)
    assert label_adaptable == "adaptable"
    
    # Test edge case
    ee_empty = ""
    ve_empty = ""
    label_empty = culture_label(ee_empty, ve_empty)
    assert label_empty == "culture-specific"