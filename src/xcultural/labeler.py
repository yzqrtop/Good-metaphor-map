from typing import Literal


def culture_label(ee: str, ve: str) -> Literal["universal", "culture-specific", "adaptable"]:
    """Add cross-cultural label to metaphor
    
    Args:
        ee: Explanation of E
        ve: Explanation of V
        
    Returns:
        Cross-cultural label: "universal", "culture-specific", or "adaptable"
    """
    # Calculate cultural specificity score
    ee_words = set(ee.split())
    ve_words = set(ve.split())
    
    if not ee_words:
        return "culture-specific"
    
    # Calculate vocabulary overlap
    intersection = len(ee_words & ve_words)
    union = len(ee_words | ve_words)
    overlap = intersection / union if union > 0 else 0
    
    # Calculate cultural specificity score (1 - overlap)
    cs = 1 - overlap
    
    # Determine label based on score
    if cs > 0.75:
        return "culture-specific"
    elif cs < 0.3:
        return "universal"
    else:
        return "adaptable"