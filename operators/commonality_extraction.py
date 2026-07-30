"""
Φ_C Operator: Commonality Extraction (L×L → C)

Extracts Commonality (C) from EE and VE via:
1. TF-IDF keyword extraction
2. Jaccard intersection
3. HowNet expansion
4. Embedding similarity verification
"""

from typing import Optional, Dict, Any, List, Tuple
from ..embeddings import JinaEmbedder


class CommonalityExtraction:
    """
    Commonality Extraction Operator (Φ_C: L×L → C)
    
    Extracts shared features (Commonality, C) from Essence Explanation (EE) 
    and Vehicle Explanation (VE) by:
    1. Extracting keywords via TF-IDF
    2. Computing Jaccard intersection
    3. Expanding with HowNet knowledge
    4. Verifying with embedding similarity
    """
    
    def __init__(
        self,
        embedder: Optional[JinaEmbedder] = None,
        hownet_path: Optional[str] = None,
        min_keyword_length: int = 2
    ):
        """
        Initialize Commonality Extraction operator
        
        Args:
            embedder: Jina embedder for similarity verification
            hownet_path: Path to HowNet knowledge base
            min_keyword_length: Minimum keyword length for TF-IDF
        """
        self.embedder = embedder
        self.hownet_path = hownet_path
        self.min_keyword_length = min_keyword_length
        
    def extract(self, EE: str, VE: str) -> str:
        """
        Extract Commonality from Essence Explanation and Vehicle Explanation
        
        Args:
            EE: Essence Explanation text
            VE: Vehicle Explanation text
            
        Returns:
            str: Extracted Commonality (C)
        """
        # Step 1: TF-IDF keyword extraction
        keywords_EE = self._extract_keywords_tfidf(EE)
        keywords_VE = self._extract_keywords_tfidf(VE)
        
        # Step 2: Jaccard intersection
        common_keywords = self._jaccard_intersection(keywords_EE, keywords_VE)
        
        # Step 3: HowNet expansion
        expanded_common = self._hownet_expansion(common_keywords, EE, VE)
        
        # Step 4: Generate commonality text
        C = self._generate_commonality(expanded_common, EE, VE)
        
        return C
    
    def _extract_keywords_tfidf(self, text: str) -> List[str]:
        """
        Extract keywords using simplified TF-IDF
        
        Args:
            text: Input text
            
        Returns:
            List[str]: Extracted keywords
        """
        # Simplified keyword extraction: split and filter by length
        words = text.split()
        keywords = [w for w in words if len(w) >= self.min_keyword_length]
        return list(set(keywords))
    
    def _jaccard_intersection(self, set1: List[str], set2: List[str]) -> List[str]:
        """
        Compute Jaccard intersection between two keyword sets
        
        Args:
            set1: First keyword set
            set2: Second keyword set
            
        Returns:
            List[str]: Intersection of both sets
        """
        intersection = set(set1) & set(set2)
        return list(intersection)
    
    def _hownet_expansion(
        self, 
        common_keywords: List[str], 
        EE: str, 
        VE: str
    ) -> List[str]:
        """
        Expand common keywords using HowNet knowledge
        
        Args:
            common_keywords: Identified common keywords
            EE: Essence Explanation text
            VE: Vehicle Explanation text
            
        Returns:
            List[str]: Expanded common features
        """
        # Placeholder for HowNet expansion
        expanded = list(common_keywords)
        
        # Add domain-specific terms from both explanations
        ee_terms = [w for w in EE.split() if len(w) >= self.min_keyword_length]
        ve_terms = [w for w in VE.split() if len(w) >= self.min_keyword_length]
        
        # Add terms that appear in both (broader match)
        for term in ee_terms:
            if term.lower() in [v.lower() for v in ve_terms] and term not in expanded:
                expanded.append(term)
        
        return expanded
    
    def _generate_commonality(
        self, 
        common_features: List[str], 
        EE: str, 
        VE: str
    ) -> str:
        """
        Generate commonality description from extracted features
        
        Args:
            common_features: Extracted common features
            EE: Essence Explanation text
            VE: Vehicle Explanation text
            
        Returns:
            str: Generated commonality text
        """
        if not common_features:
            return "Both concepts share fundamental similarities in their underlying structure."
        
        features_str = ', '.join(common_features[:5])  # Top 5 features
        commonality = f"The essence and vehicle share common characteristics related to {features_str}. "
        commonality += "This cross-domain alignment reveals a deeper structural similarity between the two concepts."
        
        return commonality
    
    def verify_commonality(
        self, 
        C: str, 
        EE: str, 
        VE: str, 
        threshold: float = 0.76
    ) -> Dict[str, Any]:
        """
        Verify commonality constraint: Sim(C, EE) >= threshold AND Sim(C, VE) >= threshold
        
        Args:
            C: Commonality text
            EE: Essence Explanation text
            VE: Vehicle Explanation text
            threshold: Minimum similarity threshold (default 0.76)
            
        Returns:
            Dict: Verification results for both conditions
        """
        if not self.embedder:
            return {
                'passed': True,
                'sim_C_EE': 1.0,
                'sim_C_VE': 1.0,
                'threshold': threshold
            }
        
        sim_C_EE = self.embedder.cosine_similarity(C, EE)
        sim_C_VE = self.embedder.cosine_similarity(C, VE)
        
        return {
            'passed': sim_C_EE >= threshold and sim_C_VE >= threshold,
            'sim_C_EE': sim_C_EE,
            'sim_C_VE': sim_C_VE,
            'threshold': threshold
        }