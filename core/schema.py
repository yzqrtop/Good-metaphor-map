"""
GoodM Core Data Structure Definition
Corresponds to the GoodM notation in the paper: (E, V, EE, VE, C, Sc, Sy)
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import json


@dataclass
class GoodMTuple:
    """
    GoodM Seven-Tuple Data Structure
    
    Attributes:
        E: Entity (本体) - The first part of the two-part allegorical saying, the thing being compared
        V: Vehicle (喻体) - The second part of the two-part allegorical saying, the thing used for comparison
        EE: Entity Explanation (本体阐释) - Detailed explanation of the entity
        VE: Vehicle Explanation (喻体阐释) - Detailed explanation of the vehicle
        C: Commonality (共性特征) - Shared features between EE and VE
        Sc: Scenario (场景泛化) - Scenario generalized from EE and VE
        Sy: Synthesis (综合意义) - Deep meaning synthesized from Sc and C
    """
    E: str = ""
    V: str = ""
    EE: str = ""
    VE: str = ""
    C: str = ""
    Sc: str = ""
    Sy: str = ""
    
    # Metadata field
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def validate(self) -> bool:
        """
        Validate the completeness of the seven-tuple
        Check that all 7 fields are non-empty
        
        Returns:
            bool: True if validation passes, False otherwise
        """
        fields = [self.E, self.V, self.EE, self.VE, self.C, self.Sc, self.Sy]
        return all(field and isinstance(field, str) and field.strip() for field in fields)
    
    def validate_with_details(self) -> Dict[str, Any]:
        """
        Detailed validation, returns the status of each field
        
        Returns:
            Dict: Contains validation results and missing field information
        """
        field_names = ['E', 'V', 'EE', 'VE', 'C', 'Sc', 'Sy']
        field_values = [self.E, self.V, self.EE, self.VE, self.C, self.Sc, self.Sy]
        
        missing_fields = []
        for name, value in zip(field_names, field_values):
            if not value or not isinstance(value, str) or not value.strip():
                missing_fields.append(name)
        
        return {
            'is_valid': len(missing_fields) == 0,
            'missing_fields': missing_fields,
            'field_count': 7 - len(missing_fields)
        }
    
    def to_prefix(self) -> str:
        """
        Concatenate the seven-tuple into LLM input prefix format
        
        Returns:
            str: Formatted prefix string
        """
        return (
            f"Essence: {self.E}\n"
            f"Vehicle: {self.V}\n"
            f"Essence Explanation: {self.EE}\n"
            f"Vehicle Explanation: {self.VE}\n"
            f"Commonality: {self.C}\n"
            f"Scenario: {self.Sc}\n"
            f"Synthesis: {self.Sy}"
        )
    
    def to_embedding_input(self) -> List[str]:
        """
        Return a list of strings for Jina Embedding encoding
        
        Returns:
            List[str]: List of strings for each field of the seven-tuple
        """
        return [self.E, self.V, self.EE, self.VE, self.C, self.Sc, self.Sy]
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary format
        
        Returns:
            Dict: Dictionary containing all fields
        """
        return {
            'E': self.E,
            'V': self.V,
            'EE': self.EE,
            'VE': self.VE,
            'C': self.C,
            'Sc': self.Sc,
            'Sy': self.Sy,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GoodMTuple':
        """
        Create a GoodMTuple instance from a dictionary
        
        Args:
            data: Dictionary containing seven-tuple fields
            
        Returns:
            GoodMTuple: Created instance
        """
        metadata = data.get('metadata', {})
        return cls(
            E=data.get('E', ''),
            V=data.get('V', ''),
            EE=data.get('EE', ''),
            VE=data.get('VE', ''),
            C=data.get('C', ''),
            Sc=data.get('Sc', ''),
            Sy=data.get('Sy', ''),
            metadata=metadata
        )
    
    def to_json(self) -> str:
        """
        Convert to JSON string
        
        Returns:
            str: JSON formatted string
        """
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'GoodMTuple':
        """
        Create an instance from a JSON string
        
        Args:
            json_str: JSON formatted string
            
        Returns:
            GoodMTuple: Created instance
        """
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    def get_core_fields(self) -> Dict[str, str]:
        """
        Get core seven-tuple fields (excluding metadata)
        
        Returns:
            Dict[str, str]: Dictionary of core fields
        """
        return {
            'E': self.E,
            'V': self.V,
            'EE': self.EE,
            'VE': self.VE,
            'C': self.C,
            'Sc': self.Sc,
            'Sy': self.Sy
        }
    
    def __repr__(self) -> str:
        """String representation"""
        return f"GoodMTuple(E='{self.E}', V='{self.V}', valid={self.validate()})"
    
    def __hash__(self) -> int:
        """Hash for sets and dictionaries"""
        return hash((self.E, self.V, self.EE, self.VE, self.C, self.Sc, self.Sy))
    
    def __eq__(self, other) -> bool:
        """Equality comparison"""
        if not isinstance(other, GoodMTuple):
            return False
        return (
            self.E == other.E and
            self.V == other.V and
            self.EE == other.EE and
            self.VE == other.VE and
            self.C == other.C and
            self.Sc == other.Sc and
            self.Sy == other.Sy
        )