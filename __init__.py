"""
GoodM - Metaphor Generation, Evaluation, and Analysis System

GoodM is a modular Python project for metaphor generation, evaluation, and analysis,
implementing the seven-tuple representation and five processing operators.

Core Features:
- Five processing operators (Φ_EE, Φ_VE, Φ_C, Φ_Sc, Φ_Sy)
- Four verifiable constraints with calibrated thresholds
- Multi-dataset support (GoodM-Structured, CTRC, OOD, Intent Subset)
- Six RQ experiments for comprehensive evaluation
- Multiple LLM and semantic model wrappers
"""

__version__ = "1.0.0"

from .core.schema import GoodMTuple
from .operators import (
    EntityExpansion,
    VehicleExpansion,
    CommonalityExtraction,
    SceneGeneralization,
    Synthesis
)
from .constraints import ConstraintVerifier
from .embeddings import JinaEmbedder
from .datasets import (
    GoodMDataset,
    CTRCDataset,
    OODDataset,
    IntentSubset
)
from .models.llm_wrappers import create_model

__all__ = [
    'GoodMTuple',
    'EntityExpansion',
    'VehicleExpansion',
    'CommonalityExtraction',
    'SceneGeneralization',
    'Synthesis',
    'ConstraintVerifier',
    'JinaEmbedder',
    'GoodMDataset',
    'CTRCDataset',
    'OODDataset',
    'IntentSubset',
    'create_model'
]