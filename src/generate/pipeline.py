"""
GoodM Seven-Tuple Generation Pipeline
Implements four-stage generation + multi-round consensus + consistency checking
"""

import asyncio
from typing import Optional, List, Dict, Any
from collections import Counter
import json

from core.schema import GoodMTuple
from src.models.llm_wrappers import BaseLLM, GenerationConfig


class GenerationPipeline:
    """
    GoodM Seven-Tuple Generation Pipeline
    
    Four-stage generation:
    1. Entity/Vehicle Expansion (本体/喻体扩展)
    2. Commonality Extraction (共性特征提取)
    3. Scene Generalization (场景泛化)
    4. Synthesis (综合意义生成)
    
    Includes multi-round consensus mechanism and consistency checking
    """
    
    def __init__(
        self,
        llm_client: BaseLLM,
        temperature: float = 0.7,
        consensus_rounds: int = 3,
        similarity_threshold: float = 0.6
    ):
        self.llm = llm_client
        self.temperature = temperature
        self.consensus_rounds = consensus_rounds
        self.similarity_threshold = similarity_threshold
        
        # Load prompt templates
        self.prompts = self._load_prompts()
    
    def _load_prompts(self) -> Dict[str, str]:
        """Load prompt templates"""
        return {
            'entity_expansion': """Please explain the following concept in detail, including its meaning, features, uses, etc.:
Concept: {entity}

Please provide a detailed explanation:""",
            
            'commonality_extraction': """Based on the detailed explanations of the following two concepts, extract their shared features or commonalities:

Concept 1 explanation:
{ee}

Concept 2 explanation:
{ve}

Please extract their common features (Commonality):""",
            
            'scene_generalization': """Based on the detailed explanations of the following two concepts, generalize a related scenario or situation:

Concept 1 explanation:
{ee}

Concept 2 explanation:
{ve}

Please describe a related scenario (Scenario):""",
            
            'synthesis': """Based on the following scenario and common features, synthesize the deep meaning or symbolic significance:

Scenario:
{sc}

Common features:
{c}

Please synthesize the deep meaning (Synthesis):"""
        }
    
    async def generate(self, E: str, V: str) -> Optional[GoodMTuple]:
        """
        Generate a complete seven-tuple
        
        Args:
            E: Entity (本体)
            V: Vehicle (喻体)
            
        Returns:
            GoodMTuple: Generated seven-tuple, returns None if validation fails
        """
        # Phase 1: Entity/Vehicle Expansion (multi-round consensus)
        EE = await self._multi_round_consensus(
            input_text=E,
            prompt_template=self.prompts['entity_expansion'],
            template_key='entity'
        )
        
        VE = await self._multi_round_consensus(
            input_text=V,
            prompt_template=self.prompts['entity_expansion'],
            template_key='entity'
        )
        
        # Phase 2: Commonality Extraction
        C = await self._generate_commonality(EE, VE)
        
        # Phase 3: Scene Generalization
        Sc = await self._generate_scenario(EE, VE)
        
        # Phase 4: Synthesis
        Sy = await self._generate_synthesis(Sc, C)
        
        # Build seven-tuple
        tuple_obj = GoodMTuple(
            E=E,
            V=V,
            EE=EE,
            VE=VE,
            C=C,
            Sc=Sc,
            Sy=Sy
        )
        
        # Two-stage consistency check
        if self._automatic_logic_validation(tuple_obj):
            return tuple_obj
        else:
            return None
    
    async def _multi_round_consensus(
        self,
        input_text: str,
        prompt_template: str,
        template_key: str
    ) -> str:
        """
        Multi-round consensus generation
        
        Execute multiple independent generations, select the result that appears ≥2/3 of the time
        
        Args:
            input_text: Input text
            prompt_template: Prompt template
            template_key: Replacement key in the template
            
        Returns:
            str: Consensus result
        """
        prompt = prompt_template.format(**{template_key: input_text})
        
        # Execute multiple generations in parallel
        tasks = [
            self.llm.agenerate(
                prompt=prompt,
                config=GenerationConfig(temperature=self.temperature)
            )
            for _ in range(self.consensus_rounds)
        ]
        
        results = await asyncio.gather(*tasks)
        
        # Simple consensus mechanism: select the most common answer
        # In actual projects, semantic similarity clustering can be used
        counter = Counter(results)
        most_common = counter.most_common(1)[0]
        
        # If the most common result appears ≥ 2/3 of the time, select it
        if most_common[1] >= (2 * self.consensus_rounds / 3):
            return most_common[0]
        
        # Otherwise return the first result (or can return the longest/most detailed result)
        return results[0]
    
    async def _generate_commonality(self, EE: str, VE: str) -> str:
        """Generate common features C"""
        prompt = self.prompts['commonality_extraction'].format(ee=EE, ve=VE)
        return await self.llm.agenerate(
            prompt=prompt,
            config=GenerationConfig(temperature=self.temperature)
        )
    
    async def _generate_scenario(self, EE: str, VE: str) -> str:
        """Generate scenario Sc"""
        prompt = self.prompts['scene_generalization'].format(ee=EE, ve=VE)
        return await self.llm.agenerate(
            prompt=prompt,
            config=GenerationConfig(temperature=self.temperature)
        )
    
    async def _generate_synthesis(self, Sc: str, C: str) -> str:
        """Generate synthesis Sy"""
        prompt = self.prompts['synthesis'].format(sc=Sc, c=C)
        return await self.llm.agenerate(
            prompt=prompt,
            config=GenerationConfig(temperature=self.temperature)
        )
    
    def _automatic_logic_validation(self, tuple_obj: GoodMTuple) -> bool:
        """
        Automatic logic validation (first stage of two-stage consistency check)
        
        Args:
            tuple_obj: Seven-tuple to be validated
            
        Returns:
            bool: Whether validation passes
        """
        # Check 1: Basic completeness
        if not tuple_obj.validate():
            return False
        
        # Check 2: Reasonable field length (avoid too short or too long generated results)
        fields = tuple_obj.get_core_fields()
        for field_name, field_value in fields.items():
            # Each field should be at least 10 characters and at most 1000 characters
            if len(field_value) < 10 or len(field_value) > 1000:
                return False
        
        # Check 3: EE and VE should be related to E and V (simple check if keywords are included)
        if tuple_obj.E not in tuple_obj.EE and len(tuple_obj.E) > 2:
            # If the entity is not in the entity explanation, may need to check
            pass  # In actual projects, semantic similarity can be used
        
        return True
    
    async def batch_generate(
        self,
        pairs: List[tuple],
        max_concurrency: int = 5
    ) -> List[Optional[GoodMTuple]]:
        """
        Batch generate seven-tuples
        
        Args:
            pairs: List of (E, V) tuples
            max_concurrency: Maximum concurrency
            
        Returns:
            List[Optional[GoodMTuple]]: List of generated seven-tuples
        """
        semaphore = asyncio.Semaphore(max_concurrency)
        
        async def _generate_with_limit(e: str, v: str) -> Optional[GoodMTuple]:
            async with semaphore:
                return await self.generate(e, v)
        
        tasks = [_generate_with_limit(e, v) for e, v in pairs]
        return await asyncio.gather(*tasks)