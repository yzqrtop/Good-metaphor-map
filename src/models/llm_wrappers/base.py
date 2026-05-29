"""
LLM Wrapper Base Class Definition
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
import asyncio
from dataclasses import dataclass


@dataclass
class GenerationConfig:
    """Generation configuration"""
    temperature: float = 0.7
    max_tokens: int = 2048
    top_p: float = 0.9
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop_sequences: Optional[List[str]] = None


class BaseLLM(ABC):
    """
    LLM Wrapper Base Class
    
    All concrete LLM implementations should inherit from this class
    """
    
    def __init__(self, model_name: str, config: Optional[GenerationConfig] = None):
        self.model_name = model_name
        self.config = config or GenerationConfig()
    
    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        config: Optional[GenerationConfig] = None
    ) -> str:
        """
        Synchronous text generation
        
        Args:
            prompt: User prompt
            system_prompt: System prompt (optional)
            config: Generation configuration (optional, overrides default configuration)
            
        Returns:
            str: Generated text
        """
        pass
    
    @abstractmethod
    async def agenerate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        config: Optional[GenerationConfig] = None
    ) -> str:
        """
        Asynchronous text generation
        
        Args:
            prompt: User prompt
            system_prompt: System prompt (optional)
            config: Generation configuration (optional, overrides default configuration)
            
        Returns:
            str: Generated text
        """
        pass
    
    def batch_generate(
        self,
        prompts: List[str],
        system_prompt: Optional[str] = None,
        config: Optional[GenerationConfig] = None
    ) -> List[str]:
        """
        Batch synchronous generation
        
        Args:
            prompts: List of prompts
            system_prompt: System prompt (optional)
            config: Generation configuration (optional)
            
        Returns:
            List[str]: List of generated texts
        """
        return [
            self.generate(p, system_prompt, config)
            for p in prompts
        ]
    
    async def batch_agenerate(
        self,
        prompts: List[str],
        system_prompt: Optional[str] = None,
        config: Optional[GenerationConfig] = None,
        max_concurrency: int = 5
    ) -> List[str]:
        """
        Batch asynchronous generation (with concurrency limit)
        
        Args:
            prompts: List of prompts
            system_prompt: System prompt (optional)
            config: Generation configuration (optional)
            max_concurrency: Maximum concurrency
            
        Returns:
            List[str]: List of generated texts
        """
        semaphore = asyncio.Semaphore(max_concurrency)
        
        async def _generate_with_limit(prompt: str) -> str:
            async with semaphore:
                return await self.agenerate(prompt, system_prompt, config)
        
        tasks = [_generate_with_limit(p) for p in prompts]
        return await asyncio.gather(*tasks)