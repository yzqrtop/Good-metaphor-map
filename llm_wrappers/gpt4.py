"""
GPT-4 LLM Wrapper
Uses OpenAI API
"""

import os
from typing import Optional
from openai import OpenAI, AsyncOpenAI

from .base import BaseLLM, GenerationConfig


class GPT4Wrapper(BaseLLM):
    """
    GPT-4 Model Wrapper
    
    Supported models:
    - gpt-4
    - gpt-4-turbo
    - gpt-4o
    """
    
    def __init__(
        self,
        model_name: str = "gpt-4",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        config: Optional[GenerationConfig] = None
    ):
        super().__init__(model_name, config)
        
        # Set API Key
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        
        # Initialize clients
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=base_url
        )
        self.async_client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=base_url
        )
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        config: Optional[GenerationConfig] = None
    ) -> str:
        """Synchronous generation"""
        cfg = config or self.config
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            top_p=cfg.top_p,
            frequency_penalty=cfg.frequency_penalty,
            presence_penalty=cfg.presence_penalty,
            stop=cfg.stop_sequences
        )
        
        return response.choices[0].message.content
    
    async def agenerate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        config: Optional[GenerationConfig] = None
    ) -> str:
        """Asynchronous generation"""
        cfg = config or self.config
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = await self.async_client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            top_p=cfg.top_p,
            frequency_penalty=cfg.frequency_penalty,
            presence_penalty=cfg.presence_penalty,
            stop=cfg.stop_sequences
        )
        
        return response.choices[0].message.content