"""
Open-source LLM Wrapper
Supports LLaMA-3.1, Mistral, Gemma, Yi models via local deployment or API
"""

import os
from typing import Optional
from .base import BaseLLM, GenerationConfig


class OpenSourceWrapper(BaseLLM):
    """
    Open-source Model Wrapper
    
    Supported models:
    - llama-3.1-8b
    - mistral-7b
    - gemma-7b
    - yi-34b-chat
    
    Supports local deployment (vLLM, Ollama) or API access
    """
    
    def __init__(
        self,
        model_name: str = "llama-3.1-8b",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        config: Optional[GenerationConfig] = None
    ):
        super().__init__(model_name, config)
        
        self.api_key = api_key or os.getenv("OPEN_SOURCE_API_KEY", "EMPTY")
        self.base_url = base_url or os.getenv("OPEN_SOURCE_BASE_URL", "http://localhost:8000/v1")
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        config: Optional[GenerationConfig] = None
    ) -> str:
        """Synchronous generation"""
        from openai import OpenAI
        
        cfg = config or self.config
        
        client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens
        )
        
        return response.choices[0].message.content
    
    async def agenerate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        config: Optional[GenerationConfig] = None
    ) -> str:
        """Asynchronous generation"""
        from openai import AsyncOpenAI
        
        cfg = config or self.config
        
        client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = await client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens
        )
        
        return response.choices[0].message.content