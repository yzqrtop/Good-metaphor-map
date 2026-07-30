"""
DeepSeek LLM Wrapper
Uses DeepSeek API
"""

import os
from typing import Optional
from .base import BaseLLM, GenerationConfig


class DeepSeekWrapper(BaseLLM):
    """
    DeepSeek Model Wrapper
    
    Supported models:
    - deepseek-v3
    - deepseek-chat
    - deepseek-coder
    """
    
    def __init__(
        self,
        model_name: str = "deepseek-v3",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        config: Optional[GenerationConfig] = None
    ):
        super().__init__(model_name, config)
        
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("DeepSeek API key is required. Set DEEPSEEK_API_KEY environment variable or pass api_key parameter.")
        
        self.base_url = base_url or "https://api.deepseek.com/v1"
    
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