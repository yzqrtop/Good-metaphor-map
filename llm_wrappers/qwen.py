"""
Qwen LLM Wrapper
Uses DashScope API
"""

import os
from typing import Optional
import dashscope
from dashscope import Generation

from .base import BaseLLM, GenerationConfig


class QwenWrapper(BaseLLM):
    """
    Qwen Model Wrapper
    
    Supported models:
    - qwen-plus (recommended)
    - qwen-max
    - qwen-turbo
    """
    
    def __init__(
        self,
        model_name: str = "qwen-plus",
        api_key: Optional[str] = None,
        config: Optional[GenerationConfig] = None
    ):
        super().__init__(model_name, config)
        
        # Set API Key
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise ValueError("DashScope API key is required. Set DASHSCOPE_API_KEY environment variable or pass api_key parameter.")
        
        dashscope.api_key = self.api_key
    
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
        
        response = Generation.call(
            model=self.model_name,
            messages=messages,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            top_p=cfg.top_p,
            result_format="message"
        )
        
        if response.status_code == 200:
            return response.output.choices[0].message.content
        else:
            raise RuntimeError(f"API call failed: {response.message}")
    
    async def agenerate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        config: Optional[GenerationConfig] = None
    ) -> str:
        """Asynchronous generation (using thread pool to wrap synchronous call)"""
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.generate,
            prompt,
            system_prompt,
            config
        )