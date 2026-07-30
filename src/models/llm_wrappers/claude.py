"""
Claude LLM Wrapper
Uses Anthropic API
"""

import os
from typing import Optional
from .base import BaseLLM, GenerationConfig


class ClaudeWrapper(BaseLLM):
    """
    Claude Model Wrapper
    
    Supported models:
    - claude-3.5-sonnet
    - claude-3-opus
    - claude-3-haiku
    """
    
    def __init__(
        self,
        model_name: str = "claude-3.5-sonnet",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        config: Optional[GenerationConfig] = None
    ):
        super().__init__(model_name, config)
        
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("Anthropic API key is required. Set ANTHROPIC_API_KEY environment variable or pass api_key parameter.")
        
        self.base_url = base_url or "https://api.anthropic.com/v1/messages"
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        config: Optional[GenerationConfig] = None
    ) -> str:
        """Synchronous generation"""
        import requests
        
        cfg = config or self.config
        
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01"
        }
        
        messages = []
        if system_prompt:
            # For Claude, system prompt is passed separately
            pass
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": cfg.max_tokens,
            "temperature": cfg.temperature
        }
        
        if system_prompt:
            payload["system"] = system_prompt
        
        response = requests.post(
            self.base_url,
            headers=headers,
            json=payload
        )
        
        if response.status_code == 200:
            result = response.json()
            return result["content"][0]["text"]
        else:
            raise RuntimeError(f"API call failed: {response.text}")
    
    async def agenerate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        config: Optional[GenerationConfig] = None
    ) -> str:
        """Asynchronous generation"""
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.generate,
            prompt,
            system_prompt,
            config
        )