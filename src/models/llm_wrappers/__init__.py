"""
LLM 包装器模块
"""

from .base import BaseLLM, GenerationConfig
from .qwen import QwenWrapper
from .gpt4 import GPT4Wrapper

__all__ = [
    'BaseLLM',
    'GenerationConfig',
    'QwenWrapper',
    'GPT4Wrapper'
]
