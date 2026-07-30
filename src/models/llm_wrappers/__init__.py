"""
LLM Wrapper Module
Supports multiple LLM backends for the GoodM framework
"""

from .base import BaseLLM, GenerationConfig
from .qwen import QwenWrapper
from .gpt4 import GPT4Wrapper
from .claude import ClaudeWrapper
from .deepseek import DeepSeekWrapper
from .opensource import OpenSourceWrapper

__all__ = [
    'BaseLLM',
    'GenerationConfig',
    'QwenWrapper',
    'GPT4Wrapper',
    'ClaudeWrapper',
    'DeepSeekWrapper',
    'OpenSourceWrapper'
]

# Model registry for easy instantiation
MODEL_REGISTRY = {
    'qwen-plus': QwenWrapper,
    'qwen-max': QwenWrapper,
    'qwen-turbo': QwenWrapper,
    'gpt-4': GPT4Wrapper,
    'gpt-4-turbo': GPT4Wrapper,
    'gpt-4o': GPT4Wrapper,
    'gpt-5': GPT4Wrapper,
    'claude-3.5-sonnet': ClaudeWrapper,
    'claude-3-opus': ClaudeWrapper,
    'deepseek-v3': DeepSeekWrapper,
    'deepseek-chat': DeepSeekWrapper,
    'llama-3.1-8b': OpenSourceWrapper,
    'mistral-7b': OpenSourceWrapper,
    'gemma-7b': OpenSourceWrapper,
    'yi-34b-chat': OpenSourceWrapper
}

def create_model(model_name: str, **kwargs) -> BaseLLM:
    """
    Factory function to create LLM wrapper by model name
    
    Args:
        model_name: Name of the model (e.g., 'qwen-plus', 'gpt-4')
        **kwargs: Additional parameters for the model wrapper
        
    Returns:
        BaseLLM: Initialized model wrapper
        
    Raises:
        ValueError: If model name is not recognized
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available models: {list(MODEL_REGISTRY.keys())}"
        )
    
    wrapper_class = MODEL_REGISTRY[model_name]
    return wrapper_class(model_name=model_name, **kwargs)