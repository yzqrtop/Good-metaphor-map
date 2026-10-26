"""
LLM Wrapper Module

Backends are imported lazily so that a missing vendor SDK (dashscope, openai)
only fails when that model is actually requested.
"""

from typing import Dict, Type

from .base import BaseLLM, GenerationConfig

# model name → (submodule, class name)
MODEL_REGISTRY: Dict[str, tuple] = {
    "qwen-plus": ("qwen", "QwenWrapper"),
    "qwen-max": ("qwen", "QwenWrapper"),
    "qwen-turbo": ("qwen", "QwenWrapper"),
    # Models named in the paper: 1.5B refines EE/VE, 7B selects Sc/Sy.
    "qwen2.5-1.5b-instruct": ("qwen", "QwenWrapper"),
    "qwen2.5-7b-instruct": ("qwen", "QwenWrapper"),
    "gpt-4": ("gpt4", "GPT4Wrapper"),
    "gpt-4-turbo": ("gpt4", "GPT4Wrapper"),
    "gpt-4o": ("gpt4", "GPT4Wrapper"),
    "gpt-5": ("gpt4", "GPT4Wrapper"),
}


def create_model(model_name: str, **kwargs) -> BaseLLM:
    """
    Instantiate an LLM wrapper by model name.

    Raises:
        ValueError: unknown model name.
        ImportError: the vendor SDK for that backend is not installed.
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available models: {list(MODEL_REGISTRY.keys())}"
        )

    module_name, class_name = MODEL_REGISTRY[model_name]
    try:
        module = __import__(f"{__name__}.{module_name}", fromlist=[class_name])
    except ImportError as exc:
        raise ImportError(
            f"Backend '{module_name}' is not available: install its SDK "
            f"(original error: {exc})."
        ) from exc

    wrapper_class: Type[BaseLLM] = getattr(module, class_name)
    return wrapper_class(model_name=model_name, **kwargs)


__all__ = ["BaseLLM", "GenerationConfig", "MODEL_REGISTRY", "create_model"]
