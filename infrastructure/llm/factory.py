# C:\Users\erees\Documents\development\nireon_v4\infrastructure\llm\factory.py
from __future__ import annotations
import importlib
import inspect # Keep inspect if needed for other things, though not directly for create_llm_instance
import logging
from types import ModuleType
from typing import Any, Dict, Type
from domain.ports.llm_port import LLMPort # MODIFIED

logger = logging.getLogger(__name__)
__all__ = ['LLMFactoryError', 'load_class_from_path', 'create_llm_instance']

class LLMFactoryError(RuntimeError):
    pass # Keep as is

def _split_path(dotted: str) -> tuple[str, str]: # Keep as is
    if ':' not in dotted:
        raise LLMFactoryError(f"Backend path '{dotted}' is missing ':' – expected 'pkg.mod:ClassName'")
    module_path, attr_name = dotted.split(':', 1)
    return (module_path, attr_name)

def _import_module(path: str) -> ModuleType: # Keep as is
    try:
        return importlib.import_module(path)
    except Exception as exc:
        raise LLMFactoryError(f"Failed to import module '{path}': {exc}") from exc

def load_class_from_path(dotted: str, *, expect_subclass: Type | None = None) -> Type[Any]: # Keep as is
    module_path, attr_name = _split_path(dotted)
    module = _import_module(module_path)
    try:
        attr = getattr(module, attr_name)
    except AttributeError as exc:
        raise LLMFactoryError(f"Attribute '{attr_name}' not found in module '{module_path}'.") from exc
    
    if not inspect.isclass(attr): # Added check
        raise LLMFactoryError(f"Loaded object '{dotted}' is not a class.")

    if expect_subclass is not None and not issubclass(attr, expect_subclass):
        raise LLMFactoryError(f"Loaded object '{dotted}' is not a subclass of {expect_subclass.__name__}.")
    return attr

def create_llm_instance(model_key: str, model_cfg_dict: Dict[str, Any]) -> LLMPort: # model_cfg_dict is the config for THIS model
    if 'backend' not in model_cfg_dict: # Check in the specific model's config
        raise LLMFactoryError(f"Model '{model_key}' is missing required 'backend' field in its configuration.")
    
    backend_path: str = model_cfg_dict['backend']
    BackendCls = load_class_from_path(backend_path, expect_subclass=LLMPort)
    
    # Prepare kwargs for the backend constructor
    # Pass the model_key as 'model_name' if the constructor expects it,
    # and also pass the full config for this model if it expects 'config'.
    # Other keys from model_cfg_dict (excluding 'backend', 'provider') can be passed as direct kwargs.
    
    constructor_kwargs = {k: v for k, v in model_cfg_dict.items() if k not in {'backend', 'provider'}}
    
    # Some backends might want the model_key as 'model_name' or 'model'
    # Some might want the whole model_cfg_dict as 'config'
    # This requires knowing the constructor signature or having a convention.
    
    # Convention: Pass 'model_name=model_key' and 'config=model_cfg_dict'
    # Also pass individual kwargs from model_cfg_dict.
    # The backend class constructor should pick what it needs.
    
    final_kwargs_for_ctor = {'model_name': model_key, 'config': model_cfg_dict, **constructor_kwargs}

    try:
        # The backend class (e.g., OpenAILLMAdapter) __init__ should handle these.
        # Example: OpenAILLMAdapter(model_name='gpt-4', config={'model':'gpt-4', 'timeout':30, ...}, timeout=30)
        # The specific adapter will decide how to use 'model_name' vs 'config.model'.
        # My updated OpenAILLMAdapter uses 'model_name' passed to __init__ first.
        instance: LLMPort = BackendCls(**final_kwargs_for_ctor)
        logger.debug(f"Created LLM backend for model_key '{model_key}' using adapter {BackendCls.__name__}")
        return instance
    except Exception as exc:
        raise LLMFactoryError(f"Failed to instantiate backend '{backend_path}' with config for model '{model_key}': {exc}") from exc