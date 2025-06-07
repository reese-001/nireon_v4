"""infrastructure.llm.factory
===========================

Utility helpers for dynamically loading and instantiating **LLM backend**
classes based on the dotted‑path declared in the YAML model config, e.g.::

    backend: "infrastructure.llm.backends.openai_llm:OpenAIChatLLM"

Why a factory function instead of wiring the class name directly in the
manifest?  Because back‑ends are *data* in V4: the router parses a config
file that lists any number of models and their parameters.  Each model entry
may point at a different provider‑specific adapter, so we need to resolve the
string at runtime.
"""

from __future__ import annotations

import importlib
import inspect
import logging
from types import ModuleType
from typing import Any, Dict, Type

from domain.ports.llm_port import LLMPort

logger = logging.getLogger(__name__)

__all__ = [
    "LLMFactoryError",
    "load_class_from_path",
    "create_llm_instance",
]


class LLMFactoryError(RuntimeError):
    """Raised when a backend class cannot be resolved or instantiated."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _split_path(dotted: str) -> tuple[str, str]:
    """Return module path and attribute name from ``pkg.mod:Class`` notation."""

    if ":" not in dotted:
        raise LLMFactoryError(
            f"Backend path '{dotted}' is missing ':' – expected 'pkg.mod:ClassName'"
        )
    module_path, attr_name = dotted.split(":", 1)
    return module_path, attr_name


def _import_module(path: str) -> ModuleType:
    try:
        return importlib.import_module(path)
    except Exception as exc:  # pylint: disable=broad-except
        raise LLMFactoryError(f"Failed to import module '{path}': {exc}") from exc


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_class_from_path(dotted: str, *, expect_subclass: Type | None = None) -> Type[Any]:
    """Load a Python object given a ``pkg.mod:ClassName`` dotted path.

    Parameters
    ----------
    dotted : str
        Fully‑qualified ``module:attribute`` reference.
    expect_subclass : Type | None, optional
        If provided, the loaded attribute *must* be a subclass of this type.

    Returns
    -------
    type
        The resolved class object.
    """

    module_path, attr_name = _split_path(dotted)
    module = _import_module(module_path)

    try:
        attr = getattr(module, attr_name)
    except AttributeError as exc:
        raise LLMFactoryError(
            f"Attribute '{attr_name}' not found in module '{module_path}'."
        ) from exc

    if expect_subclass is not None and not issubclass(attr, expect_subclass):
        raise LLMFactoryError(
            f"Loaded object '{dotted}' is not a subclass of {expect_subclass.__name__}."
        )
    return attr  # type: ignore[return-value]


def create_llm_instance(model_key: str, model_cfg: Dict[str, Any]) -> LLMPort:
    """Instantiate an LLM backend based on its model config section.

    The *router* hands each model entry here so the factory can resolve the
    Python class, merge config, and return a ready instance.

    Parameters
    ----------
    model_key : str
        Logical key from the config (`openai_chat`, `gemini_pro`, etc.). Used
        only for logging and ComponentMetadata.
    model_cfg : Dict[str, Any]
        YAML dictionary under ``llm.models.<model_key>``.  Must include at
        least a ``backend`` path plus any kwargs accepted by the adapter’s
        ``__init__``.

    Returns
    -------
    LLMPort
        A live backend instance ready to be cached by the router.
    """

    if "backend" not in model_cfg:
        raise LLMFactoryError(
            f"Model '{model_key}' is missing required 'backend' field in configuration."
        )

    backend_path: str = model_cfg["backend"]
    BackendCls = load_class_from_path(backend_path, expect_subclass=LLMPort)  # type: ignore[arg-type]

    # Remove keys that are purely declarative and should not be forwarded.
    kwargs = {k: v for k, v in model_cfg.items() if k not in {"backend", "provider"}}

    try:
        instance: LLMPort = BackendCls(model_name=model_key, **kwargs)  # type: ignore[call-arg]
        logger.debug("Created LLM backend '%s' using %s", model_key, BackendCls.__name__)
        return instance
    except Exception as exc:  # pylint: disable=broad-except
        raise LLMFactoryError(
            f"Failed to instantiate backend '{backend_path}' with config for model '{model_key}': {exc}"
        ) from exc
