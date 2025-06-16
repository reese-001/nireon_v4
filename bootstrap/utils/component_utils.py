"""nireon_v4.bootstrap.utils.component_utils
Utility helpers for dynamic component construction, configuration, and validation.

Public API (stable):
    - get_pydantic_defaults
    - create_component_instance
    - inject_dependencies
    - validate_component_interfaces
    - configure_component_logging
    - prepare_component_metadata
"""
from __future__ import annotations
from __future__ import absolute_import

import dataclasses
import functools
import inspect
import logging
import types
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Type,
    Union,
    cast,
    TYPE_CHECKING,
)

from pydantic import BaseModel

from core.base_component import NireonBaseComponent
from core.lifecycle import ComponentMetadata

if TYPE_CHECKING:  # Avoid circular import at runtime
    from bootstrap.exceptions import ComponentInstantiationError

logger = logging.getLogger(__name__)

__all__ = [
    "get_pydantic_defaults",
    "create_component_instance",
    "inject_dependencies",
    "validate_component_interfaces",
    "configure_component_logging",
    "prepare_component_metadata",
]

###############################################################################
# Internal helpers
###############################################################################


def _update_attr_if_different(obj: Any, attr: str, new_value: Any) -> None:
    """Set ``obj.attr`` to *new_value* only if it differs, bypassing frozen/slots."""
    current = getattr(obj, attr, dataclasses.MISSING)
    if current == new_value:
        return
    try:
        object.__setattr__(obj, attr, new_value)
    except Exception:
        # Fallback for non‑dataclass or protected attrs
        setattr(obj, attr, new_value)


def _is_pydantic_model(cls: Type) -> bool:
    return issubclass(cls, BaseModel)


###############################################################################
# 1. Default‑config extraction
###############################################################################

@functools.lru_cache(maxsize=256)
def _collect_pydantic_defaults(model_cls: Type[BaseModel]) -> Dict[str, Any]:
    """Handle both Pydantic v1 (`__fields__`) and v2 (`model_fields`)."""
    defaults: Dict[str, Any] = {}
    if hasattr(model_cls, "model_fields"):  # v2
        for name, info in model_cls.model_fields.items():  # type: ignore[attr-defined]
            if info.default is not ...:
                defaults[name] = info.default
            elif getattr(info, "default_factory", None):
                try:
                    defaults[name] = info.default_factory()  # type: ignore[operator]
                except Exception:  # pragma: no cover
                    logger.debug("default_factory failed for %s.%s", model_cls.__name__, name)
    elif hasattr(model_cls, "__fields__"):  # v1
        for name, field in model_cls.__fields__.items():  # type: ignore[attr-defined]
            if field.default is not ...:
                defaults[name] = field.default
            elif field.default_factory:
                try:
                    defaults[name] = field.default_factory()
                except Exception:  # pragma: no cover
                    logger.debug("default_factory failed for %s.%s (v1)", model_cls.__name__, name)
    return defaults


def get_pydantic_defaults(component_class: Type, component_name: str) -> Dict[str, Any]:
    """Return a merged mapping of configuration defaults for *component_class*.

    Order of precedence (highest → lowest):
        1. Values explicitly set in the component's Pydantic ConfigModel.
        2. Values in the legacy ``DEFAULT_CONFIG`` dict.
    """
    defaults: Dict[str, Any] = {}
    try:
        if hasattr(component_class, "ConfigModel") and _is_pydantic_model(component_class.ConfigModel):
            defaults.update(_collect_pydantic_defaults(component_class.ConfigModel))  # type: ignore[arg-type]

        if hasattr(component_class, "DEFAULT_CONFIG") and isinstance(component_class.DEFAULT_CONFIG, dict):
            # Preserve Pydantic preferences
            merged = component_class.DEFAULT_CONFIG.copy()
            merged.update(defaults)
            defaults = merged
    except Exception as exc:  # pragma: no cover
        logger.debug("Error extracting defaults from %s: %s", component_name, exc)

    logger.debug(
        "Resolved %d default(s) for component '%s'",
        len(defaults),
        component_name,
    )
    return defaults


###############################################################################
# 2. Dynamic instantiation
###############################################################################

_DEFAULT_GATEWAY_PARAMS: tuple[str, ...] = (
    "llm_router",
    "parameter_service",
    "frame_factory",
    "budget_manager",
    "event_bus",
)


async def create_component_instance(
    component_class: Type[NireonBaseComponent],
    resolved_config_for_instance: Dict[str, Any],
    instance_id: str,
    instance_metadata_object: ComponentMetadata,
    common_deps: Optional[Dict[str, Any]] = None,
) -> NireonBaseComponent:
    """Instantiate *component_class* with smart constructor matching."""
    ctor_sig = inspect.signature(component_class.__init__)
    ctor_params = ctor_sig.parameters
    kwargs: Dict[str, Any] = {}

    # Config parameter
    if "config" in ctor_params:
        kwargs["config"] = resolved_config_for_instance
    elif "cfg" in ctor_params:
        kwargs["cfg"] = resolved_config_for_instance

    # Metadata parameter
    if "metadata_definition" in ctor_params:
        kwargs["metadata_definition"] = instance_metadata_object
    elif "metadata" in ctor_params:
        kwargs["metadata"] = instance_metadata_object

    # Common dependency bundle
    for name, dep in (common_deps or {}).items():
        if name in ctor_params:
            kwargs[name] = dep

    # Ensure MechanismGateway sentinel params are present
    if component_class.__name__ == "MechanismGateway":
        for name in _DEFAULT_GATEWAY_PARAMS:
            if name in ctor_params and name not in kwargs:
                kwargs[name] = None

    logger.debug(
        "Creating component '%s' of type %s with args: %s",
        instance_id,
        component_class.__name__,
        sorted(kwargs),
    )
    try:
        instance = component_class(**kwargs)  # type: ignore[arg-type]
    except Exception as exc:
        from bootstrap.exceptions import ComponentInstantiationError  # local import to avoid loops

        raise ComponentInstantiationError(
            f"Instantiation failed for '{instance_id}': {exc}", component_id=instance_id
        ) from exc

    # Enforce id / metadata consistency
    if hasattr(instance, "component_id"):
        _update_attr_if_different(instance, "component_id", instance_id)
    else:
        _update_attr_if_different(instance, "_component_id", instance_id)

    if hasattr(instance, "metadata") and isinstance(instance.metadata, ComponentMetadata):
        _update_attr_if_different(instance.metadata, "id", instance_id)
    else:
        _update_attr_if_different(instance, "_metadata_definition", instance_metadata_object)

    logger.info("Component '%s' (%s) instantiated successfully", instance_id, component_class.__name__)
    return cast(NireonBaseComponent, instance)


###############################################################################
# 3. Dependency injection
###############################################################################

def _can_setattr(obj: Any, attr: str) -> bool:
    """Return ``True`` if *attr* can be set on *obj* without AttributeError."""
    if hasattr(obj, "__slots__") and attr not in obj.__slots__:  # type: ignore[attr-defined]
        return False
    return True


def inject_dependencies(
    instance: NireonBaseComponent,
    dependency_map: Dict[str, Any],
    registry: Optional[Any] = None,  # retained for compatibility
) -> None:
    """Inject values in *dependency_map* onto *instance* by name."""
    for name, value in dependency_map.items():
        target_attr = f"_{name}" if hasattr(instance, f"_{name}") else name
        if _can_setattr(instance, target_attr):
            try:
                setattr(instance, target_attr, value)
                logger.debug("Injected dependency '%s' into %s", name, instance.component_id)
            except Exception as exc:  # pragma: no cover
                logger.warning("Failed to inject '%s' into %s: %s", name, instance.component_id, exc)
        else:
            logger.warning(
                "Dependency '%s' could not be injected into %s (attribute missing or read‑only)",
                name,
                instance.component_id,
            )


###############################################################################
# 4. Interface validation
###############################################################################

def validate_component_interfaces(
    instance: NireonBaseComponent,
    expected_interfaces: List[Type],
) -> List[str]:
    """Return a list of error strings if *instance* is not an instance of each protocol."""
    errors: List[str] = []
    for proto in expected_interfaces:
        try:
            if not isinstance(instance, proto):
                errors.append(f"{instance.component_id} does not implement {proto.__name__}")
        except Exception as exc:  # pragma: no cover
            errors.append(f"Interface validation failed for {proto}: {exc}")
    return errors


###############################################################################
# 5. Logging helpers
###############################################################################

class _PrefixFilter(logging.Filter):
    """Inject a static prefix into every log record."""

    def __init__(self, prefix: str) -> None:
        self._prefix = prefix
        super().__init__()

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: D401
        record.msg = f"{self._prefix}{record.msg}"
        return True


def configure_component_logging(
    instance: NireonBaseComponent,
    log_level: Optional[str] = None,
    log_prefix: Optional[str] = None,
) -> None:
    """Apply runtime log settings to *instance*."""
    if hasattr(instance, "_configure_logging") and callable(instance._configure_logging):
        # Delegate to component implementation if provided
        instance._configure_logging(log_level=log_level, log_prefix=log_prefix)
        logger.debug("Delegated logging config for %s", instance.component_id)
        return

    if not hasattr(instance, "logger") or not isinstance(instance.logger, logging.Logger):
        return  # Component does not expose a logger

    comp_logger: logging.Logger = instance.logger
    if log_level:
        level_val = getattr(logging, log_level.upper(), None)
        if isinstance(level_val, int):
            comp_logger.setLevel(level_val)

    if log_prefix:
        # Clear any existing prefix filters to avoid duplication
        comp_logger.filters = [f for f in comp_logger.filters if not isinstance(f, _PrefixFilter)]
        comp_logger.addFilter(_PrefixFilter(log_prefix))


###############################################################################
# 6. Metadata preparation
###############################################################################

def prepare_component_metadata(
    base_metadata: ComponentMetadata,
    instance_id: str,
    config_overrides: Optional[Dict[str, Any]] = None,
) -> ComponentMetadata:
    """Return a ComponentMetadata for *instance_id*, applying override dicts if provided."""
    if not isinstance(base_metadata, ComponentMetadata):
        # Attempt coercion or fallback
        if isinstance(base_metadata, dict) and {"id", "name"} <= base_metadata.keys():
            base_metadata = ComponentMetadata(**base_metadata)  # type: ignore[arg-type]
        else:
            base_metadata = ComponentMetadata(
                id=instance_id,
                name=instance_id,
                version="0.0.0",
                category="unknown",
            )

    meta = dataclasses.replace(base_metadata, id=instance_id)

    if config_overrides:
        overrides = config_overrides.get("metadata_override", {})
        if overrides:
            try:
                meta_dict = dataclasses.asdict(meta)
                meta_dict.update(overrides)
                meta = ComponentMetadata(**meta_dict)  # type: ignore[arg-type]
            except Exception as exc:  # pragma: no cover
                logger.error("Failed metadata_override for %s: %s", instance_id, exc)

    return meta
