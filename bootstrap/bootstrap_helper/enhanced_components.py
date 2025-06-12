"""
Bootstrap helpers for *enhanced‑manifest* components (v4.1)
==========================================================

This module offers three public helpers that are consumed by the main bootstrap
pipeline:

* :func:`_get_pydantic_defaults` – discover Pydantic default config for a
  component or factory key.
* :func:`_create_component_instance` – instantiate a concrete component
  given fully‑resolved metadata & config.
* :func:`init_full_component` – create **and register** a component based on
  the *enhanced* manifest spec (initialisation is deferred).

The public API is **100 % backwards‑compatible** with the legacy v4.0
implementation; only internal structure, typing, and logging have changed.
"""

from __future__ import annotations

import dataclasses
import importlib
import inspect
import json
import logging
import re
import types
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Type, Union, cast

from pydantic import BaseModel

from bootstrap.bootstrap_helper.placeholders import PlaceholderEventBusImpl  # type: ignore
from bootstrap.health.reporter import BootstrapHealthReporter, ComponentStatus
from configs.config_utils import ConfigMerger
from core.base_component import NireonBaseComponent
from core.lifecycle import (
    ComponentMetadata,
    ComponentRegistryMissingError,
)
from core.registry import ComponentRegistry
from domain.context import NireonExecutionContext
from domain.ports.event_bus_port import EventBusPort
from factories.dependencies import CommonMechanismDependencies
from runtime.utils import import_by_path, load_yaml_robust

from ._exceptions import BootstrapError
from .service_resolver import _safe_register_service_instance

logger = logging.getLogger(__name__)

__all__ = (
    "_get_pydantic_defaults",
    "_create_component_instance",
    "init_full_component",
    # Additional helpers from attachment 2
    "_inject_dependencies",
    "_validate_component_interfaces",
    "_configure_component_logging",
    "_prepare_component_metadata",
)

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# Helper – Pydantic defaults discovery
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::


def _get_pydantic_defaults(
    component_or_key: Union[Type, str],
    label: str,
) -> Dict[str, Any]:
    """
    Return default‑value dict from a Pydantic *ConfigModel* associated with a
    component **class** (preferred) or, if `component_or_key` is a factory key,
    an empty dict.

    Conventions checked in order:

    1. Nested ``ConfigModel`` attribute on the class.
    2. `f"{ClassName}Config"` inside ``<module>.config``.
    3. `f"{ClassName}Config"` in the same module.
    4. `f"{ClassName}Config`` in ``<parent>.config``.
    """
    logger.debug("Discovering Pydantic defaults for '%s' (%s)", label, component_or_key)

    if isinstance(component_or_key, str):
        # Factory key – let the factory worry about defaults.
        return {}

    cls = component_or_key
    config_cls: Optional[Type[BaseModel]] = None

    # 1) Nested model
    if hasattr(cls, "ConfigModel") and inspect.isclass(cls.ConfigModel) and issubclass(cls.ConfigModel, BaseModel):  # type: ignore[attr-defined]
        config_cls = cls.ConfigModel  # type: ignore[assignment]
    else:
        # helpers
        def _import_candidate(module_path: str) -> Optional[Type[BaseModel]]:
            try:
                mod = importlib.import_module(module_path)
                cand = getattr(mod, f"{cls.__name__}Config", None)
                if inspect.isclass(cand) and issubclass(cand, BaseModel):
                    return cast(Type[BaseModel], cand)
            except ImportError:
                pass
            return None

        module_path = cls.__module__
        candidates = [f"{module_path}.config", module_path]
        if "." in module_path:
            candidates.append(f"{module_path.rsplit('.', 1)[0]}.config")

        for mod_path in candidates:
            config_cls = _import_candidate(mod_path)
            if config_cls:
                break

    if not config_cls:
        return {}

    try:
        return config_cls.model_construct().model_dump()
    except Exception as exc:
        logger.warning("Could not extract defaults from %s for '%s': %s", config_cls.__name__, label, exc)
        return {}

def _get_pydantic_defaults(component_class: Type, component_name: str) -> Dict[str, Any]:
    """Extract default values from Pydantic models associated with a component class."""
    
    defaults = {}
    
    try:
        # Check for ConfigModel attribute
        if hasattr(component_class, 'ConfigModel'):
            config_model = component_class.ConfigModel
            
            # Pydantic v2 style
            if hasattr(config_model, 'model_fields'):
                for field_name, field_info in config_model.model_fields.items():
                    if hasattr(field_info, 'default') and field_info.default is not ...:
                        defaults[field_name] = field_info.default
                    elif hasattr(field_info, 'default_factory') and field_info.default_factory is not None:
                        try:
                            defaults[field_name] = field_info.default_factory()
                        except Exception:
                            pass
            # Pydantic v1 style
            elif hasattr(config_model, '__fields__'):
                for field_name, field in config_model.__fields__.items():
                    if hasattr(field, 'default') and field.default is not ...:
                        defaults[field_name] = field.default
                    elif hasattr(field, 'default_factory') and field.default_factory is not None:
                        try:
                            defaults[field_name] = field.default_factory()
                        except Exception:
                            pass

        # Check for DEFAULT_CONFIG class attribute
        if hasattr(component_class, 'DEFAULT_CONFIG'):
            defaults.update(component_class.DEFAULT_CONFIG)

        if defaults:
            logger.debug(f'Found {len(defaults)} Pydantic defaults for {component_name}')
        else:
            logger.debug(f'No Pydantic defaults found for {component_name}')

    except Exception as e:
        logger.debug(f'Error extracting Pydantic defaults for {component_name}: {e}')

    return defaults



# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# Helper – Component instantiation
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::


async def _create_component_instance(
    component_class: Type,
    resolved_config: Dict[str, Any],
    instance_id: str,
    metadata: ComponentMetadata,
    common_deps: Optional[CommonMechanismDependencies] = None,
) -> Optional[NireonBaseComponent]:
    """
    Instantiate *component_class* with its resolved *metadata* & *config*.

    Constructor argument resolution order (same as legacy):

    1. ``config`` and ``metadata_definition`` if accepted.
    2. Dependency‑injection via ``common_deps`` (llm, embed, event_bus, …).
    3. Fallbacks: ``component_class(config=…)`` then ``component_class()``.
    """
    logger.debug(
        "[_create_component_instance] id=%s class=%s",
        instance_id,
        component_class.__name__,
    )

    if metadata.id != instance_id:
        metadata = replace(metadata, id=instance_id)

    sig = inspect.signature(component_class.__init__)
    kwargs: Dict[str, Any] = {}

    if "config" in sig.parameters:
        kwargs["config"] = resolved_config
    if "metadata_definition" in sig.parameters:
        kwargs["metadata_definition"] = metadata

    # Dependency injection
    if common_deps:
        mapping = {
            "common_deps": common_deps,
            "llm": getattr(common_deps, "llm_port", None),
            "embedding_port": getattr(common_deps, "embedding_port", None),
            "event_bus": getattr(common_deps, "event_bus", None),
        }
        for param, value in mapping.items():
            if value is not None and param in sig.parameters:
                kwargs.setdefault(param, value)

    # Primary attempt
    try:
        instance = component_class(**kwargs)
    except TypeError as err:
        logger.error("Constructor mismatch for '%s': %s", instance_id, err, exc_info=True)
        # Fallbacks
        for fallback_kwargs in ({"config": resolved_config}, {}):
            try:
                return component_class(**fallback_kwargs)
            except TypeError:
                continue
        raise BootstrapError(f"Instantiation failed for '{instance_id}': {err}") from err

    # Force‑synchronise IDs & metadata on NireonBaseComponent
    if isinstance(instance, NireonBaseComponent):
        if getattr(instance, "component_id", None) != instance_id:
            object.__setattr__(instance, "_component_id", instance_id)
        if getattr(instance, "metadata", None) is None or instance.metadata.id != instance_id:  # type: ignore[attr-defined]
            object.__setattr__(instance, "_metadata_definition", metadata)

    return cast(NireonBaseComponent, instance)


# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# Public – initialise & register component (no `.initialize()` called)
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::


async def init_full_component(  # noqa: D401
    cid_manifest: str,
    spec: Mapping[str, Any],
    registry: ComponentRegistry,
    event_bus: EventBusPort,
    run_id: str,
    replay_context: bool,
    common_deps: Optional[CommonMechanismDependencies] = None,
    global_app_config: Optional[Dict[str, Any]] = None,
    health_reporter: Optional[BootstrapHealthReporter] = None,
    validation_data_store: Any = None,
) -> None:
    """
    Create *and register* a component described by an **enhanced** manifest spec.

    Steps (same semantics as v4.0):

    1. Import target class & canonical metadata.
    2. Merge config layers: Pydantic defaults → YAML → inline override.
    3. Instantiate component (but **do not** call `.initialize()`).
    4. Register with :class:`~core.lifecycle.ComponentRegistry`.
    """
    cfg = global_app_config or {}
    strict = cfg.get("bootstrap_strict_mode", True)

    # -------- 0. early exits / bookkeeping --------------------------------
    if not spec.get("enabled", True):
        logger.info("Component '%s' disabled in manifest – skipping.", cid_manifest)
        if health_reporter:
            health_reporter.add_component_status(cid_manifest, ComponentStatus.INSTANCE_REGISTERED_NO_INIT)
        return

    class_path = spec.get("class")
    meta_path = spec.get("metadata_definition")
    if not class_path or not meta_path:
        msg = f"Component '{cid_manifest}' missing 'class' or 'metadata_definition'."
        _report_and_raise(msg, cid_manifest, health_reporter, strict, ComponentStatus.DEFINITION_ERROR)
        return

    # -------- 1. import class & canonical metadata ------------------------
    try:
        comp_cls = _import_class(class_path, cid_manifest)
        canonical_meta = _import_metadata(meta_path, cid_manifest)
    except BootstrapError as exc:
        _report_and_raise(str(exc), cid_manifest, health_reporter, strict, ComponentStatus.DEFINITION_ERROR)
        return

    # -------- 2. build instance metadata ----------------------------------
    inst_meta = _build_instance_metadata(canonical_meta, cid_manifest, spec)

    # -------- 3. resolve configuration ------------------------------------
    pyd_defaults = _get_pydantic_defaults(comp_cls, inst_meta.name)
    yaml_cfg = _load_yaml_layer(spec.get("config"), cid_manifest)
    inline_override = spec.get("config_override", {})

    merged_cfg = ConfigMerger.merge(pyd_defaults, yaml_cfg, f"{cid_manifest}_pyd_yaml")
    resolved_cfg = ConfigMerger.merge(merged_cfg, inline_override, f"{cid_manifest}_final")

    logger.info("[%s] Resolved config keys: %s", cid_manifest, list(resolved_cfg))

    if validation_data_store:
        validation_data_store.store_component_data(  # type: ignore[attr-defined]
            component_id=cid_manifest,
            original_metadata=inst_meta,
            resolved_config=resolved_cfg,
            manifest_spec=spec,
        )

    # -------- 4. instantiate ---------------------------------------------
    try:
        instance = await _create_component_instance(
            comp_cls,
            resolved_cfg,
            cid_manifest,
            inst_meta,
            common_deps=common_deps,
        )
    except BootstrapError as exc:
        _report_and_raise(str(exc), cid_manifest, health_reporter, strict, ComponentStatus.INSTANTIATION_ERROR, inst_meta)
        return

    if instance is None:
        _report_and_raise(
            f"Instantiation for '{cid_manifest}' returned None.",
            cid_manifest,
            health_reporter,
            strict,
            ComponentStatus.INSTANTIATION_ERROR,
            inst_meta,
        )
        return

    # -------- 5. register (initialisation deferred) ----------------------
    try:
        _safe_register_service_instance(
            registry,
            type(instance),
            instance,
            instance.component_id,
            instance.metadata.category,
            description_for_meta=instance.metadata.description,
            requires_initialize_override=instance.metadata.requires_initialize,
        )
        if health_reporter:
            health_reporter.add_component_status(
                instance.component_id,
                ComponentStatus.INSTANCE_REGISTERED_INIT_DEFERRED,
                instance.metadata,
            )
        logger.info("✓ Registered component '%s' (requires_init=%s)", instance.component_id, instance.metadata.requires_initialize)
    except Exception as exc:
        _report_and_raise(str(exc), cid_manifest, health_reporter, strict, ComponentStatus.BOOTSTRAP_ERROR, inst_meta)


# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# Additional helpers from attachment 2
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::


def _inject_dependencies(
    instance: NireonBaseComponent,
    dependency_map: Dict[str, Any],
    registry: Optional[Any] = None
) -> None:
    """Inject dependencies into a component instance."""
    
    try:
        for dep_name, dep_value in dependency_map.items():
            if hasattr(instance, f'_{dep_name}'):
                setattr(instance, f'_{dep_name}', dep_value)
                logger.debug(f"Injected dependency '{dep_name}' into {instance.component_id}")
            elif hasattr(instance, dep_name):
                setattr(instance, dep_name, dep_value)
                logger.debug(f"Injected dependency '{dep_name}' into {instance.component_id}")
    except Exception as e:
        logger.warning(f'Failed to inject dependencies into {instance.component_id}: {e}')


def _validate_component_interfaces(
    instance: NireonBaseComponent,
    expected_interfaces: List[Type]
) -> List[str]:
    """Validate that a component implements expected interfaces."""
    
    errors = []
    try:
        for interface in expected_interfaces:
            if not isinstance(instance, interface):
                errors.append(f'Component {instance.component_id} does not implement {interface.__name__}')
    except Exception as e:
        errors.append(f'Interface validation failed for {instance.component_id}: {e}')
    
    return errors


def _configure_component_logging(
    instance: NireonBaseComponent,
    log_level: Optional[str] = None,
    log_prefix: Optional[str] = None
) -> None:
    """Configure logging for a component instance."""
    
    try:
        if hasattr(instance, '_configure_logging'):
            instance._configure_logging(log_level=log_level, log_prefix=log_prefix)
        elif hasattr(instance, 'logger'):
            if log_level:
                instance.logger.setLevel(getattr(logging, log_level.upper()))
    except Exception as e:
        logger.warning(f'Failed to configure logging for {instance.component_id}: {e}')


def _prepare_component_metadata(
    base_metadata: ComponentMetadata,
    instance_id: str,
    config_overrides: Optional[Dict[str, Any]] = None
) -> ComponentMetadata:
    """Prepare component metadata for an instance."""
    
    instance_metadata = dataclasses.replace(base_metadata, id=instance_id)
    
    if config_overrides:
        metadata_overrides = config_overrides.get('metadata_override', {})
        if metadata_overrides:
            metadata_dict = dataclasses.asdict(instance_metadata)
            metadata_dict.update(metadata_overrides)
            instance_metadata = ComponentMetadata(**metadata_dict)
    
    return instance_metadata


# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# Internal helpers (not exported)
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::


def _import_class(path: str, cid: str) -> Type:
    obj = import_by_path(path)
    if not inspect.isclass(obj):
        raise BootstrapError(f"'{path}' for '{cid}' is not a class.")
    return cast(Type, obj)


def _import_metadata(path: str, cid: str) -> ComponentMetadata:
    obj = import_by_path(path)
    if not isinstance(obj, ComponentMetadata):
        raise BootstrapError(f"'{path}' for '{cid}' is not a ComponentMetadata instance.")
    return obj


def _build_instance_metadata(canonical: ComponentMetadata, cid: str, spec: Mapping[str, Any]) -> ComponentMetadata:
    data = asdict(canonical)
    data["id"] = cid
    data.update(spec.get("metadata_override", {}))

    if "epistemic_tags" in spec and isinstance(spec["epistemic_tags"], list):
        data["epistemic_tags"] = list(spec["epistemic_tags"])

    data.setdefault("requires_initialize", canonical.requires_initialize)
    return ComponentMetadata(**data)  # type: ignore[arg-type]


def _load_yaml_layer(path_template: Optional[str], cid: str) -> Dict[str, Any]:
    if not path_template:
        return {}
    try:
        actual = Path(path_template.replace("{id}", cid))
        return load_yaml_robust(actual)
    except Exception as exc:
        logger.warning("Failed to load YAML config for '%s': %s", cid, exc)
        return {}


def _report_and_raise(
    msg: str,
    cid: str,
    reporter: Optional[BootstrapHealthReporter],
    strict: bool,
    status: ComponentStatus,
    meta: Optional[ComponentMetadata] = None,
) -> None:
    logger.error(msg)
    if reporter:
        reporter.add_component_status(cid, status, meta or ComponentMetadata(id=cid, name=cid, version="0.0.0", category="unknown"), [msg])
    if strict:
        raise BootstrapError(msg)


# Regex pre‑compilation example kept for future extensions (possible ID validation)
_ID_RE = re.compile(r"^[A-Za-z0-9_\-\.]+$")