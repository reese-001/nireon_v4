# nireon_v4/bootstrap/bootstrap_helper/enhanced_components.py
from __future__ import annotations
from __future__ import absolute_import

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

from pydantic import BaseModel # Keep for potential local models if any

# Import the consolidated helper functions
from bootstrap.utils.component_utils import (
    get_pydantic_defaults,
    create_component_instance,
    # The following are not directly used in init_full_component but were in the original file.
    # If init_full_component logic expands to use them, they are available.
    # inject_dependencies,
    # validate_component_interfaces,
    # configure_component_logging,
    # prepare_component_metadata,
)

from bootstrap.bootstrap_helper.placeholders import PlaceholderEventBusImpl # Used by _report_and_raise if reporter is None
from bootstrap.health.reporter import BootstrapHealthReporter, ComponentStatus
from configs.config_utils import ConfigMerger # Assuming this is available
from core.base_component import NireonBaseComponent
from core.lifecycle import ComponentMetadata, ComponentRegistryMissingError
from core.registry import ComponentRegistry
from domain.context import NireonExecutionContext # Not directly used here, but often related
from domain.ports.event_bus_port import EventBusPort
from factories.dependencies import CommonMechanismDependencies # Used in init_full_component
from runtime.utils import import_by_path, load_yaml_robust
from ._exceptions import BootstrapError # Relative import within bootstrap_helper
from bootstrap.processors.service_resolver import _safe_register_service_instance # Relative import

logger = logging.getLogger(__name__)

# This __all__ should reflect what this module now primarily offers, which is init_full_component
__all__ = (
    'init_full_component',
    # Keep original private-like names if they are part of an implicit API used elsewhere,
    # though ideally, direct consumers would use the public versions from utils.
    # For now, focusing on init_full_component as the main export.
)

# _ID_RE is used by init_full_component locally, so it stays.
_ID_RE = re.compile('^[A-Za-z0-9_\\-\\.]+$')


async def init_full_component(
    cid_manifest: str,
    spec: Mapping[str, Any],
    registry: ComponentRegistry,
    event_bus: EventBusPort, # Assuming this is the actual EventBusPort, not a placeholder type
    run_id: str,
    replay_context: bool, # This parameter is not used in the current implementation
    common_deps: Optional[CommonMechanismDependencies] = None,
    global_app_config: Optional[Dict[str, Any]] = None,
    health_reporter: Optional[BootstrapHealthReporter] = None,
    validation_data_store: Any = None # e.g., BootstrapValidationData
) -> None:
    cfg = global_app_config or {}
    strict = cfg.get('bootstrap_strict_mode', True)

    if not _ID_RE.match(cid_manifest):
        msg = f"Component ID '{cid_manifest}' contains invalid characters. Must match '{_ID_RE.pattern}'."
        _report_and_raise(msg, cid_manifest, health_reporter, strict, ComponentStatus.DEFINITION_ERROR)
        return

    if not spec.get('enabled', True):
        logger.info("Component '%s' disabled in manifest - skipping.", cid_manifest)
        if health_reporter:
            # Create a minimal metadata for reporting disabled status
            minimal_meta = ComponentMetadata(id=cid_manifest, name=cid_manifest, version='N/A', category='disabled')
            health_reporter.add_component_status(cid_manifest, ComponentStatus.DISABLED, minimal_meta)
        return

    class_path = spec.get('class')
    meta_path = spec.get('metadata_definition')

    if not class_path: # meta_path can be optional if class provides METADATA_DEFINITION
        msg = f"Component '{cid_manifest}' missing 'class' path in manifest spec."
        _report_and_raise(msg, cid_manifest, health_reporter, strict, ComponentStatus.DEFINITION_ERROR)
        return

    try:
        comp_cls = _import_class(class_path, cid_manifest)
        
        # Resolve canonical_meta: from path, or from class, or fallback
        if meta_path:
            canonical_meta = _import_metadata(meta_path, cid_manifest)
        elif hasattr(comp_cls, 'METADATA_DEFINITION') and isinstance(comp_cls.METADATA_DEFINITION, ComponentMetadata):
            canonical_meta = comp_cls.METADATA_DEFINITION
            logger.debug(f"Using METADATA_DEFINITION from class {comp_cls.__name__} for '{cid_manifest}'.")
        else:
            msg = (f"Component '{cid_manifest}' (class: {class_path}) missing 'metadata_definition' path in spec "
                   f"and class does not have a 'METADATA_DEFINITION' attribute.")
            _report_and_raise(msg, cid_manifest, health_reporter, strict, ComponentStatus.DEFINITION_ERROR)
            return
            
    except BootstrapError as exc:
        _report_and_raise(str(exc), cid_manifest, health_reporter, strict, ComponentStatus.DEFINITION_ERROR)
        return
    except Exception as exc: # Catch other import errors too
        _report_and_raise(f"Failed to import class/metadata for '{cid_manifest}': {exc}",
                          cid_manifest, health_reporter, strict, ComponentStatus.DEFINITION_ERROR)
        return

    inst_meta = _build_instance_metadata(canonical_meta, cid_manifest, spec)
    
    # Use the consolidated get_pydantic_defaults
    pyd_defaults = get_pydantic_defaults(comp_cls, inst_meta.name)
    
    yaml_cfg = _load_yaml_layer(spec.get('config'), cid_manifest)
    inline_override = spec.get('config_override', {})
    
    merged_cfg = ConfigMerger.merge(pyd_defaults, yaml_cfg, f'{cid_manifest}_pyd_yaml')
    resolved_cfg = ConfigMerger.merge(merged_cfg, inline_override, f'{cid_manifest}_final_config')
    
    logger.info('[%s] Resolved config keys: %s', cid_manifest, list(resolved_cfg.keys()))

    if validation_data_store and hasattr(validation_data_store, 'store_component_data'):
        validation_data_store.store_component_data(
            component_id=cid_manifest,
            original_metadata=inst_meta, # This should be the version before instance overrides
            resolved_config=resolved_cfg,
            manifest_spec=spec
        )
    
    # Adapt common_deps for the new create_component_instance signature
    common_deps_dict: Optional[Dict[str, Any]] = None
    if common_deps:
        common_deps_dict = {
            # Map attributes of CommonMechanismDependencies to expected constructor param names
            'llm_port': getattr(common_deps, 'llm_port', None), # or 'llm' if that's the param name
            'embedding_port': getattr(common_deps, 'embedding_port', None),
            'event_bus': getattr(common_deps, 'event_bus', None),
            'idea_service': getattr(common_deps, 'idea_service', None),
            'component_registry': getattr(common_deps, 'component_registry', None),
            'rng': getattr(common_deps, 'rng', None),
            # Add 'common_deps' itself if some components expect the whole object
            'common_deps': common_deps 
        }

    instance: Optional[NireonBaseComponent] = None
    try:
        # Use the consolidated create_component_instance
        instance = await create_component_instance(
            comp_cls, 
            resolved_cfg, 
            cid_manifest, 
            inst_meta, 
            common_deps=common_deps_dict
        )
    except BootstrapError as exc:
        _report_and_raise(str(exc), cid_manifest, health_reporter, strict, ComponentStatus.INSTANTIATION_ERROR, inst_meta)
        return
    except Exception as exc: # Catch unexpected instantiation errors
        _report_and_raise(f"Unexpected error during instantiation of '{cid_manifest}': {exc}", 
                          cid_manifest, health_reporter, strict, ComponentStatus.INSTANTIATION_ERROR, inst_meta)
        return

    if instance is None: # Should be caught by create_component_instance raising an error, but as a safeguard.
        _report_and_raise(f"Instantiation for '{cid_manifest}' returned None.", 
                          cid_manifest, health_reporter, strict, ComponentStatus.INSTANTIATION_ERROR, inst_meta)
        return

    try:
        # Ensure the instance has its metadata correctly set, especially the ID.
        # create_component_instance should handle this, but double check.
        if not hasattr(instance, 'metadata') or instance.metadata.id != cid_manifest:
             # This suggests an issue in create_component_instance or component's __init__
            logger.warning(f"Metadata ID mismatch for '{cid_manifest}' post-instantiation. Instance has: {getattr(instance, 'metadata', None)}. Attempting to fix.")
            if hasattr(instance, '_metadata_definition'):
                 object.__setattr__(instance, '_metadata_definition', inst_meta)
            elif hasattr(instance, 'metadata'):
                 object.__setattr__(instance, 'metadata', inst_meta)


        _safe_register_service_instance(
            registry,
            type(instance),
            instance,
            instance.component_id, # Use ID from instance, should match cid_manifest
            instance.metadata.category,
            description_for_meta=instance.metadata.description,
            requires_initialize_override=instance.metadata.requires_initialize
        )
        if health_reporter:
            health_reporter.add_component_status(
                instance.component_id,
                ComponentStatus.INSTANCE_REGISTERED_INIT_DEFERRED, # Standard status post-registration
                instance.metadata
            )
        logger.info("✓ Registered component '%s' (type: %s, category: %s, requires_init=%s)",
                    instance.component_id, type(instance).__name__, instance.metadata.category, instance.metadata.requires_initialize)
    except Exception as exc:
        _report_and_raise(f"Error during registration of '{cid_manifest}': {exc}", 
                          cid_manifest, health_reporter, strict, ComponentStatus.REGISTRATION_ERROR, inst_meta)


# Helper functions that were part of the original module, kept for init_full_component
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
    data['id'] = cid # Ensure ID is set to the manifest component ID
    
    # Apply metadata_override from the spec
    metadata_override = spec.get('metadata_override', {})
    if metadata_override:
        logger.debug(f"Applying metadata_override for '{cid}': {metadata_override}")
        data.update(metadata_override)

    # Specific handling for epistemic_tags if present directly in spec
    if 'epistemic_tags' in spec and isinstance(spec['epistemic_tags'], list):
        data['epistemic_tags'] = list(spec['epistemic_tags'])
    
    # Ensure requires_initialize is correctly sourced
    # Priority: spec override > canonical metadata
    if 'requires_initialize' in metadata_override: # If it was in metadata_override
        data['requires_initialize'] = metadata_override['requires_initialize']
    elif 'requires_initialize' in spec: # If directly in spec (less common but possible)
         data['requires_initialize'] = spec['requires_initialize']
    else: # Fallback to canonical
        data.setdefault('requires_initialize', canonical.requires_initialize)
        
    try:
        return ComponentMetadata(**data)
    except TypeError as e:
        raise BootstrapError(f"Failed to create ComponentMetadata for '{cid}' with data {data}: {e}")


def _load_yaml_layer(path_template: Optional[Union[str, Dict[str, Any]]], cid: str) -> Dict[str, Any]:
    if not path_template:
        return {}
    
    if isinstance(path_template, dict): # Inline YAML config
        logger.debug(f"Using inline YAML config for '{cid}'")
        return path_template

    if isinstance(path_template, str):
        try:
            actual_path_str = path_template.replace('{id}', cid)
            actual_path = Path(actual_path_str)
            if actual_path.exists():
                return load_yaml_robust(actual_path)
            else:
                logger.debug(f"YAML config file not found for '{cid}' at '{actual_path_str}'. Skipping.")
                return {}
        except Exception as exc:
            logger.warning("Failed to load YAML config for '%s' from template '%s': %s", cid, path_template, exc)
            return {}
    
    logger.warning(f"Invalid 'config' type for '{cid}': {type(path_template)}. Expected string path or dict.")
    return {}


def _report_and_raise(
    msg: str,
    cid: str,
    reporter: Optional[BootstrapHealthReporter],
    strict: bool,
    status: ComponentStatus,
    meta: Optional[ComponentMetadata] = None
) -> None:
    logger.error(f"Bootstrap error for '{cid}': {msg}")
    if reporter:
        # If meta is not provided, create a minimal one for reporting
        if meta is None:
            meta = ComponentMetadata(id=cid, name=cid, version='0.0.0', category='unknown_error_state')
        reporter.add_component_status(cid, status, meta, [msg])
    if strict:
        raise BootstrapError(msg, component_id=cid)