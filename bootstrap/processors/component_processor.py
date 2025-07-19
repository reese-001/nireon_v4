from __future__ import absolute_import

import dataclasses
import inspect
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Type
import asyncio

from core.base_component import NireonBaseComponent
from core.lifecycle import ComponentMetadata, ComponentRegistryMissingError
from core.registry.component_registry import ComponentRegistry
from domain.ports.event_bus_port import EventBusPort
from factories.mechanism_factory import SimpleMechanismFactory
from runtime.utils import import_by_path, load_yaml_robust
from bootstrap.bootstrap_helper._exceptions import BootstrapError
from bootstrap.health.reporter import ComponentStatus, HealthReporter as BootstrapHealthReporter
from bootstrap.processors.metadata import get_default_metadata
from bootstrap.processors.service_resolver import _safe_register_service_instance
from bootstrap.utils.component_utils import create_component_instance, get_pydantic_defaults

# Fix: Added import for _expand_tree
from configs.config_loader import _expand_tree

logger = logging.getLogger(__name__)

try:
    from configs.config_utils import ConfigMerger
except ImportError:
    logger.warning('ConfigMerger not found; using fallback merge.')
    class ConfigMerger:
        @staticmethod
        def merge(d1: Mapping, d2: Mapping, _: str='') -> Dict:
            merged = dict(d1)
            merged.update(d2)
            return merged

_YAML_PARAMETERS_KEY = 'parameters'


def _load_config(source: Any, component_id: str) -> Dict[str, Any]:
    if source is None:
        return {}
    if isinstance(source, str):
        path = Path(source.replace('{id}', component_id))
        return load_yaml_robust(path)
    if isinstance(source, Mapping):
        return dict(source)

    logger.warning("Unexpected config type for '%s': %s - ignoring.", component_id, type(source))
    return {}

def _merge_configs(pydantic_defaults: Dict[str, Any], yaml_cfg: Dict[str, Any], override: Dict[str, Any], context: str) -> Dict[str, Any]:
    step1 = ConfigMerger.merge(pydantic_defaults, yaml_cfg, f'{context}_defaults_yaml')
    return ConfigMerger.merge(step1, override, f'{context}_final')

def _apply_meta_override(base_meta: ComponentMetadata, overrides: Mapping[str, Any]) -> ComponentMetadata:
    if not overrides:
        return base_meta
    meta_dict = dataclasses.asdict(base_meta)
    meta_dict.update(overrides)
    try:
        return ComponentMetadata(**meta_dict)
    except TypeError as exc:
        logger.warning('Metadata override error: %s', exc)
        return base_meta

def _store_validation(data_store: Any, *, component_id: str, metadata: ComponentMetadata, config: Dict[str, Any], manifest_spec: Dict[str, Any]) -> None:
    if data_store and hasattr(data_store, 'store_component_data'):
        data_store.store_component_data(
            component_id=component_id,
            original_metadata=metadata,
            resolved_config=config,
            manifest_spec=manifest_spec,
        )

async def process_simple_component(
    comp_def: Dict[str, Any],
    registry: ComponentRegistry,
    mechanism_factory: Optional[SimpleMechanismFactory],
    health_reporter: BootstrapHealthReporter,
    run_id: str,
    global_app_config: Dict[str, Any],
    validation_data_store: Any,
) -> None:
    cid: str | None = comp_def.get('component_id')
    factory_key = comp_def.get('factory_key')
    class_path = comp_def.get('class')
    ctype = comp_def.get('type')
    enabled: bool = comp_def.get('enabled', True)
    strict: bool = global_app_config.get('bootstrap_strict_mode', True)

    meta_report = ComponentMetadata(
        id=cid or 'unknown_simple',
        name=factory_key or class_path or 'unknown',
        category=ctype or 'unknown_type',
        version='0.0.0',
        requires_initialize=True,
    )

    if not enabled:
        _mark_disabled(health_reporter, cid, meta_report)
        return

    if not cid or not (factory_key or class_path) or (not ctype):
        _def_error('Missing required fields', health_reporter, cid, meta_report, strict)
        return

    base_meta = (
        get_default_metadata(factory_key)
        if factory_key else _try_class_metadata(class_path)
    ) or ComponentMetadata(
        id=cid,
        name=cid,
        version='0.1.0',
        category=ctype,
        description=f"Auto-generated metadata for {ctype} '{cid}'",
    )
    
    try:
        final_meta = _build_component_metadata(base_meta, cid, comp_def)
    except Exception as exc:
        _meta_error(f'Metadata build failed: {exc}', health_reporter, cid, meta_report, strict, exc)
        return

    cls_for_defaults: Optional[Type] = None
    if class_path:
        try:
            cls_for_defaults = import_by_path(class_path)
        except Exception:
            logger.debug('Could not import %s for pydantic defaults.', class_path)

    pydantic_defaults = get_pydantic_defaults(cls_for_defaults, final_meta.name) if cls_for_defaults else {}
    yaml_cfg = _load_config(comp_def.get('config'), cid)
    if _YAML_PARAMETERS_KEY in yaml_cfg:
        yaml_cfg = yaml_cfg[_YAML_PARAMETERS_KEY]

    cfg_override = comp_def.get('config_override', {})
    
    final_cfg_unexpanded = _merge_configs(pydantic_defaults, yaml_cfg, cfg_override, cid)
    # Expand environment variables here as well for consistency
    final_cfg = _expand_tree(final_cfg_unexpanded)

    _store_validation(
        validation_data_store,
        component_id=cid,
        metadata=final_meta,
        config=final_cfg,
        manifest_spec=comp_def,
    )

    try:
        instance = await _instantiate_simple_component(
            cid, class_path, cls_for_defaults, factory_key, ctype,
            mechanism_factory, final_meta, final_cfg
        )
    except Exception as exc:
        _inst_error(f'Instantiation error: {exc}', health_reporter, cid, final_meta, strict, exc)
        return

    try:
        _safe_register_service_instance(
            registry,
            type(instance),
            instance,
            cid,
            final_meta.category,
            description_for_meta=final_meta.description,
            requires_initialize_override=final_meta.requires_initialize
        )
        health_reporter.add_component_status(cid, ComponentStatus.INSTANCE_REGISTERED, instance.metadata, [])
        logger.info("✓ Registered simple component '%s'", cid)
    except Exception as exc:
        _inst_error(f'Registry error: {exc}', health_reporter, cid, final_meta, strict, exc)


async def _instantiate_simple_component(
    cid: str,
    class_path: Optional[str],
    cls_for_defaults: Optional[Type],
    factory_key: Optional[str],
    ctype: str,
    mechanism_factory: Optional[SimpleMechanismFactory],
    metadata: ComponentMetadata,
    cfg: Dict[str, Any]
) -> NireonBaseComponent:
    if class_path:
        cls_to_use = cls_for_defaults or import_by_path(class_path)
        if not issubclass(cls_to_use, NireonBaseComponent):
            raise BootstrapError(f"Class {class_path} for '{cid}' is not a NireonBaseComponent.", component_id=cid)
        return await create_component_instance(cls_to_use, cfg, cid, metadata, common_deps=None)
    
    if factory_key and ctype == 'mechanism' and mechanism_factory:
        instance = mechanism_factory.create_mechanism(factory_key, metadata, cfg)
        if instance is None:
            raise BootstrapError(f"Factory returned None for '{cid}' / key '{factory_key}'", component_id=cid)
        return instance
    
    raise BootstrapError(f"Cannot determine instantiation method for '{cid}'.", component_id=cid)


async def instantiate_shared_service(
    service_key_in_manifest: str,
    service_spec_from_manifest: Dict[str, Any],
    context: Any
) -> None:
    registry = context.registry
    event_bus = context.event_bus
    global_app_config = context.global_app_config
    health_reporter = context.health_reporter
    validation_data_store = getattr(context, 'validation_data_store', None)
    registry_manager = getattr(context, 'registry_manager', None)
    
    class_path = service_spec_from_manifest.get('class')
    enabled: bool = service_spec_from_manifest.get('enabled', True)
    strict: bool = global_app_config.get('bootstrap_strict_mode', True)
    port_type = service_spec_from_manifest.get('port_type')

    base_meta = ComponentMetadata(
        id=service_key_in_manifest,
        name=class_path.split(':')[-1] if class_path and ':' in class_path else (class_path or service_key_in_manifest).split('.')[-1],
        category='shared_service',
        version='0.1.0'
    )
    
    if not enabled:
        _mark_disabled(health_reporter, service_key_in_manifest, base_meta)
        return

    if not class_path:
        _def_error("Shared service definition missing 'class'", health_reporter, service_key_in_manifest, base_meta, strict)
        return

    try:
        if registry.get(service_key_in_manifest, None):
            logger.info("Service '%s' already registered - skipping.", service_key_in_manifest)
            return
    except (KeyError, ComponentRegistryMissingError):
        pass

    logger.info("→ Instantiating shared service '%s' (%s)", service_key_in_manifest, class_path)
    
    try:
        service_class = import_by_path(class_path)
    except ImportError as exc:
        _def_error(f'Import error: {exc}', health_reporter, service_key_in_manifest, base_meta, strict)
        return

    pydantic_defaults = get_pydantic_defaults(service_class, base_meta.name)
    yaml_cfg = _load_config(service_spec_from_manifest.get('config'), service_key_in_manifest)
    if _YAML_PARAMETERS_KEY in yaml_cfg:
        yaml_cfg = yaml_cfg[_YAML_PARAMETERS_KEY]
    
    cfg_override = service_spec_from_manifest.get('config_override', {})
    
    final_cfg_unexpanded = _merge_configs(pydantic_defaults, yaml_cfg, cfg_override, service_key_in_manifest)
    # --- START OF FIX ---
    # Expand environment variables on the final config dict BEFORE instantiation
    final_service_config = _expand_tree(final_cfg_unexpanded)
    # --- END OF FIX ---
    
    meta_obj = getattr(service_class, 'METADATA_DEFINITION', None) if isinstance(getattr(service_class, 'METADATA_DEFINITION', None), ComponentMetadata) else base_meta
    
    md_path = service_spec_from_manifest.get('metadata_definition')
    if md_path:
        try:
            imported_md = import_by_path(md_path)
            if isinstance(imported_md, ComponentMetadata):
                meta_obj = imported_md
                logger.debug("Using metadata from path '%s'", md_path)
        except Exception as exc:
            logger.warning("Could not import metadata_definition '%s': %s", md_path, exc)

    meta_obj = dataclasses.replace(meta_obj, id=service_key_in_manifest)
    meta_obj = _apply_meta_override(meta_obj, service_spec_from_manifest.get('metadata_override', {}))

    _store_validation(
        validation_data_store,
        component_id=service_key_in_manifest,
        metadata=meta_obj,
        config=final_service_config,
        manifest_spec=service_spec_from_manifest
    )

    try:
        instance = await _instantiate_shared_service(
            service_class, service_key_in_manifest, final_service_config, meta_obj, registry, event_bus
        )
    except Exception as exc:
        _inst_error(f'Instantiation failed: {exc}', health_reporter, service_key_in_manifest, meta_obj, strict, exc)
        return

    try:
        if registry_manager:
            registry_manager.register_with_certification(instance, meta_obj, additional_cert_data={'source_manifest': 'standard.yaml'})
        else:
            registry.register(instance, meta_obj)
            logger.warning(f"Registered '{service_key_in_manifest}' without certification (RegistryManager not available).")

        _safe_register_service_instance_with_port(
            registry,
            service_class,
            instance,
            service_key_in_manifest,
            meta_obj.category,
            port_type=port_type,
            description_for_meta=meta_obj.description,
            requires_initialize_override=meta_obj.requires_initialize
        )
        health_reporter.add_component_status(service_key_in_manifest, ComponentStatus.INSTANCE_REGISTERED, meta_obj, [])
        if not meta_obj.requires_initialize:
            health_reporter.add_component_status(service_key_in_manifest, ComponentStatus.INITIALIZATION_SKIPPED_NOT_REQUIRED, meta_obj, [])
    except Exception as exc:
        _inst_error(f'Registry error: {exc}', health_reporter, service_key_in_manifest, meta_obj, strict, exc)

async def _instantiate_shared_service(
    service_class: Type,
    svc_id: str,
    cfg: Dict[str, Any],
    meta: ComponentMetadata,
    registry: ComponentRegistry,
    event_bus: EventBusPort,
):
    deps = {'event_bus': event_bus, 'registry': registry}
    if issubclass(service_class, NireonBaseComponent):
        return await create_component_instance(service_class, cfg, svc_id, meta, common_deps=deps)

    ctor_params = inspect.signature(service_class.__init__).parameters
    ctor_kwargs: Dict[str, Any] = {}
    if 'config' in ctor_params:
        ctor_kwargs['config'] = cfg
    elif 'cfg' in ctor_params:
        ctor_kwargs['cfg'] = cfg

    for name in ('id', 'metadata', 'metadata_definition'):
        if name in ctor_params:
            ctor_kwargs[name] = meta if 'metadata' in name else svc_id
    
    ctor_kwargs.update({k: v for k, v in deps.items() if k in ctor_params})
    return service_class(**ctor_kwargs)

async def register_orchestration_command(
    cmd_id: str,
    cmd_spec: Dict[str, Any],
    registry: ComponentRegistry,
    health_reporter: BootstrapHealthReporter,
    global_app_config: Dict[str, Any],
) -> None:
    enabled: bool = cmd_spec.get('enabled', True)
    strict: bool = global_app_config.get('bootstrap_strict_mode', True)
    class_path = cmd_spec.get('class')
    
    if not enabled:
        logger.info("Orchestration command '%s' disabled.", cmd_id)
        return

    if not class_path:
        _register_virtual_command(cmd_id, cmd_spec, registry, health_reporter)
        return

    try:
        cmd_class = import_by_path(class_path)
    except ImportError as exc:
        _def_error(f'Import error: {exc}', health_reporter, cmd_id, ComponentMetadata(id=cmd_id, name=cmd_id, category='orchestration_command'), strict)
        return

    from bootstrap.bootstrap_helper.metadata import create_service_metadata
    meta = create_service_metadata(
        service_id=cmd_id,
        service_name=cmd_id,
        category='orchestration_command',
        description=f'Orchestration command: {cmd_id}',
        requires_initialize=False
    )
    cmd_instance = cmd_class()
    registry.register(cmd_instance, meta)
    health_reporter.add_component_status(cmd_id, ComponentStatus.INSTANCE_REGISTERED, meta, [])
    logger.info("✓ Registered orchestration command '%s'", cmd_id)

def _register_virtual_command(
    cmd_id: str,
    cmd_spec: Dict[str, Any],
    registry: ComponentRegistry,
    health_reporter: BootstrapHealthReporter
) -> None:
    from bootstrap.bootstrap_helper.metadata import create_orchestration_command_metadata
    overrides = cmd_spec.get('metadata_override', {})
    meta = create_orchestration_command_metadata(
        command_id=cmd_id,
        command_name=overrides.get('name', cmd_id),
        description=overrides.get('description', f'Virtual component for {cmd_id}')
    )
    meta.produces = overrides.get('produces', [])
    meta.accepts = overrides.get('accepts', [])
    
    registry.register(None, meta)
    health_reporter.add_component_status(cmd_id, ComponentStatus.INSTANCE_REGISTERED, meta, [])
    logger.info("✓ Registered virtual orchestration command '%s'", cmd_id)


def _mark_disabled(reporter: BootstrapHealthReporter, comp_id: str | None, meta: ComponentMetadata) -> None:
    reporter.add_component_status(comp_id or 'unknown_disabled', ComponentStatus.DISABLED, meta, ['Disabled in manifest'])
    logger.info("Component '%s' disabled via manifest - skipped.", comp_id)

def _def_error(msg: str, reporter: BootstrapHealthReporter, comp_id: str | None, meta: ComponentMetadata, strict: bool, exc: Exception | None = None) -> None:
    logger.error(msg)
    reporter.add_component_status(comp_id or 'definition_error', ComponentStatus.DEFINITION_ERROR, meta, [msg])
    if strict:
        raise BootstrapError(msg, component_id=comp_id) from exc
    
def _meta_error(msg: str, reporter: BootstrapHealthReporter, comp_id: str, meta: ComponentMetadata, strict: bool, exc: Exception | None = None) -> None:
    logger.error(msg)
    reporter.add_component_status(comp_id, ComponentStatus.METADATA_CONSTRUCTION_ERROR, meta, [msg])
    if strict:
        raise BootstrapError(msg, component_id=comp_id) from exc

def _inst_error(msg: str, reporter: BootstrapHealthReporter, comp_id: str | None, meta: ComponentMetadata, strict: bool, exc: Exception | None = None) -> None:
    logger.error(msg, exc_info=True)
    reporter.add_component_status(comp_id or 'inst_error', ComponentStatus.BOOTSTRAP_ERROR, meta, [msg])
    if strict:
        raise BootstrapError(msg, component_id=comp_id) from exc


def _build_component_metadata(
    base_metadata: ComponentMetadata,
    component_id_from_manifest: str,
    manifest_comp_definition: Dict[str, Any]
) -> ComponentMetadata:
    if not isinstance(base_metadata, ComponentMetadata):
        raise TypeError(f"base_metadata for '{component_id_from_manifest}' must be ComponentMetadata, got {type(base_metadata)}")
    
    meta_dict = dataclasses.asdict(base_metadata) | {'id': component_id_from_manifest}
    meta_dict = _apply_manifest_meta_fields(meta_dict, manifest_comp_definition)
    
    try:
        return ComponentMetadata(**meta_dict)
    except TypeError as exc:
        raise BootstrapError(f"Metadata construction error for '{component_id_from_manifest}': {exc}", component_id=component_id_from_manifest) from exc

def _apply_manifest_meta_fields(meta: Dict[str, Any], manifest: Dict[str, Any]) -> Dict[str, Any]:
    overrides = manifest.get('metadata_override', {})
    meta.update(overrides)

    tags = manifest.get('epistemic_tags')
    if isinstance(tags, Sequence) and all(isinstance(t, str) for t in tags):
        meta['epistemic_tags'] = list(tags)
    
    if 'requires_initialize' not in overrides and isinstance(manifest.get('requires_initialize'), bool):
        meta['requires_initialize'] = manifest['requires_initialize']
    
    return meta

def _try_class_metadata(path: Optional[str]) -> Optional[ComponentMetadata]:
    if not path:
        return None
    try:
        cls = import_by_path(path)
        md = getattr(cls, 'METADATA_DEFINITION', None)
        return md if isinstance(md, ComponentMetadata) else None
    except Exception:
        return None


def _safe_register_service_instance_with_port(
    registry: ComponentRegistry,
    service_class: Type,
    service_instance: Any,
    service_id: str,
    category: str,
    port_type: Optional[str] = None,
    description_for_meta: Optional[str] = None,
    requires_initialize_override: Optional[bool] = None,
) -> None:
    if port_type:
        try:
            port_cls = import_by_path(port_type)
            _safe_register_service_instance(
                registry,
                port_cls,
                service_instance,
                service_id,
                category,
                description_for_meta=description_for_meta,
                requires_initialize_override=requires_initialize_override
            )
            logger.debug("Registered '%s' under port type '%s'", service_id, port_type)
        except Exception as exc:
            logger.warning("Could not register '%s' by port '%s': %s", service_id, port_type, exc)
    else:
        _safe_register_service_instance(
            registry,
            service_class,
            service_instance,
            service_id,
            category,
            description_for_meta=description_for_meta,
            requires_initialize_override=requires_initialize_override
        )