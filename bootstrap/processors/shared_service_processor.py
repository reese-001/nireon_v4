from __future__ import absolute_import
import asyncio
import dataclasses
import inspect
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from core.base_component import NireonBaseComponent
from core.lifecycle import ComponentMetadata, ComponentRegistryMissingError
from core.registry.component_registry import ComponentRegistry
from domain.ports.event_bus_port import EventBusPort
from bootstrap.exceptions import BootstrapError
from bootstrap.health.reporter import HealthReporter as BootstrapHealthReporter, ComponentStatus
from runtime.utils import import_by_path, load_yaml_robust
from bootstrap.utils.component_utils import create_component_instance, get_pydantic_defaults
from .service_resolver import _safe_register_service_instance_with_port

try:
    from configs.config_utils import ConfigMerger
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning('ConfigMerger not found in configs.config_utils, using placeholder.')
    class ConfigMerger:
        @staticmethod
        def merge(dict1: Dict, dict2: Dict, context_name: str) -> Dict:
            logger.debug(f'Using placeholder ConfigMerger for: {context_name}')
            result = dict1.copy()
            result.update(dict2)
            return result

logger = logging.getLogger(__name__)

async def instantiate_shared_service(
    service_key_in_manifest: str,
    service_spec_from_manifest: Dict[str, Any],
    registry: ComponentRegistry,
    event_bus: EventBusPort,
    global_app_config: Dict[str, Any],
    health_reporter: BootstrapHealthReporter,
    validation_data_store: Any
) -> None:
    class_path = service_spec_from_manifest.get('class')
    config_source = service_spec_from_manifest.get('config')
    inline_config_override = service_spec_from_manifest.get('config_override', {})
    port_type_str = service_spec_from_manifest.get('port_type')
    is_enabled = service_spec_from_manifest.get('enabled', True)
    is_strict_mode = global_app_config.get('bootstrap_strict_mode', True)
    
    service_name_for_report = class_path.split(':')[-1] if class_path and ':' in class_path else class_path.split('.')[-1] if class_path else service_key_in_manifest
    base_meta_for_report = ComponentMetadata(
        id=service_key_in_manifest,
        name=service_name_for_report,
        category='shared_service',
        version='0.1.0'
    )
    
    if not class_path:
        msg = f"Shared service '{service_key_in_manifest}' definition missing 'class' path."
        logger.error(msg)
        health_reporter.add_component_status(service_key_in_manifest, ComponentStatus.DEFINITION_ERROR, base_meta_for_report, [msg])
        if is_strict_mode:
            raise BootstrapError(msg, component_id=service_key_in_manifest)
        return
    
    if not is_enabled:
        logger.info(f"Shared service '{service_key_in_manifest}' (Class: {class_path}) is disabled. Skipping.")
        health_reporter.add_component_status(service_key_in_manifest, ComponentStatus.DISABLED, base_meta_for_report, ['Disabled in manifest'])
        return
    
    try:
        existing_instance = registry.get(service_key_in_manifest)
        if existing_instance is not None:
            logger.info(f"Service '{service_key_in_manifest}' (Class: {class_path}) already in registry by key. Skipping manifest instantiation.")
            return
    except (KeyError, ComponentRegistryMissingError):
        logger.debug(f"Service '{service_key_in_manifest}' not found in registry by key. Proceeding with instantiation.")
    
    logger.info(f'-> Instantiating Shared Service from manifest: {service_key_in_manifest} (Class: {class_path})')
    
    try:
        service_class = import_by_path(class_path)
    except ImportError as e_import:
        msg = f"Failed to import class '{class_path}' for shared service '{service_key_in_manifest}': {e_import}"
        logger.error(msg, exc_info=True)
        health_reporter.add_component_status(service_key_in_manifest, ComponentStatus.DEFINITION_ERROR, base_meta_for_report, [msg])
        if is_strict_mode:
            raise BootstrapError(msg, component_id=service_key_in_manifest) from e_import
        return
    
    pydantic_class_defaults = get_pydantic_defaults(service_class, service_name_for_report)
    
    yaml_loaded_config = {}
    if isinstance(config_source, str):
        actual_config_path = config_source.replace('{id}', service_key_in_manifest)
        yaml_loaded_config = load_yaml_robust(Path(actual_config_path))
        if not yaml_loaded_config and Path(actual_config_path).is_file() and Path(actual_config_path).read_text(encoding='utf-8').strip():
            msg = f"Failed to parse non-empty config YAML '{actual_config_path}' for service '{service_key_in_manifest}'."
            logger.error(msg)
    elif isinstance(config_source, dict):
        yaml_loaded_config = config_source
        logger.debug(f"Using inline config from manifest for service '{service_key_in_manifest}'")
    elif config_source is not None:
        logger.warning(f"Unexpected 'config' type for service '{service_key_in_manifest}': {type(config_source)}. Expected string path or dict. Ignoring.")
    
    merged_config_step1 = ConfigMerger.merge(pydantic_class_defaults, yaml_loaded_config, f'{service_key_in_manifest}_defaults_yaml')
    final_service_config = ConfigMerger.merge(merged_config_step1, inline_config_override, f'{service_key_in_manifest}_final_config')
    
    # Start with base metadata
    service_instance_metadata = base_meta_for_report
    
    # Check if service class has METADATA_DEFINITION
    if hasattr(service_class, 'METADATA_DEFINITION') and isinstance(service_class.METADATA_DEFINITION, ComponentMetadata):
        service_instance_metadata = dataclasses.replace(service_class.METADATA_DEFINITION, id=service_key_in_manifest)
    
    # Handle metadata_definition from manifest
    metadata_definition_path = service_spec_from_manifest.get('metadata_definition')
    if metadata_definition_path:
        try:
            metadata_obj = import_by_path(metadata_definition_path)
            if isinstance(metadata_obj, ComponentMetadata):
                service_instance_metadata = dataclasses.replace(metadata_obj, id=service_key_in_manifest)
                logger.debug(f"Using metadata from manifest path: {metadata_definition_path}")
        except Exception as e:
            logger.warning(f"Could not import metadata_definition '{metadata_definition_path}': {e}")
    
    # Apply metadata overrides
    manifest_meta_override = service_spec_from_manifest.get('metadata_override', {})
    if manifest_meta_override:
        current_meta_dict = dataclasses.asdict(service_instance_metadata)
        current_meta_dict.update(manifest_meta_override)
        try:
            service_instance_metadata = ComponentMetadata(**current_meta_dict)
        except Exception as e_meta_override:
            logger.warning(f"Error applying metadata_override for '{service_key_in_manifest}': {e_meta_override}. Using metadata before override.")
    
    # Ensure the metadata ID matches the manifest key
    if service_instance_metadata.id != service_key_in_manifest:
        service_instance_metadata = dataclasses.replace(service_instance_metadata, id=service_key_in_manifest)
    
    if validation_data_store and hasattr(validation_data_store, 'store_component_data'):
        validation_data_store.store_component_data(
            component_id=service_key_in_manifest,
            original_metadata=service_instance_metadata,
            resolved_config=final_service_config,
            manifest_spec=service_spec_from_manifest
        )
    
    service_instance: Optional[Any] = None
    try:
        common_deps_for_service: Optional[Dict[str, Any]] = None
        potential_deps_map = {'event_bus': event_bus, 'registry': registry}
        
        if issubclass(service_class, NireonBaseComponent):
            service_instance = await create_component_instance(
                component_class=service_class,
                resolved_config_for_instance=final_service_config,
                instance_id=service_key_in_manifest,
                instance_metadata_object=service_instance_metadata,
                common_deps=potential_deps_map
            )
        else:
            ctor_params = inspect.signature(service_class.__init__).parameters
            kwargs_for_ctor: Dict[str, Any] = {}
            
            if 'config' in ctor_params:
                kwargs_for_ctor['config'] = final_service_config
            elif 'cfg' in ctor_params:
                kwargs_for_ctor['cfg'] = final_service_config
            
            # Add ID parameter if accepted
            if 'id' in ctor_params:
                kwargs_for_ctor['id'] = service_key_in_manifest
            
            # Add metadata if accepted
            if 'metadata_definition' in ctor_params:
                kwargs_for_ctor['metadata_definition'] = service_instance_metadata
            elif 'metadata' in ctor_params:
                kwargs_for_ctor['metadata'] = service_instance_metadata
            
            for dep_name, dep_instance in potential_deps_map.items():
                if dep_name in ctor_params:
                    kwargs_for_ctor[dep_name] = dep_instance
            
            service_instance = service_class(**kwargs_for_ctor)
    
    except Exception as e_create:
        msg = f"Instantiation failed for shared service '{service_key_in_manifest}': {e_create}"
        logger.error(msg, exc_info=True)
        health_reporter.add_component_status(service_key_in_manifest, ComponentStatus.INSTANTIATION_ERROR, service_instance_metadata, [msg])
        if is_strict_mode:
            if not isinstance(e_create, BootstrapError):
                raise BootstrapError(msg, component_id=service_key_in_manifest) from e_create
            raise
        return
    
    if service_instance:
        # Ensure the instance has the correct metadata with the manifest ID
        if hasattr(service_instance, 'metadata') and isinstance(service_instance.metadata, ComponentMetadata):
            if service_instance.metadata.id != service_key_in_manifest:
                # Update the instance's metadata to have the correct ID
                updated_metadata = dataclasses.replace(service_instance.metadata, id=service_key_in_manifest)
                if hasattr(service_instance, '_metadata_definition'):
                    object.__setattr__(service_instance, '_metadata_definition', updated_metadata)
                logger.debug(f"Updated service instance metadata ID to match manifest key: {service_key_in_manifest}")
        
        # CRITICAL: First register with the exact manifest ID
        # This ensures we can look up by 'llm_router_main' etc.
        try:
            registry.register(service_instance, service_instance_metadata)
            logger.info(f"✓ Registered '{service_key_in_manifest}' in registry with manifest ID")
        except Exception as e:
            logger.error(f"Failed to register '{service_key_in_manifest}' by manifest ID: {e}")
            raise
        
        # Determine if initialization is required
        final_requires_init = service_instance_metadata.requires_initialize
        if isinstance(service_instance, NireonBaseComponent):
            final_requires_init = getattr(service_instance.metadata, 'requires_initialize', False)
        
        # Then register by type and port for interface-based lookups
        _safe_register_service_instance_with_port(
            registry,
            service_class,
            service_instance,
            service_key_in_manifest,
            service_instance_metadata.category,
            port_type=port_type_str,
            description_for_meta=service_instance_metadata.description,
            requires_initialize_override=final_requires_init
        )
        
        logger.info(f'✓ Shared Service Instantiated and Registered: {service_key_in_manifest} -> {service_class.__name__} (Category: {service_instance_metadata.category}, Requires Init: {final_requires_init})')
        
        # Update health status appropriately
        health_reporter.add_component_status(
            service_key_in_manifest,
            ComponentStatus.INSTANCE_REGISTERED,
            service_instance_metadata,
            []
        )
        
        # If component doesn't require initialization, mark it as healthy immediately
        if not final_requires_init:
            health_reporter.add_component_status(
                service_key_in_manifest,
                ComponentStatus.INITIALIZATION_SKIPPED_NOT_REQUIRED,
                service_instance_metadata,
                []
            )
            logger.info(f"✓ Service '{service_key_in_manifest}' marked as healthy (no initialization required)")
    else:
        msg = f"Service instantiation returned None for '{service_key_in_manifest}' using class {service_class.__name__}"
        logger.error(msg)
        health_reporter.add_component_status(service_key_in_manifest, ComponentStatus.INSTANTIATION_ERROR, service_instance_metadata, [msg])
        if is_strict_mode:
            raise BootstrapError(msg, component_id=service_key_in_manifest)