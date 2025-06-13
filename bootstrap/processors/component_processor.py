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
from factories.mechanism_factory import SimpleMechanismFactory
from domain.ports.event_bus_port import EventBusPort
# Fix: Import BootstrapError directly from the helper to avoid circular import
from bootstrap.bootstrap_helper._exceptions import BootstrapError
from bootstrap.health.reporter import HealthReporter as BootstrapHealthReporter, ComponentStatus
from bootstrap.processors.metadata import get_default_metadata
from bootstrap.processors.service_resolver import _safe_register_service_instance
from runtime.utils import import_by_path, load_yaml_robust
from bootstrap.utils.component_utils import create_component_instance, get_pydantic_defaults

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


async def process_simple_component(
    comp_def: Dict[str, Any],
    registry: ComponentRegistry,
    mechanism_factory: Optional[SimpleMechanismFactory],
    health_reporter: BootstrapHealthReporter,
    run_id: str,
    global_app_config: Dict[str, Any],
    validation_data_store: Any
) -> None:
    """Process a simple component definition from manifest."""
    
    component_id = comp_def.get('component_id')
    factory_key = comp_def.get('factory_key')
    component_class_path = comp_def.get('class')
    component_type = comp_def.get('type')
    yaml_config_source = comp_def.get('config')
    inline_config_override = comp_def.get('config_override', {})
    is_enabled = comp_def.get('enabled', True)
    is_strict_mode = global_app_config.get('bootstrap_strict_mode', True)
    
    # Create metadata for reporting
    current_component_metadata_for_reporting = ComponentMetadata(
        id=component_id or 'unknown_id_simple',
        name=factory_key or component_class_path or 'UnknownNameSimple',
        version='0.0.0',
        category=component_type or 'unknown_type_simple',
        requires_initialize=True
    )
    
    if not is_enabled:
        logger.info(f"Simple component '{component_id}' is disabled via manifest. Skipping.")
        health_reporter.add_component_status(
            component_id or 'unknown_disabled_component',
            ComponentStatus.DISABLED,
            current_component_metadata_for_reporting,
            ['Disabled in manifest']
        )
        return
    
    # Validate required fields
    if not component_id or not (factory_key or component_class_path) or (not component_type):
        errmsg = f"Skipping simple component definition due to missing 'component_id', ('factory_key' or 'class'), or 'type'. Definition: {comp_def}"
        logger.error(errmsg)
        health_reporter.add_component_status(
            component_id or 'unknown_component_def_error',
            ComponentStatus.DEFINITION_ERROR,
            current_component_metadata_for_reporting,
            [errmsg]
        )
        if is_strict_mode:
            raise BootstrapError(errmsg, component_id=component_id or 'unknown_component_def_error')
        return
    
    target_class_or_factory_key = component_class_path if component_class_path else factory_key
    
    # Get base metadata
    base_metadata_for_class: Optional[ComponentMetadata] = None
    if factory_key:
        base_metadata_for_class = get_default_metadata(factory_key)
    
    if not base_metadata_for_class and component_class_path:
        try:
            cls_for_meta = import_by_path(component_class_path)
            if hasattr(cls_for_meta, 'METADATA_DEFINITION') and isinstance(cls_for_meta.METADATA_DEFINITION, ComponentMetadata):
                base_metadata_for_class = cls_for_meta.METADATA_DEFINITION
        except Exception as e:
            logger.debug(f"Could not retrieve METADATA_DEFINITION from class {component_class_path} for '{component_id}': {e}")
    
    if not base_metadata_for_class:
        logger.warning(f"No default or class-defined metadata found for '{target_class_or_factory_key}'. Using minimal fallback for component '{component_id}'.")
        base_metadata_for_class = ComponentMetadata(
            id=component_id,
            name=component_id,
            version='0.1.0',
            category=component_type,
            description=f"Auto-generated metadata for {component_type} '{component_id}'"
        )
    
    # Build instance metadata
    try:
        final_instance_metadata = _build_component_metadata(base_metadata_for_class, component_id, comp_def)
        current_component_metadata_for_reporting = final_instance_metadata
    except Exception as e_meta:
        errmsg = f"Error constructing ComponentMetadata for simple component '{component_id}': {e_meta}"
        logger.error(errmsg, exc_info=True)
        health_reporter.add_component_status(
            component_id,
            ComponentStatus.METADATA_CONSTRUCTION_ERROR,
            current_component_metadata_for_reporting,
            [errmsg]
        )
        if is_strict_mode:
            raise BootstrapError(errmsg, component_id=component_id) from e_meta
        return
    
    # Get Pydantic defaults
    pydantic_class_defaults = {}
    component_class_for_pydantic: Optional[Type] = None
    if component_class_path:
        try:
            component_class_for_pydantic = import_by_path(component_class_path)
            pydantic_class_defaults = get_pydantic_defaults(component_class_for_pydantic, final_instance_metadata.name)
        except Exception as e_pydantic:
            logger.debug(f"Could not get Pydantic defaults for simple component '{component_id}' (class: {component_class_path}): {e_pydantic}")
    
    # Load YAML config
    yaml_loaded_data = {}
    if isinstance(yaml_config_source, str):
        actual_config_path_str = yaml_config_source.replace('{id}', component_id)
        yaml_loaded_data = load_yaml_robust(Path(actual_config_path_str))
    elif isinstance(yaml_config_source, dict):
        yaml_loaded_data = yaml_config_source

    # Extract the actual component config from the 'parameters' key if it exists
    # This aligns with the new YAML structure.
    yaml_config_params = yaml_loaded_data.get('parameters', {})
    if not yaml_config_params and yaml_loaded_data:
        logger.debug(f"No 'parameters' key in config for '{component_id}'. Using entire loaded config dictionary.")
        yaml_config_params = yaml_loaded_data # Fallback for old format

    # Now merge the correct dictionaries
    merged_config_step1 = ConfigMerger.merge(pydantic_class_defaults, yaml_config_params, f'{component_id}_defaults_yaml')
    final_config = ConfigMerger.merge(merged_config_step1, inline_config_override, f'{component_id}_final_config')
    
    # Store validation data
    if validation_data_store and hasattr(validation_data_store, 'store_component_data'):
        validation_data_store.store_component_data(
            component_id=component_id,
            original_metadata=final_instance_metadata,
            resolved_config=final_config,
            manifest_spec=comp_def
        )
    
    # Instantiate component
    instance: Optional[NireonBaseComponent] = None
    try:
        if component_class_path:
            cls_to_instantiate = component_class_for_pydantic or import_by_path(component_class_path)
            if not issubclass(cls_to_instantiate, NireonBaseComponent):
                raise BootstrapError(f"Class {component_class_path} for '{component_id}' is not a NireonBaseComponent.")
            
            instance = await create_component_instance(
                cls_to_instantiate,
                final_config,
                component_id,
                final_instance_metadata,
                common_deps=None
            )
        elif factory_key and component_type == 'mechanism' and mechanism_factory:
            instance = mechanism_factory.create_mechanism(factory_key, final_instance_metadata, final_config)
        else:
            errmsg = f"Cannot determine instantiation method for simple component '{component_id}'. Need 'class' path or a 'factory_key' for a known type (e.g., mechanism)."
            raise BootstrapError(errmsg, component_id=component_id)
        
        if instance is None:
            errmsg = f"Component instantiation returned None for simple component '{component_id}' (target: {target_class_or_factory_key})"
            raise BootstrapError(errmsg, component_id=component_id)
        
        logger.info(f"Instantiated simple component '{component_id}' (Type: {component_type}, Class/FactoryKey: {target_class_or_factory_key})")
        
        # Register the component
        _safe_register_service_instance(
            registry,
            type(instance),
            instance,
            component_id,
            final_instance_metadata.category,
            description_for_meta=final_instance_metadata.description,
            requires_initialize_override=final_instance_metadata.requires_initialize
        )
        
        health_reporter.add_component_status(component_id, ComponentStatus.INSTANCE_REGISTERED, instance.metadata, [])
        
    except Exception as e:
        errmsg = f"Error during instantiation/registration of simple component '{component_id}': {e}"
        logger.error(errmsg, exc_info=True)
        health_reporter.add_component_status(
            component_id,
            ComponentStatus.BOOTSTRAP_ERROR,
            current_component_metadata_for_reporting,
            [errmsg]
        )
        if is_strict_mode:
            if not isinstance(e, BootstrapError):
                raise BootstrapError(errmsg, component_id=component_id) from e
            raise


def _build_component_metadata(
    base_metadata: ComponentMetadata,
    component_id_from_manifest: str,
    manifest_comp_definition: Dict[str, Any]
) -> ComponentMetadata:
    """Build component metadata from base metadata and manifest overrides."""
    
    if not isinstance(base_metadata, ComponentMetadata):
        raise TypeError(f"base_metadata for '{component_id_from_manifest}' must be ComponentMetadata, got {type(base_metadata)}")
    
    # Start with base metadata as dict
    instance_metadata_dict = dataclasses.asdict(base_metadata)
    instance_metadata_dict['id'] = component_id_from_manifest
    
    # Apply manifest metadata overrides
    manifest_meta_override = manifest_comp_definition.get('metadata_override', {})
    if manifest_meta_override:
        logger.debug(f'[{component_id_from_manifest}] Applying manifest metadata_override: {manifest_meta_override}')
        for key, value in manifest_meta_override.items():
            if key in instance_metadata_dict:
                instance_metadata_dict[key] = value
            elif key == 'requires_initialize' and isinstance(value, bool):
                instance_metadata_dict[key] = value
            else:
                logger.warning(f"[{component_id_from_manifest}] Unknown key '{key}' or invalid type in metadata_override. Key not present in base metadata fields. Ignoring.")
    
    # Handle epistemic_tags from manifest
    if 'epistemic_tags' in manifest_comp_definition:
        tags = manifest_comp_definition['epistemic_tags']
        if isinstance(tags, list) and all(isinstance(t, str) for t in tags):
            instance_metadata_dict['epistemic_tags'] = tags
            logger.debug(f'[{component_id_from_manifest}] Using epistemic_tags from manifest: {tags}')
        else:
            logger.warning(f"[{component_id_from_manifest}] Invalid 'epistemic_tags' in manifest, expected list of strings. Current tags from base: {instance_metadata_dict.get('epistemic_tags')}")
    
    # Handle requires_initialize
    if 'requires_initialize' not in manifest_meta_override:
        if 'requires_initialize' in manifest_comp_definition and isinstance(manifest_comp_definition['requires_initialize'], bool):
            instance_metadata_dict['requires_initialize'] = manifest_comp_definition['requires_initialize']
        else:
            instance_metadata_dict.setdefault('requires_initialize', base_metadata.requires_initialize)
    
    try:
        return ComponentMetadata(**instance_metadata_dict)
    except TypeError as e:
        logger.error(f"Failed to create ComponentMetadata for '{component_id_from_manifest}' due to TypeError: {e}. Data: {instance_metadata_dict}")
        raise BootstrapError(f"Metadata construction error for '{component_id_from_manifest}': {e}", component_id=component_id_from_manifest) from e


async def instantiate_shared_service(
    service_key_in_manifest: str,
    service_spec_from_manifest: Dict[str, Any],
    registry: ComponentRegistry,
    event_bus: EventBusPort,
    global_app_config: Dict[str, Any],
    health_reporter: BootstrapHealthReporter,
    validation_data_store: Any
) -> None:
    """Instantiate a shared service from manifest specification."""
    
    class_path = service_spec_from_manifest.get('class')
    config_source = service_spec_from_manifest.get('config')
    inline_config_override = service_spec_from_manifest.get('config_override', {})
    is_enabled = service_spec_from_manifest.get('enabled', True)
    is_strict_mode = global_app_config.get('bootstrap_strict_mode', True)
    
    # Create metadata for reporting
    service_name_for_report = (class_path.split(':')[-1] if class_path and ':' in class_path else 
                              class_path.split('.')[-1] if class_path else service_key_in_manifest)
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
    
    # Check if already exists
    try:
        existing_instance = registry.get(service_key_in_manifest)
        if existing_instance is not None:
            logger.info(f"Service '{service_key_in_manifest}' (Class: {class_path}) already in registry by key. Skipping manifest instantiation.")
            return
    except (KeyError, ComponentRegistryMissingError):
        logger.debug(f"Service '{service_key_in_manifest}' not found in registry by key. Proceeding with instantiation.")
    
    logger.info(f'-> Instantiating Shared Service from manifest: {service_key_in_manifest} (Class: {class_path})')
    
    # Import service class
    try:
        service_class = import_by_path(class_path)
    except ImportError as e_import:
        msg = f"Failed to import class '{class_path}' for shared service '{service_key_in_manifest}': {e_import}"
        logger.error(msg, exc_info=True)
        health_reporter.add_component_status(service_key_in_manifest, ComponentStatus.DEFINITION_ERROR, base_meta_for_report, [msg])
        if is_strict_mode:
            raise BootstrapError(msg, component_id=service_key_in_manifest) from e_import
        return
    
    # Get Pydantic defaults
    pydantic_class_defaults = get_pydantic_defaults(service_class, service_name_for_report)
    
    # Load YAML config
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
    
    # Merge configs
    merged_config_step1 = ConfigMerger.merge(pydantic_class_defaults, yaml_loaded_config, f'{service_key_in_manifest}_defaults_yaml')
    final_service_config = ConfigMerger.merge(merged_config_step1, inline_config_override, f'{service_key_in_manifest}_final_config')
    
    # Build metadata
    service_instance_metadata = base_meta_for_report
    
    if hasattr(service_class, 'METADATA_DEFINITION') and isinstance(service_class.METADATA_DEFINITION, ComponentMetadata):
        service_instance_metadata = dataclasses.replace(service_class.METADATA_DEFINITION, id=service_key_in_manifest)
    
    metadata_definition_path = service_spec_from_manifest.get('metadata_definition')
    if metadata_definition_path:
        try:
            metadata_obj = import_by_path(metadata_definition_path)
            if isinstance(metadata_obj, ComponentMetadata):
                service_instance_metadata = dataclasses.replace(metadata_obj, id=service_key_in_manifest)
                logger.debug(f'Using metadata from manifest path: {metadata_definition_path}')
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
    
    if service_instance_metadata.id != service_key_in_manifest:
        service_instance_metadata = dataclasses.replace(service_instance_metadata, id=service_key_in_manifest)
    
    # Store validation data
    if validation_data_store and hasattr(validation_data_store, 'store_component_data'):
        validation_data_store.store_component_data(
            component_id=service_key_in_manifest,
            original_metadata=service_instance_metadata,
            resolved_config=final_service_config,
            manifest_spec=service_spec_from_manifest
        )
    
    # Instantiate service
    service_instance: Optional[Any] = None
    try:
        potential_deps_map = {
            'event_bus': event_bus,
            'registry': registry
        }
        
        if issubclass(service_class, NireonBaseComponent):
            service_instance = await create_component_instance(
                component_class=service_class,
                resolved_config_for_instance=final_service_config,
                instance_id=service_key_in_manifest,
                instance_metadata_object=service_instance_metadata,
                common_deps=potential_deps_map
            )
        else:
            # Handle non-NireonBaseComponent services
            ctor_params = inspect.signature(service_class.__init__).parameters
            kwargs_for_ctor: Dict[str, Any] = {}
            
            if 'config' in ctor_params:
                kwargs_for_ctor['config'] = final_service_config
            elif 'cfg' in ctor_params:
                kwargs_for_ctor['cfg'] = final_service_config
            
            if 'id' in ctor_params:
                kwargs_for_ctor['id'] = service_key_in_manifest
            
            if 'metadata_definition' in ctor_params:
                kwargs_for_ctor['metadata_definition'] = service_instance_metadata
            elif 'metadata' in ctor_params:
                kwargs_for_ctor['metadata'] = service_instance_metadata
            
            # Add dependencies if constructor expects them
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
    
    # Register service
    if service_instance:
        # Update metadata ID if needed
        if hasattr(service_instance, 'metadata') and isinstance(service_instance.metadata, ComponentMetadata):
            if service_instance.metadata.id != service_key_in_manifest:
                updated_metadata = dataclasses.replace(service_instance.metadata, id=service_key_in_manifest)
                if hasattr(service_instance, '_metadata_definition'):
                    object.__setattr__(service_instance, '_metadata_definition', updated_metadata)
                logger.debug(f'Updated service instance metadata ID to match manifest key: {service_key_in_manifest}')
        
        # Register in registry
        try:
            registry.register(service_instance, service_instance_metadata)
            logger.info(f"✓ Registered '{service_key_in_manifest}' in registry with manifest ID")
        except Exception as e:
            logger.error(f"Failed to register '{service_key_in_manifest}' by manifest ID: {e}")
            raise
        
        # Determine initialization requirement
        final_requires_init = service_instance_metadata.requires_initialize
        if isinstance(service_instance, NireonBaseComponent):
            final_requires_init = getattr(service_instance.metadata, 'requires_initialize', False)
        
        # Register by type/port if needed
        port_type_str = service_spec_from_manifest.get('port_type')
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
        
        health_reporter.add_component_status(service_key_in_manifest, ComponentStatus.INSTANCE_REGISTERED, service_instance_metadata, [])
        
        if not final_requires_init:
            health_reporter.add_component_status(service_key_in_manifest, ComponentStatus.INITIALIZATION_SKIPPED_NOT_REQUIRED, service_instance_metadata, [])
            logger.info(f"✓ Service '{service_key_in_manifest}' marked as healthy (no initialization required)")
    else:
        msg = f"Service instantiation returned None for '{service_key_in_manifest}' using class {service_class.__name__}"
        logger.error(msg)
        health_reporter.add_component_status(service_key_in_manifest, ComponentStatus.INSTANTIATION_ERROR, service_instance_metadata, [msg])
        if is_strict_mode:
            raise BootstrapError(msg, component_id=service_key_in_manifest)


def _safe_register_service_instance_with_port(
    registry: ComponentRegistry,
    service_class: Type,
    service_instance: Any,
    service_id: str,
    category: str,
    port_type: Optional[str] = None,
    description_for_meta: Optional[str] = None,
    requires_initialize_override: Optional[bool] = None
) -> None:
    """Register service instance with optional port type handling."""
    
    # If port_type is specified, try to import and register by that type too
    if port_type:
        try:
            port_class = import_by_path(port_type)
            _safe_register_service_instance(
                registry,
                port_class,
                service_instance,
                service_id,
                category,
                description_for_meta=description_for_meta,
                requires_initialize_override=requires_initialize_override
            )
            logger.debug(f"Also registered '{service_id}' by port type: {port_type}")
        except Exception as e:
            logger.warning(f"Could not register '{service_id}' by port type '{port_type}': {e}")
    else:
        # Register by service class type
        _safe_register_service_instance(
            registry,
            service_class,
            service_instance,
            service_id,
            category,
            description_for_meta=description_for_meta,
            requires_initialize_override=requires_initialize_override
        )


async def register_orchestration_command(
    cmd_id: str,
    cmd_spec: Dict[str, Any],
    registry: ComponentRegistry,
    health_reporter: BootstrapHealthReporter,
    global_app_config: Dict[str, Any]
) -> None:
    """Register an orchestration command from manifest specification."""
    
    class_path = cmd_spec.get('class')
    is_enabled = cmd_spec.get('enabled', True)
    is_strict_mode = global_app_config.get('bootstrap_strict_mode', True)
    
    if not is_enabled:
        logger.info(f"Orchestration command '{cmd_id}' is disabled. Skipping.")
        return
    
    if not class_path:
        msg = f"Orchestration command '{cmd_id}' missing 'class' path."
        logger.error(msg)
        if is_strict_mode:
            raise BootstrapError(msg, component_id=cmd_id)
        return
    
    try:
        cmd_class = import_by_path(class_path)
        
        # Create basic metadata for orchestration command
        from bootstrap.bootstrap_helper.metadata import create_service_metadata
        cmd_metadata = create_service_metadata(
            service_id=cmd_id,
            service_name=cmd_id,
            category='orchestration_command',
            description=f'Orchestration command: {cmd_id}',
            requires_initialize=False
        )
        
        # Simple instantiation for orchestration commands
        cmd_instance = cmd_class()
        
        # Register the command
        registry.register(cmd_instance, cmd_metadata)
        
        health_reporter.add_component_status(cmd_id, ComponentStatus.INSTANCE_REGISTERED, cmd_metadata, [])
        logger.info(f"✓ Registered orchestration command '{cmd_id}'")
        
    except Exception as e:
        msg = f"Failed to register orchestration command '{cmd_id}': {e}"
        logger.error(msg, exc_info=True)
        if is_strict_mode:
            raise BootstrapError(msg, component_id=cmd_id) from e