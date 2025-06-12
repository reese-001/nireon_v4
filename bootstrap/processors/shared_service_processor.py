"""Shared service processor for handling instantiation and registration of shared services."""

from .component_processing_utils import *


async def instantiate_shared_service(
    service_key_in_manifest: str,
    service_spec_from_manifest: Dict[str, Any],
    registry: ComponentRegistry,
    event_bus: EventBusPort,
    global_app_config: Dict[str, Any],
    health_reporter: BootstrapHealthReporter,
    validation_data_store: Any
) -> None:
    """Instantiate and register a shared service from manifest specification."""
    
    class_path = service_spec_from_manifest.get('class')
    config_path_template = service_spec_from_manifest.get('config')
    inline_config_override = service_spec_from_manifest.get('config_override', {})
    port_type = service_spec_from_manifest.get('port_type')
    is_enabled = service_spec_from_manifest.get('enabled', True)
    is_strict_mode = global_app_config.get('bootstrap_strict_mode', True)
    
    service_name_for_report = class_path.split(':')[-1] if class_path and ':' in class_path else class_path.split('.')[-1] if class_path else service_key_in_manifest
    base_meta_for_report = ComponentMetadata(
        id=service_key_in_manifest,
        name=service_name_for_report,
        category='shared_service',
        version='unknown'
    )
    
    if not class_path:
        msg = f"Shared service '{service_key_in_manifest}' definition missing 'class' path."
        logger.error(msg)
        health_reporter.add_component_status(service_key_in_manifest, ComponentStatus.DEFINITION_ERROR, base_meta_for_report, [msg])
        if is_strict_mode:
            raise BootstrapError(msg)
        return
    
    if not is_enabled:
        logger.info(f"Shared service '{service_key_in_manifest}' (Class: {class_path}) is disabled. Skipping.")
        health_reporter.add_component_status(service_key_in_manifest, ComponentStatus.DISABLED, base_meta_for_report, ['Disabled in manifest'])
        return
    
    # FIX: ComponentRegistryMissingError is now correctly imported from core.lifecycle in utils
    try:
        existing_instance = registry.get(service_key_in_manifest)
        if existing_instance is not None:
            logger.info(f"Service '{service_key_in_manifest}' (Class: {class_path}) already in registry. Skipping manifest instantiation.")
            return
    except (KeyError, ComponentRegistryMissingError):
        logger.info(f"Service '{service_key_in_manifest}' not found in registry by key. Proceeding with instantiation.")
    
    logger.info(f'-> Instantiating Shared Service from manifest: {service_key_in_manifest} (Class: {class_path})')
    
    try:
        service_class = import_by_path(class_path)
    except ImportError as e_import:
        msg = f"Failed to import class '{class_path}' for shared service '{service_key_in_manifest}': {e_import}"
        logger.error(msg, exc_info=True)
        health_reporter.add_component_status(service_key_in_manifest, ComponentStatus.DEFINITION_ERROR, base_meta_for_report, [msg])
        if is_strict_mode:
            raise BootstrapError(msg) from e_import
        return
    
    pydantic_defaults = _get_pydantic_defaults(service_class, service_name_for_report)
    
    yaml_file_config = {}
    # FIX: Check if config_path_template is a string (path) or dict (inline config)
    if config_path_template:
        if isinstance(config_path_template, str):
            # It's a path template - process as before
            actual_config_path = config_path_template.replace('{id}', service_key_in_manifest)
            yaml_file_config = load_yaml_robust(Path(actual_config_path))
            if not yaml_file_config and Path(actual_config_path).is_file() and Path(actual_config_path).read_text(encoding='utf-8').strip():
                msg = f"Failed to parse non-empty config YAML '{actual_config_path}' for service '{service_key_in_manifest}'."
                logger.error(msg)
        elif isinstance(config_path_template, dict):
            # It's an inline config - use it directly
            yaml_file_config = config_path_template
            logger.debug(f"Using inline config from manifest for service '{service_key_in_manifest}'")
        else:
            logger.warning(f"Unexpected config type for service '{service_key_in_manifest}': {type(config_path_template)}")
    
    merged_config_step1 = ConfigMerger.merge(pydantic_defaults, yaml_file_config, f'{service_key_in_manifest}_pydantic_yaml')
    final_service_config = ConfigMerger.merge(merged_config_step1, inline_config_override, f'{service_key_in_manifest}_final')
    
    service_instance_metadata = base_meta_for_report
    if hasattr(service_class, 'METADATA_DEFINITION') and isinstance(service_class.METADATA_DEFINITION, ComponentMetadata):
        service_instance_metadata = dataclasses.replace(service_class.METADATA_DEFINITION, id=service_key_in_manifest)
    elif hasattr(service_class, 'metadata') and isinstance(service_class.metadata, ComponentMetadata):
        service_instance_metadata = dataclasses.replace(service_class.metadata, id=service_key_in_manifest)
    
    if validation_data_store:
        validation_data_store.store_component_data(
            component_id=service_key_in_manifest,
            original_metadata=service_instance_metadata,
            resolved_config=final_service_config,
            manifest_spec=service_spec_from_manifest
        )
    
    service_instance: Optional[Any] = None
    try:
        if issubclass(service_class, NireonBaseComponent):
            service_instance = await _create_component_instance(
                component_class=service_class,
                resolved_config_for_instance=final_service_config,
                instance_id=service_key_in_manifest,
                instance_metadata_object=service_instance_metadata,
                common_deps=None
            )
        else:
            ctor_params = inspect.signature(service_class.__init__).parameters
            kwargs_for_ctor = {}
            if 'config' in ctor_params:
                kwargs_for_ctor['config'] = final_service_config
            if 'cfg' in ctor_params:
                kwargs_for_ctor['cfg'] = final_service_config
            if 'event_bus' in ctor_params:
                kwargs_for_ctor['event_bus'] = event_bus
            if 'registry' in ctor_params:
                kwargs_for_ctor['registry'] = registry
            service_instance = service_class(**kwargs_for_ctor)
    except Exception as e_create:
        msg = f"Instantiation failed for shared service '{service_key_in_manifest}': {e_create}"
        logger.error(msg, exc_info=True)
        health_reporter.add_component_status(service_key_in_manifest, ComponentStatus.INSTANTIATION_ERROR, service_instance_metadata, [msg])
        if is_strict_mode:
            raise BootstrapError(msg) from e_create
        return
    
    if service_instance:
        final_requires_init = getattr(service_instance_metadata, 'requires_initialize', False)
        _safe_register_service_instance_with_port(
            registry, service_class, service_instance, service_key_in_manifest,
            service_instance_metadata.category, port_type=port_type,
            description_for_meta=service_instance_metadata.description,
            requires_initialize_override=final_requires_init
        )
        logger.info(f'✓ Shared Service Instantiated and Registered: {service_key_in_manifest} -> {service_class.__name__} (Requires Init: {final_requires_init})')
        health_reporter.add_component_status(service_key_in_manifest, ComponentStatus.INSTANCE_REGISTERED, service_instance_metadata, [])
    else:
        msg = f"Service instantiation returned None for '{service_key_in_manifest}' using class {service_class.__name__}"
        logger.error(msg)
        health_reporter.add_component_status(service_key_in_manifest, ComponentStatus.INSTANTIATION_ERROR, service_instance_metadata, [msg])
        if is_strict_mode:
            raise BootstrapError(msg)


def _safe_register_service_instance_with_port(
    registry: ComponentRegistry,
    service_class: Type,
    service_instance: Any,
    service_key: str,
    category: str,
    port_type: Optional[str] = None,
    description_for_meta: Optional[str] = None,
    requires_initialize_override: Optional[bool] = None
) -> None:
    """Register a service instance with optional port interface."""
    
    # Register by key and class
    registry.register_service_instance(service_key, service_instance)
    registry.register_service_instance(service_class, service_instance)
    
    # Register with port interface if specified
    if port_type:
        try:
            port_interface = import_by_path(port_type)
            registry.register_service_instance(port_interface, service_instance)
            logger.debug(f"Registered service '{service_key}' with port interface: {port_type}")
        except Exception as e:
            logger.warning(f"Failed to register service '{service_key}' with port type '{port_type}': {e}")