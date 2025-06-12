"""Simple component processor for handling simple component definitions."""

from .component_processing_utils import *


async def process_simple_component(
    comp_def: Dict[str, Any],
    registry: ComponentRegistry,
    mechanism_factory: SimpleMechanismFactory,
    health_reporter: BootstrapHealthReporter,
    run_id: str,
    global_app_config: Dict[str, Any],
    validation_data_store: Any
) -> None:
    """Process a simple component definition and instantiate it."""
    
    component_id = comp_def.get('component_id')
    factory_key = comp_def.get('factory_key')
    component_class_path = comp_def.get('class')
    component_type = comp_def.get('type')
    yaml_config_override = comp_def.get('config', {})
    is_enabled = comp_def.get('enabled', True)
    is_strict_mode = global_app_config.get('bootstrap_strict_mode', True)

    # Create fallback metadata for health reporting
    current_component_metadata_for_reporting = ComponentMetadata(
        id=component_id or 'unknown_id_simple',
        name=factory_key or component_class_path or 'UnknownNameSimple',
        version='0.0.0',
        category=component_type or 'unknown_type_simple',
        requires_initialize=True
    )

    if not is_enabled:
        logger.info(f"Simple component '{component_id}' is disabled. Skipping.")
        health_reporter.add_component_status(
            component_id or 'unknown_disabled_component',
            ComponentStatus.DISABLED,
            current_component_metadata_for_reporting,
            ['Disabled in manifest']
        )
        return

    # Validate required fields
    if not all([component_id, (factory_key or component_class_path), component_type]):
        errmsg = f'Skipping simple component definition due to missing id, (factory_key or class), or type: {comp_def}'
        logger.error(errmsg)
        health_reporter.add_component_status(
            component_id or 'unknown_component_def_error',
            ComponentStatus.DEFINITION_ERROR,
            current_component_metadata_for_reporting,
            [errmsg]
        )
        if is_strict_mode:
            raise BootstrapError(errmsg)
        return

    target_class_or_factory_key = component_class_path if component_class_path else factory_key

    # Get base metadata
    base_metadata_for_class = get_default_metadata(factory_key)
    if not base_metadata_for_class and component_class_path:
        try:
            cls_for_meta = import_by_path(component_class_path)
            if hasattr(cls_for_meta, 'METADATA_DEFINITION'):
                base_metadata_for_class = cls_for_meta.METADATA_DEFINITION
        except Exception:
            pass

    if not base_metadata_for_class:
        logger.warning(f"No default or class-defined metadata found for '{target_class_or_factory_key}'. Using minimal fallback for component '{component_id}'.")
        base_metadata_for_class = ComponentMetadata(
            id=component_id,
            name=component_id,
            version='unknown',
            category=component_type
        )

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
            raise BootstrapError(errmsg) from e_meta
        return

    # Get Pydantic defaults if available
    pydantic_defaults = {}
    if component_class_path:
        try:
            cls_for_pydantic = import_by_path(component_class_path)
            pydantic_defaults = _get_pydantic_defaults(cls_for_pydantic, final_instance_metadata.name)
        except Exception as e_pydantic:
            logger.debug(f"Could not get Pydantic defaults for simple component '{component_id}': {e_pydantic}")

    # Merge configuration
    final_config = ConfigMerger.merge(
        pydantic_defaults,
        yaml_config_override,
        f'{component_id}_simple_config_merge'
    )

    # Store validation data
    if validation_data_store:
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
            cls_to_instantiate = import_by_path(component_class_path)
            instance = await _create_component_instance(
                cls_to_instantiate,
                final_config,
                component_id,
                final_instance_metadata,
                None
            )
        elif factory_key and component_type == 'mechanism' and mechanism_factory:
            instance = mechanism_factory.create_mechanism(
                factory_key,
                final_instance_metadata,
                final_config
            )
        else:
            errmsg = f"Cannot determine instantiation method for simple component '{component_id}'. Need 'class' or a typed 'factory_key'."
            raise BootstrapError(errmsg)

        if instance is None:
            errmsg = f"Component instantiation returned None for simple component '{component_id}' (target: {target_class_or_factory_key})"
            health_reporter.add_component_status(
                component_id,
                ComponentStatus.INSTANTIATION_ERROR,
                final_instance_metadata,
                [errmsg]
            )
            if is_strict_mode:
                raise BootstrapError(errmsg)
            return

        logger.info(f"Instantiated simple component '{component_id}' (Type: {component_type}, Target: {target_class_or_factory_key})")

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

        health_reporter.add_component_status(
            component_id,
            ComponentStatus.INSTANCE_REGISTERED,
            instance.metadata,
            []
        )

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
            raise BootstrapError(errmsg) from e


def _build_component_metadata(
    base_metadata: ComponentMetadata,
    component_id_from_manifest: str,
    manifest_comp_definition: Dict[str, Any]
) -> ComponentMetadata:
    """Build component metadata for an instance."""
    
    if not isinstance(base_metadata, ComponentMetadata):
        raise TypeError(f"base_metadata for '{component_id_from_manifest}' must be ComponentMetadata, got {type(base_metadata)}")

    # Start with base metadata as dict
    instance_metadata_dict = dataclasses.asdict(base_metadata)
    # Override the ID
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
                logger.warning(f"[{component_id_from_manifest}] Unknown key '{key}' or invalid type in metadata_override for simple component. Ignoring.")

    # Handle epistemic tags from manifest
    if 'epistemic_tags' in manifest_comp_definition:
        tags = manifest_comp_definition['epistemic_tags']
        if isinstance(tags, list) and all(isinstance(t, str) for t in tags):
            instance_metadata_dict['epistemic_tags'] = tags
            logger.debug(f'[{component_id_from_manifest}] Using epistemic_tags from manifest: {tags}')
        else:
            logger.warning(f"[{component_id_from_manifest}] Invalid 'epistemic_tags' in manifest, expected list of strings. Using base.")

    # Ensure requires_initialize is set
    if 'requires_initialize' not in instance_metadata_dict:
        instance_metadata_dict['requires_initialize'] = base_metadata.requires_initialize

    return ComponentMetadata(**instance_metadata_dict)