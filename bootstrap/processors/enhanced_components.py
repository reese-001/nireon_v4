from __future__ import annotations
import inspect
import logging
from typing import Any, Dict, Type, Optional

from core.base_component import NireonBaseComponent
from core.lifecycle import ComponentMetadata

logger = logging.getLogger(__name__)


async def _create_component_instance(
    component_class: Type[NireonBaseComponent],
    resolved_config_for_instance: Dict[str, Any],
    instance_id: str,
    instance_metadata_object: ComponentMetadata,
    common_deps: Optional[Dict[str, Any]] = None
) -> NireonBaseComponent:
    """Create an instance of a NireonBaseComponent with proper configuration."""
    
    try:
        ctor_signature = inspect.signature(component_class.__init__)
        ctor_params = ctor_signature.parameters
        kwargs = {}

        # Add config parameter
        if 'config' in ctor_params:
            kwargs['config'] = resolved_config_for_instance
        elif 'cfg' in ctor_params:
            kwargs['cfg'] = resolved_config_for_instance

        # Add metadata parameter
        if 'metadata_definition' in ctor_params:
            kwargs['metadata_definition'] = instance_metadata_object
        elif 'metadata' in ctor_params:
            kwargs['metadata'] = instance_metadata_object

        # Add common dependencies if provided
        if common_deps:
            for dep_name, dep_value in common_deps.items():
                if dep_name in ctor_params:
                    kwargs[dep_name] = dep_value

        # Special handling for MechanismGateway
        if component_class.__name__ == 'MechanismGateway':
            gateway_params = ['llm_router', 'parameter_service', 'frame_factory', 'budget_manager', 'event_bus']
            for param in gateway_params:
                if param in ctor_params and param not in kwargs:
                    kwargs[param] = None

        logger.debug(f'Creating component {component_class.__name__} with kwargs: {list(kwargs.keys())}')
        
        # Create the instance
        instance = component_class(**kwargs)

        # Ensure the component ID matches
        if hasattr(instance, 'component_id'):
            if instance.component_id != instance_id:
                logger.debug(f"Updating component_id from '{instance.component_id}' to '{instance_id}'")
                instance.component_id = instance_id

        # Ensure the metadata ID matches
        if hasattr(instance, 'metadata'):
            if instance.metadata.id != instance_id:
                instance.metadata.id = instance_id
                logger.debug(f"Updated metadata ID to '{instance_id}'")

        logger.info(f'Successfully created component instance: {instance_id} ({component_class.__name__})')
        return instance

    except Exception as e:
        logger.error(f'Failed to create component instance {instance_id} ({component_class.__name__}): {e}', exc_info=True)
        raise


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
    expected_interfaces: list[Type]
) -> list[str]:
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
    
    import dataclasses
    
    instance_metadata = dataclasses.replace(base_metadata, id=instance_id)
    
    if config_overrides:
        metadata_overrides = config_overrides.get('metadata_override', {})
        if metadata_overrides:
            metadata_dict = dataclasses.asdict(instance_metadata)
            metadata_dict.update(metadata_overrides)
            instance_metadata = ComponentMetadata(**metadata_dict)
    
    return instance_metadata