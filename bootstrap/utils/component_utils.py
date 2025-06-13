# nireon_v4/bootstrap/utils/component_utils.py
from __future__ import annotations # Moved to the top
from __future__ import absolute_import # Moved to the top

import dataclasses
import importlib
import inspect
import logging
from typing import Any, Dict, List, Optional, Type, Union, cast, TYPE_CHECKING # Added TYPE_CHECKING

from pydantic import BaseModel

from core.base_component import NireonBaseComponent
from core.lifecycle import ComponentMetadata

# For TYPE_CHECKING to resolve ComponentInstantiationError without runtime circular import
if TYPE_CHECKING:
    from bootstrap.exceptions import ComponentInstantiationError


logger = logging.getLogger(__name__)

__all__ = [
    'get_pydantic_defaults',
    'create_component_instance',
    'inject_dependencies',
    'validate_component_interfaces',
    'configure_component_logging',
    'prepare_component_metadata',
]


def get_pydantic_defaults(component_class: Type, component_name: str) -> Dict[str, Any]:
    """
    Extracts default configuration values from a component's Pydantic ConfigModel
    or a legacy DEFAULT_CONFIG attribute.
    """
    defaults = {}
    try:
        if hasattr(component_class, 'ConfigModel'):
            config_model = component_class.ConfigModel
            if hasattr(config_model, 'model_fields'):  # Pydantic V2
                for field_name, field_info in config_model.model_fields.items():
                    if hasattr(field_info, 'default') and field_info.default is not ...:
                        defaults[field_name] = field_info.default
                    elif hasattr(field_info, 'default_factory') and field_info.default_factory is not None:
                        try:
                            defaults[field_name] = field_info.default_factory()
                        except Exception:
                            logger.debug(f"Error calling default_factory for {component_name}.{field_name}")
                            pass # Keep default as not set
            elif hasattr(config_model, '__fields__'):  # Pydantic V1
                for field_name, field in config_model.__fields__.items():
                    if hasattr(field, 'default') and field.default is not ...: # Pydantic V1
                        defaults[field_name] = field.default
                    elif hasattr(field, 'default_factory') and field.default_factory is not None:
                        try:
                            defaults[field_name] = field.default_factory()
                        except Exception:
                            logger.debug(f"Error calling default_factory for {component_name}.{field_name} (V1)")
                            pass
        # Fallback or supplement with DEFAULT_CONFIG for components not fully on Pydantic
        if hasattr(component_class, 'DEFAULT_CONFIG'):
            if isinstance(component_class.DEFAULT_CONFIG, dict):
                # Pydantic defaults should take precedence
                merged_defaults = component_class.DEFAULT_CONFIG.copy()
                merged_defaults.update(defaults)
                defaults = merged_defaults
            else:
                logger.warning(
                    f"Component {component_name} has a DEFAULT_CONFIG attribute that is not a dict. Skipping."
                )

        if defaults:
            logger.debug(f'Found {len(defaults)} Pydantic/DEFAULT_CONFIG defaults for {component_name}')
        else:
            logger.debug(f'No Pydantic/DEFAULT_CONFIG defaults found for {component_name}')
    except Exception as e:
        logger.debug(f'Error extracting Pydantic/DEFAULT_CONFIG defaults for {component_name}: {e}')
    return defaults


# Consider making this configurable as suggested
# Example:
# DEFAULT_GATEWAY_SENTINEL_PARAMS = ['llm_router', 'parameter_service', 'frame_factory', 'budget_manager', 'event_bus']
# Can be updated by application config if necessary.
# For now, keeping it hardcoded as per the provided fix.

async def create_component_instance(
    component_class: Type[NireonBaseComponent],
    resolved_config_for_instance: Dict[str, Any],
    instance_id: str,
    instance_metadata_object: ComponentMetadata,
    common_deps: Optional[Dict[str, Any]] = None,
    # extra_sentinel_params_for_gateway: Optional[List[str]] = None # Example for configurability
) -> NireonBaseComponent:
    """
    Creates an instance of a NireonBaseComponent.
    This is the canonical version, adapted from processors/enhanced_components.py
    to handle gateway_params and common_deps as a dictionary.
    """
    try:
        ctor_signature = inspect.signature(component_class.__init__)
        ctor_params = ctor_signature.parameters
        kwargs: Dict[str, Any] = {}

        if 'config' in ctor_params:
            kwargs['config'] = resolved_config_for_instance
        elif 'cfg' in ctor_params: # Support for alternative config naming
            kwargs['cfg'] = resolved_config_for_instance

        if 'metadata_definition' in ctor_params:
            kwargs['metadata_definition'] = instance_metadata_object
        elif 'metadata' in ctor_params: # Support for alternative metadata naming
            kwargs['metadata'] = instance_metadata_object
        
        # Inject common dependencies if provided and expected by constructor
        if common_deps:
            for dep_name, dep_value in common_deps.items():
                if dep_name in ctor_params:
                    kwargs[dep_name] = dep_value
                # Test for **kwargs scenario:
                # If 'kwargs' (or similar like 'extra_args') is a VAR_KEYWORD parameter,
                # do not inject individual common_deps that are not explicitly named,
                # unless a specific strategy for **kwargs is adopted.
                # Current logic correctly only injects if dep_name is an explicit parameter.

        # Special handling for MechanismGateway
        if component_class.__name__ == 'MechanismGateway':
            # gateway_params_to_check = DEFAULT_GATEWAY_SENTINEL_PARAMS + (extra_sentinel_params_for_gateway or [])
            gateway_params_to_check = ['llm_router', 'parameter_service', 'frame_factory', 'budget_manager', 'event_bus']
            for param_name in gateway_params_to_check:
                if param_name in ctor_params and param_name not in kwargs:
                    kwargs[param_name] = None
                    logger.debug(f"Ensuring '{param_name}' is in kwargs for MechanismGateway, set to None as not in common_deps.")


        logger.debug(f'Attempting to create component {instance_id} ({component_class.__name__}) with kwargs: {list(kwargs.keys())}')
        instance = component_class(**kwargs)

        if hasattr(instance, 'component_id'):
            if instance.component_id != instance_id:
                logger.debug(f"Updating component_id on instance from '{instance.component_id}' to '{instance_id}'")
                object.__setattr__(instance, 'component_id', instance_id) 
        elif isinstance(instance, NireonBaseComponent):
             object.__setattr__(instance, '_component_id', instance_id)


        if hasattr(instance, 'metadata') and isinstance(instance.metadata, ComponentMetadata):
            if instance.metadata.id != instance_id:
                logger.debug(f"Updating metadata.id on instance from '{instance.metadata.id}' to '{instance_id}'")
                try:
                    updated_meta = dataclasses.replace(instance.metadata, id=instance_id)
                    object.__setattr__(instance, '_metadata_definition', updated_meta) 
                except TypeError: 
                     instance.metadata.id = instance_id 
        elif isinstance(instance, NireonBaseComponent): 
            object.__setattr__(instance, '_metadata_definition', instance_metadata_object)


        logger.info(f'Successfully created component instance: {instance_id} ({component_class.__name__})')
        return cast(NireonBaseComponent, instance)
    except Exception as e:
        logger.error(f'Failed to create component instance {instance_id} ({component_class.__name__}): {e}', exc_info=True)
        # Local import for ComponentInstantiationError due to potential import order issues
        from bootstrap.exceptions import ComponentInstantiationError 
        raise ComponentInstantiationError(f"Instantiation failed for '{instance_id}': {e}", component_id=instance_id) from e


def inject_dependencies(instance: NireonBaseComponent, dependency_map: Dict[str, Any], registry: Optional[Any]=None) -> None:
    """
    Injects dependencies into a component instance.
    Canonical version from bootstrap_helper/enhanced_components.py.
    """
    # TODO: Consider adding _ensure_not_initialized check if component has such a state
    # if hasattr(instance, 'is_initialized') and instance.is_initialized:
    #     logger.warning(f"Attempting to inject dependencies into already initialized component {instance.component_id}")
    #     # Depending on policy, either return or raise an error
    try:
        for dep_name, dep_value in dependency_map.items():
            private_attr_name = f'_{dep_name}'
            if hasattr(instance, private_attr_name):
                setattr(instance, private_attr_name, dep_value)
                logger.debug(f"Injected dependency '{dep_name}' into {instance.component_id} (as {private_attr_name})")
            elif hasattr(instance, dep_name):
                setattr(instance, dep_name, dep_value)
                logger.debug(f"Injected dependency '{dep_name}' into {instance.component_id}")
            else:
                logger.warning(f"Component {instance.component_id} does not have an attribute '{dep_name}' or '{private_attr_name}' for dependency injection.")
    except Exception as e:
        logger.warning(f'Failed to inject dependencies into {instance.component_id}: {e}')


def validate_component_interfaces(instance: NireonBaseComponent, expected_interfaces: List[Type]) -> List[str]:
    """
    Validates if a component instance implements expected interfaces.
    Canonical version from bootstrap_helper/enhanced_components.py.
    """
    errors: List[str] = []
    try:
        for interface_protocol in expected_interfaces:
            if not isinstance(instance, interface_protocol):
                errors.append(f'Component {instance.component_id} does not implement {interface_protocol.__name__}')
    except Exception as e:
        errors.append(f'Interface validation failed for {instance.component_id}: {e}')
    return errors


def configure_component_logging(instance: NireonBaseComponent, log_level: Optional[str]=None, log_prefix: Optional[str]=None) -> None:
    """
    Configures logging for a component instance.
    Canonical version from bootstrap_helper/enhanced_components.py.
    """
    try:
        if hasattr(instance, '_configure_logging') and callable(instance._configure_logging):
            instance._configure_logging(log_level=log_level, log_prefix=log_prefix)
            logger.debug(f"Called _configure_logging for {instance.component_id}")
        elif hasattr(instance, 'logger') and isinstance(instance.logger, logging.Logger):
            if log_level:
                level_to_set = getattr(logging, log_level.upper(), None)
                if level_to_set:
                    instance.logger.setLevel(level_to_set)
                    logger.debug(f"Set log level for {instance.component_id} logger to {log_level.upper()}")
                else:
                    logger.warning(f"Invalid log level '{log_level}' for {instance.component_id}")
            if log_prefix and not (hasattr(instance, '_configure_logging') and callable(instance._configure_logging)):
                 logger.debug(f"Log prefix '{log_prefix}' provided for {instance.component_id}, but no generic way to apply it without _configure_logging method.")

    except Exception as e:
        logger.warning(f'Failed to configure logging for {instance.component_id}: {e}')


def prepare_component_metadata(
    base_metadata: ComponentMetadata,
    instance_id: str,
    config_overrides: Optional[Dict[str, Any]] = None
) -> ComponentMetadata:
    """
    Prepares the final ComponentMetadata for an instance, applying overrides.
    Canonical version from bootstrap_helper/enhanced_components.py.
    """
    if not isinstance(base_metadata, ComponentMetadata):
        logger.error(f"Base metadata for {instance_id} is not a ComponentMetadata instance. Type: {type(base_metadata)}")
        if isinstance(base_metadata, dict) and 'id' in base_metadata and 'name' in base_metadata: 
             base_metadata = ComponentMetadata(**base_metadata)
        else:
            base_metadata = ComponentMetadata(id=instance_id, name=instance_id, version="0.0.0", category="unknown")

    instance_metadata = dataclasses.replace(base_metadata, id=instance_id)

    if config_overrides:
        metadata_overrides_from_config = config_overrides.get('metadata_override', {})
        if metadata_overrides_from_config:
            logger.debug(f"Applying metadata_override from config for {instance_id}: {metadata_overrides_from_config}")
            try:
                current_meta_dict = dataclasses.asdict(instance_metadata)
                current_meta_dict.update(metadata_overrides_from_config)
                instance_metadata = ComponentMetadata(**current_meta_dict)
            except Exception as e:
                logger.error(f"Error applying metadata_override for {instance_id}: {e}. Using metadata before override attempt.")

    return instance_metadata