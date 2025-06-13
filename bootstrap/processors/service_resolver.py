import asyncio
import dataclasses
import logging
import inspect
from typing import Any, Dict, Type, Optional, Callable

from core.base_component import NireonBaseComponent
from core.lifecycle import ComponentMetadata, ComponentRegistryMissingError
from core.registry.component_registry import ComponentRegistry
from domain.ports.event_bus_port import EventBusPort
from domain.ports.idea_repository_port import IdeaRepositoryPort
from domain.ports.idea_service_port import IdeaServicePort
from application.services.idea_service import IdeaService
from bootstrap.bootstrap_helper.metadata import create_service_metadata

logger = logging.getLogger(__name__)

_ff_manager_lock = asyncio.Lock()

def _safe_register_service_instance(
    registry: ComponentRegistry,
    service_protocol_type: Type,
    instance: Any,
    service_id_for_meta: str,
    category_for_meta: str,
    description_for_meta: Optional[str] = None,
    requires_initialize_override: Optional[bool] = None
) -> None:
    """Safely register a service instance in the registry with proper metadata handling."""
    
    STANDARD_SERVICE_IDS = {
        'LLMPort': 'LLMPort',
        'EmbeddingPort': 'EmbeddingPort', 
        'EventBusPort': 'EventBusPort',
        'IdeaRepositoryPort': 'IdeaRepositoryPort',
        'IdeaService': 'IdeaService',
        'FeatureFlagsManager': 'FeatureFlagsManager',
        'ComponentRegistry': 'ComponentRegistry',
        'SimpleMechanismFactory': 'MechanismFactory',
        'InterfaceValidator': 'InterfaceValidator'
    }
    
    service_id = service_id_for_meta
    
    # Handle metadata
    if hasattr(instance, 'metadata') and isinstance(instance.metadata, ComponentMetadata):
        metadata = instance.metadata
        if metadata.id != service_id:
            logger.warning(f"Metadata ID '{metadata.id}' doesn't match service ID '{service_id}', updating metadata")
            metadata = dataclasses.replace(metadata, id=service_id)
            if hasattr(instance, '_metadata_definition'):
                object.__setattr__(instance, '_metadata_definition', metadata)
    else:
        # Create metadata if not present
        final_requires_initialize = False
        if requires_initialize_override is not None:
            final_requires_initialize = requires_initialize_override
        elif isinstance(instance, NireonBaseComponent):
            final_requires_initialize = True
            logger.warning(f"NireonBaseComponent '{service_id}' missing .metadata for requires_initialize check.")
        
        desc = description_for_meta or f'Service instance for {service_id}'
        metadata = create_service_metadata(
            service_id=service_id,
            service_name=service_id,
            category=category_for_meta,
            description=desc,
            requires_initialize=final_requires_initialize
        )
    
    # Register with manifest ID
    try:
        registry.register(instance, metadata)
        logger.info(f"✓ Registered service '{service_id}' with metadata (manifest ID)")
    except Exception as e:
        logger.error(f"Failed to register '{service_id}' by manifest ID: {e}", exc_info=True)
        raise
    
    # Register by type if registry supports it
    try:
        if hasattr(registry, 'register_service_instance'):
            registry.register_service_instance(service_protocol_type, instance)
            logger.debug(f"✓ Registered service '{service_id}' by type {service_protocol_type.__name__}")
            
            # Handle normalized type key
            if hasattr(registry, 'normalize_key') and callable(registry.normalize_key):
                normalized_type_key = registry.normalize_key(service_protocol_type)
                if normalized_type_key and normalized_type_key != service_id:
                    if hasattr(registry, '_metadata') and isinstance(registry._metadata, dict):
                        if normalized_type_key not in registry._metadata:
                            type_metadata = create_service_metadata(
                                service_id=normalized_type_key,
                                service_name=f'{service_protocol_type.__name__} (type registration)',
                                category=category_for_meta,
                                description=f'Type-based registration for {service_protocol_type.__name__}',
                                requires_initialize=metadata.requires_initialize
                            )
                            registry._metadata[normalized_type_key] = type_metadata
                            logger.debug(f"✓ Metadata registered for type key '{normalized_type_key}'")
    except Exception as e:
        logger.error(f"Failed to register '{service_id}' by type {service_protocol_type.__name__}: {e}", exc_info=True)
    
    # Register standard ID aliases
    standard_id = STANDARD_SERVICE_IDS.get(service_protocol_type.__name__, None)
    if standard_id and standard_id != service_id:
        try:
            if hasattr(registry, '_components'):
                registry._components[standard_id] = instance
                logger.debug(f"✓ Also registered service by standard ID '{standard_id}' -> '{service_id}'")
        except Exception as e:
            logger.debug(f"Could not register standard ID alias '{standard_id}': {e}")
    
    # Register normalized ID
    normalized_id = service_id.lower().replace('-', '_')
    if normalized_id != service_id:
        try:
            if hasattr(registry, '_components'):
                registry._components[normalized_id] = instance
                logger.debug(f"✓ Also registered by normalized ID '{normalized_id}'")
        except Exception:
            pass
            
    # --- START OF MODIFICATION ---
    # Register by simple class name as well for easier lookup in diagnostics and legacy code
    simple_class_name = service_protocol_type.__name__
    if not registry.has_component(simple_class_name):
        try:
            # Directly use register_service_instance if available, it's cleaner.
            if hasattr(registry, 'register_service_instance'):
                 registry.register_service_instance(simple_class_name, instance)
                 logger.debug(f"✓ Also registered service by simple class name alias '{simple_class_name}' -> '{service_id_for_meta}'")
            # Fallback for older registry versions
            elif hasattr(registry, '_components'):
                registry._components[simple_class_name] = instance
                logger.debug(f"✓ Also registered service by simple class name alias '{simple_class_name}' -> '{service_id_for_meta}' (fallback method)")
        except Exception as e:
            logger.debug(f"Could not register simple class name alias '{simple_class_name}': {e}")
    # --- END OF MODIFICATION ---
    
    logger.info(f"✓ Service '{service_id}' fully registered (type: {service_protocol_type.__name__}, category: {category_for_meta})")


def get_or_create_service(
    registry: ComponentRegistry,
    service_protocol_type: Type,
    placeholder_impl_class: Type,
    service_friendly_name: str,
    instance_id_prefix: str = 'placeholder_',
    category: str = 'placeholder_service',
    requires_initialize_for_placeholder: bool = False,
    **kwargs
) -> Any:
    """Get an existing service or create a placeholder if not found."""
    
    # Try to get by service type
    if hasattr(registry, 'get_service_instance'):
        try:
            service_instance = registry.get_service_instance(service_protocol_type)
            logger.info(f"Found '{service_friendly_name}' (type {service_protocol_type.__name__}) via get_service_instance.")
            return service_instance
        except (ComponentRegistryMissingError, AttributeError):
            pass
    
    # Try normalized key
    if hasattr(registry, 'normalize_key') and callable(registry.normalize_key):
        try:
            normalized_key = registry.normalize_key(service_protocol_type)
            service_instance = registry.get(normalized_key)
            logger.info(f"Found '{service_friendly_name}' via normalized key '{normalized_key}'.")
            return service_instance
        except (ComponentRegistryMissingError, AttributeError):
            pass
    
    # Try type matching
    try:
        for comp_id_iter in registry.list_components():
            comp = registry.get(comp_id_iter)
            if isinstance(comp, service_protocol_type):
                logger.info(f"Found '{service_friendly_name}' by type matching (ID: '{comp_id_iter}').")
                return comp
    except Exception as e:
        logger.warning(f"Error during type matching search for '{service_friendly_name}': {e}")
    
    # Create placeholder
    logger.warning(f"'{service_friendly_name}' (type: {service_protocol_type.__name__}) not found. Creating placeholder: {placeholder_impl_class.__name__}.")
    placeholder_instance = placeholder_impl_class(**kwargs)
    
    service_id_for_meta_placeholder = f"{instance_id_prefix}{service_friendly_name.replace('.', '_').replace(' ', '')}"
    
    _safe_register_service_instance(
        registry,
        service_protocol_type,
        placeholder_instance,
        service_id_for_meta_placeholder,
        category,
        description_for_meta=f'Placeholder for {service_friendly_name}',
        requires_initialize_override=requires_initialize_for_placeholder
    )
    
    logger.info(f"Placeholder '{service_friendly_name}' created and registered (ID: {service_id_for_meta_placeholder}).")
    return placeholder_instance


def get_or_create_idea_service(registry: ComponentRegistry, idea_repo: IdeaRepositoryPort, event_bus: EventBusPort) -> IdeaService:
    """Get existing IdeaService or create a new one."""
    try:
        # Try to get existing service
        if hasattr(registry, 'get_service_instance'):
            service_instance = registry.get_service_instance(IdeaService)
        else:
            service_instance = None
            for cid_iter in registry.list_components():
                candidate = registry.get(cid_iter)
                if isinstance(candidate, IdeaService):
                    service_instance = candidate
                    break
            if service_instance is None:
                raise ComponentRegistryMissingError('IdeaService not found by iteration.')
        
        logger.info('IdeaService found in registry.')
        return service_instance
        
    except (ComponentRegistryMissingError, AttributeError):
        logger.info('IdeaService not found. Creating new instance.')
        
        # Create new IdeaService
        new_idea_service: IdeaServicePort = registry.get_service_instance('idea_service')
        
        _safe_register_service_instance(
            registry,
            IdeaService,
            new_idea_service,
            'IdeaService',
            'domain_service',
            description_for_meta='IdeaService created during bootstrap'
        )
        
        logger.info('IdeaService created and registered.')
        return new_idea_service


def find_event_bus_service(registry: ComponentRegistry) -> Optional[EventBusPort]:
    """Find EventBus service in registry using various lookup strategies."""
    
    # Try by service type
    if hasattr(registry, 'get_service_instance'):
        try:
            return registry.get_service_instance(EventBusPort)
        except (ComponentRegistryMissingError, AttributeError):
            logger.debug('EventBusPort not found via get_service_instance. Trying string keys.')
    
    # Try by string keys
    for key_to_try in ['EventBusPort', 'event_bus', 'EventBus']:
        try:
            bus_candidate = registry.get(key_to_try)
            if isinstance(bus_candidate, EventBusPort):
                logger.debug(f"Found EventBusPort by string key '{key_to_try}'.")
                return bus_candidate
        except (ComponentRegistryMissingError, AttributeError):
            continue
    
    # Try duck typing
    logger.debug('EventBusPort not found by string key. Trying duck typing.')
    try:
        for comp_id_iter in registry.list_components():
            comp = registry.get(comp_id_iter)
            
            # Check for EventBus duck typing
            if (hasattr(comp, 'publish') and callable(getattr(comp, 'publish')) and
                hasattr(comp, 'subscribe') and callable(getattr(comp, 'subscribe'))):
                
                # Extra check for get_logger method (common in EventBus implementations)
                if hasattr(comp, 'get_logger') and callable(getattr(comp, 'get_logger')):
                    logger.debug(f'Found EventBusPort by duck typing (with get_logger): {comp_id_iter} ({type(comp).__name__})')
                    return comp
                    
    except Exception as e:
        logger.warning(f'Error during duck typing search for EventBusPort: {e}')
    
    logger.warning('EventBusPort not found in registry.')
    return None