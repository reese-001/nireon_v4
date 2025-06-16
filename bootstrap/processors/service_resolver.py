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
    """
    Registers a component instance against its canonical ID and its primary service protocol type.
    This simplified version prevents registry pollution from multiple alias keys.
    """
    # 1. Prepare the canonical metadata for the component instance.
    metadata: Optional[ComponentMetadata] = None
    if hasattr(instance, 'metadata') and isinstance(instance.metadata, ComponentMetadata):
        metadata = instance.metadata
        # Ensure the metadata ID always matches the canonical ID from the manifest/caller
        if metadata.id != service_id_for_meta:
            logger.warning(
                f"Correcting metadata ID mismatch for '{service_id_for_meta}'. "
                f"Instance has '{metadata.id}', but canonical ID is '{service_id_for_meta}'."
            )
            metadata = dataclasses.replace(metadata, id=service_id_for_meta)
            # Attempt to fix the instance's internal metadata reference as well
            if hasattr(instance, '_metadata_definition'):
                object.__setattr__(instance, '_metadata_definition', metadata)
    else:
        # Create metadata if the component doesn't have it.
        final_requires_initialize = requires_initialize_override if requires_initialize_override is not None else False
        desc = description_for_meta or f'Service instance for {service_id_for_meta}'
        metadata = create_service_metadata(
            service_id=service_id_for_meta,
            service_name=service_id_for_meta,
            category=category_for_meta,
            description=desc,
            requires_initialize=final_requires_initialize
        )

    # 2. Register the component instance with its canonical ID and metadata.
    # This is the primary registration. All lookups by ID (e.g., 'explorer_instance_01') will use this.
    try:
        registry.register(instance, metadata)
        logger.info(f"✓ Registered component '{service_id_for_meta}' with its canonical ID.")
    except Exception as e:
        logger.error(f"Failed to register '{service_id_for_meta}' with its canonical ID: {e}", exc_info=True)
        raise

    # 3. Register the instance against its service protocol type for dependency injection.
    # This allows `registry.get_service_instance(LLMPort)` to work.
    try:
        if hasattr(registry, 'register_service_instance'):
            # The key here is the *type* object itself (e.g., LLMPort), not its string name.
            registry.register_service_instance(service_protocol_type, instance)
            logger.debug(f"✓ Mapped protocol type {service_protocol_type.__name__} to instance '{service_id_for_meta}'.")
        else:
            logger.warning(f"Registry does not support 'register_service_instance'. Type-based lookup for {service_protocol_type.__name__} may fail.")
            
    except Exception as e:
        logger.error(f"Failed to register service by type {service_protocol_type.__name__}: {e}", exc_info=True)
        # Don't re-raise, as the primary registration by ID might be sufficient.

    logger.info(f"✓ Service '{service_id_for_meta}' fully registered (Type: {service_protocol_type.__name__}, Category: {category_for_meta})")

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
    
    def _safe_register_service_instance(
        registry: ComponentRegistry,
        service_protocol_type: Type,
        instance: Any,
        service_id_for_meta: str,
        category_for_meta: str,
        description_for_meta: Optional[str] = None,
        requires_initialize_override: Optional[bool] = None
    ) -> None:
        """
        Registers a component instance against its canonical ID and its primary service protocol type.
        This simplified version prevents registry pollution from multiple alias keys.
        """
        # 1. Prepare the canonical metadata for the component instance.
        metadata: Optional[ComponentMetadata] = None
        if hasattr(instance, 'metadata') and isinstance(instance.metadata, ComponentMetadata):
            metadata = instance.metadata
            # Ensure the metadata ID always matches the canonical ID from the manifest/caller
            if metadata.id != service_id_for_meta:
                logger.warning(
                    f"Correcting metadata ID mismatch for '{service_id_for_meta}'. "
                    f"Instance has '{metadata.id}', but canonical ID is '{service_id_for_meta}'."
                )
                metadata = dataclasses.replace(metadata, id=service_id_for_meta)
                # Attempt to fix the instance's internal metadata reference as well
                if hasattr(instance, '_metadata_definition'):
                    object.__setattr__(instance, '_metadata_definition', metadata)
        else:
            # Create metadata if the component doesn't have it.
            final_requires_initialize = requires_initialize_override if requires_initialize_override is not None else False
            desc = description_for_meta or f'Service instance for {service_id_for_meta}'
            metadata = create_service_metadata(
                service_id=service_id_for_meta,
                service_name=service_id_for_meta,
                category=category_for_meta,
                description=desc,
                requires_initialize=final_requires_initialize
            )

        # 2. Register the component instance with its canonical ID and metadata.
        # This is the primary registration. All lookups by ID (e.g., 'explorer_instance_01') will use this.
        try:
            registry.register(instance, metadata)
            logger.info(f"✓ Registered component '{service_id_for_meta}' with its canonical ID.")
        except Exception as e:
            logger.error(f"Failed to register '{service_id_for_meta}' with its canonical ID: {e}", exc_info=True)
            raise

        # 3. Register the instance against its service protocol type for dependency injection.
        # This allows `registry.get_service_instance(LLMPort)` to work.
        try:
            if hasattr(registry, 'register_service_instance'):
                # The key here is the *type* object itself (e.g., LLMPort), not its string name.
                registry.register_service_instance(service_protocol_type, instance)
                logger.debug(f"✓ Mapped protocol type {service_protocol_type.__name__} to instance '{service_id_for_meta}'.")
            else:
                logger.warning(f"Registry does not support 'register_service_instance'. Type-based lookup for {service_protocol_type.__name__} may fail.")
                
        except Exception as e:
            logger.error(f"Failed to register service by type {service_protocol_type.__name__}: {e}", exc_info=True)
            # Don't re-raise, as the primary registration by ID might be sufficient.

        logger.info(f"✓ Service '{service_id_for_meta}' fully registered (Type: {service_protocol_type.__name__}, Category: {category_for_meta})")


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