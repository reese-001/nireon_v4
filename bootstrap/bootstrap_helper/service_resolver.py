# bootstrap/bootstrap_helper/service_registration.py (or similar path)
import asyncio
import dataclasses # For dataclasses.asdict
import logging
import inspect # For inspect.signature
from typing import Any, Dict, Type, Optional, Callable

from core.base_component import NireonBaseComponent
from core.lifecycle import (
    ComponentMetadata,
    ComponentRegistryMissingError,
)
from core.registry.component_registry import ComponentRegistry
from domain.ports.event_bus_port import EventBusPort
from domain.ports.idea_repository_port import IdeaRepositoryPort
from domain.ports.idea_service_port import IdeaServicePort

from application.services.idea_service import IdeaService



from .metadata import create_service_metadata
logger = logging.getLogger(__name__)

_ff_manager_lock = asyncio.Lock() # Retained as it might be used by other code not shown



def _safe_register_service_instance(
    registry: ComponentRegistry,
    service_protocol_type: Type,
    instance: Any,
    service_id_for_meta: str,
    category_for_meta: str,
    description_for_meta: Optional[str] = None,
    requires_initialize_override: Optional[bool] = None,
) -> None:
    """
    Safely registers a service instance into the ComponentRegistry.
    Handles registration by type and by ID, and manages ComponentMetadata.
    """
    STANDARD_SERVICE_IDS = {
        'LLMPort': 'LLMPort',
        'EmbeddingPort': 'EmbeddingPort',
        'EventBusPort': 'EventBusPort',
        'IdeaRepositoryPort': 'IdeaRepositoryPort',
        'IdeaService': 'IdeaService',
        'FeatureFlagsManager': 'FeatureFlagsManager',
        'ComponentRegistry': 'ComponentRegistry',  # CHANGED: Align with PascalCase canonical ID
        'SimpleMechanismFactory': 'MechanismFactory',
        'InterfaceValidator': 'InterfaceValidator'
    }
    
    standard_id = STANDARD_SERVICE_IDS.get(service_id_for_meta, service_id_for_meta)
    
    already_by_type = False
    already_by_id = False
    
    normalized_type_key = None
    if hasattr(registry, "normalize_key") and callable(registry.normalize_key):
        normalized_type_key = registry.normalize_key(service_protocol_type)

    if hasattr(registry, "get_service_instance"):
        try:
            existing_by_type = registry.get_service_instance(service_protocol_type)
            if existing_by_type is instance:
                already_by_type = True
            elif existing_by_type is not None:
                logger.warning(
                    f"Service type {service_protocol_type.__name__} already registered with a different instance. Overwriting."
                )
        except (ComponentRegistryMissingError, AttributeError):
            pass
    
    try:
        existing_by_id = registry.get(standard_id)
        if existing_by_id is instance:
            already_by_id = True
        elif existing_by_id is not None:
            logger.warning(
                f"Service ID '{standard_id}' already registered with a different instance. Overwriting."
            )
    except (ComponentRegistryMissingError, AttributeError):
        pass

    final_requires_initialize = False
    if requires_initialize_override is not None:
        final_requires_initialize = requires_initialize_override
    elif isinstance(instance, NireonBaseComponent):
        if hasattr(instance, "metadata") and isinstance(instance.metadata, ComponentMetadata):
            final_requires_initialize = instance.metadata.requires_initialize
        else:
            final_requires_initialize = True 
            logger.warning(f"NireonBaseComponent '{standard_id}' missing .metadata for requires_initialize check.")

    desc = description_for_meta or f"Service instance for {standard_id}"
    id_metadata = create_service_metadata(
        service_id=standard_id,
        service_name=standard_id,
        category=category_for_meta,
        description=desc,
        requires_initialize=final_requires_initialize,
    )
    

    if not already_by_type and hasattr(registry, "register_service_instance"):
        try:
            registry.register_service_instance(service_protocol_type, instance)
            logger.debug(
                f"Service '{standard_id}' (type: {service_protocol_type.__name__}) registered by type key."
            )
            if normalized_type_key and normalized_type_key != standard_id:
                 if hasattr(registry, '_metadata') and isinstance(registry._metadata, dict):
                    if normalized_type_key not in registry._metadata:
                        type_metadata = create_service_metadata(
                            service_id=normalized_type_key,
                            service_name=f"{service_protocol_type.__name__} (type registration)",
                            category=category_for_meta,
                            description=f"Type-based registration for {service_protocol_type.__name__}",
                            requires_initialize=final_requires_initialize
                        )
                        registry._metadata[normalized_type_key] = type_metadata
                        logger.debug(f"Metadata registered for type key '{normalized_type_key}'.")
        except Exception as e:
            logger.error(
                f"Failed to register '{standard_id}' by type {service_protocol_type.__name__} via register_service_instance: {e}", exc_info=True
            )

    if not already_by_id:
        try:
            meta_to_register_for_id = id_metadata # Default to generated metadata
            if isinstance(instance, NireonBaseComponent) and hasattr(instance, "metadata") and isinstance(instance.metadata, ComponentMetadata):
                # If it's a Nireon component, use its own metadata, but ensure the ID matches standard_id
                component_meta = instance.metadata
                if component_meta.id != standard_id:
                    logger.warning(
                        f"NireonBaseComponent '{standard_id}': instance metadata ID is '{component_meta.id}'. "
                        f"Using '{standard_id}' for registration and ensuring instance metadata ID is canonical."
                    )
                    # Create a new metadata object with the corrected ID for registration
                    meta_dict = dataclasses.asdict(component_meta)
                    meta_dict['id'] = standard_id
                    meta_to_register_for_id = ComponentMetadata(**meta_dict)
                    # Also update the instance's internal metadata to reflect the canonical ID
                    # This direct attribute setting is generally discouraged but might be necessary here
                    # if the component doesn't offer a setter.
                    try:
                        # Attempt a safer update if a method exists, otherwise direct.
                        if hasattr(instance, 'update_metadata_id'):
                            instance.update_metadata_id(standard_id)
                        else:
                             object.__setattr__(instance, '_metadata_definition', meta_to_register_for_id)
                    except Exception as e_set_meta:
                        logger.error(f"Could not update instance metadata ID for {standard_id}: {e_set_meta}")

                else: # IDs match, use component's metadata directly
                    meta_to_register_for_id = component_meta
            
            registry.register(instance, meta_to_register_for_id)
            logger.debug(
                f"Service '{standard_id}' registered by ID with metadata (ReqInit: {meta_to_register_for_id.requires_initialize})."
            )
        except Exception as e:
            logger.error(f"Failed to register '{standard_id}' by ID with metadata: {e}", exc_info=True)
    
    if already_by_type and already_by_id:
        logger.debug(f"Service '{standard_id}' (type: {service_protocol_type.__name__}) already fully registered.")


def get_or_create_service(
    registry: ComponentRegistry,
    service_protocol_type: Type,
    placeholder_impl_class: Type,
    service_friendly_name: str,
    instance_id_prefix: str = "placeholder_",
    category: str = "placeholder_service",
    requires_initialize_for_placeholder: bool = False,
    **kwargs,
) -> Any:
    """
    Resolves a service from the registry by type. If not found, creates a placeholder,
    registers it, and returns it.
    """
    if hasattr(registry, "get_service_instance"):
        try:
            service_instance = registry.get_service_instance(service_protocol_type)
            logger.info(f"Found '{service_friendly_name}' (type {service_protocol_type.__name__}) via get_service_instance.")
            return service_instance
        except (ComponentRegistryMissingError, AttributeError):
            pass

    if hasattr(registry, "normalize_key") and callable(registry.normalize_key):
        try:
            normalized_key = registry.normalize_key(service_protocol_type)
            service_instance = registry.get(normalized_key)
            logger.info(f"Found '{service_friendly_name}' via normalized key '{normalized_key}'.")
            return service_instance
        except (ComponentRegistryMissingError, AttributeError):
            pass

    try:
        for comp_id_iter in registry.list_components(): # Use a different variable name for iteration
            comp = registry.get(comp_id_iter)
            if isinstance(comp, service_protocol_type):
                logger.info(f"Found '{service_friendly_name}' by type matching (ID: '{comp_id_iter}').")
                return comp
    except Exception as e:
        logger.warning(f"Error during type matching search for '{service_friendly_name}': {e}")

    logger.warning(
        f"'{service_friendly_name}' (type: {service_protocol_type.__name__}) not found. Creating placeholder: {placeholder_impl_class.__name__}."
    )
    placeholder_instance = placeholder_impl_class(**kwargs)
    
    service_id_for_meta_placeholder = f"{instance_id_prefix}{service_friendly_name.replace('.', '_').replace(' ', '')}"
    
    _safe_register_service_instance(
        registry,
        service_protocol_type,
        placeholder_instance,
        service_id_for_meta_placeholder,
        category,
        description_for_meta=f"Placeholder for {service_friendly_name}",
        requires_initialize_override=requires_initialize_for_placeholder
    )
    logger.info(f"Placeholder '{service_friendly_name}' created and registered (ID: {service_id_for_meta_placeholder}).")
    return placeholder_instance


def get_or_create_idea_service(
    registry: ComponentRegistry, idea_repo: IdeaRepositoryPort, event_bus: EventBusPort
) -> IdeaService:
    """Gets or creates the IdeaService."""
    try:
        if hasattr(registry, "get_service_instance"):
            service_instance = registry.get_service_instance(IdeaService)
        else:
            service_instance = None
            for cid_iter in registry.list_components(): # Use a different variable name
                candidate = registry.get(cid_iter)
                if isinstance(candidate, IdeaService):
                    service_instance = candidate
                    break
            if service_instance is None:
                raise ComponentRegistryMissingError("IdeaService not found by iteration.")
        logger.info("IdeaService found in registry.")
        return service_instance
    except (ComponentRegistryMissingError, AttributeError):
        logger.info("IdeaService not found. Creating new instance.")
   
        new_idea_service: IdeaServicePort = registry.get_service_instance("idea_service")
        _safe_register_service_instance(
            registry,
            IdeaService,
            new_idea_service,
            "IdeaService", # Use canonical ID if defined in STANDARD_SERVICE_IDS, or direct
            "domain_service",
            description_for_meta="IdeaService created during bootstrap",
        )
        logger.info("IdeaService created and registered.")
        return new_idea_service

def find_event_bus_service(registry: ComponentRegistry) -> Optional[EventBusPort]:
    """Attempts to find an EventBusPort implementation in the registry."""
    if hasattr(registry, "get_service_instance"):
        try:
            return registry.get_service_instance(EventBusPort)
        except (ComponentRegistryMissingError, AttributeError):
            logger.debug("EventBusPort not found via get_service_instance. Trying string keys.")
    
    for key_to_try in ["EventBusPort", "event_bus", "EventBus"]: # "EventBusPort" is primary
        try:
            bus_candidate = registry.get(key_to_try)
            if isinstance(bus_candidate, EventBusPort):
                logger.debug(f"Found EventBusPort by string key '{key_to_try}'.")
                return bus_candidate
        except (ComponentRegistryMissingError, AttributeError):
            continue
    
    logger.debug("EventBusPort not found by string key. Trying duck typing.")
    try:
        for comp_id_iter in registry.list_components(): # Use a different variable name
            comp = registry.get(comp_id_iter)
            if hasattr(comp, "publish") and callable(getattr(comp, "publish")) and \
               hasattr(comp, "subscribe") and callable(getattr(comp, "subscribe")):
                # Optional check for get_logger
                if hasattr(comp, "get_logger") and callable(getattr(comp, "get_logger")):
                     logger.debug(f"Found EventBusPort by duck typing (with get_logger): {comp_id_iter} ({type(comp).__name__})")
                     return comp
                # else: # If basic publish/subscribe is enough
                #    logger.debug(f"Found EventBusPort by duck typing (publish/subscribe only): {comp_id_iter} ({type(comp).__name__})")
                #    return comp
    except Exception as e:
        logger.warning(f"Error during duck typing search for EventBusPort: {e}")
            
    logger.warning("EventBusPort not found in registry.")
    return None
