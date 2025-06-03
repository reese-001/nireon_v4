import asyncio
import dataclasses # For dataclasses.asdict
import logging
import inspect # For inspect.signature
from typing import Any, Dict, Type, Optional, Callable

from application.components.base import NireonBaseComponent # V4
from application.components.lifecycle import ( # V4
    ComponentMetadata,
    ComponentRegistryMissingError,
)
from core.registry.component_registry import ComponentRegistry # V4
from application.ports.event_bus_port import EventBusPort # V4
from application.ports.idea_repository_port import IdeaRepositoryPort # V4
from application.services.idea_service import IdeaService # V4

from .metadata import create_service_metadata # V4 version from this helper package
logger = logging.getLogger(__name__)

_ff_manager_lock = asyncio.Lock() # Specific to FeatureFlagsManager, might not be needed here generally

class BootstrapError(RuntimeError): # Local exception for clarity
    pass


def _safe_register_service_instance(
    registry: ComponentRegistry,
    service_protocol_type: Type, # The protocol/interface type (e.g., LLMPort)
    instance: Any,               # The actual instance
    service_id_for_meta: str,    # The primary ID for this instance in manifest/registry
    category_for_meta: str,
    description_for_meta: Optional[str] = None,
    requires_initialize_override: Optional[bool] = None,
) -> None:
    """
    Safely registers a service instance into the V4 ComponentRegistry.
    Handles registration by type and by ID, and manages ComponentMetadata.
    """
    already_by_type = False
    already_by_id = False
    
    normalized_type_key = None
    if hasattr(registry, "normalize_key") and callable(registry.normalize_key):
        normalized_type_key = registry.normalize_key(service_protocol_type)

    # Check if already registered by type
    if hasattr(registry, "get_service_instance"): # V3 style, V4 registry might not have this
        try:
            existing_by_type = registry.get_service_instance(service_protocol_type)
            if existing_by_type is instance:
                already_by_type = True
            elif existing_by_type is not None and existing_by_type is not instance:
                logger.warning(
                    f"Service {service_protocol_type.__name__} already registered with a different instance (by type). "
                    f"Overwriting with new instance for type key."
                )
        except (ComponentRegistryMissingError, AttributeError):
            pass # Not found by type, or registry doesn't support get_service_instance
    
    # Check if already registered by ID
    try:
        existing_by_id = registry.get(service_id_for_meta) # V4 uses get() for ID lookup
        if existing_by_id is instance:
            already_by_id = True
        elif existing_by_id is not None and existing_by_id is not instance:
            logger.warning(
                f"Service '{service_id_for_meta}' already registered with a different instance (by ID). "
                f"Overwriting with new instance for ID key."
            )
    except (ComponentRegistryMissingError, AttributeError):
        pass # Not found by ID

    # Determine requires_initialize status
    final_requires_initialize = False # Default for services
    if requires_initialize_override is not None:
        final_requires_initialize = requires_initialize_override
    elif isinstance(instance, NireonBaseComponent): # If it's a Nireon component, use its metadata
        if hasattr(instance, "metadata") and isinstance(instance.metadata, ComponentMetadata):
            final_requires_initialize = instance.metadata.requires_initialize
        else: # Should not happen if NireonBaseComponent is correctly initialized by its constructor
            final_requires_initialize = True 
            logger.warning(f"NireonBaseComponent instance '{service_id_for_meta}' missing valid .metadata attribute for requires_initialize check.")

    # Create metadata for registration by ID
    desc = description_for_meta or f"Service instance for {service_id_for_meta}"
    id_metadata = create_service_metadata( # Uses the V4 helper
        service_id=service_id_for_meta,
        service_name=service_id_for_meta, # Or a more descriptive name if available
        category=category_for_meta,
        description=desc,
        requires_initialize=final_requires_initialize,
    )

    # V4: register_service_instance might be deprecated in favor of just register by ID
    # and letting type resolution happen via iteration or specific type maps if needed.
    # For now, let's assume ComponentRegistry might still have a way to hint type registrations.
    if not already_by_type and hasattr(registry, "register_service_instance"):
        try:
            registry.register_service_instance(service_protocol_type, instance)
            logger.debug(
                f"Service '{service_id_for_meta}' (type: {service_protocol_type.__name__}) "
                f"registered by type key via register_service_instance."
            )
            # If normalized_type_key is different from service_id_for_meta,
            # and registry stores metadata separately for type keys, register metadata for type key too.
            if normalized_type_key and normalized_type_key != service_id_for_meta:
                 if hasattr(registry, '_metadata') and isinstance(registry._metadata, dict): # Check internal structure (use with caution)
                    if normalized_type_key not in registry._metadata:
                        type_metadata = create_service_metadata(
                            service_id=normalized_type_key, # Use normalized_type_key as ID for this metadata entry
                            service_name=f"{service_protocol_type.__name__} (type registration)",
                            category=category_for_meta,
                            description=f"Type-based registration for {service_protocol_type.__name__}",
                            requires_initialize=final_requires_initialize
                        )
                        registry._metadata[normalized_type_key] = type_metadata
                        logger.debug(f"Metadata registered for type key '{normalized_type_key}'.")
        except Exception as e:
            logger.error(
                f"Failed to register '{service_id_for_meta}' by type key {service_protocol_type.__name__} "
                f"via register_service_instance: {e}", exc_info=True
            )

    # Register by ID (this is the primary V4 way)
    if not already_by_id:
        try:
            # If the instance is a NireonBaseComponent, it should have its own .metadata
            # This metadata (with ID corrected if needed) should be used for registration.
            if isinstance(instance, NireonBaseComponent) and hasattr(instance, "metadata") and isinstance(instance.metadata, ComponentMetadata):
                meta_to_register = instance.metadata
                if meta_to_register.id != service_id_for_meta:
                    logger.warning(
                        f"Mismatch for NireonBaseComponent '{service_id_for_meta}': instance metadata ID is '{meta_to_register.id}'. "
                        f"Using '{service_id_for_meta}' for registration key and ensuring instance metadata ID is corrected."
                    )
                    # Correct the ID on the metadata object that will be registered
                    meta_to_register_dict = dataclasses.asdict(meta_to_register)
                    meta_to_register_dict['id'] = service_id_for_meta
                    meta_to_register = ComponentMetadata(**meta_to_register_dict)
                    # Also update the instance's metadata to reflect the canonical ID
                    object.__setattr__(instance, '_metadata_definition', meta_to_register)
                
                registry.register(instance, meta_to_register) # V4 register method
                logger.debug(
                    f"NireonBaseComponent service '{service_id_for_meta}' registered with its metadata "
                    f"(ID: {meta_to_register.id}, ReqInit: {meta_to_register.requires_initialize})."
                )
            else: # For non-NireonBaseComponent services, use the generated id_metadata
                registry.register(instance, id_metadata) # V4 register method
                logger.debug(
                    f"Service '{service_id_for_meta}' registered with generated metadata "
                    f"(ID: {id_metadata.id}, ReqInit: {id_metadata.requires_initialize})."
                )
        except Exception as e:
            logger.error(f"Failed to register '{service_id_for_meta}' with metadata: {e}", exc_info=True)
    
    if already_by_type and already_by_id:
        logger.debug(f"Service '{service_id_for_meta}' (type: {service_protocol_type.__name__}) already fully registered with the same instance.")


def get_or_create_service(
    registry: ComponentRegistry,
    service_protocol_type: Type,
    placeholder_impl_class: Type,
    service_friendly_name: str, # For logging and placeholder ID
    instance_id_prefix: str = "placeholder_",
    category: str = "placeholder_service",
    requires_initialize_for_placeholder: bool = False,
    **kwargs, # Args for placeholder_impl_class constructor
) -> Any:
    """
    Resolves a service from the registry by type. If not found, creates a placeholder,
    registers it, and returns it. This is mainly for core services during early bootstrap.
    """
    # Attempt 1: Resolve by specific type using get_service_instance (if registry supports it)
    if hasattr(registry, "get_service_instance"):
        try:
            service_instance = registry.get_service_instance(service_protocol_type)
            logger.info(
                f"'{service_friendly_name}' found in V4 registry via get_service_instance (type: {service_protocol_type.__name__}). "
                f"Using existing instance: {type(service_instance).__name__}."
            )
            return service_instance
        except (ComponentRegistryMissingError, AttributeError):
            pass # Continue to other methods

    # Attempt 2: Resolve by normalized key if registry supports it
    if hasattr(registry, "normalize_key") and callable(registry.normalize_key):
        try:
            normalized_key = registry.normalize_key(service_protocol_type)
            service_instance = registry.get(normalized_key) # V4 get by ID
            logger.info(
                f"'{service_friendly_name}' found in V4 registry with normalized key '{normalized_key}'. "
                f"Using existing instance: {type(service_instance).__name__}."
            )
            return service_instance
        except (ComponentRegistryMissingError, AttributeError):
            pass # Continue to other methods

    # Attempt 3: Iterate and type-check (fallback)
    try:
        for comp_id in registry.list_components():
            comp = registry.get(comp_id)
            if isinstance(comp, service_protocol_type):
                logger.info(
                    f"'{service_friendly_name}' found in V4 registry by type matching (ID: '{comp_id}'). "
                    f"Using existing instance: {type(comp).__name__}."
                )
                return comp
    except Exception as e: # Catch errors during iteration/get, e.g., if registry is modified
        logger.warning(f"Error during type matching search for '{service_friendly_name}': {e}")


    # If not found, create and register placeholder
    logger.warning(
        f"'{service_friendly_name}' (type: {service_protocol_type.__name__}) not found in V4 registry. "
        f"Creating placeholder: {placeholder_impl_class.__name__}."
    )
    placeholder_instance = placeholder_impl_class(**kwargs)
    
    # Placeholder ID convention
    service_id_for_meta = f"{instance_id_prefix}{service_friendly_name.replace('.', '_').replace(' ', '')}"
    
    _safe_register_service_instance(
        registry,
        service_protocol_type,
        placeholder_instance,
        service_id_for_meta,
        category,
        description_for_meta=f"Placeholder implementation for {service_friendly_name}",
        requires_initialize_override=requires_initialize_for_placeholder
    )
    logger.info(
        f"Placeholder '{service_friendly_name}' instance created and registered (ID: {service_id_for_meta})."
    )
    return placeholder_instance


def get_or_create_idea_service(
    registry: ComponentRegistry, idea_repo: IdeaRepositoryPort, event_bus: EventBusPort
) -> IdeaService:
    """Gets or creates the IdeaService for V4."""
    try:
        # V4: Prefer get_service_instance if available, else iterate and check type
        if hasattr(registry, "get_service_instance"):
            idea_service_instance = registry.get_service_instance(IdeaService)
        else: # Fallback: iterate and check type
            idea_service_instance = None
            for cid in registry.list_components():
                candidate = registry.get(cid)
                if isinstance(candidate, IdeaService):
                    idea_service_instance = candidate
                    break
            if idea_service_instance is None:
                raise ComponentRegistryMissingError("IdeaService not found by iteration.")

        logger.info("IdeaService found in V4 registry. Using existing instance.")
        return idea_service_instance
    except (ComponentRegistryMissingError, AttributeError):
        logger.info("IdeaService not found in V4 registry. Creating new instance.")
        idea_service_instance = IdeaService(repository=idea_repo, event_bus=event_bus)
        _safe_register_service_instance(
            registry,
            IdeaService,
            idea_service_instance,
            "bootstrap_idea_service_v4", # V4 specific ID
            "domain_service",
            description_for_meta="IdeaService created during V4 bootstrap",
        )
        logger.info("IdeaService created and registered for V4.")
        return idea_service_instance

def find_event_bus_service(registry: ComponentRegistry) -> Optional[EventBusPort]:
    """Attempts to find an EventBusPort implementation in the V4 registry."""
    if hasattr(registry, "get_service_instance"):
        try:
            return registry.get_service_instance(EventBusPort)
        except (ComponentRegistryMissingError, AttributeError):
            logger.debug("V4 EventBusPort not found via get_service_instance. Trying string key 'EventBusPort'.")
    # Try common string keys
    for key_attempt in ["EventBusPort", "event_bus", "EventBus"]: # V4: "EventBusPort" is canonical
        try:
            bus_candidate = registry.get(key_attempt)
            if isinstance(bus_candidate, EventBusPort):
                logger.debug(f"Found EventBusPort by string key '{key_attempt}'")
                return bus_candidate
        except (ComponentRegistryMissingError, AttributeError):
            continue
    
    # Fallback: Duck typing (less reliable, but can catch custom implementations)
    logger.debug("EventBusPort not found via string key. Trying V4 duck typing.")
    for comp_id in registry.list_components():
        comp = registry.get(comp_id)
        # V4 EventBusPort defines publish and subscribe
        if hasattr(comp, "publish") and callable(getattr(comp, "publish")) and \
           hasattr(comp, "subscribe") and callable(getattr(comp, "subscribe")):
            # Optional: check for get_logger if it's a convention for your V4 event buses
            if hasattr(comp, "get_logger") and callable(getattr(comp, "get_logger")):
                 logger.debug(f"Found EventBus by duck typing (with get_logger): {comp_id} ({type(comp)})")
                 return comp
            # else: # Basic publish/subscribe is enough to match the Port
            #      logger.debug(f"Found EventBus by basic duck typing (publish/subscribe only): {comp_id} ({type(comp)})")
            #      return comp 
    
    logger.warning("EventBusPort not found in V4 registry through any method.")
    return None