"""
Service Resolver Module
======================

Helper functions for safely registering service instances with the ComponentRegistry,
handling both type-based and port-based registration patterns used in V4.
"""

from __future__ import annotations
import logging
from typing import Any, Type, Optional
from core.registry import ComponentRegistry, ComponentMetadata
from runtime.utils import import_by_path

logger = logging.getLogger(__name__)


def _safe_register_service_instance(
    registry: ComponentRegistry,
    service_class: Type,
    service_instance: Any,
    service_key: str,
    category: str,
    description_for_meta: Optional[str] = None,
    requires_initialize_override: Optional[bool] = None
) -> None:
    """
    Safely register a service instance with the ComponentRegistry.
    
    Args:
        registry: The component registry
        service_class: The class type of the service
        service_instance: The actual service instance
        service_key: The primary key to register under
        category: The category for metadata purposes
        description_for_meta: Optional description
        requires_initialize_override: Override for requires_initialize flag
    """
    try:
        # Register by service key
        registry.register_service_instance(service_key, service_instance)
        logger.debug(f"Registered service instance '{service_key}' by key")
        
        # Register by class type
        registry.register_service_instance(service_class, service_instance)
        logger.debug(f"Registered service instance '{service_key}' by class type {service_class.__name__}")
        
        # If the instance has its own metadata, ensure it's properly linked
        if hasattr(service_instance, 'metadata') and isinstance(service_instance.metadata, ComponentMetadata):
            # The instance should already have proper metadata
            logger.debug(f"Service instance '{service_key}' has built-in metadata")
        else:
            logger.debug(f"Service instance '{service_key}' registered without built-in metadata")
            
    except Exception as e:
        logger.error(f"Failed to register service instance '{service_key}': {e}", exc_info=True)
        raise


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
    """
    Enhanced version of _safe_register_service_instance that supports port type registration.
    
    Args:
        registry: The component registry
        service_class: The class type of the service
        service_instance: The actual service instance
        service_key: The primary key to register under
        category: The category for metadata purposes
        port_type: Optional port interface type path (e.g., "domain.ports.llm_port:LLMPort")
        description_for_meta: Optional description
        requires_initialize_override: Override for requires_initialize flag
    """
    try:
        # First do the standard registration
        _safe_register_service_instance(
            registry, service_class, service_instance, service_key, category,
            description_for_meta, requires_initialize_override
        )
        
        # NEW: Register by port interface type if specified
        if port_type:
            try:
                port_interface = import_by_path(port_type)
                registry.register_service_instance(port_interface, service_instance)
                logger.debug(f"Registered service '{service_key}' with port interface: {port_type}")
            except Exception as e:
                logger.warning(f"Failed to register service '{service_key}' with port type '{port_type}': {e}")
                
    except Exception as e:
        logger.error(f"Failed to register service instance with port '{service_key}': {e}", exc_info=True)
        raise


def resolve_service_by_type_or_key(
    registry: ComponentRegistry,
    service_type: Type,
    fallback_key: Optional[str] = None,
    default: Any = None
) -> Any:
    """
    Resolve a service by type first, then by key if provided, with optional default.
    
    Args:
        registry: The component registry
        service_type: The service type to look for
        fallback_key: Optional fallback key to try
        default: Default value if service not found
        
    Returns:
        The service instance or default value
    """
    try:
        # Try by type first
        return registry.get_service_instance(service_type)
    except Exception:
        if fallback_key:
            try:
                # Try by fallback key
                return registry.get(fallback_key, default)
            except Exception:
                return default
        return default


def register_with_multiple_aliases(
    registry: ComponentRegistry,
    service_instance: Any,
    primary_key: str,
    aliases: list[str],
    service_type: Optional[Type] = None
) -> None:
    """
    Register a service instance with multiple aliases.
    
    Args:
        registry: The component registry
        service_instance: The service instance to register
        primary_key: The primary registration key
        aliases: List of alias keys
        service_type: Optional service type for type-based registration
    """
    try:
        # Register with primary key
        registry.register_service_instance(primary_key, service_instance)
        logger.debug(f"Registered service with primary key: {primary_key}")
        
        # Register with type if provided
        if service_type:
            registry.register_service_instance(service_type, service_instance)
            logger.debug(f"Registered service with type: {service_type.__name__}")
        
        # Register with aliases
        for alias in aliases:
            registry.register_service_instance(alias, service_instance)
            logger.debug(f"Registered service with alias: {alias}")
            
    except Exception as e:
        logger.error(f"Failed to register service with multiple aliases: {e}", exc_info=True)
        raise