import logging
from typing import Any, Dict, List, Optional, Type, Union
from datetime import datetime, timezone
import typing

from core.lifecycle import ComponentMetadata, ComponentRegistryMissingError
from domain.ports.event_bus_port import EventBusPort

logger = logging.getLogger(__name__)

class ComponentRegistry:
    """
    Central registry for all components in the NIREON system.
    Supports registration by ID, type, and maintains metadata and certification.
    """
    
    def __init__(self, event_bus: Optional[EventBusPort] = None):
        self._components: Dict[str, Any] = {}
        self._metadata: Dict[str, ComponentMetadata] = {}
        self._certifications: Dict[str, Dict[str, Any]] = {}
        self._service_instances: Dict[Type, Any] = {}  # Type -> instance mapping
        self._event_bus = event_bus
        self._registration_order: List[str] = []
        logger.info("ComponentRegistry initialized")
    
    def normalize_key(self, key: Any) -> str:
        """
        Normalize a key for consistent lookups.
        IMPORTANT: For string keys, we preserve them as-is to maintain manifest IDs.
        """
        if isinstance(key, type):
            # For types, use the full module.class name
            return f"{key.__module__}.{key.__name__}"
        elif isinstance(key, str):
            # For strings, DON'T normalize them - keep the exact ID from manifest
            # This preserves 'llm_router_main' as-is
            return key
        else:
            # For other objects, use their string representation
            return str(key)
        
    ## REMOVE WHEN POSSIBLE: TODO
    def resolve(self, key: typing.Any):
        """
        DEPRECATED - kept only so legacy helpers don’t break.
        Prefer `get_service_instance()` for protocols or `get()` for IDs.
        """
        if isinstance(key, str):
            return self.get(key)
        return self.get_service_instance(key)
    
    
    def register(self, component: Any, metadata: ComponentMetadata) -> None:
        """
        Register a component with its metadata.
        This is the primary registration method that preserves manifest IDs.
        """
        component_id = metadata.id
        
        # Validate metadata
        if not isinstance(metadata, ComponentMetadata):
            raise TypeError(f"metadata must be ComponentMetadata instance, got {type(metadata)}")
        
        # Store the component with its exact ID
        self._components[component_id] = component
        self._metadata[component_id] = metadata
        self._registration_order.append(component_id)
        
        # If it's a NireonBaseComponent, ensure its component_id matches
        if hasattr(component, 'component_id') and component.component_id != component_id:
            logger.warning(
                f"Component ID mismatch: instance has '{component.component_id}', "
                f"metadata has '{component_id}'. Using metadata ID for registration."
            )
            # Try to update the component's ID if possible
            if hasattr(component, '_component_id'):
                object.__setattr__(component, '_component_id', component_id)
        
        # Fire registration event if event bus available
        if self._event_bus:
            try:
                self._event_bus.publish('COMPONENT_REGISTERED', {
                    'component_id': component_id,
                    'component_type': type(component).__name__,
                    'category': metadata.category,
                    'timestamp': datetime.now(timezone.utc).isoformat()
                })
            except Exception as e:
                logger.warning(f"Failed to publish registration event: {e}")
        
        logger.info(f"Registered component '{component_id}' ({type(component).__name__})")
    
    def register_service_instance(self, key: Union[str, Type], instance: Any) -> None:
        """
        Register a service instance by string key or type.
        Used for type-based and alias registrations.
        """
        if isinstance(key, type):
            # Type-based registration
            self._service_instances[key] = instance
            normalized_key = self.normalize_key(key)
            self._components[normalized_key] = instance
            logger.debug(f"Registered service by type: {key.__name__}")
        else:
            # String key registration
            self._components[str(key)] = instance
            logger.debug(f"Registered service by key: {key}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a component by its ID.
        First tries exact match, then normalized key.
        """
        # First try exact match - this is critical for manifest IDs
        if key in self._components:
            return self._components[key]
        
        # Then try normalized key (for type-based lookups)
        normalized = self.normalize_key(key)
        if normalized != key and normalized in self._components:
            return self._components[normalized]
        
        # If not found and no default, raise error with helpful message
        if default is None:
            # Get available keys for error message
            available = list(self._components.keys())
            if len(available) > 20:
                available = available[:20] + [f"... and {len(available) - 20} more"]
            
            raise ComponentRegistryMissingError(
                key,
                available_components=available
            )
        
        return default
    
    def get_service_instance(self, service_type: Type) -> Any:
        """
        Get a service instance by its type/interface.
        """
        # First check direct type mapping
        if service_type in self._service_instances:
            return self._service_instances[service_type]
        
        # Then check normalized type key in components
        normalized_key = self.normalize_key(service_type)
        if normalized_key in self._components:
            return self._components[normalized_key]
        
        # Finally, search all components for instances of the type
        for comp_id, component in self._components.items():
            if isinstance(component, service_type):
                logger.debug(f"Found {service_type.__name__} by instance check: {comp_id}")
                return component
        
        raise ComponentRegistryMissingError(
            f"Service of type {service_type.__name__}",
            available_components=list(self._components.keys())[:10]
        )
    
    def list_components(self) -> List[str]:
        """
        List all registered component IDs.
        """
        return list(self._components.keys())
    
    def get_metadata(self, component_id: str) -> ComponentMetadata:
        """
        Get metadata for a component.
        """
        if component_id in self._metadata:
            return self._metadata[component_id]
        
        # Try to get from component itself
        component = self.get(component_id)
        if hasattr(component, 'metadata') and isinstance(component.metadata, ComponentMetadata):
            return component.metadata
        
        raise ComponentRegistryMissingError(
            f"Metadata for component '{component_id}'",
            available_components=list(self._metadata.keys())[:10]
        )
    
    def register_certification(self, component_id: str, certification_data: Dict[str, Any]) -> None:
        """
        Register certification data for a component.
        """
        self._certifications[component_id] = certification_data
        logger.debug(f"Registered certification for component '{component_id}'")

    def has_component(self, component_id: str) -> bool:
        return component_id in self._components or self.normalize_key(component_id) in self._components
    
    def get_certification(self, component_id: str) -> Optional[Dict[str, Any]]:
        """
        Get certification data for a component.
        """
        return self._certifications.get(component_id)
    
    def has_component(self, component_id: str) -> bool:
        """
        Check if a component is registered.
        """
        return component_id in self._components or self.normalize_key(component_id) in self._components
    
    def get_components_by_category(self, category: str) -> List[str]:
        """
        Get all component IDs for a given category.
        """
        result = []
        for comp_id, metadata in self._metadata.items():
            if metadata.category == category:
                result.append(comp_id)
        return result
    
    def get_components_by_tag(self, tag: str) -> List[str]:
        """
        Get all component IDs that have a specific epistemic tag.
        """
        result = []
        for comp_id, metadata in self._metadata.items():
            if tag in metadata.epistemic_tags:
                result.append(comp_id)
        return result
    
    def unregister(self, component_id: str) -> None:
        """
        Remove a component from the registry.
        """
        if component_id in self._components:
            component = self._components[component_id]
            
            # Remove from main registry
            del self._components[component_id]
            
            # Remove metadata
            if component_id in self._metadata:
                del self._metadata[component_id]
            
            # Remove certification
            if component_id in self._certifications:
                del self._certifications[component_id]
            
            # Remove from registration order
            if component_id in self._registration_order:
                self._registration_order.remove(component_id)
            
            # Remove from service instances if it's there
            for svc_type, instance in list(self._service_instances.items()):
                if instance is component:
                    del self._service_instances[svc_type]
            
            logger.info(f"Unregistered component '{component_id}'")
    
    def clear(self) -> None:
        """
        Clear all registrations.
        """
        self._components.clear()
        self._metadata.clear()
        self._certifications.clear()
        self._service_instances.clear()
        self._registration_order.clear()
        logger.info("Registry cleared")
    
    def get_registration_order(self) -> List[str]:
        """
        Get the order in which components were registered.
        """
        return list(self._registration_order)
    
    def __len__(self) -> int:
        """
        Get the number of registered components.
        """
        return len(self._components)
    
    def __contains__(self, component_id: str) -> bool:
        """
        Check if a component is in the registry.
        """
        return self.has_component(component_id)