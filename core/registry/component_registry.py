# V4: ComponentRegistry class, adapted from V3's application.components.lifecycle
from __future__ import annotations

import logging
import threading
from typing import Any, Dict, List, Optional, Set, Union, Type

# V4: Import ComponentMetadata and ComponentRegistryMissingError from their new V4 location
from application.components.lifecycle import ComponentMetadata, ComponentRegistryMissingError

__all__ = ['ComponentRegistry'] # V4: Only ComponentRegistry is defined here

logger = logging.getLogger(__name__)

class ComponentRegistry:
    def __init__(self) -> None:
        self._components: Dict[str, Any] = {}
        self._metadata: Dict[str, ComponentMetadata] = {}
        self._certifications: Dict[str, Dict[str, Any]] = {} # V4: Retain from V3
        self._lock = threading.RLock() # V4: Use RLock for reentrancy
        logger.debug("ComponentRegistry (V4) initialized with thread safety")

    def normalize_key(self, key: Union[str, Type, object]) -> str: # V4: Public method
        original_key_repr = repr(key)
        if isinstance(key, str):
            normalized = key
        elif isinstance(key, type): # V4: Robust type naming
            normalized = f"{key.__module__}.{key.__qualname__}"
        elif hasattr(key, '__name__'): # Fallback for other callables
            normalized = getattr(key, '__name__')
        else:
            normalized = str(key)
        logger.debug(f"[V4.ComponentRegistry.normalize_key] Original key: {original_key_repr}, Type: {type(key)}, Normalized to: '{normalized}'")
        return normalized
    
    def _warn_re_register(self, key: str, kind: str) -> None: # V4: Internal helper
        logger.warning(f"[V4.ComponentRegistry] {kind} with key '{key}' is being re-registered.")

    def is_service_registered(self, key: Union[str, Type]) -> bool:
        normalized_key = self.normalize_key(key)
        return normalized_key in self._components

    def register(self, component_value: Any, metadata: ComponentMetadata) -> None:
        if not isinstance(metadata, ComponentMetadata):
            raise TypeError("metadata must be an instance of ComponentMetadata")
        
        registration_key = metadata.id # Use metadata.id as the primary key
        with self._lock:
            if registration_key in self._components:
                existing_instance = self._components[registration_key]
                if existing_instance is component_value:
                    logger.debug(f"Component '{registration_key}' already registered with the exact same instance. Skipping re-registration.")
                    return
                else: # V4: Detailed logging for re-registration
                    logger.debug(
                        f"[V4.ComponentRegistry] Component '{registration_key}' is being re-registered with a new instance "
                        f"(Old: {type(existing_instance)}, New: {type(component_value)}). "
                        "This might be expected during multi-phase bootstrap."
                    )
            self._components[registration_key] = component_value
            self._metadata[registration_key] = metadata # Store metadata by the same ID
            logger.info(
                f"Component '{registration_key}' registered (or re-registered) successfully "
                f"(category: {metadata.category}, epistemic_tags: {metadata.epistemic_tags})"
            )

    def register_service_instance(self, key: Union[str, Type], instance: Any) -> None:
        normalized_key = self.normalize_key(key)
        if not normalized_key.strip(): # V4: Guard against empty keys
            raise ValueError("Service key cannot resolve to empty or whitespace")
            
        with self._lock:
            if normalized_key in self._components:
                self._warn_re_register(normalized_key, "Service")
            self._components[normalized_key] = instance
            # V4: Metadata for service instances (not full components) might be handled by _safe_register_service_instance
            # or could be added here if a convention is established.
            logger.debug(f"Service instance registered: {key} -> '{normalized_key}'")

    def get_service_instance(self, key: Union[str, Type]) -> Any:
        normalized_key = self.normalize_key(key)
        logger.debug(f"[V4.ComponentRegistry.get_service_instance] Attempting to get service with normalized key: '{normalized_key}'.")
        # V4: Improved debug logging
        logger.debug(f"[V4.ComponentRegistry.get_service_instance] Available keys in _components: {sorted(list(self._components.keys()))}")
        if normalized_key not in self._components:
            available_keys = sorted(self._components.keys()) # Get keys at the time of error
            raise ComponentRegistryMissingError(
                str(key), 
                message=f"Service '{key}' (normalized: '{normalized_key}') not found in V4 registry. Available components: {available_keys}"
            )
        return self._components[normalized_key]

    def get(self, key: Union[str, Type]) -> Any: # General get, can be by ID or type
        normalized_key = self.normalize_key(key)
        if normalized_key not in self._components:
            raise ComponentRegistryMissingError(normalized_key)
        return self._components[normalized_key]

    def get_metadata(self, component_id: str) -> ComponentMetadata:
        if component_id in self._metadata:
            return self._metadata[component_id]
        
        # Fallback: if component exists but metadata wasn't directly put in _metadata
        # (e.g. registered via register_service_instance without explicit metadata store step)
        if component_id in self._components:
            comp = self._components[component_id]
            if hasattr(comp, 'metadata') and isinstance(comp.metadata, ComponentMetadata):
                with self._lock: # Cache it if found this way
                    self._metadata[component_id] = comp.metadata
                logger.debug(f"Cached metadata for component '{component_id}' from instance")
                return comp.metadata
        
        available_metadata = sorted(self._metadata.keys())
        raise ComponentRegistryMissingError(
            component_id, 
            message=f"Metadata for component '{component_id}' not found in V4 registry. Components with metadata: {available_metadata}"
        )

    def get_certification(self, component_id: str) -> Dict[str, Any]: # V4: Retain from V3
        return self._certifications.get(component_id, {})

    def register_certification(self, component_id: str, cert: Dict[str, Any]) -> None: # V4: Retain
        if not isinstance(cert, dict):
            raise TypeError("Certification data must be a dictionary")
        with self._lock:
            if component_id not in self._components: # V4: More nuanced logging
                logger.debug(f"Registering certification for '{component_id}' which is not (yet) a fully registered component. This may be acceptable during bootstrap.")
            self._certifications[component_id] = cert
            logger.debug(f"Certification registered for '{component_id}'")

    def list_components(self) -> List[str]:
        return list(self._components.keys())

    def list_service_instances(self) -> List[str]: # V4: Alias or distinct if needed
        return list(self._components.keys())
    
    # V4: Retain these useful find methods from V3
    def find_by_category(self, category: str) -> List[str]:
        return [cid for cid, md in self._metadata.items() if md.category == category]

    def find_by_capability(self, capability: str) -> List[str]:
        return [cid for cid, md in self._metadata.items() if capability in md.capabilities]

    def find_by_epistemic_tag(self, tag: str) -> List[str]:
        return [cid for cid, md in self._metadata.items() if tag in md.epistemic_tags]

    def get_stats(self) -> Dict[str, Any]: # V4: Retain from V3
        certified_count = len([cid for cid in self._certifications if self._certifications[cid]])
        categories: Dict[str, int] = {}
        epistemic_tags: Dict[str, int] = {}
        for metadata in self._metadata.values():
            categories[metadata.category] = categories.get(metadata.category, 0) + 1
            for tag in metadata.epistemic_tags:
                epistemic_tags[tag] = epistemic_tags.get(tag, 0) + 1
        
        return {
            "total_components": len(self._components),
            "components_with_metadata": len(self._metadata),
            "certified_components": certified_count,
            "categories": categories,
            "epistemic_tags": epistemic_tags,
        }