# ComponentRegistry class, adapted from V3's application.components.lifecycle
from __future__ import annotations

import logging
import threading
from typing import Any, Dict, List, Union, Type, Optional


from core.lifecycle import ComponentMetadata, ComponentRegistryMissingError



__all__ = ['ComponentRegistry'] # Only ComponentRegistry is defined here

logger = logging.getLogger(__name__)

class ComponentRegistry:
    def __init__(self) -> None:
        self._components: Dict[str, Any] = {}
        self._metadata: Dict[str, ComponentMetadata] = {}
        self._certifications: Dict[str, Dict[str, Any]] = {} # Retain from V3
        # NEW: Direct instance-to-metadata mapping for O(1) lookups
        self._instance_to_metadata: Dict[int, ComponentMetadata] = {}
        self._lock = threading.RLock() # Use RLock for reentrancy
        logger.debug("ComponentRegistry initialized with thread safety and instance-to-metadata mapping")

    def normalize_key(self, key: Union[str, Type, object]) -> str:
        """
        Simple key normalization - use service names, not complex class path logic.
        For Type objects, extract just the class name (not full module path).
        For strings, use as-is.
        """
        if isinstance(key, str):
            return key.strip()
        elif isinstance(key, type):
            # Use just the class name, not the full module path
            return key.__name__
        elif hasattr(key, '__name__'): # For function objects etc.
            return getattr(key, '__name__')
        else: # Fallback for other objects
            return str(key)

    def _warn_re_register(self, key: str, kind: str) -> None: # Internal helper
        logger.warning(f"[.ComponentRegistry] {kind} with key '{key}' is being re-registered.")

    def is_service_registered(self, key: Union[str, Type]) -> bool:
        normalized_key = self.normalize_key(key)
        return normalized_key in self._components

    def register(self, component_value: Any, metadata: ComponentMetadata) -> None:
        import inspect # Keep import here if not already at top of file
        if not isinstance(metadata, ComponentMetadata):
            raise TypeError('metadata must be an instance of ComponentMetadata')
        
        canonical_id = metadata.id
        type_key_for_alias = None
        component_type_obj = type(component_value)

        if hasattr(component_type_obj, '__name__'):
            normalized_type_name = self.normalize_key(component_type_obj)
            if normalized_type_name != canonical_id:
                type_key_for_alias = normalized_type_name
        
        with self._lock:
            if canonical_id in self._components:
                existing_instance = self._components[canonical_id]
                if existing_instance is not component_value:
                    logger.debug(f"[V4.ComponentRegistry] Component '{canonical_id}' is being re-registered with a new instance (Old: {type(existing_instance)}, New: {type(component_value)}).")
            
            self._components[canonical_id] = component_value
            self._metadata[canonical_id] = metadata # Canonical registration
            
            # NEW: Store instance-to-metadata mapping
            self._instance_to_metadata[id(component_value)] = metadata
            
            logger.info(f"Component '{canonical_id}' (canonical ID) registered (or re-registered) successfully (category: {metadata.category}, type: {type(component_value).__name__})")

            # Link type alias if it's different and not already pointing to this instance/metadata
            if type_key_for_alias:
                if type_key_for_alias not in self._components or self._components[type_key_for_alias] is not component_value:
                    self._components[type_key_for_alias] = component_value
                    logger.debug(f"Component instance '{canonical_id}' also made accessible via type alias '{type_key_for_alias}'.")
                # Ensure the type alias also points to the canonical metadata
                if type_key_for_alias not in self._metadata or self._metadata[type_key_for_alias] != metadata:
                    self._metadata[type_key_for_alias] = metadata
                    logger.debug(f"Metadata for component '{canonical_id}' also linked under type alias '{type_key_for_alias}'.")

            # Ensure all other existing aliases for this component_value also point to this canonical metadata
            for alias_key, registered_instance in list(self._components.items()):
                if registered_instance is component_value and alias_key != canonical_id and alias_key != type_key_for_alias:
                    if alias_key not in self._metadata or self._metadata[alias_key] != metadata:
                        self._metadata[alias_key] = metadata
                        logger.debug(f"Updated metadata for existing service alias '{alias_key}' to point to canonical metadata of '{canonical_id}' during canonical registration.")


    def register_service_instance(self, key: Union[str, Type], instance: Any) -> None:
        normalized_key = self.normalize_key(key)
        if not normalized_key.strip():
            raise ValueError('Service key cannot resolve to empty or whitespace')
        
        with self._lock:
            if normalized_key in self._components and self._components[normalized_key] is not instance:
                self._warn_re_register(normalized_key, 'Service (via register_service_instance)')
            
            self._components[normalized_key] = instance
            logger.debug(f"Service instance registered: {key} -> '{normalized_key}' (Type: {type(instance).__name__})")

            # NEW: Use instance-to-metadata mapping first for O(1) lookup
            canonical_metadata_to_link: Optional[ComponentMetadata] = None
            
            # Check our direct mapping first
            instance_id = id(instance)
            if instance_id in self._instance_to_metadata:
                canonical_metadata_to_link = self._instance_to_metadata[instance_id]
                logger.debug(f"Found canonical metadata for instance via direct mapping (ID: '{canonical_metadata_to_link.id}')")
            
            # If not found in direct mapping, check instance's metadata attribute
            elif hasattr(instance, 'metadata') and isinstance(getattr(instance, 'metadata', None), ComponentMetadata):
                canonical_metadata_to_link = instance.metadata
                # Store in our direct mapping for future lookups
                self._instance_to_metadata[instance_id] = canonical_metadata_to_link
                # Ensure the canonical ID's metadata is stored
                if canonical_metadata_to_link.id not in self._metadata:
                    self._metadata[canonical_metadata_to_link.id] = canonical_metadata_to_link
                    logger.debug(f"Cached metadata for '{canonical_metadata_to_link.id}' from instance.metadata")
            
            if canonical_metadata_to_link:
                # Link the alias to the canonical metadata
                if normalized_key not in self._metadata or self._metadata[normalized_key] != canonical_metadata_to_link:
                    self._metadata[normalized_key] = canonical_metadata_to_link
                    logger.debug(f"Linked metadata for service alias '{normalized_key}' to canonical metadata of '{canonical_metadata_to_link.id}'.")
            else:
                # No canonical metadata found - log for debugging
                logger.debug(f"Service '{normalized_key}' registered. No canonical metadata found yet. "
                             f"Will be resolved by canonical 'register()' call if this is an alias.")


    def get_service_instance(self, key: Union[str, Type]) -> Any:
        normalized_key = self.normalize_key(key)
        logger.debug(f"[ComponentRegistry.get_service_instance] Attempting to get service with normalized key: '{normalized_key}'.")
        
        if normalized_key in self._components:
            return self._components[normalized_key]
        
        # Fallback: If key is a Type, check if any registered component is an instance of that type
        if isinstance(key, type):
            for comp_id, comp_instance in self._components.items():
                if isinstance(comp_instance, key):
                    logger.debug(f"Found service by type match for '{normalized_key}' -> component '{comp_id}' ({type(comp_instance).__name__})")
                    return comp_instance
        
        available_keys = sorted(self._components.keys())
        raise ComponentRegistryMissingError(
            str(key),
            message=f"Service '{key}' (normalized: '{normalized_key}') not found in registry. Available component keys: {available_keys}"
        )

    def get(self, key: Union[str, Type]) -> Any: # General get, can be by ID or type
        normalized_key = self.normalize_key(key)
        if normalized_key in self._components:
            return self._components[normalized_key]

        # Fallback logic similar to get_service_instance if not found by direct key
        if isinstance(key, type):
            for comp_id, comp_instance in self._components.items():
                if isinstance(comp_instance, key):
                    logger.debug(f"Found component by type match for '{normalized_key}' -> component '{comp_id}' in get()")
                    return comp_instance
        
        # Fallback: if key is a string, it might be a canonical ID whose instance was registered under an alias
        if isinstance(key, str):
             for comp_instance in self._components.values():
                 if hasattr(comp_instance, 'metadata') and isinstance(getattr(comp_instance, 'metadata', None), ComponentMetadata):
                     if comp_instance.metadata.id == normalized_key:
                         logger.debug(f"Found component by matching instance metadata.id for key '{normalized_key}' in get()")
                         return comp_instance
                 elif hasattr(comp_instance, 'component_id') and comp_instance.component_id == normalized_key: # For NireonBaseComponent
                     logger.debug(f"Found component by matching instance component_id for key '{normalized_key}' in get()")
                     return comp_instance


        raise ComponentRegistryMissingError(normalized_key)

    def get_metadata(self, component_id_or_alias: str) -> ComponentMetadata:
        normalized_key = self.normalize_key(component_id_or_alias)

        # 1. Direct lookup by the provided key (could be ID or alias)
        if normalized_key in self._metadata:
            return self._metadata[normalized_key]

        # 2. If the key corresponds to a registered component, try to get metadata from the instance
        if normalized_key in self._components:
            comp = self._components[normalized_key]
            
            # NEW: Check instance-to-metadata mapping first
            instance_id = id(comp)
            if instance_id in self._instance_to_metadata:
                metadata = self._instance_to_metadata[instance_id]
                # Cache under the alias for future direct lookups
                with self._lock:
                    if normalized_key not in self._metadata:
                        self._metadata[normalized_key] = metadata
                logger.debug(f"Retrieved metadata for '{normalized_key}' from instance-to-metadata mapping")
                return metadata
            
            # Fallback to checking instance.metadata attribute
            if hasattr(comp, 'metadata') and isinstance(getattr(comp, 'metadata', None), ComponentMetadata):
                instance_meta: ComponentMetadata = comp.metadata
                with self._lock:
                    # Store in both places for future lookups
                    self._instance_to_metadata[instance_id] = instance_meta
                    if instance_meta.id not in self._metadata:
                        self._metadata[instance_meta.id] = instance_meta
                    if normalized_key != instance_meta.id and normalized_key not in self._metadata:
                         self._metadata[normalized_key] = instance_meta
                logger.debug(f"Retrieved and cached metadata for '{normalized_key}' from instance attribute (Canonical ID: '{instance_meta.id}')")
                return instance_meta

        available_metadata_keys = sorted(self._metadata.keys())
        available_component_keys = sorted(self._components.keys())
        raise ComponentRegistryMissingError(
            component_id_or_alias,
            message=(
                f"Metadata for key '{normalized_key}' not found in registry. "
                f"Available metadata keys: {available_metadata_keys}. "
                f"Available component keys (instances): {available_component_keys}."
            )
        )

    def get_certification(self, component_id: str) -> Dict[str, Any]: # Retain from V3
        return self._certifications.get(component_id, {})

    def register_certification(self, component_id: str, cert: Dict[str, Any]) -> None: # Retain
        if not isinstance(cert, dict):
            raise TypeError("Certification data must be a dictionary")
        with self._lock:
            # Allow certification even if component registration is pending (e.g. during NireonBaseComponent.__init__)
            # if component_id not in self._components:
            #     logger.debug(f"Registering certification for '{component_id}' which is not (yet) a fully registered component instance. This may be acceptable during bootstrap.")
            self._certifications[component_id] = cert
            logger.debug(f"Certification registered for '{component_id}'")

    def list_components(self) -> List[str]:
        # This should ideally list canonical component IDs for consistency.
        # However, _components might contain aliases.
        # A more robust way might be to return keys from _metadata if all true components have metadata.
        # Or, return a set of all keys from _components and _metadata.
        # For now, matching existing behavior:
        return list(self._components.keys())

    def list_service_instances(self) -> List[str]: # Alias or distinct if needed
        return list(self._components.keys())

    # Retain these useful find methods from V3
    def find_by_category(self, category: str) -> List[str]:
        # Iterate through _metadata, as category is a metadata property
        return [cid for cid, md in self._metadata.items() if md.category == category]

    def find_by_capability(self, capability: str) -> List[str]:
        return [cid for cid, md in self._metadata.items() if capability in md.capabilities]

    def find_by_epistemic_tag(self, tag: str) -> List[str]:
        return [cid for cid, md in self._metadata.items() if tag in md.epistemic_tags]

    def get_stats(self) -> Dict[str, Any]: # Retain from V3
        certified_count = len([cid for cid in self._certifications if self._certifications[cid]])
        categories: Dict[str, int] = {}
        epistemic_tags: Dict[str, int] = {}
        
        # Count unique metadata objects (not just keys, since aliases share metadata)
        unique_metadata = set(id(md) for md in self._metadata.values())
        
        for metadata in self._metadata.values(): # Iterate unique metadata objects
            categories[metadata.category] = categories.get(metadata.category, 0) + 1
            for tag in metadata.epistemic_tags:
                epistemic_tags[tag] = epistemic_tags.get(tag, 0) + 1
        
        # Count unique component instances, not just keys in _components (which might include aliases)
        unique_instances = set(id(inst) for inst in self._components.values())

        return {
            "total_component_instances": len(unique_instances), # More accurate count of distinct objects
            "total_registered_keys": len(self._components), # Includes aliases
            "components_with_metadata": len(self._metadata), # Number of keys with associated metadata
            "unique_metadata_objects": len(unique_metadata), # NEW: Number of unique metadata objects
            "certified_components": certified_count,
            "categories": categories,
            "epistemic_tags": epistemic_tags,
        }

    # Helper for robustly importing inspect if needed
    def _get_inspect_module(self):
        import inspect as _inspect_module
        return _inspect_module
    
    def cleanup_instance_reference(self, instance: Any) -> None:
        """Remove instance from the instance-to-metadata mapping when it's being unregistered.
        This prevents memory leaks from holding references to instances."""
        instance_id = id(instance)
        if instance_id in self._instance_to_metadata:
            del self._instance_to_metadata[instance_id]
            logger.debug(f"Cleaned up instance-to-metadata mapping for instance id {instance_id}")