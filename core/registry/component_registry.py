# ComponentRegistry class, enhanced to leverage new metadata features
from __future__ import annotations

import logging
import threading
from typing import Any, Dict, List, Union, Type, Optional, Set, Tuple
from datetime import datetime, timezone
from collections import defaultdict

from core.lifecycle import ComponentMetadata, ComponentRegistryMissingError

__all__ = ['ComponentRegistry']

logger = logging.getLogger(__name__)


class ComponentRegistry:
    """Enhanced registry leveraging advanced metadata features."""
    
    def __init__(self) -> None:
        self._components: Dict[str, Any] = {}
        self._metadata: Dict[str, ComponentMetadata] = {}
        self._certifications: Dict[str, Dict[str, Any]] = {}
        self._instance_to_metadata: Dict[int, ComponentMetadata] = {}
        
        # New: Track component relationships
        self._dependency_graph: Dict[str, Set[str]] = defaultdict(set)  # component_id -> dependencies
        self._dependents_graph: Dict[str, Set[str]] = defaultdict(set)  # component_id -> dependents
        
        # New: Version tracking
        self._version_history: Dict[str, List[Tuple[str, datetime]]] = defaultdict(list)  # component_id -> [(version, timestamp)]
        
        # New: Fingerprint cache for quick comparison
        self._fingerprint_cache: Dict[str, str] = {}
        
        self._lock = threading.RLock()
        logger.debug("ComponentRegistry initialized with enhanced features")

    def normalize_key(self, key: Union[str, Type, object]) -> str:
        """Simple key normalization."""
        if isinstance(key, str):
            return key.strip()
        elif isinstance(key, type):
            return key.__name__
        elif hasattr(key, '__name__'):
            return getattr(key, '__name__')
        else:
            return str(key)

    def _warn_re_register(self, key: str, kind: str) -> None:
        logger.warning(f"[ComponentRegistry] {kind} with key '{key}' is being re-registered.")

    def is_service_registered(self, key: Union[str, Type]) -> bool:
        """Check if a service is registered."""
        normalized_key = self.normalize_key(key)
        return normalized_key in self._components

    def register(self, component_value: Any, metadata: ComponentMetadata) -> None:
        """Enhanced registration with dependency tracking and validation."""
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
            # Check for version changes
            if canonical_id in self._metadata:
                existing_metadata = self._metadata[canonical_id]
                if existing_metadata.version != metadata.version:
                    # Track version history
                    self._version_history[canonical_id].append(
                        (metadata.version, datetime.now(timezone.utc))
                    )
                    logger.info(f"Component '{canonical_id}' version changed: "
                               f"{existing_metadata.version} -> {metadata.version}")
            
            # Validate interface compatibility
            if metadata.expected_interfaces:
                validation_errors = metadata.validate_interfaces(component_value)
                if validation_errors:
                    logger.warning(f"Interface validation issues for '{canonical_id}': "
                                  f"{'; '.join(validation_errors)}")
            
            # Check for conflicts
            for comp_id, comp_meta in self._metadata.items():
                if comp_id != canonical_id:
                    if not metadata.is_compatible_with(comp_meta):
                        logger.warning(f"Component '{canonical_id}' may not be compatible "
                                      f"with existing component '{comp_id}'")
            
            # Register the component
            self._components[canonical_id] = component_value
            self._metadata[canonical_id] = metadata
            self._instance_to_metadata[id(component_value)] = metadata
            
            # Cache fingerprint
            self._fingerprint_cache[canonical_id] = metadata.fingerprint
            
            # Update dependency graphs
            self._update_dependency_graphs(canonical_id, metadata)
            
            # Update metadata runtime state
            metadata.add_runtime_state("registered_at", datetime.now(timezone.utc).isoformat())
            metadata.add_runtime_state("registry_id", id(self))
            
            logger.info(f"Component '{canonical_id}' registered successfully "
                       f"(version: {metadata.version}, fingerprint: {metadata.fingerprint[:8]}...)")

            # Handle type alias
            if type_key_for_alias:
                if type_key_for_alias not in self._components or self._components[type_key_for_alias] is not component_value:
                    self._components[type_key_for_alias] = component_value
                    logger.debug(f"Component '{canonical_id}' also accessible via type alias '{type_key_for_alias}'")
                if type_key_for_alias not in self._metadata or self._metadata[type_key_for_alias] != metadata:
                    self._metadata[type_key_for_alias] = metadata
                    logger.debug(f"Metadata linked under type alias '{type_key_for_alias}'")

            # Update existing aliases
            for alias_key, registered_instance in list(self._components.items()):
                if registered_instance is component_value and alias_key != canonical_id and alias_key != type_key_for_alias:
                    if alias_key not in self._metadata or self._metadata[alias_key] != metadata:
                        self._metadata[alias_key] = metadata
                        logger.debug(f"Updated metadata for alias '{alias_key}'")

    def _update_dependency_graphs(self, component_id: str, metadata: ComponentMetadata) -> None:
        """Update dependency tracking graphs."""
        # Clear old dependencies
        if component_id in self._dependency_graph:
            for dep_id in self._dependency_graph[component_id]:
                self._dependents_graph[dep_id].discard(component_id)
        
        # Add new dependencies
        self._dependency_graph[component_id] = set(metadata.dependencies.keys())
        for dep_id in metadata.dependencies:
            self._dependents_graph[dep_id].add(component_id)

    def register_service_instance(self, key: Union[str, Type], instance: Any) -> None:
        """Enhanced service registration with metadata inference."""
        normalized_key = self.normalize_key(key)
        if not normalized_key.strip():
            raise ValueError('Service key cannot resolve to empty or whitespace')
        
        with self._lock:
            if normalized_key in self._components and self._components[normalized_key] is not instance:
                self._warn_re_register(normalized_key, 'Service (via register_service_instance)')
            
            self._components[normalized_key] = instance
            logger.debug(f"Service instance registered: {key} -> '{normalized_key}' (Type: {type(instance).__name__})")

            # Enhanced metadata linking
            canonical_metadata_to_link: Optional[ComponentMetadata] = None
            
            # Check instance-to-metadata mapping first (O(1) lookup)
            instance_id = id(instance)
            if instance_id in self._instance_to_metadata:
                canonical_metadata_to_link = self._instance_to_metadata[instance_id]
                logger.debug(f"Found canonical metadata via instance mapping (ID: '{canonical_metadata_to_link.id}')")
            
            # Check instance's metadata attribute
            elif hasattr(instance, 'metadata') and isinstance(getattr(instance, 'metadata', None), ComponentMetadata):
                canonical_metadata_to_link = instance.metadata
                self._instance_to_metadata[instance_id] = canonical_metadata_to_link
                if canonical_metadata_to_link.id not in self._metadata:
                    self._metadata[canonical_metadata_to_link.id] = canonical_metadata_to_link
                    logger.debug(f"Cached metadata for '{canonical_metadata_to_link.id}' from instance.metadata")
            
            if canonical_metadata_to_link:
                if normalized_key not in self._metadata or self._metadata[normalized_key] != canonical_metadata_to_link:
                    self._metadata[normalized_key] = canonical_metadata_to_link
                    logger.debug(f"Linked metadata for service alias '{normalized_key}' to canonical metadata")

    def get_service_instance(self, key: Union[str, Type]) -> Any:
        """Get service instance with enhanced error reporting."""
        normalized_key = self.normalize_key(key)
        logger.debug(f"[get_service_instance] Looking for: '{normalized_key}'")
        
        if normalized_key in self._components:
            return self._components[normalized_key]
        
        # Type-based fallback
        if isinstance(key, type):
            for comp_id, comp_instance in self._components.items():
                if isinstance(comp_instance, key):
                    logger.debug(f"Found by type match: '{normalized_key}' -> '{comp_id}'")
                    return comp_instance
        
        # Enhanced error with available components
        available_keys = sorted(self._components.keys())
        raise ComponentRegistryMissingError(
            str(key),
            message=f"Service '{key}' (normalized: '{normalized_key}') not found",
            available_components=available_keys
        )

    def get(self, key: Union[str, Type]) -> Any:
        """General get with enhanced lookup."""
        normalized_key = self.normalize_key(key)
        if normalized_key in self._components:
            return self._components[normalized_key]

        # Type-based fallback
        if isinstance(key, type):
            for comp_id, comp_instance in self._components.items():
                if isinstance(comp_instance, key):
                    logger.debug(f"Found by type: '{normalized_key}' -> '{comp_id}'")
                    return comp_instance
        
        # Metadata ID fallback
        if isinstance(key, str):
             for comp_instance in self._components.values():
                 if hasattr(comp_instance, 'metadata') and isinstance(getattr(comp_instance, 'metadata', None), ComponentMetadata):
                     if comp_instance.metadata.id == normalized_key:
                         logger.debug(f"Found by metadata.id: '{normalized_key}'")
                         return comp_instance
                 elif hasattr(comp_instance, 'component_id') and comp_instance.component_id == normalized_key:
                     logger.debug(f"Found by component_id: '{normalized_key}'")
                     return comp_instance

        available_keys = sorted(self._components.keys())
        raise ComponentRegistryMissingError(normalized_key, available_components=available_keys)

    def get_metadata(self, component_id_or_alias: str) -> ComponentMetadata:
        """Enhanced metadata retrieval."""
        normalized_key = self.normalize_key(component_id_or_alias)

        # Direct lookup
        if normalized_key in self._metadata:
            return self._metadata[normalized_key]

        # Instance-based lookup
        if normalized_key in self._components:
            comp = self._components[normalized_key]
            instance_id = id(comp)
            
            # Check instance mapping
            if instance_id in self._instance_to_metadata:
                metadata = self._instance_to_metadata[instance_id]
                with self._lock:
                    if normalized_key not in self._metadata:
                        self._metadata[normalized_key] = metadata
                logger.debug(f"Retrieved metadata via instance mapping")
                return metadata
            
            # Check instance attribute
            if hasattr(comp, 'metadata') and isinstance(getattr(comp, 'metadata', None), ComponentMetadata):
                instance_meta = comp.metadata
                with self._lock:
                    self._instance_to_metadata[instance_id] = instance_meta
                    if instance_meta.id not in self._metadata:
                        self._metadata[instance_meta.id] = instance_meta
                    if normalized_key != instance_meta.id and normalized_key not in self._metadata:
                         self._metadata[normalized_key] = instance_meta
                logger.debug(f"Retrieved metadata from instance attribute")
                return instance_meta

        available_metadata_keys = sorted(self._metadata.keys())
        available_component_keys = sorted(self._components.keys())
        raise ComponentRegistryMissingError(
            component_id_or_alias,
            message=f"Metadata for key '{normalized_key}' not found",
            available_components=available_metadata_keys
        )

    def get_certification(self, component_id: str) -> Dict[str, Any]:
        """Get certification data."""
        return self._certifications.get(component_id, {})

    def register_certification(self, component_id: str, cert: Dict[str, Any]) -> None:
        """Register certification with validation."""
        if not isinstance(cert, dict):
            raise TypeError("Certification data must be a dictionary")
        with self._lock:
            self._certifications[component_id] = cert
            # Update metadata runtime state
            if component_id in self._metadata:
                self._metadata[component_id].add_runtime_state("certified", True)
                self._metadata[component_id].add_runtime_state("certification_timestamp", 
                                                               datetime.now(timezone.utc).isoformat())
            logger.debug(f"Certification registered for '{component_id}'")

    def list_components(self) -> List[str]:
        """List all component keys."""
        return list(self._components.keys())

    def list_service_instances(self) -> List[str]:
        """List all service instance keys."""
        return list(self._components.keys())

    def find_by_category(self, category: str) -> List[str]:
        """Find components by category."""
        return [cid for cid, md in self._metadata.items() if md.category == category]

    def find_by_capability(self, capability: str) -> List[str]:
        """Find components by capability."""
        return [cid for cid, md in self._metadata.items() if capability in md.capabilities]

    def find_by_epistemic_tag(self, tag: str) -> List[str]:
        """Find components by epistemic tag."""
        return [cid for cid, md in self._metadata.items() if tag in md.epistemic_tags]

    def find_satisfying_dependency(self, dependency_id: str, version_spec: str) -> List[str]:
        """Find components that satisfy a dependency requirement."""
        results = []
        for cid, metadata in self._metadata.items():
            if cid == dependency_id or dependency_id in [cid, metadata.id]:
                if metadata.satisfies_dependency(version_spec):
                    results.append(cid)
        return results

    def get_dependents(self, component_id: str) -> Set[str]:
        """Get all components that depend on this component."""
        return self._dependents_graph.get(component_id, set()).copy()

    def get_dependencies(self, component_id: str) -> Set[str]:
        """Get all dependencies of this component."""
        return self._dependency_graph.get(component_id, set()).copy()

    def check_dependency_conflicts(self) -> List[str]:
        """Check for dependency conflicts in the registry."""
        conflicts = []
        
        for comp_id, dependencies in self._dependency_graph.items():
            if comp_id not in self._metadata:
                continue
                
            comp_meta = self._metadata[comp_id]
            
            for dep_id in dependencies:
                version_spec = comp_meta.dependencies.get(dep_id, "*")
                
                # Check if any registered component satisfies this dependency
                satisfying = self.find_satisfying_dependency(dep_id, version_spec)
                if not satisfying:
                    conflicts.append(
                        f"Component '{comp_id}' requires '{dep_id}' "
                        f"version '{version_spec}' but none found"
                    )
        
        return conflicts

    def get_version_history(self, component_id: str) -> List[Tuple[str, datetime]]:
        """Get version history for a component."""
        return self._version_history.get(component_id, []).copy()

    def get_stats(self) -> Dict[str, Any]:
        """Enhanced statistics with dependency analysis."""
        certified_count = len([cid for cid in self._certifications if self._certifications[cid]])
        categories: Dict[str, int] = {}
        epistemic_tags: Dict[str, int] = {}
        
        unique_metadata = set(id(md) for md in self._metadata.values())
        
        for metadata in self._metadata.values():
            categories[metadata.category] = categories.get(metadata.category, 0) + 1
            for tag in metadata.epistemic_tags:
                epistemic_tags[tag] = epistemic_tags.get(tag, 0) + 1
        
        unique_instances = set(id(inst) for inst in self._components.values())
        
        # Dependency analysis
        total_dependencies = sum(len(deps) for deps in self._dependency_graph.values())
        max_dependents = max(len(deps) for deps in self._dependents_graph.values()) if self._dependents_graph else 0
        
        return {
            "total_component_instances": len(unique_instances),
            "total_registered_keys": len(self._components),
            "components_with_metadata": len(self._metadata),
            "unique_metadata_objects": len(unique_metadata),
            "certified_components": certified_count,
            "categories": categories,
            "epistemic_tags": epistemic_tags,
            "total_dependencies": total_dependencies,
            "max_dependents": max_dependents,
            "dependency_conflicts": len(self.check_dependency_conflicts()),
            "components_with_version_history": len(self._version_history)
        }

    def cleanup_instance_reference(self, instance: Any) -> None:
        """Remove instance references to prevent memory leaks."""
        instance_id = id(instance)
        if instance_id in self._instance_to_metadata:
            metadata = self._instance_to_metadata[instance_id]
            del self._instance_to_metadata[instance_id]
            # Clear runtime state
            metadata.clear_runtime_state()
            logger.debug(f"Cleaned up instance reference for id {instance_id}")

    def export_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Export all metadata for persistence."""
        return {
            comp_id: metadata.to_dict()
            for comp_id, metadata in self._metadata.items()
        }

    def import_metadata(self, metadata_dict: Dict[str, Dict[str, Any]]) -> None:
        """Import metadata from persistence."""
        with self._lock:
            for comp_id, meta_data in metadata_dict.items():
                try:
                    metadata = ComponentMetadata.from_dict(meta_data)
                    self._metadata[comp_id] = metadata
                    self._fingerprint_cache[comp_id] = metadata.fingerprint
                    self._update_dependency_graphs(comp_id, metadata)
                    logger.info(f"Imported metadata for component '{comp_id}'")
                except Exception as e:
                    logger.error(f"Failed to import metadata for '{comp_id}': {e}")