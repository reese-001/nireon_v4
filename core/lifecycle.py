# application/components/lifecycle.py
from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Type, Protocol, TypeVar, Union, Callable
from functools import cached_property
import hashlib

__all__ = ['ComponentMetadata', 'ComponentRegistryMissingError', 'MetadataValidationError']

logger = logging.getLogger(__name__)

# Type variable for generic component types
T = TypeVar('T')


class MetadataValidationError(ValueError):
    """Raised when metadata validation fails."""
    pass


@dataclass
class ComponentMetadata:
    """Enhanced metadata for Nireon components with advanced features.
    
    This class maintains full backward compatibility while adding:
    - Version comparison and compatibility checking
    - Dependency tracking and validation
    - Runtime state tracking
    - Serialization support
    - Advanced introspection capabilities
    """
    
    # Core fields (backward compatible)
    id: str
    name: str
    version: str
    category: str  # e.g., 'mechanism', 'observer', 'service', 'core_service', 'persistence_service'
    subcategory: Optional[str] = None
    description: str = ""
    capabilities: Set[str] = field(default_factory=set)
    invariants: List[str] = field(default_factory=list)
    accepts: List[str] = field(default_factory=list)  # Input DTO types or concepts
    produces: List[str] = field(default_factory=list)  # Output DTO types or concepts
    author: str = "Nireon Team"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    epistemic_tags: List[str] = field(default_factory=list)
    expected_interfaces: Optional[List[Type[Any]]] = field(default_factory=list)
    requires_initialize: bool = True
    
    # New enhanced fields (with defaults for backward compatibility)
    dependencies: Dict[str, str] = field(default_factory=dict)  # component_id -> version_spec
    conflicts_with: Set[str] = field(default_factory=set)  # Set of incompatible component IDs
    runtime_state: Dict[str, Any] = field(default_factory=dict)  # For tracking runtime metadata
    _validation_rules: List[Callable[[ComponentMetadata], bool]] = field(default_factory=list, repr=False)
    
    def __post_init__(self) -> None:
        """Enhanced validation with backward compatibility."""
        # Original validation
        if not all([self.id, self.name, self.version, self.category]):
            raise ValueError("ComponentMetadata fields id, name, version, category must be non-empty.")
        
        if self.created_at.tzinfo is None:
            self.created_at = self.created_at.replace(tzinfo=timezone.utc)

        if not isinstance(self.epistemic_tags, list):
            raise TypeError("epistemic_tags must be a list.")
        for tag in self.epistemic_tags:
            if not isinstance(tag, str):
                raise TypeError(f"All epistemic_tags must be strings, got {type(tag)} for tag '{tag}'.")
        
        if self.expected_interfaces is not None:
            if not isinstance(self.expected_interfaces, list):
                raise TypeError("expected_interfaces must be a list of types or None.")
            for iface_type in self.expected_interfaces:
                if not isinstance(iface_type, type):
                    logger.warning(
                        f"Item '{iface_type}' in expected_interfaces for component '{self.id}' "
                        f"might not be a valid type/Protocol. Expected a class/type."
                    )
        
        if not isinstance(self.requires_initialize, bool):
            raise TypeError(f"'requires_initialize' for component '{self.id}' must be a boolean.")
        
        # New validation
        self._validate_version_format()
        self._validate_dependencies()
        self._run_custom_validations()
    
    def _validate_version_format(self) -> None:
        """Validate version follows semantic versioning."""
        parts = self.version.split('.')
        if len(parts) != 3:
            logger.warning(f"Version '{self.version}' for component '{self.id}' "
                         f"doesn't follow semantic versioning (x.y.z)")
            return
        
        try:
            major, minor, patch = parts
            int(major), int(minor), int(patch)
        except ValueError:
            logger.warning(f"Version '{self.version}' for component '{self.id}' "
                         f"contains non-numeric parts")
    
    def _validate_dependencies(self) -> None:
        """Validate dependency specifications."""
        for dep_id, version_spec in self.dependencies.items():
            if not isinstance(dep_id, str) or not isinstance(version_spec, str):
                raise TypeError(f"Dependencies must be string mappings, "
                              f"got {type(dep_id).__name__} -> {type(version_spec).__name__}")
    
    def _run_custom_validations(self) -> None:
        """Run any custom validation rules."""
        for rule in self._validation_rules:
            try:
                if not rule(self):
                    raise MetadataValidationError(
                        f"Custom validation failed for component '{self.id}'"
                    )
            except Exception as e:
                logger.error(f"Error in custom validation for '{self.id}': {e}")
                raise
    
    @cached_property
    def version_tuple(self) -> tuple[int, int, int]:
        """Get version as a comparable tuple."""
        try:
            parts = self.version.split('.')
            if len(parts) == 3:
                return (int(parts[0]), int(parts[1]), int(parts[2]))
        except (ValueError, AttributeError):
            pass
        return (0, 0, 0)  # Default for invalid versions
    
    def is_compatible_with(self, other: ComponentMetadata) -> bool:
        """Check if this component is compatible with another."""
        # Check for explicit conflicts
        if other.id in self.conflicts_with or self.id in other.conflicts_with:
            return False
        
        # Check if capabilities match requirements
        if self.produces and other.accepts:
            # Check if any of our outputs match their inputs
            if not any(output in other.accepts for output in self.produces):
                logger.debug(f"Component '{self.id}' produces {self.produces} "
                           f"but '{other.id}' accepts {other.accepts}")
        
        return True
    
    def satisfies_dependency(self, version_spec: str) -> bool:
        """Check if this component satisfies a version specification."""
        # Simple implementation - can be enhanced with proper version parsing
        if version_spec == "*" or version_spec == "latest":
            return True
        
        if version_spec.startswith(">="):
            required = version_spec[2:].strip()
            try:
                req_tuple = tuple(int(x) for x in required.split('.'))
                return self.version_tuple >= req_tuple
            except ValueError:
                logger.warning(f"Invalid version spec: {version_spec}")
                return False
        
        # Exact match
        return self.version == version_spec
    
    def add_capability(self, capability: str) -> None:
        """Add a capability with validation."""
        if not isinstance(capability, str):
            raise TypeError(f"Capability must be a string, got {type(capability).__name__}")
        self.capabilities.add(capability)
        logger.debug(f"Added capability '{capability}' to component '{self.id}'")
    
    def add_runtime_state(self, key: str, value: Any) -> None:
        """Add or update runtime state information."""
        self.runtime_state[key] = value
        self.runtime_state['last_updated'] = datetime.now(timezone.utc).isoformat()
    
    def get_runtime_state(self, key: str, default: Any = None) -> Any:
        """Get runtime state value."""
        return self.runtime_state.get(key, default)
    
    def clear_runtime_state(self) -> None:
        """Clear all runtime state."""
        self.runtime_state.clear()
    
    def add_validation_rule(self, rule: Callable[[ComponentMetadata], bool]) -> None:
        """Add a custom validation rule."""
        self._validation_rules.append(rule)
    
    @property
    def fingerprint(self) -> str:
        """Generate a unique fingerprint for this metadata."""
        # Create a stable representation
        data = {
            'id': self.id,
            'name': self.name,
            'version': self.version,
            'category': self.category,
            'capabilities': sorted(list(self.capabilities)),
            'accepts': sorted(self.accepts),
            'produces': sorted(self.produces),
            'requires_initialize': self.requires_initialize
        }
        
        json_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(json_str.encode()).hexdigest()[:16]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        # Convert datetime to ISO format
        data['created_at'] = self.created_at.isoformat()
        # Convert sets to lists for JSON serialization
        data['capabilities'] = list(self.capabilities)
        data['conflicts_with'] = list(self.conflicts_with)
        # Remove non-serializable fields
        data.pop('_validation_rules', None)
        data.pop('expected_interfaces', None)  # Can't easily serialize types
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ComponentMetadata:
        """Create from dictionary (deserialization)."""
        # Convert ISO string back to datetime
        if 'created_at' in data and isinstance(data['created_at'], str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        
        # Convert lists back to sets
        if 'capabilities' in data:
            data['capabilities'] = set(data['capabilities'])
        if 'conflicts_with' in data:
            data['conflicts_with'] = set(data['conflicts_with'])
        
        # Remove any unknown fields
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        data = {k: v for k, v in data.items() if k in valid_fields}
        
        return cls(**data)
    
    def validate_interfaces(self, component_instance: Any) -> List[str]:
        """Validate that a component instance implements expected interfaces."""
        errors = []
        
        if not self.expected_interfaces:
            return errors
        
        for expected_type in self.expected_interfaces:
            if not isinstance(component_instance, expected_type):
                errors.append(
                    f"Component '{self.id}' instance does not implement "
                    f"expected interface {expected_type.__name__}"
                )
        
        return errors
    
    def __str__(self) -> str:
        """Enhanced string representation."""
        return (f"{self.name} v{self.version} (ID: {self.id}, "
                f"Category: {self.category}, "
                f"Capabilities: {len(self.capabilities)})")
    
    def __hash__(self) -> int:
        """Make metadata hashable based on ID."""
        return hash(self.id)
    
    def __eq__(self, other: Any) -> bool:
        """Equality based on ID and version."""
        if not isinstance(other, ComponentMetadata):
            return False
        return self.id == other.id and self.version == other.version
    
    def __lt__(self, other: ComponentMetadata) -> bool:
        """Compare by version for sorting."""
        if self.id != other.id:
            return self.id < other.id
        return self.version_tuple < other.version_tuple


class ComponentRegistryMissingError(KeyError):
    """Custom exception for when a component is not found in the registry.
    
    Enhanced with additional context about what was being looked for.
    """
    def __init__(self, 
                 component_id: str, 
                 message: Optional[str] = None,
                 available_components: Optional[List[str]] = None):
        default_message = f"Component with ID '{component_id}' not found in the registry."
        
        if available_components:
            default_message += f" Available components: {', '.join(available_components[:5])}"
            if len(available_components) > 5:
                default_message += f" and {len(available_components) - 5} more..."
        
        final_message = message if message is not None else default_message
        super().__init__(final_message)
        self.component_id = component_id
        self.available_components = available_components or []


# Utility functions for working with metadata

def create_metadata_validator(
    required_capabilities: Optional[Set[str]] = None,
    required_category: Optional[str] = None,
    min_version: Optional[str] = None
) -> Callable[[ComponentMetadata], bool]:
    """Create a reusable metadata validator function."""
    def validator(metadata: ComponentMetadata) -> bool:
        if required_capabilities and not required_capabilities.issubset(metadata.capabilities):
            return False
        
        if required_category and metadata.category != required_category:
            return False
        
        if min_version:
            try:
                min_tuple = tuple(int(x) for x in min_version.split('.'))
                if metadata.version_tuple < min_tuple:
                    return False
            except ValueError:
                logger.warning(f"Invalid minimum version: {min_version}")
        
        return True
    
    return validator


def merge_metadata(base: ComponentMetadata, updates: Dict[str, Any]) -> ComponentMetadata:
    """Create a new metadata instance with updates applied."""
    data = base.to_dict()
    data.update(updates)
    return ComponentMetadata.from_dict(data)