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

T = TypeVar('T')

class MetadataValidationError(ValueError):
    pass

@dataclass
class ComponentMetadata:
    id: str
    name: str
    version: str
    category: str
    subcategory: Optional[str] = None
    description: str = ''
    capabilities: Set[str] = field(default_factory=set)
    invariants: List[str] = field(default_factory=list)
    accepts: List[str] = field(default_factory=list)
    produces: List[str] = field(default_factory=list)
    author: str = 'Nireon Team'
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    epistemic_tags: List[str] = field(default_factory=list)
    expected_interfaces: Optional[List[Type[Any]]] = field(default_factory=list)
    requires_initialize: bool = True
    dependencies: Dict[str, str] = field(default_factory=dict)
    conflicts_with: Set[str] = field(default_factory=set)
    runtime_state: Dict[str, Any] = field(default_factory=dict)
    _validation_rules: List[Callable[[ComponentMetadata], bool]] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        if not all([self.id, self.name, self.version, self.category]):
            raise ValueError('ComponentMetadata fields id, name, version, category must be non-empty.')
        if self.created_at.tzinfo is None:
            self.created_at = self.created_at.replace(tzinfo=timezone.utc)
        if not isinstance(self.epistemic_tags, list):
            raise TypeError('epistemic_tags must be a list.')
        for tag in self.epistemic_tags:
            if not isinstance(tag, str):
                raise TypeError(f"All epistemic_tags must be strings, got {type(tag)} for tag '{tag}'.")

        if self.expected_interfaces is not None:
            if not isinstance(self.expected_interfaces, list):
                raise TypeError('expected_interfaces must be a list of types or None.')
            for iface_type in self.expected_interfaces:
                if not isinstance(iface_type, type):
                    logger.warning(f"Item '{iface_type}' in expected_interfaces for component '{self.id}' might not be a valid type/Protocol. Expected a class/type.")

        if not isinstance(self.requires_initialize, bool):
            raise TypeError(f"'requires_initialize' for component '{self.id}' must be a boolean.")

        self._validate_version_format()
        self._validate_dependencies()
        self._run_custom_validations()

    def _validate_version_format(self) -> None:
        parts = self.version.split('.')
        if len(parts) != 3:
            logger.warning(f"Version '{self.version}' for component '{self.id}' doesn't follow semantic versioning (x.y.z)")
            return
        try:
            major, minor, patch = parts
            int(major), int(minor), int(patch)
        except ValueError:
            logger.warning(f"Version '{self.version}' for component '{self.id}' contains non-numeric parts")

    def _validate_dependencies(self) -> None:
        for dep_id, version_spec in self.dependencies.items():
            if not isinstance(dep_id, str) or not isinstance(version_spec, str):
                raise TypeError(f'Dependencies must be string mappings, got {type(dep_id).__name__} -> {type(version_spec).__name__}')

    def _run_custom_validations(self) -> None:
        for rule in self._validation_rules:
            try:
                if not rule(self):
                    raise MetadataValidationError(f"Custom validation failed for component '{self.id}'")
            except Exception as e:
                logger.error(f"Error in custom validation for '{self.id}': {e}")
                raise

    @cached_property
    def version_tuple(self) -> tuple[int, int, int]:
        try:
            parts = self.version.split('.')
            if len(parts) == 3:
                return int(parts[0]), int(parts[1]), int(parts[2])
        except (ValueError, AttributeError):
            pass
        return 0, 0, 0

    def is_compatible_with(self, other: ComponentMetadata) -> bool:
        if other.id in self.conflicts_with or self.id in other.conflicts_with:
            return False
        if self.produces and other.accepts:
            if not any(output in other.accepts for output in self.produces):
                logger.debug(f"Component '{self.id}' produces {self.produces} but '{other.id}' accepts {other.accepts}")
        return True

    def satisfies_dependency(self, version_spec: str) -> bool:
        if version_spec == '*' or version_spec == 'latest':
            return True
        if version_spec.startswith('>='):
            required = version_spec[2:].strip()
            try:
                req_tuple = tuple(int(x) for x in required.split('.'))
                return self.version_tuple >= req_tuple
            except ValueError:
                logger.warning(f'Invalid version spec: {version_spec}')
                return False
        return self.version == version_spec

    def add_capability(self, capability: str) -> None:
        if not isinstance(capability, str):
            raise TypeError(f'Capability must be a string, got {type(capability).__name__}')
        self.capabilities.add(capability)
        logger.debug(f"Added capability '{capability}' to component '{self.id}'")

    def add_runtime_state(self, key: str, value: Any) -> None:
        self.runtime_state[key] = value
        self.runtime_state['last_updated'] = datetime.now(timezone.utc).isoformat()

    def get_runtime_state(self, key: str, default: Any = None) -> Any:
        return self.runtime_state.get(key, default)

    def clear_runtime_state(self) -> None:
        self.runtime_state.clear()

    def add_validation_rule(self, rule: Callable[['ComponentMetadata'], bool]) -> None:
        self._validation_rules.append(rule)

    @property
    def fingerprint(self) -> str:
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
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        data['capabilities'] = list(self.capabilities)
        data['conflicts_with'] = list(self.conflicts_with)
        data.pop('_validation_rules', None)
        data.pop('expected_interfaces', None)
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ComponentMetadata:
        if 'created_at' in data and isinstance(data['created_at'], str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        if 'capabilities' in data:
            data['capabilities'] = set(data['capabilities'])
        if 'conflicts_with' in data:
            data['conflicts_with'] = set(data['conflicts_with'])
        
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        data = {k: v for k, v in data.items() if k in valid_fields}
        
        return cls(**data)
        
    def validate_interfaces(self, component_instance: Any) -> List[str]:
        errors = []
        if not self.expected_interfaces:
            return errors
        for expected_type in self.expected_interfaces:
            if not isinstance(component_instance, expected_type):
                errors.append(f"Component '{self.id}' instance does not implement expected interface {expected_type.__name__}")
        return errors

    def __str__(self) -> str:
        return f'{self.name} v{self.version} (ID: {self.id}, Category: {self.category}, Capabilities: {len(self.capabilities)})'
    
    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, ComponentMetadata):
            return False
        return self.id == other.id and self.version == other.version
    
    def __lt__(self, other: 'ComponentMetadata') -> bool:
        if self.id != other.id:
            return self.id < other.id
        return self.version_tuple < other.version_tuple

class ComponentRegistryMissingError(KeyError):
    def __init__(self, component_id: str, message: Optional[str]=None, available_components: Optional[List[str]]=None):
        default_message = f"Component with ID '{component_id}' not found in the registry."
        if available_components:            
            sorted_keys = sorted(available_components)
            default_message += f" Available components ({len(sorted_keys)} total): {', '.join(sorted_keys)}"
        final_message = message if message is not None else default_message
        super().__init__(final_message)
        self.component_id = component_id
        self.available_components = available_components or []

def create_metadata_validator(
    required_capabilities: Optional[Set[str]] = None,
    required_category: Optional[str] = None,
    min_version: Optional[str] = None
) -> Callable[[ComponentMetadata], bool]:
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
                logger.warning(f'Invalid minimum version: {min_version}')
        return True
    return validator

def merge_metadata(base: ComponentMetadata, updates: Dict[str, Any]) -> ComponentMetadata:
    data = base.to_dict()
    data.update(updates)
    return ComponentMetadata.from_dict(data)