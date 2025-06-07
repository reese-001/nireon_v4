# application/components/lifecycle.py
from __future__ import annotations

import logging
import threading # Kept for potential future use if other classes here might need it
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Type, Protocol # Protocol imported for type hinting clarity

__all__ = ['ComponentMetadata', 'ComponentRegistryMissingError']

logger = logging.getLogger(__name__)


@dataclass
class ComponentMetadata:
    id: str
    name: str
    version: str
    category: str # e.g., 'mechanism', 'observer', 'service', 'core_service', 'persistence_service'
    subcategory: Optional[str] = None
    description: str = ""
    capabilities: Set[str] = field(default_factory=set)
    invariants: List[str] = field(default_factory=list)
    accepts: List[str] = field(default_factory=list) # Input DTO types or concepts
    produces: List[str] = field(default_factory=list) # Output DTO types or concepts
    author: str = "Nireon Team"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    epistemic_tags: List[str] = field(default_factory=list)
    # expected_interfaces allows specifying types (ideally Protocols) the component expects to interact with.
    expected_interfaces: Optional[List[Type[Any]]] = field(default_factory=list) # Type[Any] for flexibility; could be Type[Protocol] if strictly enforced
    requires_initialize: bool = True # Indicates if the component's initialize() method must be called

    def __post_init__(self) -> None:
        if not all([self.id, self.name, self.version, self.category]):
            raise ValueError("ComponentMetadata fields id, name, version, category must be non-empty.")
        
        if self.created_at.tzinfo is None: # Ensure timezone-aware datetime
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
                # A basic check. Full Protocol type checking is complex and often done by static analyzers.
                # Here, we just ensure it's a type (class).
                if not isinstance(iface_type, type): # or (isinstance(iface_type, type) and not issubclass(iface_type, Protocol))
                    logger.warning(
                        f"Item '{iface_type}' in expected_interfaces for component '{self.id}' "
                        f"might not be a valid type/Protocol. Expected a class/type."
                    )
        
        if not isinstance(self.requires_initialize, bool):
            raise TypeError(f"'requires_initialize' for component '{self.id}' must be a boolean.")


class ComponentRegistryMissingError(KeyError):
    """Custom exception for when a component is not found in the registry."""
    def __init__(self, component_id: str, message: Optional[str] = None):
        default_message = f"Component with ID '{component_id}' not found in the registry."
        final_message = message if message is not None else default_message
        super().__init__(final_message)
        self.component_id = component_id

# Note: The ComponentRegistry class itself is now located in nireon.core.registry.component_registry