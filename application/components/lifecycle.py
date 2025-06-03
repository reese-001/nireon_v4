# Adapted from nireon_staging/nireon/application/components/lifecycle.py
# V4: ComponentRegistry class itself is moved to core.registry.component_registry
# This file now primarily holds ComponentMetadata and related errors/types.
from __future__ import annotations

import logging
import threading # Keep for potential future use if other classes here need it
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Union, Type, Protocol

__all__ = ['ComponentMetadata', 'ComponentRegistryMissingError'] # V4: ComponentRegistry removed from __all__

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
    # V4: expected_interfaces aligns with V4 Component Developer Guide
    expected_interfaces: Optional[List[Type[Any]]] = field(default_factory=list)
    requires_initialize: bool = True # V4: Added as per V3, useful for bootstrap

    def __post_init__(self) -> None:
        if not all([self.id, self.name, self.version, self.category]):
            raise ValueError("ComponentMetadata fields id, name, version, category must be non-empty")
        
        if self.created_at.tzinfo is None: # Ensure timezone-aware datetime
            self.created_at = self.created_at.replace(tzinfo=timezone.utc)

        if not isinstance(self.epistemic_tags, list):
            raise TypeError("epistemic_tags must be a list of strings")
        for tag in self.epistemic_tags:
            if not isinstance(tag, str):
                raise TypeError(f"All epistemic_tags must be strings, got {type(tag)}")
        
        if self.expected_interfaces is not None and not isinstance(self.expected_interfaces, list):
            raise TypeError("expected_interfaces must be a list of Protocol types or None")
        if self.expected_interfaces:
            for iface in self.expected_interfaces:
                # Basic check, Protocol type checking is complex
                if not isinstance(iface, type) or not hasattr(iface, '__mro__'):
                    logger.warning(f"Item '{iface}' in expected_interfaces for '{self.id}' might not be a valid Protocol type.")
        
        if not isinstance(self.requires_initialize, bool):
            raise TypeError(f"'requires_initialize' for '{self.id}' must be a boolean.")


class ComponentRegistryMissingError(KeyError):
    """Custom exception for when a component is not found in the registry."""
    def __init__(self, component_id: str, message: Optional[str] = None):
        default_message = f"Component '{component_id}' not found in registry."
        final_message = message if message is not None else default_message
        super().__init__(final_message)
        self.component_id = component_id

# V4: The ComponentRegistry class is moved to nireon_v4.core.registry.component_registry
# The content of the V3 ComponentRegistry class will be adapted there.