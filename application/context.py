# Combined nireon_v4/application/context.py
from __future__ import annotations
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Dict, Optional

# V4 imports - using the more complete import structure from version 2
from core.registry import ComponentRegistry
from application.ports.event_bus_port import EventBusPort

# Placeholders for components that may not yet be fully implemented in V4
# These can be properly typed or imported later when available
LoggerAdapter = Any
ConfigProvider = Any
StateManager = Any


class NireonExecutionContext:
    """
    Execution context for Nireon framework operations.
    
    Provides immutable-style context management with support for cloning
    and scoped modifications while maintaining thread-safe operations.
    """
    
    def __init__(
        self,
        *,
        run_id: str,
        step: int = 0,
        feature_flags: Optional[Dict[str, Any]] = None,
        component_registry: Optional[ComponentRegistry] = None,
        event_bus: Optional[EventBusPort] = None,
        config: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        component_id: Optional[str] = None,
        timestamp: Optional[datetime] = None,
        replay_mode: bool = False,
        replay_seed: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
        logger: Optional[LoggerAdapter] = None,
        config_provider: Optional[ConfigProvider] = None,
        state_manager: Optional[StateManager] = None,
    ) -> None:
        self.run_id: str = run_id
        self.step: int = step
        self.feature_flags: Dict[str, Any] = feature_flags or {}
        self.component_registry = component_registry
        self.event_bus = event_bus
        self.logger = logger
        self.config_provider = config_provider
        self.state_manager = state_manager
        self.config = config or {}
        self.session_id = session_id
        self.component_id = component_id
        self.timestamp: datetime = timestamp or datetime.now(timezone.utc)
        self.replay_mode = replay_mode
        self.replay_seed = replay_seed
        self.metadata: Dict[str, Any] = metadata or {}
        self._custom_data: Dict[str, Any] = {}  # For arbitrary data passing

    def is_flag_enabled(self, flag_name: str, default: bool = False) -> bool:
        """Check if a feature flag is enabled."""
        return bool(self.feature_flags.get(flag_name, default))

    def get_custom_data(self, key: str, default: Any = None) -> Any:
        """Retrieve custom data by key."""
        return self._custom_data.get(key, default)

    def set_custom_data(self, key: str, value: Any) -> "NireonExecutionContext":
        """Set custom data and return a new context instance."""
        new_custom = deepcopy(self._custom_data)
        new_custom[key] = value
        return self._clone(_internal_custom_data_override=new_custom)

    def with_component_scope(self, component_id: str) -> "NireonExecutionContext":
        """Create a new context scoped to a specific component."""
        return self._clone(component_id=component_id)

    def with_step(self, step: int) -> "NireonExecutionContext":
        """Create a new context with an updated step."""
        return self._clone(step=step)

    def with_metadata(self, **updates) -> "NireonExecutionContext":
        """Create a new context with updated metadata."""
        new_meta = {**self.metadata, **updates}
        return self._clone(metadata=new_meta)

    def with_flags(self, **flag_updates) -> "NireonExecutionContext":
        """Create a new context with updated feature flags."""
        new_flags = {**self.feature_flags, **flag_updates}
        return self._clone(feature_flags=new_flags)

    def advance_step(self, new_step: int) -> "NireonExecutionContext":
        """Advance the execution step and return a new context."""
        return self._clone(step=new_step)

    def _clone(self, **overrides) -> "NireonExecutionContext":
        """
        Create a deep clone of the context with optional overrides.
        
        Uses deepcopy for mutable fields to ensure true isolation between
        context instances while sharing immutable/singleton resources.
        """
        params = {
            "run_id": self.run_id,
            "step": self.step,
            "feature_flags": deepcopy(self.feature_flags),
            "component_registry": self.component_registry,  # Shared singleton
            "event_bus": self.event_bus,  # Shared singleton
            "config": deepcopy(self.config),
            "session_id": self.session_id,
            "component_id": self.component_id,
            "timestamp": self.timestamp,  # Keep original timestamp
            "replay_mode": self.replay_mode,
            "replay_seed": self.replay_seed,
            "metadata": deepcopy(self.metadata),
            "logger": self.logger,  # Shared or re-scoped
            "config_provider": self.config_provider,  # Shared singleton
            "state_manager": self.state_manager,  # Shared singleton
        }
        
        # Handle _custom_data separately for precise control
        internal_custom_data_override = overrides.pop("_internal_custom_data_override", None)
        params.update(overrides)
        
        new_instance = NireonExecutionContext(**params)
        
        if internal_custom_data_override is not None:
            new_instance._custom_data = internal_custom_data_override
        else:
            new_instance._custom_data = deepcopy(self._custom_data)
            
        return new_instance

    def __repr__(self) -> str:
        return (
            f"NireonExecutionContext(run_id={self.run_id!r}, step={self.step}, "
            f"component_id={self.component_id!r}, session_id={self.session_id!r}, "
            f"feature_flags={self.feature_flags})"
        )


# Alias for backward compatibility with V3 code
ExecutionContext = NireonExecutionContext