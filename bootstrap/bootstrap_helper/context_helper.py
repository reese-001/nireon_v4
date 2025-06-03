"""
Execution context builder for NIREON V4 bootstrap process.

This module provides utilities for creating and configuring NireonExecutionContext
instances used throughout the bootstrap process and component lifecycle.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from core.registry import ComponentRegistry
from application.context import NireonExecutionContext
from application.ports.event_bus_port import EventBusPort

logger = logging.getLogger(__name__)


class SimpleConfigProvider:
    """
    Simple configuration provider for execution contexts.
    
    This provider wraps feature flags and configuration data
    to provide a consistent interface for component configuration.
    """
    
    def __init__(self, flags: Dict[str, Any]):
        self._flags = flags or {}
        logger.debug(f"SimpleConfigProvider initialized with {len(self._flags)} flags")
    
    def get_config(self, component_id: str, key: str, default: Any = None) -> Any:
        """
        Get configuration value for a component.
        
        Args:
            component_id: Component requesting configuration
            key: Configuration key
            default: Default value if not found
            
        Returns:
            Configuration value or default
        """
        # First try component-specific config
        component_key = f"{component_id}.{key}"
        if component_key in self._flags:
            return self._flags[component_key]
        
        # Fall back to global config
        return self._flags.get(key, default)
    
    def has_config(self, key: str) -> bool:
        """Check if a configuration key exists."""
        return key in self._flags
    
    def get_all_config(self) -> Dict[str, Any]:
        """Get all configuration as a dictionary."""
        return dict(self._flags)


def build_execution_context(
    component_id: str,
    run_id: str,
    registry: ComponentRegistry,
    event_bus: EventBusPort,
    feature_flags: Optional[Dict[str, Any]] = None,
    replay: bool = False,
    step: int = 0,
    session_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None
) -> NireonExecutionContext:
    """
    Build a NireonExecutionContext for component operations.
    
    Args:
        component_id: ID of the component this context is for
        run_id: Unique run identifier
        registry: Component registry instance
        event_bus: Event bus for communication
        feature_flags: Feature flags and configuration
        replay: Whether this is a replay execution
        step: Current execution step
        session_id: Optional session identifier
        metadata: Additional metadata
        config: Additional configuration
        
    Returns:
        Configured NireonExecutionContext
    """
    logger.debug(f"Building execution context for component '{component_id}' (run_id: {run_id})")
    
    # Create logger instance for the component
    logger_instance = _get_component_logger(component_id, event_bus)
    
    # Create config provider
    config_provider = SimpleConfigProvider(feature_flags or {})
    
    # Build context
    context = NireonExecutionContext(
        run_id=run_id,
        step=step,
        feature_flags=feature_flags or {},
        component_registry=registry,
        event_bus=event_bus,
        config=config or {},
        session_id=session_id,
        component_id=component_id,
        timestamp=datetime.now(timezone.utc),
        replay_mode=replay,
        replay_seed=12345 if replay else None,
        metadata=metadata or {},
        logger=logger_instance,
        config_provider=config_provider,
        state_manager=None  # Will be set up later if needed
    )
    
    logger.debug(f"Execution context built for '{component_id}' with {len(feature_flags or {})} feature flags")
    return context


def build_bootstrap_context(
    run_id: str,
    registry: ComponentRegistry,
    event_bus: EventBusPort,
    global_config: Dict[str, Any],
    step: int = 0
) -> NireonExecutionContext:
    """
    Build an execution context specifically for bootstrap operations.
    
    Args:
        run_id: Bootstrap run identifier
        registry: Component registry
        event_bus: Event bus
        global_config: Global application configuration
        step: Current bootstrap step
        
    Returns:
        Bootstrap execution context
    """
    feature_flags = global_config.get('feature_flags', {})
    
    return build_execution_context(
        component_id='bootstrap_system',
        run_id=run_id,
        registry=registry,
        event_bus=event_bus,
        feature_flags=feature_flags,
        replay=False,
        step=step,
        session_id=f"bootstrap_{run_id}",
        metadata={
            'bootstrap_version': '4.0',
            'bootstrap_mode': 'standard',
            'strict_mode': global_config.get('bootstrap_strict_mode', True)
        },
        config=global_config
    )


def build_component_init_context(
    component_id: str,
    base_context: NireonExecutionContext,
    component_config: Optional[Dict[str, Any]] = None
) -> NireonExecutionContext:
    """
    Build an execution context for component initialization.
    
    Args:
        component_id: Component being initialized
        base_context: Base context to derive from
        component_config: Component-specific configuration
        
    Returns:
        Component initialization context
    """
    return base_context.with_component_scope(component_id).with_metadata(
        initialization_step=True,
        component_config_keys=list((component_config or {}).keys())
    )


def build_validation_context(
    component_id: str,
    base_context: NireonExecutionContext,
    validation_data: Optional[Dict[str, Any]] = None
) -> NireonExecutionContext:
    """
    Build an execution context for component validation.
    
    Args:
        component_id: Component being validated
        base_context: Base context to derive from
        validation_data: Validation-specific data
        
    Returns:
        Component validation context
    """
    return base_context.with_component_scope(component_id).with_metadata(
        validation_step=True,
        validation_data=validation_data or {}
    )


def _get_component_logger(component_id: str, event_bus: EventBusPort) -> logging.Logger:
    """
    Get or create a logger for a component.
    
    Args:
        component_id: Component identifier
        event_bus: Event bus (may provide logger if it has get_logger method)
        
    Returns:
        Logger instance for the component
    """
    # Try to get logger from event bus if it supports it
    if hasattr(event_bus, 'get_logger') and callable(event_bus.get_logger):
        try:
            return event_bus.get_logger(component_id)
        except Exception as e:
            logger.warning(f"Could not get logger from event bus for '{component_id}': {e}")
    
    # Fall back to standard logger
    return logging.getLogger(f'nireon.{component_id}')


class ContextBuilder:
    """
    Builder class for creating execution contexts with fluent interface.
    
    This provides a more flexible way to build contexts when you need
    to set multiple optional parameters.
    """
    
    def __init__(self, component_id: str, run_id: str):
        self.component_id = component_id
        self.run_id = run_id
        self.registry: Optional[ComponentRegistry] = None
        self.event_bus: Optional[EventBusPort] = None
        self.feature_flags: Dict[str, Any] = {}
        self.replay = False
        self.step = 0
        self.session_id: Optional[str] = None
        self.metadata: Dict[str, Any] = {}
        self.config: Dict[str, Any] = {}
    
    def with_registry(self, registry: ComponentRegistry) -> 'ContextBuilder':
        """Set the component registry."""
        self.registry = registry
        return self
    
    def with_event_bus(self, event_bus: EventBusPort) -> 'ContextBuilder':
        """Set the event bus."""
        self.event_bus = event_bus
        return self
    
    def with_feature_flags(self, flags: Dict[str, Any]) -> 'ContextBuilder':
        """Set feature flags."""
        self.feature_flags = flags
        return self
    
    def with_replay(self, replay: bool = True) -> 'ContextBuilder':
        """Enable or disable replay mode."""
        self.replay = replay
        return self
    
    def with_step(self, step: int) -> 'ContextBuilder':
        """Set the execution step."""
        self.step = step
        return self
    
    def with_session_id(self, session_id: str) -> 'ContextBuilder':
        """Set the session ID."""
        self.session_id = session_id
        return self
    
    def with_metadata(self, **metadata) -> 'ContextBuilder':
        """Add metadata to the context."""
        self.metadata.update(metadata)
        return self
    
    def with_config(self, config: Dict[str, Any]) -> 'ContextBuilder':
        """Set configuration."""
        self.config = config
        return self
    
    def build(self) -> NireonExecutionContext:
        """Build the execution context."""
        if not self.registry:
            raise ValueError("Component registry is required")
        if not self.event_bus:
            raise ValueError("Event bus is required")
        
        return build_execution_context(
            component_id=self.component_id,
            run_id=self.run_id,
            registry=self.registry,
            event_bus=self.event_bus,
            feature_flags=self.feature_flags,
            replay=self.replay,
            step=self.step,
            session_id=self.session_id,
            metadata=self.metadata,
            config=self.config
        )


def create_context_builder(component_id: str, run_id: str) -> ContextBuilder:
    """
    Create a new context builder.
    
    Args:
        component_id: Component identifier
        run_id: Run identifier
        
    Returns:
        New ContextBuilder instance
    """
    return ContextBuilder(component_id, run_id)


# Convenience functions for common context patterns
def create_test_context(
    component_id: str = "test_component",
    run_id: str = "test_run",
    registry: Optional[ComponentRegistry] = None,
    event_bus: Optional[EventBusPort] = None
) -> NireonExecutionContext:
    """
    Create a minimal context for testing.
    
    Args:
        component_id: Test component ID
        run_id: Test run ID
        registry: Optional registry (creates placeholder if None)
        event_bus: Optional event bus (creates placeholder if None)
        
    Returns:
        Test execution context
    """
    if not registry:
        registry = ComponentRegistry()
    
    if not event_bus:
        from .placeholders import PlaceholderEventBusImpl
        event_bus = PlaceholderEventBusImpl()
    
    return build_execution_context(
        component_id=component_id,
        run_id=run_id,
        registry=registry,
        event_bus=event_bus,
        feature_flags={'test_mode': True},
        metadata={'test_context': True}
    )


def create_minimal_context(component_id: str, run_id: str) -> NireonExecutionContext:
    """
    Create a minimal context with placeholder services.
    
    Args:
        component_id: Component identifier
        run_id: Run identifier
        
    Returns:
        Minimal execution context with placeholders
    """
    from .placeholders import PlaceholderEventBusImpl
    
    registry = ComponentRegistry()
    event_bus = PlaceholderEventBusImpl()
    
    return build_execution_context(
        component_id=component_id,
        run_id=run_id,
        registry=registry,
        event_bus=event_bus
    )