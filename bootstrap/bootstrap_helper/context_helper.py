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
        component_key = f"{component_id}.{key}"
        if component_key in self._flags:
            return self._flags[component_key]
        
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
    """
    logger.debug(f"Building execution context for component '{component_id}' (run_id: {run_id})")
    
    logger_instance = _get_component_logger(component_id, event_bus)
    config_provider = SimpleConfigProvider(feature_flags or {})
    
    context = NireonExecutionContext(
        run_id=run_id,
        step=step,
        feature_flags=feature_flags or {},
        component_registry=registry,  # Correctly uses component_registry kwarg
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
        state_manager=None
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
    """
    return base_context.with_component_scope(component_id).with_metadata(
        validation_step=True,
        validation_data=validation_data or {}
    )


def _get_component_logger(component_id: str, event_bus: EventBusPort) -> logging.Logger:
    """
    Get or create a logger for a component.
    """
    if hasattr(event_bus, 'get_logger') and callable(event_bus.get_logger):
        try:
            return event_bus.get_logger(component_id)
        except Exception as e:
            logger.warning(f"Could not get logger from event bus for '{component_id}': {e}")
    
    return logging.getLogger(f'nireon.{component_id}')


class ContextBuilder:
    """
    Builder class for creating execution contexts with fluent interface.
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
        self.registry = registry
        return self
    
    def with_event_bus(self, event_bus: EventBusPort) -> 'ContextBuilder':
        self.event_bus = event_bus
        return self
    
    def with_feature_flags(self, flags: Dict[str, Any]) -> 'ContextBuilder':
        self.feature_flags = flags
        return self
    
    def with_replay(self, replay: bool = True) -> 'ContextBuilder':
        self.replay = replay
        return self
    
    def with_step(self, step: int) -> 'ContextBuilder':
        self.step = step
        return self
    
    def with_session_id(self, session_id: str) -> 'ContextBuilder':
        self.session_id = session_id
        return self
    
    def with_metadata(self, **metadata) -> 'ContextBuilder':
        self.metadata.update(metadata)
        return self
    
    def with_config(self, config: Dict[str, Any]) -> 'ContextBuilder':
        self.config = config
        return self
    
    def build(self) -> NireonExecutionContext:
        """Build the execution context."""
        if not self.registry:
            raise ValueError("Component registry is required for ContextBuilder.build()")
        if not self.event_bus:
            raise ValueError("Event bus is required for ContextBuilder.build()")
        
        return build_execution_context(
            component_id=self.component_id,
            run_id=self.run_id,
            registry=self.registry,  # FIXED: Use self.registry
            event_bus=self.event_bus,
            feature_flags=self.feature_flags,
            replay=self.replay,
            step=self.step,
            session_id=self.session_id,
            metadata=self.metadata,
            config=self.config
        )


def create_context_builder(component_id: str, run_id: str) -> ContextBuilder:
    """Create a new context builder."""
    return ContextBuilder(component_id, run_id)


def create_test_context(
    component_id: str = "test_component",
    run_id: str = "test_run",
    registry: Optional[ComponentRegistry] = None,
    event_bus: Optional[EventBusPort] = None
) -> NireonExecutionContext:
    """Create a minimal context for testing."""
    if not registry:
        registry = ComponentRegistry() # Assumes ComponentRegistry can be init'd without args for testing
    
    if not event_bus:
        # Assuming placeholders is a module in the same directory (bootstrap)
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
    """Create a minimal context with placeholder services."""
    # Assuming placeholders is a module in the same directory (bootstrap)
    from .placeholders import PlaceholderEventBusImpl
    
    registry = ComponentRegistry() # Assumes ComponentRegistry can be init'd without args
    event_bus = PlaceholderEventBusImpl()
    
    return build_execution_context(
        component_id=component_id,
        run_id=run_id,
        registry=registry,
        event_bus=event_bus
    )