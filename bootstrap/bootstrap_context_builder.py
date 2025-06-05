"""
Bootstrap Context Builder - Consistent context creation for NIREON V4 bootstrap.

This module provides a builder pattern for creating BootstrapContext objects
with proper validation, consistent dependencies, and null-safety.
"""

from __future__ import annotations
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from core.registry import ComponentRegistry
from application.ports.event_bus_port import EventBusPort
from bootstrap.bootstrap_helper.placeholders import PlaceholderEventBusImpl
from bootstrap.health.reporter import HealthReporter
from bootstrap.registry.registry_manager import RegistryManager
from bootstrap.signals.bootstrap_signals import BootstrapSignalEmitter
from bootstrap.validation_data import BootstrapValidationData

if TYPE_CHECKING:
    # from bootstrap.orchestrator import BootstrapConfig, BootstrapContext
    from bootstrap.bootstrap_config import BootstrapConfig, BootstrapContext

logger = logging.getLogger(__name__)


class BootstrapContextBuilder:
    """
    Builder for creating properly configured BootstrapContext objects.
    
    Ensures consistent dependency resolution, proper null handling,
    and validation of required components.
    """
    
    def __init__(self, run_id: str):
        self.run_id = run_id
        self._config: Optional[BootstrapConfig] = None
        self._registry: Optional[ComponentRegistry] = None
        self._event_bus: Optional[EventBusPort] = None
        self._global_app_config: Dict[str, Any] = {}
        self._strict_mode: bool = True
        self._built = False
        
        logger.debug(f"BootstrapContextBuilder created for run_id: {run_id}")
    
    def with_config(self, config: BootstrapConfig) -> 'BootstrapContextBuilder':
        """Set the bootstrap configuration."""
        if self._built:
            raise RuntimeError("Cannot modify builder after build() is called")
        
        self._config = config
        self._strict_mode = config.effective_strict_mode
        return self
    
    def with_registry(self, registry: Optional[ComponentRegistry] = None) -> 'BootstrapContextBuilder':
        """
        Set the component registry.
        
        Args:
            registry: Existing registry to use, or None to create new one
        """
        if self._built:
            raise RuntimeError("Cannot modify builder after build() is called")
        
        self._registry = registry or ComponentRegistry()
        logger.debug(f"Registry set: {type(self._registry).__name__}")
        return self
    
    def with_event_bus(self, event_bus: Optional[EventBusPort] = None) -> 'BootstrapContextBuilder':
        """
        Set the event bus with null-safety.
        
        Args:
            event_bus: Existing event bus, or None to create placeholder
        """
        if self._built:
            raise RuntimeError("Cannot modify builder after build() is called")
        
        if event_bus is None:
            logger.warning("No event bus provided - creating placeholder for bootstrap safety")
            self._event_bus = PlaceholderEventBusImpl()
        else:
            self._event_bus = event_bus
        
        logger.debug(f"Event bus set: {type(self._event_bus).__name__}")
        return self
    
    def with_global_config(self, global_config: Dict[str, Any]) -> 'BootstrapContextBuilder':
        """Set the global application configuration."""
        if self._built:
            raise RuntimeError("Cannot modify builder after build() is called")
        
        self._global_app_config = global_config or {}
        
        # Extract strict mode from config if not explicitly set
        if 'bootstrap_strict_mode' in self._global_app_config:
            self._strict_mode = bool(self._global_app_config['bootstrap_strict_mode'])
        
        logger.debug(f"Global config set with {len(self._global_app_config)} keys, strict_mode: {self._strict_mode}")
        return self
    
    def build(self) -> BootstrapContext:
        """
        Build the BootstrapContext with validation.
        
        Returns:
            Fully configured BootstrapContext
            
        Raises:
            ValueError: If required components are missing
            RuntimeError: If builder already used
        """
        if self._built:
            raise RuntimeError("Builder can only be used once - create new builder for additional contexts")
        
        # Validate required components
        self._validate_required_components()
        
        # Create dependent components with consistent patterns
        registry_manager = RegistryManager(self._registry)
        health_reporter = HealthReporter(self._registry)
        signal_emitter = BootstrapSignalEmitter(self._event_bus, self.run_id)
        validation_data_store = BootstrapValidationData(self._global_app_config)
        
        # Import here to avoid circular dependency
        from bootstrap.bootstrap_context import BootstrapContext
        
        context = BootstrapContext(
            config=self._config,
            run_id=self.run_id,
            registry=self._registry,
            registry_manager=registry_manager,
            health_reporter=health_reporter,
            signal_emitter=signal_emitter,
            global_app_config=self._global_app_config,
            validation_data_store=validation_data_store
        )
        
        self._built = True
        logger.info(f"BootstrapContext built successfully for run_id: {self.run_id}")
        
        return context
    
    def _validate_required_components(self) -> None:
        """Validate that all required components are set."""
        errors = []
        
        if self._config is None:
            errors.append("BootstrapConfig is required")
        
        if self._registry is None:
            errors.append("ComponentRegistry is required")
        
        if self._event_bus is None:
            errors.append("EventBusPort is required")
        
        if not self.run_id or not self.run_id.strip():
            errors.append("run_id must be non-empty")
        
        if errors:
            error_msg = f"Cannot build BootstrapContext - missing required components: {', '.join(errors)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    @classmethod
    def create_default(cls, run_id: str, config: BootstrapConfig, global_config: Dict[str, Any]) -> BootstrapContext:
        """
        Convenience method to create a context with standard defaults.
        
        Args:
            run_id: Unique identifier for this bootstrap run
            config: Bootstrap configuration
            global_config: Global application configuration
            
        Returns:
            Configured BootstrapContext
        """
        return (cls(run_id)
                .with_config(config)
                .with_registry(config.existing_registry)
                .with_event_bus(config.existing_event_bus)
                .with_global_config(global_config)
                .build())


def create_bootstrap_context(
    run_id: str,
    config: BootstrapConfig,
    global_config: Dict[str, Any],
    registry: Optional[ComponentRegistry] = None,
    event_bus: Optional[EventBusPort] = None
) -> BootstrapContext:
    """
    Factory function for creating BootstrapContext with explicit parameters.
    
    Args:
        run_id: Unique identifier for this bootstrap run
        config: Bootstrap configuration
        global_config: Global application configuration
        registry: Optional existing registry (defaults to config.existing_registry)
        event_bus: Optional existing event bus (defaults to config.existing_event_bus)
        
    Returns:
        Configured BootstrapContext
    """
    return (BootstrapContextBuilder(run_id)
            .with_config(config)
            .with_registry(registry or config.existing_registry)
            .with_event_bus(event_bus or config.existing_event_bus)
            .with_global_config(global_config)
            .build())