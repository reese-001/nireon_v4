# bootstrap/bootstrap_context.py
# -------------------------------------------------
# Canonical bootstrap-time context for Nireon V4
# -------------------------------------------------
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from application.context import NireonExecutionContext
from bootstrap.bootstrap_config import BootstrapConfig
from core.registry.component_registry import ComponentRegistry


@dataclass
class BootstrapContext:
    """
    Immutable object that is threaded through every bootstrap phase.

    It also acts as a *factory* for per-component execution contexts
    that components use during their own initialisation.
    """
    config: BootstrapConfig
    run_id: str
    registry: ComponentRegistry
    registry_manager: Any            # RegistryManager
    health_reporter: Any             # HealthReporter (has .add_phase_result, etc.)
    signal_emitter: Any              # BootstrapSignalEmitter
    global_app_config: Dict[str, Any]
    validation_data_store: Any       # BootstrapValidationData

    # ------------------------------------------------------------------ #
    # Convenience properties
    # ------------------------------------------------------------------ #
    @property
    def strict_mode(self) -> bool:
        """Effective strict-mode flag resolved from CLI + config."""
        return self.config.effective_strict_mode

    @property
    def event_bus(self) -> Optional[Any]:
        """Expose the event bus if the signal emitter is wiring one in."""
        return getattr(self.signal_emitter, "event_bus", None)

    # ------------------------------------------------------------------ #
    # Context helpers
    # ------------------------------------------------------------------ #
    def with_component_scope(self, component_id: str) -> NireonExecutionContext:
        """
        Build a *runtime* execution context scoped to a single component.

        That execution context provides:
          • .logger           – standard `logging.LoggerAdapter`
          • .with_metadata()  – for extra diagnostic tags
          • feature-flag helpers, registry access, …
        It is exactly what `build_component_init_context()` and your
        component `initialize()` implementations expect.
        """
        # NB: We deliberately *do not* carry over bootstrap-only objects
        # such as `health_reporter`; components shouldn’t poke at those.
        return NireonExecutionContext(
            run_id=self.run_id,
            component_id=component_id,
            logger=self._component_logger(component_id),
            feature_flags=self.global_app_config.get("feature_flags", {}),
            component_registry=self.registry,
            event_bus=self.event_bus,
            config=self.global_app_config,
        )

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _component_logger(self, component_id: str):
        """
        Derive a logger for an individual component.

        If the health-reporter already created a logger adapter we reuse it;
        otherwise fall back to `logging.getLogger(component_id)`.
        """
        import logging

        if hasattr(self.health_reporter, "logger") and self.health_reporter.logger:
            # Clone the adapter but change the component ID to keep
            # per-component prefixes consistent.
            adapter = self.health_reporter.logger
            return logging.LoggerAdapter(adapter.logger, {"component": component_id})

        return logging.getLogger(component_id)
