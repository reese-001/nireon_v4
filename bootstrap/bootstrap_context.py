from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Optional
from domain.context import NireonExecutionContext
from bootstrap.bootstrap_config import BootstrapConfig
from core.registry.component_registry import ComponentRegistry
from domain.ports.event_bus_port import EventBusPort 
if TYPE_CHECKING:
    from validators.interface_validator import InterfaceValidator


@dataclass
class BootstrapContext:
    config: BootstrapConfig
    run_id: str
    registry: ComponentRegistry
    registry_manager: Any
    health_reporter: Any
    signal_emitter: Any
    global_app_config: Dict[str, Any]
    validation_data_store: Any
    interface_validator: Optional[Any] = None

    @property
    def strict_mode(self) -> bool:
        return self.config.effective_strict_mode

    @property
    def event_bus(self) -> Optional[EventBusPort]: 
        try:
            return self.registry.get_service_instance(EventBusPort)
        except Exception:
            return getattr(self.signal_emitter, 'event_bus', None)

    def with_component_scope(self, component_id: str) -> NireonExecutionContext:
        return NireonExecutionContext(
            run_id=self.run_id,
            component_id=component_id,
            logger=self._component_logger(component_id),
            feature_flags=self.global_app_config.get('feature_flags', {}),
            component_registry=self.registry,
            event_bus=self.event_bus,
            config=self.global_app_config,
            interface_validator=self.interface_validator
        )

    def _component_logger(self, component_id: str):
        import logging
        if hasattr(self.health_reporter, 'logger') and self.health_reporter.logger:
            adapter = self.health_reporter.logger
            return logging.LoggerAdapter(adapter.logger, {'component': component_id})
        return logging.getLogger(component_id)