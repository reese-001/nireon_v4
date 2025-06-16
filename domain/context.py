from __future__ import annotations
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Dict, Optional, TYPE_CHECKING
from domain.ports.event_bus_port import EventBusPort
from domain.epistemic_stage import EpistemicStage 
if TYPE_CHECKING:
    from core.registry import ComponentRegistry
    from validators.interface_validator import InterfaceValidator

LoggerAdapter = Any
ConfigProvider = Any
StateManager = Any


class NireonExecutionContext:
    def __init__(self, *, 
                 run_id: str, 
                 step: int = 0, 
                 feature_flags: Optional[Dict[str, Any]] = None, 
                 component_registry: Optional['ComponentRegistry'] = None, 
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
                 interface_validator: Optional['InterfaceValidator'] = None) -> None:
        self.run_id: str = run_id
        self.step: int = step
        self.feature_flags: Dict[str, Any] = feature_flags or {}
        self.component_registry: Optional['ComponentRegistry'] = component_registry
        self.event_bus: Optional[EventBusPort] = event_bus
        self.logger: Optional[LoggerAdapter] = logger
        self.config_provider: Optional[ConfigProvider] = config_provider
        self.state_manager: Optional[StateManager] = state_manager
        self.config: Dict[str, Any] = config or {}
        self.session_id: Optional[str] = session_id
        self.component_id: Optional[str] = component_id
        self.timestamp: datetime = timestamp or datetime.now(timezone.utc)
        self.replay_mode: bool = replay_mode
        self.replay_seed: Optional[int] = replay_seed
        self.metadata: Dict[str, Any] = metadata or {}
        self.interface_validator: Optional['InterfaceValidator'] = interface_validator
        self._custom_data: Dict[str, Any] = {}

    def is_flag_enabled(self, flag_name: str, default: bool = False) -> bool:
        return bool(self.feature_flags.get(flag_name, default))

    def get_custom_data(self, key: str, default: Any = None) -> Any:
        return self._custom_data.get(key, default)

    def set_custom_data(self, key: str, value: Any) -> 'NireonExecutionContext':
        new_custom = deepcopy(self._custom_data)
        new_custom[key] = value
        return self._clone(_internal_custom_data_override=new_custom)

    def with_component_scope(self, component_id: str) -> 'NireonExecutionContext':
        return self._clone(component_id=component_id)

    def with_step(self, step: int) -> 'NireonExecutionContext':
        return self._clone(step=step)

    def with_metadata(self, **updates) -> 'NireonExecutionContext':
        new_meta = {**self.metadata, **updates}
        return self._clone(metadata=new_meta)

    def with_flags(self, **flag_updates) -> 'NireonExecutionContext':
        new_flags = {**self.feature_flags, **flag_updates}
        return self._clone(feature_flags=new_flags)

    def advance_step(self, new_step: int) -> 'NireonExecutionContext':
        return self._clone(step=new_step)

    def _clone(self, **overrides) -> 'NireonExecutionContext':
        params = {
            'run_id': self.run_id,
            'step': self.step,
            'feature_flags': deepcopy(self.feature_flags),
            'component_registry': self.component_registry,
            'event_bus': self.event_bus,
            'config': deepcopy(self.config),
            'session_id': self.session_id,
            'component_id': self.component_id,
            'timestamp': self.timestamp,
            'replay_mode': self.replay_mode,
            'replay_seed': self.replay_seed,
            'metadata': deepcopy(self.metadata),
            'logger': self.logger,
            'config_provider': self.config_provider,
            'state_manager': self.state_manager,
            'interface_validator': self.interface_validator
        }

        internal_custom_data_override = overrides.pop('_internal_custom_data_override', None)
        params.update(overrides)

        new_instance = NireonExecutionContext(**params)
        if internal_custom_data_override is not None:
            new_instance._custom_data = internal_custom_data_override
        else:
            new_instance._custom_data = deepcopy(self._custom_data)

        return new_instance

    def __repr__(self) -> str:
        return f'NireonExecutionContext(run_id={self.run_id!r}, step={self.step}, component_id={self.component_id!r}, session_id={self.session_id!r}, feature_flags={self.feature_flags})'


ExecutionContext = NireonExecutionContext