# nireon_v4\bootstrap\bootstrap_helper\context_helper.py
from __future__ import annotations
import logging
from datetime import datetime, timezone
from functools import cached_property
from typing import Any, Dict, Final, Optional
from core.registry import ComponentRegistry
from domain.context import NireonExecutionContext
from domain.ports.event_bus_port import EventBusPort
__all__: Final = ['SimpleConfigProvider', 'build_execution_context', 'build_bootstrap_context', 'build_component_init_context', 'build_validation_context', 'ContextBuilder', 'create_context_builder', 'create_test_context', 'create_minimal_context']
logger = logging.getLogger(__name__)
DEFAULT_REPLAY_SEED: Final[int] = 12345
def _utc_now() -> datetime:
    return datetime.now(timezone.utc)
def _component_logger(component_id: str, event_bus: EventBusPort) -> logging.Logger:
    if callable(getattr(event_bus, 'get_logger', None)):
        try:
            return event_bus.get_logger(component_id)
        except Exception as exc:
            logger.warning("Event‑bus logger retrieval failed for '%s': %s", component_id, exc)
    return logging.getLogger(f'nireon.{component_id}')
def _ensure(assertion: bool, message: str) -> None:
    if not assertion:
        raise ValueError(message)
class SimpleConfigProvider:
    def __init__(self, flags: Optional[Dict[str, Any]]=None) -> None:
        self._flags: Dict[str, Any] = dict(flags or {})
        logger.debug('SimpleConfigProvider initialised with %d flag(s)', len(self._flags))
    def get_config(self, component_id: str, key: str, default: Any=None) -> Any:
        scoped_key = f'{component_id}.{key}'
        return self._flags[scoped_key] if scoped_key in self._flags else self._flags.get(key, default)
    def has_config(self, key: str) -> bool:
        return key in self._flags
    def get_all_config(self) -> Dict[str, Any]:
        return dict(self._flags)
    def __contains__(self, key: str) -> bool:
        return self.has_config(key)
    def __getitem__(self, key: str) -> Any:
        return self._flags[key]
def build_execution_context(*, component_id: str, run_id: str, registry: ComponentRegistry, event_bus: EventBusPort, feature_flags: Optional[Dict[str, Any]]=None, replay: bool=False, step: int=0, session_id: Optional[str]=None, metadata: Optional[Dict[str, Any]]=None, config: Optional[Dict[str, Any]]=None) -> NireonExecutionContext:
    logger.debug("Building execution context for component '%s' (run_id=%s, step=%s)", component_id, run_id, step)
    context = NireonExecutionContext(run_id=run_id, step=step, feature_flags=feature_flags or {}, component_registry=registry, event_bus=event_bus, config=config or {}, session_id=session_id, component_id=component_id, timestamp=_utc_now(), replay_mode=replay, replay_seed=DEFAULT_REPLAY_SEED if replay else None, metadata=metadata or {}, logger=_component_logger(component_id, event_bus), config_provider=SimpleConfigProvider(feature_flags or {}), state_manager=None)
    logger.debug("Execution context ready for '%s' with %d feature flag(s)", component_id, len(feature_flags or {}))
    return context
def build_bootstrap_context(*, run_id: str, registry: ComponentRegistry, event_bus: EventBusPort, global_config: Dict[str, Any], step: int=0) -> NireonExecutionContext:
    return build_execution_context(component_id='bootstrap_system', run_id=run_id, registry=registry, event_bus=event_bus, feature_flags=global_config.get('feature_flags', {}), replay=False, step=step, session_id=f'bootstrap_{run_id}', metadata={'bootstrap_version': '4.0', 'bootstrap_mode': 'standard', 'strict_mode': global_config.get('bootstrap_strict_mode', True)}, config=global_config)
def build_component_init_context(*, component_id: str, base_context: NireonExecutionContext, component_config: Optional[Dict[str, Any]]=None) -> NireonExecutionContext:
    return base_context.with_component_scope(component_id).with_metadata(initialization_step=True, component_config_keys=list((component_config or {}).keys()))
def build_validation_context(*, component_id: str, base_context: NireonExecutionContext, validation_data: Optional[Dict[str, Any]]=None) -> NireonExecutionContext:
    return base_context.with_component_scope(component_id).with_metadata(validation_step=True, validation_data=validation_data or {})
class ContextBuilder:
    # Added '__dict__' to allow @cached_property to work
    __slots__ = ('_component_id', '_run_id', '_registry', '_event_bus', '_feature_flags', '_replay', '_step', '_session_id', '_metadata', '_config', '__dict__')
    def __init__(self, component_id: str, run_id: str) -> None:
        self._component_id = component_id
        self._run_id = run_id
        self._registry: Optional[ComponentRegistry] = None
        self._event_bus: Optional[EventBusPort] = None
        self._feature_flags: Dict[str, Any] = {}
        self._replay = False
        self._step = 0
        self._session_id: Optional[str] = None
        self._metadata: Dict[str, Any] = {}
        self._config: Dict[str, Any] = {}
    def with_registry(self, registry: ComponentRegistry) -> 'ContextBuilder':
        self._registry = registry
        return self
    def with_event_bus(self, event_bus: EventBusPort) -> 'ContextBuilder':
        self._event_bus = event_bus
        return self
    def with_feature_flags(self, flags: Dict[str, Any]) -> 'ContextBuilder':
        self._feature_flags = flags
        return self
    def with_replay(self, replay: bool=True) -> 'ContextBuilder':
        self._replay = replay
        return self
    def with_step(self, step: int) -> 'ContextBuilder':
        self._step = step
        return self
    def with_session_id(self, session_id: str) -> 'ContextBuilder':
        self._session_id = session_id
        return self
    def with_metadata(self, **metadata: Any) -> 'ContextBuilder':
        self._metadata.update(metadata)
        return self
    def with_config(self, config: Dict[str, Any]) -> 'ContextBuilder':
        self._config = config
        return self
    @cached_property
    def _validated_registry(self) -> ComponentRegistry:
        _ensure(self._registry is not None, 'ComponentRegistry is required.')
        return self._registry
    @cached_property
    def _validated_event_bus(self) -> EventBusPort:
        _ensure(self._event_bus is not None, 'EventBusPort is required.')
        return self._event_bus
    def build(self) -> NireonExecutionContext:
        return build_execution_context(component_id=self._component_id, run_id=self._run_id, registry=self._validated_registry, event_bus=self._validated_event_bus, feature_flags=self._feature_flags, replay=self._replay, step=self._step, session_id=self._session_id, metadata=self._metadata, config=self._config)
def create_context_builder(component_id: str, run_id: str) -> ContextBuilder:
    return ContextBuilder(component_id, run_id)
def create_test_context(*, component_id: str='test_component', run_id: str='test_run', registry: Optional[ComponentRegistry]=None, event_bus: Optional[EventBusPort]=None) -> NireonExecutionContext:
    if registry is None:
        registry = ComponentRegistry()
    if event_bus is None:
        from .placeholders import PlaceholderEventBusImpl
        event_bus = PlaceholderEventBusImpl()
    return build_execution_context(component_id=component_id, run_id=run_id, registry=registry, event_bus=event_bus, feature_flags={'test_mode': True}, metadata={'test_context': True})
def create_minimal_context(component_id: str, run_id: str) -> NireonExecutionContext:
    from .placeholders import PlaceholderEventBusImpl
    return create_test_context(component_id=component_id, run_id=run_id, registry=ComponentRegistry(), event_bus=PlaceholderEventBusImpl())