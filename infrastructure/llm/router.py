# nireon_v4\infrastructure\llm\router.py
from __future__ import annotations
import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Mapping, List
from core.base_component import NireonBaseComponent
from core.lifecycle import ComponentMetadata
from domain.ports.llm_port import LLMPort, LLMResponse
from domain.ports.event_bus_port import EventBusPort
from infrastructure.llm.parameter_service import ParameterService
from infrastructure.llm.factory import create_llm_instance
from domain.context import NireonExecutionContext
from domain.epistemic_stage import EpistemicStage
from core.results import ProcessResult
try:
    from infrastructure.llm.config_validator import validate_and_log_config
    from infrastructure.llm.circuit_breaker import CircuitBreaker, CircuitBreakerLLMAdapter
    from infrastructure.llm.metrics import get_metrics_collector, record_llm_call_metrics
    from infrastructure.llm.exceptions import LLMConfigurationError, LLMBackendNotAvailableError, LLMError, LLMProviderError
    ENHANCEMENTS_AVAILABLE = True
except ImportError as e:
    ENHANCEMENTS_AVAILABLE = False
    def validate_and_log_config(config, logger_instance):
        logger_instance.debug('Configuration validation not available - enhancements not installed')
        return True
    def get_metrics_collector():
        return None
    def record_llm_call_metrics(*args, **kwargs):
        pass
    class LLMConfigurationError(Exception):
        pass
    class LLMBackendNotAvailableError(Exception):
        pass
    class LLMError(Exception):
        pass
    class LLMProviderError(Exception):
        def __init__(self, message, cause=None, details=None):
            super().__init__(message)
            self.cause = cause
            self.details = details or {}
logger = logging.getLogger(__name__)
LLM_ROUTER_METADATA = ComponentMetadata(id='llm_router_main', name='LLMRouter', version='2.0.0', category='service', description='Enhanced LLM router with optional circuit breakers, metrics, and health monitoring.', requires_initialize=True, epistemic_tags=['router', 'llm_orchestrator', 'policy_enforcer'])
@dataclass(slots=True)
class _ModelConfig:
    name: str
    provider: str
    backend_class: str
    kwargs: Dict[str, Any] = field(default_factory=dict)
    circuit_breaker_config: Dict[str, Any] = field(default_factory=dict)
@dataclass
class BackendHealth:
    backend_key: str
    is_healthy: bool
    last_check: float
    consecutive_failures: int = 0
    last_error: Optional[str] = None
class LLMRouter(NireonBaseComponent, LLMPort):
    METADATA_DEFINITION = LLM_ROUTER_METADATA
    def __init__(self, config: dict, metadata_definition: Optional[ComponentMetadata]=None):
        if ENHANCEMENTS_AVAILABLE:
            validate_and_log_config(config, logger)
        if metadata_definition is None:
            component_id_from_config = config.get('id', 'llm_router_main')
            metadata_definition = ComponentMetadata(id=component_id_from_config, name='LLMRouter', version='2.0.0', category='service', description='Enhanced LLM router with optional circuit breakers, metrics, and health monitoring.', requires_initialize=True, epistemic_tags=[])
        super().__init__(config=config, metadata_definition=metadata_definition)
        logger.info(f"LLMRouter instantiated with component_id='{self.component_id}' (from metadata.id)")
        self._event_bus: Optional[EventBusPort] = None
        self._default_route: str = self.config.get('default', 'default')
        self._route_map: Dict[str, str] = self.config.get('routes', {})
        self._model_defs: Dict[str, _ModelConfig] = {}
        self._backends: Dict[str, LLMPort] = {}
        self._metrics_collector = get_metrics_collector() if ENHANCEMENTS_AVAILABLE else None
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}
        self._backend_health: Dict[str, BackendHealth] = {}
        self._enable_circuit_breakers = self.config.get('enable_circuit_breakers', ENHANCEMENTS_AVAILABLE)
        self._enable_health_checks = self.config.get('enable_health_checks', ENHANCEMENTS_AVAILABLE)
        self._enable_metrics = self.config.get('enable_metrics', ENHANCEMENTS_AVAILABLE)
        self._health_check_interval = self.config.get('health_check_interval', 300)
        self._default_circuit_breaker_config = self.config.get('circuit_breaker_defaults', {'failure_threshold': 5, 'recovery_timeout': 60, 'success_threshold': 3, 'timeout': 30.0})
        param_service_config = self.config.get('parameters', {})
        param_service_metadata = ComponentMetadata(id='llm_parameter_service_for_router', name='LLMParameterServiceForRouter', version='1.0.0', category='service_internal', description='Parameter resolver for the LLMRouter.', requires_initialize=False)
        self._param_service = ParameterService(config=param_service_config, metadata_definition=param_service_metadata)
        self._parse_model_configurations()
        enhancement_status = 'with enhancements' if ENHANCEMENTS_AVAILABLE else 'basic mode'
        logger.info(f"LLMRouter '{self.component_id}' instantiated {enhancement_status} with {len(self._model_defs)} model definitions. Default route: '{self._default_route}'")
    def _parse_model_configurations(self):
        for model_key, model_cfg_dict in self.config.get('models', {}).items():
            if not isinstance(model_cfg_dict, dict):
                logger.warning(f"Model configuration for '{model_key}' is not a dictionary. Skipping.")
                continue
            circuit_breaker_config = {}
            if self._enable_circuit_breakers and ENHANCEMENTS_AVAILABLE:
                circuit_breaker_config = {**self._default_circuit_breaker_config, **model_cfg_dict.get('circuit_breaker', {})}
            self._model_defs[model_key] = _ModelConfig(name=model_key, provider=model_cfg_dict.get('provider', 'unknown'), backend_class=model_cfg_dict.get('backend', ''), kwargs={k: v for k, v in model_cfg_dict.items() if k not in {'provider', 'backend', 'circuit_breaker'}}, circuit_breaker_config=circuit_breaker_config)
            if self._enable_health_checks:
                self._backend_health[model_key] = BackendHealth(backend_key=model_key, is_healthy=True, last_check=time.time())
    async def _initialize_impl(self, context: NireonExecutionContext) -> None:
        component_logger = context.logger or logger
        component_logger.info(f"LLMRouter '{self.component_id}' starting _initialize_impl")
        component_logger.info(f"LLMRouter '{self.component_id}' instance at {hex(id(self))}, _initialized_properly={self._initialized_properly}")
        if context.component_registry and context.event_bus:
            self._event_bus = context.event_bus
        elif context.component_registry:
            try:
                self._event_bus = context.component_registry.get_service_instance(EventBusPort)
            except Exception:
                component_logger.warning(f"LLMRouter '{self.component_id}': EventBusPort not found in registry during initialization.")
        try:
            default_backend_key = self._route_map.get(self._default_route, self._default_route)
            if default_backend_key in self._model_defs:
                component_logger.info(f"LLMRouter '{self.component_id}': Pre-warming default backend '{default_backend_key}'")
                await self._lazy_get_backend(default_backend_key)
                component_logger.info(f"LLMRouter '{self.component_id}': Default backend '{default_backend_key}' pre-warmed successfully.")
            else:
                component_logger.warning(f"LLMRouter '{self.component_id}': Default backend key '{default_backend_key}' not found in model definitions.")
        except Exception as e:
            component_logger.error(f"LLMRouter '{self.component_id}': Error pre-warming default backend: {e}", exc_info=True)
        if self._enable_health_checks and ENHANCEMENTS_AVAILABLE:
            asyncio.create_task(self._health_check_loop())
            component_logger.info(f"LLMRouter '{self.component_id}': Health monitoring enabled")
        component_logger.info(f"LLMRouter '{self.component_id}' _initialize_impl completed. Event bus: {('Configured' if self._event_bus else 'Not configured')}")
    async def _health_check_loop(self):
        while True:
            try:
                await asyncio.sleep(self._health_check_interval)
                await self._perform_health_checks()
            except Exception as e:
                logger.error(f'Health check loop error: {e}')
    async def _perform_health_checks(self):
        if not self._enable_health_checks:
            return
        current_time = time.time()
        for backend_key, backend in self._backends.items():
            health = self._backend_health.get(backend_key)
            if not health:
                continue
            try:
                test_context = NireonExecutionContext(run_id=f'health_check_{backend_key}')
                test_settings = {'max_tokens': 5, 'temperature': 0}
                response = await backend.call_llm_async('OK', stage=EpistemicStage.DEFAULT, role='health_check', context=test_context, settings=test_settings)
                if response.text and 'error' not in response:
                    health.is_healthy = True
                    health.consecutive_failures = 0
                    health.last_error = None
                else:
                    health.consecutive_failures += 1
                    health.is_healthy = health.consecutive_failures < 3
                    health.last_error = response.get('error', 'Health check failed')
            except Exception as e:
                health.consecutive_failures += 1
                health.is_healthy = health.consecutive_failures < 3
                health.last_error = str(e)
                logger.warning(f"Health check failed for backend '{backend_key}': {e}")
            finally:
                health.last_check = current_time
    async def _process_impl(self, data: Any, context: NireonExecutionContext) -> ProcessResult:
        logger.warning(f"LLMRouter '{self.component_id}' received generic process() call. This is not its primary mode of operation. Use call_llm_async() or call_llm_sync(). Data (first 100 chars): {str(data)[:100]}")
        return ProcessResult(success=False, component_id=self.component_id, message="LLMRouter's generic process() method is not intended for direct use. Use specific LLM call methods.", error_code='METHOD_NOT_SUITABLE')
    async def call_llm_async(self, prompt: str, *, stage: EpistemicStage, role: str, context: NireonExecutionContext, settings: Optional[Mapping[str, Any]]=None) -> LLMResponse:
        component_logger = context.logger or logger
        if not self.is_initialized:
            error_msg = f"LLMRouter '{self.component_id}' (instance at {hex(id(self))}) called before initialization! is_initialized={self.is_initialized}, _initialized_properly={self._initialized_properly}"
            component_logger.error(error_msg)
            return LLMResponse({'text': 'LLMRouter not initialized.', 'error': 'NotInitialized', 'details': error_msg})
        ce_event_id_from_context = getattr(context, 'ce_event_id', None) or (settings.get('ce_event_id') if settings else None)
        call_id = str(ce_event_id_from_context or uuid.uuid4())
        start_time = time.time()
        final_llm_params = dict(settings or {})
        route_override = final_llm_params.get('route')
        logical_route = route_override or self._default_route
        backend_key = self._route_map.get(logical_route, logical_route)
        component_logger.debug(f"LLMRouter '{self.component_id}': Routing call {call_id}. Logical route: '{logical_route}', Backend key: '{backend_key}'")
        if self._enable_health_checks:
            health = self._backend_health.get(backend_key)
            if health and (not health.is_healthy):
                component_logger.warning(f"Backend '{backend_key}' is marked as unhealthy, attempting anyway")
        try:
            backend = await self._lazy_get_backend(backend_key)
        except KeyError as e:
            component_logger.error(f"LLMRouter '{self.component_id}': Backend key '{backend_key}' (from route '{logical_route}') not found or failed to load. {e}")
            raise LLMBackendNotAvailableError(f"LLM backend '{backend_key}' not available.", details={'backend_key': backend_key}) from e
        if 'route' in final_llm_params:
            del final_llm_params['route']
        if 'ce_event_id' in final_llm_params:
            del final_llm_params['ce_event_id']
        response_text = 'Error: LLM call failed.'
        error_details = None
        llm_response_obj: LLMResponse
        try:
            llm_response_obj = await backend.call_llm_async(prompt, stage=stage, role=role, context=context, settings=final_llm_params)
            response_text = llm_response_obj.text
            if self._enable_metrics and ENHANCEMENTS_AVAILABLE:
                duration_ms = (time.time() - start_time) * 1000
                model_def = self._model_defs.get(backend_key)
                record_llm_call_metrics(call_id=call_id, model=model_def.name if model_def else backend_key, provider=model_def.provider if model_def else 'unknown', stage=stage.value if isinstance(stage, EpistemicStage) else str(stage), role=role, prompt_length=len(prompt), response_length=len(response_text), duration_ms=duration_ms, success=not llm_response_obj.get('error'))
            return llm_response_obj
        except Exception as e:
            component_logger.error(f"LLMRouter '{self.component_id}': Error calling backend '{backend_key}': {e}", exc_info=True)
            error_details = str(e)
            if self._enable_metrics and ENHANCEMENTS_AVAILABLE:
                duration_ms = (time.time() - start_time) * 1000
                model_def = self._model_defs.get(backend_key)
                record_llm_call_metrics(call_id=call_id, model=model_def.name if model_def else backend_key, provider=model_def.provider if model_def else 'unknown', stage=stage.value if isinstance(stage, EpistemicStage) else str(stage), role=role, prompt_length=len(prompt), response_length=0, duration_ms=duration_ms, success=False, error_type=type(e).__name__)
            if isinstance(e, LLMError):
                raise
            raise LLMProviderError(f"LLM backend '{backend_key}' failed.", cause=e, details={'backend_key': backend_key}) from e
    def call_llm_sync(self, prompt: str, *, stage: EpistemicStage, role: str, context: NireonExecutionContext, settings: Optional[Mapping[str, Any]]=None) -> LLMResponse:
        if not self.is_initialized:
            return LLMResponse({'text': 'LLMRouter not initialized.', 'error': 'NotInitialized'})
        ce_event_id_from_context = getattr(context, 'ce_event_id', None) or (settings.get('ce_event_id') if settings else None)
        call_id = str(ce_event_id_from_context or uuid.uuid4())
        final_llm_params = dict(settings or {})
        route_override = final_llm_params.get('route')
        logical_route = route_override or self._default_route
        backend_key = self._route_map.get(logical_route, logical_route)
        logger.debug(f"LLMRouter '{self.component_id}': Routing SYNC call {call_id}. Logical route: '{logical_route}', Backend key: '{backend_key}'")
        if self._enable_health_checks:
            health = self._backend_health.get(backend_key)
            if health and (not health.is_healthy):
                logger.warning(f"Backend '{backend_key}' is marked as unhealthy for sync call")
        try:
            if backend_key not in self._backends:
                model_def = self._model_defs.get(backend_key)
                if not model_def:
                    raise KeyError(f"LLM model definition '{backend_key}' not found.")
                if not model_def.backend_class:
                    raise ValueError(f"Backend class not defined for model '{backend_key}'.")
                logger.debug(f"LLMRouter '{self.component_id}': Creating SYNC backend '{backend_key}' via factory: {model_def.backend_class}")
                backend_config_for_factory = {'backend': model_def.backend_class, **model_def.kwargs}
                base_backend = create_llm_instance(model_def.name, backend_config_for_factory)
                if self._enable_circuit_breakers and ENHANCEMENTS_AVAILABLE:
                    circuit_breaker = CircuitBreaker(**model_def.circuit_breaker_config)
                    protected_backend = CircuitBreakerLLMAdapter(base_backend, {})
                    protected_backend.circuit_breaker = circuit_breaker
                    self._backends[backend_key] = protected_backend
                    self._circuit_breakers[backend_key] = circuit_breaker
                else:
                    self._backends[backend_key] = base_backend
            backend = self._backends[backend_key]
        except KeyError as e:
            logger.error(f"LLMRouter '{self.component_id}': SYNC Backend key '{backend_key}' not found. {e}")
            return LLMResponse({'text': f"Error: LLM backend '{backend_key}' not available.", 'error': str(e), 'call_id': call_id})
        if 'route' in final_llm_params:
            del final_llm_params['route']
        if 'ce_event_id' in final_llm_params:
            del final_llm_params['ce_event_id']
        if self._event_bus:
            self._event_bus.publish('LLM_SYNC_CALL_STARTED', {'router_id': self.component_id, 'route': logical_route, 'backend_key': backend_key, 'component_id': context.component_id, 'call_id': call_id})
        response_text = 'Error: LLM sync call failed.'
        error_details = None
        try:
            llm_response_obj = backend.call_llm_sync(prompt, stage=stage, role=role, context=context, settings=final_llm_params)
            response_text = llm_response_obj.text
            return llm_response_obj
        except Exception as e:
            logger.error(f"LLMRouter '{self.component_id}': Error calling SYNC backend '{backend_key}': {e}", exc_info=True)
            error_details = str(e)
            return LLMResponse({LLMResponse.TEXT_KEY: response_text, 'error': error_details, 'backend_key': backend_key, 'call_id': call_id})
        finally:
            if self._event_bus:
                self._event_bus.publish('LLM_SYNC_CALL_COMPLETED', {'router_id': self.component_id, 'route': logical_route, 'backend_key': backend_key, 'success': error_details is None, 'error': error_details, 'component_id': context.component_id, 'call_id': call_id})
    async def _lazy_get_backend(self, model_key: str) -> LLMPort:
        if model_key not in self._backends:
            model_def = self._model_defs.get(model_key)
            if not model_def:
                aliased_model_key = None
                for route_name, mapped_key in self._route_map.items():
                    if route_name == model_key and mapped_key in self._model_defs:
                        aliased_model_key = mapped_key
                        break
                if aliased_model_key:
                    model_def = self._model_defs[aliased_model_key]
                    logger.info(f"LLMRouter '{self.component_id}': Resolved alias '{model_key}' to model definition '{aliased_model_key}'.")
                else:
                    raise KeyError(f"LLM model definition or route alias '{model_key}' not found.")
            if not model_def.backend_class:
                raise ValueError(f"Backend class not defined for model '{model_def.name}'.")
            logger.debug(f"LLMRouter '{self.component_id}': Creating backend '{model_def.name}' via factory: {model_def.backend_class}")
            backend_config_for_factory = {'backend': model_def.backend_class, **model_def.kwargs}
            base_backend = create_llm_instance(model_def.name, backend_config_for_factory)
            if self._enable_circuit_breakers and ENHANCEMENTS_AVAILABLE:
                circuit_breaker = CircuitBreaker(**model_def.circuit_breaker_config)
                protected_backend = CircuitBreakerLLMAdapter(base_backend, {})
                protected_backend.circuit_breaker = circuit_breaker
                self._backends[model_key] = protected_backend
                self._circuit_breakers[model_key] = circuit_breaker
            else:
                self._backends[model_key] = base_backend
            if model_key != model_def.name and model_def.name not in self._backends:
                self._backends[model_def.name] = self._backends[model_key]
        return self._backends[model_key]
    def get_available_routes(self) -> Dict[str, str]:
        return dict(self._route_map)
    def get_defined_models(self) -> List[str]:
        return list(self._model_defs.keys())
    def get_backend_health(self) -> Dict[str, Dict[str, Any]]:
        if not self._enable_health_checks:
            return {}
        return {key: {'is_healthy': health.is_healthy, 'last_check': health.last_check, 'consecutive_failures': health.consecutive_failures, 'last_error': health.last_error} for key, health in self._backend_health.items()}
    def get_circuit_breaker_stats(self) -> Dict[str, Dict[str, Any]]:
        if not self._enable_circuit_breakers or not ENHANCEMENTS_AVAILABLE:
            return {}
        return {key: cb.get_stats() for key, cb in self._circuit_breakers.items()}
    def get_metrics_summary(self, time_window_seconds: Optional[int]=None) -> Dict[str, Any]:
        if not self._enable_metrics or not self._metrics_collector:
            return {'metrics_available': False}
        return self._metrics_collector.get_llm_summary(time_window_seconds)
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        stats = {'router_info': {'component_id': self.component_id, 'default_route': self._default_route, 'total_models': len(self._model_defs), 'active_backends': len(self._backends), 'enhancements_available': ENHANCEMENTS_AVAILABLE, 'features_enabled': {'circuit_breakers': self._enable_circuit_breakers, 'health_checks': self._enable_health_checks, 'metrics': self._enable_metrics}}, 'routes': self.get_available_routes(), 'models': self.get_defined_models()}
        if self._enable_health_checks:
            stats['backend_health'] = self.get_backend_health()
        if self._enable_circuit_breakers:
            stats['circuit_breaker_stats'] = self.get_circuit_breaker_stats()
        if self._enable_metrics:
            stats['metrics_summary'] = self.get_metrics_summary(3600)
        return stats
    def reset_backend(self, backend_key: str):
        if backend_key in self._backends:
            del self._backends[backend_key]
            logger.info(f"Backend '{backend_key}' removed from cache")
        if backend_key in self._circuit_breakers:
            self._circuit_breakers[backend_key].reset()
            logger.info(f"Circuit breaker for '{backend_key}' reset")
        if backend_key in self._backend_health:
            health = self._backend_health[backend_key]
            health.is_healthy = True
            health.consecutive_failures = 0
            health.last_error = None
            logger.info(f"Health status for '{backend_key}' reset")
    def force_backend_offline(self, backend_key: str):
        if backend_key in self._circuit_breakers:
            self._circuit_breakers[backend_key].force_open()
            logger.warning(f"Circuit breaker for '{backend_key}' forced open")
        if backend_key in self._backend_health:
            health = self._backend_health[backend_key]
            health.is_healthy = False
            health.last_error = 'Manually forced offline'
            logger.warning(f"Backend '{backend_key}' manually marked as unhealthy")