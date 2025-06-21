# nireon_v4/infrastructure/gateway/mechanism_gateway.py
# process [math_engine, principia_agent]

from __future__ import annotations
import logging
import time
import uuid
from typing import Any, Dict, Optional, TypedDict
from asyncio import TimeoutError as AsyncioTimeoutError

from application.services.budget_manager import BudgetExceededError, BudgetManagerPort
from application.services.frame_factory_service import FrameFactoryService, FrameNotFoundError
from bootstrap.bootstrap_helper.placeholders import PlaceholderEventBusImpl, PlaceholderLLMPortImpl
from core.base_component import NireonBaseComponent
from core.lifecycle import ComponentMetadata
from core.results import ComponentHealth, ProcessResult, SystemSignal, create_error_result, create_success_result
from domain.cognitive_events import CognitiveEvent, LLMRequestPayload
from domain.context import NireonExecutionContext
from domain.frames import Frame
from domain.ports.event_bus_port import EventBusPort
from domain.ports.llm_port import LLMPort, LLMResponse
from domain.ports.mechanism_gateway_port import MechanismGatewayPort
from infrastructure.llm.parameter_service import ParameterService
from infrastructure.llm.router import LLMRouter
from infrastructure.llm.router_backed_port import RouterBackedLLMPort
from .mechanism_gateway_metadata import MECHANISM_GATEWAY_METADATA

# NEW: Import the MathPort protocol
from domain.ports.math_port import MathPort

__all__ = ('MechanismGateway',)
logger = logging.getLogger(__name__)

class _EpisodeData(TypedDict, total=False):
    cognitive_event: CognitiveEvent
    response: Any
    duration_ms: float
    success: bool
    frame_name: str
    owning_agent_id: Optional[str]
    epistemic_intent: Optional[str]
    performance_metrics: Dict[str, Any]
    correlation_id: str


class MechanismGateway(NireonBaseComponent, MechanismGatewayPort):
    def __init__(
        self,
        llm_router: Optional[LLMPort],
        parameter_service: Optional[ParameterService],
        frame_factory: Optional[FrameFactoryService],
        budget_manager: Optional[BudgetManagerPort],
        event_bus: Optional[EventBusPort] = None,
        # NEW: Added math_port dependency
        math_port: Optional[MathPort] = None,
        config: Optional[Dict[str, Any]] = None,
        metadata_definition: Optional[ComponentMetadata] = None,
    ) -> None:
        super().__init__(config=config or {}, metadata_definition=metadata_definition or MECHANISM_GATEWAY_METADATA)
        self._llm_router: Optional[LLMPort] = llm_router
        self._parameter_service: Optional[ParameterService] = parameter_service
        self._frame_factory: Optional[FrameFactoryService] = frame_factory
        self._budget_manager: Optional[BudgetManagerPort] = budget_manager
        self._event_bus: Optional[EventBusPort] = event_bus
        # NEW: Store the math_port
        self._math_port: Optional[MathPort] = math_port

        self.metadata.add_runtime_state('total_requests', 0)
        self.metadata.add_runtime_state('llm_requests', 0)
        self.metadata.add_runtime_state('event_publishes', 0)
        # NEW: Add a metric for math computations
        self.metadata.add_runtime_state('math_computes', 0)
        self.metadata.add_runtime_state('budget_failures', 0)

    async def _initialize_impl(self, context: NireonExecutionContext) -> None:
        log = context.logger or logger
        log.info("MechanismGateway '%s' initializing", self.component_id)

        self.metadata.dependencies = {
            'LLMPort': '*',
            'parameter_service_global': '>=1.0.0',
            'frame_factory_service': '*',
            'BudgetManagerPort': '*',
            # NEW: Declare MathPort as a potential dependency
            'MathPort': '*'
        }

        self._llm_router = await self._resolve_llm_router(context, log)
        self._parameter_service = self._resolve_from_registry(ParameterService, self._parameter_service, context, log)
        self._frame_factory = self._resolve_from_registry(FrameFactoryService, self._frame_factory, context, log)
        self._budget_manager = self._resolve_from_registry(BudgetManagerPort, self._budget_manager, context, log)
        # NEW: Resolve the MathPort, marking it as optional
        self._math_port = self._resolve_from_registry(MathPort, self._math_port, context, log, is_optional=True)
        self._event_bus = self._resolve_event_bus(self._event_bus, context, log)

        self.metadata.add_runtime_state('initialization_complete', True)
        self.metadata.add_runtime_state('event_bus_available', self._event_bus is not None)
        # NEW: Record if the MathPort was successfully resolved
        self.metadata.add_runtime_state('math_port_available', self._math_port is not None)
        log.info("MechanismGateway '%s' initialization complete", self.component_id)

    async def _process_impl(self, data: Any, context: NireonExecutionContext) -> ProcessResult:
        if not isinstance(data, CognitiveEvent):
            return create_error_result(self.component_id, TypeError(f'Expected CognitiveEvent, got {type(data).__name__}'))
        
        ce: CognitiveEvent = data
        corr_id = ce.event_id or str(uuid.uuid4())
        start = time.time()
        
        try:
            result_payload = await self.process_cognitive_event(ce, context)
            result = create_success_result(component_id=self.component_id, message=f'Processed {ce.service_call_type}', output_data=result_payload)
            result.set_correlation_id(corr_id)
            result.add_metric('processing_time_ms', (time.time() - start) * 1000)
            result.add_metric('event_type', ce.service_call_type)
            self._inc_runtime('total_requests')
            return result
        except Exception as exc:
            err_res = create_error_result(self.component_id, exc)
            err_res.set_correlation_id(corr_id)
            err_sig = SystemSignal.create_error(self.component_id, f'Failed to process {ce.service_call_type}', exc)
            err_sig.set_correlation_id(corr_id)
            if self._event_bus:
                self._event_bus.publish(err_sig.to_event()['event_type'], err_sig.to_event())
            return err_res

    async def health_check(self, context: NireonExecutionContext) -> ComponentHealth:
        health = ComponentHealth(component_id=self.component_id, status='HEALTHY', message='MechanismGateway operational')
        health.add_check('llm_router', bool(self._llm_router), 'LLMRouter dependency present')
        health.add_check('parameter_service', bool(self._parameter_service), 'ParameterService dependency present')
        health.add_check('frame_factory', bool(self._frame_factory), 'FrameFactoryService dependency present')
        
        # NEW: Check optional MathPort
        if not self._math_port:
            health.add_check('math_port', False, 'MathPort not available (optional dependency)')
        else:
            health.add_check('math_port', True, 'MathPort dependency present')

        # ... (rest of health_check is the same)
        if not self._event_bus:
            health.add_check('event_bus', False, 'EventBus not available (optional)')

        if self._llm_router and hasattr(self._llm_router, 'health_check'):
            try:
                llm_ctx = context.with_component_scope(getattr(self._llm_router, 'component_id', 'llm_router'))
                llm_health = await self._llm_router.health_check(llm_ctx)
                health.add_dependency_health(llm_ctx.component_id, llm_health.status)
                if not llm_health.is_healthy():
                    health.add_check('llm_router_health', False, f'LLMRouter is {llm_health.status}')
            except Exception as exc:
                health.add_check('llm_router_health', False, f'LLMRouter health error: {exc}')
        
        health.metrics = {
            'total_requests': self.metadata.get_runtime_state('total_requests', 0),
            'llm_requests': self.metadata.get_runtime_state('llm_requests', 0),
            'event_publishes': self.metadata.get_runtime_state('event_publishes', 0),
            'math_computes': self.metadata.get_runtime_state('math_computes', 0), # NEW
            'budget_failures': self.metadata.get_runtime_state('budget_failures', 0)
        }
        health.status = health.calculate_overall_health()
        return health

    async def process_cognitive_event(self, ce: CognitiveEvent, context: NireonExecutionContext) -> Any:
        ce.gateway_received_ts = time.time()
        if ce.service_call_type == 'LLM_ASK':
            return await self.ask_llm(ce, context)
        if ce.service_call_type == 'EVENT_PUBLISH':
            return await self.publish_event(ce, context)
        # NEW: Add routing for MATH_COMPUTE
        if ce.service_call_type == 'MATH_COMPUTE':
            return await self._handle_math_compute(ce, context)

        raise ValueError(f'Unsupported service_call_type: {ce.service_call_type}')

    # NEW: Method to handle math computation events
    async def _handle_math_compute(self, ce: CognitiveEvent, context: NireonExecutionContext) -> Dict[str, Any]:
        log = context.logger or logger
        if not self._math_port:
            log.error(f'[{self.component_id}] Received MATH_COMPUTE event but no MathPort is configured.')
            return {'success': False, 'error': 'MathPort not available.'}
        
        log.info(f"[{self.component_id}] Processing MATH_COMPUTE event from agent '{ce.owning_agent_id}'.")
        self._inc_runtime('math_computes')

        try:
            if self._budget_manager and ce.frame_id:
                self._budget_manager.consume_resource_or_raise(ce.frame_id, 'math_calls', 1.0)
        except BudgetExceededError as exc:
            self._inc_runtime('budget_failures')
            log.warning(f"[{self.component_id}] Math compute budget exceeded for frame '{ce.frame_id}'.")
            return {'success': False, 'error': str(exc)}
        except KeyError:
            log.debug(f"[{self.component_id}] No 'math_calls' budget defined for frame '{ce.frame_id}'. Proceeding without check.")
            
        try:
            # The .compute() call is what will raise the TimeoutError from its internal asyncio.wait_for
            result = await self._math_port.compute(ce.payload)
            return result
        # --- START OF MODIFIED SECTION ---
        except AsyncioTimeoutError as e:
            # This is the crucial part. We catch the timeout from the adapter...
            error_msg = f"MathPort computation timed out within MechanismGateway: {e}"
            log.error(f'[{self.component_id}] {error_msg}')
            # ...and we return a clean failure dictionary, stopping exception propagation.
            return {'success': False, 'error': error_msg}
        except Exception as e:
            log.error(f'[{self.component_id}] Error during math computation via MathPort: {e}', exc_info=True)
            return {'success': False, 'error': str(e)}

    # ... (publish_event, ask_llm, and other methods remain the same) ...
    async def publish_event(self, ce: CognitiveEvent, context: NireonExecutionContext) -> ProcessResult:
        if ce.service_call_type != 'EVENT_PUBLISH':
            raise ValueError("CognitiveEvent must have service_call_type 'EVENT_PUBLISH'")
        if not isinstance(ce.payload, dict) or {'event_type', 'event_data'} - ce.payload.keys():
            raise ValueError("Payload must contain 'event_type' and 'event_data' keys")

        log = context.logger or logger
        start = time.time()
        
        frame = await self._frame_factory.get_frame_by_id(context, ce.frame_id)
        if not frame:
            raise FrameNotFoundError(f'Frame {ce.frame_id} not found')
            
        if not frame.is_active():
            log.warning("Publishing from non-active frame '%s' (status=%s)", frame.id, frame.status)

        bus = context.event_bus or self._event_bus
        if not bus:
            return create_error_result(self.component_id, RuntimeError('Event bus not available'))
            
        e_type, e_data = ce.payload['event_type'], ce.payload['event_data']
        
        try:
            bus.publish(e_type, e_data)
            duration_ms = (time.time() - start) * 1000
            self._inc_runtime('event_publishes')
            result = create_success_result(self.component_id, f"Published event '{e_type}'", {'published_event_type': e_type, 'publication_successful': True})
            result.add_metric('publish_duration_ms', duration_ms)
            result.add_metric('frame_id', frame.id)
            await self._publish_event_episode(ce, result, frame, context)
            return result
        except Exception as exc:
            err_res = create_error_result(self.component_id, exc)
            err_res.metadata['event_type'] = e_type
            await self._publish_event_episode(ce, err_res, frame, context)
            return err_res

    async def ask_llm(self, ce: CognitiveEvent, context: NireonExecutionContext) -> LLMResponse:
        log = context.logger or logger
        if not isinstance(ce.payload, LLMRequestPayload):
            raise ValueError('Payload must be LLMRequestPayload')

        frame = await self._frame_factory.get_frame_by_id(context, ce.frame_id)
        if not frame:
            return LLMResponse({
                'text': f"Error: Frame '{ce.frame_id}' not found.",
                'error': 'Frame not found',
                'error_type': 'FrameNotFoundError',
                'call_id': ce.event_id
            })

        if frame.resource_budget:
            self._budget_manager.initialize_frame_budget(frame.id, frame.resource_budget)
            
        if not frame.is_active():
            return await self._handle_inactive_frame(ce, frame, context)

        try:
            self._budget_manager.consume_resource_or_raise(frame.id, 'llm_calls', 1.0)
        except BudgetExceededError as exc:
            self._inc_runtime('budget_failures')
            return await self._handle_budget_exceeded(ce, frame, exc, context)
        
        final_settings = self._compose_llm_settings(frame, ce.payload)
        
        llm_ctx = context.with_component_scope(ce.owning_agent_id).with_metadata(
            frame_id=frame.id, frame_name=frame.name,
            epistemic_intent=ce.epistemic_intent, cognitive_event_id=ce.event_id
        )
        
        try:
            start = time.time()
            llm_resp = await self._llm_router.call_llm_async(
                prompt=ce.payload.prompt, stage=ce.payload.stage, role=ce.payload.role,
                context=llm_ctx, settings=final_settings
            )
            duration_ms = (time.time() - start) * 1000

            if llm_resp and llm_resp.get('error'):
                err_res = ProcessResult(success=False, component_id=self.component_id,
                                        message=str(llm_resp['error']),
                                        error_code=llm_resp.get('error_type', 'LLM_ERROR'),
                                        output_data=llm_resp)
                await self._publish_llm_episode(ce, err_res, frame, context)
            else:
                succ = create_success_result(self.component_id, 'LLM call successful', output_data=llm_resp)
                succ.add_metric('llm_call_duration_ms', duration_ms)
                succ.add_metric('prompt_length', len(ce.payload.prompt))
                if llm_resp and LLMResponse.TEXT_KEY in llm_resp:
                    succ.add_metric('response_length', len(llm_resp[LLMResponse.TEXT_KEY]))
                
                self._inc_runtime('llm_requests')
                await self._publish_llm_episode(ce, succ, frame, context)
            return llm_resp
        except Exception as exc:
            log.error('ask_llm: exception for CE %s: %s', ce.event_id, exc, exc_info=True)
            llm_resp = LLMResponse({
                'text': f'Error processing LLM request: {exc}',
                'error': str(exc), 'error_type': type(exc).__name__, 'call_id': ce.event_id
            })
            err_res = create_error_result(self.component_id, exc)
            err_res.output_data = llm_resp
            await self._publish_llm_episode(ce, err_res, frame, context)
            return llm_resp

    async def _publish_llm_episode(self, ce: CognitiveEvent, result: ProcessResult, frame: Frame, context: NireonExecutionContext) -> None:
        await self._publish_episode('GATEWAY_LLM_EPISODE_COMPLETED', ce, result, frame, context)

    async def _publish_event_episode(self, ce: CognitiveEvent, result: ProcessResult, frame: Frame, context: NireonExecutionContext) -> None:
        await self._publish_episode('GATEWAY_EVENT_EPISODE_COMPLETED', ce, result, frame, context)

    async def _publish_episode(self, event_type: str, ce: CognitiveEvent, result: ProcessResult, frame: Frame, context: NireonExecutionContext) -> None:
        episode: _EpisodeData = {
            'cognitive_event': ce, 'response': result.output_data,
            'duration_ms': result.get_metric('llm_call_duration_ms', 0) or result.get_metric('publish_duration_ms', 0),
            'success': result.success, 'frame_name': frame.name,
            'owning_agent_id': ce.owning_agent_id, 'epistemic_intent': ce.epistemic_intent,
            'performance_metrics': result.performance_metrics, 'correlation_id': result.correlation_id
        }
        bus = context.event_bus or self._event_bus
        if bus:
            try:
                bus.publish(event_type, episode)
            except Exception as exc:
                (context.logger or logger).error('Failed to publish episode: %s', exc)
        else:
            (context.logger or logger).info('%s (no bus): %s', event_type, episode)

    async def _resolve_llm_router(self, context: NireonExecutionContext, log: logging.Logger) -> LLMPort:
        if self._llm_router and not isinstance(self._llm_router, PlaceholderLLMPortImpl):
            return self._llm_router

        router: LLMPort = self._resolve_from_registry(LLMPort, self._llm_router, context, log)

        if isinstance(router, PlaceholderLLMPortImpl):
            main_router = context.component_registry.get('llm_router_main', None)
            if isinstance(main_router, LLMRouter):
                if getattr(main_router, 'is_initialized', False) is False and hasattr(main_router, 'initialize'):
                    await main_router.initialize(context.with_component_scope(main_router.component_id))
                router = RouterBackedLLMPort(main_router)
                log.info("Upgraded LLMPort to RouterBackedLLMPort via '%s'", main_router.component_id)
        return router

    # UPDATED: Added `is_optional` flag
    def _resolve_from_registry(
        self,
        cls: type,
        existing: Optional[Any],
        context: NireonExecutionContext,
        log: logging.Logger,
        is_optional: bool = False
    ):
        if existing and not isinstance(existing, (PlaceholderEventBusImpl, PlaceholderLLMPortImpl)):
            return existing
        try:
            svc = context.component_registry.get_service_instance(cls)
            log.info('%s resolved from registry', cls.__name__)
            return svc
        except Exception as exc:
            if is_optional:
                log.warning("Optional dependency %s not found in registry: %s", cls.__name__, exc)
                return None
            log.error('Failed to resolve %s from registry: %s', cls.__name__, exc, exc_info=True)
            raise RuntimeError(f'MechanismGateway missing {cls.__name__} and cannot resolve.') from exc

    def _resolve_event_bus(self, current: Optional[EventBusPort], context: NireonExecutionContext, log: logging.Logger) -> Optional[EventBusPort]:
        try:
            reg_bus = context.component_registry.get_service_instance(EventBusPort)
            if reg_bus and not isinstance(reg_bus, PlaceholderEventBusImpl):
                if current is None or isinstance(current, PlaceholderEventBusImpl):
                    log.info("EventBus upgraded to registry instance '%s'", getattr(reg_bus, 'component_id', type(reg_bus).__name__))
                return reg_bus
        except Exception:
            pass
        return current or context.event_bus

    def _compose_llm_settings(self, frame: Frame, payload: LLMRequestPayload) -> Dict[str, Any]:
        settings = self._parameter_service.get_parameters(stage=payload.stage, role=payload.role, ctx=None)
        if frame.llm_policy:
            settings.update(frame.llm_policy)
            if 'preferred_route' in frame.llm_policy and not payload.target_model_route:
                payload.target_model_route = frame.llm_policy['preferred_route']

        if payload.llm_settings:
            settings.update(payload.llm_settings)
            
        if payload.target_model_route:
            settings['route'] = payload.target_model_route
        
        return settings

    async def _handle_inactive_frame(self, ce: CognitiveEvent, frame: Frame, context: NireonExecutionContext) -> LLMResponse:
        err = f"Frame '{frame.id}' is not active (status: {frame.status})"
        resp = LLMResponse({
            'text': f'Error: {err}',
            'error': err,
            'error_type': 'FrameNotActiveError',
            'call_id': ce.event_id
        })
        result = ProcessResult(success=False, component_id=self.component_id, message=err,
                                error_code='FRAME_NOT_ACTIVE', output_data=resp)
        await self._publish_llm_episode(ce, result, frame, context)
        return resp

    async def _handle_budget_exceeded(self, ce: CognitiveEvent, frame: Frame, exc: BudgetExceededError, context: NireonExecutionContext) -> LLMResponse:
        msg = f'LLM call budget exceeded: {exc}'
        resp = LLMResponse({
            'text': f'Error: {msg}',
            'error': str(exc),
            'error_type': 'BudgetExceededError',
            'call_id': ce.event_id,
            'error_payload': {'code': 'BUDGET_EXCEEDED_HARD', 'message': str(exc)}
        })
        result = ProcessResult(success=False, component_id=self.component_id, message=msg,
                                error_code='BUDGET_EXCEEDED', output_data=resp)
        await self._publish_llm_episode(ce, result, frame, context)
        return resp
        
    def _inc_runtime(self, key: str) -> None:
        self.metadata.add_runtime_state(key, self.metadata.get_runtime_state(key, 0) + 1)