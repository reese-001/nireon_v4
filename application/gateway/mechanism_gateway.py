from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional
import uuid

from core.base_component import NireonBaseComponent
from core.lifecycle import ComponentMetadata
from core.results import (
    ComponentHealth, ProcessResult, SystemSignal, SignalType,
    create_success_result, create_error_result
)
from domain.cognitive_events import CognitiveEvent, LLMRequestPayload
from domain.ports.llm_port import LLMResponse, LLMPort
from domain.ports.event_bus_port import EventBusPort
from domain.ports.mechanism_gateway_port import MechanismGatewayPort
from domain.context import NireonExecutionContext
from domain.frames import Frame

from application.services.frame_factory_service import FrameFactoryService, FrameNotFoundError
from application.services.budget_manager import BudgetManagerPort, BudgetExceededError  # New Import
from infrastructure.llm.parameter_service import ParameterService
from bootstrap.bootstrap_helper.placeholders import PlaceholderEventBusImpl, PlaceholderLLMPortImpl
from infrastructure.llm.router_backed_port import RouterBackedLLMPort
from infrastructure.llm.router import LLMRouter
from .mechanism_gateway_metadata import MECHANISM_GATEWAY_METADATA


class MechanismGateway(NireonBaseComponent, MechanismGatewayPort):
    """Enhanced MechanismGateway leveraging advanced result tracking."""

    def __init__(self,
                 llm_router: LLMPort,
                 parameter_service: ParameterService,
                 frame_factory: FrameFactoryService,
                 budget_manager: BudgetManagerPort,  # New dependency
                 event_bus: Optional[EventBusPort] = None,
                 config: Optional[Dict[str, Any]] = None,
                 metadata_definition: Optional[ComponentMetadata] = None):
        """Initialize the gateway with required dependencies."""
        # Use provided metadata or default to MECHANISM_GATEWAY_METADATA
        super().__init__(
            config=config or {}, 
            metadata_definition=metadata_definition or MECHANISM_GATEWAY_METADATA
        )
        
        self._llm_router = llm_router
        self._parameter_service = parameter_service
        self._frame_factory = frame_factory
        self._budget_manager = budget_manager  # Store it
        self._event_bus = event_bus  # Initial event_bus, could be a placeholder
        
        # Track gateway-specific metadata
        self.metadata.add_runtime_state("total_requests", 0)
        self.metadata.add_runtime_state("llm_requests", 0)
        self.metadata.add_runtime_state("event_publishes", 0)
        self.metadata.add_runtime_state("budget_failures", 0)  # New

    async def _initialize_impl(self, context: NireonExecutionContext) -> None:
        """Initialize with dependency validation and health tracking."""
        component_logger = context.logger or logging.getLogger(__name__)
        component_logger.info(f"MechanismGateway '{self.component_id}' initializing")
        
        # Track dependencies in metadata - use actual component IDs
        self.metadata.dependencies = {
            "LLMPort": "*",
            "parameter_service_global": ">=1.0.0",
            "frame_factory_service": "*",
            "BudgetManagerPort": "*"  # New dependency
        }
        
        # Resolve dependencies robustly if not provided or if they need updating
        # LLMRouter
        if not self._llm_router:
            try:
                self._llm_router = context.component_registry.get_service_instance(LLMPort)
                component_logger.info(f"LLMPort resolved from registry for '{self.component_id}'.")
            except Exception as e:
                component_logger.error(f"Failed to resolve LLMPort for '{self.component_id}' from registry: {e}", exc_info=True)
                raise RuntimeError(f"MechanismGateway '{self.component_id}' missing LLMPort and cannot resolve from registry.")
        
        # ---- Swap in router-backed LLMPort if current is placeholder ----
        if isinstance(self._llm_router, PlaceholderLLMPortImpl):
            router_instance = context.component_registry.get("llm_router_main")  # id from manifest
            if router_instance and isinstance(router_instance, LLMRouter):
                self._llm_router = RouterBackedLLMPort(router_instance)  # thin adapter
                component_logger.info(f"↪  Upgraded LLMPort for '{self.component_id}' to RouterBackedLLMPort via '{router_instance.component_id}'")
        
        # ParameterService
        if not self._parameter_service:
            try:
                self._parameter_service = context.component_registry.get_service_instance(ParameterService)
                component_logger.info(f"ParameterService resolved from registry for '{self.component_id}'.")
            except Exception as e:
                component_logger.error(f"Failed to resolve ParameterService for '{self.component_id}' from registry: {e}", exc_info=True)
                raise RuntimeError(f"MechanismGateway '{self.component_id}' missing ParameterService and cannot resolve from registry.")

        # FrameFactoryService
        if not self._frame_factory:
            try:
                self._frame_factory = context.component_registry.get_service_instance(FrameFactoryService)
                component_logger.info(f"FrameFactoryService resolved from registry for '{self.component_id}'.")
            except Exception as e:
                component_logger.error(f"Failed to resolve FrameFactoryService for '{self.component_id}' from registry: {e}", exc_info=True)
                raise RuntimeError(f"MechanismGateway '{self.component_id}' missing FrameFactoryService and cannot resolve from registry.")

        # BudgetManagerPort
        if not self._budget_manager:
            try:
                self._budget_manager = context.component_registry.get_service_instance(BudgetManagerPort)
                component_logger.info(f"BudgetManagerPort resolved from registry for '{self.component_id}'.")
            except Exception as e:
                component_logger.error(f"Failed to resolve BudgetManagerPort for '{self.component_id}' from registry: {e}", exc_info=True)
                raise RuntimeError(f"MechanismGateway '{self.component_id}' missing BudgetManager and cannot resolve from registry.")

        # EventBusPort - Enhanced resolution
        # Try to get the most up-to-date EventBus from the registry.
        # This is important if the gateway was instantiated with a placeholder event bus.
        current_registry_event_bus = None
        if context.component_registry:
            try:
                current_registry_event_bus = context.component_registry.get_service_instance(EventBusPort)
            except Exception:
                component_logger.debug(f"EventBusPort not found in registry during '{self.component_id}' initialization. Will use existing or context's.")

        # ---- Swap in real EventBus if we were bootstrapped with a placeholder ----
        if current_registry_event_bus:
            if self._event_bus is None or isinstance(self._event_bus, PlaceholderEventBusImpl):
                self._event_bus = current_registry_event_bus
                if isinstance(self._event_bus, PlaceholderEventBusImpl):
                    component_logger.info(f"EventBus for '{self.component_id}' resolved from component registry, but it's still a placeholder.")
                else:
                    component_logger.info(f"↪  Upgraded EventBus for '{self.component_id}' to real EventBus '{getattr(self._event_bus, 'component_id', type(self._event_bus).__name__)}' from component registry.")
            elif self._event_bus is not current_registry_event_bus:
                component_logger.info(f"Updating EventBus for '{self.component_id}': Replacing current EventBus ({type(self._event_bus).__name__}) with instance from component registry ({type(current_registry_event_bus).__name__}).")
                self._event_bus = current_registry_event_bus
        elif (self._event_bus is None or isinstance(self._event_bus, PlaceholderEventBusImpl)) and context.event_bus:
            if not isinstance(context.event_bus, PlaceholderEventBusImpl):
                self._event_bus = context.event_bus
                component_logger.info(f"↪  Upgraded EventBus for '{self.component_id}' to real EventBus from initialization context.")
            elif self._event_bus is None:  # Only if it was truly None, not if it was already a placeholder
                self._event_bus = context.event_bus
                component_logger.info(f"EventBus for '{self.component_id}' taken from initialization context (still placeholder type).")
        
        if not self._event_bus:
            component_logger.warning(f"EventBus still not available for MechanismGateway '{self.component_id}'. Episode publishing will be limited.")
        else:
            component_logger.info(f"MechanismGateway '{self.component_id}' will use EventBus of type: {type(self._event_bus).__name__}")
        
        # Update runtime state
        self.metadata.add_runtime_state("initialization_complete", True)
        self.metadata.add_runtime_state("event_bus_available", self._event_bus is not None)
        
        component_logger.info(f"MechanismGateway '{self.component_id}' initialization complete")

    async def _process_impl(self, data: Any, context: NireonExecutionContext) -> ProcessResult:
        if not isinstance(data, CognitiveEvent):
            return create_error_result(self.component_id, TypeError(f'Expected CognitiveEvent but got {type(data).__name__}'))
        
        ce: CognitiveEvent = data
        correlation_id = ce.event_id or str(uuid.uuid4())
        start_time = time.time()
        
        try:
            result_data = await self.process_cognitive_event(ce, context)
            result = create_success_result(
                component_id=self.component_id,
                message=f'Successfully processed {ce.service_call_type} event',
                output_data=result_data
            )
            duration_ms = (time.time() - start_time) * 1000
            result.add_metric('processing_time_ms', duration_ms)
            result.add_metric('event_type', ce.service_call_type)
            result.set_correlation_id(correlation_id)
            
            total = self.metadata.get_runtime_state('total_requests', 0) + 1
            self.metadata.add_runtime_state('total_requests', total)
            return result
        except Exception as e:
            error_result = create_error_result(component_id=self.component_id, error=e)
            error_result.set_correlation_id(correlation_id)
            
            error_signal = SystemSignal.create_error(
                component_id=self.component_id,
                message=f'Failed to process {ce.service_call_type} event',
                error=e
            )
            error_signal.set_correlation_id(correlation_id)
            
            if self._event_bus:
                # Remove 'await' - EventBusPort.publish is synchronous
                self._event_bus.publish(error_signal.to_event()['event_type'], error_signal.to_event())
            
            return error_result

    async def health_check(self, context: NireonExecutionContext) -> ComponentHealth:
        """Enhanced health check with dependency tracking."""
        health = ComponentHealth(
            component_id=self.component_id,
            status="HEALTHY",
            message="MechanismGateway operational"
        )
        
        # Check core dependencies
        health.add_check("llm_router", self._llm_router is not None, 
                        "LLMRouter dependency present")
        health.add_check("parameter_service", self._parameter_service is not None,
                        "ParameterService dependency present")
        health.add_check("frame_factory", self._frame_factory is not None,
                        "FrameFactoryService dependency present")
        
        # Check optional dependencies
        if not self._event_bus:
            health.add_check("event_bus", False, "EventBus not available (optional)")
        
        # Check dependency health if they support it
        if self._llm_router and hasattr(self._llm_router, 'health_check'):
            try:
                llm_router_id = getattr(self._llm_router, 'component_id', 'llm_router')
                dep_context = context.with_component_scope(llm_router_id)
                llm_health = await self._llm_router.health_check(dep_context)
                health.add_dependency_health(llm_router_id, llm_health.status)
                
                if not llm_health.is_healthy():
                    health.add_check("llm_router_health", False, 
                                   f"LLMRouter is {llm_health.status}")
            except Exception as e:
                health.add_check("llm_router_health", False, 
                               f"Error checking LLMRouter health: {e}")
        
        # Add runtime metrics to health
        health.metrics = {
            "total_requests": self.metadata.get_runtime_state("total_requests", 0),
            "llm_requests": self.metadata.get_runtime_state("llm_requests", 0),
            "event_publishes": self.metadata.get_runtime_state("event_publishes", 0),
            "budget_failures": self.metadata.get_runtime_state("budget_failures", 0)  # New metric
        }
        
        # Calculate overall health
        overall_status = health.calculate_overall_health()
        health.status = overall_status
        
        return health

    async def process_cognitive_event(self, ce: CognitiveEvent, context: NireonExecutionContext) -> Any:
        """Process a cognitive event by routing it to the appropriate service."""
        component_logger = context.logger or logging.getLogger(__name__)
        ce.gateway_received_ts = time.time()

        if ce.service_call_type == 'LLM_ASK':
            return await self.ask_llm(ce, context)
        elif ce.service_call_type == 'EVENT_PUBLISH':
            return await self.publish_event(ce, context)
        else:
            component_logger.error(f'Unsupported service_call_type: {ce.service_call_type} in CE {ce.event_id}')
            raise ValueError(f'Unsupported service_call_type: {ce.service_call_type}')

    async def publish_event(self, ce: CognitiveEvent, context: NireonExecutionContext) -> ProcessResult:
        component_logger = context.logger or logging.getLogger(__name__)
        if ce.service_call_type != 'EVENT_PUBLISH':
            raise ValueError("CognitiveEvent must have service_call_type 'EVENT_PUBLISH'")
        if not isinstance(ce.payload, dict) or 'event_type' not in ce.payload or 'event_data' not in ce.payload:
            raise ValueError("Payload must be a dict with 'event_type' and 'event_data' keys")
        
        start_time = time.time()
        frame = await self._frame_factory.get_frame_by_id(context, ce.frame_id)
        if not frame:
            raise FrameNotFoundError(f'Frame {ce.frame_id} not found for CE {ce.event_id}')
        
        if not frame.is_active():
            component_logger.warning(f"Publishing event from non-active Frame '{frame.id}' (status: {frame.status}) for CE {ce.event_id}")
        
        event_type_to_publish = ce.payload['event_type']
        event_data_to_publish = ce.payload['event_data']
        event_bus = context.event_bus or self._event_bus
        
        if event_bus:
            try:
                # Remove 'await' - EventBusPort.publish is synchronous
                event_bus.publish(event_type_to_publish, event_data_to_publish)
                result = create_success_result(
                    component_id=self.component_id,
                    message=f"Published event '{event_type_to_publish}'",
                    output_data={'published_event_type': event_type_to_publish, 'publication_successful': True}
                )
                duration_ms = (time.time() - start_time) * 1000
                result.add_metric('publish_duration_ms', duration_ms)
                result.add_metric('frame_id', frame.id)
                publishes = self.metadata.get_runtime_state('event_publishes', 0) + 1
                self.metadata.add_runtime_state('event_publishes', publishes)
                await self._publish_event_episode(ce, result, frame, context)
                return result
            except Exception as e:
                error_result = create_error_result(component_id=self.component_id, error=e)
                error_result.metadata['event_type'] = event_type_to_publish
                await self._publish_event_episode(ce, error_result, frame, context)
                return error_result
        else:
            return create_error_result(component_id=self.component_id, error=RuntimeError('Event bus not available'))
           

    async def ask_llm(self, ce: CognitiveEvent, context: NireonExecutionContext) -> LLMResponse:
        """Enhanced LLM routing with performance tracking."""
        component_logger = context.logger or logging.getLogger(__name__)
        
        if ce.service_call_type != 'LLM_ASK':
            raise ValueError("CognitiveEvent must have service_call_type 'LLM_ASK'")
        
        if not isinstance(ce.payload, LLMRequestPayload):
            raise ValueError("Payload must be LLMRequestPayload")

        llm_payload = ce.payload
        start_time = time.time()

        # Get the frame to check policies
        frame = await self._frame_factory.get_frame_by_id(context, ce.frame_id)
        if not frame:
            # This case should ideally not happen if Explorer creates the frame first
            # and uses its ID in the CE. But if it could, handle it.
            component_logger.error(f"Frame {ce.frame_id} not found for CE {ce.event_id}")
            return LLMResponse({
                LLMResponse.TEXT_KEY: f"Error: Frame '{ce.frame_id}' not found.",
                'error': f'Frame not found: {ce.frame_id}',
                'error_type': 'FrameNotFoundError',
                'call_id': ce.event_id
            })

        # Initialize frame budget in BudgetManager if it's the first time or budget changed
        if frame.resource_budget:
            self._budget_manager.initialize_frame_budget(frame.id, frame.resource_budget)

        # Check frame status
        if not frame.is_active():
            component_logger.warning(f"LLM_ASK from non-active Frame '{frame.id}' "
                                    f"(status: {frame.status}) for CE {ce.event_id}")
            
            # Create error response
            err_resp = LLMResponse({
                LLMResponse.TEXT_KEY: f"Error: Frame '{frame.id}' is not active (status: {frame.status})",
                'error': f'Frame not active: {frame.status}',
                'error_type': 'FrameNotActiveError',
                'call_id': ce.event_id
            })
            
            # Create error result for tracking
            error_result = ProcessResult(
                success=False,
                component_id=self.component_id,
                message=f"Frame not active: {frame.status}",
                error_code="FRAME_NOT_ACTIVE",
                output_data=err_resp
            )
            
            await self._publish_llm_episode(ce, error_result, frame, context)
            return err_resp

        # Budget Check
        try:
            # Assuming 1 LLM call consumes 1 unit of 'llm_calls' budget
            self._budget_manager.consume_resource_or_raise(frame.id, "llm_calls", 1.0)
            component_logger.info(f"Budget consumed for LLM call in frame '{frame.id}'.")
        except BudgetExceededError as budget_error:
            component_logger.warning(f"LLM_ASK budget exceeded for Frame '{frame.id}': {budget_error}")
            self.metadata.add_runtime_state('budget_failures', self.metadata.get_runtime_state('budget_failures', 0) + 1)
            err_resp = LLMResponse({
                LLMResponse.TEXT_KEY: f"Error: LLM call budget exceeded. {budget_error}",
                'error': str(budget_error),
                'error_type': 'BudgetExceededError',  # This specific type is important
                'call_id': ce.event_id,
                # For Explorer to potentially parse
                'error_payload': {'code': 'BUDGET_EXCEEDED_HARD', 'message': str(budget_error)}
            })
            error_result = ProcessResult(success=False, component_id=self.component_id,
                                         message=str(budget_error), error_code='BUDGET_EXCEEDED',
                                         output_data=err_resp)
            await self._publish_llm_episode(ce, error_result, frame, context)
            return err_resp  # Return the LLMResponse object with error details
        
        # Get base parameters from parameter service
        base_params = self._parameter_service.get_parameters(
            stage=llm_payload.stage,
            role=llm_payload.role,
            ctx=context
        )
        
        # Build final LLM settings by merging policies
        final_llm_settings = {**base_params}
        
        # Apply frame LLM policy
        if frame.llm_policy:
            final_llm_settings.update(frame.llm_policy)
            if 'preferred_route' in frame.llm_policy and not llm_payload.target_model_route:
                llm_payload.target_model_route = frame.llm_policy['preferred_route']
        
        # Apply payload-specific settings (highest priority)
        if llm_payload.llm_settings:
            final_llm_settings.update(llm_payload.llm_settings)

        # Handle routing
        if llm_payload.target_model_route:
            final_llm_settings['route'] = llm_payload.target_model_route
        elif 'route' in final_llm_settings and not llm_payload.target_model_route:
            llm_payload.target_model_route = final_llm_settings['route']

        # Make the LLM call
        try:
            # Create context for LLM call with additional metadata
            llm_context = context.with_metadata(
                frame_id=frame.id,
                frame_name=frame.name,
                epistemic_intent=ce.epistemic_intent,
                cognitive_event_id=ce.event_id
            )
            
            # Add event ID to settings if not present
            if 'call_id' not in final_llm_settings:  # Ensure ce.event_id is used if no call_id
                final_llm_settings['ce_event_id'] = ce.event_id

            llm_response_obj = await self._llm_router.call_llm_async(
                prompt=llm_payload.prompt,
                stage=llm_payload.stage,
                role=llm_payload.role,
                context=llm_context,
                settings=final_llm_settings
            )
            
            # Check for errors in response
            if llm_response_obj and llm_response_obj.get('error'):
                # Create error result
                error_result = ProcessResult(
                    success=False,
                    component_id=self.component_id,
                    message=str(llm_response_obj.get('error')),
                    error_code=llm_response_obj.get('error_type', 'LLM_ERROR'),
                    output_data=llm_response_obj
                )
                await self._publish_llm_episode(ce, error_result, frame, context)  # Publish episode even for LLM errors
            else:
                # Create success result with metrics
                success_result = create_success_result(
                    component_id=self.component_id,
                    message="LLM call successful",
                    output_data=llm_response_obj
                )
                
                # Add performance metrics
                duration_ms = (time.time() - start_time) * 1000
                success_result.add_metric("llm_call_duration_ms", duration_ms)
                success_result.add_metric("prompt_length", len(llm_payload.prompt))
                if llm_response_obj and LLMResponse.TEXT_KEY in llm_response_obj:
                    success_result.add_metric("response_length", 
                                            len(llm_response_obj[LLMResponse.TEXT_KEY]))
                
                # Update runtime state
                llm_requests = self.metadata.get_runtime_state("llm_requests", 0) + 1
                self.metadata.add_runtime_state("llm_requests", llm_requests)
                
                # Publish episode
                await self._publish_llm_episode(ce, success_result, frame, context)
            
            return llm_response_obj  # Always return the LLMResponse object

        except Exception as e:
            # ---- MODIFICATION START ----
            component_logger.error(
                f"ask_llm: Caught exception for CE {ce.event_id}. Exception Type: {type(e).__name__}, Exception Msg: {str(e)}",
                exc_info=True
            )
            
            # Attempt to get a safe string representation if str(e) itself fails
            safe_error_str = "Error stringifying exception"
            try:
                safe_error_str = str(e)
            except Exception as str_e:
                component_logger.error(f"ask_llm: Failed to str(e): {str_e}")
            # ---- MODIFICATION END ----
            
            # Ensure an LLMResponse is created for the episode and returned
            llm_response_obj = LLMResponse({
                LLMResponse.TEXT_KEY: f'Error processing LLM request: {safe_error_str}',
                'error': safe_error_str,  # Use safe_error_str
                'error_type': type(e).__name__,
                'call_id': ce.event_id
            })
            
            # Create error result
            error_result = create_error_result(
                component_id=self.component_id,
                error=e
            )
            error_result.output_data = llm_response_obj  # Attach the LLMResponse to the ProcessResult
            
            # Publish episode
            await self._publish_llm_episode(ce, error_result, frame, context)
            
            return llm_response_obj  # Return the LLMResponse with error

    async def _publish_llm_episode(self, ce: CognitiveEvent, result: ProcessResult, frame: Frame, context: NireonExecutionContext) -> None:
        component_logger = context.logger or logging.getLogger(__name__)
        episode_data = {
            'cognitive_event': ce,
            'response': result.output_data,
            'duration_ms': result.get_metric('llm_call_duration_ms', 0),
            'success': result.success,
            'frame_name': frame.name,
            'owning_agent_id': ce.owning_agent_id,
            'epistemic_intent': ce.epistemic_intent,
            'performance_metrics': result.performance_metrics,
            'correlation_id': result.correlation_id
        }
        event_bus = context.event_bus or self._event_bus
        if event_bus:
            try:
                # Remove 'await' - EventBusPort.publish is synchronous
                event_bus.publish('GATEWAY_LLM_EPISODE_COMPLETED', episode_data)
            except Exception as e:
                component_logger.error(f'Failed to publish LLM episode: {e}')
        else:
            component_logger.info(f'LLM Episode (no bus): {episode_data}')


    async def _publish_event_episode(self, ce: CognitiveEvent, result: ProcessResult, frame: Frame, context: NireonExecutionContext) -> None:
        component_logger = context.logger or logging.getLogger(__name__)
        episode_data = {
            'cognitive_event': ce,
            'response': result.output_data,
            'duration_ms': result.get_metric('publish_duration_ms', 0),
            'success': result.success,
            'frame_name': frame.name,
            'owning_agent_id': ce.owning_agent_id,
            'epistemic_intent': ce.epistemic_intent,
            'performance_metrics': result.performance_metrics,
            'correlation_id': result.correlation_id
        }
        event_bus = context.event_bus or self._event_bus
        if event_bus:
            try:
                # Remove 'await' - EventBusPort.publish is synchronous
                event_bus.publish('GATEWAY_EVENT_EPISODE_COMPLETED', episode_data)
            except Exception as e:
                component_logger.error(f"Failed to publish event episode: {e}")
        else:
            component_logger.info(f'Event Episode (no bus): {episode_data}')