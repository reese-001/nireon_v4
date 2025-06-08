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
from infrastructure.llm.parameter_service import ParameterService
from .mechanism_gateway_metadata import MECHANISM_GATEWAY_METADATA


class MechanismGateway(NireonBaseComponent, MechanismGatewayPort):
    """Enhanced MechanismGateway leveraging advanced result tracking."""

    def __init__(self,
                 llm_router: LLMPort,
                 parameter_service: ParameterService,
                 frame_factory: FrameFactoryService,
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
        self._event_bus = event_bus
        
        # Track gateway-specific metadata
        self.metadata.add_runtime_state("total_requests", 0)
        self.metadata.add_runtime_state("llm_requests", 0)
        self.metadata.add_runtime_state("event_publishes", 0)

    async def _initialize_impl(self, context: NireonExecutionContext) -> None:
        """Initialize with dependency validation and health tracking."""
        component_logger = context.logger or logging.getLogger(__name__)
        component_logger.info(f"MechanismGateway '{self.component_id}' initializing")
        
        # Track dependencies in metadata - use actual component IDs
        self.metadata.dependencies = {
            "LLMPort": "*",
            "parameter_service_global": ">=1.0.0",
            "frame_factory_service": "*"
        }
        
        # Validate required dependencies
        if not self._llm_router:
            raise RuntimeError(f"MechanismGateway '{self.component_id}' missing LLMRouter")
        if not self._parameter_service:
            raise RuntimeError(f"MechanismGateway '{self.component_id}' missing ParameterService")
        if not self._frame_factory:
            raise RuntimeError(f"MechanismGateway '{self.component_id}' missing FrameFactoryService")
        
        # Event bus is optional but we can try to resolve it from context if not injected
        if not self._event_bus and context.event_bus:
            self._event_bus = context.event_bus
            component_logger.info("EventBus resolved from initialization context")
        
        if not self._event_bus:
            component_logger.warning(f"EventBus not available for MechanismGateway '{self.component_id}'. "
                                     "Episode publishing will be limited.")
        
        # Update runtime state
        self.metadata.add_runtime_state("initialization_complete", True)
        self.metadata.add_runtime_state("event_bus_available", self._event_bus is not None)
        
        component_logger.info(f"MechanismGateway '{self.component_id}' initialization complete")

    async def _process_impl(self, data: Any, context: NireonExecutionContext) -> ProcessResult:
        """Process with enhanced result tracking and correlation."""
        if not isinstance(data, CognitiveEvent):
            return create_error_result(
                self.component_id,
                TypeError(f"Expected CognitiveEvent but got {type(data).__name__}")
            )
        
        ce: CognitiveEvent = data
        correlation_id = ce.event_id or str(uuid.uuid4())
        start_time = time.time()
        
        try:
            # Process the cognitive event
            result_data = await self.process_cognitive_event(ce, context)
            
            # Create success result with metrics
            result = create_success_result(
                component_id=self.component_id,
                message=f"Successfully processed {ce.service_call_type} event",
                output_data=result_data
            )
            
            # Add performance metrics
            duration_ms = (time.time() - start_time) * 1000
            result.add_metric("processing_time_ms", duration_ms)
            result.add_metric("event_type", ce.service_call_type)
            
            # Set correlation ID for tracing
            result.set_correlation_id(correlation_id)
            
            # Update runtime state
            total = self.metadata.get_runtime_state("total_requests", 0) + 1
            self.metadata.add_runtime_state("total_requests", total)
            
            return result
            
        except Exception as e:
            # Create error result with correlation
            error_result = create_error_result(
                component_id=self.component_id,
                error=e
            )
            error_result.set_correlation_id(correlation_id)
            
            # Emit error signal
            error_signal = SystemSignal.create_error(
                component_id=self.component_id,
                message=f"Failed to process {ce.service_call_type} event",
                error=e
            )
            error_signal.set_correlation_id(correlation_id)
            
            if self._event_bus:
                await self._event_bus.publish(
                    error_signal.to_event()["event_type"],
                    error_signal.to_event()
                )
            
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
            "event_publishes": self.metadata.get_runtime_state("event_publishes", 0)
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
        """Enhanced event publishing with result tracking."""
        component_logger = context.logger or logging.getLogger(__name__)
        
        if ce.service_call_type != 'EVENT_PUBLISH':
            raise ValueError("CognitiveEvent must have service_call_type 'EVENT_PUBLISH'")
        
        if not isinstance(ce.payload, dict) or 'event_type' not in ce.payload or 'event_data' not in ce.payload:
            raise ValueError("Payload must be a dict with 'event_type' and 'event_data' keys")

        start_time = time.time()
        
        # Get the frame to check policies
        frame = await self._frame_factory.get_frame_by_id(context, ce.frame_id)
        if not frame:
            raise FrameNotFoundError(f'Frame {ce.frame_id} not found for CE {ce.event_id}')

        if not frame.is_active():
            component_logger.warning(f"Publishing event from non-active Frame '{frame.id}' "
                                    f"(status: {frame.status}) for CE {ce.event_id}")
            
        event_type_to_publish = ce.payload['event_type']
        event_data_to_publish = ce.payload['event_data']
        
        # Use event bus from context first, then fall back to injected instance
        event_bus = context.event_bus or self._event_bus
        
        if event_bus:
            try:
                await event_bus.publish(event_type_to_publish, event_data_to_publish)
                
                # Create success result
                result = create_success_result(
                    component_id=self.component_id,
                    message=f"Published event '{event_type_to_publish}'",
                    output_data={
                        'published_event_type': event_type_to_publish,
                        'publication_successful': True
                    }
                )
                
                # Track metrics
                duration_ms = (time.time() - start_time) * 1000
                result.add_metric("publish_duration_ms", duration_ms)
                result.add_metric("frame_id", frame.id)
                
                # Update runtime state
                publishes = self.metadata.get_runtime_state("event_publishes", 0) + 1
                self.metadata.add_runtime_state("event_publishes", publishes)
                
                # Publish episode
                await self._publish_event_episode(ce, result, frame, context)
                
                return result
                
            except Exception as e:
                error_result = create_error_result(
                    component_id=self.component_id,
                    error=e
                )
                error_result.metadata["event_type"] = event_type_to_publish
                
                await self._publish_event_episode(ce, error_result, frame, context)
                
                return error_result
        else:
            # No event bus available
            return create_error_result(
                component_id=self.component_id,
                error=RuntimeError("Event bus not available")
            )

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
            raise FrameNotFoundError(f'Frame {ce.frame_id} not found for CE {ce.event_id}')

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
            if 'call_id' not in final_llm_settings:
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
                
                return llm_response_obj

        except Exception as e:
            component_logger.error(f'Error calling LLM for CE {ce.event_id}: {e}', exc_info=True)
            
            # Create error response
            llm_response_obj = LLMResponse({
                LLMResponse.TEXT_KEY: f'Error processing LLM request: {e}',
                'error': str(e),
                'error_type': type(e).__name__,
                'call_id': ce.event_id
            })
            
            # Create error result
            error_result = create_error_result(
                component_id=self.component_id,
                error=e
            )
            error_result.output_data = llm_response_obj
            
            # Publish episode
            await self._publish_llm_episode(ce, error_result, frame, context)
            
            return llm_response_obj

    async def _publish_llm_episode(
        self,
        ce: CognitiveEvent,
        result: ProcessResult,
        frame: Frame,
        context: NireonExecutionContext
    ) -> None:
        """Publish enhanced LLM episode with result tracking."""
        component_logger = context.logger or logging.getLogger(__name__)
        
        episode_data = {
            'cognitive_event': ce,
            'response': result.output_data,
            'duration_ms': result.get_metric("llm_call_duration_ms", 0),
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
                await event_bus.publish('GATEWAY_LLM_EPISODE_COMPLETED', episode_data)
            except Exception as e:
                component_logger.error(f"Failed to publish LLM episode: {e}")
        else:
            component_logger.info(f'LLM Episode (no bus): {episode_data}')

    async def _publish_event_episode(
        self,
        ce: CognitiveEvent,
        result: ProcessResult,
        frame: Frame,
        context: NireonExecutionContext
    ) -> None:
        """Publish enhanced event episode with result tracking."""
        component_logger = context.logger or logging.getLogger(__name__)
        
        episode_data = {
            'cognitive_event': ce,
            'response': result.output_data,
            'duration_ms': result.get_metric("publish_duration_ms", 0),
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
                await event_bus.publish('GATEWAY_EVENT_EPISODE_COMPLETED', episode_data)
            except Exception as e:
                component_logger.error(f"Failed to publish event episode: {e}")
        else:
            component_logger.info(f'Event Episode (no bus): {episode_data}')