from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

from core.base_component import NireonBaseComponent
from core.lifecycle import ComponentMetadata
from core.results import ComponentHealth, ProcessResult
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
    """Unified facade for mechanism-initiated cognitive events and service interactions.
    
    This gateway:
    - Routes cognitive events to appropriate services (LLM, Event Bus)
    - Applies frame-specific policies to service calls
    - Logs episodes for observability
    - Provides a simplified interface for mechanisms
    
    Note: This component follows pure context handling - it never stores context
    as instance state. All context is passed through method parameters.
    """

    def __init__(self,
                 llm_router: LLMPort,
                 parameter_service: ParameterService,
                 frame_factory: FrameFactoryService,
                 event_bus: Optional[EventBusPort] = None,
                 config: Optional[Dict[str, Any]] = None,
                 metadata_definition: Optional[ComponentMetadata] = None):
        """Initialize the gateway with required dependencies.
        
        Args:
            llm_router: Service for routing LLM requests
            parameter_service: Service for resolving LLM parameters
            frame_factory: Service for frame lifecycle management
            event_bus: Optional event bus for publishing episodes
            config: Optional configuration dictionary
            metadata_definition: Optional metadata (defaults to MECHANISM_GATEWAY_METADATA)
        """
        # Use provided metadata or default to MECHANISM_GATEWAY_METADATA
        super().__init__(
            config=config or {}, 
            metadata_definition=metadata_definition or MECHANISM_GATEWAY_METADATA
        )
        
        self._llm_router = llm_router
        self._parameter_service = parameter_service
        self._frame_factory = frame_factory
        self._event_bus = event_bus

    async def _initialize_impl(self, context: NireonExecutionContext) -> None:
        """Initialize the gateway, validating dependencies.
        
        This method is called by the framework during ComponentInitializationPhase
        when requires_initialize=True in the metadata.
        """
        component_logger = context.logger or logging.getLogger(__name__)
        component_logger.info(f"MechanismGateway '{self.component_id}' initializing")
        
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
        
        component_logger.info(f"MechanismGateway '{self.component_id}' initialization complete")

    async def _process_impl(self, data: Any, context: NireonExecutionContext) -> ProcessResult:
        """Process a cognitive event by delegating to process_cognitive_event.
        
        This method is called by the framework when the gateway is used as a component.
        The context parameter is provided by the runtime.
        """
        if not isinstance(data, CognitiveEvent):
            return ProcessResult(
                success=False,
                component_id=self.component_id,
                message=f"Expected CognitiveEvent but got {type(data).__name__}",
                error_code="INVALID_INPUT_TYPE"
            )
        
        try:
            result = await self.process_cognitive_event(data, context)
            return ProcessResult(
                success=True,
                component_id=self.component_id,
                message=f"Successfully processed {data.service_call_type} event",
                data=result
            )
        except Exception as e:
            return ProcessResult(
                success=False,
                component_id=self.component_id,
                message=f"Failed to process cognitive event: {e}",
                error_code="COGNITIVE_EVENT_PROCESSING_ERROR"
            )

    async def health_check(self, context: NireonExecutionContext) -> ComponentHealth:
        """Check the health of the gateway and its dependencies."""
        status = "HEALTHY"
        details = {
            "llm_router_available": self._llm_router is not None,
            "parameter_service_available": self._parameter_service is not None,
            "frame_factory_available": self._frame_factory is not None,
            "event_bus_available": self._event_bus is not None,
        }
        message = "MechanismGateway operational"
        
        # Check critical dependencies
        if not all([details["llm_router_available"], 
                    details["parameter_service_available"], 
                    details["frame_factory_available"]]):
            status = "UNHEALTHY"
            message = "MechanismGateway has missing critical dependencies"
        
        # Check health of dependencies if they support it
        if self._llm_router and hasattr(self._llm_router, 'health_check'):
            try:
                # Create a scoped context for the dependency
                llm_router_id = getattr(self._llm_router, 'component_id', 'llm_router')
                dep_context = context.with_component_scope(llm_router_id)
                llm_health = await self._llm_router.health_check(dep_context)
                details["llm_router_health"] = llm_health.status
                if not llm_health.is_healthy():
                    status = "DEGRADED"
                    message += f". LLMRouter is {llm_health.status}"
            except Exception as e:
                details["llm_router_health"] = f"ERROR: {type(e).__name__}"
                status = "DEGRADED"
                message += ". Error checking LLMRouter health"
        
        return ComponentHealth(
            component_id=self.component_id,
            status=status,
            message=message,
            details=details
        )

    async def process_cognitive_event(self, ce: CognitiveEvent, context: NireonExecutionContext) -> Any:
        """Process a cognitive event by routing it to the appropriate service.
        
        Args:
            ce: The cognitive event to process
            context: The execution context for this operation
            
        Returns:
            The result of processing (varies by service_call_type)
            
        Raises:
            ValueError: If the service_call_type is not supported
        """
        component_logger = context.logger or logging.getLogger(__name__)
        ce.gateway_received_ts = time.time()

        if ce.service_call_type == 'LLM_ASK':
            return await self.ask_llm(ce, context)
        elif ce.service_call_type == 'EVENT_PUBLISH':
            return await self.publish_event(ce, context)
        else:
            component_logger.error(f'Unsupported service_call_type: {ce.service_call_type} in CE {ce.event_id}')
            raise ValueError(f'Unsupported service_call_type: {ce.service_call_type}')

    async def publish_event(self, ce: CognitiveEvent, context: NireonExecutionContext) -> None:
        """Publish an event through the event bus on behalf of a mechanism.
        
        Args:
            ce: Cognitive event with EVENT_PUBLISH type
            context: The execution context for this operation
            
        Raises:
            ValueError: If the cognitive event is not properly formatted
            FrameNotFoundError: If the frame specified in the event doesn't exist
        """
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
        publication_success = False
        error_info = None

        # Use event bus from context first, then fall back to injected instance
        event_bus = context.event_bus or self._event_bus
        
        if event_bus:
            try:
                await event_bus.publish(event_type_to_publish, event_data_to_publish)
                publication_success = True
                component_logger.debug(f"Published event '{event_type_to_publish}' from CE {ce.event_id}")
            except Exception as e:
                error_info = f"Failed to publish event: {e}"
                component_logger.error(f'Error publishing event for CE {ce.event_id}: {e}', exc_info=True)
        else:
            error_info = 'Event bus not available'
            component_logger.warning(f"Event bus not available for CE {ce.event_id}")

        # Create response surrogate for the episode
        duration_ms = (time.time() - start_time) * 1000
        response_surrogate = {
            'published_event_type': event_type_to_publish,
            'publication_successful': publication_success,
            'details': 'Event published successfully' if publication_success else error_info
        }
        
        if error_info and not publication_success:
            response_surrogate['error'] = error_info
            response_surrogate['error_type'] = 'EventPublishingError'

        # Publish episode data
        episode_data = {
            'cognitive_event': ce,
            'response': response_surrogate,
            'duration_ms': duration_ms,
            'success': publication_success,
            'frame_name': frame.name,
            'owning_agent_id': ce.owning_agent_id,
            'epistemic_intent': ce.epistemic_intent
        }

        if event_bus:
            try:
                await event_bus.publish('GATEWAY_EVENT_EPISODE_COMPLETED', episode_data)
            except Exception as e:
                component_logger.error(f"Failed to publish episode data: {e}")
        else:
            component_logger.info(f'Event Episode (no bus): {episode_data}')

    async def ask_llm(self, ce: CognitiveEvent, context: NireonExecutionContext) -> LLMResponse:
        """Route an LLM request through the LLM router with appropriate policies.
        
        Args:
            ce: Cognitive event with LLM_ASK type
            context: The execution context for this operation
            
        Returns:
            LLMResponse from the LLM router
            
        Raises:
            ValueError: If the cognitive event is not properly formatted
            FrameNotFoundError: If the frame specified in the event doesn't exist
        """
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
            
            # Publish episode and return
            await self._publish_llm_episode(
                ce, err_resp, frame, start_time, False, context
            )
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
        llm_response_obj = None
        error_info = None
        error_type = None

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
                error_info = str(llm_response_obj.get('error'))
                error_type = llm_response_obj.get('error_type', 'LLMProcessingError')

        except Exception as e:
            component_logger.error(f'Error calling LLM for CE {ce.event_id}: {e}', exc_info=True)
            error_info = str(e)
            error_type = type(e).__name__
            llm_response_obj = LLMResponse({
                LLMResponse.TEXT_KEY: f'Error processing LLM request: {error_info}',
                'error': error_info,
                'error_type': error_type,
                'call_id': ce.event_id
            })

        # Publish episode
        await self._publish_llm_episode(
            ce, llm_response_obj, frame, start_time, 
            error_info is None, context
        )
        
        return llm_response_obj

    async def _publish_llm_episode(
        self,
        ce: CognitiveEvent,
        response: LLMResponse,
        frame: Frame,
        start_time: float,
        success: bool,
        context: NireonExecutionContext
    ) -> None:
        """Publish an LLM episode completed event.
        
        This is a helper method to avoid code duplication.
        """
        component_logger = context.logger or logging.getLogger(__name__)
        duration_ms = (time.time() - start_time) * 1000
        episode_data = {
            'cognitive_event': ce,
            'response': response,
            'duration_ms': duration_ms,
            'success': success,
            'frame_name': frame.name,
            'owning_agent_id': ce.owning_agent_id,
            'epistemic_intent': ce.epistemic_intent
        }

        event_bus = context.event_bus or self._event_bus
        if event_bus:
            try:
                await event_bus.publish('GATEWAY_LLM_EPISODE_COMPLETED', episode_data)
            except Exception as e:
                component_logger.error(f"Failed to publish LLM episode: {e}")
        else:
            component_logger.info(f'LLM Episode (no bus): {episode_data}')