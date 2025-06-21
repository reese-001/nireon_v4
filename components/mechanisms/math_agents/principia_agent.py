import logging
from typing import Any, Dict, List, Optional
from core.base_component import NireonBaseComponent
from core.lifecycle import ComponentMetadata
from core.results import ProcessResult
from domain.cognitive_events import CognitiveEvent, LLMRequestPayload
from domain.context import NireonExecutionContext
from domain.epistemic_stage import EpistemicStage
from domain.ports.mechanism_gateway_port import MechanismGatewayPort
from signals.core import MathResultSignal

logger = logging.getLogger(__name__)

# Constants
ROOT_FRAME_ID = 'F-ROOT-00000000-0000-0000-0000-000000000000'

PRINCIPIA_METADATA = ComponentMetadata(
    id='principia_agent_primary',
    name='Principia Agent',
    version='1.0.1',  # Bumped version for improvements
    category='mechanism',
    description='An agent specializing in symbolic and deterministic mathematics by offloading computation to a dedicated MathPort.',
    epistemic_tags=['evaluator', 'symbolic_math', 'computation', 'reasoning'],
    accepts=['MathQuerySignal'],
    produces=['MathResultSignal', 'ExplanationSignal'],
    requires_initialize=True,
    dependencies={'MechanismGatewayPort': '>=1.0.0'}
)

class PrincipiaAgent(NireonBaseComponent):
    """
    Mathematical computation agent that processes MathQuerySignal inputs.
    
    Frame ID Resolution Order:
    1. Signal payload 'frame_id'
    2. Execution context metadata 'current_frame_id'
    3. Fallback to ROOT_FRAME_ID
    """
    
    METADATA_DEFINITION = PRINCIPIA_METADATA
    
    def __init__(self, config: Dict[str, Any], metadata_definition: ComponentMetadata, 
                 gateway: Optional[MechanismGatewayPort] = None):
        super().__init__(config, metadata_definition)
        self.gateway = gateway
        logger.info(f"PrincipiaAgent '{self.component_id}' instance created.")
    
    async def _initialize_impl(self, context: NireonExecutionContext) -> None:
        """Initialize the agent and resolve the gateway dependency."""
        if not self.gateway:
            self.gateway = context.component_registry.get_service_instance(MechanismGatewayPort)
        context.logger.info(f"PrincipiaAgent '{self.component_id}' initialized successfully.")
    
    def _resolve_frame_id(self, data: Dict[str, Any], context: NireonExecutionContext) -> str:
        """
        Resolve frame ID with multiple fallback options.
        
        Returns the first available from:
        1. Signal payload 'frame_id'
        2. Context metadata 'current_frame_id'
        3. ROOT_FRAME_ID as final fallback
        """
        # Try signal payload first
        frame_id_from_payload = data.get('payload', {}).get('frame_id')
        if frame_id_from_payload:
            logger.debug(f'[{self.component_id}] Using frame_id from signal payload: {frame_id_from_payload}')
            return frame_id_from_payload
        
        # Try context metadata
        frame_id_from_context = context.metadata.get('current_frame_id')
        if frame_id_from_context:
            logger.debug(f'[{self.component_id}] Using frame_id from context: {frame_id_from_context}')
            return frame_id_from_context
        
        # Use root frame as fallback
        logger.warning(f'[{self.component_id}] No frame_id found, using ROOT frame: {ROOT_FRAME_ID}')
        return ROOT_FRAME_ID
    
    async def _process_impl(self, data: Dict[str, Any], context: NireonExecutionContext) -> ProcessResult:
        """Process a MathQuerySignal and return results."""
        # Extract required data
        expression = data.get('expression')
        operations = data.get('operations', [])
        natural_language_query = data.get('natural_language_query', "the user's math question")
        
        # Resolve frame ID
        frame_id = self._resolve_frame_id(data, context)
        
        # Validate input
        if not expression:
            return ProcessResult(
                success=False,
                message="Input data must contain an 'expression' for computation.",
                component_id=self.component_id
            )
        
        # Log processing start
        logger.info(f'[{self.component_id}] Processing math query: "{natural_language_query}"')
        logger.debug(f'[{self.component_id}] Expression: {expression}, Operations: {operations}')
        
        # Perform computation
        computation_result = await self._perform_computation(
            frame_id, expression, operations, context
        )
        
        if not computation_result['success']:
            # Handle computation failure
            return await self._handle_computation_failure(
                frame_id, natural_language_query, computation_result, context
            )
        
        # Generate explanation for successful computation
        explanation = await self._generate_explanation(
            frame_id, natural_language_query, computation_result, context
        )
        
        # Publish final result
        await self._publish_result(
            frame_id, natural_language_query, explanation, computation_result, context
        )
        
        return ProcessResult(
            success=True,
            message='Mathematical computation and explanation completed successfully.',
            component_id=self.component_id,
            output_data={
                'natural_language_query': natural_language_query,
                'explanation': explanation,
                'computation_details': computation_result
            }
        )
    
    async def _perform_computation(self, frame_id: str, expression: str, 
                                   operations: List[Dict[str, Any]], 
                                   context: NireonExecutionContext) -> Dict[str, Any]:
        """Perform the mathematical computation via the gateway."""
        math_payload = {
            'engine': 'sympy',
            'expression': expression,
            'operations': operations
        }
        
        math_ce = CognitiveEvent(
            frame_id=frame_id,
            owning_agent_id=self.component_id,
            service_call_type='MATH_COMPUTE',
            payload=math_payload,
            epistemic_intent='PERFORM_SYMBOLIC_COMPUTATION'
        )
        
        logger.info(f'[{self.component_id}] Sending computation task to gateway: {math_payload}')
        
        try:
            return await self.gateway.process_cognitive_event(math_ce, context)
        except Exception as e:
            logger.error(f'[{self.component_id}] Gateway processing failed: {e}')
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _generate_explanation(self, frame_id: str, query: str, 
                                    computation_result: Dict[str, Any],
                                    context: NireonExecutionContext) -> str:
        """Generate a natural language explanation of the computation result."""
        final_answer = computation_result.get('final_result', 'N/A')
        steps = computation_result.get('steps', [])
        
        # Build a detailed prompt
        explanation_prompt = f"""The user asked: '{query}'.

The computation resulted in: {final_answer}

The following steps were performed:
{self._format_steps(steps)}

Please provide a clear, step-by-step explanation of how this result was obtained. 
Explain the mathematical concepts involved in simple terms suitable for the user's understanding."""
        
        llm_payload = LLMRequestPayload(
            prompt=explanation_prompt,
            stage=EpistemicStage.SYNTHESIS,
            role='math_explainer'
        )
        
        llm_ce = CognitiveEvent(
            frame_id=frame_id,
            owning_agent_id=self.component_id,
            service_call_type='LLM_ASK',
            payload=llm_payload,
            epistemic_intent='GENERATE_MATH_EXPLANATION'
        )
        
        logger.info(f'[{self.component_id}] Requesting LLM explanation for math result.')
        
        try:
            llm_response = await self.gateway.process_cognitive_event(llm_ce, context)
            return llm_response.text
        except Exception as e:
            logger.error(f'[{self.component_id}] Failed to get LLM explanation: {e}')
            # Provide a fallback explanation
            return self._generate_fallback_explanation(query, final_answer, steps)
    
    def _format_steps(self, steps: List[Dict[str, Any]]) -> str:
        """Format computation steps for the LLM prompt."""
        if not steps:
            return "No intermediate steps recorded."
        
        formatted = []
        for i, step in enumerate(steps, 1):
            operation = step.get('operation', 'unknown')
            result = step.get('result', 'N/A')
            formatted.append(f"{i}. {operation}: {result}")
        
        return "\n".join(formatted)
    
    def _generate_fallback_explanation(self, query: str, result: str, 
                                        steps: List[Dict[str, Any]]) -> str:
        """Generate a basic explanation when LLM is unavailable."""
        explanation = f"For the question '{query}', the computed result is: {result}."
        
        if steps:
            explanation += "\n\nComputation steps:"
            for i, step in enumerate(steps, 1):
                operation = step.get('operation', 'unknown')
                step_result = step.get('result', 'N/A')
                explanation += f"\n{i}. {operation}: {step_result}"
        
        return explanation
    
    async def _handle_computation_failure(self, frame_id: str, query: str,
                                          computation_result: Dict[str, Any],
                                          context: NireonExecutionContext) -> ProcessResult:
        """Handle cases where mathematical computation fails."""
        error = computation_result.get('error', 'Unknown computation error.')
        logger.error(f'[{self.component_id}] Math computation failed: {error}')
        
        # Publish error signal
        error_signal = MathResultSignal(
            source_node_id=self.component_id,
            natural_language_query=query,
            explanation=f'The mathematical computation could not be completed. Error: {error}',
            computation_details=computation_result
        )
        
        await self._publish_signal(frame_id, error_signal, 'PUBLISH_MATH_FAILURE', context)
        
        return ProcessResult(
            success=False,
            message=f'Mathematical computation failed: {error}',
            component_id=self.component_id,
            output_data={
                'error': error,
                'computation_details': computation_result
            }
        )
    
    async def _publish_result(self, frame_id: str, query: str, explanation: str,
                              computation_result: Dict[str, Any],
                              context: NireonExecutionContext) -> None:
        """Publish the final MathResultSignal."""
        result_signal = MathResultSignal(
            source_node_id=self.component_id,
            natural_language_query=query,
            explanation=explanation,
            computation_details=computation_result
        )
        
        await self._publish_signal(frame_id, result_signal, 'PUBLISH_MATH_SUCCESS', context)
        logger.info(f'[{self.component_id}] Successfully published final MathResultSignal.')
    
    async def _publish_signal(self, frame_id: str, signal: MathResultSignal,
                              intent: str, context: NireonExecutionContext) -> None:
        """Publish a signal via the gateway."""
        try:
            publish_payload = {
                'event_type': signal.signal_type,
                'event_data': signal.model_dump()
            }
            
            publish_ce = CognitiveEvent(
                frame_id=frame_id,
                owning_agent_id=self.component_id,
                service_call_type='EVENT_PUBLISH',
                payload=publish_payload,
                epistemic_intent=intent
            )
            
            await self.gateway.process_cognitive_event(publish_ce, context)
        except Exception as e:
            logger.error(f'[{self.component_id}] Failed to publish signal: {e}')