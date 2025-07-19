import logging
from typing import Any, Dict, List, Optional
import yaml
from core.base_component import NireonBaseComponent
from core.lifecycle import ComponentMetadata
from core.results import ProcessResult
from domain.cognitive_events import CognitiveEvent, LLMRequestPayload
from domain.context import NireonExecutionContext
from domain.epistemic_stage import EpistemicStage
from domain.ports.mechanism_gateway_port import MechanismGatewayPort
from signals.core import MathResultSignal, FormalResultSignal

logger = logging.getLogger(__name__)

ROOT_FRAME_ID = 'F-ROOT-00000000-0000-0000-0000-000000000000'

PRINCIPIA_METADATA = ComponentMetadata(
    id='principia_agent_primary',
    name='Principia Agent',
    version='1.1.0',
    category='mechanism',
    description='An agent specializing in symbolic and deterministic mathematics, emitting structured formal results.',
    epistemic_tags=['evaluator', 'symbolic_math', 'computation', 'reasoning', 'formalizer'],
    accepts=['MathQuerySignal'],
    produces=['FormalResultSignal', 'MathResultSignal', 'ExplanationSignal'],
    requires_initialize=True,
    dependencies={'MechanismGatewayPort': '>=1.0.0'}
)

class PrincipiaAgent(NireonBaseComponent):
    METADATA_DEFINITION = PRINCIPIA_METADATA

    def __init__(self, config: Dict[str, Any], metadata_definition: ComponentMetadata, gateway: Optional[MechanismGatewayPort] = None):
        super().__init__(config, metadata_definition)
        self.gateway = gateway
        logger.info(f"PrincipiaAgent '{self.component_id}' instance created.")

    async def _initialize_impl(self, context: NireonExecutionContext) -> None:
        if not self.gateway:
            self.gateway = context.component_registry.get_service_instance(MechanismGatewayPort)
        context.logger.info(f"PrincipiaAgent '{self.component_id}' initialized successfully.")

    def _resolve_frame_id(self, data: Dict[str, Any], context: NireonExecutionContext) -> str:
        frame_id_from_payload = data.get('payload', {}).get('frame_id')
        if frame_id_from_payload:
            logger.debug(f'[{self.component_id}] Using frame_id from signal payload: {frame_id_from_payload}')
            return frame_id_from_payload
        frame_id_from_context = context.metadata.get('current_frame_id')
        if frame_id_from_context:
            logger.debug(f'[{self.component_id}] Using frame_id from context: {frame_id_from_context}')
            return frame_id_from_context
        logger.warning(f'[{self.component_id}] No frame_id found, using ROOT frame: {ROOT_FRAME_ID}')
        return ROOT_FRAME_ID

    async def _process_impl(self, data: Dict[str, Any], context: NireonExecutionContext) -> ProcessResult:
        expression = data.get('expression')
        operations = data.get('operations', [])
        natural_language_query = data.get('natural_language_query', "the user's math question")
        frame_id = self._resolve_frame_id(data, context)
        
        if not expression:
            return ProcessResult(success=False, message="Input data must contain an 'expression' for computation.", component_id=self.component_id)

        logger.info(f'[{self.component_id}] Processing math query: "{natural_language_query}"')
        logger.debug(f'[{self.component_id}] Expression: {expression}, Operations: {operations}')

        computation_result = await self._perform_computation(frame_id, expression, operations, context)

        if not computation_result['success']:
            return await self._handle_computation_failure(frame_id, natural_language_query, computation_result, context)
        
        formal_template = await self._format_as_formal_template(frame_id, natural_language_query, computation_result, context)
        
        if formal_template:
            await self._publish_formal_result(frame_id, natural_language_query, formal_template, computation_result, context)
            return ProcessResult(success=True, message='Formal mathematical analysis completed successfully.', component_id=self.component_id, output_data=formal_template)
        else:
            explanation = self._generate_fallback_explanation(computation_result.get('final_result', 'N/A'), computation_result.get('steps', []))
            await self._publish_legacy_result(frame_id, natural_language_query, explanation, computation_result, context)
            return ProcessResult(success=True, message='Mathematical computation completed, but formal formatting failed. Emitting legacy result.', component_id=self.component_id, output_data={'explanation': explanation, 'computation_details': computation_result})

    async def _perform_computation(self, frame_id: str, expression: str, operations: List[Dict[str, Any]], context: NireonExecutionContext) -> Dict[str, Any]:
        math_payload = {'engine': 'sympy', 'expression': expression, 'operations': operations}
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
            return {'success': False, 'error': str(e)}

    async def _format_as_formal_template(self, frame_id: str, query: str, computation_result: Dict[str, Any], context: NireonExecutionContext) -> Optional[Dict[str, Any]]:
        final_answer = computation_result.get('final_result', 'N/A')
        steps = self._format_steps(computation_result.get('steps', []))
        
        prompt = f"""
        Based on the user's query and the computational result, structure the output into a 5-part formal template.
        The output MUST be a valid YAML object.

        User Query: "{query}"
        Computational Result: {final_answer}
        Computational Steps:
        {steps}

        Format this into the following YAML structure:
        
        definition: "A formal, mathematical statement of the problem."
        computation_table: # (Optional) A dictionary of key-value pairs if the result is a table. Omit if not applicable.
        result_extract: "A single, concise sentence stating the final answer."
        monotonicity_analysis: # (Optional) A brief analysis of the function's behavior (e.g., increasing/decreasing). Omit if not applicable.
        conclusion: "A concluding sentence that relates the formal result back to the user's original query."
        is_proof_valid: # (Optional) true or false if the query was a proof check. Omit otherwise.
        """
        
        llm_payload = LLMRequestPayload(prompt=prompt, stage=EpistemicStage.SYNTHESIS, role='formal_formatter', llm_settings={'temperature': 0.0, 'max_tokens': 1024})
        llm_ce = CognitiveEvent(
            frame_id=frame_id,
            owning_agent_id=self.component_id,
            service_call_type='LLM_ASK',
            payload=llm_payload,
            epistemic_intent='FORMAT_FORMAL_RESULT'
        )
        
        try:
            llm_response = await self.gateway.process_cognitive_event(llm_ce, context)
            yaml_content = llm_response.text.strip().replace("```yaml", "").replace("```", "")
            return yaml.safe_load(yaml_content)
        except Exception as e:
            logger.error(f"[{self.component_id}] Failed to format result into template: {e}")
            return None

    def _format_steps(self, steps: List[Dict[str, Any]]) -> str:
        if not steps:
            return 'No intermediate steps recorded.'
        formatted = []
        for i, step in enumerate(steps, 1):
            operation = step.get('operation', 'unknown')
            result = step.get('result', 'N/A')
            formatted.append(f'{i}. {operation}: {result}')
        return '\n'.join(formatted)

    def _generate_fallback_explanation(self, result: str, steps: List[Dict[str, Any]]) -> str:
        explanation = f"The computed result is: {result}."
        if steps:
            explanation += "\n\nComputation steps:"
            for i, step in enumerate(steps, 1):
                operation = step.get('operation', 'unknown')
                step_result = step.get('result', 'N/A')
                explanation += f"\n{i}. {operation}: {step_result}"
        return explanation

    async def _handle_computation_failure(self, frame_id: str, query: str, computation_result: Dict[str, Any], context: NireonExecutionContext) -> ProcessResult:
        error = computation_result.get('error', 'Unknown computation error.')
        logger.error(f'[{self.component_id}] Math computation failed: {error}')
        explanation = f'The mathematical computation could not be completed. Error: {error}'
        await self._publish_legacy_result(frame_id, query, explanation, computation_result, context)
        return ProcessResult(success=False, message=f'Mathematical computation failed: {error}', component_id=self.component_id, output_data={'error': error, 'computation_details': computation_result})

    async def _publish_legacy_result(self, frame_id: str, query: str, explanation: str, computation_result: Dict[str, Any], context: NireonExecutionContext) -> None:
        result_signal = MathResultSignal(
            source_node_id=self.component_id,
            natural_language_query=query,
            explanation=explanation,
            computation_details=computation_result
        )
        await self._publish_signal(frame_id, result_signal, 'PUBLISH_MATH_FAILURE', context)

    async def _publish_formal_result(self, frame_id: str, query: str, template: Dict[str, Any], computation_result: Dict[str, Any], context: NireonExecutionContext) -> None:
        signal = FormalResultSignal(
            source_node_id=self.component_id,
            original_idea_id=context.metadata.get('original_idea_id', 'unknown'),
            original_query=query,
            definition=template.get('definition', 'N/A'),
            computation_table=template.get('computation_table'),
            result_extract=template.get('result_extract', 'N/A'),
            monotonicity_analysis=template.get('monotonicity_analysis'),
            conclusion=template.get('conclusion', 'N/A'),
            is_proof_valid=template.get('is_proof_valid'),
            computation_details=computation_result
        )
        await self._publish_signal(frame_id, signal, 'PUBLISH_FORMAL_RESULT', context)
        logger.info(f"[{self.component_id}] Successfully published FormalResultSignal.")

    async def _publish_signal(self, frame_id: str, signal: MathResultSignal | FormalResultSignal, intent: str, context: NireonExecutionContext) -> None:
        try:
            publish_payload = {'event_type': signal.signal_type, 'event_data': signal.model_dump()}
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