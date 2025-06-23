# C:\Users\erees\Documents\development\nireon_v4\components\mechanisms\analyst_agents\quantifier_agent.py
from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Optional, Final

from core.base_component import NireonBaseComponent
from core.lifecycle import ComponentMetadata
from core.results import ProcessResult
from domain.context import NireonExecutionContext
from signals.core import ProtoTaskSignal, GenerativeLoopFinishedSignal

__all__ = ["QuantifierAgent"]

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
#  Module-level constants
# --------------------------------------------------------------------------- #
QUANTIFIER_METADATA: Final = ComponentMetadata(
    id='quantifier_agent_primary',
    name='Quantifier Agent',
    version='1.1.0',
    category='mechanism',
    description='Analyzes qualitative ideas and formulates quantitative Proto tasks to model them.',
    epistemic_tags=['analyzer', 'translator', 'modeler', 'quantifier'],
    accepts=['TrustAssessmentSignal'],
    produces=['ProtoTaskSignal', 'GenerativeLoopFinishedSignal'],
    requires_initialize=True,
    dependencies={'ProtoGenerator': '*', 'IdeaService': '*'}
)

# Trigger keywords for proto generation
PROTO_TRIGGER_KEYWORDS: Final = ['tariff', 'business', 'margin', 'supply chain', 'retail']

# Standard proto request template
DEFAULT_PROTO_REQUEST: Final = (
    "Model the impact of a 25% tariff on a simplified retail business's gross margin, "
    "assuming a starting Cost of Goods Sold (COGS) of 70% of revenue. "
    "The function should calculate the new gross margin if the business passes 0%, 50%, "
    "and 100% of the tariff cost onto the consumer. "
    "Visualize the three margin scenarios as a bar chart named 'tariff_impact.png'. "
    "The function should return a dictionary summarizing the three margin results."
)


# --------------------------------------------------------------------------- #
#  Main component class
# --------------------------------------------------------------------------- #
class QuantifierAgent(NireonBaseComponent):
    """
    Analyzes qualitative ideas and generates quantitative modeling tasks.
    
    The agent evaluates incoming ideas to determine if they warrant quantitative
    modeling via ProtoGenerator, and emits completion signals to manage the
    generative loop lifecycle.
    """
    
    METADATA_DEFINITION = QUANTIFIER_METADATA

    def __init__(
        self, 
        config: Dict[str, Any], 
        metadata_definition: ComponentMetadata
    ) -> None:
        super().__init__(config, metadata_definition)
        self.proto_generator = None
        self.idea_service = None

    # --------------------------------------------------------------------- #
    #  Lifecycle methods
    # --------------------------------------------------------------------- #
    async def _initialize_impl(self, context: NireonExecutionContext) -> None:
        """Initialize dependencies via component registry."""
        self._resolve_dependencies(context)
        self._validate_dependencies()
        context.logger.info("QuantifierAgent '%s' initialized successfully.", self.component_id)

    async def _process_impl(
        self, 
        data: Any, 
        context: NireonExecutionContext
    ) -> ProcessResult:
        """
        Process incoming idea data and potentially trigger proto generation.
        
        Args:
            data: Input data containing idea_id, idea_text, and assessment_details
            context: Execution context
            
        Returns:
            ProcessResult indicating success/failure and any output data
        """
        logger.info("--- QUANTIFIER AGENT PROCESS START --- Data keys: %s", data.keys())
        
        # Extract and validate input data
        idea_id = data.get('idea_id')
        idea_text = data.get('idea_text')
        assessment_details = data.get('assessment_details', {})

        if not self._validate_input_data(idea_id, idea_text):
            return self._create_error_result("Input data missing 'idea_id' or 'idea_text'.")
        
        context.logger.info(
            "[%s] Received idea to quantify: '%s...'",
            self.component_id,
            idea_text[:80]
        )

        # Determine if proto generation is warranted
        nl_request = self._formulate_proto_request(idea_text)
        quantifier_triggered = nl_request is not None

        # Always emit completion signal to ensure loop termination
        await self._emit_completion_signal(assessment_details, context, quantifier_triggered)
        
        if not nl_request:
            return self._create_success_result('Idea was not suitable for quantitative modeling.')

        # Trigger proto generation
        return await self._trigger_proto_generation(nl_request, idea_id, context)

    # --------------------------------------------------------------------- #
    #  Private helper methods
    # --------------------------------------------------------------------- #
    def _resolve_dependencies(self, context: NireonExecutionContext) -> None:
        """Resolve required dependencies from component registry."""
        from proto_generator.service import ProtoGenerator
        from application.services.idea_service import IdeaService
        
        # Dependency injection should populate these, but provide fallback
        if not self.proto_generator:
            self.proto_generator = context.component_registry.get("proto_generator_main")
        if not self.idea_service:
            self.idea_service = context.component_registry.get_service_instance(IdeaService)

    def _validate_dependencies(self) -> None:
        """Validate that all required dependencies are available."""
        if not self.proto_generator or not self.idea_service:
            raise RuntimeError(
                f"QuantifierAgent '{self.component_id}' failed to resolve dependencies on initialization."
            )

    @staticmethod
    def _validate_input_data(idea_id: Any, idea_text: Any) -> bool:
        """Validate that required input data is present."""
        return bool(idea_id and idea_text)

    def _create_error_result(self, message: str) -> ProcessResult:
        """Create a standardized error result."""
        return ProcessResult(
            success=False,
            component_id=self.component_id,
            message=message
        )

    def _create_success_result(
        self, 
        message: str, 
        output_data: Optional[Dict[str, Any]] = None
    ) -> ProcessResult:
        """Create a standardized success result."""
        return ProcessResult(
            success=True,
            component_id=self.component_id,
            message=message,
            output_data=output_data
        )

    async def _trigger_proto_generation(
        self, 
        nl_request: str, 
        idea_id: str, 
        context: NireonExecutionContext
    ) -> ProcessResult:
        """Trigger proto generation via ProtoGenerator."""
        context.logger.info(
            "[%s] Formulated NL request for ProtoGenerator: '%s'",
            self.component_id,
            nl_request
        )
        
        try:
            generator_result = await self.proto_generator.process(
                {'natural_language_request': nl_request},
                context
            )
            
            if generator_result.success:
                return self._create_success_result(
                    f"Successfully triggered ProtoGenerator for idea '{idea_id}'.",
                    generator_result.output_data
                )
            else:
                return ProcessResult(
                    success=False,
                    component_id=self.component_id,
                    message=f'Failed to trigger ProtoGenerator: {generator_result.message}',
                    output_data=generator_result.output_data
                )
        except Exception as exc:
            logger.exception("Error triggering proto generation for idea %s", idea_id)
            return self._create_error_result(f"Proto generation failed: {exc}")

    async def _emit_completion_signal(
        self, 
        assessment_details: Dict[str, Any], 
        context: NireonExecutionContext, 
        quantifier_triggered: bool
    ) -> None:
        """Emit completion signal to manage generative loop lifecycle."""
        if not context.event_bus:
            logger.warning('Event bus not available, cannot emit completion signal.')
            return

        try:
            final_payload = self._build_completion_payload(
                assessment_details, 
                quantifier_triggered
            )
            
            completion_signal = GenerativeLoopFinishedSignal(
                source_node_id=self.component_id,
                payload=final_payload
            )
            
            await asyncio.to_thread(
                context.event_bus.publish,
                completion_signal.signal_type,
                completion_signal.model_dump(mode='json')
            )
            
            logger.info(
                '[%s] Emitted GenerativeLoopFinishedSignal (Quantifier triggered: %s)',
                self.component_id,
                quantifier_triggered
            )
        except Exception as exc:
            logger.exception("Failed to emit completion signal")

    def _build_completion_payload(
        self, 
        assessment_details: Dict[str, Any], 
        quantifier_triggered: bool
    ) -> Dict[str, Any]:
        """Build the payload for the completion signal."""
        # Safely extract depth from nested metadata structure
        final_depth = assessment_details.get('metadata', {}).get('depth', -1)
        
        return {
            'status': 'completed_one_branch',
            'final_idea_id': assessment_details.get('idea_id'),
            'final_trust_score': assessment_details.get('trust_score'),
            'final_depth': final_depth,
            'quantifier_triggered': quantifier_triggered
        }

    def _formulate_proto_request(self, idea_text: str) -> Optional[str]:
        """
        Determine if idea warrants proto generation and return request text.
        
        Args:
            idea_text: The text content of the idea to analyze
            
        Returns:
            Proto request string if idea should trigger generation, None otherwise
        """
        idea_lower = idea_text.lower()
        
        if any(keyword in idea_lower for keyword in PROTO_TRIGGER_KEYWORDS):
            return DEFAULT_PROTO_REQUEST
            
        return None