from __future__ import annotations
import logging
from typing import Any, Dict, Optional
from core.base_component import NireonBaseComponent
from core.lifecycle import ComponentMetadata
from core.results import ProcessResult
from domain.context import NireonExecutionContext
from domain.ports.idea_service_port import IdeaServicePort
from signals.core import FormalResultSignal, TrustAssessmentSignal
from components.service_resolution_mixin import ServiceResolutionMixin

logger = logging.getLogger(__name__)

REINTEGRATION_METADATA = ComponentMetadata(
    id='reintegration_observer_main',
    name='Reintegration Observer',
    version='1.0.0',
    category='observer',
    description='Observes formal computation results and reintegrates them into the narrative idea space.',
    epistemic_tags=['observer', 'integrator', 'reconciler', 'grounding'],
    requires_initialize=True,
    dependencies={'IdeaServicePort': '*', 'EventBusPort': '*'}
)

class ReintegrationObserver(NireonBaseComponent, ServiceResolutionMixin):
    METADATA_DEFINITION = REINTEGRATION_METADATA

    def __init__(self, config: Dict[str, Any], metadata_definition: Optional[ComponentMetadata] = None):
        super().__init__(config, metadata_definition or self.METADATA_DEFINITION)
        self.idea_service: Optional[IdeaServicePort] = None
        self.event_bus = None # Resolved in initialize
        self.reintegration_count = 0
        self.failed_updates = 0

    async def _initialize_impl(self, context: NireonExecutionContext) -> None:
        from domain.ports.event_bus_port import EventBusPort
        service_map = {
            'idea_service': IdeaServicePort,
            'event_bus': EventBusPort
        }
        self.resolve_services(context, service_map, raise_on_missing=True)
        context.logger.info(f"[{self.component_id}] initialized and ready to reintegrate formal results.")

    async def _process_impl(self, data: Dict[str, Any], context: NireonExecutionContext) -> ProcessResult:
        signal: Optional[FormalResultSignal] = data.get('signal')
        if not isinstance(signal, FormalResultSignal):
            return ProcessResult(success=False, message=f"Expected FormalResultSignal, got {type(signal).__name__}", component_id=self.component_id)

        original_idea_id = signal.original_idea_id
        logger.info(f"[{self.component_id}] Reintegrating formal result for idea '{original_idea_id}'")

        try:
            original_idea = self.idea_service.get_by_id(original_idea_id)
            if not original_idea:
                self.failed_updates += 1
                return ProcessResult(success=False, message=f"Original idea '{original_idea_id}' not found.", component_id=self.component_id)

            # 1. Update the idea text with the formal summary
            formal_summary = f"\n\n--- Formal Analysis Summary ---\nDefinition: {signal.definition}\nResult: {signal.result_extract}\nConclusion: {signal.conclusion}\n--- End Formal Analysis ---"
            updated_text = original_idea.text + formal_summary
            
            # 2. Update trust score and stability based on proof validation
            new_trust_score = original_idea.trust_score or 5.0
            new_is_stable = original_idea.is_stable
            
            if signal.is_proof_valid is True:
                new_trust_score = 9.5  # High confidence for validated proofs
                new_is_stable = True
            elif signal.is_proof_valid is False:
                new_trust_score = 2.0  # Low confidence for invalidated proofs
                new_is_stable = False
            
            # 3. Create a new version of the idea object with all updates
            updated_idea = original_idea.with_metadata_update({
                'formal_result_id': signal.signal_id,
                'reintegrated_at': context.timestamp.isoformat()
            })
            updated_idea.text = updated_text
            updated_idea = updated_idea.with_scores(
                trust_score=new_trust_score, 
                is_stable=new_is_stable, 
                novelty_score=original_idea.novelty_score
            )

            # 4. Save the updated idea
            self.idea_service.save(updated_idea)
            
            # 5. Publish a new TrustAssessmentSignal to notify the system
            await self._publish_new_assessment(updated_idea, signal, context)

            self.reintegration_count += 1
            return ProcessResult(success=True, message=f"Reintegrated formal results into idea '{original_idea_id}'.", output_data={'updated_idea_id': original_idea_id})

        except Exception as e:
            self.failed_updates += 1
            logger.error(f"[{self.component_id}] Failed to reintegrate result for '{original_idea_id}': {e}", exc_info=True)
            return ProcessResult(success=False, message=f"Reintegration failed: {e}", component_id=self.component_id)

    async def _publish_new_assessment(self, idea, original_signal: FormalResultSignal, context: NireonExecutionContext):
        if not self.event_bus:
            logger.warning(f"[{self.component_id}] EventBus not available, cannot publish updated assessment.")
            return

        assessment_payload = {
            'idea_id': idea.idea_id,
            'idea_text': idea.text,
            'is_stable': idea.is_stable,
            'rejection_reason': 'N/A' if idea.is_stable else 'Formal analysis contradicted or could not validate the claim.',
            'assessment_details': {
                'trust_score': idea.trust_score,
                'is_stable': idea.is_stable,
                'metadata': {'source': 'reintegration_observer', 'formal_result_id': original_signal.signal_id}
            }
        }
        
        new_assessment_signal = TrustAssessmentSignal(
            source_node_id=self.component_id,
            target_id=idea.idea_id,
            target_type='Idea',
            trust_score=idea.trust_score,
            assessment_rationale=f"Updated based on formal analysis (Proof valid: {original_signal.is_proof_valid})",
            payload=assessment_payload,
            parent_signal_ids=[original_signal.signal_id]
        )
        
        self.event_bus.publish(new_assessment_signal.signal_type, new_assessment_signal)
        logger.info(f"[{self.component_id}] Published new TrustAssessmentSignal for formally updated idea '{idea.idea_id}'.")