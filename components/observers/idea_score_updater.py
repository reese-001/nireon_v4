# 1. Create a new observer component to update idea scores
# Save as: components/observers/idea_score_updater.py

import logging
from typing import Dict, Any, Optional
from core.base_component import NireonBaseComponent
from core.lifecycle import ComponentMetadata
from core.results import ProcessResult
from domain.context import NireonExecutionContext
from signals.core import TrustAssessmentSignal
from components.service_resolution_mixin import ServiceResolutionMixin

logger = logging.getLogger(__name__)

IDEA_SCORE_UPDATER_METADATA = ComponentMetadata(
    id="idea_score_updater",
    name="Idea Score Updater",
    version="1.0.0",
    category="observer",
    description="Updates idea trust and novelty scores in the database when assessments are made",
    epistemic_tags=["score_tracker", "database_updater"],
    requires_initialize=True,
    dependencies={
        "IdeaRepositoryPort": "*"
    }
)

class IdeaScoreUpdater(NireonBaseComponent, ServiceResolutionMixin):
    """Automatically updates idea scores when trust assessments are published"""
    
    METADATA_DEFINITION = IDEA_SCORE_UPDATER_METADATA
    
    def __init__(self, config: Dict[str, Any], 
                 metadata_definition: Optional[ComponentMetadata] = None):
        super().__init__(config, metadata_definition or self.METADATA_DEFINITION)
        self.idea_repository = None
        self.update_count = 0
        
    async def _initialize_impl(self, context: NireonExecutionContext) -> None:
        self.resolve_services(context, {
            "idea_repository": "IdeaRepositoryPort"
        }, raise_on_missing=True)
        
        context.logger.info(f"[{self.component_id}] Initialized - ready to update idea scores")
        
    async def _process_impl(self, data: Dict[str, Any], context: NireonExecutionContext) -> ProcessResult:
        signal: Optional[TrustAssessmentSignal] = data.get('signal')
        if not isinstance(signal, TrustAssessmentSignal):
            return ProcessResult(success=False, message='Expected TrustAssessmentSignal', component_id=self.component_id)
        
        idea_id = signal.target_id
        trust_score = signal.trust_score
        novelty_score = signal.novelty_score
        is_stable = None
        
        if hasattr(signal, 'payload') and isinstance(signal.payload, dict):
            # Check for 'is_stable' in the top-level payload first
            is_stable = signal.payload.get('is_stable')
            # If not found, check inside 'assessment_details' as a fallback
            if is_stable is None:
                 details = signal.payload.get('assessment_details', {})
                 if isinstance(details, dict):
                     is_stable = details.get('is_stable')
        
        if not idea_id or trust_score is None:
            return ProcessResult(success=False, message=f'Missing required data: idea_id={idea_id}, trust_score={trust_score}', component_id=self.component_id)
        
        try:
            success = self.idea_repository.update_scores(idea_id=idea_id, trust_score=trust_score, novelty_score=novelty_score, is_stable=is_stable)
            if success:
                self.update_count += 1
                # FIX: Correctly format the novelty score before the f-string
                novelty_str = f"{novelty_score:.2f}" if novelty_score is not None else 'N/A'
                context.logger.info(f"[{self.component_id}] Updated scores for idea {idea_id[:8]}...: trust={trust_score:.2f}, novelty={novelty_str}, stable={is_stable}")
                
                return ProcessResult(
                    success=True, 
                    message=f'Updated scores for idea {idea_id}', 
                    component_id=self.component_id, 
                    output_data={
                        'idea_id': idea_id, 
                        'trust_score': trust_score, 
                        'novelty_score': novelty_score, 
                        'is_stable': is_stable, 
                        'total_updates': self.update_count
                    }
                )
            else:
                return ProcessResult(success=False, message=f'Idea {idea_id} not found in repository', component_id=self.component_id)
        except Exception as e:
            context.logger.error(f'[{self.component_id}] Error updating scores for {idea_id}: {e}', exc_info=True)
            return ProcessResult(success=False, message=f'Error updating scores: {str(e)}', component_id=self.component_id)
