from __future__ import annotations
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Tuple
from domain.context import NireonExecutionContext
from core.results import AnalysisResult
from domain.evaluation.assessment import IdeaAssessment
from domain.ideas.idea import Idea
from ..errors import SentinelAssessmentError
if TYPE_CHECKING:
    from ..service import SentinelMechanism
logger = logging.getLogger(__name__)


class AnalysisHelper:
    def __init__(self, sentinel: 'SentinelMechanism'):
        self.sentinel = sentinel

    async def analyze(self, context: NireonExecutionContext) -> AnalysisResult:
        if self.sentinel.idea_to_assess_buffer is None:
            base_metrics = {'buffered_items': 0, 'status': 'idle'}
            return AnalysisResult(success=True, component_id=self.sentinel.component_id, message='Sentinel is idle. No ideas in assessment buffer.', metrics=base_metrics, confidence=0.9)
        idea_to_assess, reference_ideas, original_ctx = self.sentinel.idea_to_assess_buffer
        self.sentinel.idea_to_assess_buffer = None
        current_context = original_ctx if original_ctx and hasattr(original_ctx, 'logger') else context
        current_context.logger.info(f'[{self.sentinel.component_id}] Analyzing buffered idea: {idea_to_assess.idea_id}')
        try:
            if self.sentinel.idea_service is None:
                await self.sentinel._init_helper.late_initialize(current_context)
                if self.sentinel.idea_service is None:
                    raise SentinelAssessmentError('IdeaService unavailable for analysis phase after late initialization attempt.')
            assess_obj = await self.sentinel.assessment_core.perform_assessment(idea_to_assess, reference_ideas, current_context)
            analysis_result, _ = await self.prepare_analysis_result(assess_obj, idea_to_assess, current_context)
            return analysis_result
        except SentinelAssessmentError as e:
            current_context.logger.error(f'[{self.sentinel.component_id}] Assessment failed during analysis: {e}', exc_info=True)
            self.sentinel._error_count += 1
            return AnalysisResult(success=False, component_id=self.sentinel.component_id, message=str(e), insights=[f'Assessment failed in analysis phase: {e}'], metrics={'idea_id': idea_to_assess.idea_id})
        except Exception as e:
            current_context.logger.critical(f'[{self.sentinel.component_id}] Critical error during analysis: {e}', exc_info=True)
            self.sentinel._error_count += 1
            return AnalysisResult(success=False, component_id=self.sentinel.component_id, message=f'Critical error during analysis: {str(e)}', insights=[f'Critical failure during analysis: {e}'], metrics={'idea_id': idea_to_assess.idea_id})

    async def prepare_analysis_result(self, assess_obj: IdeaAssessment, idea: Idea, ctx: NireonExecutionContext) -> Tuple[AnalysisResult, dict]:
        metrics = {'trust_score': assess_obj.trust_score, 'is_stable': assess_obj.is_stable, 'idea_id': idea.idea_id, 'rejection_reason_summary': assess_obj.rejection_reason[:200] if assess_obj.rejection_reason else None}
        for score in assess_obj.axis_scores or []:
            metrics[f'axis_{score.name}_score'] = score.score
        msg = f"Assessed idea '{idea.idea_id}': Trust={assess_obj.trust_score:.2f}, Stable={assess_obj.is_stable}. Reason: {assess_obj.rejection_reason or 'N/A'}"
        
        # REMOVED: Direct event publishing
        # The Reactor will handle signal promotion based on the ProcessResult
        
        full_assessment_data = assess_obj.model_dump()
        ctx.set_custom_data('last_sentinel_assessment_object', full_assessment_data)
        analysis_result = AnalysisResult(success=True, component_id=self.sentinel.component_id, metrics=metrics, confidence=min(1.0, max(0.0, assess_obj.trust_score / 10.0)), message=msg, insights=[f"Idea is {('stable' if assess_obj.is_stable else 'unstable')}"])
        return (analysis_result, full_assessment_data)