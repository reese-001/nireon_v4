"""
Handles the analyze() lifecycle phase for the SentinelMechanism.
This is where the buffered idea is actually assessed.
"""
from __future__ import annotations
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING

# V4 CHANGE: Flattened import paths
from domain.context import NireonExecutionContext
from core.results import AnalysisResult
from domain.evaluation.assessment import IdeaAssessment
from domain.ideas.idea import Idea
from ..errors import SentinelAssessmentError

if TYPE_CHECKING:
    from ..service import SentinelMechanism

logger = logging.getLogger(__name__)


class AnalysisHelper:
    """A helper class to encapsulate analysis logic for Sentinel."""
    def __init__(self, sentinel: 'SentinelMechanism'):
        self.sentinel = sentinel

    async def analyze(self, context: NireonExecutionContext) -> AnalysisResult:
        """
        Pulls a buffered idea and performs assessment. This is the core of the
        two-phase (process/buffer -> analyze/assess) pattern for Sentinel.
        """
        if self.sentinel.idea_to_assess_buffer is None:
            # If there's no buffered idea, perform a basic health/status analysis.
            # In V4, we can provide a more meaningful base analysis.
            base_metrics = {'buffered_items': 0, 'status': 'idle'}
            return AnalysisResult(
                success=True,
                component_id=self.sentinel.component_id,
                message="Sentinel is idle. No ideas in assessment buffer.",
                metrics=base_metrics,
                confidence=0.9
            )

        # Unpack the buffered data
        idea_to_assess, reference_ideas, original_ctx = self.sentinel.idea_to_assess_buffer
        self.sentinel.idea_to_assess_buffer = None  # Clear buffer immediately

        # Use the context from when the process was called, as it's more relevant
        current_context = (
            original_ctx if original_ctx and hasattr(original_ctx, 'logger') else context
        )
        current_context.logger.info(
            f'[{self.sentinel.component_id}] Analyzing buffered idea: {idea_to_assess.idea_id}'
        )

        try:
            # Ensure idea_service is available, attempt late init if needed
            if self.sentinel.idea_service is None:
                await self.sentinel._init_helper.late_initialize(current_context)
                if self.sentinel.idea_service is None:
                    raise SentinelAssessmentError(
                        'IdeaService unavailable for analysis phase after late initialization attempt.'
                    )

            assess_obj = await self.sentinel.assessment_core.perform_assessment(
                idea_to_assess, reference_ideas, current_context
            )

            # This now needs to be awaited
            return await self.prepare_analysis_result(assess_obj, idea_to_assess, current_context)

        except SentinelAssessmentError as e:
            current_context.logger.error(
                f'[{self.sentinel.component_id}] Assessment failed during analysis: {e}',
                exc_info=True
            )
            self.sentinel._error_count += 1
            return AnalysisResult(
                success=False,
                component_id=self.sentinel.component_id,
                message=str(e),
                insights=[f"Assessment failed in analysis phase: {e}"],
                metrics={'idea_id': idea_to_assess.idea_id}
            )

        except Exception as e:
            current_context.logger.critical(
                f'[{self.sentinel.component_id}] Critical error during analysis: {e}',
                exc_info=True
            )
            self.sentinel._error_count += 1
            return AnalysisResult(
                success=False,
                component_id=self.sentinel.component_id,
                message=f'Critical error during analysis: {str(e)}',
                insights=[f"Critical failure during analysis: {e}"],
                metrics={'idea_id': idea_to_assess.idea_id}
            )

    async def prepare_analysis_result(self, assess_obj: IdeaAssessment, idea: Idea, ctx: NireonExecutionContext) -> tuple[AnalysisResult, dict]:
        metrics = {
            'trust_score': assess_obj.trust_score,
            'is_stable': assess_obj.is_stable,
            'idea_id': idea.idea_id,
            'rejection_reason_summary': assess_obj.rejection_reason[:200] if assess_obj.rejection_reason else None
        }
        for score in assess_obj.axis_scores or []:
            metrics[f'axis_{score.name}_score'] = score.score
            
        msg = f"Assessed idea '{idea.idea_id}': Trust={assess_obj.trust_score:.2f}, Stable={assess_obj.is_stable}. Reason: {assess_obj.rejection_reason or 'N/A'}"
        
        if self.sentinel.event_bus:
            # This is now a synchronous call
            self.sentinel._event_publisher.publish_assessment_event(assess_obj, idea, ctx)
            
        full_assessment_data = assess_obj.model_dump()
        ctx.set_custom_data('last_sentinel_assessment_object', full_assessment_data)
        
        analysis_result = AnalysisResult(
            success=True,
            component_id=self.sentinel.component_id,
            metrics=metrics,
            confidence=min(1.0, max(0.0, assess_obj.trust_score / 10.0)),
            message=msg,
            insights=[f"Idea is {('stable' if assess_obj.is_stable else 'unstable')}"]
        )
        # Return both objects for the calling function to use
        return analysis_result, full_assessment_data