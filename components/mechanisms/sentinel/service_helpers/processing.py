from __future__ import annotations
import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence, Union

from domain.context import NireonExecutionContext
from core.results import ProcessResult
from domain.ideas.idea import Idea
from ..errors import SentinelAssessmentError

if TYPE_CHECKING:
    from ..service import SentinelMechanism

logger = logging.getLogger(__name__)


class ProcessingHelper:
    def __init__(self, sentinel: 'SentinelMechanism'):
        self.sentinel = sentinel

    async def process_impl(self, data: Any, context: NireonExecutionContext) -> ProcessResult:
        if not self.sentinel.is_initialized:
            await self.sentinel._init_helper.initialize_impl(context)
            if not self.sentinel.is_initialized:
                return self.create_error_result('Component failed to initialize.', 'INITIALIZATION_FAILURE')

        if not isinstance(data, dict) or 'target_idea_id' not in data:
            return self.create_error_result("Input data must be a dict with a 'target_idea_id' key.", 'INVALID_INPUT')

        target_idea_id = data.get('target_idea_id')
        if not target_idea_id:
            return self.create_error_result("'target_idea_id' cannot be empty.", 'MISSING_TARGET_IDEA_ID')

        idea_to_assess = await self._fetch_idea(target_idea_id, context)

        if not isinstance(idea_to_assess, Idea):
            return self.create_error_result(f"Idea with ID '{target_idea_id}' not found.", 'IDEA_NOT_FOUND')

        reference_ideas = self._get_reference_ideas(data)

        try:
            assessment = await self.sentinel.assessment_core.perform_assessment(
                idea_to_assess, reference_ideas, context
            )
            # FIX: Get both the analysis result and the full assessment data
            analysis_result, full_assessment_data = await self.sentinel._analysis_helper.prepare_analysis_result(assessment, idea_to_assess, context)
            return ProcessResult(
                success=True,
                component_id=self.sentinel.component_id,
                message=f"Assessment for idea '{idea_to_assess.idea_id}' complete. Trust={assessment.trust_score:.2f}, Stable={assessment.is_stable}.",
                # FIX: Use the full data as output
                output_data=full_assessment_data,
                performance_metrics=analysis_result.metrics
            )
        
        except SentinelAssessmentError as e:
            logger.error(f'[{self.sentinel.component_id}] Assessment failed for {target_idea_id}: {e}', exc_info=True)
            return self.create_error_result(f'Assessment failed: {e}', 'ASSESSMENT_ERROR')
        except Exception as e:
            logger.error(f'[{self.sentinel.component_id}] Unexpected error during processing: {e}', exc_info=True)
            return self.create_error_result(f'Unexpected processing error: {e}', 'UNEXPECTED_PROCESS_ERROR')


    async def _fetch_idea(self, idea_id: str, context: NireonExecutionContext) -> Optional[Idea]:
        if not self.sentinel.idea_service:
            context.logger.error('IdeaService is not available.')
            return None
        # The 'get_idea' method is synchronous, so it should not be awaited.
        return self.sentinel.idea_service.get_idea(idea_id)

    def _get_reference_ideas(self, data: dict) -> Optional[Sequence[Idea]]:
        return data.get('reference_ideas')

    def create_error_result(self, message: str, error_code: str) -> ProcessResult:
        logger.error(f'[{self.sentinel.component_id}] Error: {message} ({error_code})')
        self.sentinel._error_count += 1
        return ProcessResult(
            success=False,
            component_id=self.sentinel.component_id,
            message=message,
            error_code=error_code
        )