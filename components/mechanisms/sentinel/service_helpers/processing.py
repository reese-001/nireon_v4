from __future__ import annotations
import asyncio
import logging
import time
from typing import Any, Optional, Sequence, TYPE_CHECKING

from core.results import ProcessResult
from domain.context import NireonExecutionContext
from domain.ideas.idea import Idea
from ..errors import SentinelAssessmentError

if TYPE_CHECKING:
    from ..service import SentinelMechanism

logger = logging.getLogger(__name__)

class ProcessingHelper:
    def __init__(self, sentinel: 'SentinelMechanism') -> None:
        self.sentinel = sentinel

    async def process_impl(self, data: Any, context: NireonExecutionContext) -> ProcessResult:
        if not self.sentinel.is_initialized:
            await self.sentinel._init_helper.initialize_impl(context)
            if not self.sentinel.is_initialized:
                return self._err('Component failed to initialize.', 'INITIALIZATION_FAILURE')

        if not isinstance(data, dict) or 'target_idea_id' not in data:
            return self._err("Input data must be a dict with a 'target_idea_id' key.", 'INVALID_INPUT')

        # --- START: THE FIX ---
        # The Explorer passes the frame_id in the signal payload. The reactor rule then maps it
        # to the input_data for this component. We need to ensure this frame_id is used.
        frame_id_from_data = data.get('frame_id')
        if frame_id_from_data:
            # Create a new context scoped to the correct frame
            assessment_context = context.with_metadata(current_frame_id=frame_id_from_data)
            logger.debug(f"[{self.sentinel.component_id}] Context updated with frame_id from data: {frame_id_from_data}")
        else:
            assessment_context = context
            logger.warning(f"[{self.sentinel.component_id}] No 'frame_id' in input data for Sentinel. Assessment may fail if a frame is required.")
        # --- END: THE FIX ---

        target_idea_id = data.get('target_idea_id') or ''
        if not target_idea_id:
            return self._err("'target_idea_id' cannot be empty.", 'MISSING_TARGET_IDEA_ID')

        idea_to_assess = await self._fetch_idea_with_backoff(target_idea_id, assessment_context)
        if idea_to_assess is None:
            return self._err(f"Idea with ID '{target_idea_id}' not found.", 'IDEA_NOT_FOUND')

        reference_ideas = self._get_reference_ideas(data)

        try:
            # Pass the corrected context to the assessment core
            assessment = await self.sentinel.assessment_core.perform_assessment(idea_to_assess, reference_ideas, assessment_context)
            trust_score = assessment.trust_score if assessment.trust_score is not None else 0.0
            analysis_result, full_data = await self.sentinel._analysis_helper.prepare_analysis_result(assessment, idea_to_assess, assessment_context)

            return ProcessResult(
                success=True,
                component_id=self.sentinel.component_id,
                message=f"Assessment for idea '{idea_to_assess.idea_id}' complete. Trust={trust_score:.2f}, Stable={assessment.is_stable}.",
                output_data=full_data,
                performance_metrics=analysis_result.metrics
            )
        except SentinelAssessmentError as exc:
            logger.error('[%s] Assessment failed for %s: %s', self.sentinel.component_id, target_idea_id, exc, exc_info=True)
            return self._err(f'Assessment failed: {exc}', 'ASSESSMENT_ERROR')
        except Exception as exc:
            logger.exception('[%s] Unexpected error during processing', self.sentinel.component_id)
            return self._err(f'Unexpected processing error: {exc}', 'UNEXPECTED_PROCESS_ERROR')

    async def _fetch_idea_with_backoff(self, idea_id: str, context: NireonExecutionContext, *, retries: int = 3, delay: float = 0.2) -> Optional[Idea]:
        if not self.sentinel.idea_service:
            context.logger.error('IdeaService is not available.')
            return None

        for attempt in range(retries + 1):
            idea = self.sentinel.idea_service.get_idea(idea_id)
            if idea:
                return idea
            if attempt < retries:
                await asyncio.sleep(delay)
        return None

    @staticmethod
    def _get_reference_ideas(data: dict) -> Optional[Sequence[Idea]]:
        return data.get('reference_ideas')

    def _err(self, message: str, code: str) -> ProcessResult:
        logger.error('[%s] %s (%s)', self.sentinel.component_id, message, code)
        self.sentinel._error_count += 1
        return ProcessResult(success=False, component_id=self.sentinel.component_id, message=message, error_code=code)
