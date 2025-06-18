# nireon_v4/components/mechanisms/sentinel/service_helpers/processing.py
from __future__ import annotations
import asyncio
import logging
import time
from typing import Any, Optional, Sequence, TYPE_CHECKING, List

from core.results import ProcessResult
from domain.context import NireonExecutionContext
from domain.ideas.idea import Idea
from ..errors import SentinelAssessmentError

if TYPE_CHECKING:
    from ..service import SentinelMechanism

logger = logging.getLogger(__name__)


class ProcessingHelper:
    """Handles the main Sentinel .process() path."""

    MAX_REF_IDS: int = 20
    BACKOFF_DELAY: float = 0.2
    MAX_BACKOFF_RETRIES: int = 3

    def __init__(self, sentinel: 'SentinelMechanism') -> None:
        self.sentinel = sentinel

    async def process_impl(self, data: Any,
                           context: NireonExecutionContext) -> ProcessResult:
        if not self.sentinel.is_initialized:
            await self.sentinel._init_helper.initialize_impl(context)
            if not self.sentinel.is_initialized:
                return self._error('Component failed to initialize.', 'INITIALIZATION_FAILURE')

        if not isinstance(data, dict) or 'target_idea_id' not in data:
            return self._error("Input data must be a dict with a 'target_idea_id' key.", 'INVALID_INPUT')

        frame_id = data.get('frame_id')
        ctx = context.with_metadata(current_frame_id=frame_id) if frame_id else context

        target_idea_id: str = data.get('target_idea_id') or ''
        if not target_idea_id:
            return self._error("'target_idea_id' cannot be empty.", 'MISSING_TARGET_IDEA_ID')

        idea_to_assess = await self._fetch_idea_with_backoff(target_idea_id, ctx)
        if idea_to_assess is None:
            return self._error(f'Idea with ID {target_idea_id!r} not found.', 'IDEA_NOT_FOUND')

        ref_ideas: Sequence[Idea] | None = data.get('reference_ideas')
        if not ref_ideas:
            ref_ideas = await self._gather_reference_ideas(idea_to_assess, ctx)

        try:
            assessment = await self.sentinel.assessment_core.perform_assessment(
                idea_to_assess, ref_ideas, ctx
            )
            analysis_result, full_data = (
                await self.sentinel._analysis_helper.prepare_analysis_result(
                    assessment, idea_to_assess, ctx
                )
            )
            msg = (f"Assessment for idea '{idea_to_assess.idea_id}' complete. "
                   f"Trust={assessment.trust_score:.2f}, Stable={assessment.is_stable}.")
            return ProcessResult(success=True,
                                 component_id=self.sentinel.component_id,
                                 message=msg,
                                 output_data=full_data,
                                 performance_metrics=analysis_result.metrics)
        except SentinelAssessmentError as exc:
            logger.error('[%s] Assessment failed for %s: %s',
                         self.sentinel.component_id, target_idea_id, exc,
                         exc_info=True)
            return self._error(f'Assessment failed: {exc}', 'ASSESSMENT_ERROR')
        except Exception as exc:
            logger.exception('[%s] Unexpected processing error', self.sentinel.component_id)
            return self._error(f'Unexpected processing error: {exc}', 'UNEXPECTED_PROCESS_ERROR')

    async def _fetch_idea_with_backoff(self, idea_id: str, ctx: NireonExecutionContext) -> Optional[Idea]:
        if self.sentinel.idea_service is None:
            ctx.logger.error('IdeaService is not available.')
            return None

        for attempt in range(self.MAX_BACKOFF_RETRIES + 1):
            idea = self.sentinel.idea_service.get_idea(idea_id)
            if idea:
                return idea
            if attempt < self.MAX_BACKOFF_RETRIES:
                await asyncio.sleep(self.BACKOFF_DELAY)
        return None

    async def _gather_reference_ideas(self, idea: Idea,
                                      ctx: NireonExecutionContext
                                      ) -> List[Idea]:
        service = self.sentinel.idea_service
        if service is None:
            ctx.logger.warning(f'[{self.sentinel.component_id}] IdeaService unavailable; '
                               'novelty will use empty references.')
            return []

        ref_candidates: List[Idea] = []
        try:
            parent_ids = getattr(idea, 'parent_ids', []) or []
            for pid in parent_ids:
                p = service.get_idea(pid)
                if p:
                    ref_candidates.append(p)
                    # +++ FIX #2: Changed get_child_ideas to get_children +++
                    for sib in service.get_children(pid) or []:
                        if sib.idea_id != idea.idea_id:
                            ref_candidates.append(sib)

            seen: set[str] = set()
            unique_refs = []
            max_refs = self.sentinel.sentinel_cfg.max_reference_ideas
            for r in ref_candidates:
                if r.idea_id not in seen:
                    unique_refs.append(r)
                    seen.add(r.idea_id)
                if len(unique_refs) >= max_refs:
                    break
            return unique_refs
        except (AttributeError, TypeError, ValueError) as exc:
            ctx.logger.warning(f'[{self.sentinel.component_id}] '
                               f'Failed to gather reference ideas due to data error: {exc}', exc_info=True)
            return []

    def _error(self, message: str, code: str) -> ProcessResult:
        logger.error('[%s] %s (%s)', self.sentinel.component_id, message, code)
        self.sentinel._error_count += 1
        return ProcessResult(success=False,
                             component_id=self.sentinel.component_id,
                             message=message,
                             error_code=code)