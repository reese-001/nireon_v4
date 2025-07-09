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
    MAX_REF_IDS: int = 20
    BACKOFF_DELAY: float = 0.2
    MAX_BACKOFF_RETRIES: int = 3

    def __init__(self, sentinel: 'SentinelMechanism') -> None:
        self.sentinel = sentinel

    async def process_impl(self, data: Any, context: NireonExecutionContext) -> ProcessResult:
        if not self.sentinel.is_initialized:
            await self.sentinel._init_helper.initialize_impl(context)
            if not self.sentinel.is_initialized:
                return self._error('Component failed to initialize.', 'INITIALIZATION_FAILURE')
        try:
            original_data = data.get('original_data', data) if isinstance(data, dict) else data
        except Exception as e:
            logger.error(f'[{self.sentinel.component_id}] Error unwrapping data: {e}')
            original_data = data
        if not isinstance(original_data, dict) or 'target_idea_id' not in original_data:
            return self._error("Input data must be a dict with a 'target_idea_id' key.", 'INVALID_INPUT')
        frame_id = original_data.get('frame_id')
        ctx = context.with_metadata(current_frame_id=frame_id) if frame_id else context
        target_idea_id: str = str(original_data.get('target_idea_id', ''))
        objective: Optional[str] = original_data.get('objective')
        if not target_idea_id:
            return self._error("'target_idea_id' cannot be empty.", 'MISSING_TARGET_IDEA_ID')
        idea_to_assess = await self._fetch_idea_with_backoff(target_idea_id, ctx)
        if idea_to_assess is None:
            return self._error(f'Idea with ID {target_idea_id!r} not found.', 'IDEA_NOT_FOUND')
        ref_ideas: Sequence[Idea] | None = original_data.get('reference_ideas')
        if not ref_ideas:
            ref_ideas = await self._gather_reference_ideas(idea_to_assess, ctx)
        try:
            assessment = await self.sentinel.assessment_core.perform_assessment(idea_to_assess, ref_ideas, ctx, objective)
            if not assessment:
                return self._error('Assessment returned None', 'NULL_ASSESSMENT')
            try:
                trust_score_str = f'{assessment.trust_score:.2f}' if assessment.trust_score is not None else 'None'
                ctx.logger.info(f'[{self.sentinel.component_id}] Assessment created - idea_id: {assessment.idea_id}, trust_score: {trust_score_str}, is_stable: {assessment.is_stable}, rejection_reason: {assessment.rejection_reason}')
            except Exception as e:
                ctx.logger.warning(f'[{self.sentinel.component_id}] Error logging assessment: {e}')
            if assessment.is_stable is None:
                ctx.logger.warning(f'[{self.sentinel.component_id}] is_stable is None, computing from thresholds...')
                try:
                    trust_ok = assessment.trust_score is not None and assessment.trust_score >= self.sentinel.trust_th
                    axes_ok = True
                    if assessment.axis_scores:
                        for axis in assessment.axis_scores:
                            if axis.score is None or axis.score < self.sentinel.min_axis:
                                axes_ok = False
                                break
                    assessment.is_stable = trust_ok and axes_ok
                except Exception as e:
                    ctx.logger.error(f'[{self.sentinel.component_id}] Error computing is_stable: {e}')
                    assessment.is_stable = False
                ctx.logger.info(f'[{self.sentinel.component_id}] Computed is_stable = {assessment.is_stable}')
            
            # REMOVED: Direct event publishing
            # The Reactor will handle signal promotion based on our ProcessResult
            
            # Prepare the full assessment data
            axis_scores_dict = {}
            if assessment.axis_scores:
                for axis in assessment.axis_scores:
                    axis_scores_dict[axis.name] = {
                        'score': axis.score,
                        'explanation': axis.explanation
                    }
            
            # Get assessment metadata
            assessment_metadata = assessment.metadata if hasattr(assessment, 'metadata') else {}
            
            # Build comprehensive output data
            full_data = {
                'type': 'trust_assessment',
                'idea_id': assessment.idea_id,
                'idea_text': idea_to_assess.text,
                'trust_score': assessment.trust_score,
                'is_stable': assessment.is_stable,
                'rejection_reason': assessment.rejection_reason,
                'axis_scores': axis_scores_dict,
                'assessment_details': {
                    'trust_score': assessment.trust_score,
                    'axis_scores': axis_scores_dict,
                    'is_stable': assessment.is_stable,
                    'idea_parent_id': getattr(idea_to_assess, 'parent_id', None),
                    'metadata': {
                        'objective': objective,
                        'assessment_timestamp': time.time(),
                        **assessment_metadata
                    }
                },
                'metadata': {
                    'objective': objective,
                    'assessment_timestamp': time.time(),
                    **assessment_metadata
                }
            }
            
            try:
                trust_str = f'{assessment.trust_score:.2f}' if assessment.trust_score is not None else 'N/A'
                msg = f"Assessment for idea '{idea_to_assess.idea_id}' complete. Trust={trust_str}, Stable={assessment.is_stable}."
            except:
                msg = f"Assessment for idea '{idea_to_assess.idea_id}' complete."
            
            return ProcessResult(
                success=True, 
                component_id=self.sentinel.component_id, 
                message=msg, 
                output_data=full_data
            )
        except SentinelAssessmentError as exc:
            logger.error(f'[{self.sentinel.component_id}] Assessment failed for {target_idea_id}: {exc}', exc_info=True)
            return self._error(f'Assessment failed: {exc}', 'ASSESSMENT_ERROR')
        except Exception as exc:
            logger.exception(f'[{self.sentinel.component_id}] Unexpected processing error')
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

    async def _gather_reference_ideas(self, idea: Idea, ctx: NireonExecutionContext) -> List[Idea]:
        service = self.sentinel.idea_service
        if service is None:
            ctx.logger.warning(f'[{self.sentinel.component_id}] IdeaService unavailable; novelty will use empty references.')
            return []
        ref_candidates: List[Idea] = []
        try:
            parent_ids = getattr(idea, 'parent_ids', []) or []
            for pid in parent_ids:
                p = service.get_idea(pid)
                if p:
                    ref_candidates.append(p)
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
            ctx.logger.warning(f'[{self.sentinel.component_id}] Failed to gather reference ideas due to data error: {exc}', exc_info=True)
            return []

    def _error(self, message: str, code: str) -> ProcessResult:
        logger.error('[%s] %s (%s)', self.sentinel.component_id, message, code)
        self.sentinel._error_count += 1
        return ProcessResult(success=False, component_id=self.sentinel.component_id, message=message, error_code=code)