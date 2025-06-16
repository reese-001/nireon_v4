import logging
import json
import re
from datetime import datetime, timezone
from typing import Optional, Sequence, Tuple, TYPE_CHECKING
import numpy as np

from domain.context import NireonExecutionContext
from domain.epistemic_stage import EpistemicStage
from domain.evaluation.assessment import AxisScore, IdeaAssessment
from domain.ideas.idea import Idea
from domain.cognitive_events import CognitiveEvent, LLMRequestPayload
from domain.ports.llm_port import LLMResponse
from .errors import SentinelAssessmentError, SentinelLLMParsingError
from .constants import DEFAULT_LLM_SCORE

if TYPE_CHECKING:
    from .service import SentinelMechanism

logger = logging.getLogger(__name__)

class AssessmentCore:
    def __init__(self, sentinel: 'SentinelMechanism'):
        self.sentinel = sentinel

    @staticmethod
    def _get_explanation_snippet(explanation: Optional[str], length: int = 70) -> str:
        if not explanation:
            return 'No explanation provided.'
        return explanation[:length] + '...' if len(explanation) > length else explanation

    async def perform_assessment(
        self, idea: Idea, refs: Optional[Sequence[Idea]], ctx: NireonExecutionContext
    ) -> IdeaAssessment:
        ctx.logger.debug(f'[{self.sentinel.component_id}] Performing assessment for idea {idea.idea_id}')

        # FIX: The core issue is that Sentinel was trying to spawn a new frame from a parent
        # that the Explorer had already closed. The correct behavior is for Sentinel to operate
        # within the *existing* frame context provided by the triggering signal.
        # We will now use the context directly instead of creating a new frame.

        # The `assessment_context` is now just the context we were given, which already
        # has the correct frame_id from the triggering `IdeaGeneratedSignal`.
        assessment_ctx = ctx
        assessment_frame_id = assessment_ctx.metadata.get('current_frame_id')

        if not assessment_frame_id:
            raise SentinelAssessmentError(f"Cannot perform assessment for idea '{idea.idea_id}' without a frame_id in the context.")

        ctx.logger.info(f"[{self.sentinel.component_id}] Performing assessment within existing frame '{assessment_frame_id}'.")

        stage_enum = self._determine_stage(idea)
        assessment_params = self.sentinel.stage_evaluation_service.get_assessment_parameters(stage_enum, assessment_ctx)
        self._update_assessment_parameters(assessment_params)

        objective = self.sentinel.sentinel_cfg.objective_override or assessment_ctx.get_custom_data('objective', 'Generate and refine ideas')

        llm_prompt = self.sentinel.stage_evaluation_service.build_stage_assessment_prompt(
            idea_text=idea.text,
            objective=objective,
            stage=stage_enum or EpistemicStage.DEFAULT,
            context=assessment_ctx
        )

        align_score, feas_score, llm_explanation = await self._get_llm_scores_via_gateway(llm_prompt, assessment_ctx)

        nov_score, nov_reason = await self.sentinel.novelty_calculator.calculate_novelty(idea, refs or [])

        axes = [
            AxisScore(name='align', score=align_score, explanation=llm_explanation),
            AxisScore(name='feas', score=feas_score, explanation=llm_explanation),
            AxisScore(name='novel', score=nov_score, explanation=nov_reason),
        ]

        trust = self._calculate_trust_score(axes)
        final_trust, length_penalty = self.sentinel.scoring_adjustments.apply_length_penalty(
            trust, len(idea.text), assessment_ctx
        )

        progression_bonus = 0.0
        has_progressive = False
        if assessment_ctx.is_flag_enabled('sentinel_enable_progression_adjustment', self.sentinel.sentinel_cfg.enable_progression_adjustment):
            final_trust, progression_bonus, has_progressive = self.sentinel.scoring_adjustments.apply_progression_adjustment(
                final_trust, idea.text, getattr(idea, 'step', -1), assessment_ctx
            )

        is_stable = self._determine_stability(final_trust, align_score, feas_score, nov_score)
        rejection_reason = None if is_stable else self._generate_rejection_reason(is_stable, align_score, feas_score, nov_score, final_trust, llm_explanation)

        # FIX: Removed the call to update the frame status, as we are no longer creating
        # and managing a sub-frame for this assessment.
        ctx.logger.info(f"[{self.sentinel.component_id}] Assessment complete for idea {idea.idea_id} in frame '{assessment_frame_id}'.")

        return self._build_assessment(
            idea, final_trust, is_stable, rejection_reason, axes, stage_enum,
            trust, length_penalty, progression_bonus, has_progressive
        )

    async def _get_llm_scores_via_gateway(self, prompt: str, ctx: NireonExecutionContext) -> Tuple[float, float, str]:
        if not self.sentinel.gateway:
            raise SentinelAssessmentError('MechanismGateway is not available to Sentinel.')

        frame_id = ctx.metadata.get('current_frame_id')
        if not frame_id:
            raise SentinelAssessmentError("Context for LLM score assessment is missing 'current_frame_id'.")

        llm_payload = LLMRequestPayload(
            prompt=prompt,
            stage=EpistemicStage.CRITIQUE,
            role='sentinel_evaluator',
            llm_settings={'temperature': 0.2, 'max_tokens': 512, 'route': 'sentinel_axis_scorer'}
        )
        cognitive_event = CognitiveEvent(
            frame_id=frame_id,
            owning_agent_id=self.sentinel.component_id,
            service_call_type='LLM_ASK',
            payload=llm_payload,
            epistemic_intent='ASSESS_IDEA_AXES'
        )

        try:
            llm_response: LLMResponse = await self.sentinel.gateway.process_cognitive_event(cognitive_event, ctx)
            if llm_response.get('error'):
                error_message = f"Gateway returned an error for LLM_ASK: {llm_response.get('error')}"
                ctx.logger.error(f'[{self.sentinel.component_id}] {error_message}')
                return (DEFAULT_LLM_SCORE, DEFAULT_LLM_SCORE, error_message)

            raw_resp_text = llm_response.text
            return self._parse_llm_response_for_scores(raw_resp_text, ctx)
        except Exception as e:
            ctx.logger.critical(f'[{self.sentinel.component_id}] Critical error calling MechanismGateway: {e}', exc_info=True)
            raise SentinelAssessmentError(f'Unexpected error during Gateway call: {e}') from e

    def _parse_llm_response_for_scores(self, raw_resp: str, ctx: NireonExecutionContext) -> Tuple[float, float, str]:
        if not raw_resp:
            ctx.logger.warning(f'[{self.sentinel.component_id}] Empty LLM response for scores.')
            raise SentinelLLMParsingError('Empty LLM response.')

        try:
            start = raw_resp.find('{')
            end = raw_resp.rfind('}')
            if start != -1 and end > start:
                data = json.loads(raw_resp[start:end + 1])
                align_val = float(data.get('align_score', DEFAULT_LLM_SCORE))
                feas_val = float(data.get('feas_score', DEFAULT_LLM_SCORE))
                explanation = str(data.get('explanation', 'No LLM explanation provided.'))
                ctx.logger.debug(f'[{self.sentinel.component_id}] JSON parse ok (align={align_val}, feas={feas_val})')
                return self._clamp_score(align_val), self._clamp_score(feas_val), explanation
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            ctx.logger.debug(f'[{self.sentinel.component_id}] Could not parse LLM response as JSON ({e}), trying regex.')

        try:
            al_match = re.search('align(?:_score)?\\s*[:=]?\\s*["\\\']?(\\d+\\.?\\d*)["\\\']?', raw_resp, re.I)
            fe_match = re.search('feas(?:_score)?\\s*[:=]?\\s*["\\\']?(\\d+\\.?\\d*)["\\\']?', raw_resp, re.I)
            ex_match = re.search('expl(?:anation)?\\s*[:=]?\\s*["\\\'](.*?)["\\\']', raw_resp, re.I | re.S)

            align_val = float(al_match.group(1)) if al_match else DEFAULT_LLM_SCORE
            feas_val = float(fe_match.group(1)) if fe_match else DEFAULT_LLM_SCORE
            explanation = ex_match.group(1).strip() if ex_match else 'No explanation found via regex.'

            ctx.logger.debug(f'[{self.sentinel.component_id}] Regex parse ok (align={align_val}, feas={feas_val})')
            return self._clamp_score(align_val), self._clamp_score(feas_val), explanation
        except Exception as e:
            ctx.logger.warning(f'[{self.sentinel.component_id}] Regex parsing also failed: {e}. Defaulting scores.')

        raise SentinelLLMParsingError(f"Cannot parse LLM response. Raw response (first 200 chars): {raw_resp[:200]}")

    def _clamp_score(self, score: float) -> float:
        return max(1.0, min(10.0, score))

    def _determine_stage(self, idea: Idea) -> Optional[EpistemicStage]:
        stage_name = idea.metadata.get('stage') if idea.metadata else None
        if isinstance(stage_name, str):
            return EpistemicStage.__members__.get(stage_name.upper())
        return None

    def _update_assessment_parameters(self, params: dict) -> None:
        self.sentinel.weights = np.asarray([params['alignment_weight'], params['feasibility_weight'], params['novelty_weight']], dtype=float)
        self.sentinel.trust_th = params.get('trust_threshold', self.sentinel.trust_th)
        self.sentinel.min_axis = params['min_axis_score']

    def _calculate_trust_score(self, axes: list[AxisScore]) -> float:
        scores = np.array([a.score for a in axes])
        trust = float(np.dot(self.sentinel.weights, scores))
        return max(1.0, min(10.0, trust))

    def _determine_stability(self, trust: float, align: float, feas: float, nov: float) -> bool:
        return trust >= self.sentinel.trust_th and align >= self.sentinel.min_axis and feas >= self.sentinel.min_axis and nov >= self.sentinel.min_axis

    def _generate_rejection_reason(self, is_stable: bool, al: float, fe: float, nov: float, trust: float, llm_expl: str) -> str:
        expl_snippet = self._get_explanation_snippet(llm_expl)
        if is_stable:
            return f'Stable (Trust:{trust:.1f} Align:{al:.1f} Feas:{fe:.1f} Novel:{nov:.1f}). LLM: "{expl_snippet}"'

        reasons = []
        if trust < self.sentinel.trust_th:
            reasons.append(f'Low Trust ({trust:.1f} < {self.sentinel.trust_th:.1f})')
        if al < self.sentinel.min_axis:
            reasons.append(f'Low Alignment ({al:.1f} < {self.sentinel.min_axis:.1f})')
        if fe < self.sentinel.min_axis:
            reasons.append(f'Low Feasibility ({fe:.1f} < {self.sentinel.min_axis:.1f})')
        if nov < self.sentinel.min_axis:
            reasons.append(f'Low Novelty ({nov:.1f} < {self.sentinel.min_axis:.1f})')

        final_reason = '; '.join(reasons) or 'One or more assessment criteria failed.'
        return f'Unstable: {final_reason}. LLM: "{expl_snippet}"'

    def _build_assessment(self, idea: Idea, final_trust: float, is_stable: bool, rejection_reason: Optional[str],
                          axes: list[AxisScore], stage_enum: Optional[EpistemicStage],
                          original_trust: float, length_penalty: float, progression_bonus: float, has_progressive: bool) -> IdeaAssessment:
        llm_explanation_snippet = self._get_explanation_snippet(axes[0].explanation if axes else None)
        return IdeaAssessment(
            idea_id=idea.idea_id,
            trust_score=final_trust,
            is_stable=is_stable,
            rejection_reason=rejection_reason,
            axis_scores=axes,
            metadata={
                'step_created': idea.step,
                'stage_assessed': stage_enum.name if stage_enum else None,
                'method_created': idea.method,
                'original_trust_score': original_trust,
                'length_penalty_applied': length_penalty,
                'progression_bonus_applied': progression_bonus,
                'contains_progressive_elements': has_progressive,
                'assessment_weights_used': self.sentinel.weights.tolist(),
                'trust_threshold_used': self.sentinel.trust_th,
                'min_axis_score_used': self.sentinel.min_axis,
                'llm_explanation_snippet': llm_explanation_snippet,
                'assessment_timestamp': datetime.now(timezone.utc).isoformat()
            }
        )