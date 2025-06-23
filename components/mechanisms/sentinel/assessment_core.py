from __future__ import annotations
import json
import logging
import re
from datetime import datetime, timezone
from typing import Final, Optional, Sequence, Tuple, TYPE_CHECKING, Dict, Any
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

_JSON_BRACE_RE: Final = re.compile('\\{.*\\}', re.S)
_ALIGN_RE: Final = re.compile('align(?:_score)?\\s*[:=]?\\s*[\'\\"]?(\\d+(?:\\.\\d*)?)[\'\\"]?', re.I)
_FEAS_RE: Final = re.compile('feas(?:_score)?\\s*[:=]?\\s*[\'\\"]?(\\d+(?:\\.\\d*)?)[\'\\"]?', re.I)
_EXPL_RE: Final = re.compile('expla(?:nation)?\\s*[:=]?\\s*[\'\\"](.+?)[\'\\"]', re.I | re.S)

def _clamp(v: float, lo: float=1.0, hi: float=10.0) -> float:
    return float(np.clip(v, lo, hi))

def _snippet(text: Optional[str], length: int=70) -> str:
    if not text:
        return 'No explanation provided.'
    return f'{text[:length]}...' if len(text) > length else text

__all__ = ['AssessmentCore']

class AssessmentCore:
    def __init__(self, sentinel: 'SentinelMechanism') -> None:
        self.sentinel = sentinel

    async def perform_assessment(self, idea: Idea, refs: Optional[Sequence[Idea]], ctx: NireonExecutionContext, objective: Optional[str] = None) -> IdeaAssessment:
        ctx.logger.debug('[%s] Assessing idea %s', self.sentinel.component_id, idea.idea_id)
        ctx.logger.info('[DEPTH_DEBUG] Assessing idea %s metadata=%s', idea.idea_id, idea.metadata)
        parent_frame_id = ctx.metadata.get('current_frame_id')

        if not parent_frame_id:
            raise SentinelAssessmentError(f"Cannot assess idea '{idea.idea_id}' without a parent frame.")
        parent_frame = await self.sentinel.frame_factory.get_frame_by_id(ctx, parent_frame_id)
        if not parent_frame:
            raise SentinelAssessmentError(f"Parent frame '{parent_frame_id}' not found for assessment.")


        assessment_frame = await self.sentinel.frame_factory.create_frame(
            context=ctx,
            name=f"{parent_frame.name}_sentinel_eval_{idea.idea_id[:8]}", 
            owner_agent_id=self.sentinel.component_id,
            description=f"Assessment frame for idea '{idea.idea_id}'",
            parent_frame_id=parent_frame_id,
            llm_policy={'temperature': 0.2, 'route': 'sentinel_axis_scorer'},
            resource_budget={'llm_calls': 2}
        )
        ctx.logger.info('[%s] Spawned assessment sub-frame %s (parent=%s)', self.sentinel.component_id, assessment_frame.id, parent_frame_id)
        assessment_ctx = ctx.with_metadata(current_frame_id=assessment_frame.id)
        
        stage_enum = self._determine_stage(idea)
        
        base_params = self.sentinel.sentinel_cfg.model_dump()
        self._explode_weights(base_params)
        params = self.sentinel.stage_evaluation_service.get_assessment_parameters(stage_enum, assessment_ctx, base_params=base_params)
        self._update_assessment_parameters(params)
        
        final_objective = objective or self.sentinel.sentinel_cfg.objective_override or 'Generate and refine ideas'
        prompt = self.sentinel.stage_evaluation_service.build_stage_assessment_prompt(
            idea_text=idea.text,
            objective=final_objective,
            stage=stage_enum or EpistemicStage.DEFAULT,
            context=assessment_ctx
        )

        align_score, feas_score, llm_expl = await self._get_llm_scores(prompt, assessment_ctx)
        
        nov_score, nov_reason = await self.sentinel.novelty_calculator.calculate_novelty(idea, refs or [])
        
        axes = [
            AxisScore(name='align', score=align_score, explanation=llm_expl),
            AxisScore(name='feas', score=feas_score, explanation=llm_expl),
            AxisScore(name='novel', score=nov_score, explanation=nov_reason),
        ]
        
        trust_raw = self._calculate_trust_score(axes)
        trust_adj, length_pen = self.sentinel.scoring_adjustments.apply_length_penalty(trust_raw, len(idea.text), ctx)
        
        prog_bonus = 0.0
        has_prog = False
        if assessment_ctx.is_flag_enabled('sentinel_enable_progression_adjustment', self.sentinel.sentinel_cfg.enable_progression_adjustment):
            trust_adj, prog_bonus, has_prog = self.sentinel.scoring_adjustments.apply_progression_adjustment(
                trust_adj, idea.text, getattr(idea, 'step', -1), assessment_ctx
            )

        edge_support, edge_dist = self._edge_trust_context(idea)
        trust_final = self.sentinel.scoring_adjustments.apply_edge_trust_adjustment(trust_adj, edge_support, edge_dist, assessment_ctx)

        is_stable = self._determine_stability(trust_final, align_score, feas_score, nov_score)
        rejection_reason = None if is_stable else self._generate_rejection_reason(align_score, feas_score, nov_score, trust_final, llm_expl)
        
        await self.sentinel.frame_factory.update_frame_status(assessment_ctx, assessment_frame.id, 'completed_ok')
        ctx.logger.info('[%s] Assessment frame %s complete.', self.sentinel.component_id, assessment_frame.id)

        return self._build_assessment(
            idea=idea,
            final_trust=trust_final,
            is_stable=is_stable,
            rejection_reason=rejection_reason,
            axes=axes,
            stage_enum=stage_enum,
            original_trust=trust_raw,
            length_penalty=length_pen,
            progression_bonus=prog_bonus,
            has_progressive=has_prog,
            has_edge_support=edge_support,
            edge_distance=edge_dist,
            objective=final_objective
        )

    @staticmethod
    def _explode_weights(cfg: Dict[str, Any]) -> None:
        if 'weights' in cfg and len(cfg['weights']) == 3:
            cfg['alignment_weight'], cfg['feasibility_weight'], cfg['novelty_weight'] = cfg['weights']

    async def _get_llm_scores(self, prompt: str, ctx: NireonExecutionContext) -> Tuple[float, float, str]:
        if not self.sentinel.gateway:
            raise SentinelAssessmentError('MechanismGateway is unavailable to Sentinel.')
        
        frame_id = ctx.metadata.get('current_frame_id')
        if not frame_id:
            raise SentinelAssessmentError("Context missing 'current_frame_id'.")
            
        payload = LLMRequestPayload(
            prompt=prompt,
            stage=EpistemicStage.CRITIQUE,
            role='sentinel_evaluator',
            llm_settings={'temperature': 0.2, 'max_tokens': 512, 'route': 'sentinel_axis_scorer'}
        )
        event = CognitiveEvent(
            frame_id=frame_id,
            owning_agent_id=self.sentinel.component_id,
            service_call_type='LLM_ASK',
            payload=payload,
            epistemic_intent='ASSESS_IDEA_AXES'
        )
        
        try:
            response: LLMResponse = await self.sentinel.gateway.process_cognitive_event(event, ctx)
        except Exception as exc:
            ctx.logger.critical('[%s] Gateway call failed: %s', self.sentinel.component_id, exc, exc_info=True)
            raise SentinelAssessmentError(f'Gateway error: {exc}') from exc
        
        if response.get('error'):
            err = str(response.get('error'))
            ctx.logger.error('[%s] Gateway returned error: %s', self.sentinel.component_id, err)
            return (DEFAULT_LLM_SCORE, DEFAULT_LLM_SCORE, err)
            
        return self._parse_llm_response(response.text, ctx)

    def _parse_llm_response(self, raw: str, ctx: NireonExecutionContext) -> Tuple[float, float, str]:
        if not raw:
            raise SentinelLLMParsingError('Empty LLM response.')

        brace_match = _JSON_BRACE_RE.search(raw)
        if brace_match:
            try:
                data = json.loads(brace_match.group(0))
                align = _clamp(float(data.get('align_score', DEFAULT_LLM_SCORE)))
                feas = _clamp(float(data.get('feas_score', DEFAULT_LLM_SCORE)))
                expl = str(data.get('explanation', 'No LLM explanation provided.'))
                ctx.logger.debug('[%s] Parsed JSON response.', self.sentinel.component_id)
                return (align, feas, expl)
            except (ValueError, TypeError, json.JSONDecodeError):
                ctx.logger.debug('[%s] JSON parse failed; trying regex.', self.sentinel.component_id)

        try:
            align = _clamp(float(_ALIGN_RE.search(raw).group(1)))
            feas = _clamp(float(_FEAS_RE.search(raw).group(1)))
            expl_match = _EXPL_RE.search(raw)
            expl = expl_match.group(1).strip() if expl_match else 'No explanation found.'
            ctx.logger.debug('[%s] Parsed via regex.', self.sentinel.component_id)
            return (align, feas, expl)
        except Exception as exc:
            ctx.logger.warning('[%s] Regex parsing failed: %s', self.sentinel.component_id, exc)
        
        raise SentinelLLMParsingError(f'Unparseable LLM response: {raw[:200]}')

    @staticmethod
    def _determine_stage(idea: Idea) -> Optional[EpistemicStage]:
        stage_name = idea.metadata.get('stage') if idea.metadata else None
        return EpistemicStage.__members__.get(stage_name.upper()) if isinstance(stage_name, str) else None

    def _update_assessment_parameters(self, params: Dict[str, Any]) -> None:
        weights = np.asarray([params['alignment_weight'], params['feasibility_weight'], params['novelty_weight']], dtype=float)
        if abs(weights.sum() - 1.0) > 0.01:
            logger.warning('[%s] Weights sum to %.3f (expected 1.0).', self.sentinel.component_id, weights.sum())
        
        self.sentinel.weights = weights
        self.sentinel.trust_th = params.get('trust_threshold', self.sentinel.trust_th)
        self.sentinel.min_axis = params.get('min_axis_score', self.sentinel.min_axis)

    def _calculate_trust_score(self, axes: Sequence[AxisScore]) -> float:
        scores = np.array([a.score for a in axes], dtype=float)
        trust = float(np.dot(self.sentinel.weights, scores))
        return _clamp(trust)

    def _determine_stability(self, trust: float, align: float, feas: float, nov: float) -> bool:
        return trust >= self.sentinel.trust_th and align >= self.sentinel.min_axis and (feas >= self.sentinel.min_axis) and (nov >= self.sentinel.min_axis)

    def _generate_rejection_reason(self, align: float, feas: float, nov: float, trust: float, llm_expl: str) -> str:
        reasons: list[str] = []
        if trust < self.sentinel.trust_th:
            reasons.append(f'Low Trust ({trust:.1f} < {self.sentinel.trust_th:.1f})')
        if align < self.sentinel.min_axis:
            reasons.append(f'Low Alignment ({align:.1f} < {self.sentinel.min_axis:.1f})')
        if feas < self.sentinel.min_axis:
            reasons.append(f'Low Feasibility ({feas:.1f} < {self.sentinel.min_axis:.1f})')
        if nov < self.sentinel.min_axis:
            reasons.append(f'Low Novelty ({nov:.1f} < {self.sentinel.min_axis:.1f})')
        
        joined = '; '.join(reasons) or 'One or more criteria failed.'
        return f'Unstable: {joined}. LLM: "{_snippet(llm_expl)}"'

    def _edge_trust_context(self, idea: Idea) -> Tuple[bool, int]:
        idea_service = self.sentinel.idea_service
        if idea_service is None:
            return (False, 0)

        for pid in getattr(idea, 'parent_ids', []) or []:
            parent = idea_service.get_idea(pid)
            if parent and getattr(parent, 'trust_score', 0.0) >= self.sentinel.trust_th:
                return (True, 1)

        return (False, 0)

    def _build_assessment(self, *, idea: Idea, final_trust: float, is_stable: bool, rejection_reason: Optional[str], axes: Sequence[AxisScore], stage_enum: Optional[EpistemicStage], original_trust: float, length_penalty: float, progression_bonus: float, has_progressive: bool, has_edge_support: bool, edge_distance: int, objective: str) -> IdeaAssessment:
        metadata: Dict[str, Any] = {
            'step_created': idea.step,
            'stage_assessed': stage_enum.name if stage_enum else None,
            'method_created': idea.method,
            'original_trust_score': original_trust,
            'length_penalty_applied': length_penalty,
            'progression_bonus_applied': progression_bonus,
            'contains_progressive_elements': has_progressive,
            'edge_support': has_edge_support,
            'edge_distance': edge_distance,
            'assessment_weights_used': self.sentinel.weights.tolist(),
            'trust_threshold_used': self.sentinel.trust_th,
            'min_axis_score_used': self.sentinel.min_axis,
            'llm_explanation_snippet': _snippet(axes[0].explanation if axes else None),
            'assessment_timestamp': datetime.now(timezone.utc).isoformat(),
            'idea_text': idea.text,
            'objective': objective,
            'depth': idea.metadata.get('depth', 0) if idea.metadata else 0,
        }
        logger.debug('[DEPTH_DEBUG] Final assessment metadata: %s', metadata)
        
        return IdeaAssessment(
            idea_id=idea.idea_id,
            trust_score=final_trust,
            is_stable=is_stable,
            rejection_reason=rejection_reason,
            axis_scores=list(axes),
            metadata=metadata,
        )