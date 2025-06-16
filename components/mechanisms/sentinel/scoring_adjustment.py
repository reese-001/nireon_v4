# C:\Users\erees\Documents\development\nireon_staging\nireon\application\mechanisms\sentinel\scoring_adjustments.py

import logging
from typing import Tuple, TYPE_CHECKING

from domain.context import NireonExecutionContext

if TYPE_CHECKING:
    from .service import SentinelMechanism

logger = logging.getLogger(__name__)


class ScoringAdjustments:
    PROGRESSIVE_INDICATORS = [
        'implementation timeline', 'metrics for success', 'detailed roadmap',
        'based on previous', 'building upon', 'address critique', 'specific steps',
        'detailed implementation', 'execution plan', 'milestones', 'measurement criteria',
        'success indicators', 'phased approach', 'strategic timeline',
        'implementation strategy', 'next steps', 'future work', 'iteration'
    ]

    def __init__(self, sentinel: 'SentinelMechanism'):
        self.sentinel = sentinel

    def apply_length_penalty(
        self,
        trust: float,
        length: int,
        ctx: NireonExecutionContext
    ) -> Tuple[float, float]:
        cfg = self.sentinel.sentinel_cfg

        if not cfg.enable_length_penalty or length <= cfg.length_penalty_threshold:
            return trust, 0.0

        penalty = ((length - cfg.length_penalty_threshold) / 1000.0) * cfg.length_penalty_factor
        adjusted_trust = max(1.0, trust - penalty)

        ctx.logger.debug(
            f'[{self.sentinel.component_id}] Length penalty -{penalty:.2f} '
            f'(text length {length}) applied. Trust: {trust:.2f} -> {adjusted_trust:.2f}.'
        )

        return adjusted_trust, penalty

    def apply_progression_adjustment(
        self,
        trust: float,
        idea_txt: str,
        idea_step: int,
        ctx: NireonExecutionContext
    ) -> Tuple[float, float, bool]:
        cfg = self.sentinel.sentinel_cfg

        flag_enabled = ctx.is_flag_enabled(
            'sentinel_enable_progression_adjustment',
            cfg.enable_progression_adjustment
        )
        has_progressive = self._contains_progressive_elements(idea_txt)

        if not flag_enabled or idea_step < cfg.progression_adjustment_min_step or not has_progressive:
            return trust, 0.0, False

        steps_past_min = idea_step - cfg.progression_adjustment_min_step + 1
        bonus = min(
            steps_past_min * cfg.progression_adjustment_bonus_factor,
            cfg.progression_adjustment_bonus_cap
        )
        adjusted_trust = min(10.0, trust + bonus)

        ctx.logger.debug(
            f'[{self.sentinel.component_id}] Progression bonus +{bonus:.2f} (step {idea_step}) applied. '
            f'Trust: {trust:.2f} -> {adjusted_trust:.2f}.'
        )

        return adjusted_trust, bonus, True

    def _contains_progressive_elements(self, text: str) -> bool:
        text_lower = text.lower()
        indicator_count = sum(1 for indicator in self.PROGRESSIVE_INDICATORS if indicator in text_lower)
        return indicator_count >= 3

    def apply_edge_trust_adjustment(
        self,
        trust: float,
        has_edge_support: bool,
        edge_distance: int,
        ctx: NireonExecutionContext
    ) -> float:
        cfg = self.sentinel.sentinel_cfg

        if not cfg.enable_edge_trust:
            return trust

        if has_edge_support:
            adjusted_trust = min(10.0, trust + cfg.edge_support_boost)
            ctx.logger.debug(
                f'[{self.sentinel.component_id}] Edge support boost +{cfg.edge_support_boost:.2f} applied.'
            )
        elif edge_distance > 3:
            decay = cfg.edge_trust_decay * (edge_distance - 3)
            adjusted_trust = max(1.0, trust - decay)
            ctx.logger.debug(
                f'[{self.sentinel.component_id}] Edge decay -{decay:.2f} applied (distance={edge_distance}).'
            )
        else:
            adjusted_trust = trust

        return adjusted_trust

    def calculate_composite_trust(
        self,
        base_trust: float,
        adjustments: dict[str, float]
    ) -> float:
        total_adjustment = sum(adjustments.values())
        final_trust = base_trust + total_adjustment
        return max(1.0, min(10.0, final_trust))
