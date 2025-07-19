# nireon_v4/components/observers/adaptive_triggers.py
from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Optional, Tuple

from pydantic import BaseModel, Field

from core.base_component import NireonBaseComponent
from core.lifecycle import ComponentMetadata
from core.results import ProcessResult
from domain.context import NireonExecutionContext
from signals.core import TrustAssessmentSignal
from components.service_resolution_mixin import ServiceResolutionMixin

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------#
#  Shared helpers
# ---------------------------------------------------------------------------#

def _best_objective(payload: Dict[str, Any]) -> str:
    """Fast-path extractor with minimal nesting checks."""
    return (
        payload.get('assessment_details', {})
        .get('metadata', {})
        .get('objective', 'Generate novel concepts.')
    )

# ---------------------------------------------------------------------------#
#  StagnationDetector
# ---------------------------------------------------------------------------#

STAGNATION_DETECTOR_METADATA = ComponentMetadata(
    id='stagnation_detector_main',
    name='Stagnation Detector',
    version='1.0.0',
    category='observer',
    description='Monitors idea novelty and triggers a catalyst when stagnation is detected.',
    epistemic_tags=[
        'monitor', 'state_tracker', 'adaptive_trigger', 'diversity_controller'
    ],
    requires_initialize=True,
    dependencies={'catalyst_instance_01': '*', 'IdeaService': '*'},
)


class StagnationDetectorConfig(BaseModel):
    stagnation_threshold: int = Field(5, ge=2)
    novelty_score_threshold: float = Field(6.0, ge=0.0, le=10.0)
    catalyst_trigger_id: str = Field('catalyst_instance_01')
    idea_selection_trust_threshold: float = Field(7.0, ge=0.0, le=10.0)


class StagnationDetector(NireonBaseComponent, ServiceResolutionMixin):
    """
    Counts consecutive low-novelty ideas; when the streak hits the threshold,
    sends the most recent high-trust idea to the Catalyst.
    """

    __slots__ = (
        'cfg',
        'low_novelty_streak',
        'last_high_trust_candidate',
        'catalyst_mechanism',
        'idea_service',
    )

    METADATA_DEFINITION = STAGNATION_DETECTOR_METADATA
    ConfigModel = StagnationDetectorConfig

    # --------------------------------------------------------------- init  --
    def __init__(
        self,
        config: Dict[str, Any],
        metadata_definition: Optional[ComponentMetadata] = None,
    ):
        super().__init__(config, metadata_definition or self.METADATA_DEFINITION)
        self.cfg = self.ConfigModel(**self.config)
        self.low_novelty_streak: int = 0
        self.last_high_trust_candidate: Optional[Tuple[str, str, str]] = None
        self.catalyst_mechanism: Optional[NireonBaseComponent] = None
        self.idea_service = None  # reserved for later use

    # ------------------------------------------------------ lifecycle init --
    async def _initialize_impl(self, context: NireonExecutionContext) -> None:
        self.resolve_services(
            context,
            {
                'catalyst_mechanism': self.cfg.catalyst_trigger_id,
                # IdeaService kept for future extensions
            },
            raise_on_missing=True,
        )
        context.logger.info(
            f"[{self.component_id}] armed (threshold={self.cfg.stagnation_threshold},"
            f" novelty≤{self.cfg.novelty_score_threshold})"
        )

    # ----------------------------------------------------------- processing --
    async def _process_impl(
        self, data: Dict[str, Any], context: NireonExecutionContext
    ) -> ProcessResult:
        sig: Optional[TrustAssessmentSignal] = data.get('signal')
        if not isinstance(sig, TrustAssessmentSignal):
            return ProcessResult(False, "Expected TrustAssessmentSignal")

        # 1.  Track best candidate ----------------------------------------------------
        score = sig.trust_score or 0.0
        if score >= self.cfg.idea_selection_trust_threshold:
            self.last_high_trust_candidate = (
                sig.target_id,
                sig.payload.get('idea_text'),
                _best_objective(sig.payload),
            )
            context.logger.debug(
                f"[{self.component_id}] candidate={sig.target_id[:8]} "
                f"(trust={score:.2f})"
            )

        # 2.  Check novelty and update streak ----------------------------------------
        novelty = (
            sig.payload.get('axis_scores', {})
            .get('novel', {})
            .get('score')
        )
        if novelty is not None and novelty < self.cfg.novelty_score_threshold:
            self.low_novelty_streak += 1
            context.logger.info(
                f"[{self.component_id}] low-novelty streak "
                f"{self.low_novelty_streak}/{self.cfg.stagnation_threshold}"
            )
            if self.low_novelty_streak >= self.cfg.stagnation_threshold:
                return await self._trigger_catalyst(context)
        else:
            self.low_novelty_streak = 0  # reset on any high-novelty item

        return ProcessResult(
            True,
            component_id=self.component_id,
            message=f"streak={self.low_novelty_streak}",
        )

    # ----------------------------------------------------------- internals --
    async def _trigger_catalyst(
        self, context: NireonExecutionContext
    ) -> ProcessResult:
        if not self.last_high_trust_candidate:
            self.low_novelty_streak = 0
            msg = "Stagnation detected but no high-trust idea available."
            context.logger.warning(f"[{self.component_id}] {msg}")
            return ProcessResult(True, message=msg)

        idea_id, _, objective = self.last_high_trust_candidate
        context.logger.warning(
            f"[{self.component_id}] stagnation threshold hit → Catalyst <= {idea_id[:8]}"
        )
        # Fire-and-forget
        asyncio.create_task(
            self.catalyst_mechanism.process(
                {'target_idea_id': idea_id, 'objective': objective}, context
            )
        )
        self.low_novelty_streak = 0
        return ProcessResult(
            True,
            message=f"Catalyst triggered with idea {idea_id}",
            component_id=self.component_id,
        )

# ---------------------------------------------------------------------------#
#  ShakeUpCoordinator
# ---------------------------------------------------------------------------#

SHAKEUP_COORDINATOR_METADATA = ComponentMetadata(
    id='shake_up_coordinator',
    name='Shake-Up Coordinator',
    version='1.0.0',
    category='observer',
    description='Periodically triggers the Catalyst mechanism to ensure diversity.',
    epistemic_tags=['coordinator', 'scheduler', 'periodic_trigger'],
    requires_initialize=True,
    dependencies={'catalyst_instance_01': '*'},
)


class ShakeUpCoordinatorConfig(BaseModel):
    trigger_interval: int = Field(20, ge=5)
    catalyst_trigger_id: str = Field('catalyst_instance_01')


class ShakeUpCoordinator(NireonBaseComponent, ServiceResolutionMixin):
    """
    After *N* processed signals, forwards the latest high-trust idea
    (provided by the caller) to the Catalyst.
    """

    __slots__ = ('cfg', 'counter', 'catalyst_mechanism')

    METADATA_DEFINITION = SHAKEUP_COORDINATOR_METADATA
    ConfigModel = ShakeUpCoordinatorConfig

    # --------------------------------------------------------------- init  --
    def __init__(
        self,
        config: Dict[str, Any],
        metadata_definition: Optional[ComponentMetadata] = None,
    ):
        super().__init__(config, metadata_definition or self.METADATA_DEFINITION)
        self.cfg = self.ConfigModel(**self.config)
        self.counter: int = 0
        self.catalyst_mechanism: Optional[NireonBaseComponent] = None

    # ------------------------------------------------------ lifecycle init --
    async def _initialize_impl(self, context: NireonExecutionContext) -> None:
        self.resolve_services(
            context,
            {'catalyst_mechanism': self.cfg.catalyst_trigger_id},
            raise_on_missing=True,
        )
        context.logger.info(
            f"[{self.component_id}] armed (interval={self.cfg.trigger_interval})"
        )

    # ----------------------------------------------------------- processing --
    async def _process_impl(
        self, data: Dict[str, Any], context: NireonExecutionContext
    ) -> ProcessResult:
        self.counter += 1
        if self.counter < self.cfg.trigger_interval:
            return ProcessResult(
                True,
                message=f"counter={self.counter}/{self.cfg.trigger_interval}",
            )

        # Interval hit - try to trigger
        idea_id = data.get('idea_id')
        if not idea_id:
            msg = "Shake-up trigger fired but no idea supplied."
            context.logger.warning(f"[{self.component_id}] {msg}")
            self.counter = 0
            return ProcessResult(True, message=msg)

        asyncio.create_task(
            self.catalyst_mechanism.process(
                {
                    'target_idea_id': idea_id,
                    'objective': data.get('objective'),
                },
                context,
            )
        )
        context.logger.info(
            f"[{self.component_id}] Catalyst <= {idea_id[:8]} (periodic)"
        )
        self.counter = 0
        return ProcessResult(True, message=f"Catalyst triggered with {idea_id}")
