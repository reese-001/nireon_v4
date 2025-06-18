# nireon_v4/components/mechanisms/sentinel/service.py
from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional
import numpy as np

from core.base_component import NireonBaseComponent
from core.lifecycle import ComponentMetadata
from core.results import ProcessResult, AnalysisResult, SystemSignal, AdaptationAction
from domain.context import NireonExecutionContext
from domain.ports.embedding_port import EmbeddingPort
from domain.ports.event_bus_port import EventBusPort
from domain.ports.llm_port import LLMPort
from domain.ports.mechanism_gateway_port import MechanismGatewayPort
from application.services.idea_service import IdeaService
from application.services.frame_factory_service import FrameFactoryService
from application.services.stage_evaluation_service import StageEvaluationService

# +++ MODIFIED: Imports now use the new helper classes +++
from .metadata import SENTINEL_METADATA
from .config import SentinelMechanismConfig
from .errors import SentinelAssessmentError
from .assessment_core import AssessmentCore
from .novelty_calculator import NoveltyCalculator
from .scoring_adjustment import ScoringAdjustments
from .service_helpers import (
    InitializationHelper, ProcessingHelper, AnalysisHelper,
    AdaptationHelper, EventPublisher
)

logger = logging.getLogger(__name__)


class SentinelMechanism(NireonBaseComponent):
    """
    Evaluates ideas against multiple axes to ensure quality and relevance.

    This class acts as a coordinator, delegating its core lifecycle methods
    to specialized helper classes for improved modularity and clarity.
    """
    ConfigModel = SentinelMechanismConfig

    def __init__(self,
                 config: Dict[str, Any],
                 metadata_definition: Optional[ComponentMetadata] = None
                 ) -> None:
        super().__init__(config=config, metadata_definition=metadata_definition or SENTINEL_METADATA)
        # --- Dependencies (resolved during initialization) ---
        self.gateway: Optional[MechanismGatewayPort] = None
        self.llm: Optional[LLMPort] = None
        self.embed: Optional[EmbeddingPort] = None
        self.event_bus: Optional[EventBusPort] = None
        self.idea_service: Optional[IdeaService] = None
        self.frame_factory: Optional[FrameFactoryService] = None

        # --- Configuration & State ---
        self.sentinel_cfg = SentinelMechanismConfig(**self.config)
        self.weights = np.array([], dtype=float)  # Normalized weights
        self.trust_th: float = self.sentinel_cfg.trust_threshold
        self.min_axis: float = self.sentinel_cfg.min_axis_score

        # +++ MODIFIED: Instantiate all helper classes +++
        self._init_helper = InitializationHelper(self)
        self._process_helper = ProcessingHelper(self)
        self._analysis_helper = AnalysisHelper(self)
        self._adapt_helper = AdaptationHelper(self)
        self._event_publisher = EventPublisher(self)

        # --- Sub-components for core logic ---
        self._init_helper.initialize_weights() # Initial setup before full init
        self.stage_evaluation_service = StageEvaluationService(config=self.config)
        self.assessment_core = AssessmentCore(self)
        self.novelty_calculator = NoveltyCalculator(self)
        self.scoring_adjustments = ScoringAdjustments(self)

        logger.info(f'[{self.component_id}] instance created. TrustThrCfg={self.sentinel_cfg.trust_threshold}, '
                    f'LiveTrust={self.trust_th}, MinAxis={self.min_axis}, Weights={self.weights.tolist()}')

    # +++ MODIFIED: All lifecycle methods are now delegated to helpers +++
    async def _initialize_impl(self, context: NireonExecutionContext) -> None:
        await self._init_helper.initialize_impl(context)

    async def _process_impl(self, data: Any, context: NireonExecutionContext) -> ProcessResult:
        return await self._process_helper.process_impl(data, context)

    async def analyze(self, context: NireonExecutionContext) -> AnalysisResult:
        return await self._analysis_helper.analyze(context)

    async def react(self, context: NireonExecutionContext) -> List[SystemSignal]:
        signals = await self._adapt_helper.handle_react_phase(context)
        base_signals = await super().react(context)
        return signals + base_signals

    async def adapt(self, context: NireonExecutionContext) -> List[AdaptationAction]:
        actions = await self._adapt_helper.adapt(context)
        base_actions = await super().adapt(context)
        return actions + base_actions

    async def recover_from_error(self, error: Exception, context: NireonExecutionContext) -> bool:
        context.logger.warning(f'[{self.component_id}] Attempting recovery from error: {error}')
        if isinstance(error, SentinelAssessmentError):
            context.logger.info(f'[{self.component_id}] No specific state to reset for SentinelAssessmentError.')
            return True
        return await super().recover_from_error(error, context)

    async def shutdown(self, context: NireonExecutionContext) -> None:
        context.logger.info(f'[{self.component_id}] Performing Sentinel-specific shutdown procedures.')
        if self._event_publisher and self.event_bus:
            await self._event_publisher.publish_lifecycle_event('sentinel_shutdown_initiated', context)
        await super().shutdown(context)