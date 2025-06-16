from __future__ import annotations
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
from core.base_component import NireonBaseComponent
from core.lifecycle import ComponentMetadata
from core.results import ProcessResult, AnalysisResult, SystemSignal, AdaptationAction
from domain.context import NireonExecutionContext
from domain.ideas.idea import Idea
from domain.ports.embedding_port import EmbeddingPort
from domain.ports.event_bus_port import EventBusPort
from domain.ports.llm_port import LLMPort
from domain.ports.mechanism_gateway_port import MechanismGatewayPort
from application.services.idea_service import IdeaService
from application.services.frame_factory_service import FrameFactoryService # <-- Add this import
from application.services.stage_evaluation_service import StageEvaluationService
from components.mechanisms.sentinel.metadata import SENTINEL_METADATA
from components.mechanisms.sentinel.config import SentinelMechanismConfig
from components.mechanisms.sentinel.errors import SentinelAssessmentError
from components.mechanisms.sentinel.assessment_core import AssessmentCore
from components.mechanisms.sentinel.novelty_calculator import NoveltyCalculator
from components.mechanisms.sentinel.scoring_adjustment import ScoringAdjustments
from components.mechanisms.sentinel.service_helpers import InitializationHelper, ProcessingHelper, AnalysisHelper, AdaptationHelper, EventPublisher

logger = logging.getLogger(__name__)

class SentinelMechanism(NireonBaseComponent):
    ConfigModel = SentinelMechanismConfig

    def __init__(self, config: Dict[str, Any], metadata_definition: Optional[ComponentMetadata]=None):
        super().__init__(config=config, metadata_definition=metadata_definition or SENTINEL_METADATA)
        self.gateway: Optional[MechanismGatewayPort] = None
        self.llm: Optional[LLMPort] = None
        self.embed: Optional[EmbeddingPort] = None
        self.event_bus: Optional[EventBusPort] = None
        self.idea_service: Optional[IdeaService] = None
        self.frame_factory: Optional[FrameFactoryService] = None # <-- Add this attribute
        self.sentinel_cfg = SentinelMechanismConfig(**self.config)
        self.weights = np.array([], dtype=float)
        self.trust_th: float = self.sentinel_cfg.trust_threshold
        self.min_axis: float = self.sentinel_cfg.min_axis_score
        self._init_helper = InitializationHelper(self)
        self._process_helper = ProcessingHelper(self)
        self._analysis_helper = AnalysisHelper(self)
        self._adapt_helper = AdaptationHelper(self)
        self._event_publisher = EventPublisher(self)
        self._init_helper.initialize_weights()
        self.stage_evaluation_service = StageEvaluationService(config=self.config)
        self.assessment_core = AssessmentCore(self)
        self.novelty_calculator = NoveltyCalculator(self)
        self.scoring_adjustments = ScoringAdjustments(self)
        logger.info(f'[{self.component_id}] instance created. TrustThrCfg={self.sentinel_cfg.trust_threshold}, LiveTrust={self.trust_th}, MinAxis={self.min_axis}, Weights={self.weights.tolist()}')

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
            context.logger.info(f'[{self.component_id}] No specific state to reset for SentinelAssessmentError in V4 model.')
            return True
        return await super().recover_from_error(error, context)
        
    async def shutdown(self, context: NireonExecutionContext) -> None:
        context.logger.info(f'[{self.component_id}] Performing Sentinel-specific shutdown procedures.')
        if self._event_publisher and self.event_bus:
            await self._event_publisher.publish_lifecycle_event('sentinel_shutdown_initiated', context)
        await super().shutdown(context)