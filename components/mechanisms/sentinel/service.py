# In nireon_v4/components/mechanisms/sentinel/service.py
# Complete fixes addressing all review points + conversion to ProcessorMechanism

from __future__ import annotations
import asyncio
import logging
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
from datetime import datetime, timezone

from core.base_component import NireonBaseComponent
from core.lifecycle import ComponentMetadata
from core.results import ComponentHealth, ProcessResult, AnalysisResult, SystemSignal, AdaptationAction
from domain.context import NireonExecutionContext
from domain.ports.embedding_port import EmbeddingPort
from domain.ports.llm_port import LLMPort
from domain.ports.mechanism_gateway_port import MechanismGatewayPort
from domain.evaluation.assessment import AxisScore  # For type hints
from application.services.idea_service import IdeaService
from application.services.frame_factory_service import FrameFactoryService
from application.services.stage_evaluation_service import StageEvaluationService
from components.mechanisms.base import ProcessorMechanism 
from components.service_resolution_mixin import ServiceResolutionMixin

from .metadata import SENTINEL_METADATA
from .config import SentinelMechanismConfig
from .errors import SentinelAssessmentError
from .assessment_core import AssessmentCore
from .novelty_calculator import NoveltyCalculator
from .scoring_adjustment import ScoringAdjustments
from .service_helpers import InitializationHelper, ProcessingHelper, AnalysisHelper, AdaptationHelper

logger = logging.getLogger(__name__)


class SentinelMechanism(ProcessorMechanism, ServiceResolutionMixin):
    """
    Sentinel is a PROCESSOR mechanism that evaluates ideas and returns assessment data.
    It does NOT publish signals directly; the Reactor will promote its output to TrustAssessmentSignal.
    
    Required Services:
        - gateway (MechanismGatewayPort): For LLM communication
        - embed (EmbeddingPort): For vector operations and novelty calculation
        - idea_service (IdeaService): For idea management
        - frame_factory (FrameFactoryService): For frame management
        
    Optional Services:
        - stage_evaluation_service (StageEvaluationService): For stage-specific evaluation
    """

    ConfigModel = SentinelMechanismConfig

    def __init__(self, config: Dict[str, Any], metadata_definition: Optional[ComponentMetadata]=None) -> None:
        super().__init__(config=config, metadata_definition=metadata_definition or SENTINEL_METADATA)
        self.gateway: Optional[MechanismGatewayPort] = None
        self.llm: Optional[LLMPort] = None
        self.embed: Optional[EmbeddingPort] = None
        self.idea_service: Optional[IdeaService] = None
        self.frame_factory: Optional[FrameFactoryService] = None
        self.stage_evaluation_service: Optional[StageEvaluationService] = None
        
        # Configuration
        self.sentinel_cfg = SentinelMechanismConfig(**self.config)
        self.weights = np.array([], dtype=float)
        self.trust_th: float = self.sentinel_cfg.trust_threshold
        self.min_axis: float = self.sentinel_cfg.min_axis_score
        
        # FIX 3.3: Initialize trust_axes for consistency
        self.trust_axes: List[AxisScore] = []  # Populated during assessment
        
        # FIX 3.1 & 3.2: Initialize all state variables properly
        self._idea_reload_executor = ThreadPoolExecutor(
            max_workers=2, 
            thread_name_prefix="sentinel_idea_reload"
        )
        self._assessment_count = 0
        self._error_count = 0  # General error counter
        
        # New metrics for parsing success tracking
        self._llm_parse_attempts = 0
        self._llm_parse_failures = 0
        self._llm_error_types = defaultdict(int)  # Track types of errors
        self._default_score_usage_count = 0
        
        # Helper instances
        self._init_helper = InitializationHelper(self)
        self._process_helper = ProcessingHelper(self)
        self._analysis_helper = AnalysisHelper(self)
        self._adapt_helper = AdaptationHelper(self)
        
        # Core assessment components
        self.assessment_core = AssessmentCore(self)
        self.novelty_calculator = NoveltyCalculator(self)
        self.scoring_adjustments = ScoringAdjustments(self)
        
        # Initialize weights
        self._init_helper.initialize_weights()
        
        logger.info(
            f'[{self.component_id}] instance created. TrustThrCfg={self.sentinel_cfg.trust_threshold}, '
            f'LiveTrust={self.trust_th}, MinAxis={self.min_axis}, Weights={self.weights.tolist()}'
        )

    async def _initialize_impl(self, context: NireonExecutionContext) -> None:
        """Initialize Sentinel with dependency resolution and validation."""
        context.logger.info(f'[{self.component_id}] Performing Sentinel-specific initialization.')
        
        try:
            # Resolve all required services using the mixin
            await self._resolve_all_dependencies(context)
            
            # Validate that all required services are present
            self._validate_dependencies(context)
            
            # Special handling: llm is an alias for gateway
            self.llm = self.gateway
            
            context.logger.info(f'[{self.component_id}] Initialization complete')
            
        except Exception as e:
            context.logger.error(f'[{self.component_id}] Initialization failed: {e}')
            raise

    async def _resolve_all_dependencies(self, context: NireonExecutionContext) -> None:
        """Resolve all required and optional dependencies using the ServiceResolutionMixin."""
        
        # Define required services
        required_services = {
            'gateway': MechanismGatewayPort,
            'embed': EmbeddingPort,
            'idea_service': IdeaService,
            'frame_factory': FrameFactoryService
        }
        
        # Define optional services
        optional_services = {
            'stage_evaluation_service': 'stage_evaluation_service'  # String-based lookup
        }
        
        try:
            # Resolve required services (will raise if any are missing)
            resolved_required = self.resolve_services(
                context=context,
                service_map=required_services,
                raise_on_missing=True,
                log_resolution=True
            )
            
            context.logger.debug(
                f"[{self.component_id}] Resolved {len(resolved_required)} required services"
            )
            
            # Resolve optional services (won't raise if missing)
            resolved_optional = self.resolve_services(
                context=context,
                service_map=optional_services,
                raise_on_missing=False,
                log_resolution=True
            )
            
            if resolved_optional:
                context.logger.debug(
                    f"[{self.component_id}] Resolved {len(resolved_optional)} optional services"
                )
            else:
                context.logger.info(
                    f"[{self.component_id}] No optional services were resolved"
                )
                
        except RuntimeError as e:
            # The mixin will provide detailed error messages
            context.logger.error(f"[{self.component_id}] Failed to resolve dependencies: {e}")
            raise

    def _validate_dependencies(self, context: NireonExecutionContext) -> None:
        """Validate that all required dependencies are available."""
        
        required_services = ['gateway', 'embed', 'idea_service', 'frame_factory']
        
        # Use the mixin's validation method
        if not self.validate_required_services(required_services, context):
            # The mixin will have already logged which services are missing
            missing = [s for s in required_services if not getattr(self, s, None)]
            raise ValueError(
                f'[{self.component_id}] Missing required dependencies: {", ".join(missing)}'
            )

    def _ensure_services_available(self, context: NireonExecutionContext) -> bool:
        """
        Ensure all required services are available at runtime.
        This can be used for lazy resolution or recovery after failures.
        """
        required_services = ['gateway', 'embed', 'idea_service']
        
        # Quick check if all services are already available
        if self.validate_required_services(required_services):
            return True
        
        # Attempt to re-resolve missing services
        context.logger.warning(
            f"[{self.component_id}] Some services missing at runtime, attempting re-resolution"
        )
        
        try:
            # Build service map only for missing services
            service_map = {}
            if not self.gateway:
                service_map['gateway'] = MechanismGatewayPort
            if not self.embed:
                service_map['embed'] = EmbeddingPort
            if not self.idea_service:
                service_map['idea_service'] = IdeaService
            if not self.frame_factory:
                service_map['frame_factory'] = FrameFactoryService
            
            if service_map:
                self.resolve_services(
                    context=context,
                    service_map=service_map,
                    raise_on_missing=False,  # Don't raise, we'll check below
                    log_resolution=True
                )
                
                # Re-establish llm alias if gateway was re-resolved
                if self.gateway and not self.llm:
                    self.llm = self.gateway
            
            # Check again after resolution attempt
            return self.validate_required_services(required_services)
            
        except Exception as e:
            context.logger.error(f"[{self.component_id}] Failed to ensure services: {e}")
            return False

    def _extract_session_metadata(self, data: Any) -> Dict[str, Any]:
        """Extract metadata from various nested structures.
        
        FIX 4: Add return type annotation
        """
        session_id = None
        metadata = {}
        
        if isinstance(data, dict):
            # Try multiple locations for session_id
            session_id = data.get('session_id')
            if not session_id and 'metadata' in data:
                session_id = data['metadata'].get('session_id')
                metadata.update(data['metadata'])
            if not session_id and 'payload' in data and isinstance(data['payload'], dict):
                payload_meta = data['payload'].get('metadata', {})
                session_id = payload_meta.get('session_id')
                metadata.update(payload_meta)
            
            # Extract other relevant fields
            for key in ['objective', 'depth', 'planner_action', 'parent_trust_score', 'target_component_id']:
                if key in data:
                    metadata[key] = data[key]
        
        if session_id:
            metadata['session_id'] = session_id
            
        return metadata

    def _create_assessment_details(
        self,
        final_trust_score: float,
        axis_scores: List[float],
        is_stable: bool,
        parent_id: Optional[str],
        objective: Optional[str],
        extracted_metadata: Dict[str, Any],
        start_time: Optional[float] = None
    ) -> Dict[str, Any]:
        """Create detailed assessment metadata for analysis.
        
        FIX 3.3: Handle trust_axes properly
        """
        assessment_details = {
            'trust_score': final_trust_score,
            'axis_scores': {
                # Use actual axis names if available, fallback to indices
                axis.name if hasattr(axis, 'name') else f'axis_{i}': score 
                for i, (axis, score) in enumerate(zip(self.trust_axes or [], axis_scores))
            },
            'is_stable': is_stable,
            'idea_parent_id': parent_id,
            'metadata': {
                'objective': objective or extracted_metadata.get('objective'),
                'depth': extracted_metadata.get('depth', 0),  # FIX 3.3: Consistent default
                'session_id': extracted_metadata.get('session_id'),
                'planner_action': extracted_metadata.get('planner_action'),
                'parent_trust_score': extracted_metadata.get('parent_trust_score'),
                'target_component_id': extracted_metadata.get('target_component_id'),
                'duration_ms': (time.time() - start_time) * 1000 if start_time else None
            }
        }
        
        return assessment_details

    async def _process_impl(self, data: Any, context: NireonExecutionContext) -> ProcessResult:
        """Main processing entry point with comprehensive logging"""
        logger.info(f"[{self.component_id}] === SENTINEL PROCESS START ===")
        logger.info(f"[{self.component_id}] Input data type: {type(data)}")
        logger.info(f"[{self.component_id}] Input data keys: {list(data.keys()) if isinstance(data, dict) else 'N/A'}")
        
        # Ensure services are available (defensive check with potential re-resolution)
        if not self._ensure_services_available(context):
            return ProcessResult(
                success=False,
                component_id=self.component_id,
                message="Required services not available",
                error_code='MISSING_DEPENDENCIES'
            )
        
        start_time = time.time()
        extracted_metadata = self._extract_session_metadata(data)
        context.logger.info(f'[{self.component_id}] Processing with extracted metadata: {extracted_metadata}')
        
        # Validate input
        if not isinstance(data, dict) or 'target_idea_id' not in data:
            logger.error(f"[{self.component_id}] Invalid input - expected dict with 'target_idea_id'")
            return ProcessResult(
                success=False, 
                component_id=self.component_id, 
                message="Input data must be a dict with a 'target_idea_id' key.", 
                error_code='INVALID_INPUT'
            )
        
        target_idea_id = str(data.get('target_idea_id', ''))
        logger.info(f"[{self.component_id}] Target idea ID: {target_idea_id}")
        
        try:
            # Delegate to ProcessingHelper
            result = await self._process_helper.process_impl(data, context)
            
            elapsed = time.time() - start_time
            logger.info(f"[{self.component_id}] === SENTINEL PROCESS COMPLETE ===")
            logger.debug(f"[{self.component_id}] Processing took {elapsed:.2f}s")
            logger.debug(f"[{self.component_id}] Result success: {result.success}")
            logger.debug(f"[{self.component_id}] Result message: {result.message}")
            
            if result.output_data:
                logger.debug(f"[{self.component_id}] Output data type: {result.output_data.get('type')}")
                logger.debug(f"[{self.component_id}] Output data keys: {list(result.output_data.keys())}")
                if result.output_data.get('type') == 'trust_assessment':
                    logger.info(f"[{self.component_id}] TRUST ASSESSMENT READY:")
                    logger.info(f"[{self.component_id}]   - idea_id: {result.output_data.get('idea_id')}")
                    logger.info(f"[{self.component_id}]   - trust_score: {result.output_data.get('trust_score')}")
                    logger.info(f"[{self.component_id}]   - is_stable: {result.output_data.get('is_stable')}")
            
            return result
            
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[{self.component_id}] === SENTINEL PROCESS FAILED ===")
            logger.error(f"[{self.component_id}] Failed after {elapsed:.2f}s with error: {e}", exc_info=True)
            return ProcessResult(
                success=False,
                component_id=self.component_id,
                message=f"Unexpected error in Sentinel processing: {str(e)}",
                error_code='SENTINEL_PROCESS_ERROR'
            )

    async def _fetch_idea_with_backoff(self, idea_id: str, context: NireonExecutionContext) -> Optional[Any]:
        """Fetch an idea with exponential backoff retry logic."""
        max_retries = 3
        base_delay = 0.1
        
        for attempt in range(max_retries):
            try:
                idea = self.idea_service.get_idea(idea_id)
                if idea:
                    return idea
                    
                # If not found, don't retry
                context.logger.debug(f'[{self.component_id}] Idea {idea_id} not found')
                return None
                
            except Exception as e:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    context.logger.warning(
                        f'[{self.component_id}] Retry {attempt + 1}/{max_retries} '
                        f'for idea {idea_id} after {delay}s: {e}'
                    )
                    await asyncio.sleep(delay)
                else:
                    context.logger.error(
                        f'[{self.component_id}] Failed to fetch idea {idea_id} '
                        f'after {max_retries} attempts: {e}'
                    )
                    raise

    async def _gather_reference_ideas(self, idea: Any, context: NireonExecutionContext) -> List[Any]:
        """Gather reference ideas for comparison."""
        # Implementation depends on your specific logic
        # This is a placeholder
        return []

    async def analyze(self, context: NireonExecutionContext) -> AnalysisResult:
        """Analyze recent assessment activity with enhanced metrics."""
        # Calculate parsing success rate
        parse_success_rate = 0.0
        if self._llm_parse_attempts > 0:
            parse_success_rate = 1.0 - (self._llm_parse_failures / self._llm_parse_attempts)
        
        # Calculate default score usage rate
        default_usage_rate = 0.0
        if self._assessment_count > 0:
            default_usage_rate = self._default_score_usage_count / self._assessment_count
        
        # Check service availability
        service_health = self.validate_required_services(
            ['gateway', 'embed', 'idea_service'],
            context
        )
        
        return AnalysisResult(
            component_id=self.component_id,
            success=True,
            findings={
                'total_assessments': self._assessment_count,
                'error_count': self._error_count,
                'error_rate': self._error_count / max(1, self._assessment_count),
                'current_weights': self.weights.tolist(),
                'trust_threshold': self.trust_th,
                # New metrics
                'llm_parse_attempts': self._llm_parse_attempts,
                'llm_parse_failures': self._llm_parse_failures,
                'llm_parse_success_rate': parse_success_rate,
                'default_score_usage_rate': default_usage_rate,
                'llm_error_breakdown': dict(self._llm_error_types),
                'circuit_breaker_active': getattr(self.assessment_core, '_circuit_breaker_active', False),
                # Service health
                'all_services_available': service_health,
                'has_optional_stage_service': self.stage_evaluation_service is not None
            },
            recommendations=self._generate_analysis_recommendations(parse_success_rate, default_usage_rate),
            insights=self._generate_analysis_insights(parse_success_rate, default_usage_rate)
        )

    def _generate_analysis_recommendations(self, parse_success_rate: float, default_usage_rate: float) -> List[str]:
        """Generate recommendations based on current metrics."""
        recommendations = []
        
        if parse_success_rate < 0.8:
            recommendations.append('LLM parsing success rate is low. Consider reviewing prompt templates.')
        
        if default_usage_rate > 0.2:
            recommendations.append('High default score usage. Check LLM service health and response quality.')
        
        if self._llm_error_types.get('rate_limit', 0) > 5:
            recommendations.append('Frequent rate limit errors. Consider adjusting request frequency or upgrading quota.')
        
        # Check for missing optional services
        if not self.stage_evaluation_service:
            recommendations.append('StageEvaluationService not available. Consider enabling for stage-specific assessments.')
        
        return recommendations

    def _generate_analysis_insights(self, parse_success_rate: float, default_usage_rate: float) -> List[str]:
        """Generate insights based on current metrics."""
        insights = []
        
        if parse_success_rate >= 0.95:
            insights.append('LLM parsing is highly reliable.')
        
        if self.assessment_core._circuit_breaker_active:
            insights.append('Circuit breaker is currently active due to repeated parsing failures.')
        
        # Most common error type
        if self._llm_error_types:
            most_common_error = max(self._llm_error_types.items(), key=lambda x: x[1])
            insights.append(f'Most common LLM error type: {most_common_error[0]} ({most_common_error[1]} occurrences)')
        
        return insights

    async def react(self, context: NireonExecutionContext) -> List[SystemSignal]:
        """React phase - generate system signals based on recent activity."""
        signals = await self._adapt_helper.handle_react_phase(context)
        base_signals = await super().react(context)
        
        # Add signals for critical conditions
        if hasattr(self.assessment_core, '_circuit_breaker_active') and self.assessment_core._circuit_breaker_active:
            signals.append(SystemSignal(
                signal_type='WARNING',
                component_id=self.component_id,
                message='Sentinel circuit breaker is active - LLM parsing repeatedly failed',
                payload={'circuit_breaker_active': True, 'consecutive_failures': self.assessment_core._consecutive_parse_failures}
            ))
        
        return signals + base_signals

    async def adapt(self, context: NireonExecutionContext) -> List[AdaptationAction]:
        """Adapt phase - adjust parameters based on context."""
        actions = await self._adapt_helper.adapt(context)
        base_actions = await super().adapt(context)
        
        # Add adaptations based on parsing metrics
        if self._llm_parse_attempts > 100:  # Only adapt after sufficient data
            parse_success_rate = 1.0 - (self._llm_parse_failures / self._llm_parse_attempts)
            
            if parse_success_rate < 0.5:
                # Very low success rate - consider increasing default score weight
                actions.append(AdaptationAction(
                    action_type='PARAMETER_ADJUST',
                    component_id=self.component_id,
                    description='Increasing reliance on default scores due to poor LLM parsing',
                    parameters={
                        'parameter_name': 'default_score_weight',
                        'adjustment': 'increase',
                        'reason': f'Parse success rate is only {parse_success_rate:.1%}'
                    }
                ))
        
        return actions + base_actions

    async def recover_from_error(self, error: Exception, context: NireonExecutionContext) -> bool:
        """Attempt recovery from specific error types."""
        context.logger.warning(f'[{self.component_id}] Attempting recovery from error: {error}')
        
        if isinstance(error, SentinelAssessmentError):
            # Assessment errors are often transient
            context.logger.info(f'[{self.component_id}] Assessment error is recoverable')
            return True
            
        return await super().recover_from_error(error, context)

    async def shutdown(self, context: NireonExecutionContext) -> None:
        """Clean shutdown with proper resource cleanup."""
        context.logger.info(f'[{self.component_id}] Beginning Sentinel shutdown...')
        
        # Shutdown thread pool executor
        if hasattr(self, '_idea_reload_executor') and self._idea_reload_executor:
            context.logger.debug(f'[{self.component_id}] Shutting down idea reload executor...')
            try:
                # Cancel all pending futures before shutdown
                self._idea_reload_executor.shutdown(wait=False)
                
                # Give it a moment to clean up with a timeout
                await asyncio.sleep(0.1)
                
                # Now force shutdown with timeout
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None, 
                    lambda: self._idea_reload_executor.shutdown(wait=True, timeout=2.0)
                )
            except Exception as e:
                context.logger.warning(
                    f'[{self.component_id}] Error during executor shutdown: {e}'
                )
            finally:
                self._idea_reload_executor = None
            context.logger.debug(f'[{self.component_id}] Executor shutdown complete')
        
        # Shutdown helpers if they have cleanup methods
        for helper_name, helper in [
            ('_process_helper', self._process_helper),
            ('_analysis_helper', self._analysis_helper),
            ('_adapt_helper', self._adapt_helper),
            ('_init_helper', self._init_helper)
        ]:
            if hasattr(helper, 'shutdown'):
                try:
                    context.logger.debug(f'[{self.component_id}] Shutting down {helper_name}')
                    await helper.shutdown(context)
                except Exception as e:
                    context.logger.warning(f'[{self.component_id}] Error shutting down {helper_name}: {e}')
        
        # Log final metrics
        context.logger.info(
            f'[{self.component_id}] Shutdown complete. '
            f'Total assessments: {self._assessment_count}, '
            f'Total errors: {self._error_count}, '
            f'Parse failures: {self._llm_parse_failures}/{self._llm_parse_attempts}'
        )
        
        # Call parent shutdown
        await super().shutdown(context)

    # FIX 3.4: Public metric accessors for dashboards
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics for monitoring."""
        parse_success_rate = 0.0
        if self._llm_parse_attempts > 0:
            parse_success_rate = 1.0 - (self._llm_parse_failures / self._llm_parse_attempts)
        
        # Check service availability
        all_services_available = all([
            self.gateway is not None,
            self.embed is not None,
            self.idea_service is not None,
            self.frame_factory is not None
        ])
            
        return {
            'total_assessments': self._assessment_count,
            'total_errors': self._error_count,
            'error_rate': self._error_count / max(1, self._assessment_count),
            'weights': self.weights.tolist(),
            'trust_threshold': self.trust_th,
            'min_axis_score': self.min_axis,
            # Enhanced metrics
            'llm_parse_success_rate': parse_success_rate,
            'llm_parse_attempts': self._llm_parse_attempts,
            'llm_parse_failures': self._llm_parse_failures,
            'default_score_usage_count': self._default_score_usage_count,
            'circuit_breaker_active': getattr(self.assessment_core, '_circuit_breaker_active', False),
            # Service status
            'all_services_available': all_services_available,
            'has_optional_stage_service': self.stage_evaluation_service is not None
        }

    async def health_check(self, context: NireonExecutionContext) -> ComponentHealth:
        """Perform health check including service availability."""
        from core.results import ComponentHealth
        
        # Check required services
        required_services = ['gateway', 'embed', 'idea_service', 'frame_factory']
        service_health = self.validate_required_services(required_services, context)
        
        # Determine overall health status
        if not service_health:
            status = "unhealthy"
            message = "Required services not available"
        elif self._assessment_count > 0 and (self._error_count / self._assessment_count) > 0.3:
            status = "degraded"
            message = "High error rate in assessments"
        else:
            status = "healthy"
            message = "All systems operational"
        
        metrics = self.get_metrics()
        
        return ComponentHealth(
            component_id=self.component_id,
            status=status,
            message=message,
            metrics=metrics
        )