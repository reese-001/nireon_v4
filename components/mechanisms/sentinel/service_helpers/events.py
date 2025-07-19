# events.py
from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Final, Optional

from pydantic import ValidationError

from domain.cognitive_events import CognitiveEvent
from domain.context import NireonExecutionContext
from domain.evaluation.assessment import IdeaAssessment
from domain.ideas.idea import Idea
from domain.ports.mechanism_gateway_port import MechanismGatewayPort
from signals.core import TrustAssessmentSignal

if TYPE_CHECKING:  # avoids run-time import cycle
    from ..service import SentinelMechanism

logger: Final = logging.getLogger(__name__)

# Module-level constants for publish_assessment_event
AXIS_NOVEL = "novel"
GATEWAY_TIMEOUT_SEC = 4.0
IDEA_RELOAD_TIMEOUT_SEC = 2.0
DEPRECATED_TARGET_ID_REMOVAL = False  # Set True after migration period
GATEWAY_RETRY_THROTTLE_SEC = 60.0  # Don't retry gateway acquisition for 1 minute

__all__ = ["EventPublisher", "SentinelPublishError", "ComponentNotAvailableError"]


class SentinelPublishError(Exception):
    """Raised when assessment publication fails."""
    pass


class ComponentNotAvailableError(Exception):
    """Raised when a required component is unavailable."""
    pass


class EventPublisher:
    """Publishes Sentinel-related signals and bookkeeping events."""

    def __init__(self, sentinel: "SentinelMechanism") -> None:
        self.sentinel = sentinel

    # ------------------------------------------------------------------ #
    # 1. Trust-assessment signal via Gateway (Production-hardened)
    # ------------------------------------------------------------------ #
    async def publish_assessment_event(
        self,
        assess_obj: IdeaAssessment | None,
        idea: Idea | None,
        ctx: NireonExecutionContext,
    ) -> None:
        """Publish TrustAssessmentSignal with guaranteed idea_text.
        
        Args:
            assess_obj: The assessment result to publish
            idea: The idea being assessed (may be reloaded if text missing)
            ctx: Execution context with logger and metrics
            
        Raises:
            SentinelPublishError: If required data cannot be obtained
            ComponentNotAvailableError: If gateway is unavailable
            asyncio.TimeoutError: If gateway call exceeds timeout
            
        Metrics:
            Increments _failed_publishes on any failure
            Increments _successful_publishes on success
            Records timing histogram if ctx.metrics available
        """
        
        # Track timing
        start_time = time.perf_counter()
        
        # Validate inputs
        if not assess_obj:
            ctx.logger.error('[%s] Cannot publish null assessment', self.sentinel.component_id)
            self.sentinel._failed_publishes += 1
            raise SentinelPublishError('Assessment object is None')

        # Ensure gateway availability with throttled retry
        gateway = self.sentinel.gateway
        if gateway is None:
            current_time = time.time()
            if current_time - self.sentinel._gateway_last_attempt_ts < GATEWAY_RETRY_THROTTLE_SEC:
                # Don't spam registry lookups
                self.sentinel._failed_publishes += 1
                raise ComponentNotAvailableError(
                    f'Gateway unavailable (last attempt {current_time - self.sentinel._gateway_last_attempt_ts:.1f}s ago)'
                )
            
            ctx.logger.warning('[%s] Gateway not cached, attempting to acquire', self.sentinel.component_id)
            self.sentinel._gateway_last_attempt_ts = current_time
            
            try:
                gateway = ctx.component_registry.get_service_instance(MechanismGatewayPort)
                if gateway:
                    self.sentinel.gateway = gateway  # Cache on success
                    ctx.logger.info('[%s] Successfully acquired gateway', self.sentinel.component_id)
                else:
                    self.sentinel._failed_publishes += 1
                    raise ComponentNotAvailableError('MechanismGateway not found in registry')
            except Exception as e:
                self.sentinel._failed_publishes += 1
                raise ComponentNotAvailableError(f'Failed to acquire gateway: {e}')

        # Validate frame_id (required)
        frame_id = ctx.metadata.get('current_frame_id')
        if not frame_id:
            ctx.logger.error('[%s] current_frame_id missing - cannot publish', self.sentinel.component_id)
            self.sentinel._failed_publishes += 1
            raise SentinelPublishError('current_frame_id missing in context metadata')

        # Ensure we have idea with text (reload if needed)
        if not idea or not getattr(idea, 'text', None):
            ctx.logger.warning(
                '[%s] idea.text missing for %s - attempting reload',
                self.sentinel.component_id, assess_obj.idea_id
            )
            
            if not self.sentinel.idea_service:
                # Can't reload without service
                ctx.logger.error('[%s] IdeaService unavailable - cannot reload idea', self.sentinel.component_id)
                self.sentinel._failed_publishes += 1
                raise SentinelPublishError(f'No idea text and IdeaService unavailable for {assess_obj.idea_id}')
            
            try:
                # Try async first
                if hasattr(self.sentinel.idea_service, 'get_idea_async'):
                    idea = await asyncio.wait_for(
                        self.sentinel.idea_service.get_idea_async(assess_obj.idea_id),
                        timeout=IDEA_RELOAD_TIMEOUT_SEC
                    )
                else:
                    # Wrap sync call in dedicated executor to avoid blocking
                    idea = await asyncio.wait_for(
                        asyncio.get_event_loop().run_in_executor(
                            self.sentinel._idea_reload_executor,
                            self.sentinel.idea_service.get_idea,
                            assess_obj.idea_id
                        ),
                        timeout=IDEA_RELOAD_TIMEOUT_SEC
                    )
            except asyncio.TimeoutError:
                ctx.logger.error(
                    '[%s] Timeout reloading idea %s after %.1fs', 
                    self.sentinel.component_id, assess_obj.idea_id, IDEA_RELOAD_TIMEOUT_SEC
                )
                self.sentinel._failed_publishes += 1
                raise SentinelPublishError(f'Timeout reloading idea {assess_obj.idea_id}')
            except Exception as e:
                ctx.logger.error('[%s] Failed to reload idea: %s', self.sentinel.component_id, e)
                idea = None
        
        # Final check - fail if still no text
        if not idea or not getattr(idea, 'text', None):
            ctx.logger.error(
                '[%s] Still no idea.text for %s - aborting publish (failed_publishes=%d)',
                self.sentinel.component_id, assess_obj.idea_id, self.sentinel._failed_publishes + 1
            )
            self.sentinel._failed_publishes += 1
            raise SentinelPublishError(f'No text available for idea {assess_obj.idea_id}')
        
        # Build payload with guaranteed fresh idea
        signal_payload = {
            'idea_id': assess_obj.idea_id,
            'idea_text': idea.text,
            'is_stable': assess_obj.is_stable if assess_obj.is_stable is not None else True,
            'rejection_reason': assess_obj.rejection_reason,
            'assessment_details': assess_obj.model_dump(),
            'idea_metadata': getattr(idea, 'metadata', {}),
            # Add observability data
            '_assessment_id': assess_obj.idea_id,
            '_axis_count': len(assess_obj.axis_scores or [])
        }
        
        # Conditionally add deprecated field
        if not DEPRECATED_TARGET_ID_REMOVAL:
            signal_payload['target_id'] = assess_obj.idea_id
        
        ctx.logger.debug(
            '[%s] Publishing TrustAssessmentSignal for %s with text (%d chars), axes=%d',
            self.sentinel.component_id, assess_obj.idea_id, len(idea.text), 
            len(assess_obj.axis_scores or [])
        )
        
        # Extract novelty score safely (could use max if multiple novel axes)
        novelty_score = None
        try:
            # Find the highest novelty score if multiple exist
            novelty_scores = [
                float(ax.score) / 10.0 
                for ax in (assess_obj.axis_scores or [])
                if ax.name == AXIS_NOVEL and ax.score is not None
            ]
            if novelty_scores:
                novelty_score = max(novelty_scores)
        except Exception as e:
            ctx.logger.debug('[%s] Could not extract novelty: %s', self.sentinel.component_id, e)
        
        # Create signal with validation
        try:
            trust_signal = TrustAssessmentSignal(
                source_node_id=self.sentinel.component_id,
                target_id=assess_obj.idea_id,
                target_type='Idea',
                trust_score=assess_obj.trust_score,
                novelty_score=novelty_score,
                assessment_rationale=assess_obj.rejection_reason,
                payload=signal_payload,
                context_tags={'frame_id': frame_id}
            )
        except ValidationError as e:
            ctx.logger.error(
                '[%s] TrustAssessmentSignal validation failed: %s. Payload keys: %s',
                self.sentinel.component_id, e, list(signal_payload.keys())
            )
            self.sentinel._failed_publishes += 1
            raise SentinelPublishError(f'Signal validation failed: {e}')
        
        # Wrap in cognitive event
        ce_payload = {
            'event_type': trust_signal.signal_type,
            'event_data': trust_signal.model_dump(mode='json')
        }
        ce_meta = {
            'publisher_component_id': self.sentinel.component_id,
            'publisher_version': self.sentinel.metadata.version
        }
        
        cognitive_event = CognitiveEvent(
            frame_id=frame_id,
            owning_agent_id=self.sentinel.component_id,
            service_call_type='EVENT_PUBLISH',
            payload=ce_payload,
            epistemic_intent='PUBLISH_TRUST_ASSESSMENT',
            custom_metadata=ce_meta
        )
        
        # Publish with timeout and tracking
        try:
            await asyncio.wait_for(
                gateway.process_cognitive_event(cognitive_event, ctx),
                timeout=GATEWAY_TIMEOUT_SEC
            )
            
            # Success - update metrics
            self.sentinel._successful_publishes += 1
            publish_duration = time.perf_counter() - start_time
            
            ctx.logger.info(
                '[%s] Published TrustAssessmentSignal for %s in %.3fs (total_publishes=%d)',
                self.sentinel.component_id, assess_obj.idea_id, publish_duration,
                self.sentinel._successful_publishes
            )
            
            # Record timing if metrics available
            if hasattr(ctx, 'metrics') and hasattr(ctx.metrics, 'timing'):
                ctx.metrics.timing('sentinel.publish_assessment.duration', publish_duration)
                
        except asyncio.TimeoutError:
            self.sentinel._failed_publishes += 1
            ctx.logger.error(
                '[%s] Gateway timeout after %.1fs for %s (failed_publishes=%d)',
                self.sentinel.component_id, GATEWAY_TIMEOUT_SEC, assess_obj.idea_id,
                self.sentinel._failed_publishes
            )
            raise
        except Exception as exc:
            self.sentinel._failed_publishes += 1
            ctx.logger.error(
                '[%s] Failed to publish signal: %s (failed_publishes=%d)',
                self.sentinel.component_id, exc, self.sentinel._failed_publishes,
                exc_info=True
            )
            raise

    # ------------------------------------------------------------------ #
    # 2. Adaptation-parameter change event
    # ------------------------------------------------------------------ #
    async def publish_adaptation_event(
        self,
        param_type: str,
        new_value: Any,
        context: NireonExecutionContext,
    ) -> None:
        bus = self.sentinel.event_bus
        if bus is None:
            return

        try:
            event_type = f"sentinel_{param_type}_updated"
            event_data = {
                "event_type": event_type,
                "component_id": self.sentinel.component_id,
                param_type: new_value,
                "run_id": context.run_id,
                "step": context.step,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            bus.publish("component_adaptation_events", event_data)
        except Exception as exc:  # pragma: no cover
            context.logger.warning(
                "[%s] Failed to publish adaptation event: %s",
                self.sentinel.component_id,
                exc,
                exc_info=True,
            )

    # ------------------------------------------------------------------ #
    # 3. Generic lifecycle event
    # ------------------------------------------------------------------ #
    async def publish_lifecycle_event(
        self,
        event_type: str,
        context: NireonExecutionContext,
        additional_data: Optional[dict] = None,
    ) -> None:
        bus = self.sentinel.event_bus
        if bus is None:
            return

        try:
            event_data = {
                "component_id": self.sentinel.component_id,
                "event_type": event_type,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "run_id": context.run_id,
                "step": context.step,
            }
            if additional_data:
                event_data.update(additional_data)

            bus.publish("component_lifecycle_event", event_data)
        except Exception as exc:  # pragma: no cover
            context.logger.error(
                "[%s] Error publishing %s event: %s",
                self.sentinel.component_id,
                event_type,
                exc,
                exc_info=True,
            )