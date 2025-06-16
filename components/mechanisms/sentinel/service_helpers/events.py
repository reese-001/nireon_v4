from __future__ import annotations
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Optional

from domain.context import NireonExecutionContext
from domain.evaluation.assessment import IdeaAssessment
from domain.ideas.idea import Idea
from signals.core import TrustAssessmentSignal # FIX: Import the correct signal class

if TYPE_CHECKING:
    from ..service import SentinelMechanism

logger = logging.getLogger(__name__)

class EventPublisher:
    def __init__(self, sentinel: 'SentinelMechanism'):
        self.sentinel = sentinel

    def publish_assessment_event(self, assess_obj: IdeaAssessment, idea: Idea, ctx: NireonExecutionContext) -> None:
        if not self.sentinel.event_bus:
            ctx.logger.warning(f'[{self.sentinel.component_id}] Event bus not available, skipping assessment event publication.')
            return
        try:
            # FIX: Create and publish a proper TrustAssessmentSignal object
            # This is what the Explorer is waiting for.
            
            # Extract the frame_id from the context the Sentinel used for its assessment
            frame_id = ctx.metadata.get('current_frame_id')

            signal_payload = {
                'idea_id': assess_obj.idea_id,
                'is_stable': assess_obj.is_stable,
                'rejection_reason': assess_obj.rejection_reason,
                'assessment_details': assess_obj.model_dump(),
            }
            
            trust_assessment_signal = TrustAssessmentSignal(
                source_node_id=self.sentinel.component_id,
                target_id=assess_obj.idea_id,
                target_type='Idea',
                trust_score=assess_obj.trust_score,
                assessment_rationale=assess_obj.rejection_reason,
                payload=signal_payload,
                context_tags={'frame_id': frame_id} # FIX: Pass the frame_id in the tags
            )

            # The event bus expects the signal object itself as the payload
            self.sentinel.event_bus.publish(TrustAssessmentSignal.__name__, trust_assessment_signal)
            
            ctx.logger.info(f"[{self.sentinel.component_id}] Published TrustAssessmentSignal for idea '{idea.idea_id}' in frame '{frame_id}'.")

        except Exception as e:
            ctx.logger.warning(f'[{self.sentinel.component_id}] Failed to publish assessment event: {e}', exc_info=True)


    async def publish_adaptation_event(self, param_type: str, new_value: Any, context: NireonExecutionContext) -> None:
        if not self.sentinel.event_bus:
            return
        try:
            event_type = f'sentinel_{param_type}_updated'
            event_data = {
                'event_type': event_type,
                'component_id': self.sentinel.component_id,
                param_type: new_value,
                'run_id': context.run_id,
                'step': context.step,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            # This is likely a custom event, so publishing a dict is fine.
            self.sentinel.event_bus.publish('component_adaptation_events', event_data)
        except Exception as e:
            context.logger.warning(f'[{self.sentinel.component_id}] Failed to publish adaptation event: {e}', exc_info=True)


    async def publish_lifecycle_event(self, event_type: str, context: NireonExecutionContext, additional_data: Optional[dict] = None) -> None:
        if not self.sentinel.event_bus:
            return
        try:
            event_data = {
                'component_id': self.sentinel.component_id,
                'event_type': event_type,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'run_id': context.run_id,
                'step': context.step,
            }
            if additional_data:
                event_data.update(additional_data)
            # This is likely a custom event, so publishing a dict is fine.
            self.sentinel.event_bus.publish('component_lifecycle_event', event_data)
        except Exception as e:
            context.logger.error(f'[{self.sentinel.component_id}] Error publishing {event_type} event: {e}', exc_info=True)