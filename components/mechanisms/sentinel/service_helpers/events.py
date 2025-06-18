# nireon_v4\components\mechanisms\sentinel\service_helpers\events.py
from __future__ import annotations
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Optional

from domain.cognitive_events import CognitiveEvent
from domain.context import NireonExecutionContext
from domain.evaluation.assessment import IdeaAssessment
from domain.ideas.idea import Idea
from signals.core import TrustAssessmentSignal

if TYPE_CHECKING:
    from ..service import SentinelMechanism
logger = logging.getLogger(__name__)


class EventPublisher:
    def __init__(self, sentinel: 'SentinelMechanism'):
        self.sentinel = sentinel

    async def publish_assessment_event(self, assess_obj: IdeaAssessment, idea: Idea, ctx: NireonExecutionContext) -> None:
        if not self.sentinel.gateway:
            ctx.logger.error(f'[{self.sentinel.component_id}] Gateway not available for event publication.')
            return
        
        try:
            frame_id = ctx.metadata.get('current_frame_id')
            if not frame_id:
                ctx.logger.error(f"[{self.sentinel.component_id}] Frame ID missing, cannot publish TrustAssessmentSignal.")
                return

            # Construct the signal object
            signal_payload = {'idea_id': assess_obj.idea_id, 'is_stable': assess_obj.is_stable, 'rejection_reason': assess_obj.rejection_reason, 'assessment_details': assess_obj.model_dump()}
            trust_assessment_signal = TrustAssessmentSignal(
                source_node_id=self.sentinel.component_id,
                target_id=assess_obj.idea_id,
                target_type='Idea',
                trust_score=assess_obj.trust_score,
                assessment_rationale=assess_obj.rejection_reason,
                payload=signal_payload,
                context_tags={'frame_id': frame_id}
            )

            # Create a CognitiveEvent to publish the signal via the gateway
            ce_payload = {'event_type': trust_assessment_signal.signal_type, 'event_data': trust_assessment_signal.model_dump(mode='json')}
            ce_custom_metadata = {'publisher_component_id': self.sentinel.component_id, 'publisher_version': self.sentinel.metadata.version}
            cognitive_event = CognitiveEvent(
                frame_id=frame_id, 
                owning_agent_id=self.sentinel.component_id, 
                service_call_type='EVENT_PUBLISH', 
                payload=ce_payload, 
                epistemic_intent='PUBLISH_TRUST_ASSESSMENT', 
                custom_metadata=ce_custom_metadata
            )
            
            await self.sentinel.gateway.process_cognitive_event(cognitive_event, ctx)
            ctx.logger.info(f"[{self.sentinel.component_id}] Published TrustAssessmentSignal for idea '{idea.idea_id}' via Gateway in frame '{frame_id}'.")
        except Exception as e:
            ctx.logger.warning(f'[{self.sentinel.component_id}] Failed to publish assessment event via Gateway: {e}', exc_info=True)


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
            self.sentinel.event_bus.publish('component_adaptation_events', event_data)
        except Exception as e:
            context.logger.warning(f'[{self.sentinel.component_id}] Failed to publish adaptation event: {e}', exc_info=True)


    async def publish_lifecycle_event(self, event_type: str, context: NireonExecutionContext, additional_data: Optional[dict]=None) -> None:
        if not self.sentinel.event_bus:
            return

        try:
            event_data = {
                'component_id': self.sentinel.component_id,
                'event_type': event_type,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'run_id': context.run_id,
                'step': context.step
            }
            if additional_data:
                event_data.update(additional_data)
            self.sentinel.event_bus.publish('component_lifecycle_event', event_data)
        except Exception as e:
            context.logger.error(f'[{self.sentinel.component_id}] Error publishing {event_type} event: {e}', exc_info=True)