"""
Handles the creation and publication of events and signals originating from
the Sentinel mechanism, ensuring consistent event schemas.
"""
from __future__ import annotations
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Optional

# V4 CHANGE: Flattened import paths
from domain.context import NireonExecutionContext
from domain.evaluation.assessment import IdeaAssessment
from domain.ideas.idea import Idea

if TYPE_CHECKING:
    from ..service import SentinelMechanism

logger = logging.getLogger(__name__)


class EventPublisher:
    """A helper class to encapsulate all event publication logic for Sentinel."""
    def __init__(self, sentinel: 'SentinelMechanism'):
        self.sentinel = sentinel

    def publish_assessment_event(self, assess_obj: IdeaAssessment, idea: Idea, ctx: NireonExecutionContext) -> None:
        if not self.sentinel.event_bus:
            ctx.logger.warning(f'[{self.sentinel.component_id}] Event bus not available, skipping assessment event publication.')
            return
        try:
            event_data = {
                'idea_id': idea.idea_id,
                'trust_score': assess_obj.trust_score,
                'is_stable': assess_obj.is_stable,
                'stage_assessed': assess_obj.metadata.get('stage_assessed', 'UNKNOWN') if assess_obj.metadata else 'UNKNOWN',
                'method_created': assess_obj.metadata.get('method_created', 'UNKNOWN') if assess_obj.metadata else 'UNKNOWN',
                'run_id': ctx.run_id,
                'step': ctx.step,
                'assessment_details': assess_obj.model_dump(),
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'component_id': self.sentinel.component_id
            }
            self.sentinel.event_bus.publish('idea_assessed', event_data)
        except Exception as e:
            ctx.logger.warning(f'[{self.sentinel.component_id}] Failed to publish assessment event: {e}', exc_info=True)

    async def publish_adaptation_event(
        self,
        param_type: str,
        new_value: Any,
        context: NireonExecutionContext
    ) -> None:
        """Publishes an event when Sentinel adapts its own parameters."""
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
            # V4 CHANGE: The event bus publish method is now async
            await self.sentinel.event_bus.publish('component_adaptation_events', event_data)

        except Exception as e:
            context.logger.warning(
                f'[{self.sentinel.component_id}] Failed to publish adaptation event: {e}',
                exc_info=True
            )

    async def publish_lifecycle_event(
        self,
        event_type: str,
        context: NireonExecutionContext,
        additional_data: Optional[dict] = None
    ) -> None:
        """Publishes a general lifecycle event for the Sentinel component."""
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

            # V4 CHANGE: The event bus publish method is now async
            await self.sentinel.event_bus.publish('component_lifecycle_event', event_data)

        except Exception as e:
            context.logger.error(
                f'[{self.sentinel.component_id}] Error publishing {event_type} event: {e}',
                exc_info=True
            )