# events.py
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Final, Optional

from domain.cognitive_events import CognitiveEvent
from domain.context import NireonExecutionContext
from domain.evaluation.assessment import IdeaAssessment
from domain.ideas.idea import Idea
from signals.core import TrustAssessmentSignal

if TYPE_CHECKING:  # avoids run‑time import cycle
    from ..service import SentinelMechanism

logger: Final = logging.getLogger(__name__)

__all__ = ["EventPublisher"]


class EventPublisher:
    """Publishes Sentinel‑related signals and bookkeeping events."""

    def __init__(self, sentinel: "SentinelMechanism") -> None:
        self.sentinel = sentinel

    # ------------------------------------------------------------------ #
    # 1. Trust‑assessment signal via Gateway
    # ------------------------------------------------------------------ #
    async def publish_assessment_event(
        self,
        assess_obj: IdeaAssessment,
        idea: Idea,
        ctx: NireonExecutionContext,
    ) -> None:
        gateway = self.sentinel.gateway
        if gateway is None:
            ctx.logger.error("[%s] Gateway unavailable – skipping publish.", self.sentinel.component_id)
            return

        frame_id: Optional[str] = ctx.metadata.get("current_frame_id")
        if not frame_id:
            ctx.logger.error("[%s] current_frame_id missing – cannot publish TrustAssessmentSignal.", self.sentinel.component_id)
            return

        # ---------- diff‑required change: compute novelty_score -----------
        novelty_axis = next((ax for ax in assess_obj.axis_scores if ax.name == "novel"), None)
        novelty_score_val: Optional[float] = None
        if novelty_axis:
            novelty_score_val = novelty_axis.score / 10.0  # 1‑10 → 0‑1

        # Compact, top‑level payload
        signal_payload = {
            "idea_id": assess_obj.idea_id,
            "is_stable": assess_obj.is_stable,
            "rejection_reason": assess_obj.rejection_reason,
            "assessment_details": assess_obj.model_dump(),
        }

        trust_signal = TrustAssessmentSignal(
            source_node_id=self.sentinel.component_id,
            target_id=assess_obj.idea_id,
            target_type="Idea",
            trust_score=assess_obj.trust_score,
            novelty_score=novelty_score_val,              # <‑‑ NEW FIELD
            assessment_rationale=assess_obj.rejection_reason,
            payload=signal_payload,
            context_tags={"frame_id": frame_id},
        )

        ce_payload = {
            "event_type": trust_signal.signal_type,
            "event_data": trust_signal.model_dump(mode="json"),
        }
        ce_meta = {
            "publisher_component_id": self.sentinel.component_id,
            "publisher_version": self.sentinel.metadata.version,
        }

        cognitive_event = CognitiveEvent(
            frame_id=frame_id,
            owning_agent_id=self.sentinel.component_id,
            service_call_type="EVENT_PUBLISH",
            payload=ce_payload,
            epistemic_intent="PUBLISH_TRUST_ASSESSMENT",
            custom_metadata=ce_meta,
        )

        try:
            await gateway.process_cognitive_event(cognitive_event, ctx)
            ctx.logger.info(
                "[%s] TrustAssessmentSignal for idea '%s' published (frame %s).",
                self.sentinel.component_id,
                idea.idea_id,
                frame_id,
            )
        except Exception as exc:  # pragma: no cover
            ctx.logger.warning(
                "[%s] Failed to publish TrustAssessmentSignal: %s",
                self.sentinel.component_id,
                exc,
                exc_info=True,
            )

    # ------------------------------------------------------------------ #
    # 2. Adaptation‑parameter change event
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
