"""
Handles the react() and adapt() lifecycle phases for the SentinelMechanism.
This includes emitting signals based on assessment results and processing
adaptation requests to change its own parameters.
"""
from __future__ import annotations
import logging
from typing import TYPE_CHECKING, List, Optional

import numpy as np

# V4 CHANGE: Flattened import paths
from domain.context import NireonExecutionContext
from core.results import SystemSignal, AdaptationAction, SignalType, AdaptationActionType

if TYPE_CHECKING:
    from ..service import SentinelMechanism

logger = logging.getLogger(__name__)


class AdaptationHelper:
    """A helper class to encapsulate adaptation and reaction logic for Sentinel."""
    def __init__(self, sentinel: 'SentinelMechanism'):
        self.sentinel = sentinel

    async def handle_react_phase(self, context: NireonExecutionContext) -> List[SystemSignal]:
        """
        Generates SystemSignals based on the last assessment performed.
        This is the primary way Sentinel communicates its findings to the rest of the system.
        """
        signals: List[SystemSignal] = []
        assessment_data = context.get_custom_data('last_sentinel_assessment_object')

        if assessment_data and isinstance(assessment_data, dict):
            is_stable = assessment_data.get('is_stable')
            idea_id = assessment_data.get('idea_id')
            rejection_reason = assessment_data.get('rejection_reason')
            trust_score = assessment_data.get('trust_score', 0.0)

            if idea_id is not None:
                signal_type = SignalType.STATUS_UPDATE  # Use the V4 Enum
                priority = 2 if is_stable else 4  # Example priority levels
                
                payload = {
                    'idea_id': idea_id,
                    'is_stable': is_stable,
                    'trust_score': trust_score,
                    'rejection_reason': rejection_reason if not is_stable else None,
                    'full_assessment_summary': {k: v for k, v in assessment_data.items()}
                }

                msg = (
                    f"Idea '{idea_id}' assessed: {'Stable' if is_stable else 'Unstable'} "
                    f"(Trust: {trust_score:.2f})."
                )
                if not is_stable and rejection_reason:
                    msg += f" Reason: {rejection_reason[:100]}{'...' if len(rejection_reason) > 100 else ''}"
                
                # V4 CHANGE: Construct SystemSignal with the new structure
                signals.append(SystemSignal(
                    signal_type=signal_type,
                    component_id=self.sentinel.component_id,
                    message=msg,
                    payload=payload,
                    priority=priority
                ))
        else:
            context.logger.debug(
                f"[{self.sentinel.component_id}] No 'last_sentinel_assessment_object' found in context for react phase."
            )

        return signals

    async def adapt(self, context: NireonExecutionContext) -> List[AdaptationAction]:
        """
        Proposes adaptations to Sentinel's own configuration based on data
        placed in the execution context.
        """
        actions: List[AdaptationAction] = []

        requested_weights = context.get_custom_data('adapt_sentinel_weights_to')
        if isinstance(requested_weights, list):
            action = self.adapt_weights(requested_weights, context)
            if action:
                actions.append(action)

        requested_threshold = context.get_custom_data('adapt_sentinel_threshold_to')
        if isinstance(requested_threshold, (int, float)):
            action = self.adapt_threshold(float(requested_threshold), context)
            if action:
                actions.append(action)

        return actions

    def adapt_weights(self, req_weights: List[float], ctx: NireonExecutionContext) -> Optional[AdaptationAction]:
        """Handles a request to adapt the scoring weights."""
        try:
            # Basic validation
            if not (isinstance(req_weights, list) and len(req_weights) == 3):
                ctx.logger.warning(
                    f"[{self.sentinel.component_id}] adapt_weights: expected list[3], got {req_weights}"
                )
                return None

            try:
                raw = np.asarray(req_weights, dtype=float)
            except ValueError:
                ctx.logger.warning(
                    f"[{self.sentinel.component_id}] adapt_weights: non-numeric values {req_weights}"
                )
                return None

            if (raw < 0).any():
                ctx.logger.warning(
                    f"[{self.sentinel.component_id}] adapt_weights: negatives not allowed {req_weights}"
                )
                return None

            # Capture state before change
            old_live = self.sentinel.weights.tolist()

            # Step 1 – write raw request into config
            self.sentinel.sentinel_cfg.weights = raw.tolist()
            ctx.logger.debug(
                f"[{self.sentinel.component_id}] adapt_weights: cfg updated to raw {raw.tolist()}"
            )

            # Step 2 – re-normalise & apply via initialise helper
            self.sentinel._init_helper.initialize_weights()
            new_live = self.sentinel.weights.tolist()

            if np.allclose(old_live, new_live):
                ctx.logger.debug(
                    f"[{self.sentinel.component_id}] adapt_weights: no operational change (already {new_live})"
                )
                return None

            params = {
                "parameter_name": "axis_weights",
                "old_operational_value": old_live,
                "new_operational_value": new_live,
            }
            msg = f"Axis-weights adapted from {old_live} -> {new_live}"

            logger.info(f"[{self.sentinel.component_id}] {msg}")
            if self.sentinel.event_bus:
                self.sentinel._event_publisher.publish_adaptation_event("weights", new_live, ctx)

            # V4 CHANGE: Use AdaptationActionType enum
            return AdaptationAction(
                action_type=AdaptationActionType.PARAMETER_ADJUST,
                component_id=self.sentinel.component_id,
                description=msg,
                parameters=params
            )
        except Exception as e:
            ctx.logger.error(
                f"[{self.sentinel.component_id}] adapt_weights: unexpected error {e}",
                exc_info=True,
            )
            self.sentinel._error_count += 1
            return None

    def adapt_threshold(self, req_threshold: float, ctx: NireonExecutionContext) -> Optional[AdaptationAction]:
        """Handles a request to adapt the trust threshold."""
        try:
            if not 0.0 <= req_threshold <= 10.0:
                ctx.logger.warning(
                    f"[{self.sentinel.component_id}] adapt_threshold: {req_threshold} outside 0-10"
                )
                return None

            old_live = self.sentinel.trust_th
            if abs(req_threshold - old_live) < 1e-9:
                ctx.logger.debug(
                    f"[{self.sentinel.component_id}] adapt_threshold: no change from live value {old_live}"
                )
                return None

            self.sentinel.sentinel_cfg.trust_threshold = req_threshold
            self.sentinel.trust_th = req_threshold

            params = {
                "parameter_name": "trust_threshold",
                "old_value": old_live,
                "new_value": req_threshold,
            }
            msg = f"Trust-threshold adapted from {old_live:.2f} -> {req_threshold:.2f}"

            logger.info(f"[{self.sentinel.component_id}] {msg}")
            if self.sentinel.event_bus:
                self.sentinel._event_publisher.publish_adaptation_event("threshold", req_threshold, ctx)

            # V4 CHANGE: Use AdaptationActionType enum
            return AdaptationAction(
                action_type=AdaptationActionType.PARAMETER_ADJUST,
                component_id=self.sentinel.component_id,
                description=msg,
                parameters=params
            )
        except Exception as e:
            ctx.logger.error(
                f"[{self.sentinel.component_id}] adapt_threshold: unexpected error {e}",
                exc_info=True,
            )
            self.sentinel._error_count += 1
            return None