# nireon_v4/components/mechanisms/catalyst/service_helpers/catalyst_event_helper.py

import logging
from typing import Dict, Optional, Any
from domain.cognitive_events import CognitiveEvent
from domain.ports.mechanism_gateway_port import MechanismGatewayPort
from domain.context import NireonExecutionContext

logger = logging.getLogger(__name__)

class CatalystEventHelper:
    """Helper class to publish signals via the MechanismGateway."""
    def __init__(self, gateway: MechanismGatewayPort, owning_agent_id: str, component_version: str):
        if gateway is None:
            raise ValueError('MechanismGatewayPort cannot be None for CatalystEventHelper')
        self.gateway = gateway
        self.owning_agent_id = owning_agent_id
        self.component_version = component_version

    async def publish_signal(
        self,
        frame_id: str,
        signal_type_name: str,
        signal_payload: Dict[str, Any],
        context: NireonExecutionContext,
        epistemic_intent: Optional[str] = None,
        custom_metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Publishes a signal by creating and processing an EVENT_PUBLISH Cognitive Event."""
        if not frame_id:
            logger.error(f"[{self.owning_agent_id}] Frame ID is required to publish signal '{signal_type_name}'.")
            return

        ce_payload = {'event_type': signal_type_name, 'event_data': signal_payload}
        ce_custom_metadata = {
            'publisher_component_id': self.owning_agent_id,
            'publisher_version': self.component_version,
            **(custom_metadata or {})
        }

        cognitive_event = CognitiveEvent(
            frame_id=frame_id,
            owning_agent_id=self.owning_agent_id,
            service_call_type='EVENT_PUBLISH',
            payload=ce_payload,
            epistemic_intent=epistemic_intent or f'PUBLISH_{signal_type_name.upper()}',
            custom_metadata=ce_custom_metadata
        )

        try:
            logger.debug(f"[{self.owning_agent_id}] Publishing '{signal_type_name}' via CE for Frame '{frame_id}'.")
            await self.gateway.process_cognitive_event(cognitive_event, context)
            logger.info(f"[{self.owning_agent_id}] Successfully published signal '{signal_type_name}' for Frame '{frame_id}'.")
        except Exception as e:
            logger.error(f"[{self.owning_agent_id}] Failed to publish signal '{signal_type_name}' for Frame '{frame_id}': {e}", exc_info=True)