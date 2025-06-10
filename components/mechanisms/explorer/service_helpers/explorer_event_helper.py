# nireon_v4/components/mechanisms/explorer/service_helpers/explorer_event_helper.py
import logging
from typing import Dict, Optional, Any

from domain.cognitive_events import CognitiveEvent
from domain.ports.mechanism_gateway_port import MechanismGatewayPort
from domain.context import NireonExecutionContext  # Import NireonExecutionContext

logger = logging.getLogger(__name__)

class ExplorerEventHelper:
    """
    A helper class for ExplorerMechanism to build and publish CognitiveEvents
    of type EVENT_PUBLISH via the MechanismGateway.
    """
    def __init__(self, gateway: MechanismGatewayPort, owning_agent_id: str, component_version: str):
        if gateway is None:
            raise ValueError("MechanismGatewayPort cannot be None for ExplorerEventHelper")
        self.gateway = gateway
        self.owning_agent_id = owning_agent_id
        self.component_version = component_version

    async def publish_signal(
        self,
        frame_id: str,
        signal_type_name: str,
        signal_payload: Dict[str, Any],
        context: NireonExecutionContext,  # Add context as an argument
        epistemic_intent: Optional[str] = None,
        custom_metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Constructs and publishes an EVENT_PUBLISH CognitiveEvent.

        Args:
            frame_id: The ID of the frame this event belongs to.
            signal_type_name: The specific type name of the signal being published (e.g., "IdeaGeneratedSignal").
                              This is used as 'event_type' in the CE payload.
            signal_payload: The actual data payload of the signal. This is used as 'event_data'.
            context: The NireonExecutionContext required by the gateway.
            epistemic_intent: The epistemic intent of publishing this event.
            custom_metadata: Additional custom metadata for the CognitiveEvent.
        """
        if not frame_id:
            logger.error(f"[{self.owning_agent_id}] Frame ID is required to publish signal '{signal_type_name}'.")
            return

        ce_payload = {
            "event_type": signal_type_name,
            "event_data": signal_payload
        }
        
        ce_custom_metadata = {
            "publisher_component_id": self.owning_agent_id,
            "publisher_version": self.component_version,
            **(custom_metadata or {})
        }

        cognitive_event = CognitiveEvent(
            frame_id=frame_id,
            owning_agent_id=self.owning_agent_id,
            service_call_type='EVENT_PUBLISH',
            payload=ce_payload,
            epistemic_intent=epistemic_intent or f"PUBLISH_{signal_type_name.upper()}",
            custom_metadata=ce_custom_metadata,
        )

        try:
            logger.debug(f"[{self.owning_agent_id}] Publishing signal '{signal_type_name}' via CE for Frame '{frame_id}'. CE ID: {cognitive_event.event_id}")
            # Pass the context to the gateway
            await self.gateway.process_cognitive_event(cognitive_event, context)  # Pass context
            logger.info(f"[{self.owning_agent_id}] Successfully published signal '{signal_type_name}' for Frame '{frame_id}'.")
        except Exception as e:
            logger.error(f"[{self.owning_agent_id}] Failed to publish signal '{signal_type_name}' for Frame '{frame_id}': {e}", exc_info=True)