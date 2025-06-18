# nireon_v4\components\mechanisms\catalyst\service_helpers\catalyst_event_helper.py
import logging
from typing import Dict, Optional, Any, List, Tuple

from domain.cognitive_events import CognitiveEvent
from domain.ports.mechanism_gateway_port import MechanismGatewayPort
from domain.context import NireonExecutionContext
from domain.ports.idea_service_port import IdeaServicePort
from domain.ideas.idea import Idea

logger = logging.getLogger(__name__)

class CatalystEventHelper:
    def __init__(self, gateway: MechanismGatewayPort, owning_agent_id: str, component_version: str, *, idea_service: IdeaServicePort):
        if gateway is None:
            raise ValueError('MechanismGatewayPort cannot be None for CatalystEventHelper')
        if idea_service is None:
            raise ValueError('IdeaServicePort cannot be None for CatalystEventHelper')
        
        self.gateway = gateway
        self.owning_agent_id = owning_agent_id
        self.component_version = component_version
        self._idea_service = idea_service

    def create_and_persist_idea(self, text: str, parent_id: Optional[str], context: NireonExecutionContext, metadata: Dict[str, Any]) -> Idea:
        """
        Creates an Idea instance and persists it via the IdeaService.
        """
        idea = self._idea_service.create_idea(
            text=text, 
            parent_id=parent_id,
            context=context,
            metadata=metadata,
            method=self.owning_agent_id
        )
        return idea

    async def publish_signal(self, frame_id: str, signal_type_name: str, signal_payload: Dict[str, Any], context: NireonExecutionContext, epistemic_intent: Optional[str] = None, custom_metadata: Optional[Dict[str, Any]] = None) -> None:
        if not frame_id:
            logger.error(f"[{self.owning_agent_id}] Frame ID is required to publish signal '{signal_type_name}'.")
            return

        ce_payload = {
            'event_type': signal_type_name,
            'event_data': signal_payload
        }
        
        ce_custom_metadata = {
            'publisher_component_id': self.owning_agent_id,
            'publisher_version': self.component_version,
            'catalyst_specific': True,
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
            
    async def publish_batch_signals(self, frame_id: str, signals: List[Tuple[str, Dict[str, Any]]], context: NireonExecutionContext, epistemic_intent_prefix: str = 'BATCH') -> int:
        """Publishes a batch of signals."""
        successful = 0
        for signal_type, payload in signals:
            try:
                await self.publish_signal(
                    frame_id=frame_id,
                    signal_type_name=signal_type,
                    signal_payload=payload,
                    context=context,
                    epistemic_intent=f'{epistemic_intent_prefix}_{signal_type}'
                )
                successful += 1
            except Exception as e:
                logger.error(f"[{self.owning_agent_id}] Failed to publish signal '{signal_type}' in batch: {e}")
        return successful