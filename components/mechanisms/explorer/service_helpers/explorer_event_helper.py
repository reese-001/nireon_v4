# Modified components\mechanisms\explorer\service_helpers\explorer_event_helper.py (full class with new method)

import asyncio
import logging
from typing import Dict, Optional, Any
from core.registry.component_registry import ComponentRegistry
from domain.cognitive_events import CognitiveEvent
from domain.ports.mechanism_gateway_port import MechanismGatewayPort
from domain.ideas.idea import Idea
from domain.ports.idea_service_port import IdeaServicePort
from domain.context import NireonExecutionContext
from signals.base import EpistemicSignal
from signals.core import IdeaGeneratedSignal  # Add this import

logger = logging.getLogger(__name__)

class ExplorerEventHelper:
    def __init__(self, gateway: MechanismGatewayPort, owning_agent_id: str, component_version: str, *, registry: ComponentRegistry | None=None, idea_service: IdeaServicePort | None=None):
        if gateway is None:
            raise ValueError('MechanismGatewayPort cannot be None for ExplorerEventHelper')
        self.gateway = gateway
        self.owning_agent_id = owning_agent_id
        self.component_version = component_version
        if idea_service is not None:
            self._idea_service = idea_service
        else:
            reg = registry
            if reg is None and hasattr(gateway, 'component_registry'):
                reg = gateway.component_registry
            if reg is None:
                raise RuntimeError('ExplorerEventHelper could not resolve IdeaService; pass it explicitly or provide a gateway with .component_registry.')
            try:
                self._idea_service = reg.get_service_instance(IdeaServicePort)
            except Exception:
                self._idea_service = reg.get('IdeaService')

    def create_and_persist_idea(self, text: str, parent_id: Optional[str], context: NireonExecutionContext, metadata: Optional[Dict[str, Any]]=None, idea_id: Optional[str]=None) -> Idea:
        # Pass idea_id to IdeaService if provided (e.g., for seeds)
        idea = self._idea_service.create_idea(text=text, parent_id=parent_id, context=context, metadata=metadata, idea_id=idea_id)
        self._publish_idea_generated(idea, context)
        return idea

    def _publish_idea_generated(self, idea: Idea, context: NireonExecutionContext) -> None:
        idea_payload = {
            'id': idea.idea_id,
            'text': idea.text,
            'parent_id': idea.parent_ids[0] if idea.parent_ids else None,
            'source_mechanism': self.owning_agent_id,
            'derivation_method': idea.method,
            'seed_idea_text_preview': idea.text[:50],
            'frame_id': context.metadata.get('current_frame_id'),
            'objective': idea.metadata.get('objective'),
            'metadata': idea.metadata
        }
        signal = IdeaGeneratedSignal(
            source_node_id=self.owning_agent_id,
            idea_id=idea.idea_id,
            idea_content=idea.text,
            generation_method=idea.method,
            payload=idea_payload,
            context_tags={'frame_id': context.metadata.get('current_frame_id')}
        )
        # Since publish_signal is async, run it in a task
        asyncio.create_task(self.publish_signal(signal, context))

    async def publish_signal(self, signal_to_publish: EpistemicSignal, context: NireonExecutionContext) -> None:
        frame_id = signal_to_publish.context_tags.get('frame_id')
        if not frame_id:
            logger.error(f"[{self.owning_agent_id}] Frame ID not found in signal context_tags. Cannot publish '{signal_to_publish.signal_type}'.")
            return
        ce_payload = {'event_type': signal_to_publish.signal_type, 'event_data': signal_to_publish.model_dump(mode='json')}
        ce_custom_metadata = {'publisher_component_id': self.owning_agent_id, 'publisher_version': self.component_version, 'original_signal_id': signal_to_publish.signal_id}
        cognitive_event = CognitiveEvent(frame_id=frame_id, owning_agent_id=self.owning_agent_id, service_call_type='EVENT_PUBLISH', payload=ce_payload, epistemic_intent=f'PUBLISH_{signal_to_publish.signal_type.upper()}', custom_metadata=ce_custom_metadata)
        try:
            logger.debug(f"[{self.owning_agent_id}] Publishing '{signal_to_publish.signal_type}' via CE for Frame '{frame_id}'. CE ID: {cognitive_event.event_id}")
            await self.gateway.process_cognitive_event(cognitive_event, context)
            logger.info(f"[{self.owning_agent_id}] Successfully published signal '{signal_to_publish.signal_type}' for Frame '{frame_id}'.")
        except Exception as e:
            logger.error(f"[{self.owning_agent_id}] Failed to publish signal '{signal_to_publish.signal_type}' for Frame '{frame_id}': {e}", exc_info=True)