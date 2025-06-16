# nireon/domain/ports/mechanism_gateway_port.py
from __future__ import annotations
from typing import Protocol, Any, runtime_checkable 
from domain.cognitive_events import CognitiveEvent
from domain.ports.llm_port import LLMResponse
# from domain.context import ContextSnapshot # Uncomment if you add get_context_snapshot

@runtime_checkable 
class MechanismGatewayPort(Protocol):
    async def process_cognitive_event(self, ce: CognitiveEvent) -> Any:
        ...

    async def ask_llm(self, ce_llm_request: CognitiveEvent) -> LLMResponse: # Assuming ce_llm_request implies a CognitiveEvent specific to LLM asks
        ...

    async def publish_event(self, ce_event_publish: CognitiveEvent) -> None: # Assuming ce_event_publish implies a CognitiveEvent specific to event publishing
        ...
    
    async def get_context_snapshot(self, ce_snapshot_request: CognitiveEvent) -> Any: # Replace Any with actual ContextSnapshot type
        ...
