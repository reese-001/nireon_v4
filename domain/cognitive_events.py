# nireon/domain/cognitive_events.py
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Literal

from domain.epistemic_stage import EpistemicStage

SERVICE_CALL_TYPE = Literal['LLM_ASK', 'EVENT_PUBLISH', 'CONTEXT_SNAPSHOT_GET']

@dataclass
class LLMRequestPayload:
    prompt: str
    stage: EpistemicStage
    role: str
    llm_settings: Optional[Dict[str, Any]] = field(default_factory=dict)
    target_model_route: Optional[str] = None

@dataclass
class CognitiveEvent:
    # Non-default arguments first
    frame_id: str
    owning_agent_id: str
    service_call_type: SERVICE_CALL_TYPE
    payload: LLMRequestPayload | Dict[str, Any]

    # Default arguments follow
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    epistemic_intent: Optional[str] = None
    trust_hint: Optional[float] = None
    response_timeout_ms: Optional[int] = None
    created_ts: float = field(default_factory=time.time)
    gateway_received_ts: Optional[float] = None
    custom_metadata: Dict[str, Any] = field(default_factory=dict)
    schema_version: str = '1.0'
    # is_reflexive: bool = False # From design doc, add if needed
    # t_star_proximity: float | None = None # From design doc, add if needed


    @classmethod
    def for_llm_ask(
        cls,
        frame_id: str,
        owning_agent_id: str,
        prompt: str,
        stage: EpistemicStage,
        role: str,
        llm_settings: Optional[Dict[str, Any]] = None,
        target_model_route: Optional[str] = None,
        intent: Optional[str] = None,
        # Allow passing other optional CE fields directly
        event_id: Optional[str] = None,
        trust_hint: Optional[float] = None,
        response_timeout_ms: Optional[int] = None,
        custom_metadata: Optional[Dict[str, Any]] = None

    ) -> CognitiveEvent:
        payload_obj = LLMRequestPayload(
            prompt=prompt,
            stage=stage,
            role=role,
            llm_settings=llm_settings or {},
            target_model_route=target_model_route
        )
        
        # Prepare kwargs for CognitiveEvent constructor, handling optional None values
        ce_kwargs = {
            "frame_id": frame_id,
            "owning_agent_id": owning_agent_id,
            "service_call_type": 'LLM_ASK',
            "payload": payload_obj,
            "epistemic_intent": intent
        }
        if event_id is not None:
            ce_kwargs["event_id"] = event_id
        if trust_hint is not None:
            ce_kwargs["trust_hint"] = trust_hint
        if response_timeout_ms is not None:
            ce_kwargs["response_timeout_ms"] = response_timeout_ms
        if custom_metadata is not None:
            ce_kwargs["custom_metadata"] = custom_metadata
            
        return cls(**ce_kwargs)