# nireon_v4/core/tracing.py
from __future__ import annotations
import uuid
from typing import Optional, Dict, Any, List
from uuid import UUID
from datetime import datetime
from pydantic import BaseModel, Field

class BlockTrace(BaseModel):
    """
    An immutable record of a single, completed generative-evaluative step.
    This serves as the basis for reinforcement learning.
    """
    trace_id: UUID = Field(default_factory=uuid.uuid4)
    session_id: str
    
    # Context (State)
    parent_idea_id: Optional[str] = None
    parent_trust_score: Optional[float] = None
    parent_depth: int = 0
    
    # Action
    planner_policy_id: str
    chosen_action: str
    chosen_mechanism_id: str
    
    # Outcome
    generated_idea_id: str
    generated_trust_score: float
    
    # Reward Signal (calculated by the sink)
    reward: Optional[float] = None
    
    # Cost
    duration_ms: float
    llm_calls: int = 0
    
    # NEW (from review): Context for Idea Space principles
    frame_id: Optional[str] = None
    interpreter_set: Optional[List[str]] = None
    
    # Metadata
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    trace_metadata: Dict[str, Any] = Field(default_factory=dict)