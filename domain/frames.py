from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import time

@dataclass
class Frame:
    # Fields without defaults first
    id: str
    name: str
    description: str
    owner_agent_id: str
    created_ts: float
    updated_ts: float

    # Fields with defaults
    parent_frame_id: Optional[str] = None
    epistemic_goals: List[str] = field(default_factory=list)
    trust_basis: Dict[str, float] = field(default_factory=dict)
    llm_policy: Optional[Dict[str, Any]] = field(default_factory=dict)
    resource_budget: Optional[Dict[str, Any]] = None
    domain_rules: List[Any] = field(default_factory=list)
    context_tags: Dict[str, Any] = field(default_factory=dict)
    status: str = "active"
    expires_at: Optional[float] = None
    schema_version: str = "1.0"

    def is_active(self) -> bool:
        """Checks if the frame is currently active."""
        return self.status == 'active'

    def update_status(self, new_status: str) -> None:
        """Updates the frame's status and its updated_ts."""
        if self.status == new_status:
            return
        self.status = new_status
        self.updated_ts = time.time()