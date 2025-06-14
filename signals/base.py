# nireon_v4/signals/base.py

from __future__ import annotations
from pydantic import BaseModel, Field, field_validator, field_serializer, ConfigDict 
from datetime import datetime, timezone
import uuid
from typing import Dict, Any, Optional, List

class EpistemicSignal(BaseModel):
    model_config = ConfigDict(
        extra='allow',
    )

    signal_type: str = Field(..., description="The type of the signal (e.g., 'IdeaGenerated', 'TrustAssessment').")
    source_node_id: str = Field(..., description='The unique ID of the component that emitted the signal.')
    payload: Dict[str, Any] = Field(default_factory=dict, description='The data payload of the signal.')
    trust_score: Optional[float] = Field(default=None, ge=0.0, le=1.0, description='Trust/confidence score for the signal content (0-1).')
    novelty_score: Optional[float] = Field(default=None, ge=0.0, le=1.0, description='Novelty score indicating how new/unique the signal content is (0-1).')
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0, description='Confidence level of the emitting component (0-1).')
    signal_id: str = Field(default_factory=lambda: f'sig_{uuid.uuid4()}', description='A unique identifier for this specific signal instance.')
    parent_signal_ids: List[str] = Field(default_factory=list, description='IDs of parent signals that led to this signal (for lineage tracking).')
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description='The UTC timestamp of when the signal was created.')
    context_tags: Dict[str, Any] = Field(default_factory=dict, description='Additional context tags for filtering or analysis.')

    @field_validator('timestamp')
    @classmethod
    def ensure_utc_timezone(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v
    
    # <<< ADD A SERIALIZER TO REPLACE json_encoders
    @field_serializer('timestamp')
    def serialize_dt(self, dt: datetime, _info):
        return dt.isoformat()

    def with_lineage(self, parent_signals: List['EpistemicSignal']) -> 'EpistemicSignal':
        parent_ids = [sig.signal_id for sig in parent_signals]
        return self.copy(update={'parent_signal_ids': parent_ids})

    def is_high_confidence(self, threshold: float=0.7) -> bool:
        scores = [s for s in [self.trust_score, self.confidence] if s is not None]
        return all((s >= threshold for s in scores)) if scores else False

    def is_novel(self, threshold: float=0.7) -> bool:
        return self.novelty_score >= threshold if self.novelty_score is not None else False

