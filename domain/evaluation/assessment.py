"""
Defines the core data structures for representing the assessment of an idea
within the NIREON system. These models are used by evaluation mechanisms
like Sentinel to provide structured feedback.
"""
from __future__ import annotations
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, confloat
from datetime import datetime, timezone


class AxisScore(BaseModel):
    """
    Represents the score and explanation for a single evaluation axis.
    """
    name: str = Field(..., description="The name of the evaluation axis (e.g., 'align', 'feas', 'novel').")
    score: confloat(ge=1.0, le=10.0) = Field(..., description="The score on this axis, typically from 1.0 to 10.0.") # type: ignore
    explanation: Optional[str] = Field(None, description="The rationale or evidence supporting this score.")

    class Config:
        validate_assignment = True


class IdeaAssessment(BaseModel):
    """
    Represents the complete assessment result for a single idea, including
    the final trust score, stability, and breakdown by axis.
    """
    idea_id: str = Field(..., description="The unique identifier of the idea that was assessed.")
    trust_score: confloat(ge=0.0, le=10.0) = Field(..., description="The final, weighted trust score for the idea.") # type: ignore
    is_stable: bool = Field(..., description="Indicates if the idea passed all thresholds and is considered stable.")
    rejection_reason: Optional[str] = Field(None, description="A summary of why the idea was deemed unstable, if applicable.")
    axis_scores: List[AxisScore] = Field(default_factory=list, description="A list of scores for each evaluation axis.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata related to the assessment process (e.g., weights used).")
    assessment_timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="The UTC timestamp when the assessment was created.")

    class Config:
        validate_assignment = True


__all__ = ['AxisScore', 'IdeaAssessment']