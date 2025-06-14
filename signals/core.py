from __future__ import annotations
from pydantic import Field
from .base import EpistemicSignal
from typing import List, Literal, Optional, Dict, Any

class SeedSignal(EpistemicSignal):
    """Initial signal to bootstrap system execution."""
    signal_type: Literal['SeedSignal'] = 'SeedSignal'
    
    # Seed-specific fields
    seed_content: str = Field(
        default="",
        description="Initial seed content to begin epistemic processing."
    )
    
    def __init__(self, **data):
        # Ensure seed signals have high confidence by default
        if 'confidence' not in data:
            data['confidence'] = 1.0
        super().__init__(**data)


class LoopSignal(EpistemicSignal):
    """Signal to control execution loops."""
    signal_type: Literal['LoopSignal'] = 'LoopSignal'
    
    loop_action: Literal['continue', 'pause', 'stop'] = Field(
        default='continue',
        description="Action to take for the execution loop."
    )
    loop_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata about the current loop state."
    )


class IdeaGeneratedSignal(EpistemicSignal):
    """Signal emitted when a new idea is generated."""
    signal_type: Literal['IdeaGeneratedSignal'] = 'IdeaGeneratedSignal'
    
    idea_id: str = Field(
        ...,
        description="Unique identifier of the generated idea."
    )
    idea_content: str = Field(
        ...,
        description="The actual content of the generated idea."
    )
    generation_method: str = Field(
        default="unknown",
        description="Method used to generate the idea (e.g., 'exploration', 'synthesis')."
    )


class TrustAssessmentSignal(EpistemicSignal):
    """Signal emitted when trust is assessed for an idea or component."""
    signal_type: Literal['TrustAssessmentSignal'] = 'TrustAssessmentSignal'
    
    target_id: str = Field(
        ...,
        description="ID of the entity being assessed (idea, component, etc.)."
    )
    target_type: str = Field(
        ...,
        description="Type of entity being assessed."
    )
    assessment_rationale: Optional[str] = Field(
        default=None,
        description="Explanation for the trust assessment."
    )
    
    def __init__(self, **data):
        # Trust assessment signals should include their trust score
        if 'trust_score' not in data:
            raise ValueError("TrustAssessmentSignal requires a trust_score")
        super().__init__(**data)


class StagnationDetectedSignal(EpistemicSignal):
    """Signal emitted when epistemic stagnation is detected."""
    signal_type: Literal['StagnationDetectedSignal'] = 'StagnationDetectedSignal'
    
    stagnation_metrics: Dict[str, float] = Field(
        default_factory=dict,
        description="Metrics indicating the level and type of stagnation."
    )
    suggested_interventions: List[str] = Field(
        default_factory=list,
        description="Suggested actions to break stagnation."
    )


class ErrorSignal(EpistemicSignal):
    """Signal emitted when an error occurs in processing."""
    signal_type: Literal['ErrorSignal'] = 'ErrorSignal'
    
    error_type: str = Field(
        ...,
        description="Type/category of the error."
    )
    error_message: str = Field(
        ...,
        description="Human-readable error message."
    )
    error_context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional context about the error."
    )
    recoverable: bool = Field(
        default=True,
        description="Whether the error is recoverable."
    )