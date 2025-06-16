from __future__ import annotations
from pydantic import Field
from .base import EpistemicSignal
from typing import List, Literal, Optional, Dict, Any

class SeedSignal(EpistemicSignal):
    signal_type: Literal['SeedSignal'] = 'SeedSignal'
    seed_content: str = Field(default="", description="Initial seed content to begin epistemic processing.")
    
    def __init__(self, **data):
        if 'confidence' not in data:
            data['confidence'] = 1.0
        super().__init__(**data)

class LoopSignal(EpistemicSignal):
    signal_type: Literal['LoopSignal'] = 'LoopSignal'
    loop_action: Literal['continue', 'pause', 'stop'] = Field(default='continue', description="Action to take for the execution loop.")
    loop_metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata about the current loop state.")

class IdeaGeneratedSignal(EpistemicSignal):
    signal_type: Literal['IdeaGeneratedSignal'] = 'IdeaGeneratedSignal'
    idea_id: str = Field(..., description="Unique identifier of the generated idea.")
    idea_content: str = Field(..., description="The actual content of the generated idea.")
    generation_method: str = Field(default="unknown", description="Method used to generate the idea (e.g., 'exploration', 'synthesis').")

class TrustAssessmentSignal(EpistemicSignal):
    signal_type: Literal['TrustAssessmentSignal'] = 'TrustAssessmentSignal'
    target_id: str = Field(..., description="ID of the entity being assessed (idea, component, etc.).")
    target_type: str = Field(..., description="Type of entity being assessed.")
    assessment_rationale: Optional[str] = Field(default=None, description="Explanation for the trust assessment.")
    
    # FIX: Change the trust_score range from 0-1 to 0-10 to match the Sentinel's output.
    trust_score: Optional[float] = Field(default=None, ge=0.0, le=10.0, description="Trust/confidence score for the signal content (0-10).")

    def __init__(self, **data):
        # The base signal uses a 0-1 scale, but this specific signal uses 0-10.
        # We can remove the check here as it's now handled by the Field validator.
        super().__init__(**data)


class StagnationDetectedSignal(EpistemicSignal):
    signal_type: Literal['StagnationDetectedSignal'] = 'StagnationDetectedSignal'
    stagnation_metrics: Dict[str, float] = Field(default_factory=dict, description="Metrics indicating the level and type of stagnation.")
    suggested_interventions: List[str] = Field(default_factory=list, description="Suggested actions to break stagnation.")

class ErrorSignal(EpistemicSignal):
    signal_type: Literal['ErrorSignal'] = 'ErrorSignal'
    error_type: str = Field(..., description="Type/category of the error.")
    error_message: str = Field(..., description="Human-readable error message.")
    error_context: Dict[str, Any] = Field(default_factory=dict, description="Additional context about the error.")
    recoverable: bool = Field(default=True, description="Whether the error is recoverable.")

class GenerativeLoopFinishedSignal(EpistemicSignal):
    signal_type: Literal['GenerativeLoopFinishedSignal'] = 'GenerativeLoopFinishedSignal'