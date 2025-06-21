# nireon_v4/signals/core.py
# process [math_engine, principia_agent]

from __future__ import annotations
from pydantic import Field
from .base import EpistemicSignal
from typing import List, Literal, Optional, Dict, Any, Union

# ... (all existing signals like SeedSignal, LoopSignal, etc. remain here) ...

class SeedSignal(EpistemicSignal):
    signal_type: Literal['SeedSignal'] = 'SeedSignal'
    seed_content: str = Field(default="", description="Initial seed content to begin epistemic processing.")
    
    def __init__(self, **data: Any):
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
    generation_method: str = Field(default='unknown', description="Method used to generate the idea (e.g., 'exploration', 'synthesis').")

class TrustAssessmentSignal(EpistemicSignal):
    signal_type: Literal['TrustAssessmentSignal'] = 'TrustAssessmentSignal'
    target_id: str = Field(..., description="ID of the entity being assessed (idea, component, etc.).")
    target_type: str = Field(..., description="Type of entity being assessed.")
    assessment_rationale: Optional[str] = Field(default=None, description="Explanation for the trust assessment.")
    trust_score: Optional[float] = Field(default=None, ge=0.0, le=10.0, description="Trust/confidence score for the signal content (0-10).")

    def __init__(self, **data: Any):
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

class MathQuerySignal(EpistemicSignal):
    signal_type: Literal['MathQuerySignal'] = 'MathQuerySignal'
    natural_language_query: str = Field(description="The original user query in natural language.")
    expression: str = Field(description="The mathematical expression to be computed, formatted for the tool (e.g., SymPy).")
    operations: List[Dict[str, Any]] = Field(description="A structured list of operations to perform on the expression.")

# NEW: The signal our test script will wait for
class MathResultSignal(EpistemicSignal):
    signal_type: Literal['MathResultSignal'] = 'MathResultSignal'
    natural_language_query: str = Field(description="The original user query.")
    explanation: str = Field(description="The final, LLM-generated explanation of the result.")
    computation_details: Dict[str, Any] = Field(description="The raw computational result from the MathPort.")

class ProtoTaskSignal(EpistemicSignal):
    """Signal carrying a Proto block for type expansion and execution by a ProtoEngine."""
    signal_type: Literal['ProtoTaskSignal'] = 'ProtoTaskSignal'
    proto_block: Dict[str, Any] = Field(description="The raw Proto block dictionary to be executed.")
    execution_priority: int = Field(default=5, ge=1, le=10, description="Execution priority (1-10, higher is more important).")
    dialect: str = Field(description="The Proto dialect (eidos value) of the block.")

class ProtoResultSignal(EpistemicSignal):
    """Signal carrying the results from a successful Proto block execution."""
    signal_type: Literal['ProtoResultSignal'] = 'ProtoResultSignal'
    proto_block_id: str = Field(description="ID of the executed Proto block for correlation.")
    dialect: str = Field(description="The Proto dialect that was executed.")
    success: bool = Field(description="Flag indicating if the execution was successful.")
    result: Any = Field(description="The primary result returned by the executed function.")
    artifacts: List[str] = Field(default_factory=list, description="List of paths to generated artifacts (e.g., plots, data files).")
    execution_time_sec: float = Field(description="Total execution duration in seconds.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata from the execution.")

class ProtoErrorSignal(EpistemicSignal):
    """Signal indicating a failure during Proto block validation or execution."""
    signal_type: Literal['ProtoErrorSignal'] = 'ProtoErrorSignal'
    proto_block_id: str = Field(description="ID of the failed Proto block.")
    dialect: str = Field(description="The Proto dialect that was attempted.")
    error_type: str = Field(description="Category of error (e.g., 'validation', 'execution', 'timeout', 'budget_exceeded').")
    error_message: str = Field(description="Detailed error message or traceback snippet.")
    execution_context: Dict[str, Any] = Field(default_factory=dict, description="Contextual information at the time of failure.")



class MathProtoResultSignal(ProtoResultSignal):
    """A specialized result signal for the 'math' dialect with additional structured fields."""
    dialect: Literal['math'] = 'math'
    equation_latex: Optional[str] = Field(default=None, description="LaTeX representation of the primary equation, if applicable.")
    numeric_result: Optional[Union[float, List[float], Dict[str, float]]] = Field(default=None, description="Structured numeric result, if computable.")
