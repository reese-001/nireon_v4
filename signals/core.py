from __future__ import annotations
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import Field, field_validator
from .base import EpistemicSignal
from core.tracing import BlockTrace

class LoopAction(str, Enum):
    CONTINUE = 'continue'
    PAUSE = 'pause'
    STOP = 'stop'

class BranchCompletionStatus(str, Enum):
    TERMINAL_SUCCESS = 'terminal_success'
    TERMINAL_FAILURE = 'terminal_failure'
    TERMINAL_NO_OP = 'terminal_no_op'
    CONTINUE = 'continue'

class SeedSignal(EpistemicSignal):
    signal_type: Literal['SeedSignal'] = 'SeedSignal'
    seed_content: str = Field('', description='Initial seed content to begin epistemic processing.')
    def __init__(self, **data: Any):
        data.setdefault('confidence', 1.0)
        data.setdefault('trust_score', 1.0)
        super().__init__(**data)

class LoopSignal(EpistemicSignal):
    signal_type: Literal['LoopSignal'] = 'LoopSignal'
    loop_action: LoopAction = Field(LoopAction.CONTINUE, description='Control flag for the main execution loop.')
    loop_metadata: Dict[str, Any] = Field(default_factory=dict, description='Opaque runtime metadata.')

class IdeaGeneratedSignal(EpistemicSignal):
    signal_type: Literal['IdeaGeneratedSignal'] = 'IdeaGeneratedSignal'
    idea_id: str = Field(..., description='Unique identifier of the generated idea.')
    idea_content: str = Field(..., description='Natural-language description of the idea.')
    generation_method: str = Field('unknown', description='Mechanism that generated the idea.')

class TrustAssessmentSignal(EpistemicSignal):
    signal_type: Literal['TrustAssessmentSignal'] = 'TrustAssessmentSignal'
    target_id: str = Field(..., description='ID of the entity being assessed.')
    target_type: str = Field(..., description='Type of entity being assessed.')
    assessment_rationale: Optional[str] = Field(None, description='Explanation for the trust judgement.')
    trust_score: Optional[float] = Field(None, ge=0.0, le=10.0, description='Trust score (0-10).')

    @property
    def trust_score_normalised(self) -> Optional[float]:
        return None if self.trust_score is None else self.trust_score / 10.0

    @field_validator('assessment_rationale')
    @classmethod
    def _rationale_requires_score(cls, v: str, info):
        if v and info.data.get('trust_score') is None:
            raise ValueError('An assessment_rationale requires trust_score to be provided')
        return v

class StagnationDetectedSignal(EpistemicSignal):
    signal_type: Literal['StagnationDetectedSignal'] = 'StagnationDetectedSignal'
    stagnation_metrics: Dict[str, float] = Field(default_factory=dict)
    suggested_interventions: List[str] = Field(default_factory=list)

class ErrorSignal(EpistemicSignal):
    signal_type: Literal['ErrorSignal'] = 'ErrorSignal'
    error_type: str = Field(..., description='Type/category of the error.')
    error_message: str = Field(..., description='Human-readable error message.')
    error_context: Dict[str, Any] = Field(default_factory=dict)
    recoverable: bool = Field(True)

    def raise_if_fatal(self) -> None:
        if not self.recoverable:
            raise RuntimeError(f'[{self.error_type}] {self.error_message}')

class GenerativeLoopFinishedSignal(EpistemicSignal):
    signal_type: Literal['GenerativeLoopFinishedSignal'] = 'GenerativeLoopFinishedSignal'
    completion_status: Optional[BranchCompletionStatus] = Field(None, description='The completion status of this branch')
    completion_reason: Optional[str] = Field(None, description='Human-readable reason for the completion status')
    status: Optional[str] = Field(None, description='Legacy status field for backward compatibility')
    quantifier_triggered: Optional[bool] = Field(None, description='Legacy field indicating if quantifier was triggered')

class MathQuerySignal(EpistemicSignal):
    signal_type: Literal['MathQuerySignal'] = 'MathQuerySignal'
    natural_language_query: str
    expression: str
    operations: List[Dict[str, Any]]
    latex_definition: Optional[str] = Field(None, description='LaTeX representation of the formal definition.')

class MathResultSignal(EpistemicSignal):
    signal_type: Literal['MathResultSignal'] = 'MathResultSignal'
    natural_language_query: str
    explanation: str
    computation_details: Dict[str, Any]

class FormalResultSignal(EpistemicSignal):
    signal_type: Literal['FormalResultSignal'] = 'FormalResultSignal'
    original_idea_id: str = Field(..., description="The ID of the narrative idea that was formalized.")
    original_query: str = Field(..., description="The natural language query that initiated the formalization.")
    definition: str = Field(..., description="The formal mathematical definition derived from the idea.")
    computation_table: Optional[Dict[str, Any]] = Field(None, description="A table of computed values, if applicable.")
    result_extract: str = Field(..., description="A concise summary of the main computational result.")
    monotonicity_analysis: Optional[str] = Field(None, description="Analysis of the function's behavior (e.g., increasing, decreasing).")
    conclusion: str = Field(..., description="The final conclusion, relating the formal result back to the original idea.")
    is_proof_valid: Optional[bool] = Field(None, description="True if a proof was validated, False if invalidated, None if not a proof.")
    computation_details: Dict[str, Any] = Field(default_factory=dict, description="The raw computational steps and results from the math kernel.")

class ProtoTaskSignal(EpistemicSignal):
    signal_type: Literal['ProtoTaskSignal'] = 'ProtoTaskSignal'
    proto_block: Dict[str, Any]
    execution_priority: int = Field(5, ge=1, le=10)
    dialect: str

class ProtoResultSignal(EpistemicSignal):
    signal_type: Literal['ProtoResultSignal'] = 'ProtoResultSignal'
    proto_block_id: str
    dialect: str
    success: bool
    result: Any
    artifacts: List[str] = Field(default_factory=list)
    execution_time_sec: float
    metadata: Dict[str, Any] = Field(default_factory=dict)

class ProtoErrorSignal(EpistemicSignal):
    signal_type: Literal['ProtoErrorSignal'] = 'ProtoErrorSignal'
    proto_block_id: str
    dialect: str
    error_type: str
    error_message: str
    execution_context: Dict[str, Any] = Field(default_factory=dict)

class MathProtoResultSignal(ProtoResultSignal):
    dialect: Literal['math'] = 'math'
    equation_latex: Optional[str] = None
    numeric_result: Optional[Union[float, List[float], Dict[str, float]]] = None

class PlanNextStepSignal(EpistemicSignal):
    signal_type: Literal['PlanNextStepSignal'] = 'PlanNextStepSignal'
    session_id: str
    current_idea_id: str
    current_idea_text: str
    current_trust_score: float
    current_depth: int
    objective: str

    @field_validator('session_id', mode='before')
    @classmethod
    def validate_session_id(cls, v: Any) -> str:
        return v if v is not None else 'unknown_session'

    @field_validator('current_depth', mode='before')
    @classmethod
    def validate_depth(cls, v: Any) -> int:
        return v if v is not None else 0

class TraceEmittedSignal(EpistemicSignal):
    signal_type: Literal['TraceEmittedSignal'] = 'TraceEmittedSignal'
    trace: BlockTrace