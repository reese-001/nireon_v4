#  nireon_v4/signals/core.py
#  Optimised 2025-06-22 – public interface unchanged
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import Field, field_validator

from .base import EpistemicSignal
from core.tracing import BlockTrace

# ---------------------------------------------------------------------- #
# Reusable enums (optional, but safer than string Literals everywhere)
# ---------------------------------------------------------------------- #
class LoopAction(str, Enum):
    CONTINUE = 'continue'
    PAUSE = 'pause'
    STOP = 'stop'


# NEW: Add completion status enum
class BranchCompletionStatus(str, Enum):
    """Represents the completion state of an exploration branch."""
    TERMINAL_SUCCESS = 'terminal_success'  # Branch completed successfully with results
    TERMINAL_FAILURE = 'terminal_failure'  # Branch failed with error
    TERMINAL_NO_OP = 'terminal_no_op'     # Branch completed but no operation performed (e.g., not viable)
    CONTINUE = 'continue'                   # Branch should continue processing


# ---------------------------------------------------------------------- #
# Domain-specific signals
# ---------------------------------------------------------------------- #
class SeedSignal(EpistemicSignal):
    signal_type: Literal["SeedSignal"] = "SeedSignal"
    seed_content: str = Field("", description="Initial seed content to begin epistemic processing.")

    def __init__(self, **data: Any):  # noqa: D401
        # Default seeds are assumed *fully* trusted by definition
        data.setdefault("confidence", 1.0)
        data.setdefault("trust_score", 1.0)
        super().__init__(**data)


class LoopSignal(EpistemicSignal):
    signal_type: Literal["LoopSignal"] = "LoopSignal"
    loop_action: LoopAction = Field(
        LoopAction.CONTINUE,
        description="Control flag for the main execution loop.",
    )
    loop_metadata: Dict[str, Any] = Field(default_factory=dict, description="Opaque runtime metadata.")


class IdeaGeneratedSignal(EpistemicSignal):
    signal_type: Literal["IdeaGeneratedSignal"] = "IdeaGeneratedSignal"
    idea_id: str = Field(..., description="Unique identifier of the generated idea.")
    idea_content: str = Field(..., description="Natural-language description of the idea.")
    generation_method: str = Field("unknown", description="Mechanism that generated the idea.")


class TrustAssessmentSignal(EpistemicSignal):
    """
    Trust assessments historically used a 0–10 scale, so we keep that
    range **for backward compatibility** but also expose a normalised
    0–1 view for consistency with other signals.
    """

    signal_type: Literal["TrustAssessmentSignal"] = "TrustAssessmentSignal"
    target_id: str = Field(..., description="ID of the entity being assessed.")
    target_type: str = Field(..., description="Type of entity being assessed.")
    assessment_rationale: Optional[str] = Field(None, description="Explanation for the trust judgement.")
    trust_score: Optional[float] = Field(None, ge=0.0, le=10.0, description="Trust score (0–10).")

    # Derived helper for consumers expecting the 0–1 convention
    @property
    def trust_score_normalised(self) -> Optional[float]:
        return None if self.trust_score is None else self.trust_score / 10.0

    # Validation rule: if rationales are present, score must be set
    @field_validator("assessment_rationale")
    @classmethod
    def _rationale_requires_score(cls, v: str, info):  # noqa: D401, ANN001
        if v and info.data.get("trust_score") is None:
            raise ValueError("An assessment_rationale requires trust_score to be provided")
        return v


class StagnationDetectedSignal(EpistemicSignal):
    signal_type: Literal["StagnationDetectedSignal"] = "StagnationDetectedSignal"
    stagnation_metrics: Dict[str, float] = Field(default_factory=dict)
    suggested_interventions: List[str] = Field(default_factory=list)


class ErrorSignal(EpistemicSignal):
    signal_type: Literal["ErrorSignal"] = "ErrorSignal"
    error_type: str = Field(..., description="Type/category of the error.")
    error_message: str = Field(..., description="Human-readable error message.")
    error_context: Dict[str, Any] = Field(default_factory=dict)
    recoverable: bool = Field(True)

    def raise_if_fatal(self) -> None:
        """Utility to raise self as an exception if not recoverable."""
        if not self.recoverable:
            raise RuntimeError(f"[{self.error_type}] {self.error_message}")


class GenerativeLoopFinishedSignal(EpistemicSignal):
    signal_type: Literal['GenerativeLoopFinishedSignal'] = 'GenerativeLoopFinishedSignal'
    
    completion_status: Optional[BranchCompletionStatus] = Field(
        None, 
        description="The completion status of this branch"
    )
    completion_reason: Optional[str] = Field(
        None, 
        description="Human-readable reason for the completion status"
    )
    # Keep these for backward compatibility, but they're now optional
    status: Optional[str] = Field(
        None,
        description="Legacy status field for backward compatibility"
    )
    quantifier_triggered: Optional[bool] = Field(
        None,
        description="Legacy field indicating if quantifier was triggered"
    )
# ---------------------------------------------------------------------- #
# Math dialect signals
# ---------------------------------------------------------------------- #
class MathQuerySignal(EpistemicSignal):
    signal_type: Literal["MathQuerySignal"] = "MathQuerySignal"
    natural_language_query: str
    expression: str
    operations: List[Dict[str, Any]]


class MathResultSignal(EpistemicSignal):
    signal_type: Literal["MathResultSignal"] = "MathResultSignal"
    natural_language_query: str
    explanation: str
    computation_details: Dict[str, Any]


# ---------------------------------------------------------------------- #
# Proto-dialect signals
# ---------------------------------------------------------------------- #
class ProtoTaskSignal(EpistemicSignal):
    """Carries a Proto block for execution by a ProtoEngine."""
    signal_type: Literal["ProtoTaskSignal"] = "ProtoTaskSignal"
    proto_block: Dict[str, Any]
    execution_priority: int = Field(5, ge=1, le=10)
    dialect: str


class ProtoResultSignal(EpistemicSignal):
    """Successful execution of a Proto block."""
    signal_type: Literal["ProtoResultSignal"] = "ProtoResultSignal"
    proto_block_id: str
    dialect: str
    success: bool
    result: Any
    artifacts: List[str] = Field(default_factory=list)
    execution_time_sec: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ProtoErrorSignal(EpistemicSignal):
    """Failure during Proto block validation or execution."""
    signal_type: Literal["ProtoErrorSignal"] = "ProtoErrorSignal"
    proto_block_id: str
    dialect: str
    error_type: str
    error_message: str
    execution_context: Dict[str, Any] = Field(default_factory=dict)


class MathProtoResultSignal(ProtoResultSignal):
    """Specialised result signal for the *math* dialect."""
    dialect: Literal["math"] = "math"
    equation_latex: Optional[str] = None
    numeric_result: Optional[Union[float, List[float], Dict[str, float]]] = None

# Tracing Signals
class PlanNextStepSignal(EpistemicSignal):
    """
    Signal to instruct the active planner to choose the next action.
    This is the entry point to the learning-driven part of the loop.
    """
    signal_type: Literal['PlanNextStepSignal'] = 'PlanNextStepSignal'
    session_id: str
    current_idea_id: str
    current_idea_text: str
    current_trust_score: float
    current_depth: int
    objective: str


class TraceEmittedSignal(EpistemicSignal):
    """
    Broadcasts a completed BlockTrace for sinks and monitoring components.
    """
    signal_type: Literal['TraceEmittedSignal'] = 'TraceEmittedSignal'
    trace: BlockTrace