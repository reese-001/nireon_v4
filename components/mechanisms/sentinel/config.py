# nireon_v4/components/mechanisms/sentinel/config.py
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from .constants import (
    DEFAULT_ALIGNMENT_WEIGHT,
    DEFAULT_FEASIBILITY_WEIGHT,
    DEFAULT_NOVELTY_WEIGHT
)

class SentinelMechanismConfig(BaseModel):
    class Config:
        extra = 'forbid'

    # --- Thresholds & Axis Weights ---
    trust_threshold: float = Field(5.0, ge=0.0, le=10.0,
        description="The minimum trust score an idea must have to be considered stable.")
    min_axis_score: float = Field(4.0, ge=0.0, le=10.0,
        description="The minimum score an idea must achieve on any single axis (alignment, feasibility, novelty).")

    # --- Length Penalty ---
    enable_length_penalty: bool = Field(True,
        description="Whether to penalize excessively long ideas.")
    length_penalty_threshold: int = Field(1500, ge=0,
        description="Character count above which length penalties start to apply.")
    length_penalty_factor: float = Field(0.1, ge=0.0,
        description="Severity of the length penalty. Higher values penalize more.")

    # --- Edge Trust ---
    enable_edge_trust: bool = Field(True,
        description="Whether to consider graph-based trust from parent/child ideas.")
    edge_trust_decay: float = Field(0.05, ge=0.0, le=1.0,
        description="Penalty factor applied per step of distance from a supporting edge.")
    edge_support_boost: float = Field(0.1,  ge=0.0, le=1.0,
        description="Bonus applied to trust score if there is direct edge support.")

    # --- Progression Bonus ---
    enable_progression_adjustment: bool = Field(True,
        description="Whether to apply a bonus for ideas that show iterative progress.")
    progression_adjustment_min_step: int = Field(5, ge=0,
        description="The simulation step after which progression bonuses can be applied.")
    progression_adjustment_bonus_factor: float = Field(0.1, ge=0.0,
        description="Bonus points added per step past the minimum step.")
    progression_adjustment_bonus_cap: float = Field(1.0, ge=0.0, le=10.0,
        description="The maximum bonus that can be applied from progression adjustment.")

    # --- Axis Weights ---
    weights: List[float] = Field(
        default_factory=lambda: [
            DEFAULT_ALIGNMENT_WEIGHT,
            DEFAULT_FEASIBILITY_WEIGHT,
            DEFAULT_NOVELTY_WEIGHT,
        ],
        description="Weights for [alignment, feasibility, novelty]. Must have 3 elements."
    )

    # +++ ADDED: Configurable cap for reference ideas (addresses Issue #4) +++
    max_reference_ideas: int = Field(20, ge=5, le=100,
        description="Maximum number of parent/sibling ideas to fetch for novelty scoring.")

    # --- Misc ---
    objective_override: Optional[str] = Field(None,
        description="If set, this string overrides any objective passed in the context for evaluation.")

    default_llm_score_on_error: float = Field(
        5.0, 
        ge=1.0, 
        le=10.0,
        description='Score to use when LLM response parsing fails. This ensures assessments can continue even with LLM issues.'
    )

    @classmethod
    def get_default_dict(cls) -> Dict[str, Any]:
        return cls().model_dump()