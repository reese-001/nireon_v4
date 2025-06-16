from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

# Import the default weights from the constants file. This is good practice
# as it centralizes these default values if they need to be used elsewhere.
from .constants import DEFAULT_ALIGNMENT_WEIGHT, DEFAULT_FEASIBILITY_WEIGHT, DEFAULT_NOVELTY_WEIGHT

class SentinelMechanismConfig(BaseModel):
    """
    Configuration for the SentinelMechanism, validated by Pydantic.
    This model serves as the single source of truth for default values.
    """
    # --- HIGH-PRIORITY FIX APPLIED ---
    # Added inner Config class to forbid extra fields, hardening the config parsing
    # as per the V4 specification guidelines (§ 3.5, § 4.4).
    class Config:
        extra = "forbid"
    # ---------------------------------

    trust_threshold: float = Field(
        default=5.0,
        ge=0.0,
        le=10.0,
        description="The minimum trust score an idea must have to be considered stable."
    )

    min_axis_score: float = Field(
        default=4.0,
        ge=0.0,
        le=10.0,
        description="The minimum score an idea must achieve on any single axis (alignment, feasibility, novelty)."
    )

    enable_length_penalty: bool = Field(
        default=True,
        description="Whether to penalize excessively long ideas."
    )

    length_penalty_threshold: int = Field(
        default=1500,
        ge=0,
        description="Character count above which length penalties start to apply."
    )

    length_penalty_factor: float = Field(
        default=0.1,
        ge=0.0,
        description="Severity of the length penalty. Higher values penalize more."
    )

    enable_edge_trust: bool = Field(
        default=True,
        description="Whether to consider graph-based trust from parent/child ideas."
    )

    edge_trust_decay: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Penalty factor applied per step of distance from a supporting edge."
    )

    edge_support_boost: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Bonus applied to trust score if there is direct edge support."
    )

    enable_progression_adjustment: bool = Field(
        default=False,
        description="Whether to apply a bonus for ideas that show iterative progress."
    )

    progression_adjustment_min_step: int = Field(
        default=5,
        ge=0,
        description="The simulation step after which progression bonuses can be applied."
    )

    progression_adjustment_bonus_factor: float = Field(
        default=0.1,
        ge=0.0,
        description="Bonus points added per step past the minimum step."
    )

    progression_adjustment_bonus_cap: float = Field(
        default=1.0,
        ge=0.0,
        le=10.0,
        description="The maximum bonus that can be applied from progression adjustment."
    )

    weights: List[float] = Field(
        default_factory=lambda: [DEFAULT_ALIGNMENT_WEIGHT, DEFAULT_FEASIBILITY_WEIGHT, DEFAULT_NOVELTY_WEIGHT],
        description="Weights for [alignment, feasibility, novelty]. Must have 3 elements."
    )

    objective_override: Optional[str] = Field(
        default=None,
        description="If set, this string overrides any objective passed in the context for evaluation."
    )

    @classmethod
    def get_default_dict(cls) -> Dict[str, Any]:
        """Returns the default configuration as a dictionary."""
        return cls().model_dump()