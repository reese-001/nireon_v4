# nireon_v4/components/mechanisms/explorer/config.py
from typing import Literal, Optional, Dict, Any, List
from pydantic import BaseModel, Field, validator

class ExplorerConfig(BaseModel):
    # Existing fields from original implementation
    divergence_strength: float = Field(default=0.2, ge=0.001, le=1.0, description="Base mutation strength for vector perturbation.")
    exploration_timeout_seconds: float = Field(default=60.0, ge=1.0, le=300.0, description="Timeout for a single exploration operation in seconds.")
    max_depth: int = Field(default=3, ge=1, le=10, description="Maximum depth for exploration (number of levels to explore).")
    application_rate: float = Field(default=0.5, ge=0.0, le=1.0, description="Rate at which exploration is applied (0.0 = minimal, 1.0 = maximum).")
    exploration_strategy: Literal['depth_first', 'breadth_first', 'random', 'llm_guided'] = Field(default='depth_first', description="Strategy for exploring the idea space.")
    max_variations_per_level: int = Field(default=3, ge=1, le=20, description="Maximum number of variations to generate per exploration level.")
    enable_semantic_exploration: bool = Field(default=True, description="Enable semantic-based exploration using embeddings.")
    enable_llm_enhancement: bool = Field(default=True, description="Enable LLM-based idea enhancement and variation generation.")
    creativity_factor: float = Field(default=0.7, ge=0.0, le=1.0, description="Creativity factor for LLM prompts (0.0 = conservative, 1.0 = highly creative).")
    seed_randomness: Optional[int] = Field(default=None, description="Random seed for reproducible exploration (None = random). Used for Frame RNG seeding.")
    minimum_idea_length: int = Field(default=10, ge=5, le=1000, description="Minimum length for generated idea text.")
    maximum_idea_length: int = Field(default=500, ge=50, le=5000, description="Maximum length for generated idea text.")
    enable_diversity_filter: bool = Field(default=True, description="Enable filtering to ensure diversity in generated variations.")
    diversity_threshold: float = Field(default=0.3, ge=0.0, le=1.0, description="Minimum diversity threshold for accepting variations (0.0 = no filter, 1.0 = maximum diversity).")

    # New fields from DESIGN_AFCE.md for A-F-CE integration
    default_llm_policy_for_exploration: Dict[str, Any] = Field(
        default_factory=lambda: {"temperature": 0.7, "max_tokens": 250, "model_preference": "balanced"},
        description="Default LLM policy to be used when creating ExplorationFrames. Can be overridden by MechanismGateway or parent Frame."
    )
    default_resource_budget_for_exploration: Dict[str, Any] = Field(
        default_factory=lambda: {"llm_calls": 10, "embedding_calls": 20, "cpu_seconds": 300},
        description="Desired initial resource budget for an ExplorationFrame. Subject to system constraints."
    )
    max_parallel_llm_calls_per_frame: int = Field(
        default=3, ge=1, le=10,
        description="Maximum number of concurrent LLM calls Explorer will make within a single Frame."
    )
    embedding_response_timeout_s: float = Field(
        default=30.0, ge=1.0, le=120.0,
        description="Timeout in seconds to wait for an EmbeddingComputedSignal before fallback."
    )

    # From legacy Explorer config, adapted for V4
    reperturb_multiplier: float = Field(1.5, ge=1.0, le=5.0, description='Factor to increase perturbation on retries if novelty threshold not met.')
    max_retries_for_novelty: int = Field(4, ge=1, le=10, description='Max attempts to find a novel vector by re-perturbing.')
    min_novelty_threshold_for_acceptance: float = Field(0.1, ge=0.0, le=1.0, description='Minimum novelty score required for an idea to be accepted after perturbation.')
    default_prompt_template: Optional[str] = Field(
        default="Generate a creative variation of the following idea: '{seed_idea_text}'. Consider its novelty and divergence. The previous exploration attempt had a vector distance of {vector_distance:.3f}. Aim for {objective}.",
        description="Default prompt template for LLM-based idea generation. Can use placeholders like {seed_idea_text}, {vector_distance}, {objective}."
    )

    # Preferred gateway IDs
    preferred_gateway_ids: str = Field(
        default="gw_llm_fast_default,gw_llm_main_backup",
        description="Comma-separated list of preferred MechanismGateway IDs. Explorer will try to use them in order."
    )

    # Add the missing fields that are in DESIGN_AFCE.md and used in your manifest/code
    request_embeddings_for_variations: bool = Field(
        default=True, 
        description="Whether to request embeddings for newly generated idea variations."
    )
    max_pending_embedding_requests: int = Field(
        default=10, 
        ge=1, 
        le=100, 
        description="Maximum number of embedding requests Explorer can have pending."
    )
    embedding_request_metadata: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Additional metadata to include in EmbeddingRequestSignals."
    )

    @validator('maximum_idea_length')
    def validate_max_length(cls, v, values):
        if 'minimum_idea_length' in values and v < values['minimum_idea_length']:
            raise ValueError('maximum_idea_length must be greater than or equal to minimum_idea_length')
        return v

    def get_exploration_params(self) -> dict:
        return {
            'max_depth': self.max_depth,
            'strategy': self.exploration_strategy,
            'creativity': self.creativity_factor,
            'max_variations': self.max_variations_per_level,
            'timeout': self.exploration_timeout_seconds
        }

    def is_llm_enabled(self) -> bool:
        return self.enable_llm_enhancement

    def is_semantic_enabled(self) -> bool:
        return self.enable_semantic_exploration

    def should_filter_diversity(self) -> bool:
        return self.enable_diversity_filter and self.diversity_threshold > 0.0

    class Config:
        extra = 'forbid'
        validate_assignment = True
        use_enum_values = True
        json_schema_extra = {
            "example": {
                "divergence_strength": 0.2,
                "application_rate": 0.5,
                "exploration_strategy": "depth_first",
                "max_variations_per_level": 3,
                "default_llm_policy_for_exploration": {"temperature": 0.75},
                "default_resource_budget_for_exploration": {"llm_calls": 5},
                "max_parallel_llm_calls_per_frame": 2,
                "embedding_response_timeout_s": 20.0,
                "reperturb_multiplier": 1.2,
                "max_retries_for_novelty": 3,
                "min_novelty_threshold_for_acceptance": 0.15,
                "preferred_gateway_ids": "primary_gateway,secondary_gateway"
            }
        }