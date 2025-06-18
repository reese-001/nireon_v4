# nireon_v4\components\mechanisms\explorer\config.py
from __future__ import annotations
import json
import pathlib
from typing import Any, Dict, List, Literal, Optional, Sequence
from pydantic import BaseModel, Field, validator

__all__: Sequence[str] = ('ExplorerConfig',)

class ExplorerConfig(BaseModel):
    divergence_strength: float = Field(..., ge=0.001, le=1.0, description='Base mutation strength for vector perturbation.')
    exploration_timeout_seconds: float = Field(..., ge=1.0, le=300.0, description='Timeout (s) for a single exploration operation.')
    max_depth: int = Field(..., ge=1, le=10, description='Maximum recursion depth (levels) to explore.')
    application_rate: float = Field(..., ge=0.0, le=1.0, description='Probability of applying exploration to an idea.')
    exploration_strategy: Literal['depth_first', 'breadth_first', 'random', 'llm_guided'] = Field(..., description='Strategy for traversing the idea space.')
    max_variations_per_level: int = Field(..., ge=1, le=20, description='Number of candidate variations to generate per depth level.')
    enable_semantic_exploration: bool = Field(..., description='Use embedding distance to drive semantic exploration.')
    enable_llm_enhancement: bool = Field(..., description='Allow LLM to refine or expand generated ideas.')
    creativity_factor: float = Field(..., ge=0.0, le=1.0, description='`temperature`‑like multiplier for LLM prompts (1.0 = max).')
    seed_randomness: Optional[int] = Field(..., description='Random seed for deterministic runs (None = nondeterministic).')
    minimum_idea_length: int = Field(..., ge=5, le=1000, description='Lower bound on generated idea length (tokens/words).')
    maximum_idea_length: int = Field(..., ge=50, le=5000, description='Upper bound on generated idea length (tokens/words).')
    enable_diversity_filter: bool = Field(..., description='Reject candidates that are too similar to their seed.')
    diversity_threshold: float = Field(..., ge=0.0, le=1.0, description='Cosine/semantic distance below which a variation is discarded.')
    default_resource_budget_for_exploration: Dict[str, Any] = Field(..., description='Initial soft budget (may be clipped by the orchestrator).')
    max_parallel_llm_calls_per_frame: int = Field(..., ge=1, le=10, description='Concurrency guard for LLM calls inside a single frame.')
    embedding_response_timeout_s: float = Field(..., ge=1.0, le=120.0, description='Seconds to await an *EmbeddingComputedSignal* before fallback.')
    reperturb_multiplier: float = Field(..., ge=1.0, le=5.0, description='Factor by which perturbation is increased on each retry.')
    max_retries_for_novelty: int = Field(..., ge=1, le=10, description='Attempts to achieve the desired novelty before giving up.')
    min_novelty_threshold_for_acceptance: float = Field(..., ge=0.0, le=1.0, description='Minimum novelty score to accept a perturbed idea.')
    default_prompt_template: Optional[str] = Field(..., description='Jinja‑style template available to LLM prompt builders. Named placeholders **must** remain unchanged for compatibility.')
    preferred_gateway_ids: str = Field(..., description='Comma‑separated priority list of MechanismGateway IDs.')
    request_embeddings_for_variations: bool = Field(..., description='Trigger embedding requests for each new variation.')
    max_pending_embedding_requests: int = Field(..., ge=1, le=100, description='Back‑pressure guard against embedding queue overflow.')
    embedding_request_metadata: Dict[str, Any] = Field(..., description='Extra metadata included in *EmbeddingRequestSignal*s.')

    @validator('maximum_idea_length')
    def _validate_max_length(cls, v: int, values: Dict[str, Any]) -> int:
        min_len = values.get('minimum_idea_length')
        if min_len is not None and v < min_len:
            raise ValueError(f'maximum_idea_length must be ≥ minimum_idea_length (got {v} < {min_len})')
        return v

    def copy_for(self, **overrides: Any) -> 'ExplorerConfig':
        return self.copy(update=overrides)

    @classmethod
    def from_yaml(cls, path: str | pathlib.Path) -> 'ExplorerConfig':
        import yaml
        with pathlib.Path(path).expanduser().open('r', encoding='utf‑8') as fh:
            data = yaml.safe_load(fh)
        return cls(**data)

    def get_exploration_params(self) -> Dict[str, Any]:
        return {
            'max_depth': self.max_depth,
            'strategy': self.exploration_strategy,
            'creativity': self.creativity_factor,
            'max_variations': self.max_variations_per_level,
            'timeout': self.exploration_timeout_seconds,
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
        from_attributes = True
        json_encoders = {
            pathlib.Path: str
        }