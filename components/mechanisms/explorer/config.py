"""
Explorer‑mechanism configuration model (v4.1)

This module defines :class:`ExplorerConfig`, the **single source of truth** for every
tunable parameter that controls divergence‑based exploration inside NIREON.  It is a
strict, self‑documenting Pydantic model that guarantees:

* **Backwards compatibility** – every field that existed in v4.0 is still here with the
  same name, type, and default.
* **Forward extensibility**   – ergonomic helpers (`ModelExtra`, `from_yaml`, `copy_for`)
  make it trivial to script or compose configs without ad‑hoc boilerplate.
* **Strictness by default**   – illegal/unknown keys raise immediately (`extra = "forbid"`).
* **Runtime safety**          – inexpensive validators protect against silent
  mis‑configuration while keeping the hot path zero‑cost.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any, Dict, List, Literal, Optional, Sequence

from pydantic import BaseModel, Field, validator

__all__: Sequence[str] = ("ExplorerConfig",)


class ExplorerConfig(BaseModel):
    """
    Configuration for the *Explorer* mechanism.

    For quick programmatic access, prefer the `get_exploration_params()` helper or
    the rich `copy_for(**overrides)` constructor.
    """

    # -------------------------------------------------------------------------
    # Core exploration knobs
    # -------------------------------------------------------------------------
    divergence_strength: float = Field(
        0.2,
        ge=0.001,
        le=1.0,
        description="Base mutation strength for vector perturbation.",
    )
    exploration_timeout_seconds: float = Field(
        60.0,
        ge=1.0,
        le=300.0,
        description="Timeout (s) for a single exploration operation.",
    )
    max_depth: int = Field(
        3,
        ge=1,
        le=10,
        description="Maximum recursion depth (levels) to explore.",
    )
    application_rate: float = Field(
        0.5,
        ge=0.0,
        le=1.0,
        description="Probability of applying exploration to an idea.",
    )
    exploration_strategy: Literal[
        "depth_first", "breadth_first", "random", "llm_guided"
    ] = Field("depth_first", description="Strategy for traversing the idea space.")
    max_variations_per_level: int = Field(
        3,
        ge=1,
        le=20,
        description="Number of candidate variations to generate per depth level.",
    )
    enable_semantic_exploration: bool = Field(
        True, description="Use embedding distance to drive semantic exploration."
    )
    enable_llm_enhancement: bool = Field(
        True, description="Allow LLM to refine or expand generated ideas."
    )
    creativity_factor: float = Field(
        0.7,
        ge=0.0,
        le=1.0,
        description="`temperature`‑like multiplier for LLM prompts (1.0 = max).",
    )
    seed_randomness: Optional[int] = Field(
        None,
        description="Random seed for deterministic runs (None = nondeterministic).",
    )
    minimum_idea_length: int = Field(
        10,
        ge=5,
        le=1000,
        description="Lower bound on generated idea length (tokens/words).",
    )
    maximum_idea_length: int = Field(
        500,
        ge=50,
        le=5000,
        description="Upper bound on generated idea length (tokens/words).",
    )
    enable_diversity_filter: bool = Field(
        True, description="Reject candidates that are too similar to their seed."
    )
    diversity_threshold: float = Field(
        0.3,
        ge=0.0,
        le=1.0,
        description="Cosine/semantic distance below which a variation is discarded.",
    )

    # -------------------------------------------------------------------------
    # LLM / resource governance
    # -------------------------------------------------------------------------
    default_llm_policy_for_exploration: Dict[str, Any] = Field(
        default_factory=lambda: {
            "temperature": 0.7,
            "max_tokens": 250,
            "model_preference": "balanced",
        },
        description="LLM policy template for new *ExplorationFrame*s.",
    )
    default_resource_budget_for_exploration: Dict[str, Any] = Field(
        default_factory=lambda: {
            "llm_calls": 10,
            "embedding_calls": 20,
            "cpu_seconds": 300,
        },
        description="Initial soft budget (may be clipped by the orchestrator).",
    )
    max_parallel_llm_calls_per_frame: int = Field(
        3,
        ge=1,
        le=10,
        description="Concurrency guard for LLM calls inside a single frame.",
    )
    embedding_response_timeout_s: float = Field(
        30.0,
        ge=1.0,
        le=120.0,
        description="Seconds to await an *EmbeddingComputedSignal* before fallback.",
    )

    # -------------------------------------------------------------------------
    # Retry / novelty tuning
    # -------------------------------------------------------------------------
    reperturb_multiplier: float = Field(
        1.5,
        ge=1.0,
        le=5.0,
        description="Factor by which perturbation is increased on each retry.",
    )
    max_retries_for_novelty: int = Field(
        4,
        ge=1,
        le=10,
        description="Attempts to achieve the desired novelty before giving up.",
    )
    min_novelty_threshold_for_acceptance: float = Field(
        0.1,
        ge=0.0,
        le=1.0,
        description="Minimum novelty score to accept a perturbed idea.",
    )
    default_prompt_template: Optional[str] = Field(
        (
            "Generate a creative variation of the following idea: "
            "'{seed_idea_text}'. Previous exploration distance: "
            "{vector_distance:.3f}. Objective: {objective}."
        ),
        description=(
            "Jinja‑style template available to LLM prompt builders. "
            "Named placeholders **must** remain unchanged for compatibility."
        ),
    )

    # -------------------------------------------------------------------------
    # Gateway & embedding orchestration
    # -------------------------------------------------------------------------
    preferred_gateway_ids: str = Field(
        "gw_llm_fast_default,gw_llm_main_backup",
        description="Comma‑separated priority list of MechanismGateway IDs.",
    )
    request_embeddings_for_variations: bool = Field(
        True, description="Trigger embedding requests for each new variation."
    )
    max_pending_embedding_requests: int = Field(
        10,
        ge=1,
        le=100,
        description="Back‑pressure guard against embedding queue overflow.",
    )
    embedding_request_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Extra metadata included in *EmbeddingRequestSignal*s.",
    )

    # -------------------------------------------------------------------------
    # Validators & helpers
    # -------------------------------------------------------------------------

    @validator("maximum_idea_length")
    def _validate_max_length(
        cls, v: int, values: Dict[str, Any]  # noqa: D401, N805 (Pydantic signature)
    ) -> int:
        min_len = values.get("minimum_idea_length")
        if min_len is not None and v < min_len:
            raise ValueError(
                "maximum_idea_length must be ≥ minimum_idea_length "
                f"(got {v} < {min_len})"
            )
        return v

    # --- convenience API ----------------------------------------------------

    def copy_for(self, **overrides: Any) -> "ExplorerConfig":
        """
        Lightweight wrapper around :pymeth:`pydantic.BaseModel.copy(update=...)`
        that returns *ExplorerConfig* ensuring correct type.
        """
        return self.copy(update=overrides)  # type: ignore[arg-type]

    @classmethod
    def from_yaml(cls, path: str | pathlib.Path) -> "ExplorerConfig":
        """Load configuration from a YAML or JSON file on disk."""
        import yaml  # lazy import to avoid hard dependency for JSON users

        with pathlib.Path(path).expanduser().open("r", encoding="utf‑8") as fh:
            data = yaml.safe_load(fh)
        return cls(**data)

    # Re‑export of original helper names – untouched for compatibility
    def get_exploration_params(self) -> Dict[str, Any]:  # noqa: D401
        return {
            "max_depth": self.max_depth,
            "strategy": self.exploration_strategy,
            "creativity": self.creativity_factor,
            "max_variations": self.max_variations_per_level,
            "timeout": self.exploration_timeout_seconds,
        }

    def is_llm_enabled(self) -> bool:  # noqa: D401
        return self.enable_llm_enhancement

    def is_semantic_enabled(self) -> bool:  # noqa: D401
        return self.enable_semantic_exploration

    def should_filter_diversity(self) -> bool:  # noqa: D401
        return self.enable_diversity_filter and self.diversity_threshold > 0.0

    # -------------------------------------------------------------------------
    # Pydantic model‑level settings
    # -------------------------------------------------------------------------
    class Config:
        extra = "forbid"
        validate_assignment = True
        use_enum_values = True
        orm_mode = True  # future‑proofing for DB‑backed configs
        json_encoders = {pathlib.Path: str}
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
                "preferred_gateway_ids": "primary_gateway,secondary_gateway",
            }
        }

    # -------------------------------------------------------------------------
    # Introspection utilities (out of hot path)
    # -------------------------------------------------------------------------
    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialize the config to a human‑friendly JSON string."""
        return self.json(indent=indent, sort_keys=True)

    def dump(self, path: str | pathlib.Path, *, indent: int | None = 2) -> None:
        """Persist the config to disk as pretty‑printed JSON."""
        p = pathlib.Path(path).expanduser()
        p.write_text(self.to_json(indent=indent), encoding="utf‑8")

    # keep the model hashable
    def __hash__(self) -> int:  # noqa: D401
        return hash(self.json())
