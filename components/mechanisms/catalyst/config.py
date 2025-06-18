# nireon_v4/components/mechanisms/catalyst/config.py
"""Configuration for the Catalyst mechanism."""
from typing import Optional, Dict, Any, ClassVar
from pydantic import BaseModel, Field, validator

class CatalystMechanismConfig(BaseModel):
    """Configuration for the Catalyst mechanism.
    
    The Catalyst mechanism performs cross-domain concept injection for creative synthesis
    by blending agent ideas with pre-encoded domain vectors.
    """
    
    # Core blending parameters
    application_rate: float = Field(
        0.2,  # Updated to match 20% from spec
        ge=0.0, 
        le=1.0,
        description='Chance of applying catalyst to an agent (20% default per spec)'
    )
    
    blend_low: float = Field(
        0.1, 
        ge=0.0, 
        le=1.0,
        description='Minimum blend strength for cross-domain injection'
    )
    
    blend_high: float = Field(
        0.3, 
        ge=0.0, 
        le=1.0,
        description='Maximum blend strength for cross-domain injection'
    )
    
    # Advanced blending parameters
    max_blend_low: float = Field(
        0.8, 
        ge=0.0, 
        le=1.0,
        description='Maximum allowed value for blend_low during adaptation'
    )
    
    max_blend_high: float = Field(
        0.95, 
        ge=0.0, 
        le=1.0,
        description='Maximum allowed value for blend_high during adaptation'
    )
    
    MIN_BLEND_GAP: ClassVar[float] = 0.01
    
    # Duplication detection
    duplication_check_enabled: bool = Field(
        False,
        description='Enable detection of semantic duplication in generated ideas'
    )
    
    duplication_check_probability: float = Field(
        0.05, 
        ge=0.0, 
        le=1.0,
        description='Probability of checking for duplication per catalysis'
    )
    
    duplication_cooldown_steps: int = Field(
        5, 
        ge=0,
        description='Steps before resetting blend range after duplication detection'
    )
    
    duplication_aggressiveness: float = Field(
        0.5, 
        ge=0.0, 
        le=5.0,
        description='How aggressively to adjust blend range on duplication (0.5 = moderate)'
    )
    
    # Anti-constraints for diversity
    anti_constraints_enabled: bool = Field(
        False,
        description='Enable anti-constraints to force diversity when semantic distance is low'
    )
    
    anti_constraints_count: int = Field(
        3, 
        ge=1, 
        le=10,
        description='Maximum number of anti-constraints to apply'
    )
    
    anti_constraints_diversity_threshold: float = Field(
        0.15, 
        ge=0.0, 
        le=1.0,
        description='Semantic distance threshold below which anti-constraints activate'
    )
    
    # LLM integration
    prompt_template: Optional[str] = Field(
        None,
        description='Custom prompt template for LLM-based text regeneration after blending'
    )
    
    # Frame-based processing
    default_llm_policy_for_catalysis: Dict[str, Any] = Field(
        default_factory=lambda: {
            'temperature': 0.75,
            'max_tokens': 300,
            'model_preference': 'creative'
        },
        description='LLM policy for CatalystFrame cognitive events'
    )
    
    default_resource_budget_for_catalysis: Dict[str, Any] = Field(
        default_factory=lambda: {
            'llm_calls': 20,
            'event_publishes': 40,
            'cpu_seconds': 300
        },
        description='Resource budget for catalysis task frames'
    )
    
    # Domain vector management
    normalize_domain_vectors: bool = Field(
        True,
        description='Whether to normalize domain vectors before blending'
    )
    
    preserve_semantic_direction: bool = Field(
        True,
        description='Whether to preserve original semantic direction during blending'
    )
    
    @validator('blend_high')
    def blend_high_greater_than_low(cls, v, values):
        """Ensure blend_high > blend_low."""
        if 'blend_low' in values and v <= values['blend_low']:
            raise ValueError(
                f"blend_high ({v}) must be greater than blend_low ({values['blend_low']})"
            )
        return v
    
    @validator('max_blend_high')
    def max_blend_high_greater_than_max_low(cls, v, values):
        """Ensure max_blend_high > max_blend_low."""
        if 'max_blend_low' in values and v <= values['max_blend_low']:
            raise ValueError(
                f"max_blend_high ({v}) must be greater than max_blend_low "
                f"({values['max_blend_low']})"
            )
        return v
    
    @validator('anti_constraints_count')
    def validate_anti_constraints_count(cls, v):
        """Validate anti_constraints_count is positive."""
        if v < 1:
            raise ValueError('anti_constraints_count must be at least 1')
        return v
    
    @validator('duplication_check_probability')
    def validate_duplication_probability(cls, v):
        """Validate duplication probability is in valid range."""
        if not 0.0 <= v <= 1.0:
            raise ValueError('duplication_check_probability must be between 0.0 and 1.0')
        return v
    
    @classmethod
    def get_default_dict(cls) -> Dict[str, Any]:
        """Get default configuration as dictionary."""
        return cls().model_dump()
    
    class Config:
        """Pydantic configuration."""
        extra = 'forbid'
        validate_assignment = True