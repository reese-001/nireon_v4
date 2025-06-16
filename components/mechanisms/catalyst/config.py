# nireon_v4\components\mechanisms\catalyst\config.py
from typing import Optional, Dict, Any, ClassVar
from pydantic import BaseModel, Field, validator

class CatalystMechanismConfig(BaseModel):
    application_rate: float = Field(0.1, ge=0.0, le=1.0, description='Chance of applying to an agent')
    blend_low: float = Field(0.1, ge=0.0, le=1.0, description='Minimum blend strength')
    blend_high: float = Field(0.3, ge=0.0, le=1.0, description='Maximum blend strength')
    max_blend_low: float = Field(0.8, ge=0.0, le=1.0, description='Maximum allowed value for blend_low')
    max_blend_high: float = Field(0.95, ge=0.0, le=1.0, description='Maximum allowed value for blend_high')
    
    MIN_BLEND_GAP: ClassVar[float] = 0.01  # <--- THIS IS THE FIX

    duplication_check_enabled: bool = Field(False, description='Enable duplication detection')
    duplication_check_probability: float = Field(0.05, ge=0.0, le=1.0, description='Probability of checking for duplication')
    duplication_cooldown_steps: int = Field(5, ge=0, description='Steps before resetting blend range after a duplication event')
    duplication_aggressiveness: float = Field(0.5, ge=0.0, le=5.0, description='How quickly to adjust blend range on duplication')
    anti_constraints_enabled: bool = Field(False, description='Enable anti-constraints')
    anti_constraints_count: int = Field(3, ge=1, le=10, description='Maximum number of anti-constraints')
    anti_constraints_diversity_threshold: float = Field(0.15, ge=0.0, le=1.0, description='When to apply anti-constraints')
    prompt_template: Optional[str] = Field(None, description='Custom prompt template for the LLM')
    default_llm_policy_for_catalysis: Dict[str, Any] = Field(default_factory=lambda: {'temperature': 0.75, 'max_tokens': 300, 'model_preference': 'creative'}, description='LLM policy template for new *CatalystFrame*s.')
    default_resource_budget_for_catalysis: Dict[str, Any] = Field(default_factory=lambda: {'llm_calls': 20, 'event_publishes': 40, 'cpu_seconds': 300}, description='Initial soft budget for a catalysis task frame.')

    @validator('blend_high')
    def blend_high_greater_than_low(cls, v, values):
        if 'blend_low' in values and v <= values['blend_low']:
            raise ValueError(f"blend_high ({v}) must be greater than blend_low ({values['blend_low']})")
        return v

    @validator('max_blend_high')
    def max_blend_high_greater_than_max_low(cls, v, values):
        if 'max_blend_low' in values and v <= values['max_blend_low']:
            raise ValueError(f"max_blend_high ({v}) must be greater than max_blend_low ({values['max_blend_low']})")
        return v

    @validator('anti_constraints_count')
    def validate_anti_constraints_count(cls, v):
        if v < 1:
            raise ValueError('anti_constraints_count must be at least 1')
        return v
        
    @validator('duplication_check_probability')
    def validate_duplication_probability(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError('duplication_check_probability must be between 0.0 and 1.0')
        return v

    @classmethod
    def get_default_dict(cls) -> Dict[str, Any]:
        return cls().model_dump()

    class Config:
        extra = 'forbid'
        validate_assignment = True