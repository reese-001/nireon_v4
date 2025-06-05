"""
Configuration model for Explorer mechanism using Pydantic
"""
from typing import Literal, Optional
from pydantic import BaseModel, Field, validator

class ExplorerConfig(BaseModel):
    """
    Configuration for Explorer mechanism
    
    Defines exploration parameters, strategies, and behavioral settings
    for idea generation and variation.
    """

    divergence_strength: float = 0.2
    exploration_timeout: int = 60
    
    max_depth: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum depth for exploration (number of levels to explore)"
    )
    
    application_rate: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Rate at which exploration is applied (0.0 = minimal, 1.0 = maximum)"
    )
    
    exploration_strategy: Literal['depth_first', 'breadth_first', 'random'] = Field(
        default='depth_first',
        description="Strategy for exploring the idea space"
    )
    
    max_variations_per_level: int = Field(
        default=3,
        ge=1,
        le=20,
        description="Maximum number of variations to generate per exploration level"
    )
    
    enable_semantic_exploration: bool = Field(
        default=True,
        description="Enable semantic-based exploration using embeddings"
    )
    
    enable_llm_enhancement: bool = Field(
        default=True,
        description="Enable LLM-based idea enhancement and variation generation"
    )
    
    creativity_factor: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Creativity factor for exploration (0.0 = conservative, 1.0 = highly creative)"
    )
    
    seed_randomness: Optional[int] = Field(
        default=None,
        description="Random seed for reproducible exploration (None = random)"
    )
    
    minimum_idea_length: int = Field(
        default=10,
        ge=5,
        le=1000,
        description="Minimum length for generated idea text"
    )
    
    maximum_idea_length: int = Field(
        default=500,
        ge=50,
        le=5000,
        description="Maximum length for generated idea text"
    )
    
    exploration_timeout_seconds: float = Field(
        default=30.0,
        ge=1.0,
        le=300.0,
        description="Timeout for exploration operations in seconds"
    )
    
    enable_diversity_filter: bool = Field(
        default=True,
        description="Enable filtering to ensure diversity in generated variations"
    )
    
    diversity_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Minimum diversity threshold for accepting variations (0.0 = no filter, 1.0 = maximum diversity)"
    )
    
    @validator('maximum_idea_length')
    def validate_max_length(cls, v, values):
        """Ensure maximum length is greater than minimum length"""
        if 'minimum_idea_length' in values and v <= values['minimum_idea_length']:
            raise ValueError('maximum_idea_length must be greater than minimum_idea_length')
        return v
    
    @validator('exploration_strategy')
    def validate_strategy(cls, v):
        """Validate exploration strategy"""
        valid_strategies = ['depth_first', 'breadth_first', 'random']
        if v not in valid_strategies:
            raise ValueError(f'exploration_strategy must be one of: {valid_strategies}')
        return v
    
    @validator('creativity_factor')
    def validate_creativity_factor(cls, v):
        """Validate creativity factor range"""
        if not 0.0 <= v <= 1.0:
            raise ValueError('creativity_factor must be between 0.0 and 1.0')
        return v
    
    def get_exploration_params(self) -> dict:
        """Get exploration parameters as a dictionary"""
        return {
            'max_depth': self.max_depth,
            'strategy': self.exploration_strategy,
            'creativity': self.creativity_factor,
            'max_variations': self.max_variations_per_level,
            'timeout': self.exploration_timeout_seconds
        }
    
    def is_llm_enabled(self) -> bool:
        """Check if LLM enhancement is enabled"""
        return self.enable_llm_enhancement
    
    def is_semantic_enabled(self) -> bool:
        """Check if semantic exploration is enabled"""
        return self.enable_semantic_exploration
    
    def should_filter_diversity(self) -> bool:
        """Check if diversity filtering should be applied"""
        return self.enable_diversity_filter and self.diversity_threshold > 0.0
    
    class Config:
        """Pydantic model configuration"""
        # Allow extra fields for forward compatibility
        extra = "forbid"
        
        # Validate assignment
        validate_assignment = True
        
        # Use enum values for serialization
        use_enum_values = True
        
        # Schema extras for documentation
        schema_extra = {
            "example": {
                "max_depth": 4,
                "application_rate": 0.6,
                "exploration_strategy": "depth_first",
                "max_variations_per_level": 3,
                "enable_semantic_exploration": True,
                "enable_llm_enhancement": True,
                "creativity_factor": 0.7,
                "seed_randomness": 42,
                "minimum_idea_length": 20,
                "maximum_idea_length": 300,
                "exploration_timeout_seconds": 45.0,
                "enable_diversity_filter": True,
                "diversity_threshold": 0.4
            }
        }