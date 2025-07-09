"""
Configuration schema for the QuantifierAgent.
"""

from typing import Dict, List, Literal
from pydantic import BaseModel, Field

class QuantifierConfig(BaseModel):
    """Configuration for the QuantifierAgent mechanism."""
    
    class Config:
        extra = "forbid"
        validate_assignment = True
    
    # Visualization approach
    max_visualizations: int = Field(
        default=1, 
        ge=1, 
        le=3,
        description="Maximum number of visualizations to generate per idea"
    )
    
    llm_approach: Literal["single_call", "iterative"] = Field(
        default="single_call",
        description="LLM call strategy: single comprehensive call vs multiple iterative calls"
    )
    
    # Library constraints
    available_libraries: Dict[str, List[str]] = Field(
        default={
            "core_data": ["numpy", "pandas", "scipy"],
            "visualization": ["matplotlib", "seaborn", "plotly"], 
            "specialized_viz": ["networkx", "wordcloud", "graphviz"],
            "analysis": ["scikit-learn", "statsmodels"]
        },
        description="Curated libraries available for analysis"
    )
    
    # Quality thresholds
    min_request_length: int = Field(
        default=100,
        description="Minimum length for generated Proto requests"
    )
    
    viability_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0, 
        description="Confidence threshold for visualization viability"
    )
    
    # Timeouts and limits
    llm_timeout_seconds: int = Field(
        default=30,
        description="Timeout for individual LLM calls"
    )
    
    enable_mermaid_output: bool = Field(
        default=True,
        description="Allow Mermaid diagram generation as alternative to Python visualizations"
    )