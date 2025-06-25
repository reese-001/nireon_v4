# nireon_v4/components/planners/bandit_planner/config.py
from pydantic import BaseModel, Field
from typing import Dict, List

class BanditPlannerConfig(BaseModel):
    model_path: str = Field(description="Path to the serialized contextual bandit model (.pkl).")
    
    # FIX: exploration_epsilon is now configurable.
    exploration_epsilon: float = Field(
        default=0.3, ge=0.0, le=1.0,
        description="Probability of choosing a random action. Can be tuned."
    )
    
    available_actions: List[str] = Field(
        default=["EXPLORE", "SYNTHESIZE"], 
        description="List of possible abstract actions the bandit can choose."
    )
    
    default_action: str = Field(
        "EXPLORE", 
        description="Action to take if the model is not loaded or fails."
    )
    
    # FIX: action_to_mechanism_map is now configurable.
    action_to_mechanism_map: Dict[str, str] = Field(
        default={
            "EXPLORE": "explorer_instance_01",
            "SYNTHESIZE": "catalyst_instance_01"
        },
        description="Maps abstract planner actions to concrete mechanism component IDs."
    )