# nireon_v4/components/planners/bandit_planner/metadata.py
from core.lifecycle import ComponentMetadata

BANDIT_PLANNER_METADATA = ComponentMetadata(
    id="bandit_planner_default",
    name="Bandit-based Epistemic Planner",
    version="1.0.0",
    description="Uses a contextual bandit model to choose the next epistemic action (e.g., Explore vs. Synthesize) to maximize trust gain.",
    category="planner",
    epistemic_tags=["planner", "learning", "reinforcement_learning", "bandit"],
    requires_initialize=True,
    dependencies={'EventBusPort': '*'}
)