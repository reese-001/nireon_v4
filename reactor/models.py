from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Any, Dict, List

# These are the concrete data objects the Reactor works with.

class RuleContext(BaseModel):
    """
    The context object passed to every rule evaluation. It provides a
    snapshot of the system state relevant to the decision.
    """
    signal: Any  # The triggering EpistemicSignal instance
    run_id: str
    component_registry: Any  # The system's ComponentRegistry
    logger: Any  # A pre-configured logger instance
    recursion_depth: int = Field(default=0, description="Tracks signal-chain depth to prevent loops.")
    rule_params: Dict[str, Any] = Field(default_factory=dict, description="Parameters specific to the rule being evaluated.")

    class Config:
        extra = "forbid"
        arbitrary_types_allowed = True


class TriggerComponentAction(BaseModel):
    """
    An action that tells the Reactor to invoke a specific component.
    """
    component_id: str = Field(description="The unique ID of the component instance to trigger.")
    template_id: str | None = Field(default=None, description="Optional ID of a template defining how to run the component.")
    input_data: Dict[str, Any] | None = Field(default=None, description="Data to be passed to the component's process method.")

    class Config:
        extra = "forbid"


class EmitSignalAction(BaseModel):
    """
    An action that tells the Reactor to create and publish a new EpistemicSignal.
    """
    signal_type: str = Field(description="The type of the new signal to emit.")
    payload: Dict[str, Any] = Field(description="The payload for the new signal.")
    source_node_id_override: str | None = Field(default=None, description="Optionally override the signal's source ID (e.g., to the rule's ID).")

    class Config:
        extra = "forbid"

# We can combine these into a type hint for clarity later on.
Action = TriggerComponentAction | EmitSignalAction