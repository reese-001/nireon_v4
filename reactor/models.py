# reactor/models.py
from __future__ import annotations

from typing import Any, Dict

from pydantic import BaseModel, ConfigDict, Field


class RuleContext(BaseModel):
    signal: Any
    run_id: str
    component_registry: Any
    logger: Any
    recursion_depth: int = Field(0, description="Depth in the signal chain")
    rule_params: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)


class TriggerComponentAction(BaseModel):
    component_id: str
    template_id: str | None = None
    input_data: Dict[str, Any] | None = None

    model_config = ConfigDict(extra="forbid")


class EmitSignalAction(BaseModel):
    signal_type: str
    payload: Dict[str, Any]
    source_node_id_override: str | None = None

    # model_config = ConfigDict(extra="forbid")
    model_config = ConfigDict(extra='allow')

Action = TriggerComponentAction | EmitSignalAction
