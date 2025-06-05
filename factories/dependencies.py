# nireon/factories/dependencies.py
from __future__ import annotations
import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    # V4 imports
    from application.ports.llm_port import LLMPort
    from application.services.llm_router import LLMRouter # V4
    from application.ports.embedding_port import EmbeddingPort
    from application.ports.event_bus_port import EventBusPort
    from application.services.idea_service import IdeaService # V4
    from core.registry import ComponentRegistry # V4

@dataclass(frozen=True)
class CommonMechanismDependencies:
    embedding_port: EmbeddingPort
    component_registry: ComponentRegistry # V4 ComponentRegistry

    # Optional dependencies, as in V3
    llm_port: Optional[LLMPort] = None
    llm_router: Optional[LLMRouter] = None # V4 LLMRouter
    event_bus: Optional[EventBusPort] = None
    idea_service: Optional[IdeaService] = None # V4 IdeaService
    rng: random.Random = field(default_factory=lambda: random.Random())

    def __post_init__(self):
        # V3 had a check if both llm_port and llm_router are None.
        # This might be too strict if some mechanisms don't need LLMs.
        # The V4 Module Contract Spec for factories doesn't mandate this check.
        # For now, aligning with V3's presence of this check.
        if self.llm_port is None and self.llm_router is None:
            # Consider logging a warning instead of raising an error,
            # as some mechanisms might not require an LLM.
            # logger.warning("Neither LLMPort nor LLMRouter provided in CommonMechanismDependencies.")
            pass # V3's pass is fine for now