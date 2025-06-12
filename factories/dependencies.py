# C:\Users\erees\Documents\development\nireon\factories\dependencies.py
from __future__ import annotations
import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from domain.ports.llm_port import LLMPort # MODIFIED
    from domain.ports.embedding_port import EmbeddingPort
    from domain.ports.event_bus_port import EventBusPort
    from domain.ports.idea_service_port import IdeaServicePort # Assuming this is the correct IdeaService interface
    from core.registry import ComponentRegistry
    # If IdeaService is a concrete class used directly (less ideal for DI)
    # from application.services.idea_service import IdeaService


@dataclass(frozen=True)
class CommonMechanismDependencies:
    embedding_port: EmbeddingPort
    component_registry: ComponentRegistry
    llm_port: Optional[LLMPort] = None # This is the interface
    llm_router: Optional[LLMPort] = None # If router itself is also injected, and it implements LLMPort
    event_bus: Optional[EventBusPort] = None
    idea_service: Optional[IdeaServicePort] = None # Depend on the interface
    rng: random.Random = field(default_factory=lambda: random.Random())

    def __post_init__(self):
        # This check might be too simplistic if llm_port can be None by design
        # if self.llm_port is None and self.llm_router is None:
        #     pass # Or log a warning if one is expected
        pass