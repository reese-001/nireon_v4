# domain/services/idea_service_port.py
from typing import Protocol, runtime_checkable
from domain.ideas.idea import Idea
from domain.context import NireonExecutionContext

@runtime_checkable
class IdeaServicePort(Protocol):
    def save(self, idea: Idea) -> None:
        ...
    
    def get_by_id(self, idea_id: str) -> Idea:
        ...

    def create_idea(self, *, text: str, parent_id: str | None = None, context: NireonExecutionContext | None = None) -> Idea:
        ...

    def get_all_ideas(self) -> list[Idea]: # Use list instead of List for Python 3.9+
        ...

    def add_world_fact(self, *, idea_id: str, fact_id: str, context: NireonExecutionContext | None = None) -> bool:
        ...
        
    def get_children(self, idea_id: str) -> list[Idea]:
        ...
