# domain/services/idea_service_port.py
from typing import Protocol
from domain.ideas.idea import Idea

class IdeaServicePort(Protocol):
    def save(self, idea: Idea) -> None:
        ...
    
    def get_by_id(self, idea_id: str) -> Idea:
        ...
