# nireon/application/ports/idea_repository_port.py

from typing import List, Optional, Protocol, runtime_checkable

from domain.ideas.idea import Idea


@runtime_checkable
class IdeaRepositoryPort(Protocol):
    """Interface for idea persistence operations."""
    
    def save(self, idea: Idea) -> None:
        """Save an idea to the repository."""
        ...
    
    def get_by_id(self, idea_id: str) -> Optional[Idea]:
        """Retrieve an idea by its ID."""
        ...
    
    def get_all(self) -> List[Idea]:
        """Retrieve all ideas from the repository."""
        ...
    
    def get_by_parent_id(self, parent_id: str) -> List[Idea]:
        """Retrieve all ideas with the specified parent ID."""
        ...
    
    def add_child_relationship(self, parent_id: str, child_id: str) -> bool:
        """Add a child relationship between ideas."""
        ...
    
    def add_world_fact(self, idea_id: str, fact_id: str) -> bool:
        """Add a world fact to an idea."""
        ...