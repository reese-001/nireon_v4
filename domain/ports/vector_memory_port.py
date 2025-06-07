"""
NIREON V4 Vector Memory Port Protocol
Defines the contract for vector storage and retrieval
"""
from typing import Protocol
from dataclasses import dataclass
from domain.embeddings.vector import Vector


@dataclass
class VectorMemoryStats:
    """Statistics about vector memory state."""
    total_count: int
    average_novelty: float
    min_novelty: float
    max_novelty: float
    high_novelty_count: int
    novelty_threshold: float


class VectorMemoryPort(Protocol):
    """Protocol for vector memory storage that persists embeddings with metadata."""
    
    def upsert(self, text: str, vector: Vector, meta: dict) -> None:
        """
        Insert or update a vector with associated text and metadata.
        
        Args:
            text: Original text that was embedded
            vector: The vector embedding
            meta: Additional metadata (novelty_score, source, etc.)
        """
        ...
    
    def query_last(self, n: int = 1000) -> list[tuple[str, Vector, dict]]:
        """
        Query the last N vectors by insertion time.
        
        Args:
            n: Number of recent vectors to retrieve
            
        Returns:
            List of (text, vector, metadata) tuples
        """
        ...
    
    def stats(self) -> VectorMemoryStats:
        """
        Get statistics about the vector memory.
        
        Returns:
            VectorMemoryStats with counts and novelty metrics
        """
        ...