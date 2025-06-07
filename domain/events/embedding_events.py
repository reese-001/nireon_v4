"""
NIREON V4 Embedding Event Definitions
Event schemas for embedding subsystem
"""
from dataclasses import dataclass
from typing import Any


@dataclass
class EmbeddingComputedEvent:
    """Event fired when an embedding is successfully computed."""
    text: str
    vector_id: str
    similarity_max: float
    novelty_score: float
    provider: str


@dataclass
class HighNoveltyDetectedEvent:
    """Event fired when high novelty content is detected."""
    text: str
    novelty_score: float


@dataclass
class EmbeddingErrorEvent:
    """Event fired when embedding computation fails."""
    text: str
    error: str
    provider: str


# Event type constants
EMBEDDING_COMPUTED = "EMBEDDING_COMPUTED"
HIGH_NOVELTY_DETECTED = "HIGH_NOVELTY_DETECTED" 
EMBEDDING_ERROR = "EMBEDDING_ERROR"