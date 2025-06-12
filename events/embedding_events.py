# nireon/events/embedding_events.py
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from dataclasses import dataclass # Import dataclass

class EmbeddingRequestSignalPayload(BaseModel):
    """Payload for requesting an embedding computation."""
    request_id: str = Field(..., description="Unique ID for correlation")
    text_to_embed: str = Field(..., description="Text content to embed")
    target_artifact_id: str = Field(..., description="ID of the artifact (e.g., idea) this embedding is for")
    request_timestamp_ms: int = Field(..., description="Timestamp when request was made")
    embedding_vector_dtype: str = Field(default="float32", description="Desired data type for embedding vector")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata for the request")

class EmbeddingComputedSignalPayload(BaseModel):
    """Payload containing computed embedding results."""
    request_id: str = Field(..., description="ID matching the original request for correlation")
    text_embedded: str = Field(..., description="Original text that was embedded")
    target_artifact_id: str = Field(..., description="ID of the artifact this embedding is for")
    embedding_vector: List[float] = Field(..., description="The computed embedding vector")
    embedding_vector_dtype: str = Field(..., description="Data type of the embedding vector")
    embedding_dimensions: int = Field(..., description="Number of dimensions in the embedding")
    provider_info: Dict[str, Any] = Field(default_factory=dict, description="Information about the embedding provider")
    error_message: Optional[str] = Field(None, description="Error message if embedding computation failed")

# --- Define the missing Event classes ---
@dataclass
class EmbeddingComputedEvent:
    text: str
    vector_id: str # Assuming this corresponds to target_artifact_id or a new UUID
    similarity_max: float
    novelty_score: float
    provider: str
    # You might also include the full EmbeddingComputedSignalPayload if needed
    # payload: Optional[EmbeddingComputedSignalPayload] = None

@dataclass
class HighNoveltyDetectedEvent:
    text: str
    novelty_score: float
    # payload: Optional[Any] = None # If it had a specific payload structure

@dataclass
class EmbeddingErrorEvent:
    text: str
    error: str
    provider: str
    # payload: Optional[Any] = None # If it had a specific payload structure

# --- Constants for event types (already present and correct) ---
EMBEDDING_COMPUTED = 'EMBEDDING_COMPUTED'
HIGH_NOVELTY_DETECTED = 'HIGH_NOVELTY_DETECTED'
EMBEDDING_ERROR = 'EMBEDDING_ERROR'