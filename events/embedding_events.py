# nireon_v4/events/embedding_events.py
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional

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