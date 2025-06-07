"""
NIREON V4 Embedding Configuration Models
Pydantic models for embedding and vector memory configuration
"""
from typing import Literal
from pathlib import Path
from pydantic import BaseModel, Field


class EmbeddingConfig(BaseModel):
    """Configuration for embedding providers."""
    
    provider: Literal['sentence_transformers', 'openai', 'mock'] = Field(
        ..., description="Embedding provider to use"
    )
    model_name: str = Field(
        default="all-MiniLM-L6-v2", 
        description="Model name for the provider"
    )
    dimensions: int = Field(
        default=384, 
        description="Expected embedding dimensions (must match provider output)"
    )
    cache_size: int = Field(
        default=1000, 
        description="In-memory LRU cache size"
    )
    novelty_threshold: float = Field(
        default=0.75, 
        description="Threshold for high novelty detection (1-similarity)"
    )
    vector_memory_ref: str = Field(
        default="vector_store_sqlite", 
        description="Registry key for vector memory implementation"
    )
    
    # Provider-specific configs
    openai_api_key: str | None = Field(
        default=None, 
        description="OpenAI API key (if using openai provider)"
    )


class VectorMemoryConfig(BaseModel):
    """Configuration for vector memory storage."""
    
    provider: Literal['sqlite', 'postgres'] = Field(
        default="sqlite",
        description="Vector memory storage provider"
    )
    db_path: Path = Field(
        default=Path("runtime/vector_memory.db"), 
        description="Database file path for SQLite implementation"
    )
    pool_size: int = Field(
        default=4, 
        description="Connection pool size for Postgres implementation"
    )
    enable_wal_mode: bool = Field(
        default=True,
        description="Enable WAL mode for SQLite (better concurrency)"
    )