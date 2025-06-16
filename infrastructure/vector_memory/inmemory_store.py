# nireon_v4\infrastructure\vector_memory\inmemory_store.py
from __future__ import annotations
import logging
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple
import numpy as np
from pydantic import BaseModel, Field

from core.base_component import NireonBaseComponent
from core.lifecycle import ComponentMetadata
from core.results import ProcessResult
from domain.context import NireonExecutionContext
from domain.embeddings.vector import Vector
from domain.ports.vector_memory_port import SearchResult, VectorMemoryPort, VectorMemoryStats

logger = logging.getLogger(__name__)

INMEMORY_VECTOR_STORE_METADATA = ComponentMetadata(
    id='vector_memory_inmemory',
    name='InMemoryVectorStore',
    version='1.0.0',
    category='shared_service',
    description='A simple, thread-safe, in-memory vector store for development and testing.',
    requires_initialize=True,
    epistemic_tags=['vector_storage', 'retriever'],
)

class InMemoryVectorStoreConfig(BaseModel):
    capacity: int = Field(default=10000, ge=1, description="Maximum number of vectors to store.")
    dimensions: int = Field(..., description="The dimensionality of the vectors to be stored.")
    similarity_metric: str = Field(default='cosine', description="Similarity metric to use ('cosine').")

class InMemoryVectorStore(NireonBaseComponent, VectorMemoryPort):
    METADATA_DEFINITION = INMEMORY_VECTOR_STORE_METADATA

    def __init__(self, config: Dict[str, Any], metadata_definition: Optional[ComponentMetadata] = None):
        super().__init__(config=config, metadata_definition=metadata_definition or self.METADATA_DEFINITION)
        self.store_cfg = InMemoryVectorStoreConfig(**self.config)
        self._store: Deque[Tuple[str, Vector, Dict[str, Any]]] = deque(maxlen=self.store_cfg.capacity)
        logger.info(
            f"[{self.component_id}] created with capacity={self.store_cfg.capacity}, "
            f"dimensions={self.store_cfg.dimensions}"
        )

    async def _initialize_impl(self, context: NireonExecutionContext) -> None:
        context.logger.info(f"[{self.component_id}] initialized successfully.")
        pass

    async def _process_impl(self, data: Any, context: NireonExecutionContext) -> ProcessResult:
        return ProcessResult.ok("InMemoryVectorStore does not implement a generic process method. Use specific methods like upsert, query, etc.")

    def upsert(self, text: str, vector: Vector, meta: Dict[str, Any]) -> None:
        if vector.dims != self.store_cfg.dimensions:
            raise ValueError(
                f"Vector dimension mismatch. Store configured for {self.store_cfg.dimensions}, "
                f"but received vector with {vector.dims}."
            )
        self._store.append((text, vector, meta))

    def query_last(self, n: int = 1000) -> List[Tuple[str, Vector, Dict[str, Any]]]:
        if n <= 0:
            return []
        # Return a copy to avoid external modifications to the deque
        return list(self._store)[-n:]

    def similarity_search(self, query_vector: Vector, top_k: int = 5) -> List[SearchResult]:
        if query_vector.dims != self.store_cfg.dimensions:
            raise ValueError(
                f"Query vector dimension mismatch. Store configured for {self.store_cfg.dimensions}, "
                f"but received query with {query_vector.dims}."
            )
        
        if not self._store:
            return []

        # This is a brute-force O(N) search, suitable for a simple in-memory store.
        scored_results = []
        for text, stored_vector, meta in self._store:
            similarity = query_vector.similarity(stored_vector)
            scored_results.append(SearchResult(
                text=text,
                vector=stored_vector,
                similarity=similarity,
                metadata=meta
            ))

        # Sort by similarity in descending order and take the top k
        scored_results.sort(key=lambda x: x.similarity, reverse=True)
        return scored_results[:top_k]

    def stats(self) -> VectorMemoryStats:
        total_count = len(self._store)
        if total_count == 0:
            return VectorMemoryStats(
                total_count=0,
                average_novelty=0.0,
                min_novelty=0.0,
                max_novelty=0.0,
                high_novelty_count=0,
                novelty_threshold=0.75  # A reasonable default
            )

        novelty_scores = [meta.get('novelty_score', 0.5) for _, _, meta in self._store]
        # Using a dummy threshold, as the store itself doesn't know the embedding service's config
        # This could be improved by passing the threshold to the stats method if needed.
        novelty_threshold = 0.75 

        return VectorMemoryStats(
            total_count=total_count,
            average_novelty=float(np.mean(novelty_scores)),
            min_novelty=float(np.min(novelty_scores)),
            max_novelty=float(np.max(novelty_scores)),
            high_novelty_count=sum(1 for s in novelty_scores if s > novelty_threshold),
            novelty_threshold=novelty_threshold
        )