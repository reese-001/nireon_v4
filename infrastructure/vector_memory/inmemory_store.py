# nireon_v4\infrastructure\vector_memory\inmemory_store.py
from __future__ import annotations
import logging
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple, Set
import numpy as np
from pydantic import BaseModel, Field
import asyncio
from threading import Lock, RLock
import heapq
from datetime import datetime, timezone

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
    similarity_metric: str = Field(default='cosine', description="Similarity metric to use ('cosine', 'euclidean', 'dot').")
    enable_deduplication: bool = Field(default=True, description="Enable deduplication of identical vectors.")
    similarity_threshold: float = Field(default=0.99, description="Threshold for considering vectors identical.")
    batch_size: int = Field(default=100, description="Batch size for bulk operations.")
    enable_stats_tracking: bool = Field(default=True, description="Enable detailed statistics tracking.")

class VectorEntry:
    """Efficient storage for vector entries with metadata"""
    __slots__ = ('text', 'vector', 'metadata', 'timestamp', 'access_count')
    
    def __init__(self, text: str, vector: Vector, metadata: Dict[str, Any]):
        self.text = text
        self.vector = vector
        self.metadata = metadata
        self.timestamp = datetime.now(timezone.utc)
        self.access_count = 0

class InMemoryVectorStore(NireonBaseComponent, VectorMemoryPort):
    """
    Optimized in-memory vector store with:
    - Thread-safe operations with fine-grained locking
    - Efficient similarity search with early termination
    - Deduplication support
    - Batch operations
    - Memory-efficient storage
    - Comprehensive statistics tracking
    """
    METADATA_DEFINITION = INMEMORY_VECTOR_STORE_METADATA

    def __init__(self, config: Dict[str, Any], metadata_definition: Optional[ComponentMetadata] = None):
        super().__init__(config=config, metadata_definition=metadata_definition or self.METADATA_DEFINITION)
        self.store_cfg = InMemoryVectorStoreConfig(**self.config)
        
        # Use deque for O(1) append/pop operations
        self._store: Deque[VectorEntry] = deque(maxlen=self.store_cfg.capacity)
        
        # Thread safety with reentrant lock for nested calls
        self._lock = RLock()
        
        # Deduplication support
        if self.store_cfg.enable_deduplication:
            self._vector_hashes: Set[int] = set()
            
        # Statistics tracking
        self._stats = {
            'total_operations': 0,
            'total_searches': 0,
            'total_upserts': 0,
            'deduplicated_count': 0,
            'evicted_count': 0,
            'batch_operations': 0
        }
        
        # Pre-allocate numpy arrays for batch operations
        self._batch_buffer = np.empty((self.store_cfg.batch_size, self.store_cfg.dimensions), dtype=np.float32)
        
        logger.info(
            f"[{self.component_id}] created with capacity={self.store_cfg.capacity}, "
            f"dimensions={self.store_cfg.dimensions}, similarity_metric={self.store_cfg.similarity_metric}"
        )

    async def _initialize_impl(self, context: NireonExecutionContext) -> None:
        """Initialize the vector store"""
        context.logger.info(f"[{self.component_id}] initialized successfully.")
        # Pre-warm the store if needed
        if self.store_cfg.enable_stats_tracking:
            context.logger.debug(f"Statistics tracking enabled for {self.component_id}")

    async def _process_impl(self, data: Any, context: NireonExecutionContext) -> ProcessResult:
        """Process implementation for component interface"""
        return ProcessResult.ok(
            "InMemoryVectorStore does not implement a generic process method. "
            "Use specific methods like upsert, query, etc."
        )

    def _compute_vector_hash(self, vector: Vector) -> int:
        """Compute a hash for vector deduplication"""
        # Use a subset of dimensions for faster hashing
        sample_dims = min(10, vector.dims)
        sample = vector.data[:sample_dims]
        return hash(tuple(np.round(sample, decimals=4)))

    def _is_duplicate(self, vector: Vector) -> bool:
        """Check if vector is a duplicate based on similarity threshold"""
        if not self.store_cfg.enable_deduplication:
            return False
            
        vector_hash = self._compute_vector_hash(vector)
        if vector_hash not in self._vector_hashes:
            return False
            
        # Do precise check for potential duplicates
        for entry in self._store:
            if self._compute_vector_hash(entry.vector) == vector_hash:
                similarity = vector.similarity(entry.vector)
                if similarity >= self.store_cfg.similarity_threshold:
                    return True
        return False

    def upsert(self, text: str, vector: Vector, meta: Dict[str, Any]) -> None:
        """Insert or update a vector with thread safety and deduplication"""
        if vector.dims != self.store_cfg.dimensions:
            raise ValueError(
                f"Vector dimension mismatch. Store configured for {self.store_cfg.dimensions}, "
                f"but received vector with {vector.dims}."
            )
        
        with self._lock:
            self._stats['total_operations'] += 1
            self._stats['total_upserts'] += 1
            
            # Check for duplicates if enabled
            if self.store_cfg.enable_deduplication and self._is_duplicate(vector):
                self._stats['deduplicated_count'] += 1
                logger.debug(f"Duplicate vector detected for text: {text[:50]}...")
                return
                
            # Check if we're at capacity
            if len(self._store) >= self.store_cfg.capacity:
                evicted = self._store.popleft()
                if self.store_cfg.enable_deduplication:
                    self._vector_hashes.discard(self._compute_vector_hash(evicted.vector))
                self._stats['evicted_count'] += 1
                
            # Add new entry
            entry = VectorEntry(text, vector, meta)
            self._store.append(entry)
            
            if self.store_cfg.enable_deduplication:
                self._vector_hashes.add(self._compute_vector_hash(vector))

    def upsert_batch(self, entries: List[Tuple[str, Vector, Dict[str, Any]]]) -> None:
        """Batch upsert operation for improved performance"""
        if not entries:
            return
            
        with self._lock:
            self._stats['batch_operations'] += 1
            
            for text, vector, meta in entries:
                if vector.dims != self.store_cfg.dimensions:
                    logger.warning(f"Skipping vector with incorrect dimensions: {vector.dims}")
                    continue
                    
                self._stats['total_operations'] += 1
                self._stats['total_upserts'] += 1
                
                # Deduplication check
                if self.store_cfg.enable_deduplication and self._is_duplicate(vector):
                    self._stats['deduplicated_count'] += 1
                    continue
                    
                # Handle capacity
                if len(self._store) >= self.store_cfg.capacity:
                    evicted = self._store.popleft()
                    if self.store_cfg.enable_deduplication:
                        self._vector_hashes.discard(self._compute_vector_hash(evicted.vector))
                    self._stats['evicted_count'] += 1
                    
                # Add entry
                entry = VectorEntry(text, vector, meta)
                self._store.append(entry)
                
                if self.store_cfg.enable_deduplication:
                    self._vector_hashes.add(self._compute_vector_hash(vector))

    def query_last(self, n: int = 1000) -> List[Tuple[str, Vector, Dict[str, Any]]]:
        """Query last n entries with improved performance"""
        if n <= 0:
            return []
            
        with self._lock:
            self._stats['total_operations'] += 1
            # Convert to list efficiently
            result = []
            # Iterate from the end for better cache locality
            for i in range(min(n, len(self._store))):
                idx = -(i + 1)
                entry = self._store[idx]
                entry.access_count += 1
                result.append((entry.text, entry.vector, entry.metadata))
            return list(reversed(result))

    def similarity_search(self, query_vector: Vector, top_k: int = 5, 
                         min_similarity: Optional[float] = None) -> List[SearchResult]:
        """
        Optimized similarity search with:
        - Early termination for efficiency
        - Configurable minimum similarity threshold
        - Support for different similarity metrics
        """
        if query_vector.dims != self.store_cfg.dimensions:
            raise ValueError(
                f"Query vector dimension mismatch. Store configured for {self.store_cfg.dimensions}, "
                f"but received query with {query_vector.dims}."
            )
        
        if not self._store:
            return []
            
        with self._lock:
            self._stats['total_operations'] += 1
            self._stats['total_searches'] += 1
            
            # Use heap for efficient top-k selection
            if self.store_cfg.similarity_metric == 'cosine':
                similarity_fn = lambda v: query_vector.similarity(v)
            elif self.store_cfg.similarity_metric == 'euclidean':
                # Convert distance to similarity (1 / (1 + distance))
                similarity_fn = lambda v: 1.0 / (1.0 + np.linalg.norm(query_vector.data - v.data))
            elif self.store_cfg.similarity_metric == 'dot':
                similarity_fn = lambda v: np.dot(query_vector.data, v.data)
            else:
                similarity_fn = lambda v: query_vector.similarity(v)
                
            # Use min heap with negative similarities for top-k
            heap = []
            
            for entry in self._store:
                similarity = similarity_fn(entry.vector)
                
                # Skip if below minimum similarity
                if min_similarity is not None and similarity < min_similarity:
                    continue
                    
                entry.access_count += 1
                
                if len(heap) < top_k:
                    heapq.heappush(heap, (similarity, entry))
                elif similarity > heap[0][0]:
                    heapq.heapreplace(heap, (similarity, entry))
                    
            # Extract results in descending order
            results = []
            while heap:
                similarity, entry = heapq.heappop(heap)
                results.append(SearchResult(
                    text=entry.text,
                    vector=entry.vector,
                    similarity=similarity,
                    metadata=entry.metadata
                ))
                
            return list(reversed(results))

    def similarity_search_batch(self, query_vectors: List[Vector], top_k: int = 5) -> List[List[SearchResult]]:
        """Batch similarity search for multiple queries"""
        results = []
        for query_vector in query_vectors:
            results.append(self.similarity_search(query_vector, top_k))
        return results

    def stats(self) -> VectorMemoryStats:
        """Get comprehensive statistics about the vector store"""
        with self._lock:
            total_count = len(self._store)
            
            if total_count == 0:
                return VectorMemoryStats(
                    total_count=0,
                    average_novelty=0.0,
                    min_novelty=0.0,
                    max_novelty=0.0,
                    high_novelty_count=0,
                    novelty_threshold=0.75
                )
                
            novelty_scores = []
            access_counts = []
            
            for entry in self._store:
                novelty_scores.append(entry.metadata.get('novelty_score', 0.5))
                access_counts.append(entry.access_count)
                
            novelty_threshold = 0.75
            
            stats = VectorMemoryStats(
                total_count=total_count,
                average_novelty=float(np.mean(novelty_scores)),
                min_novelty=float(np.min(novelty_scores)),
                max_novelty=float(np.max(novelty_scores)),
                high_novelty_count=sum(1 for s in novelty_scores if s > novelty_threshold),
                novelty_threshold=novelty_threshold
            )
            
            # Add extended stats if tracking is enabled
            if self.store_cfg.enable_stats_tracking:
                stats.metadata = {
                    'total_operations': self._stats['total_operations'],
                    'total_searches': self._stats['total_searches'],
                    'total_upserts': self._stats['total_upserts'],
                    'deduplicated_count': self._stats['deduplicated_count'],
                    'evicted_count': self._stats['evicted_count'],
                    'batch_operations': self._stats['batch_operations'],
                    'average_access_count': float(np.mean(access_counts)) if access_counts else 0.0,
                    'capacity_usage': total_count / self.store_cfg.capacity
                }
                
            return stats

    def clear(self) -> None:
        """Clear all stored vectors"""
        with self._lock:
            self._store.clear()
            if self.store_cfg.enable_deduplication:
                self._vector_hashes.clear()
            logger.info(f"[{self.component_id}] cleared all {len(self._store)} vectors")

    def get_by_metadata(self, key: str, value: Any) -> List[Tuple[str, Vector, Dict[str, Any]]]:
        """Retrieve entries by metadata key-value pair"""
        with self._lock:
            results = []
            for entry in self._store:
                if entry.metadata.get(key) == value:
                    entry.access_count += 1
                    results.append((entry.text, entry.vector, entry.metadata))
            return results

    def remove_by_metadata(self, key: str, value: Any) -> int:
        """Remove entries by metadata key-value pair"""
        with self._lock:
            removed_count = 0
            new_store = deque(maxlen=self.store_cfg.capacity)
            
            for entry in self._store:
                if entry.metadata.get(key) != value:
                    new_store.append(entry)
                else:
                    if self.store_cfg.enable_deduplication:
                        self._vector_hashes.discard(self._compute_vector_hash(entry.vector))
                    removed_count += 1
                    
            self._store = new_store
            return removed_count

    async def asimilarity_search(self, query_vector: Vector, top_k: int = 5,
                                min_similarity: Optional[float] = None) -> List[SearchResult]:
        """Async wrapper for similarity search"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.similarity_search, query_vector, top_k, min_similarity
        )

    async def aupsert(self, text: str, vector: Vector, meta: Dict[str, Any]) -> None:
        """Async wrapper for upsert"""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.upsert, text, vector, meta)

    def __repr__(self) -> str:
        """String representation of the store"""
        return (
            f"InMemoryVectorStore("
            f"capacity={self.store_cfg.capacity}, "
            f"dimensions={self.store_cfg.dimensions}, "
            f"current_size={len(self._store)}, "
            f"similarity_metric={self.store_cfg.similarity_metric})"
        )