# nireon_v4\domain\ports\vector_memory_port.py
from typing import Protocol, runtime_checkable, List, Tuple, Dict, Any
from dataclasses import dataclass
from domain.embeddings.vector import Vector

@dataclass
class VectorMemoryStats:
    total_count: int
    average_novelty: float
    min_novelty: float
    max_novelty: float
    high_novelty_count: int
    novelty_threshold: float

@dataclass
class SearchResult:
    text: str
    vector: Vector
    similarity: float
    metadata: Dict[str, Any]

@runtime_checkable
class VectorMemoryPort(Protocol):
    def upsert(self, text: str, vector: Vector, meta: dict) -> None:
        ...

    def query_last(self, n: int = 1000) -> list[tuple[str, Vector, dict]]:
        ...

    def stats(self) -> VectorMemoryStats:
        ...
    
    def similarity_search(self, query_vector: Vector, top_k: int = 5) -> List[SearchResult]:
        ...