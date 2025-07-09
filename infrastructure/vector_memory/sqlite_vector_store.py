# nireon_v4\infrastructure\vector_memory\sqlite_vector_store.py
from __future__ import annotations
import logging
import json
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timezone
import pickle
import asyncio

from pydantic import BaseModel, Field

from core.base_component import NireonBaseComponent
from core.lifecycle import ComponentMetadata
from core.results import ProcessResult
from domain.context import NireonExecutionContext
from domain.embeddings.vector import Vector
from domain.ports.vector_memory_port import SearchResult, VectorMemoryPort, VectorMemoryStats
logger = logging.getLogger(__name__)
# Import the shared SQLite base
try:
    from infrastructure.persistence.sqlite_base import SQLiteBaseRepository, build_in_clause
    HAS_SQLITE_BASE = True
except ImportError:
    HAS_SQLITE_BASE = False
    logger.warning("SQLiteBaseRepository not available, using inline implementation")



SQLITE_VECTOR_STORE_METADATA = ComponentMetadata(
    id='vector_memory_sqlite',
    name='SQLiteVectorStore',
    version='1.0.0',
    category='shared_service',
    description='A persistent SQLite-based vector store with efficient similarity search.',
    requires_initialize=True,
    epistemic_tags=['vector_storage', 'retriever', 'persistent'],
)

class SQLiteVectorStoreConfig(BaseModel):
    db_path: str = Field(default='runtime/vectors.db', description="Path to SQLite database file.")
    dimensions: int = Field(..., description="The dimensionality of the vectors to be stored.")
    similarity_metric: str = Field(default='cosine', description="Similarity metric to use.")
    enable_fts: bool = Field(default=True, description="Enable full-text search on text content.")
    batch_size: int = Field(default=100, description="Batch size for bulk operations.")
    vacuum_threshold: int = Field(default=10000, description="Operations before auto-vacuum.")
    # Base config options that will be passed to SQLiteBaseRepository
    enable_wal_mode: bool = Field(default=True, description="Enable WAL mode for better concurrency.")
    pool_size: int = Field(default=5, description="Connection pool size.")
    page_size: int = Field(default=4096, description="SQLite page size.")
    cache_size: int = Field(default=2000, description="SQLite cache size in pages.")


class SQLiteVectorStoreImpl(SQLiteBaseRepository[Tuple[str, Vector, Dict[str, Any]]]):
    """SQLite implementation using the shared base class"""
    
    def __init__(self, config: SQLiteVectorStoreConfig):
        self.store_cfg = config
        # Convert to base config format
        base_config = {
            'db_path': config.db_path,
            'enable_wal_mode': config.enable_wal_mode,
            'pool_size': config.pool_size,
            'page_size': config.page_size,
            'cache_size': config.cache_size,
        }
        super().__init__(base_config)
        
    def _init_database_schema(self):
        """Initialize SQLite database schema for vectors"""
        with self._get_connection() as conn:
            # Create main vectors table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS vectors (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    text TEXT NOT NULL,
                    vector_data BLOB NOT NULL,
                    metadata TEXT DEFAULT '{}',
                    novelty_score REAL DEFAULT 0.5,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    accessed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    access_count INTEGER DEFAULT 0
                )
            """)
            
            # Create indexes
            conn.execute('CREATE INDEX IF NOT EXISTS idx_novelty ON vectors(novelty_score DESC)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_created ON vectors(created_at DESC)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_accessed ON vectors(accessed_at DESC)')
            
            # Create unique index on text for deduplication
            conn.execute('CREATE UNIQUE INDEX IF NOT EXISTS idx_text_unique ON vectors(text)')
            
            # Create FTS table if enabled
            if self.store_cfg.enable_fts:
                conn.execute("""
                    CREATE VIRTUAL TABLE IF NOT EXISTS vectors_fts
                    USING fts5(text, content='vectors', content_rowid='id')
                """)
                
                # Create triggers to keep FTS in sync
                conn.execute("""
                    CREATE TRIGGER IF NOT EXISTS vectors_ai
                    AFTER INSERT ON vectors BEGIN
                        INSERT INTO vectors_fts(rowid, text) VALUES (new.id, new.text);
                    END
                """)
                
                conn.execute("""
                    CREATE TRIGGER IF NOT EXISTS vectors_ad
                    AFTER DELETE ON vectors BEGIN
                        DELETE FROM vectors_fts WHERE rowid = old.id;
                    END
                """)
                
                conn.execute("""
                    CREATE TRIGGER IF NOT EXISTS vectors_au
                    AFTER UPDATE ON vectors BEGIN
                        UPDATE vectors_fts SET text = new.text WHERE rowid = new.id;
                    END
                """)
            
            conn.commit()


class SQLiteVectorStore(NireonBaseComponent, VectorMemoryPort):
    """
    SQLite-based vector store using shared SQLite base for:
    - Connection pooling
    - Retry logic
    - Common optimizations
    - Consistent configuration
    """
    METADATA_DEFINITION = SQLITE_VECTOR_STORE_METADATA

    def __init__(self, config: Dict[str, Any], metadata_definition: Optional[ComponentMetadata] = None):
        super().__init__(config=config, metadata_definition=metadata_definition or self.METADATA_DEFINITION)
        self.store_cfg = SQLiteVectorStoreConfig(**self.config)
        
        # Use the shared SQLite implementation if available
        if HAS_SQLITE_BASE:
            self._impl = SQLiteVectorStoreImpl(self.store_cfg)
        else:
            raise RuntimeError("SQLiteBaseRepository is required for SQLiteVectorStore")
        
        logger.info(
            f"[{self.component_id}] created with db_path={self.store_cfg.db_path}, "
            f"dimensions={self.store_cfg.dimensions}"
        )

    def _serialize_vector(self, vector: Vector) -> bytes:
        """Serialize vector data for storage"""
        return pickle.dumps(vector.data)

    def _deserialize_vector(self, data: bytes) -> Vector:
        """Deserialize vector data from storage"""
        array = pickle.loads(data)
        return Vector(data=array, dims=self.store_cfg.dimensions)

    async def _initialize_impl(self, context: NireonExecutionContext) -> None:
        """Initialize the vector store"""
        context.logger.info(f"[{self.component_id}] initialized successfully.")
        # Check if maintenance needed
        if self._impl.operation_count > self.store_cfg.vacuum_threshold:
            await self._run_maintenance()

    async def _process_impl(self, data: Any, context: NireonExecutionContext) -> ProcessResult:
        """Process implementation for component interface"""
        return ProcessResult.ok(
            "SQLiteVectorStore does not implement a generic process method. "
            "Use specific methods like upsert, query, etc."
        )

    def upsert(self, text: str, vector: Vector, meta: Dict[str, Any]) -> None:
        """Insert or update a vector"""
        if vector.dims != self.store_cfg.dimensions:
            raise ValueError(
                f"Vector dimension mismatch. Store configured for {self.store_cfg.dimensions}, "
                f"but received vector with {vector.dims}."
            )
        
        vector_data = self._serialize_vector(vector)
        metadata_json = json.dumps(meta)
        novelty_score = meta.get('novelty_score', 0.5)
        
        # Use INSERT OR REPLACE for upsert behavior
        self._impl.execute_write("""
            INSERT OR REPLACE INTO vectors (text, vector_data, metadata, novelty_score)
            VALUES (?, ?, ?, ?)
        """, (text, vector_data, metadata_json, novelty_score))
        
        logger.debug(f"Upserted vector for text: {text[:50]}...")

    def upsert_batch(self, entries: List[Tuple[str, Vector, Dict[str, Any]]]) -> None:
        """Batch upsert operation"""
        if not entries:
            return
            
        # Prepare batch data
        batch_data = []
        for text, vector, meta in entries:
            if vector.dims != self.store_cfg.dimensions:
                logger.warning(f"Skipping vector with incorrect dimensions: {vector.dims}")
                continue
                
            vector_data = self._serialize_vector(vector)
            metadata_json = json.dumps(meta)
            novelty_score = meta.get('novelty_score', 0.5)
            batch_data.append((text, vector_data, metadata_json, novelty_score))
        
        if batch_data:
            self._impl.execute_many("""
                INSERT OR REPLACE INTO vectors (text, vector_data, metadata, novelty_score)
                VALUES (?, ?, ?, ?)
            """, batch_data)
            
            logger.debug(f"Batch upserted {len(batch_data)} vectors")

    def query_last(self, n: int = 1000) -> List[Tuple[str, Vector, Dict[str, Any]]]:
        """Query last n entries"""
        if n <= 0:
            return []
            
        rows = self._impl.execute_query("""
            SELECT text, vector_data, metadata
            FROM vectors
            ORDER BY created_at DESC
            LIMIT ?
        """, (n,))
        
        results = []
        for row in rows:
            vector = self._deserialize_vector(row['vector_data'])
            metadata = json.loads(row['metadata'])
            results.append((row['text'], vector, metadata))
            
        # Update access stats for retrieved items
        if results:
            ids = [row['id'] for row in rows]
            placeholders = build_in_clause(len(ids))
            self._impl.execute_write(f"""
                UPDATE vectors 
                SET accessed_at = CURRENT_TIMESTAMP, access_count = access_count + 1
                WHERE id IN {placeholders}
            """, tuple(ids))
            
        return results

    def similarity_search(self, query_vector: Vector, top_k: int = 5,
                         min_similarity: Optional[float] = None) -> List[SearchResult]:
        """Perform similarity search"""
        if query_vector.dims != self.store_cfg.dimensions:
            raise ValueError(
                f"Query vector dimension mismatch. Store configured for {self.store_cfg.dimensions}, "
                f"but received query with {query_vector.dims}."
            )
        
        # For SQLite, we need to fetch all vectors and compute similarity in Python
        # For production use, consider using a specialized vector database
        rows = self._impl.execute_query("""
            SELECT id, text, vector_data, metadata
            FROM vectors
        """)
        
        # Calculate similarities
        similarities = []
        for row in rows:
            stored_vector = self._deserialize_vector(row['vector_data'])
            
            if self.store_cfg.similarity_metric == 'cosine':
                similarity = query_vector.similarity(stored_vector)
            elif self.store_cfg.similarity_metric == 'euclidean':
                similarity = 1.0 / (1.0 + np.linalg.norm(query_vector.data - stored_vector.data))
            elif self.store_cfg.similarity_metric == 'dot':
                similarity = np.dot(query_vector.data, stored_vector.data)
            else:
                similarity = query_vector.similarity(stored_vector)
            
            if min_similarity is None or similarity >= min_similarity:
                metadata = json.loads(row['metadata'])
                similarities.append((similarity, row['id'], row['text'], stored_vector, metadata))
        
        # Sort and get top-k
        similarities.sort(key=lambda x: x[0], reverse=True)
        top_results = similarities[:top_k]
        
        # Update access stats
        if top_results:
            ids = [r[1] for r in top_results]
            placeholders = build_in_clause(len(ids))
            self._impl.execute_write(f"""
                UPDATE vectors 
                SET accessed_at = CURRENT_TIMESTAMP, access_count = access_count + 1
                WHERE id IN {placeholders}
            """, tuple(ids))
        
        # Build results
        results = []
        for similarity, _, text, vector, metadata in top_results:
            results.append(SearchResult(
                text=text,
                vector=vector,
                similarity=similarity,
                metadata=metadata
            ))
            
        return results

    def text_search(self, query: str, top_k: int = 5) -> List[Tuple[str, Vector, Dict[str, Any]]]:
        """Full-text search on text content"""
        if not self.store_cfg.enable_fts:
            raise RuntimeError("Full-text search is not enabled")
            
        rows = self._impl.execute_query("""
            SELECT v.id, v.text, v.vector_data, v.metadata
            FROM vectors v
            JOIN vectors_fts ON v.id = vectors_fts.rowid
            WHERE vectors_fts MATCH ?
            ORDER BY rank
            LIMIT ?
        """, (query, top_k))
        
        results = []
        for row in rows:
            vector = self._deserialize_vector(row['vector_data'])
            metadata = json.loads(row['metadata'])
            results.append((row['text'], vector, metadata))
            
        return results

    def stats(self) -> VectorMemoryStats:
        """Get statistics about the vector store"""
        # Get basic database stats
        base_stats = self._impl.get_database_stats()
        
        # Get vector-specific stats
        stats_rows = self._impl.execute_query("""
            SELECT 
                COUNT(*) as total,
                AVG(novelty_score) as avg_novelty,
                MIN(novelty_score) as min_novelty,
                MAX(novelty_score) as max_novelty,
                COUNT(CASE WHEN novelty_score > 0.75 THEN 1 END) as high_novelty_count,
                AVG(access_count) as avg_access_count,
                MAX(access_count) as max_access_count,
                COUNT(CASE WHEN datetime(accessed_at) > datetime('now', '-1 hour') THEN 1 END) as recent_access_count
            FROM vectors
        """)
        
        if not stats_rows or stats_rows[0]['total'] == 0:
            return VectorMemoryStats(
                total_count=0,
                average_novelty=0.0,
                min_novelty=0.0,
                max_novelty=0.0,
                high_novelty_count=0,
                novelty_threshold=0.75
            )
        
        stats_row = stats_rows[0]
        
        stats = VectorMemoryStats(
            total_count=int(stats_row['total']),
            average_novelty=float(stats_row['avg_novelty'] or 0),
            min_novelty=float(stats_row['min_novelty'] or 0),
            max_novelty=float(stats_row['max_novelty'] or 0),
            high_novelty_count=int(stats_row['high_novelty_count'] or 0),
            novelty_threshold=0.75
        )
        
        # Add extended metadata
        stats.metadata = {
            **base_stats,
            'avg_access_count': float(stats_row['avg_access_count'] or 0),
            'max_access_count': int(stats_row['max_access_count'] or 0),
            'recent_access_count': int(stats_row['recent_access_count'] or 0),
            'fts_enabled': self.store_cfg.enable_fts,
            'similarity_metric': self.store_cfg.similarity_metric
        }
        
        return stats

    async def _run_maintenance(self):
        """Run database maintenance tasks"""
        try:
            await asyncio.get_event_loop().run_in_executor(
                None, self._impl.vacuum_database
            )
            logger.info(f"[{self.component_id}] maintenance completed")
        except Exception as e:
            logger.error(f"Maintenance failed: {e}")

    def clear(self) -> None:
        """Clear all stored vectors"""
        self._impl.execute_write('DELETE FROM vectors')
        if self.store_cfg.enable_fts:
            self._impl.execute_write('DELETE FROM vectors_fts')
        logger.info(f"[{self.component_id}] cleared all vectors")

    def close(self):
        """Close all database connections"""
        self._impl.close()

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
            f"SQLiteVectorStore("
            f"db_path={self.store_cfg.db_path}, "
            f"dimensions={self.store_cfg.dimensions}, "
            f"similarity_metric={self.store_cfg.similarity_metric})"
        )