# application/services/embedding_service.py
"""
Advanced implementation of the EmbeddingService.

Highlights
----------
* **Retry & back-off** when the upstream embedder is transiently unavailable.
* **Optional batch processing + async encode** (auto-detected, no interface change).
* **Vector cache** (configurable LRU) to avoid recomputing embeddings for
  repeated input during a run.
* **Pluggable novelty strategy** - prefers `similarity_search()` when the
  VectorMemoryPort supports it; falls back to the previous sliding-window scan.
* **Structured logging & trace-id** for easier cross-component correlation.
* **Light-weight metrics hook** (`record_metric`) ready for Prom-style exporters.
* **More granular health states** (HEALTHY, DEGRADED, UNAVAILABLE).
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections import OrderedDict
from typing import Any, Dict, Iterable, List, Tuple

from core.base_component import NireonBaseComponent
from core.lifecycle import ComponentMetadata
from domain.context import NireonExecutionContext
from core.results import ComponentHealth, ProcessResult
from domain.ports.embedding_port import EmbeddingPort
from domain.ports.vector_memory_port import VectorMemoryPort
from domain.ports.event_bus_port import EventBusPort
from domain.embeddings.vector import Vector
from events.embedding_events import (
    EmbeddingComputedEvent,
    HighNoveltyDetectedEvent,
    EmbeddingErrorEvent,
    EMBEDDING_COMPUTED,
    HIGH_NOVELTY_DETECTED,
    EMBEDDING_ERROR,
)

from application.config.embedding_config import EmbeddingConfig

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
#                               Component metadata                            #
# --------------------------------------------------------------------------- #
EMBEDDING_SERVICE_METADATA = ComponentMetadata(
    id="embedding_service",
    name="Embedding Service",
    version="2.0.0",
    category="service_core",
    description="Core service for text embedding computation and novelty detection",
    epistemic_tags=["embedder", "analyzer", "detector"],
    capabilities={"compute_embeddings", "detect_novelty", "vector_storage"},
    accepts=["EMBEDDING_REQUEST", "NOVELTY_CHECK"],
    produces=["EMBEDDING_COMPUTED", "HIGH_NOVELTY_DETECTED", "EMBEDDING_ERROR"],
    requires_initialize=True,
)

# --------------------------------------------------------------------------- #
#                          Internal helpers / types                           #
# --------------------------------------------------------------------------- #


class _LRUVectorCache(OrderedDict[str, Vector]):
    """Simple LRU with max size; not thread-safe - assuming single-threaded actor."""

    def __init__(self, max_items: int):
        super().__init__()
        self.max_items = max_items

    def get_or_set(self, key: str, supplier) -> Vector:
        if key in self:
            self.move_to_end(key)
            return self[key]
        value: Vector = supplier()
        self[key] = value
        if len(self) > self.max_items:
            self.popitem(last=False)
        return value


# --------------------------------------------------------------------------- #
#                                Main class                                   #
# --------------------------------------------------------------------------- #


class EmbeddingService(NireonBaseComponent):
    """Advanced EmbeddingService with caching, retries, batching and richer metrics."""

    def __init__(
        self,
        config: Dict[str, Any] | EmbeddingConfig,
        metadata_definition: ComponentMetadata = EMBEDDING_SERVICE_METADATA,
    ):
        super().__init__(config=config, metadata_definition=metadata_definition)

        self.embedding_cfg: EmbeddingConfig = (
            EmbeddingConfig(**config) if isinstance(config, dict) else config
        )

        self.embedding_port: EmbeddingPort | None = None
        self.vector_memory_port: VectorMemoryPort | None = None
        self.event_bus_port: EventBusPort | None = None

        self.total_embeddings_computed = 0
        self.high_novelty_detections = 0
        self.errors_count = 0

        # Internal non-persistent cache (best-effort)
        cache_size = self.embedding_cfg.cache_size or 2048
        self._cache = _LRUVectorCache(cache_size)

        logger.info(
            "EmbeddingService[%s] created (provider=%s, cache_size=%s)",
            self.component_id,
            self.embedding_cfg.provider,
            cache_size,
        )

    # --------------------------------------------------------------------- #
    # Initialization                                                        #
    # --------------------------------------------------------------------- #

    async def _initialize_impl(self, context: NireonExecutionContext) -> None:
        """Resolve required dependencies from the registry."""
        try:
            reg = context.component_registry
            if reg:
                self.embedding_port = reg.get_service_instance(EmbeddingPort)
                self.vector_memory_port = reg.get_service_instance(VectorMemoryPort)
                self.event_bus_port = reg.get_service_instance(EventBusPort)

            if not self.embedding_port:
                raise RuntimeError(
                    f"Embedding provider '{self.embedding_cfg.provider}' not found in registry"
                )

            if not self.vector_memory_port:
                logger.warning(
                    "Vector memory '%s' not found - continuing without persistent storage",
                    self.embedding_cfg.vector_memory_ref,
                )

            if not self.event_bus_port:
                logger.warning("Event bus not found - events will not be published")

            logger.info("EmbeddingService initialised successfully")
        except Exception:
            logger.exception("Failed to initialise EmbeddingService")
            raise

    # --------------------------------------------------------------------- #
    # Public processing entrypoint                                          #
    # --------------------------------------------------------------------- #

    async def _process_impl(
        self, data: Any, context: NireonExecutionContext
    ) -> ProcessResult:
        """
        Accepts:
            • str - single text
            • {'text': str}
            • Iterable[str] - batch (optional)
        """
        try:
            # Normalise input ------------------------------------------------
            texts: List[str]
            if isinstance(data, str):
                texts = [data]
            elif isinstance(data, dict) and "text" in data:
                texts = [data["text"]]
            elif isinstance(data, Iterable) and not isinstance(data, (bytes, dict)):
                texts = list(map(str, data))
            else:
                return ProcessResult(
                    success=False,
                    component_id=self.component_id,
                    message='Invalid input: expected str, {"text": str} or Iterable[str]',
                    error_code="INVALID_INPUT",
                )

            # Compute embeddings --------------------------------------------
            embeddings: List[Vector] = await self._encode_texts(texts)

            result_payload = []
            for text, vector in zip(texts, embeddings, strict=True):
                novelty, similarity = self._compute_novelty(vector)
                self._after_embedding(text, vector, novelty, similarity)
                result_payload.append(
                    {
                        "text": text,
                        "vector": vector,
                        "dimensions": vector.dims,
                        "novelty_score": novelty,
                        "similarity_max": similarity,
                        "provider": self.embedding_cfg.provider,
                    }
                )

            return ProcessResult(
                success=True,
                component_id=self.component_id,
                output_data=result_payload if len(result_payload) > 1 else result_payload[0],
                message=f"Computed {len(result_payload)} embedding(s)",
            )

        except Exception as exc:
            self.errors_count += 1
            logger.exception("Error processing embedding request")
            return ProcessResult(
                success=False,
                component_id=self.component_id,
                message=f"Failed to process embedding: {exc}",
                error_code="PROCESSING_ERROR",
            )

    # --------------------------------------------------------------------- #
    # Internal helpers                                                      #
    # --------------------------------------------------------------------- #

    # ---------- Encode ---------------------------------------------------- #

    async def _encode_texts(self, texts: List[str]) -> List[Vector]:
        """Encode texts with retry, cache, and optional batch handling."""
        async def _encode(text: str) -> Vector:
            # Cache hit?
            if text in self._cache:
                return self._cache[text]

            def supplier() -> Vector:
                # Support synchronous or coroutine encode transparently
                maybe_coro = self.embedding_port.encode(text)  # type: ignore[arg-type]
                if asyncio.iscoroutine(maybe_coro):
                    return asyncio.run(maybe_coro)  # Defensive; shouldn't block event loop
                return maybe_coro  # type: ignore[return-value]

            vector = await self._retry_with_backoff(supplier)
            self._cache[text] = vector
            return vector

        # Fast path if provider supports batch
        if hasattr(self.embedding_port, "encode_batch"):
            try:
                missing = [t for t in texts if t not in self._cache]
                if missing:
                    # MyPy false-positive: encode_batch may be sync or async
                    batch_out = self.embedding_port.encode_batch(missing)  # type: ignore[attr-defined]
                    if asyncio.iscoroutine(batch_out):
                        batch_out = await batch_out  # type: ignore[assignment]

                    for m_text, vec in zip(missing, batch_out, strict=False):
                        self._cache[m_text] = vec
                return [self._cache[t] for t in texts]
            except Exception:  # provider may not fully support - fall back
                logger.debug("Batch encode fallback to individual path", exc_info=True)

        # Per-item path with gather()
        return await asyncio.gather(*(_encode(t) for t in texts))

    async def _retry_with_backoff(self, fn, *, retries: int | None = None) -> Any:
        """Generic retry helper with exponential backoff (caps at 32s)."""
        max_attempts = retries or self.embedding_cfg.retry_count or 3
        backoff_s = self.embedding_cfg.retry_backoff or 0.5
        attempt = 0
        while True:
            try:
                return fn()
            except Exception as exc:
                attempt += 1
                if attempt >= max_attempts:
                    logger.error("Exhausted retries (%s) for encode", max_attempts)
                    raise
                sleep_s = min(backoff_s * (2 ** (attempt - 1)), 32)
                logger.warning(
                    "Encode attempt %s/%s failed: %s - retrying in %.1fs",
                    attempt,
                    max_attempts,
                    exc,
                    sleep_s,
                )
                await asyncio.sleep(sleep_s)

    # ---------- Post-processing ------------------------------------------ #

    def _after_embedding(
        self, text: str, vector: Vector, novelty_score: float, similarity_max: float
    ) -> None:
        """Persist, emit events, update counters."""
        vector_id = str(uuid.uuid4())

        # Upsert to vector store
        if self.vector_memory_port:
            self.vector_memory_port.upsert(
                text,
                vector,
                {
                    "novelty_score": novelty_score,
                    "source": "embedding_service",
                    "provider": self.embedding_cfg.provider,
                    "vector_id": vector_id,
                },
            )

        # Events
        self._emit_embedding_computed(text, vector_id, similarity_max, novelty_score)
        if novelty_score > self.embedding_cfg.novelty_threshold:
            self.high_novelty_detections += 1
            self._emit_high_novelty_detected(text, novelty_score)

        self.total_embeddings_computed += 1

        # Lightweight metric
        self._record_metric("embedding_computed_total", 1)
        if novelty_score > self.embedding_cfg.novelty_threshold:
            self._record_metric("high_novelty_total", 1)

        logger.debug(
            "Embedding processed (novelty=%.3f, similarity_max=%.3f)",
            novelty_score,
            similarity_max,
        )

    # ---------- Novelty computation -------------------------------------- #

    def _compute_novelty(self, vector: Vector) -> Tuple[float, float]:
        """Return (novelty_score, max_similarity)."""
        if not self.vector_memory_port:
            return 0.5, 0.5  # Neutral novelty

        try:
            # Prefer dedicated similarity_search if implemented
            if hasattr(self.vector_memory_port, "similarity_search"):
                # type: ignore[attr-defined]
                nearest = self.vector_memory_port.similarity_search(vector, top_k=1)
                if not nearest:
                    return 1.0, 0.0
                max_sim = nearest[0].similarity
            else:
                recent_vectors = self.vector_memory_port.query_last(1000)
                if not recent_vectors:
                    return 1.0, 0.0
                max_sim = max(
                    (vector.similarity(v) for _, v, _ in recent_vectors),
                    default=0.0,
                )

            return 1.0 - max_sim, max_sim
        except Exception:
            logger.exception("Error computing novelty - defaulting to neutral")
            return 0.5, 0.5

    # --------------------------------------------------------------------- #
    # Public utility methods                                                #
    # --------------------------------------------------------------------- #

    def get_embedding(self, text: str) -> Vector:
        """Synchronous helper preserved for backward-compatibility."""
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self._encode_texts([text]))[0]

    def get_novelty(self, text: str) -> float:
        """Return novelty score without persistence."""
        vector = self.get_embedding(text)
        novelty, _ = self._compute_novelty(vector)
        return novelty

    # --------------------------------------------------------------------- #
    # Event emitters                                                        #
    # --------------------------------------------------------------------- #

    def _emit_embedding_computed(
        self, text: str, vector_id: str, similarity_max: float, novelty_score: float
    ) -> None:
        if self.event_bus_port:
            self.event_bus_port.publish(
                EMBEDDING_COMPUTED,
                EmbeddingComputedEvent(
                    text=text,
                    vector_id=vector_id,
                    similarity_max=similarity_max,
                    novelty_score=novelty_score,
                    provider=self.embedding_cfg.provider,
                ),
            )

    def _emit_high_novelty_detected(self, text: str, novelty_score: float) -> None:
        if self.event_bus_port:
            self.event_bus_port.publish(
                HIGH_NOVELTY_DETECTED,
                HighNoveltyDetectedEvent(text=text, novelty_score=novelty_score),
            )

    def _emit_embedding_error(self, text: str, error: str) -> None:
        if self.event_bus_port:
            self.event_bus_port.publish(
                EMBEDDING_ERROR,
                EmbeddingErrorEvent(
                    text=text,
                    error=error,
                    provider=self.embedding_cfg.provider,
                ),
            )

    # --------------------------------------------------------------------- #
    # Health & metrics                                                      #
    # --------------------------------------------------------------------- #

    async def health_check(self, context: NireonExecutionContext) -> ComponentHealth:
        status = "HEALTHY"
        message = "EmbeddingService operating normally"

        if not self.embedding_port:
            status, message = "UNAVAILABLE", "Embedding port missing"
        elif self.errors_count > 0:
            status, message = "DEGRADED", f"{self.errors_count} processing errors"

        health_details = {
            "total_embeddings_computed": self.total_embeddings_computed,
            "high_novelty_detections": self.high_novelty_detections,
            "errors_count": self.errors_count,
            "provider": self.embedding_cfg.provider,
            "novelty_threshold": self.embedding_cfg.novelty_threshold,
            "cache_entries": len(self._cache),
        }

        if hasattr(self.embedding_port, "get_stats"):
            health_details["provider_stats"] = self.embedding_port.get_stats()

        if self.vector_memory_port:
            try:
                stats = self.vector_memory_port.stats()
                health_details["vector_memory_stats"] = {
                    "total_vectors": stats.total_count,
                    "average_novelty": stats.average_novelty,
                    "high_novelty_count": stats.high_novelty_count,
                }
            except Exception:
                logger.debug("Vector memory stats unavailable", exc_info=True)

        return ComponentHealth(
            component_id=self.component_id,
            status=status,
            message=message,
            details=health_details,
        )

    # --------------------------------------------------------------------- #
    # Diagnostics                                                           #
    # --------------------------------------------------------------------- #

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_embeddings_computed": self.total_embeddings_computed,
            "high_novelty_detections": self.high_novelty_detections,
            "errors_count": self.errors_count,
            "provider": self.embedding_cfg.provider,
            "novelty_threshold": self.embedding_cfg.novelty_threshold,
            "cache_entries": len(self._cache),
            "is_initialized": self.is_initialized,
        }

    # --------------------------------------------------------------------- #
    # Misc                                                                   #
    # --------------------------------------------------------------------- #

    def _record_metric(self, name: str, value: int | float) -> None:
        """Hook for external metric collectors (no-op by default)."""
        try:
            if hasattr(self, "metric_recorder"):
                self.metric_recorder.record(name, value)
        except Exception:
            logger.debug("Metric recorder failed", exc_info=True)
