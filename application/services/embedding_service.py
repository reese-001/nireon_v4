# application/services/embedding_service.py
import logging
import uuid
from typing import Any, Dict

from core.base_component import NireonBaseComponent
from core.lifecycle import ComponentMetadata
from domain.context import NireonExecutionContext
from core.results import ComponentHealth, ProcessResult
from domain.ports.embedding_port import EmbeddingPort
from domain.ports.vector_memory_port import VectorMemoryPort
from domain.ports.event_bus_port import EventBusPort
from domain.embeddings.vector import Vector
from events.embedding_events import (
    EmbeddingComputedEvent, HighNoveltyDetectedEvent, EmbeddingErrorEvent,
    EMBEDDING_COMPUTED, HIGH_NOVELTY_DETECTED, EMBEDDING_ERROR
)

from config.embedding_config import EmbeddingConfig

logger = logging.getLogger(__name__)

# Define metadata for the EmbeddingService
EMBEDDING_SERVICE_METADATA = ComponentMetadata(
    id='embedding_service',
    name='Embedding Service',
    version='1.0.0',
    category='service_core',
    description='Core service for text embedding computation and novelty detection',
    epistemic_tags=['embedder', 'analyzer', 'detector'],
    capabilities={'compute_embeddings', 'detect_novelty', 'vector_storage'},
    accepts=['EMBEDDING_REQUEST', 'NOVELTY_CHECK'],
    produces=['EMBEDDING_COMPUTED', 'HIGH_NOVELTY_DETECTED', 'EMBEDDING_ERROR'],
    requires_initialize=True
)

class EmbeddingService(NireonBaseComponent):
    def __init__(self, config: Dict[str, Any], metadata_definition: ComponentMetadata = EMBEDDING_SERVICE_METADATA):
        super().__init__(config=config, metadata_definition=metadata_definition)
        
        # Parse the config into EmbeddingConfig if it's a dict
        if isinstance(config, dict):
            self.embedding_cfg = EmbeddingConfig(**config)
        else:
            self.embedding_cfg = config
            
        # Initialize service-specific attributes
        self.embedding_port: EmbeddingPort | None = None
        self.vector_memory_port: VectorMemoryPort | None = None
        self.event_bus_port: EventBusPort | None = None
        self.total_embeddings_computed = 0
        self.high_novelty_detections = 0
        self.errors_count = 0
        
        logger.info(f'EmbeddingService created with provider: {self.embedding_cfg.provider}')

    async def _initialize_impl(self, context: NireonExecutionContext) -> None:
        """Initialize the embedding service with required dependencies."""
        try:
            # Get dependencies from registry
            if context.component_registry:
                self.embedding_port = context.component_registry.get_service_instance(EmbeddingPort)
                self.vector_memory_port = context.component_registry.get_service_instance(VectorMemoryPort)
                self.event_bus_port = context.component_registry.get_service_instance(EventBusPort)
            
            if not self.embedding_port:
                raise RuntimeError(f"Embedding provider '{self.embedding_cfg.provider}' not found in registry")
            
            if not self.vector_memory_port:
                logger.warning(f"Vector memory '{self.embedding_cfg.vector_memory_ref}' not found - continuing without vector storage")
            
            if not self.event_bus_port:
                logger.warning('Event bus not found in registry - events will not be published')
                
            logger.info('EmbeddingService initialized successfully')
        except Exception as e:
            logger.error(f'Failed to initialize EmbeddingService: {e}')
            raise

    async def _process_impl(self, data: Any, context: NireonExecutionContext) -> ProcessResult:
        """Process embedding requests."""
        try:
            # Extract text from input data
            if isinstance(data, str):
                text = data
            elif isinstance(data, dict) and 'text' in data:
                text = data['text']
            else:
                return ProcessResult(
                    success=False,
                    component_id=self.component_id,
                    message='Invalid input: expected string or dict with "text" key',
                    error_code='INVALID_INPUT'
                )
            
            # Compute embedding
            vector = self.get_embedding(text)
            
            # Return successful result
            return ProcessResult(
                success=True,
                component_id=self.component_id,
                output_data={
                    'text': text,
                    'vector': vector,
                    'dimensions': vector.dims,
                    'provider': self.embedding_cfg.provider
                },
                message=f'Successfully computed embedding for text of length {len(text)}'
            )
            
        except Exception as e:
            self.errors_count += 1
            logger.error(f'Error processing embedding request: {e}')
            return ProcessResult(
                success=False,
                component_id=self.component_id,
                message=f'Failed to process embedding: {e}',
                error_code='PROCESSING_ERROR'
            )

    def get_embedding(self, text: str) -> Vector:
        """Get embedding for text with novelty detection."""
        if not self.embedding_port:
            raise RuntimeError('EmbeddingService not properly initialized')

        try:
            vector = self.embedding_port.encode(text)
            vector_id = str(uuid.uuid4())
            
            # Compute novelty score
            novelty_score, similarity_max = self._compute_novelty(vector)
            
            # Store in vector memory if available
            if self.vector_memory_port:
                metadata = {
                    'novelty_score': novelty_score,
                    'source': 'embedding_service',
                    'provider': self.embedding_cfg.provider,
                    'vector_id': vector_id
                }
                self.vector_memory_port.upsert(text, vector, metadata)

            # Emit events
            self._emit_embedding_computed(text, vector_id, similarity_max, novelty_score)
            
            if novelty_score > self.embedding_cfg.novelty_threshold:
                self._emit_high_novelty_detected(text, novelty_score)
                self.high_novelty_detections += 1

            self.total_embeddings_computed += 1
            logger.debug(f'Computed embedding for text with novelty {novelty_score:.3f}')
            return vector
            
        except Exception as e:
            self.errors_count += 1
            self._emit_embedding_error(text, str(e))
            logger.error(f'Error computing embedding: {e}')
            raise

    def get_novelty(self, text: str) -> float:
        """Get novelty score for text without storing the embedding."""
        if not self.embedding_port:
            raise RuntimeError('EmbeddingService not properly initialized')
        
        try:
            vector = self.embedding_port.encode(text)
            novelty_score, _ = self._compute_novelty(vector)
            return novelty_score
        except Exception as e:
            logger.error(f'Error computing novelty: {e}')
            return 0.5  # Default neutral novelty

    def _compute_novelty(self, vector: Vector) -> tuple[float, float]:
        """Compute novelty score based on similarity to stored vectors."""
        if not self.vector_memory_port:
            return (0.5, 0.5)  # Neutral novelty if no memory available
        
        try:
            recent_vectors = self.vector_memory_port.query_last(1000)
            if not recent_vectors:
                return (1.0, 0.0)  # Maximum novelty if no stored vectors
            
            max_similarity = 0.0
            for _, stored_vector, _ in recent_vectors:
                try:
                    similarity = vector.similarity(stored_vector)
                    max_similarity = max(max_similarity, similarity)
                except Exception as e:
                    logger.debug(f'Error comparing vectors: {e}')
                    continue
            
            novelty_score = 1.0 - max_similarity
            return (novelty_score, max_similarity)
            
        except Exception as e:
            logger.error(f'Error computing novelty: {e}')
            return (0.5, 0.5)  # Default neutral values

    def _emit_embedding_computed(self, text: str, vector_id: str, similarity_max: float, novelty_score: float):
        """Emit embedding computed event."""
        if self.event_bus_port:
            event = EmbeddingComputedEvent(
                text=text,
                vector_id=vector_id,
                similarity_max=similarity_max,
                novelty_score=novelty_score,
                provider=self.embedding_cfg.provider
            )
            self.event_bus_port.publish(EMBEDDING_COMPUTED, event)

    def _emit_high_novelty_detected(self, text: str, novelty_score: float):
        """Emit high novelty detected event."""
        if self.event_bus_port:
            event = HighNoveltyDetectedEvent(
                text=text,
                novelty_score=novelty_score
            )
            self.event_bus_port.publish(HIGH_NOVELTY_DETECTED, event)

    def _emit_embedding_error(self, text: str, error: str):
        """Emit embedding error event."""
        if self.event_bus_port:
            event = EmbeddingErrorEvent(
                text=text,
                error=error,
                provider=self.embedding_cfg.provider
            )
            self.event_bus_port.publish(EMBEDDING_ERROR, event)

    async def health_check(self, context: NireonExecutionContext) -> ComponentHealth:
        """Perform health check on the embedding service."""
        from core.results import ComponentHealth
        
        try:
            health_status = {
                'service_status': 'healthy',
                'total_embeddings_computed': self.total_embeddings_computed,
                'high_novelty_detections': self.high_novelty_detections,
                'errors_count': self.errors_count,
                'provider': self.embedding_cfg.provider,
                'novelty_threshold': self.embedding_cfg.novelty_threshold
            }
            
            # Add provider stats if available
            if hasattr(self.embedding_port, 'get_stats'):
                health_status['provider_stats'] = self.embedding_port.get_stats()
            
            # Add vector memory stats if available
            if self.vector_memory_port:
                memory_stats = self.vector_memory_port.stats()
                health_status['vector_memory_stats'] = {
                    'total_vectors': memory_stats.total_count,
                    'average_novelty': memory_stats.average_novelty,
                    'high_novelty_count': memory_stats.high_novelty_count
                }
            
            status = 'HEALTHY'
            message = 'EmbeddingService operating normally'
            
            if self.errors_count > 0:
                status = 'DEGRADED'
                message = f'EmbeddingService has {self.errors_count} errors'
            
            if not self.embedding_port:
                status = 'UNHEALTHY'
                message = 'EmbeddingService missing embedding port'
            
            return ComponentHealth(
                component_id=self.component_id,
                status=status,
                message=message,
                details=health_status
            )
            
        except Exception as e:
            logger.error(f'Error in health check: {e}')
            return ComponentHealth(
                component_id=self.component_id,
                status='UNHEALTHY',
                message=f'Health check failed: {e}',
                details={'error': str(e)}
            )

    def get_stats(self) -> dict[str, Any]:
        """Get service statistics."""
        return {
            'total_embeddings_computed': self.total_embeddings_computed,
            'high_novelty_detections': self.high_novelty_detections,
            'errors_count': self.errors_count,
            'provider': self.embedding_cfg.provider,
            'novelty_threshold': self.embedding_cfg.novelty_threshold,
            'is_initialized': self.is_initialized
        }