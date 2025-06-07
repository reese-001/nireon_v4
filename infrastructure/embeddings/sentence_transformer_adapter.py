"""
NIREON V4 Sentence Transformers Adapter
Local embedding provider using Sentence Transformers
"""
import logging
from typing import Sequence
from domain.ports.embedding_port import EmbeddingPort
from domain.embeddings.vector import Vector
import numpy as np

logger = logging.getLogger(__name__)


class SentenceTransformerAdapter(EmbeddingPort):
    """
    Local embedding adapter using Sentence Transformers models.
    Provides fallback to deterministic random embeddings when model unavailable.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", dimensions: int = 384, cache_size: int = 1000):
        self.model_name = model_name
        self.dimensions = dimensions
        self.cache_size = cache_size
        
        # Initialize model
        self.model = None
        self._init_model()
        
        # In-memory cache
        self._cache: dict[str, Vector] = {}
        self.encode_count = 0
        
        logger.info(f'SentenceTransformerAdapter initialized (model: {self.model_name}, dimensions: {self.dimensions})')
    
    def _init_model(self):
        """Initialize the Sentence Transformers model."""
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
            logger.info(f'Loaded Sentence Transformer model: {self.model_name}')
        except ImportError as e:
            logger.error(f'sentence-transformers not installed: {e}')
            logger.error('Please install: pip install sentence-transformers')
            self.model = None
        except Exception as e:
            logger.error(f'Failed to load model {self.model_name}: {e}')
            self.model = None
    
    def encode(self, text: str) -> Vector:
        """Encode text to vector embedding."""
        self.encode_count += 1
        
        # Check cache first
        if text in self._cache:
            logger.debug(f"Retrieved cached embedding for text (call #{self.encode_count})")
            return self._cache[text]
        
        if self.model is None:
            # Fallback to deterministic random embedding
            logger.warning(f"Model not available, generating deterministic random embedding (call #{self.encode_count})")
            return self._generate_fallback_embedding(text)
        
        try:
            # Generate embedding using Sentence Transformers
            embedding_array = self.model.encode(text, normalize_embeddings=True)
            
            # Convert to our Vector format
            vector = Vector(data=embedding_array.astype(np.float64))
            
            # Cache the result
            if len(self._cache) < self.cache_size:
                self._cache[text] = vector
            
            logger.debug(f"Generated embedding for text (call #{self.encode_count})")
            return vector
            
        except Exception as e:
            logger.error(f"Error generating embedding (call #{self.encode_count}): {e}")
            return self._generate_fallback_embedding(text)
    
    def encode_batch(self, texts: Sequence[str]) -> list[Vector]:
        """Encode multiple texts to vector embeddings."""
        if self.model is None:
            return [self._generate_fallback_embedding(text) for text in texts]
        
        try:
            # Check cache for all texts
            cached_results = {}
            uncached_texts = []
            uncached_indices = []
            
            for i, text in enumerate(texts):
                if text in self._cache:
                    cached_results[i] = self._cache[text]
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(i)
            
            # Generate embeddings for uncached texts
            if uncached_texts:
                embeddings_array = self.model.encode(uncached_texts, normalize_embeddings=True)
                
                for i, (text, embedding_array) in enumerate(zip(uncached_texts, embeddings_array)):
                    vector = Vector(data=embedding_array.astype(np.float64))
                    original_index = uncached_indices[i]
                    cached_results[original_index] = vector
                    
                    # Cache if space available
                    if len(self._cache) < self.cache_size:
                        self._cache[text] = vector
            
            self.encode_count += len(uncached_texts)
            
            # Return results in original order
            return [cached_results[i] for i in range(len(texts))]
            
        except Exception as e:
            logger.error(f"Error in batch encoding: {e}")
            return [self._generate_fallback_embedding(text) for text in texts]
    
    def _generate_fallback_embedding(self, text: str) -> Vector:
        """Generate deterministic random embedding when model is unavailable."""
        text_hash = hash(text)
        rng = np.random.default_rng(seed=abs(text_hash) % (2**32))
        embedding_data = rng.standard_normal(self.dimensions, dtype=np.float64)
        
        # Normalize the vector
        norm = np.linalg.norm(embedding_data)
        if norm > 0:
            embedding_data = embedding_data / norm
        
        vector = Vector(data=embedding_data)
        
        # Cache the fallback embedding
        if len(self._cache) < self.cache_size:
            self._cache[text] = vector
        
        return vector
    
    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._cache.clear()
        logger.debug('Embedding cache cleared')
    
    def get_stats(self) -> dict:
        """Get adapter statistics."""
        return {
            'encode_count': self.encode_count,
            'model_name': self.model_name,
            'dimensions': self.dimensions,
            'cache_size': len(self._cache),
            'max_cache_size': self.cache_size,
            'has_model': self.model is not None
        }