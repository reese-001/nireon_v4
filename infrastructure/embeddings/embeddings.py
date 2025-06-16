import logging
import os
import httpx
from typing import Any, Dict, List, Sequence, Optional
from domain.ports.embedding_port import EmbeddingPort
from domain.embeddings.vector import Vector
import numpy as np

logger = logging.getLogger(__name__)

class EmbeddingAdapter(EmbeddingPort):
    """Generic embeddings adapter for NIREON V4 that supports multiple providers"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.provider = self.config.get('provider', 'sentence_transformers')
        self.model = self.config.get('model', 'all-MiniLM-L6-v2')
        self.dimensions = self.config.get('dimensions', 384)
        self.timeout = self.config.get('timeout', 30)
        self.max_retries = self.config.get('max_retries', 3)
        
        self._cache: Dict[str, Vector] = {}
        self.cache_size = self.config.get('cache_size', 1000)
        self.encode_count = 0
        
        # Initialize based on provider
        self._init_provider()
        
        logger.info(f'EmbeddingAdapter initialized (provider: {self.provider}, model: {self.model}, dimensions: {self.dimensions})')

    def _init_provider(self):
        """Initialize the embedding provider based on configuration"""
        if self.provider == 'sentence_transformers':
            self._init_sentence_transformers()
        elif self.provider == 'remote_api':
            self._init_remote_api()
        elif self.provider == 'mock':
            self._init_mock()
        else:
            logger.warning(f"Unknown provider '{self.provider}', falling back to mock")
            self._init_mock()

    def _init_sentence_transformers(self):
        """Initialize Sentence Transformers provider"""
        try:
            from sentence_transformers import SentenceTransformer
            self.st_model = SentenceTransformer(self.model)
            self._use_sentence_transformers = True
            logger.info(f'Sentence Transformers model loaded: {self.model}')
        except ImportError:
            logger.warning("sentence-transformers not installed, falling back to mock")
            self._init_mock()
        except Exception as e:
            logger.warning(f"Failed to load Sentence Transformers model: {e}, falling back to mock")
            self._init_mock()

    def _init_remote_api(self):
        """Initialize remote API provider"""
        self.api_key = os.getenv('EMBEDDING_API_KEY') or self.config.get('api_key')
        self.base_url = self.config.get('base_url', 'https://api.example.com/v1')
        
        if not self.api_key:
            logger.warning("No API key found for remote provider, falling back to mock")
            self._init_mock()
            return
            
        self.client = httpx.Client(
            timeout=self.timeout,
            headers={
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }
        )
        self._use_remote_api = True
        logger.info(f'Remote API provider initialized (base_url: {self.base_url})')

    def _init_mock(self):
        """Initialize mock provider for development/testing"""
        self._use_mock = True
        logger.info('Mock embedding provider initialized')

    def encode(self, text: str) -> Vector:
        """Encode a single text string into a vector"""
        self.encode_count += 1
        
        # Check cache first
        if text in self._cache:
            logger.debug(f'Retrieved cached embedding for text (call #{self.encode_count})')
            return self._cache[text]
        
        # Route to appropriate provider
        if hasattr(self, '_use_sentence_transformers') and self._use_sentence_transformers:
            vector = self._encode_sentence_transformers(text)
        elif hasattr(self, '_use_remote_api') and self._use_remote_api:
            vector = self._encode_remote_api(text)
        else:
            vector = self._encode_mock(text)
        
        # Cache the result
        if len(self._cache) < self.cache_size:
            self._cache[text] = vector
            
        return vector

    def encode_batch(self, texts: Sequence[str]) -> List[Vector]:
        """Encode multiple texts in a batch"""
        # Check cache for each text
        cached_results = {}
        uncached_texts = []
        uncached_indices = []
        
        for i, text in enumerate(texts):
            if text in self._cache:
                cached_results[i] = self._cache[text]
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        # Process uncached texts
        if uncached_texts:
            if hasattr(self, '_use_sentence_transformers') and self._use_sentence_transformers:
                new_vectors = self._encode_batch_sentence_transformers(uncached_texts)
            elif hasattr(self, '_use_remote_api') and self._use_remote_api:
                new_vectors = self._encode_batch_remote_api(uncached_texts)
            else:
                new_vectors = [self._encode_mock(text) for text in uncached_texts]
            
            # Store results and cache
            for i, (text, vector) in enumerate(zip(uncached_texts, new_vectors)):
                original_index = uncached_indices[i]
                cached_results[original_index] = vector
                
                if len(self._cache) < self.cache_size:
                    self._cache[text] = vector
        
        self.encode_count += len(uncached_texts)
        return [cached_results[i] for i in range(len(texts))]

    def _encode_sentence_transformers(self, text: str) -> Vector:
        """Encode using Sentence Transformers"""
        try:
            embedding_array = self.st_model.encode(text, normalize_embeddings=True)
            vector_data = np.array(embedding_array, dtype=np.float64)
            return Vector(data=vector_data)
        except Exception as e:
            logger.error(f'Sentence Transformers encoding failed: {e}')
            return self._encode_mock(text)

    def _encode_batch_sentence_transformers(self, texts: List[str]) -> List[Vector]:
        """Batch encode using Sentence Transformers"""
        try:
            embeddings_array = self.st_model.encode(texts, normalize_embeddings=True)
            return [Vector(data=np.array(emb, dtype=np.float64)) for emb in embeddings_array]
        except Exception as e:
            logger.error(f'Sentence Transformers batch encoding failed: {e}')
            return [self._encode_mock(text) for text in texts]

    def _encode_remote_api(self, text: str) -> Vector:
        """Encode using remote API"""
        try:
            response = self._call_remote_api([text])
            if response and len(response) > 0:
                vector_data = np.array(response[0], dtype=np.float64)
                return Vector(data=vector_data)
            else:
                return self._encode_mock(text)
        except Exception as e:
            logger.error(f'Remote API encoding failed: {e}')
            return self._encode_mock(text)

    def _encode_batch_remote_api(self, texts: List[str]) -> List[Vector]:
        """Batch encode using remote API"""
        try:
            response = self._call_remote_api(texts)
            if response and len(response) == len(texts):
                return [Vector(data=np.array(emb, dtype=np.float64)) for emb in response]
            else:
                return [self._encode_mock(text) for text in texts]
        except Exception as e:
            logger.error(f'Remote API batch encoding failed: {e}')
            return [self._encode_mock(text) for text in texts]

    def _call_remote_api(self, texts: List[str]) -> Optional[List[List[float]]]:
        """Call remote embeddings API"""
        try:
            payload = {
                'model': self.model,
                'input': texts
            }
            
            response = self.client.post(f'{self.base_url}/embeddings', json=payload)
            response.raise_for_status()
            
            result = response.json()
            if 'data' in result:
                embeddings = []
                for item in result['data']:
                    if 'embedding' in item:
                        embeddings.append(item['embedding'])
                return embeddings
            else:
                logger.error(f"Unexpected API response structure: {result}")
                return None
                
        except Exception as e:
            logger.error(f'Remote API request failed: {e}')
            return None

    def _encode_mock(self, text: str) -> Vector:
        """Generate a deterministic mock embedding"""
        text_hash = hash(text)
        rng = np.random.default_rng(seed=abs(text_hash) % (2**32))
        
        embedding_data = rng.standard_normal(self.dimensions, dtype=np.float64)
        
        # Normalize to unit vector
        norm = np.linalg.norm(embedding_data)
        if norm > 0:
            embedding_data = embedding_data / norm
            
        return Vector(data=embedding_data)

    def clear_cache(self) -> None:
        """Clear the embedding cache"""
        self._cache.clear()
        logger.debug('Embedding cache cleared')

    def get_stats(self) -> Dict[str, Any]:
        """Get adapter statistics"""
        return {
            'encode_count': self.encode_count,
            'provider': self.provider,
            'model': self.model,
            'dimensions': self.dimensions,
            'cache_size': len(self._cache),
            'max_cache_size': self.cache_size,
            'has_api_key': hasattr(self, 'api_key') and bool(getattr(self, 'api_key', None)),
            'using_mock': hasattr(self, '_use_mock') and self._use_mock
        }