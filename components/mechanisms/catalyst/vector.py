from asyncio.log import logger
import numpy as np
import random
from typing import Optional, Tuple, Dict
from domain.embeddings.vector import Vector

class VectorOperations:
    @staticmethod
    def unit_vector(vec: np.ndarray) -> np.ndarray:
        """Normalize a vector to unit length."""
        if not isinstance(vec, np.ndarray) or vec.ndim == 0:
            logger.warning('Invalid input to unit_vector. Returning zero vector.')
            return np.zeros(vec.shape if isinstance(vec, np.ndarray) else 1)
        norm = np.linalg.norm(vec)
        if norm == 0:
            logger.debug('Zero norm vector encountered in unit_vector. Returning zero vector.')
            return np.zeros(vec.shape)
        return vec / norm

    @staticmethod
    def choose_domain(cross_domain_vectors: Dict[str, np.ndarray], rng: random.Random) -> Optional[str]:
        """Choose a random domain from available cross-domain vectors."""
        valid_keys = [k for k, v in cross_domain_vectors.items() if v is not None]
        if not valid_keys:
            logger.warning('No valid cross_domain_vectors available to choose from.')
            return None
        try:
            return rng.choice(valid_keys)
        except IndexError:
            logger.error('Error choosing random domain from valid keys.')
            return None

    @staticmethod
    def blend_vectors(original_vector: Vector, domain_vector_raw: np.ndarray, blend_strength: float) -> Tuple[Vector, float, float]:
        """Blend two vectors with actual computation instead of placeholder."""
        if not isinstance(original_vector, Vector):
            raise TypeError('original_vector must be a nireon.domain.embeddings.vector.Vector')
        
        # Convert domain vector to Vector type
        domain_vector = Vector.from_raw(domain_vector_raw)
        
        if original_vector.dims != domain_vector.dims:
            raise ValueError('Vectors must have the same dimensionality for blending.')
        
        # Actual blending computation
        blended_data = original_vector.data * (1.0 - blend_strength) + domain_vector.data * blend_strength
        blended_vector_unnormalised = Vector.from_raw(blended_data)
        
        # Normalize the result
        new_normed_vector = blended_vector_unnormalised.get_normalised()
        
        # Calculate distances
        euclidean_dist = original_vector.distance_euclidean(new_normed_vector)
        semantic_dist = 1.0 - original_vector.similarity(new_normed_vector)
        
        return (new_normed_vector, float(euclidean_dist), float(semantic_dist))

    @staticmethod
    def calculate_semantic_distance(vec1: Vector, vec2: Vector) -> Optional[float]:
        """Calculate semantic distance between two vectors."""
        if vec1 is None or vec2 is None:
            return None
        if not isinstance(vec1, Vector) or not isinstance(vec2, Vector):
            logger.warning(f'Cannot calculate semantic distance between non-Vector types: {type(vec1)} and {type(vec2)}')
            return None
        if vec1.dims != vec2.dims:
            logger.warning(f'Cannot calculate semantic distance between different dimensions: {vec1.dims} and {vec2.dims}')
            return None
        try:
            return 1.0 - vec1.similarity(vec2)
        except Exception as e:
            logger.error(f'Error calculating semantic distance: {e}')
            return None