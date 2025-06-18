# nireon_v4\components\mechanisms\catalyst\vector.py
import logging
import numpy as np
import random
from typing import Optional, Tuple, Dict, List
from domain.embeddings.vector import Vector

logger = logging.getLogger(__name__)

class VectorOperations:
    @staticmethod
    def unit_vector(vec: np.ndarray) -> np.ndarray:
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
        valid_keys = [k for k, v in cross_domain_vectors.items() if v is not None and isinstance(v, np.ndarray) and (v.size > 0)]
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
        if not isinstance(original_vector, Vector):
            raise TypeError('original_vector must be a nireon.domain.embeddings.vector.Vector')
        domain_vector = Vector.from_raw(domain_vector_raw)
        if original_vector.dims != domain_vector.dims:
            raise ValueError('Vectors must have the same dimensionality for blending.')
        blended_data = original_vector.data * (1.0 - blend_strength) + domain_vector.data * blend_strength
        blended_vector_unnormalised = Vector.from_raw(blended_data)
        new_normed_vector = blended_vector_unnormalised.get_normalised()
        euclidean_dist = original_vector.distance_euclidean(new_normed_vector)
        semantic_dist = 1.0 - original_vector.similarity(new_normed_vector)
        return (new_normed_vector, float(euclidean_dist), float(semantic_dist))
    @staticmethod
    def calculate_semantic_distance(vec1: Vector, vec2: Vector) -> Optional[float]:
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
    @staticmethod
    def weighted_domain_selection(cross_domain_vectors: Dict[str, np.ndarray], weights: Dict[str, float], rng: random.Random) -> Optional[str]:
        valid_domains = [k for k, v in cross_domain_vectors.items() if v is not None and isinstance(v, np.ndarray) and (v.size > 0)]
        if not valid_domains:
            return None
        domain_weights = []
        for domain in valid_domains:
            weight = weights.get(domain, 1.0)
            domain_weights.append(max(0.0, weight))
        total_weight = sum(domain_weights)
        if total_weight == 0:
            domain_weights = [1.0] * len(valid_domains)
            total_weight = len(valid_domains)
        probabilities = [w / total_weight for w in domain_weights]
        return rng.choices(valid_domains, weights=probabilities, k=1)[0]
    @staticmethod
    def multi_domain_blend(original_vector: Vector, domain_vectors: List[Tuple[str, np.ndarray]], blend_strengths: List[float]) -> Tuple[Vector, Dict[str, float]]:
        if len(domain_vectors) != len(blend_strengths):
            raise ValueError('Number of domains must match number of blend strengths')
        total_blend = sum(blend_strengths)
        if total_blend > 1.0:
            blend_strengths = [s / total_blend for s in blend_strengths]
            total_blend = 1.0
        original_weight = 1.0 - total_blend
        blended_data = original_vector.data * original_weight
        contributions = {'original': original_weight}
        for (domain_name, domain_vec_raw), strength in zip(domain_vectors, blend_strengths):
            domain_vector = Vector.from_raw(domain_vec_raw)
            if domain_vector.dims != original_vector.dims:
                logger.warning(f'Skipping domain {domain_name} - dimension mismatch')
                continue
            blended_data += domain_vector.data * strength
            contributions[domain_name] = strength
        blended_vector = Vector.from_raw(blended_data)
        normalized_vector = blended_vector.get_normalised()
        return (normalized_vector, contributions)
    @staticmethod
    def compute_interdisciplinary_score(original_vector: Vector, blended_vector: Vector, domain_vectors_used: List[np.ndarray]) -> float:
        if not domain_vectors_used:
            return 0.0
        novelty = 1.0 - original_vector.similarity(blended_vector)
        domain_similarities = []
        for domain_vec_raw in domain_vectors_used:
            domain_vec = Vector.from_raw(domain_vec_raw)
            sim = blended_vector.similarity(domain_vec)
            domain_similarities.append(sim)
        avg_domain_similarity = np.mean(domain_similarities) if domain_similarities else 0.0
        domain_sim_variance = np.var(domain_similarities) if len(domain_similarities) > 1 else 0.0
        balance_score = 1.0 - min(1.0, domain_sim_variance * 10)
        interdisciplinary_score = 0.4 * novelty + 0.4 * avg_domain_similarity + 0.2 * balance_score
        return float(np.clip(interdisciplinary_score, 0.0, 1.0))
    @staticmethod
    def find_semantic_midpoint(vectors: List[Vector]) -> Optional[Vector]:
        if not vectors:
            return None
        if len(vectors) == 1:
            return vectors[0]
        dims = vectors[0].dims
        if not all((v.dims == dims for v in vectors)):
            raise ValueError('All vectors must have the same dimensionality')
        avg_data = np.mean([v.data for v in vectors], axis=0)
        midpoint = Vector.from_raw(avg_data)
        return midpoint.get_normalised()
    @staticmethod
    def detect_semantic_clusters(vectors: Dict[str, Vector], similarity_threshold: float=0.8) -> List[List[str]]:
        if not vectors:
            return []
        names = list(vectors.keys())
        clusters = []
        assigned = set()
        for i, name1 in enumerate(names):
            if name1 in assigned:
                continue
            cluster = [name1]
            assigned.add(name1)
            for j, name2 in enumerate(names[i + 1:], i + 1):
                if name2 in assigned:
                    continue
                similarity = vectors[name1].similarity(vectors[name2])
                if similarity >= similarity_threshold:
                    cluster.append(name2)
                    assigned.add(name2)
            clusters.append(cluster)
        return clusters