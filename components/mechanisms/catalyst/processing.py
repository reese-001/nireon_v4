# nireon_v4\components\mechanisms\catalyst\processing.py
import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from domain.ideas.idea import Idea
from domain.embeddings.vector import Vector as DomainVector
from .types import DomainVectors
from .vector import VectorOperations

logger = logging.getLogger(__name__)

def validate_cross_domain_vectors(cross_domain_vectors: Dict[str, Any]) -> Tuple[bool, Optional[str], DomainVectors]:
    """Validates and normalizes the input cross-domain vectors."""
    if not cross_domain_vectors:
        return False, 'No cross-domain vectors provided', {}
    
    normalized_vectors: DomainVectors = {}
    for domain, vector_data in cross_domain_vectors.items():
        try:
            if isinstance(vector_data, list):
                vector_array = np.array(vector_data, dtype=np.float32)
            elif isinstance(vector_data, np.ndarray):
                vector_array = vector_data.astype(np.float32)
            else:
                return False, f"Invalid vector type for domain '{domain}'", {}

            if vector_array.ndim != 1:
                return False, f"Vector for domain '{domain}' must be 1-dimensional", {}

            normalized = VectorOperations.unit_vector(vector_array)
            if np.allclose(normalized, 0):
                logger.warning(f"Zero vector detected for domain '{domain}', skipping")
                continue
            normalized_vectors[domain] = normalized
        except Exception as e:
            return False, f"Error processing vector for domain '{domain}': {str(e)}", {}

    if not normalized_vectors:
        return False, 'No valid vectors after normalization', {}

    return True, None, normalized_vectors

def select_ideas_for_catalysis(ideas: List[Idea], application_rate: float, rng: Any, max_ideas: Optional[int] = None) -> List[Idea]:
    """Selects a subset of ideas for catalysis based on the application rate."""
    selected = []
    for idea in ideas:
        if rng.random() < application_rate:
            selected.append(idea)
            if max_ideas and len(selected) >= max_ideas:
                break
    return selected

def compute_blend_metrics(original_vector: DomainVector, blended_vector: DomainVector, domain_vector: np.ndarray, blend_strength: float) -> Dict[str, float]:
    """Computes various metrics related to the vector blending process."""
    metrics = {
        'blend_strength': blend_strength,
        'vector_distance': float(VectorOperations.calculate_semantic_distance(original_vector, blended_vector) or 0.0),
        'semantic_similarity_to_original': float(original_vector.similarity(blended_vector)),
        'semantic_similarity_to_domain': float(blended_vector.similarity(DomainVector.from_raw(domain_vector))),
        'magnitude_change': float(np.linalg.norm(blended_vector.data) / np.linalg.norm(original_vector.data))
    }
    return metrics

def should_apply_anti_constraints(semantic_distances: List[float], threshold: float, min_samples: int = 5) -> bool:
    """Determines if anti-constraints should be applied based on recent semantic distances."""
    if len(semantic_distances) < min_samples:
        return False
    
    avg_distance = sum(semantic_distances) / len(semantic_distances)
    return avg_distance < threshold

def create_catalysis_summary(ideas_processed: int, ideas_catalyzed: int, domains_used: Dict[str, int], avg_blend_strength: float, avg_semantic_distance: float) -> Dict[str, Any]:
    """Creates a summary dictionary of a catalysis operation."""
    return {
        'ideas_processed': ideas_processed,
        'ideas_catalyzed': ideas_catalyzed,
        'catalysis_rate': ideas_catalyzed / ideas_processed if ideas_processed > 0 else 0,
        'domains_used': domains_used,
        'most_used_domain': max(domains_used.items(), key=lambda x: x[1])[0] if domains_used else None,
        'avg_blend_strength': avg_blend_strength,
        'avg_semantic_distance': avg_semantic_distance,
        'diversity_score': min(1.0, avg_semantic_distance / 0.5) # Heuristic
    }