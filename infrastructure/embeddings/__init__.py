# infrastructure/embeddings/__init__.py
# Generic embeddings module - provider-agnostic

try:
    from .embeddings import EmbeddingAdapter
    __all__ = ['EmbeddingAdapter']
except ImportError as e:
    # Graceful fallback if dependencies are missing
    import logging
    logger = logging.getLogger(__name__)
    logger.warning(f"Could not import EmbeddingAdapter: {e}")
    __all__ = []