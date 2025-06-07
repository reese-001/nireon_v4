# infrastructure/persistence/__init__.py
# Generic persistence module - database-agnostic

try:
    from .idea_repository import IdeaRepository
    __all__ = ['IdeaRepository']
except ImportError as e:
    # Graceful fallback if dependencies are missing
    import logging
    logger = logging.getLogger(__name__)
    logger.warning(f"Could not import IdeaRepository: {e}")
    __all__ = []