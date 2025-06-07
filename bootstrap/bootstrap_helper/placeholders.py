"""
Placeholder implementations for NIREON V4 ports during bootstrap.

This module provides basic placeholder implementations for all major NIREON ports
to ensure the system can bootstrap even when real implementations are not available.
These placeholders log their operations and provide minimal functionality.
"""

from datetime import datetime
import logging
import random
from typing import Any, Dict, List, Optional, Sequence
import numpy as np

from domain.ports.llm_port import LLMPort
from domain.ports.embedding_port import EmbeddingPort
from domain.ports.event_bus_port import EventBusPort
from domain.ports.idea_repository_port import IdeaRepositoryPort
from domain.embeddings.vector import Vector, DEFAULT_DTYPE
from domain.ideas.idea import Idea

logger = logging.getLogger(__name__)


class PlaceholderLLMPortImpl(LLMPort):
    """
    Placeholder LLM port that provides mock responses.
    
    This implementation logs all requests and returns simple mock responses
    without calling any external LLM service.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.call_count = 0
        logger.info("PlaceholderLLMPort initialized - provides mock LLM responses")
    
    async def call_llm_async(self, prompt: str, **kwargs) -> str:
        """Async LLM call that returns a mock response."""
        self.call_count += 1
        logger.debug(f'PlaceholderLLMPort: Async call_llm #{self.call_count} with prompt: {prompt[:50]}...')
        
        # Simulate processing delay based on prompt length
        import asyncio
        delay = min(0.1, len(prompt) / 1000)  # Up to 100ms delay
        await asyncio.sleep(delay)
        
        return f'Placeholder LLM response #{self.call_count} to: "{prompt[:30]}{"..." if len(prompt) > 30 else ""}"'
    
    def call_llm(self, prompt: str, **kwargs) -> str:
        """Sync LLM call that returns a mock response."""
        self.call_count += 1
        logger.debug(f'PlaceholderLLMPort: Sync call_llm #{self.call_count} with prompt: {prompt[:50]}...')
        
        return f'Placeholder LLM response #{self.call_count} to: "{prompt[:30]}{"..." if len(prompt) > 30 else ""}"'
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate method that delegates to call_llm."""
        return self.call_llm(prompt, **kwargs)
    
    async def generate_async(self, prompt: str, **kwargs) -> str:
        """Async generate method that delegates to call_llm_async."""
        return await self.call_llm_async(prompt, **kwargs)


class PlaceholderEmbeddingPortImpl(EmbeddingPort):
    """
    Placeholder embedding port that generates random vectors.
    
    This implementation creates consistent random embeddings for the same input
    to support testing and development scenarios.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.dimensions = self.config.get('dimensions', 384)  # Default embedding size
        self._cache: Dict[str, Vector] = {}
        self.encode_count = 0
        logger.info(f"PlaceholderEmbeddingPort initialized - generates {self.dimensions}D random vectors")
    
    def encode(self, text: str) -> Vector:
        """Encode text to a deterministic random vector."""
        self.encode_count += 1
        
        # Use cached result for consistency
        if text in self._cache:
            logger.debug(f"PlaceholderEmbeddingPort: Retrieved cached embedding for '{text[:50]}...'")
            return self._cache[text]
        
        logger.debug(f"PlaceholderEmbeddingPort: encode #{self.encode_count} for '{text[:50]}...'")
        
        # Create deterministic random vector based on text hash
        text_hash = hash(text)
        rng = np.random.default_rng(seed=abs(text_hash) % (2**32))
        
        # Generate random vector with appropriate dtype
        embedding_data = rng.standard_normal(self.dimensions, dtype=DEFAULT_DTYPE)
        
        # Normalize to unit vector
        norm = np.linalg.norm(embedding_data)
        if norm > 0:
            embedding_data = embedding_data / norm
        
        vector = Vector(data=embedding_data)
        self._cache[text] = vector
        
        return vector
    
    def encode_batch(self, texts: Sequence[str]) -> List[Vector]:
        """Encode multiple texts to vectors."""
        logger.debug(f'PlaceholderEmbeddingPort: encode_batch for {len(texts)} texts.')
        return [self.encode(text) for text in texts]
    
    def clear_cache(self) -> None:
        """Clear the embedding cache (useful for testing)."""
        self._cache.clear()
        logger.debug("PlaceholderEmbeddingPort: Cache cleared")


class PlaceholderEventBusImpl(EventBusPort):
    """
    Placeholder event bus that logs events and maintains subscriptions.
    
    This implementation provides a working event bus for bootstrap and testing
    scenarios without requiring external message queue infrastructure.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self._subscribers: Dict[str, List[Any]] = {}
        self._event_history: List[Dict[str, Any]] = []
        self._loggers: Dict[str, logging.Logger] = {}
        self.publish_count = 0
        logger.info("PlaceholderEventBus initialized - logs events and maintains subscriptions")
    
    def publish(self, event_type: str, payload: Any) -> None:
        """Publish an event to all subscribers."""
        self.publish_count += 1
        
        event_record = {
            'event_type': event_type,
            'payload': payload,
            'timestamp': datetime.now(),
            'sequence': self.publish_count
        }
        self._event_history.append(event_record)
        
        logger.debug(f"PlaceholderEventBus: Event #{self.publish_count} '{event_type}' published with payload: {payload}")
        
        # Notify subscribers
        subscribers = self._subscribers.get(event_type, [])
        for handler in subscribers:
            try:
                handler(payload)
            except Exception as e:
                logger.error(f"PlaceholderEventBus: Error in event handler for '{event_type}': {e}")
    
    def subscribe(self, event_type: str, handler: Any) -> None:
        """Subscribe a handler to an event type."""
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        
        self._subscribers[event_type].append(handler)
        logger.debug(f"PlaceholderEventBus: Subscribed handler to '{event_type}' (total handlers: {len(self._subscribers[event_type])})")
    
    def get_logger(self, component_id: str) -> logging.Logger:
        """Get a logger for a specific component."""
        if component_id not in self._loggers:
            self._loggers[component_id] = logging.getLogger(f'nireon.{component_id}')
        return self._loggers[component_id]
    
    def get_event_history(self) -> List[Dict[str, Any]]:
        """Get the history of published events (useful for testing)."""
        return list(self._event_history)
    
    def get_subscriber_count(self, event_type: str) -> int:
        """Get the number of subscribers for an event type."""
        return len(self._subscribers.get(event_type, []))
    
    def clear_history(self) -> None:
        """Clear event history (useful for testing)."""
        self._event_history.clear()
        logger.debug("PlaceholderEventBus: Event history cleared")


class PlaceholderIdeaRepositoryImpl(IdeaRepositoryPort):
    """
    Placeholder idea repository that stores ideas in memory.
    
    This implementation provides a working idea repository for bootstrap
    and testing scenarios without requiring persistent storage.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self._ideas: Dict[str, Idea] = {}
        self._child_relationships: Dict[str, List[str]] = {}
        self._world_facts: Dict[str, List[str]] = {}
        self.operation_count = 0
        logger.info('PlaceholderIdeaRepository initialized with in-memory store.')
    
    def save(self, idea: Idea) -> None:
        """Save an idea to the repository."""
        self.operation_count += 1
        logger.debug(f"PlaceholderIdeaRepository: save idea '{idea.idea_id}' (operation #{self.operation_count})")
        self._ideas[idea.idea_id] = idea
    
    def get_by_id(self, idea_id: str) -> Optional[Idea]:
        """Retrieve an idea by its ID."""
        self.operation_count += 1
        idea = self._ideas.get(idea_id)
        
        if idea:
            logger.debug(f"PlaceholderIdeaRepository: get_by_id '{idea_id}' -> FOUND (operation #{self.operation_count})")
        else:
            logger.debug(f"PlaceholderIdeaRepository: get_by_id '{idea_id}' -> None (operation #{self.operation_count})")
        
        return idea
    
    def get_all(self) -> List[Idea]:
        """Get all ideas in the repository."""
        self.operation_count += 1
        all_ideas = list(self._ideas.values())
        logger.debug(f'PlaceholderIdeaRepository: get_all -> {len(all_ideas)} ideas (operation #{self.operation_count})')
        return all_ideas
    
    def get_by_parent_id(self, parent_id: str) -> List[Idea]:
        """Get all child ideas of a parent idea."""
        self.operation_count += 1
        children_ids = self._child_relationships.get(parent_id, [])
        children = [self._ideas[cid] for cid in children_ids if cid in self._ideas]
        logger.debug(f"PlaceholderIdeaRepository: get_by_parent_id '{parent_id}' -> {len(children)} ideas (operation #{self.operation_count})")
        return children
    
    def add_child_relationship(self, parent_id: str, child_id: str) -> bool:
        """Add a parent-child relationship between ideas."""
        self.operation_count += 1
        
        if parent_id not in self._ideas or child_id not in self._ideas:
            logger.warning(f"PlaceholderIdeaRepository: add_child_relationship failed. Parent '{parent_id}' or child '{child_id}' not found.")
            return False
        
        # Add to relationships
        if parent_id not in self._child_relationships:
            self._child_relationships[parent_id] = []
        
        if child_id not in self._child_relationships[parent_id]:
            self._child_relationships[parent_id].append(child_id)
        
        # Update idea objects if they support it
        if hasattr(self._ideas[parent_id], 'children') and isinstance(self._ideas[parent_id].children, list):
            if child_id not in self._ideas[parent_id].children:
                self._ideas[parent_id].children.append(child_id)
        
        if hasattr(self._ideas[child_id], 'parent_ids') and isinstance(self._ideas[child_id].parent_ids, list):
            if parent_id not in self._ideas[child_id].parent_ids:
                self._ideas[child_id].parent_ids.append(parent_id)
        
        logger.debug(f"PlaceholderIdeaRepository: add_child_relationship parent='{parent_id}', child='{child_id}' -> True (operation #{self.operation_count})")
        return True
    
    def add_world_fact(self, idea_id: str, fact_id: str) -> bool:
        """Add a world fact to an idea."""
        self.operation_count += 1
        
        if idea_id not in self._ideas:
            logger.warning(f"PlaceholderIdeaRepository: add_world_fact failed. Idea '{idea_id}' not found.")
            return False
        
        # Add to world facts
        if idea_id not in self._world_facts:
            self._world_facts[idea_id] = []
        
        if fact_id not in self._world_facts[idea_id]:
            self._world_facts[idea_id].append(fact_id)
        
        # Update idea object if it supports it
        if hasattr(self._ideas[idea_id], 'world_facts') and isinstance(self._ideas[idea_id].world_facts, list):
            if fact_id not in self._ideas[idea_id].world_facts:
                self._ideas[idea_id].world_facts.append(fact_id)
        
        logger.debug(f"PlaceholderIdeaRepository: add_world_fact idea='{idea_id}', fact='{fact_id}' -> True (operation #{self.operation_count})")
        return True
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get repository statistics (useful for monitoring)."""
        return {
            'total_ideas': len(self._ideas),
            'total_relationships': sum(len(children) for children in self._child_relationships.values()),
            'total_world_facts': sum(len(facts) for facts in self._world_facts.values()),
            'total_operations': self.operation_count
        }
    
    def clear(self) -> None:
        """Clear all data (useful for testing)."""
        self._ideas.clear()
        self._child_relationships.clear()
        self._world_facts.clear()
        self.operation_count = 0
        logger.debug("PlaceholderIdeaRepository: All data cleared")


# Convenience function to create all placeholder services
def create_placeholder_services(config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Create all placeholder service implementations.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Dictionary mapping service names to placeholder instances
    """
    base_config = config or {}
    
    return {
        'llm_port': PlaceholderLLMPortImpl(base_config.get('llm', {})),
        'embedding_port': PlaceholderEmbeddingPortImpl(base_config.get('embedding', {})),
        'event_bus': PlaceholderEventBusImpl(base_config.get('event_bus', {})),
        'idea_repository': PlaceholderIdeaRepositoryImpl(base_config.get('idea_repository', {}))
    }