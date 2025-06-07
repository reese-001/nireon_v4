"""
Memory-based Event Bus implementation for NIREON V4
"""
import logging
from typing import Any, Callable, Dict, List
from datetime import datetime, timezone
from domain.ports.event_bus_port import EventBusPort

logger = logging.getLogger(__name__)


class MemoryEventBus(EventBusPort):
    """
    In-memory event bus implementation that stores events and handlers in memory.
    Suitable for development and testing environments.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.max_history = self.config.get('max_history', 1000)
        self.enable_persistence = self.config.get('enable_persistence', False)
        
        # Event storage
        self._subscribers: Dict[str, List[Callable[[Any], None]]] = {}
        self._event_history: List[Dict[str, Any]] = []
        self._sequence_counter = 0
        
        logger.info(f'MemoryEventBus initialized (max_history: {self.max_history})')
    
    def publish(self, event_type: str, payload: Any) -> None:
        """Publish an event to all subscribers of the event type."""
        self._sequence_counter += 1
        
        # Record the event
        event_record = {
            'event_type': event_type,
            'payload': payload,
            'timestamp': datetime.now(timezone.utc),
            'sequence': self._sequence_counter
        }
        
        # Add to history (with size limit)
        self._event_history.append(event_record)
        if len(self._event_history) > self.max_history:
            self._event_history.pop(0)
        
        logger.debug(f"Publishing event '{event_type}' (sequence: {self._sequence_counter})")
        
        # Notify subscribers
        subscribers = self._subscribers.get(event_type, [])
        for handler in subscribers:
            try:
                handler(payload)
            except Exception as e:
                logger.error(f"Error in event handler for '{event_type}': {e}", exc_info=True)
    
    def subscribe(self, event_type: str, handler: Callable[[Any], None]) -> None:
        """Subscribe a handler to an event type."""
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        
        self._subscribers[event_type].append(handler)
        logger.debug(f"Subscribed handler to '{event_type}' (total: {len(self._subscribers[event_type])})")
    
    def unsubscribe(self, event_type: str, handler: Callable[[Any], None]) -> bool:
        """Unsubscribe a handler from an event type."""
        if event_type in self._subscribers:
            try:
                self._subscribers[event_type].remove(handler)
                logger.debug(f"Unsubscribed handler from '{event_type}'")
                return True
            except ValueError:
                pass
        return False
    
    def get_event_history(self) -> List[Dict[str, Any]]:
        """Get a copy of the event history."""
        return list(self._event_history)
    
    def get_subscriber_count(self, event_type: str) -> int:
        """Get the number of subscribers for an event type."""
        return len(self._subscribers.get(event_type, []))
    
    def get_all_event_types(self) -> List[str]:
        """Get all event types that have subscribers."""
        return list(self._subscribers.keys())
    
    def clear_history(self) -> None:
        """Clear the event history."""
        self._event_history.clear()
        logger.debug('Event history cleared')
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the event bus."""
        return {
            'total_events_published': self._sequence_counter,
            'events_in_history': len(self._event_history),
            'total_event_types': len(self._subscribers),
            'total_subscribers': sum(len(handlers) for handlers in self._subscribers.values()),
            'max_history': self.max_history
        }