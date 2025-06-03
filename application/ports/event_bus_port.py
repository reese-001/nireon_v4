# nireon/application/ports/event_bus_port.py

"""Event bus interface for application layer."""

from typing import Any, Callable, Protocol, runtime_checkable


@runtime_checkable
class EventBusPort(Protocol):
    """Interface for event publishing and subscription."""
    
    def publish(self, event_type: str, payload: Any) -> None:
        """
        Publish an event to subscribers.
        
        Args:
            event_type: Type of event
            payload: Event data
        """
        ...

    def subscribe(self, event_type: str, handler: Callable[[Any], None]) -> None:
        """
        Subscribe to an event type.
        
        Args:
            event_type: Type of event
            handler: Function to call when event occurs
        """
        ...