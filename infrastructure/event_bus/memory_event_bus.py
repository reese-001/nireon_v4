# nireon_v4\infrastructure\event_bus\memory_event_bus.py
from __future__ import annotations

import asyncio
import inspect
import logging
import time
import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Dict, List, Sequence, Optional, Tuple

from domain.ports.event_bus_port import EventBusPort

logger = logging.getLogger(__name__)

@dataclass(slots=True)
class _RecordedEvent:
    ts: float
    signal_name: str
    payload: Any


class MemoryEventBus(EventBusPort):
    def __init__(self, component_id: str = "event_bus_memory", max_history: int = 1000, 
                 duplicate_detection_window_seconds: float = 5.0) -> None:
        self.component_id = component_id
        self._subs: Dict[str, List[Callable[[Any], Any | Coroutine]]] = defaultdict(list)
        self._max_history = max_history
        self._history: List[_RecordedEvent] = []
        
        # Duplicate detection
        self._duplicate_detection_window = duplicate_detection_window_seconds
        self._recent_publishes: Dict[Tuple[str, str], float] = {}  # (signal_name, payload_hash) -> timestamp
        self._duplicate_count = 0
        
        logger.info("[%s] constructed (max_history=%s, duplicate_window=%.1fs)", 
                   self.component_id, self._max_history, self._duplicate_detection_window)

    def subscribe(self, signal_name: str, handler: Callable[[Any], Any]) -> None:
        self._subs[signal_name].append(handler)
        logger.info(
            '[%s] SUBSCRIBED to signal "%s". Total subscribers for this signal: %d.',
            self.component_id,
            signal_name,
            len(self._subs[signal_name]),
        )

    def unsubscribe(self, signal_name: str, handler: Callable[[Any], Any]) -> None:
        try:
            self._subs[signal_name].remove(handler)
            logger.debug("[%s] unsubscribed %s → %s", self.component_id, signal_name, handler)
        except (KeyError, ValueError):
            pass

    def _compute_payload_hash(self, payload: Any) -> str:
        """Compute a hash of the payload for duplicate detection."""
        try:
            # Convert payload to a stable string representation
            if hasattr(payload, 'model_dump'):
                # Pydantic model
                payload_str = json.dumps(payload.model_dump(mode='json'), sort_keys=True)
            elif hasattr(payload, '__dict__'):
                # Regular object
                payload_str = json.dumps(payload.__dict__, sort_keys=True, default=str)
            elif isinstance(payload, dict):
                payload_str = json.dumps(payload, sort_keys=True, default=str)
            else:
                payload_str = str(payload)
            
            return hashlib.md5(payload_str.encode()).hexdigest()
        except Exception as e:
            # If we can't hash it, use object id as fallback
            logger.debug("[%s] Failed to hash payload: %s", self.component_id, e)
            return str(id(payload))

    def _is_duplicate(self, signal_name: str, payload: Any) -> bool:
        """Check if this signal/payload combination was recently published."""
        if self._duplicate_detection_window <= 0:
            return False
        
        payload_hash = self._compute_payload_hash(payload)
        cache_key = (signal_name, payload_hash)
        current_time = time.time()
        
        # Clean old entries
        cutoff_time = current_time - self._duplicate_detection_window
        self._recent_publishes = {
            k: v for k, v in self._recent_publishes.items() 
            if v > cutoff_time
        }
        
        # Check for duplicate
        if cache_key in self._recent_publishes:
            self._duplicate_count += 1
            return True
        
        # Record this publish
        self._recent_publishes[cache_key] = current_time
        return False

    def publish(self, signal_name: str, payload: Any | None = None) -> None:
        # Check for duplicates
        if self._is_duplicate(signal_name, payload):
            logger.debug(
                '[%s] DUPLICATE signal "%s" detected within %.1fs window. Total duplicates: %d. Skipping publish.',
                self.component_id,
                signal_name,
                self._duplicate_detection_window,
                self._duplicate_count
            )
            # Still record in history for analysis
            self._record(signal_name, payload)
            return
        
        self._record(signal_name, payload)
        
        handlers = tuple(self._subs.get(signal_name, ()))

        # LOGGING: Add detailed logging for publish action
        logger.info(
            '[%s] PUBLISHING signal "%s". Found %d subscriber(s).',
            self.component_id,
            signal_name,
            len(handlers),
        )

        if not handlers:
            logger.debug(f"[{self.component_id}] No subscribers for '{signal_name}', publish is a no-op.")
            return

         
        # The reconstruction logic is now handled by the subscriber if needed.
        if self._inside_running_loop():
            asyncio.create_task(self._dispatch(signal_name, payload, handlers))
        else:
            # For sync calls, create a new loop to run the dispatch
            asyncio.run(self._dispatch(signal_name, payload, handlers))

    async def _dispatch(self, signal_name: str, payload: Any, handlers: Sequence[Callable[[Any], Any | Coroutine]]) -> None:
        tasks = []
        for i, handler in enumerate(handlers):
            try:
                # LOGGING: Add log for which handler is being called
                logger.debug(f"[{self.component_id}] Dispatching '{signal_name}' to handler #{i+1} ({getattr(handler, '__qualname__', str(handler))})")
                if inspect.iscoroutinefunction(handler):
                    tasks.append(handler(payload))
                else:
                    handler(payload)
            except Exception as exc:
                logger.exception("[%s] Error in handler for signal %s: %s", self.component_id, signal_name, exc)
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def _record(self, signal_name: str, payload: Any) -> None:
        self._history.append(_RecordedEvent(time.time(), signal_name, payload))
        if len(self._history) > self._max_history:
            self._history.pop(0)

    def history(self) -> List[_RecordedEvent]:
        return list(self._history)

    def get_stats(self) -> Dict[str, Any]:
        """Get event bus statistics."""
        return {
            'subscribers': {k: len(v) for k, v in self._subs.items()},
            'history_size': len(self._history),
            'duplicate_count': self._duplicate_count,
            'recent_publish_cache_size': len(self._recent_publishes),
            'duplicate_detection_window_seconds': self._duplicate_detection_window
        }

    @staticmethod
    def _inside_running_loop() -> bool:
        try:
            asyncio.get_running_loop()
            return True
        except RuntimeError:
            return False