# nireon_v4\infrastructure\event_bus\memory_event_bus.py
from __future__ import annotations

import asyncio
import inspect
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Dict, List, Sequence

from domain.ports.event_bus_port import EventBusPort

logger = logging.getLogger(__name__)

@dataclass(slots=True)
class _RecordedEvent:
    ts: float
    signal_name: str
    payload: Any


class MemoryEventBus(EventBusPort):
    def __init__(self, component_id: str = "event_bus_memory", max_history: int = 1000) -> None:
        self.component_id = component_id
        self._subs: Dict[str, List[Callable[[Any], Any | Coroutine]]] = defaultdict(list)
        self._max_history = max_history
        self._history: List[_RecordedEvent] = []
        logger.info("[%s] constructed (max_history=%s)", self.component_id, self._max_history)

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

    def publish(self, signal_name: str, payload: Any | None = None) -> None:
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

    @staticmethod
    def _inside_running_loop() -> bool:
        try:
            asyncio.get_running_loop()
            return True
        except RuntimeError:
            return False