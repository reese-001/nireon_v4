"""
Memory‑backed event‑bus for NIREON V4
-------------------------------------

* Thread‑safe (single `RLock` protecting all mutating state)
* Exact same public API & behaviour as the legacy implementation
* Richer logging with sequence IDs and handler counts
* Read‑only :pyattr:`publish_count` property fixes the missing attribute that the
  old code tried to log (`self.publish_count`)
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Mapping, MutableMapping

from domain.ports.event_bus_port import EventBusPort

__all__ = ("MemoryEventBus",)

logger = logging.getLogger(__name__)


class MemoryEventBus(EventBusPort):
    """
    In‑memory event‑bus for development & unit‑testing.

    The bus keeps a bounded history of the last *N* events (default 1000) and
    a mapping of event‑type → list[handler].  All operations are O(1) or O(k)
    where *k* is the number of handlers for a single event type.

    Parameters
    ----------
    config:
        Optional mapping supporting the legacy keys

        * ``max_history`` – maximum events retained in history (int, ≥ 1)
        * ``enable_persistence`` – reserved for future use (bool)

    Notes
    -----
    * The class is **thread‑safe** via an internal :class:`threading.RLock`.
    * The constructor accepts any mapping; unexpected keys are ignored.
    """

    # --------------------------------------------------------------------- #
    # Construction / private state                                          #
    # --------------------------------------------------------------------- #

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:  # noqa: D401
        self.config: Mapping[str, Any] = config or {}
        self.max_history: int = max(int(self.config.get("max_history", 1000)), 1)
        self.enable_persistence: bool = bool(self.config.get("enable_persistence", False))

        self._subscribers: MutableMapping[str, List[Callable[[Any], None]]] = {}
        self._event_history: List[Dict[str, Any]] = []
        self._sequence_counter: int = 0
        self._lock = threading.RLock()

        logger.info("MemoryEventBus initialised (max_history=%d)", self.max_history)

    # --------------------------------------------------------------------- #
    # Public API                                                            #
    # --------------------------------------------------------------------- #

    def publish(self, event_type: str, payload: Any) -> None:  # noqa: D401
        """Publish *payload* under *event_type* to all subscribed handlers."""
        with self._lock:
            self._sequence_counter += 1
            seq = self._sequence_counter

            event_record = {
                "event_type": event_type,
                "payload": payload,
                "timestamp": datetime.now(timezone.utc),
                "sequence": seq,
            }
            self._event_history.append(event_record)
            if len(self._event_history) > self.max_history:
                self._event_history.pop(0)

            # Low‑frequency human‑oriented summaries
            if event_type == "IdeaGeneratedSignal":
                snippet = str(payload.get("text", "NO TEXT"))[:200]
                logger.info("IdeaGeneratedSignal #%d – %s…", seq, snippet)
            elif event_type == "ExplorationCompleteSignal":
                cnt = payload.get("variations_generated_count", 0)
                logger.info("ExplorationCompleteSignal #%d – %d variations", seq, cnt)

            logger.debug("Event #%d '%s' published", seq, event_type)

            for handler in list(self._subscribers.get(event_type, [])):
                try:
                    handler(payload)
                except Exception as exc:  # pragma: no cover – defensive
                    logger.error(
                        "Error in handler for '%s' (seq=%d): %s",
                        event_type,
                        seq,
                        exc,
                        exc_info=True,
                    )

    def subscribe(self, event_type: str, handler: Callable[[Any], None]) -> None:  # noqa: D401
        """Register *handler* for *event_type* notifications."""
        with self._lock:
            self._subscribers.setdefault(event_type, []).append(handler)
            logger.debug(
                "Handler subscribed to '%s' (count=%d)",
                event_type,
                len(self._subscribers[event_type]),
            )

    def unsubscribe(self, event_type: str, handler: Callable[[Any], None]) -> bool:  # noqa: D401
        """Remove *handler* from *event_type*.  Returns ``True`` if removed."""
        with self._lock:
            handlers = self._subscribers.get(event_type)
            if handlers and handler in handlers:
                handlers.remove(handler)
                logger.debug(
                    "Handler unsubscribed from '%s' (remaining=%d)",
                    event_type,
                    len(handlers),
                )
                return True
            return False

    # ------------------------------------------------------------------#
    # Introspection / utility methods                                   #
    # ------------------------------------------------------------------#

    @property
    def publish_count(self) -> int:  # noqa: D401
        """Total number of events published since start‑up (read‑only)."""
        return self._sequence_counter

    def get_event_history(self) -> List[Dict[str, Any]]:  # noqa: D401
        """Return a shallow **copy** of the internal history list."""
        with self._lock:
            return list(self._event_history)

    def get_subscriber_count(self, event_type: str) -> int:  # noqa: D401
        """Number of handlers currently subscribed to *event_type*."""
        with self._lock:
            return len(self._subscribers.get(event_type, []))

    def get_all_event_types(self) -> List[str]:  # noqa: D401
        """List of all event types that have at least one subscriber."""
        with self._lock:
            return list(self._subscribers.keys())

    def clear_history(self) -> None:  # noqa: D401
        """Erase the entire in‑memory event history."""
        with self._lock:
            self._event_history.clear()
            logger.debug("Event history cleared")

    def get_stats(self) -> Dict[str, Any]:  # noqa: D401
        """Return a snapshot of internal metrics for diagnostics."""
        with self._lock:
            return {
                "total_events_published": self._sequence_counter,
                "events_in_history": len(self._event_history),
                "total_event_types": len(self._subscribers),
                "total_subscribers": sum(len(v) for v in self._subscribers.values()),
                "max_history": self.max_history,
            }
