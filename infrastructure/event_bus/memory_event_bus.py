from __future__ import annotations

"""
Nireon V4 – In‑memory event‑bus implementation.

• Thread‑safe, bounded‑history pub/sub for dev & tests.
• Implements EventBusPort and NireonBaseComponent contracts.
"""

import logging
import threading
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Mapping, MutableMapping, Optional

from core.base_component import NireonBaseComponent
from core.lifecycle import ComponentMetadata
from core.results import ProcessResult
from domain.context import NireonExecutionContext
from domain.ports.event_bus_port import EventBusPort

__all__ = ["MemoryEventBus"]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Metadata constant (imported by manifest & bootstrap)
# ---------------------------------------------------------------------------

MEMORY_EVENT_BUS_METADATA = ComponentMetadata(
    id="event_bus_memory",
    name="MemoryEventBus",
    version="1.0.5",
    category="shared_service",
    description="Thread‑safe in‑memory event bus (dev / unit‑test grade).",
    requires_initialize=True,  # run init phase for self‑certification
    epistemic_tags=["contextualizer", "facade"],
)


class MemoryEventBus(NireonBaseComponent, EventBusPort):
    """Concrete in‑process EventBus implementation."""

    METADATA_DEFINITION = MEMORY_EVENT_BUS_METADATA

    # ------------------------------------------------------------------ life‑cycle
    def __init__(
        self,
        config: Optional[Mapping[str, Any]] = None,
        *,
        metadata_definition: Optional[ComponentMetadata] = None,
    ) -> None:
        super().__init__(
            config=dict(config or {}),
            metadata_definition=metadata_definition or self.METADATA_DEFINITION,
        )

        self.max_history: int = max(int(self.config.get("max_history", 1000)), 1)
        self.enable_persistence: bool = bool(
            self.config.get("enable_persistence", False)
        )

        self._subscribers: MutableMapping[str, List[Callable[[Any], None]]] = {}
        self._event_history: List[Dict[str, Any]] = []
        self._sequence_counter: int = 0
        self._lock = threading.RLock()

        logger.info(
            "MemoryEventBus '%s' constructed (max_history=%d)",
            self.component_id,
            self.max_history,
        )

    # ------------------------------------------------------------------ EventBusPort
    def publish(self, event_type: str, payload: Any) -> None:  # type: ignore[override]
        with self._lock:
            self._sequence_counter += 1
            seq = self._sequence_counter
            self._event_history.append(
                {
                    "event_type": event_type,
                    "payload": payload,
                    "timestamp": datetime.now(timezone.utc),
                    "sequence": seq,
                }
            )
            if len(self._event_history) > self.max_history:
                self._event_history.pop(0)

            logger.debug("Event #%d '%s' published", seq, event_type)

            for handler in list(self._subscribers.get(event_type, [])):
                try:
                    handler(payload)
                except Exception as exc:  # noqa: BLE001
                    logger.error(
                        "Handler error for '%s' (seq=%d): %s",
                        event_type,
                        seq,
                        exc,
                        exc_info=True,
                    )

    def subscribe(self, event_type: str, handler: Callable[[Any], None]) -> None:  # type: ignore[override]
        with self._lock:
            self._subscribers.setdefault(event_type, []).append(handler)
            logger.debug(
                "Subscribed to '%s' (cnt=%d)",
                event_type,
                len(self._subscribers[event_type]),
            )

    def unsubscribe(self, event_type: str, handler: Callable[[Any], None]) -> bool:  # type: ignore[override]
        with self._lock:
            handlers = self._subscribers.get(event_type)
            if handlers and handler in handlers:
                handlers.remove(handler)
                logger.debug(
                    "Unsubscribed from '%s' (remaining=%d)",
                    event_type,
                    len(handlers),
                )
                return True
            return False

    # ------------------------------------------------------------------ helpers
    @property
    def publish_count(self) -> int:
        return self._sequence_counter

    def get_event_history(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._event_history)

    def get_stats(self) -> Dict[str, Any]:  # type: ignore[override]
        with self._lock:
            return {
                "total_events_published": self._sequence_counter,
                "events_in_history": len(self._event_history),
                "max_history": self.max_history,
                "event_types": len(self._subscribers),
            }

    # ------------------------------------------------------------------ NireonBaseComponent hooks
    async def _initialize_impl(self, context: NireonExecutionContext) -> None:  # noqa: D401
        """No‑op initialise – mark self‑certification now."""
        try:
            self.self_certify()
        except AttributeError:
            logger.debug("self_certify not available; assuming auto‑certified.")
        logger.debug("MemoryEventBus '%s' _initialize_impl completed.", self.component_id)

    async def _process_impl(
        self, data: Any, context: NireonExecutionContext
    ) -> ProcessResult:  # noqa: D401
        return ProcessResult.ok("No processing required.")
