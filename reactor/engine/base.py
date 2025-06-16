"""Abstract contract for reactor engines."""
from __future__ import annotations

import abc
from typing import Protocol, runtime_checkable, TYPE_CHECKING

if TYPE_CHECKING:  # no run‑time dependency
    from signals.base import EpistemicSignal


@runtime_checkable
class ReactorEngine(Protocol):
    """Minimal contract every engine must satisfy."""

    @abc.abstractmethod
    async def process_signal(self, signal: "EpistemicSignal") -> None:
        """Process a single signal asynchronously."""
