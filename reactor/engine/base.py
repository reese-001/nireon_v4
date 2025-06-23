# reactor/engine/base.py
from __future__ import annotations

import abc
from typing import Protocol, runtime_checkable, TYPE_CHECKING

if TYPE_CHECKING:
    from signals.base import EpistemicSignal


@runtime_checkable
class ReactorEngine(Protocol):
    @abc.abstractmethod
    async def process_signal(self, signal: "EpistemicSignal") -> None: ...
