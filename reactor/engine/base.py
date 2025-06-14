# nireon_v4/reactor/engine/base.py
from __future__ import annotations
from typing import Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from signals import EpistemicSignal


class ReactorEngine(Protocol):
    """
    The public interface for the NIREON Reactor.
    Its job is to process signals and orchestrate component actions based on rules.
    """

    async def process_signal(self, signal: "EpistemicSignal") -> None:
        """
        The main entry point for the reactor. It takes a signal, finds
        matching rules, and executes the resulting actions.
        """
        ...