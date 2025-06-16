# nireon_v4/domain/ports/reactor_port.py
"""
Domain-layer interface for the Reactor.

Why a separate port?
--------------------
* Keeps cross-layer dependencies flowing *outward* from `domain` → `reactor`,
  never the reverse (aligns with the other *Port* contracts).
* Lets components depend on an abstract “ReactorPort” instead of the concrete
  engine implementation.
* Future-proofs the API: if we grow methods like `get_stats()` or
  `run_cycle()` they can be added here without leaking implementation details.

The contract purposefully stays minimal: only the methods that external
code calls today are included.  Extra `*args, **kwargs` give us latitude
to extend the signatures later without breaking implementors.
"""

from __future__ import annotations

import typing as _t
from typing import Protocol, runtime_checkable

if _t.TYPE_CHECKING:  # pragma: no cover
    from bootstrap.signals import EpistemicSignal
    from reactor.protocols import ReactorRule


@runtime_checkable
class ReactorPort(Protocol):
    """
    Thin façade over the Reactor engine.

    Any concrete engine (e.g. ``reactor.engine.main.MainReactorEngine``)
    should satisfy this protocol automatically.
    """

    async def process_signal(
        self,
        signal: "EpistemicSignal",
        *args: _t.Any,
        **kwargs: _t.Any,
    ) -> None:
        """
        Dispatch an incoming signal through the rule engine.

        Extra positional / keyword arguments are accepted so that the port
        stays forward-compatible with the current_depth parameter used by
        ``MainReactorEngine.process_signal`` :contentReference[oaicite:0]{index=0}.
        """
        ...

    def add_rule(
        self,
        rule: "ReactorRule",
        *args: _t.Any,
        **kwargs: _t.Any,
    ) -> None:  # noqa: D401 (imperative)
        """Register a rule instance with the Reactor at runtime."""
        ...


# Optional convenience alias so legacy code can do:
# `from domain.ports.reactor_port import ReactorEngine`
ReactorEngine = ReactorPort  # type: ignore
