# reactor/protocols.py
from __future__ import annotations

from typing import List, Protocol, TYPE_CHECKING, runtime_checkable

if TYPE_CHECKING:
    from .models import Action, RuleContext
    from signals.base import EpistemicSignal


@runtime_checkable
class Condition(Protocol):
    async def evaluate(self, signal: "EpistemicSignal", context: "RuleContext") -> bool: ...


@runtime_checkable
class ReactorRule(Protocol):
    rule_id: str

    async def matches(
        self, signal: "EpistemicSignal", context: "RuleContext"
    ) -> bool: ...

    async def execute(
        self, signal: "EpistemicSignal", context: "RuleContext"
    ) -> List["Action"]: ...
