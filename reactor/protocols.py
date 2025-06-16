"""Structural typing contracts for rules and conditions."""

from __future__ import annotations

from typing import Protocol, List, TYPE_CHECKING, runtime_checkable

if TYPE_CHECKING:
    from .models import RuleContext, Action
    from signals.base import EpistemicSignal


@runtime_checkable
class Condition(Protocol):
    async def evaluate(self, signal: "EpistemicSignal", context: "RuleContext") -> bool:  # noqa: D401
        ...


@runtime_checkable
class ReactorRule(Protocol):
    rule_id: str

    async def matches(self, signal: "EpistemicSignal", context: "RuleContext") -> bool:
        """Return *True* if rule should run."""
        ...

    async def execute(self, signal: "EpistemicSignal", context: "RuleContext") -> List["Action"]:
        """Return list of actions to run when rule matches."""
        ...
