from __future__ import annotations
from typing import Protocol, List, TYPE_CHECKING
if TYPE_CHECKING:
    from .models import RuleContext, Action
    from bootstrap.signals import EpistemicSignal

# Remove duplicate Action protocol - it's already defined in models.py

class Condition(Protocol):
    async def evaluate(self, signal: 'EpistemicSignal', context: 'RuleContext') -> bool:
        ...

class ReactorRule(Protocol):
    rule_id: str
    
    async def matches(self, signal: 'EpistemicSignal', context: 'RuleContext') -> bool:
        ...
    
    async def execute(self, signal: 'EpistemicSignal', context: 'RuleContext') -> List['Action']:
        ...