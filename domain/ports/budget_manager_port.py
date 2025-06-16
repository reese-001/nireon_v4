# domain/ports/budget_manager_port.py
"""
Port-level contract for budget / quota managers.
Only the methods bootstrap (and your code) actually need are declared.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict


class BudgetManagerPort(ABC):
    """Abstract contract that concrete budget-manager implementations satisfy."""

    # ---------- Introspection ----------
    @abstractmethod
    def all_budgets(self) -> Dict[str, float]: ...

    # ---------- Consumption / credit ----------
    @abstractmethod
    def remaining(self, key: str) -> float: ...

    @abstractmethod
    def consume(self, key: str, amount: float) -> bool: ...

    @abstractmethod
    def credit(self, key: str, amount: float) -> None: ...
