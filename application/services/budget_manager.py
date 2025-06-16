"""
Nireon V4 – In-memory budget manager (thread-safe).

Implements BudgetManagerPort + NireonBaseComponent.
Suitable for development / unit tests – NOT durable storage.
"""

from __future__ import annotations

from asyncio.log import logger
import inspect
import threading
from typing import Any, Dict, Mapping, MutableMapping, Optional

from core.base_component import NireonBaseComponent
from core.lifecycle import ComponentMetadata
from core.results import ProcessResult
from domain.context import NireonExecutionContext

# ---------------------------------------------------------------------------
# Optional shim – if the real interface isn't on PYTHONPATH in this env
# ---------------------------------------------------------------------------
try:
    from domain.ports.budget_manager_port import BudgetManagerPort
except ModuleNotFoundError:  # pragma: no cover
    class BudgetManagerPort:  # type: ignore
        def all_budgets(self) -> Dict[str, float]: ...
        def remaining(self, key: str) -> float: ...
        def consume(self, key: str, amount: float) -> bool: ...
        def credit(self, key: str, amount: float) -> None: ...


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
class BudgetKeyError(KeyError):
    """The requested budget key does not exist."""


class BudgetExceededError(RuntimeError):
    """`consume()` would put the budget into the red."""


__all__ = [
    "BudgetKeyError",
    "BudgetExceededError",
    "InMemoryBudgetManager",
    "BUDGET_MANAGER_METADATA",
]

# ---------------------------------------------------------------------------
# Component-level metadata
# ---------------------------------------------------------------------------
BUDGET_MANAGER_METADATA = ComponentMetadata(
    id="budget_manager_inmemory",
    name="InMemoryBudgetManager",
    version="1.0.3",
    category="shared_service",
    description="Thread-safe in-memory quota / budget manager.",
    requires_initialize=True,
    epistemic_tags=["policy_enforcer", "facade"],
)


# ---------------------------------------------------------------------------
# Implementation
# ---------------------------------------------------------------------------
class InMemoryBudgetManager(NireonBaseComponent, BudgetManagerPort):
    """Simple in-process budget tracker."""

    METADATA_DEFINITION = BUDGET_MANAGER_METADATA

    # ------------------------------------------------------------------ ctor
    def __init__(
        self,
        *,
        config: Optional[Mapping[str, Any]] = None,
        initial_budgets: Optional[Mapping[str, float]] = None,
        metadata_definition: Optional[ComponentMetadata] = None,
    ) -> None:
        """
        Parameters
        ----------
        config
            Passed by bootstrap (kept for parity even when unused).
        initial_budgets
            Mapping of budget_key → float quota to pre-seed the manager.
        """
        super().__init__(
            config=dict(config or {}),
            metadata_definition=metadata_definition or self.METADATA_DEFINITION,
        )

        self._budgets: MutableMapping[str, float] = dict(initial_budgets or {})
        self._lock = threading.RLock()

    def initialize_frame_budget(self, frame_id: str, budget: Dict[str, float]) -> None:
        """Initializes budget entries for a specific frame."""
        with self._lock:
            for resource_key, limit in budget.items():
                frame_resource_key = f"{frame_id}:{resource_key}"
                if frame_resource_key not in self._budgets:
                    self._budgets[frame_resource_key] = float(limit)
                    logger.debug(f"Initialized budget for '{frame_resource_key}' to {limit}")

    def consume_resource_or_raise(self, frame_id: str, resource_key: str, amount: float):
        """Consumes a resource from a frame's budget, raising BudgetExceededError on failure."""
        if amount < 0:
            raise ValueError('amount must be >= 0')
        
        frame_resource_key = f"{frame_id}:{resource_key}"
        with self._lock:
            if frame_resource_key not in self._budgets:
                # Fallback to a global budget if one exists, otherwise raise an error
                if resource_key in self._budgets:
                    frame_resource_key = resource_key # Use the global key
                else:
                    raise BudgetKeyError(f"Budget key '{frame_resource_key}' not found and no global fallback.")

            if self._budgets[frame_resource_key] < amount:
                raise BudgetExceededError(f"Budget for '{frame_resource_key}' exhausted (remaining {self._budgets[frame_resource_key]:.2f}, requested {amount:.2f})")
            
            self._budgets[frame_resource_key] -= amount
            return True

    # ------------------------------------------------------------------ lifecycle
    async def _initialize_impl(self, context: NireonExecutionContext) -> None:
        cert_fn = getattr(self, 'self_certify', None) or getattr(self, '_self_certify', None)
        if cert_fn is None:
            return
        # We know _self_certify is async, so we can just await it.
        if len(inspect.signature(cert_fn).parameters):
            await cert_fn(context)
        else:
            await cert_fn()
    # ------------------------------------------------------------------ BudgetManagerPort API
    def all_budgets(self) -> Dict[str, float]:
        with self._lock:
            return dict(self._budgets)

    def remaining(self, key: str) -> float:
        with self._lock:
            if key not in self._budgets:
                raise BudgetKeyError(key)
            return self._budgets[key]

    def consume(self, key: str, amount: float) -> bool:
        if amount < 0:
            raise ValueError("amount must be >= 0")
        with self._lock:
            if key not in self._budgets:
                raise BudgetKeyError(key)
            if self._budgets[key] < amount:
                raise BudgetExceededError(
                    f"Budget '{key}' exhausted "
                    f"(remaining {self._budgets[key]:.2f}, requested {amount:.2f})"
                )
            self._budgets[key] -= amount
            return True

    def credit(self, key: str, amount: float) -> None:
        if amount < 0:
            raise ValueError("amount must be >= 0")
        with self._lock:
            self._budgets[key] = self._budgets.get(key, 0.0) + amount

    # ------------------------------------------------------------------ NireonBaseComponent stub
    async def _process_impl(
        self, data: Any, context: NireonExecutionContext
    ) -> ProcessResult:
        """Budget manager is not a stream processor – no-op."""
        return ProcessResult.ok("No processing performed.")