# nireon_v4/application/services/budget_manager.py
from typing import Protocol, Dict, Any, Optional, runtime_checkable
import logging

logger = logging.getLogger(__name__)

class BudgetExceededError(Exception):
    def __init__(self, frame_id: str, resource_key: str, requested: float, available: float):
        self.frame_id = frame_id
        self.resource_key = resource_key
        self.requested = requested
        self.available = available
        super().__init__(f"Budget for '{resource_key}' in frame '{frame_id}' exceeded. "
                         f"Requested: {requested}, Available: {available:.2f}")

@runtime_checkable
class BudgetManagerPort(Protocol):
    def initialize_frame_budget(self, frame_id: str, budgets: Dict[str, float]) -> None:
        """Initializes or resets all budgets for a given frame."""
        ...

    def try_consume_resource(self, frame_id: str, resource_key: str, amount_to_consume: float) -> bool:
        """
        Attempts to consume a specified amount of a resource for a given frame.
        Returns True if successful, False if budget would be exceeded.
        """
        ...

    def consume_resource_or_raise(self, frame_id: str, resource_key: str, amount_to_consume: float) -> None:
        """
        Attempts to consume a resource. Raises BudgetExceededError if budget is insufficient.
        """
        ...

    def get_budget_status(self, frame_id: str, resource_key: str) -> Dict[str, float]:
        """Returns {'total': X, 'consumed': Y, 'remaining': Z} for a resource."""
        ...

    def get_all_budgets_for_frame(self, frame_id: str) -> Dict[str, Dict[str, float]]:
        """Returns all budget statuses for a frame."""
        ...

class InMemoryBudgetManager(BudgetManagerPort):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        # frame_id -> resource_key -> {'total': X, 'consumed': Y}
        self._budgets: Dict[str, Dict[str, Dict[str, float]]] = {}
        self.config = config or {}
        # Default budgets if a frame's budget isn't explicitly initialized by FrameFactory
        self.default_initial_budgets = self.config.get('default_initial_budgets', {
            "llm_calls": 10.0,
            "event_publishes": 100.0,
            "embedding_calls": 50.0
        })
        logger.info(f"InMemoryBudgetManager initialized. Default initial budgets: {self.default_initial_budgets}")

    def initialize_frame_budget(self, frame_id: str, budgets: Dict[str, float]) -> None:
        if frame_id not in self._budgets:
            self._budgets[frame_id] = {}
        
        for resource_key, total_amount in budgets.items():
            self._budgets[frame_id][resource_key] = {'total': float(total_amount), 'consumed': 0.0}
            logger.info(f"Budget initialized for frame '{frame_id}', resource '{resource_key}': total={total_amount}")
        # Ensure all default budget types are present if not specified
        for res_key, default_total in self.default_initial_budgets.items():
            if res_key not in self._budgets[frame_id]:
                self._budgets[frame_id][res_key] = {'total': float(default_total), 'consumed': 0.0}
                logger.debug(f"Applied default budget for frame '{frame_id}', resource '{res_key}': total={default_total}")


    def _ensure_frame_and_resource_key_exists(self, frame_id: str, resource_key: str):
        if frame_id not in self._budgets:
            logger.debug(f"Frame '{frame_id}' not explicitly budgeted. Initializing with defaults.")
            self.initialize_frame_budget(frame_id, self.default_initial_budgets.copy())
        
        if resource_key not in self._budgets[frame_id]:
            default_total = self.default_initial_budgets.get(resource_key, float('inf')) # Default to infinite if not in defaults
            self._budgets[frame_id][resource_key] = {'total': default_total, 'consumed': 0.0}
            if default_total == float('inf'):
                 logger.warning(f"Resource '{resource_key}' for frame '{frame_id}' not in default budgets. "
                               f"Using infinite budget. This resource should ideally be part of default_initial_budgets or explicitly set.")
            logger.debug(f"Lazily initialized budget for frame '{frame_id}', resource '{resource_key}' to total: {default_total}")


    def try_consume_resource(self, frame_id: str, resource_key: str, amount_to_consume: float) -> bool:
        self._ensure_frame_and_resource_key_exists(frame_id, resource_key)
        budget_info = self._budgets[frame_id][resource_key]
        
        if budget_info['total'] == float('inf'): # Infinite budget
            budget_info['consumed'] += amount_to_consume
            logger.debug(f"Consumed {amount_to_consume} of '{resource_key}' (infinite budget) for frame '{frame_id}'. "
                        f"Total consumed: {budget_info['consumed']:.2f}.")
            return True

        if budget_info['consumed'] + amount_to_consume <= budget_info['total']:
            budget_info['consumed'] += amount_to_consume
            logger.debug(f"Consumed {amount_to_consume} of '{resource_key}' for frame '{frame_id}'. "
                        f"New consumed: {budget_info['consumed']:.2f}, Total: {budget_info['total']:.2f}.")
            return True
        else:
            logger.warning(f"Budget CHECK FAILED for '{resource_key}' in frame '{frame_id}'. "
                           f"Requested: {amount_to_consume}, Consumed: {budget_info['consumed']:.2f}, Total: {budget_info['total']:.2f}, "
                           f"Available: {budget_info['total'] - budget_info['consumed']:.2f}.")
            return False

    def consume_resource_or_raise(self, frame_id: str, resource_key: str, amount_to_consume: float) -> None:
        self._ensure_frame_and_resource_key_exists(frame_id, resource_key)
        budget_info = self._budgets[frame_id][resource_key]
        available = budget_info['total'] - budget_info['consumed'] if budget_info['total'] != float('inf') else float('inf')

        if not self.try_consume_resource(frame_id, resource_key, amount_to_consume): # try_consume handles logging
            raise BudgetExceededError(frame_id, resource_key, amount_to_consume, available)

    def get_budget_status(self, frame_id: str, resource_key: str) -> Dict[str, float]:
        self._ensure_frame_and_resource_key_exists(frame_id, resource_key)
        budget_info = self._budgets[frame_id][resource_key]
        total = budget_info['total']
        consumed = budget_info['consumed']
        remaining = total - consumed if total != float('inf') else float('inf')
        return {'total': total, 'consumed': consumed, 'remaining': remaining}

    def get_all_budgets_for_frame(self, frame_id: str) -> Dict[str, Dict[str, float]]:
        if frame_id not in self._budgets:
            # Ensure defaults are populated if frame is queried but had no budget activity
            self._ensure_frame_and_resource_key_exists(frame_id, next(iter(self.default_initial_budgets), "llm_calls"))

        return {
            res_key: self.get_budget_status(frame_id, res_key)
            for res_key in self._budgets.get(frame_id, {}).keys()
        }