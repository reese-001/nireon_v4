# nireon_v4\application\services\budget_manager.py
from typing import Protocol, Dict, Any, Optional, runtime_checkable
from dataclasses import dataclass
import logging
logger = logging.getLogger(__name__)
class BudgetExceededError(Exception):
    def __init__(self, frame_id: str, resource_key: str, requested: float, available: float):
        self.frame_id = frame_id
        self.resource_key = resource_key
        self.requested = requested
        self.available = available
        super().__init__(f"Budget for '{resource_key}' in frame '{frame_id}' exceeded. Requested: {requested}, Available: {available:.2f}")
@dataclass
class BudgetEntry:
    total: float
    consumed: float = 0.0
@runtime_checkable
class BudgetManagerPort(Protocol):
    def initialize_frame_budget(self, frame_id: str, budgets: Dict[str, float]) -> None:
        ...
    def try_consume_resource(self, frame_id: str, resource_key: str, amount_to_consume: float) -> bool:
        ...
    def consume_resource_or_raise(self, frame_id: str, resource_key: str, amount_to_consume: float) -> None:
        ...
    def get_budget_status(self, frame_id: str, resource_key: str) -> Dict[str, float]:
        ...
    def get_all_budgets_for_frame(self, frame_id: str) -> Dict[str, Dict[str, float]]:
        ...
class InMemoryBudgetManager(BudgetManagerPort):
    def __init__(self, config: Optional[Dict[str, Any]]=None):
        self._budgets: Dict[str, Dict[str, BudgetEntry]] = {}
        self.config = config or {}
        self.default_initial_budgets = self.config.get('default_initial_budgets', {'llm_calls': 10.0, 'event_publishes': 100.0, 'embedding_calls': 50.0})
        logger.info(f'InMemoryBudgetManager initialized. Default initial budgets: {self.default_initial_budgets}')
    def initialize_frame_budget(self, frame_id: str, budgets: Dict[str, float]) -> None:
        new_frame_initialization = False
        if frame_id not in self._budgets:
            self._budgets[frame_id] = {}
            new_frame_initialization = True
            logger.debug(f"Initializing new budget for frame '{frame_id}'.")
        else:
            logger.debug(f"Frame '{frame_id}' already has budget entries. Applying new/default budgets if not explicitly set.")

        # Apply explicit budgets provided in the call
        for resource_key, total_amount_float in budgets.items():
            # If it's a brand new frame, or if the specific resource key wasn't set before for this frame,
            # or if we want to allow overriding (this part can be debated, current logic prefers not to overwrite loudly)
            if new_frame_initialization or resource_key not in self._budgets[frame_id]:
                self._budgets[frame_id][resource_key] = BudgetEntry(total=float(total_amount_float))
                logger.info(f"Budget for frame '{frame_id}', resource '{resource_key}' set to: total={total_amount_float}")
            else:
                # If the frame existed and this specific budget key also existed.
                # The original warning logic is preserved here for now, as overwriting could be risky without an explicit 'update'
                existing_budget = self._budgets[frame_id][resource_key]
                if existing_budget.total != float(total_amount_float): # Only warn if new value is different
                    logger.warning(
                        f"Budget for frame '{frame_id}', resource '{resource_key}' already exists "
                        f"(current: {existing_budget}). Not overwriting with explicit value {total_amount_float}. "
                        f"Use a dedicated update method if override is intended."
                    )
                else:
                     logger.debug(f"Budget for frame '{frame_id}', resource '{resource_key}' already set to {total_amount_float}. No change needed.")


        # Apply default budgets for any keys not covered by explicit budgets
        for res_key, default_total in self.default_initial_budgets.items():
            if res_key not in self._budgets[frame_id]:
                self._budgets[frame_id][res_key] = BudgetEntry(total=float(default_total))
                logger.debug(f"Applied default budget for frame '{frame_id}', resource '{res_key}': total={default_total}")
    def _ensure_frame_and_resource_key_exists(self, frame_id: str, resource_key: str):
        if frame_id not in self._budgets:
            logger.debug(f"Frame '{frame_id}' not explicitly budgeted. Initializing with defaults.")
            self.initialize_frame_budget(frame_id, self.default_initial_budgets.copy()) # Pass a copy
        if resource_key not in self._budgets[frame_id]:
            default_total = self.default_initial_budgets.get(resource_key, float('inf'))
            self._budgets[frame_id][resource_key] = BudgetEntry(total=default_total)
            if default_total == float('inf'):
                logger.warning(f"Resource '{resource_key}' for frame '{frame_id}' not in default budgets. Using infinite budget. This resource should ideally be part of default_initial_budgets or explicitly set.")
            logger.debug(f"Lazily initialized budget for frame '{frame_id}', resource '{resource_key}' to total: {default_total}")
    def try_consume_resource(self, frame_id: str, resource_key: str, amount_to_consume: float) -> bool:
        self._ensure_frame_and_resource_key_exists(frame_id, resource_key)
        budget_info = self._budgets[frame_id][resource_key]
        if budget_info.total == float('inf'):
            budget_info.consumed += amount_to_consume
            logger.debug(f"Consumed {amount_to_consume} of '{resource_key}' (infinite budget) for frame '{frame_id}'. Total consumed: {budget_info.consumed:.2f}.")
            return True
        if budget_info.consumed + amount_to_consume <= budget_info.total:
            budget_info.consumed += amount_to_consume
            logger.debug(f"Consumed {amount_to_consume} of '{resource_key}' for frame '{frame_id}'. New consumed: {budget_info.consumed:.2f}, Total: {budget_info.total:.2f}.")
            return True
        else:
            logger.warning(f"Budget CHECK FAILED for '{resource_key}' in frame '{frame_id}'. Requested: {amount_to_consume}, Consumed: {budget_info.consumed:.2f}, Total: {budget_info.total:.2f}, Available: {budget_info.total - budget_info.consumed:.2f}.")
            return False
    def consume_resource_or_raise(self, frame_id: str, resource_key: str, amount_to_consume: float) -> None:
        self._ensure_frame_and_resource_key_exists(frame_id, resource_key)
        budget_info = self._budgets[frame_id][resource_key]
        available = budget_info.total - budget_info.consumed if budget_info.total != float('inf') else float('inf')
        if not self.try_consume_resource(frame_id, resource_key, amount_to_consume):
            raise BudgetExceededError(frame_id, resource_key, amount_to_consume, available)
    def get_budget_status(self, frame_id: str, resource_key: str) -> Dict[str, float]:
        self._ensure_frame_and_resource_key_exists(frame_id, resource_key)
        budget_info = self._budgets[frame_id][resource_key]
        total = budget_info.total
        consumed = budget_info.consumed
        remaining = total - consumed if total != float('inf') else float('inf')
        return {'total': total, 'consumed': consumed, 'remaining': remaining}
    def get_all_budgets_for_frame(self, frame_id: str) -> Dict[str, Dict[str, float]]:
        if frame_id not in self._budgets:
            self._ensure_frame_and_resource_key_exists(frame_id, next(iter(self.default_initial_budgets), 'llm_calls'))
        return {res_key: self.get_budget_status(frame_id, res_key) for res_key in self._budgets.get(frame_id, {}).keys()}