# nireon_v4/domain/ports/math_port.py
# process [math_engine, principia_agent]

from __future__ import annotations
from typing import Protocol, runtime_checkable, Any, Dict

@runtime_checkable
class MathPort(Protocol):
    """
    Defines the interface for a computational engine capable of handling
    structured mathematical or symbolic operations.
    """

    async def compute(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes a payload containing a mathematical task and returns the result.

        Args:
            payload: A dictionary describing the computation (e.g., expression, operations).

        Returns:
            A dictionary containing the results of the computation.
        """
        ...