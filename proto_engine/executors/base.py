from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from domain.proto.base_schema import ProtoBlock
    from domain.context import NireonExecutionContext

class ExternalExecutor(ABC):
    """
    Abstract base class for external executors (e.g., Docker, subprocess).
    Defines the contract for running sandboxed code.
    """
    
    @abstractmethod
    async def execute(self, proto: "ProtoBlock", context: "NireonExecutionContext") -> Dict[str, Any]:
        """
        Execute a Proto block and return a dictionary with the results.
        
        The result dictionary must contain:
        - success (bool): Whether the execution completed without error.
        - result (Any): The return value from the user's code.
        - artifacts (List[str]): A list of paths to generated files.
        - execution_time_sec (float): The duration of the execution.
        - error (str, optional): An error message if success is False.
        - error_type (str, optional): The type of error.
        """
        pass

    async def cleanup(self):
        """Optional method for cleaning up executor resources."""
        pass