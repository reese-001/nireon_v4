from __future__ import annotations
import abc
from typing import Any, Dict, TYPE_CHECKING
from core.base_component import NireonBaseComponent
from core.results import ProcessResult
if TYPE_CHECKING:
    from domain.context import NireonExecutionContext
    from domain.ports.event_bus_port import EventBusPort


class ProducerMechanism(NireonBaseComponent, abc.ABC):
    def __init__(self, config: Dict[str, Any], metadata_definition=None):
        super().__init__(config, metadata_definition)
        # Fix: Check if metadata is a ComponentMetadata object
        if hasattr(self, '_metadata_definition') and self._metadata_definition:
            self._metadata_definition.interaction_pattern = 'producer'
    
    @abc.abstractmethod
    async def _process_impl(self, data: Any, context: NireonExecutionContext) -> ProcessResult:
        raise NotImplementedError('ProducerMechanism subclasses must implement _process_impl')


class ProcessorMechanism(NireonBaseComponent, abc.ABC):
    def __init__(self, config: Dict[str, Any], metadata_definition=None):
        super().__init__(config, metadata_definition)
        # Fix: Set interaction_pattern on the ComponentMetadata object
        if hasattr(self, '_metadata_definition') and self._metadata_definition:
            self._metadata_definition.interaction_pattern = 'processor'
    
    @abc.abstractmethod
    async def _process_impl(self, data: Any, context: NireonExecutionContext) -> ProcessResult:
        raise NotImplementedError('ProcessorMechanism subclasses must implement _process_impl')
    
    def _validate_no_event_bus(self):
        if hasattr(self, 'event_bus') and self.event_bus is not None:
            raise RuntimeError(f'ProcessorMechanism {self.__class__.__name__} should not have event_bus dependency. Processors return data; Reactor publishes signals.')    