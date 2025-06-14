# core/base_component.py
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, List, Dict, Optional, TypeVar, Generic
from datetime import datetime, timezone
import hashlib
import logging

# Adjusted imports:
# Original: from ..context import NireonExecutionContext
# Original: from .lifecycle import ComponentMetadata, ComponentRegistryMissingError
from domain.context import NireonExecutionContext
from core.lifecycle import ComponentMetadata, ComponentRegistryMissingError
from core.results import ProcessResult as GeneralProcessResult, AnalysisResult, SystemSignal, AdaptationAction, ComponentHealth
from core.results import ProcessResult

logger = logging.getLogger(__name__) # This will now be 'core.base_component'

TInput = TypeVar('TInput')
TOutput = TypeVar('TOutput')

class ComponentLifecycle(ABC):
    @abstractmethod
    async def initialize(self, context: NireonExecutionContext) -> None:
        ...

    @abstractmethod
    async def process(self, data: Any, context: NireonExecutionContext) -> ProcessResult:
        ...

    @abstractmethod
    async def analyze(self, context: NireonExecutionContext) -> AnalysisResult:
        ...

    @abstractmethod
    async def react(self, context: NireonExecutionContext) -> List[SystemSignal]:
        ...

    @abstractmethod
    async def adapt(self, context: NireonExecutionContext) -> List[AdaptationAction]:
        ...

    @abstractmethod
    async def recover_from_error(self, error: Exception, context: NireonExecutionContext) -> bool:
        ...

    @abstractmethod
    async def shutdown(self, context: NireonExecutionContext) -> None:
        ...

    @abstractmethod
    async def health_check(self, context: NireonExecutionContext) -> ComponentHealth:
        ...

class NireonBaseComponent(ComponentLifecycle, ABC):
    def __init__(self, config: Dict[str, Any], metadata_definition: ComponentMetadata):
        if not isinstance(metadata_definition, ComponentMetadata):
            raise TypeError('metadata_definition must be a valid ComponentMetadata object')
        self._config = config or {}
        self._component_id = metadata_definition.id
        self._metadata_definition = metadata_definition
        self._initialized_properly: bool = False
        self._initialization_timestamp: Optional[datetime] = None
        self._last_process_timestamp: Optional[datetime] = None
        self._process_count: int = 0
        self._error_count: int = 0
        logger.debug(f"NireonBaseComponent '{self._component_id}' instantiated")

    @property
    def metadata(self) -> ComponentMetadata:
        return self._metadata_definition

    @property
    def config(self) -> Dict[str, Any]:
        return self._config

    @property
    def component_id(self) -> str:
        return self._component_id

    @property
    def is_initialized(self) -> bool:
        return self._initialized_properly

    @property
    def process_count(self) -> int:
        return self._process_count

    @property
    def error_count(self) -> int:
        return self._error_count

    async def initialize(self, context: NireonExecutionContext) -> None:
        component_logger = context.logger or logger
        if self._initialized_properly:
            component_logger.warning(f"Component '{self.component_id}' already initialized. Re-initializing.")
        component_logger.info(f'Initializing component: {self.component_id} (v{self.metadata.version}, category: {self.metadata.category})')

        if context.component_registry is None:
            raise ValueError(f"ComponentRegistry missing in ExecutionContext for '{self.component_id}' initialization")

        try:
            existing = context.component_registry.get(self.component_id)
            if existing is not self:
                component_logger.warning(f"Overwriting existing registration for '{self.component_id}' in registry during initialization.")
        except ComponentRegistryMissingError:
            pass # Component not yet in registry, which is fine during initialization of the component itself.

        context.component_registry.register(self, self.metadata)
        await self._self_certify(context)
        
        try:
            component_logger.debug(f"Component '{self.component_id}' starting _initialize_impl")
            await self._initialize_impl(context)
            component_logger.info(f"Component '{self.component_id}' _initialize_impl completed. Setting _initialized_properly to True.")
            self._initialized_properly = True
            self._initialization_timestamp = datetime.now(timezone.utc)
            component_logger.info(f"Component '{self.component_id}' _initialized_properly is now {self._initialized_properly}.")
            component_logger.info(f"Component '{self.component_id}' initialized successfully")
        except Exception as e:
            component_logger.error(f"Component '{self.component_id}' initialization failed: {e}", exc_info=True)
            self._initialized_properly = False
            raise

    async def _initialize_impl(self, context: NireonExecutionContext) -> None:
        # Default implementation, can be overridden by subclasses
        pass

    async def _self_certify(self, context: NireonExecutionContext) -> None:
        component_logger = context.logger or logger
        config_hash = hashlib.sha256(str(sorted(self.config.items())).encode()).hexdigest()
        certification_data = {
            'component_id': self.component_id,
            'component_name': self.metadata.name,
            'version': self.metadata.version,
            'category': self.metadata.category,
            'status': 'initializing',
            'config_hash': config_hash,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        if context.component_registry and hasattr(context.component_registry, 'register_certification'):
            context.component_registry.register_certification(self.component_id, certification_data)
            component_logger.info(f"Component '{self.component_id}' self-certification completed")
        else:
            component_logger.warning(f"ComponentRegistry for '{self.component_id}' either missing or does not support register_certification. Skipping self-certification registration.")

    async def process(self, data: Any, context: NireonExecutionContext, **kwargs) -> ProcessResult:
        component_logger = context.logger or logger
        if not self._initialized_properly:
            self._error_count += 1
            component_logger.error(f"Component '{self.component_id}' process called before initialization. is_initialized={self.is_initialized}")
            return ProcessResult(success=False, component_id=self.component_id, message='Component not initialized', error_code='NOT_INITIALIZED')
        self._process_count += 1
        self._last_process_timestamp = datetime.now(timezone.utc)
        try:
            # Pass the kwargs down to the implementation
            return await self._process_impl(data, context, **kwargs)
        except Exception as e:
            self._error_count += 1
            component_logger.error(f"Component '{self.component_id}' process error: {e}", exc_info=True)
            recovered = False
            try:
                recovered = await self.recover_from_error(e, context)
            except Exception as recovery_e:
                component_logger.error(f"Component '{self.component_id}' failed during recover_from_error: {recovery_e}", exc_info=True)

            if recovered:
                component_logger.info(f"Component '{self.component_id}' recovered from error: {e}")
                return ProcessResult(success=True, component_id=self.component_id, message=f'Recovered from error: {e}', error_code='RECOVERED_AFTER_ERROR')
            return ProcessResult(success=False, component_id=self.component_id, message=f'Processing error: {e}', error_code='PROCESS_ERROR')

    @abstractmethod
    async def _process_impl(self, data: Any, context: NireonExecutionContext) -> ProcessResult:
        ...

    async def shutdown(self, context: NireonExecutionContext) -> None:
        component_logger = context.logger or logger
        component_logger.info(f"Shutting down component: '{self.component_id}'")
        try:
            await self._shutdown_impl(context)
        except Exception as e:
            component_logger.error(f"Error during _shutdown_impl for '{self.component_id}': {e}", exc_info=True)
        self._initialized_properly = False
        component_logger.info(f"Component '{self.component_id}' shutdown complete")

    async def _shutdown_impl(self, context: NireonExecutionContext) -> None:
        # Default implementation, can be overridden by subclasses
        pass

    async def analyze(self, context: NireonExecutionContext) -> AnalysisResult:
        return AnalysisResult(success=True, component_id=self.component_id, metrics={}, confidence=0.5, message='Default analysis: No specific metrics.')

    async def react(self, context: NireonExecutionContext) -> List[SystemSignal]:
        return []

    async def adapt(self, context: NireonExecutionContext) -> List[AdaptationAction]:
        return []

    async def recover_from_error(self, error: Exception, context: NireonExecutionContext) -> bool:
        component_logger = context.logger or logger
        component_logger.warning(f"Default recover_from_error for '{self.component_id}': No recovery implemented for error: {error}")
        return False

    async def health_check(self, context: NireonExecutionContext) -> ComponentHealth:
        status = 'HEALTHY'
        messages = ['Basic health check passed.']
        if not self._initialized_properly:
            status = 'UNINITIALIZED'
            messages.append('Component is not initialized.')
        elif self._error_count > 0:
            status = 'DEGRADED'
            messages.append(f'Component has encountered {self._error_count} errors.')

        return ComponentHealth(
            component_id=self.component_id,
            status=status,
            message=' '.join(messages),
            details={'error_count': self._error_count, 'process_count': self.process_count}
        )

class TypedNireonComponent(NireonBaseComponent, Generic[TInput, TOutput]):
    @abstractmethod
    async def _process_impl(self, data: TInput, context: NireonExecutionContext) -> ProcessResult:
        ...

    async def process(self, data: TInput, context: NireonExecutionContext) -> ProcessResult:
        # This explicit override ensures type hinting from Generic works as expected
        # for users of TypedNireonComponent, while still leveraging the base class's logic.
        return await super().process(data, context)