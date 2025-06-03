# Adapted from nireon_staging/nireon/application/components/base.py
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, List, Dict, Optional, TypeVar, Generic
from datetime import datetime, timezone
import hashlib
import logging
# V4: Relative import for NireonExecutionContext
from ..context import NireonExecutionContext
# V4: Relative import for ComponentMetadata, ComponentRegistry, ComponentRegistryMissingError
from .lifecycle import ComponentMetadata, ComponentRegistry, ComponentRegistryMissingError
# V4: Relative import for results
from .results import ProcessResult, AnalysisResult, SystemSignal, AdaptationAction, ComponentHealth

logger = logging.getLogger(__name__)

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
        logger.debug(f"NireonBaseComponent '{self._component_id}' instantiated (V4)")

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
        if self._initialized_properly:
            context.logger.warning(f"Component '{self.component_id}' already initialized. Re-initializing.")
        
        context.logger.info(f'Initializing component: {self.component_id} (v{self.metadata.version}, category: {self.metadata.category})')
        
        if context.component_registry is None:
            # V4: ComponentRegistry might be optional for some minimal contexts, but vital for initialize
            raise ValueError(f"ComponentRegistry missing in ExecutionContext for '{self.component_id}'")

        try:
            existing = context.component_registry.get(self.component_id)
            if existing is not self:
                context.logger.warning(f"Overwriting existing registration for '{self.component_id}'")
        except ComponentRegistryMissingError: # V4: Using specific error type
            pass # Not registered yet, which is fine.
        
        context.component_registry.register(self, self.metadata)
        await self._self_certify(context) # Assuming _self_certify is part of V4 base or will be added
        
        await self._initialize_impl(context)
        self._initialized_properly = True
        self._initialization_timestamp = datetime.now(timezone.utc)
        context.logger.info(f"Component '{self.component_id}' initialized successfully")

    async def _initialize_impl(self, context: NireonExecutionContext) -> None:
        # Default implementation, can be overridden by subclasses
        pass

    async def _self_certify(self, context: NireonExecutionContext) -> None:
        # Simplified from V3 for Phase 1, can be expanded later
        config_hash = hashlib.sha256(str(sorted(self.config.items())).encode()).hexdigest()
        certification_data = {
            'component_id': self.component_id,
            'component_name': self.metadata.name,
            'version': self.metadata.version,
            'category': self.metadata.category,
            'status': 'initializing',
            'config_hash': config_hash,
            'timestamp': datetime.now(timezone.utc).isoformat(),
        }
        if hasattr(context.component_registry, 'register_certification'): # V4: Check if method exists
            context.component_registry.register_certification(self.component_id, certification_data)
            context.logger.info(f"Component '{self.component_id}' self-certification completed")
        else:
            context.logger.warning(f"ComponentRegistry for '{self.component_id}' does not support register_certification. Skipping.")


    async def process(self, data: Any, context: NireonExecutionContext) -> ProcessResult:
        if not self._initialized_properly:
            self._error_count += 1
            return ProcessResult(success=False, component_id=self.component_id, message='Component not initialized', error_code='NOT_INITIALIZED')
        
        self._process_count += 1
        self._last_process_timestamp = datetime.now(timezone.utc)
        
        try:
            return await self._process_impl(data, context)
        except Exception as e:
            self._error_count += 1
            context.logger.error(f"Component '{self.component_id}' process error: {e}", exc_info=True)
            if await self.recover_from_error(e, context):
                context.logger.info(f"Component '{self.component_id}' recovered from error")
                return ProcessResult(success=True, component_id=self.component_id, message=f'Recovered from error: {e}', error_code='RECOVERED')
            return ProcessResult(success=False, component_id=self.component_id, message=f'Processing error: {e}', error_code='PROCESS_ERROR')

    @abstractmethod
    async def _process_impl(self, data: Any, context: NireonExecutionContext) -> ProcessResult:
        ...

    async def shutdown(self, context: NireonExecutionContext) -> None:
        context.logger.info(f"Shutting down component: '{self.component_id}'")
        try:
            await self._shutdown_impl(context)
        except Exception as e:
            context.logger.error(f"Shutdown error for '{self.component_id}': {e}")
        self._initialized_properly = False
        context.logger.info(f"Component '{self.component_id}' shutdown complete")

    async def _shutdown_impl(self, context: NireonExecutionContext) -> None:
        # Default implementation, can be overridden by subclasses
        pass
        
    async def analyze(self, context: NireonExecutionContext) -> AnalysisResult:
        # Default implementation, can be overridden
        return AnalysisResult(success=True, component_id=self.component_id, metrics={}, confidence=0.5)

    async def react(self, context: NireonExecutionContext) -> List[SystemSignal]:
        # Default implementation, can be overridden
        return []

    async def adapt(self, context: NireonExecutionContext) -> List[AdaptationAction]:
        # Default implementation, can be overridden
        return []

    async def recover_from_error(self, error: Exception, context: NireonExecutionContext) -> bool:
        # Default implementation, can be overridden
        context.logger.warning(f"No recovery implemented for '{self.component_id}': {error}")
        return False

    async def health_check(self, context: NireonExecutionContext) -> ComponentHealth:
        # Default implementation, can be overridden
        status = "HEALTHY" if self._initialized_properly and self._error_count == 0 else "DEGRADED"
        return ComponentHealth(component_id=self.component_id, status=status, message="Basic health check.")

class TypedNireonComponent(NireonBaseComponent, Generic[TInput, TOutput]):
    @abstractmethod
    async def _process_impl(self, data: TInput, context: NireonExecutionContext) -> ProcessResult:
        ...

    async def process(self, data: TInput, context: NireonExecutionContext) -> ProcessResult: # type: ignore[override]
        return await super().process(data, context)