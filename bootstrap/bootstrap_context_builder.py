# C:\Users\erees\Documents\development\nireon_v4\bootstrap\bootstrap_context_builder.py
from __future__ import annotations
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING, Union
from pydantic import BaseModel, Field, ValidationError, validator
from bootstrap.bootstrap_context import BootstrapContext
from bootstrap.bootstrap_helper.placeholders import PlaceholderEventBusImpl
from core.registry.component_registry import ComponentRegistry
from bootstrap.exceptions import BootstrapError, BootstrapValidationError, BootstrapContextBuildError
from bootstrap.health.reporter import HealthReporter # Corrected import path
from bootstrap.registry.registry_manager import RegistryManager
from bootstrap.signals.bootstrap_signals import BootstrapSignalEmitter
from bootstrap.validation_data import BootstrapValidationData
from domain.ports.event_bus_port import EventBusPort
if TYPE_CHECKING:
    from bootstrap.bootstrap_config import BootstrapConfig # Corrected import path

logger = logging.getLogger(__name__)
class BootstrapContextBuildConfig(BaseModel):
    run_id: str = Field(description='Unique identifier for bootstrap run')
    strict_mode: bool = Field(default=True, description='Whether to enforce strict validation')
    enable_placeholders: bool = Field(default=True, description='Whether to create placeholder implementations for missing services')
    validate_dependencies: bool = Field(default=True, description='Whether to validate component dependencies during build')
    class Config:
        extra = 'forbid'
        validate_assignment = True
    @validator('run_id')
    def validate_run_id(cls, v):
        if not v or not v.strip():
            raise ValueError('run_id must be non-empty')
        if len(v) > 200:
            raise ValueError('run_id must be less than 200 characters')
        return v.strip()
class BootstrapContextBuilder:
    def __init__(self, run_id: str):
        try:
            self._build_config = BootstrapContextBuildConfig(run_id=run_id)
        except ValidationError as e:
            raise BootstrapValidationError(f'Invalid run_id: {e}') from e
        self._config: Optional['BootstrapConfig'] = None
        self._registry: Optional[ComponentRegistry] = None
        self._event_bus: Optional[EventBusPort] = None
        self._global_app_config: Dict[str, Any] = {}
        self._built = False
        self._registry_manager: Optional[RegistryManager] = None
        self._health_reporter: Optional[HealthReporter] = None
        self._signal_emitter: Optional[BootstrapSignalEmitter] = None
        self._validation_data_store: Optional[BootstrapValidationData] = None
        self._interface_validator: Optional[Any] = None # Added for clarity, though not strictly needed for this fix
        logger.debug(f'BootstrapContextBuilder created for run_id: {self.run_id}')
    @property
    def run_id(self) -> str:
        return self._build_config.run_id
    @property
    def is_built(self) -> bool:
        return self._built
    def with_config(self, config: 'BootstrapConfig') -> 'BootstrapContextBuilder':
        self._ensure_not_built('with_config')
        if config is None:
            raise BootstrapContextBuildError('BootstrapConfig cannot be None')
        self._config = config
        self._build_config.strict_mode = config.effective_strict_mode
        logger.debug(f'Bootstrap config set: strict_mode={self._build_config.strict_mode}')
        return self
    def with_registry(self, registry: Optional[ComponentRegistry]=None) -> 'BootstrapContextBuilder':
        self._ensure_not_built('with_registry')
        if registry is None:
            registry = ComponentRegistry()
            logger.debug('Created new ComponentRegistry')
        else:
            logger.debug(f'Using existing registry: {type(registry).__name__}')
        self._registry = registry
        return self
    def with_event_bus(self, event_bus: Optional[EventBusPort]=None) -> 'BootstrapContextBuilder':
        self._ensure_not_built('with_event_bus')
        if event_bus is None:
            if self._build_config.enable_placeholders:
                logger.warning('No event bus provided - creating temporary placeholder for bootstrap safety')
                self._event_bus = PlaceholderEventBusImpl()
            else:
                raise BootstrapContextBuildError('EventBusPort is required when placeholders are disabled')
        else:
            self._event_bus = event_bus
        logger.debug(f'Event bus set: {type(self._event_bus).__name__}')
        return self
    def with_global_config(self, global_config: Optional[Dict[str, Any]]=None) -> 'BootstrapContextBuilder':
        self._ensure_not_built('with_global_config')
        self._global_app_config = global_config or {}
        if 'bootstrap_strict_mode' in self._global_app_config:
            self._build_config.strict_mode = bool(self._global_app_config['bootstrap_strict_mode'])
        logger.debug(f'Global config set with {len(self._global_app_config)} keys, strict_mode: {self._build_config.strict_mode}')
        return self
    def with_build_options(self, strict_mode: Optional[bool]=None, enable_placeholders: Optional[bool]=None, validate_dependencies: Optional[bool]=None) -> 'BootstrapContextBuilder':
        self._ensure_not_built('with_build_options')
        if strict_mode is not None:
            self._build_config.strict_mode = strict_mode
        if enable_placeholders is not None:
            self._build_config.enable_placeholders = enable_placeholders
        if validate_dependencies is not None:
            self._build_config.validate_dependencies = validate_dependencies
        logger.debug(f'Build options updated: {self._build_config.dict()}')
        return self
    async def build(self) -> 'BootstrapContext':
        if self._built:
            raise BootstrapContextBuildError('Builder can only be used once - create new builder for additional contexts')
        try:
            await self._validate_required_components()
            await self._create_dependent_components()
            if self._build_config.validate_dependencies:
                await self._validate_component_dependencies()
            context = await self._create_bootstrap_context()
            await self._validate_final_context(context)
            self._built = True
            logger.info(f'BootstrapContext built successfully for run_id: {self.run_id}')
            return context
        except Exception as e:
            logger.error(f'Failed to build BootstrapContext: {e}')
            if isinstance(e, (BootstrapContextBuildError, BootstrapValidationError)):
                raise
            raise BootstrapContextBuildError(f'Context build failed: {e}') from e
    def _ensure_not_built(self, method_name: str) -> None:
        if self._built:
            raise BootstrapContextBuildError(f'Cannot call {method_name} after build() is called')
    async def _validate_required_components(self) -> None:
        errors = []
        if self._config is None:
            errors.append('BootstrapConfig is required')
        if self._registry is None:
            errors.append('ComponentRegistry is required')
        if self._event_bus is None:
            errors.append('EventBusPort is required')
        if errors:
            error_msg = f"Cannot build BootstrapContext - missing required components: {', '.join(errors)}"
            logger.error(error_msg)
            raise BootstrapContextBuildError(error_msg)
    async def _create_dependent_components(self) -> None:
        try:
            self._registry_manager = RegistryManager(self._registry)
            logger.debug('Created RegistryManager')
            self._health_reporter = HealthReporter(self._registry)
            logger.debug('Created HealthReporter')
            self._signal_emitter = BootstrapSignalEmitter(event_bus=self._event_bus, run_id=self.run_id)
            logger.debug('Created BootstrapSignalEmitter')
            self._validation_data_store = BootstrapValidationData(global_config=self._global_app_config, run_id=self.run_id)
            logger.debug('Created BootstrapValidationData')
            # self._interface_validator could be initialized here if needed by default,
            # or left as None to be set by a later phase.
            # For now, it defaults to None in BootstrapContext.
        except Exception as e:
            raise BootstrapContextBuildError(f'Failed to create dependent components: {e}') from e
    async def _validate_component_dependencies(self) -> None:
        try:
            if hasattr(self._registry, 'validate_health'):
                await self._registry.validate_health()
            if not isinstance(self._event_bus, PlaceholderEventBusImpl) and hasattr(self._event_bus, 'health_check'):
                await self._event_bus.health_check()
            logger.debug('Component dependency validation completed')
        except Exception as e:
            if self._build_config.strict_mode:
                raise BootstrapContextBuildError(f'Component dependency validation failed: {e}') from e
            else:
                logger.warning(f'Dependency validation failed (non-strict mode): {e}')
    async def _create_bootstrap_context(self) -> 'BootstrapContext':
        # The interface_validator is typically set by the FactorySetupPhase later.
        # It's optional in BootstrapContext, so it's fine to initialize it to None here.
        return BootstrapContext(
            config=self._config,
            run_id=self.run_id,
            registry=self._registry,
            registry_manager=self._registry_manager,
            health_reporter=self._health_reporter,
            signal_emitter=self._signal_emitter,
            global_app_config=self._global_app_config,
            validation_data_store=self._validation_data_store,
            interface_validator=self._interface_validator # Pass the builder's validator, which might be None
            # Removed: strict_mode=self._build_config.strict_mode
        )
    async def _validate_final_context(self, context: 'BootstrapContext') -> None:
        if not hasattr(context, 'run_id') or context.run_id != self.run_id:
            raise BootstrapContextBuildError('Context run_id mismatch')
        if not hasattr(context, 'registry') or context.registry is None:
            raise BootstrapContextBuildError('Context missing registry')
        logger.debug('Final context validation completed')
    @classmethod
    async def create_default(cls, run_id: str, config: 'BootstrapConfig', global_config: Optional[Dict[str, Any]]=None) -> 'BootstrapContext':
        builder = cls(run_id).with_config(config).with_registry(config.existing_registry).with_event_bus(config.existing_event_bus).with_global_config(global_config)
        return await builder.build()
async def create_bootstrap_context(run_id: str, config: 'BootstrapConfig', global_config: Optional[Dict[str, Any]]=None, registry: Optional[ComponentRegistry]=None, event_bus: Optional[EventBusPort]=None, **build_options) -> 'BootstrapContext':
    try:
        builder = BootstrapContextBuilder(run_id).with_config(config).with_registry(registry or config.existing_registry).with_event_bus(event_bus or config.existing_event_bus).with_global_config(global_config)
        if build_options:
            builder = builder.with_build_options(**build_options)
        return await builder.build()
    except Exception as e:
        logger.error(f'Bootstrap context creation failed: {e}')
        raise
__all__ = ['BootstrapContextBuildConfig', 'BootstrapContextBuilder', 'BootstrapContextBuildError', 'create_bootstrap_context']