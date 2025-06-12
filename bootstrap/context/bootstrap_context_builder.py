from __future__ import annotations
import logging
from typing import Any, Dict, Optional, TYPE_CHECKING

from pydantic import BaseModel, Field, ValidationError, validator

from bootstrap.context.bootstrap_context import BootstrapContext
from bootstrap.exceptions import BootstrapContextBuildError, BootstrapError, BootstrapValidationError
from bootstrap.health.reporter import HealthReporter
from bootstrap.registry.registry_manager import RegistryManager
from bootstrap.signals.bootstrap_signals import BootstrapSignalEmitter
from bootstrap.validation_data import BootstrapValidationData
from core.registry.component_registry import ComponentRegistry
from domain.ports.event_bus_port import EventBusPort

if TYPE_CHECKING:
    from bootstrap.config.bootstrap_config import BootstrapConfig

__all__ = ['BootstrapContextBuildConfig', 'BootstrapContextBuilder', 'BootstrapContextBuildError', 'create_bootstrap_context']

logger = logging.getLogger(__name__)

class BootstrapContextBuildConfig(BaseModel):
    run_id: str = Field(..., description='Unique identifier for this bootstrap run')
    strict_mode: bool = Field(True, description='Fail hard on validation errors')
    enable_placeholders: bool = Field(True, description='Create placeholder services when missing')
    validate_dependencies: bool = Field(True, description='Perform dependency health checks')

    class Config:
        extra = 'forbid'
        validate_assignment = True

    @validator('run_id')
    def _non_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError('run_id must be non‑empty')
        if len(v) > 200:
            raise ValueError('run_id must be <\u202f200 chars')
        return v

class BootstrapContextBuilder:
    def __init__(self, run_id: str) -> None:
        try:
            self._build_cfg = BootstrapContextBuildConfig(run_id=run_id)
        except ValidationError as exc:
            raise BootstrapValidationError(f'Invalid run_id: {exc}') from exc

        self._config: Optional['BootstrapConfig'] = None
        self._registry: Optional[ComponentRegistry] = None
        self._event_bus: Optional[EventBusPort] = None
        self._global_app_cfg: Dict[str, Any] = {}
        self._built: bool = False
        self._registry_manager: Optional[RegistryManager] = None
        self._health_reporter: Optional[HealthReporter] = None
        self._signal_emitter: Optional[BootstrapSignalEmitter] = None
        self._validation_store: Optional[BootstrapValidationData] = None
        self._interface_validator: Optional[Any] = None

        logger.debug("BootstrapContextBuilder created for run_id='%s'", self.run_id)

    @property
    def run_id(self) -> str:
        return self._build_cfg.run_id

    @property
    def is_built(self) -> bool:
        return self._built

    def with_config(self, config: 'BootstrapConfig') -> 'BootstrapContextBuilder':
        self._guard_unbuilt('with_config')
        if config is None:
            raise BootstrapContextBuildError('BootstrapConfig cannot be None')
        self._config = config
        self._build_cfg.strict_mode = config.effective_strict_mode
        logger.debug('BootstrapConfig set (strict_mode=%s)', self._build_cfg.strict_mode)
        return self

    def with_registry(self, registry: Optional[ComponentRegistry] = None) -> 'BootstrapContextBuilder':
        self._guard_unbuilt('with_registry')
        self._registry = registry or ComponentRegistry()
        logger.debug('ComponentRegistry set: %s', type(self._registry).__name__)
        return self

    def with_event_bus(self, event_bus: Optional[EventBusPort] = None) -> 'BootstrapContextBuilder':
        self._guard_unbuilt('with_event_bus')
        if event_bus is None:
            if self._build_cfg.enable_placeholders:
                logger.warning('No EventBus provided – creating placeholder.')
                # Import placeholder here to avoid circular imports
                try:
                    from bootstrap.bootstrap_helper.placeholders import PlaceholderEventBusImpl
                    self._event_bus = PlaceholderEventBusImpl()
                except ImportError:
                    logger.error('Failed to import PlaceholderEventBusImpl')
                    if not self._build_cfg.enable_placeholders:
                        raise BootstrapContextBuildError('EventBusPort required when placeholders are disabled')
                    # Create a minimal placeholder
                    self._event_bus = self._create_minimal_event_bus()
            else:
                raise BootstrapContextBuildError('EventBusPort required when placeholders are disabled')
        else:
            self._event_bus = event_bus
        logger.debug('EventBus set: %s', type(self._event_bus).__name__)
        return self

    def _create_minimal_event_bus(self) -> EventBusPort:
        """Create a minimal event bus implementation when placeholders fail to import"""
        class MinimalEventBus:
            def publish(self, event_type: str, payload: Any) -> None:
                logger.debug(f"MinimalEventBus: {event_type} - {payload}")
            
            def subscribe(self, event_type: str, handler: Any) -> None:
                logger.debug(f"MinimalEventBus: Subscribed to {event_type}")
            
            def get_logger(self, component_id: str):
                return logging.getLogger(f'nireon.{component_id}')
        
        return MinimalEventBus()

    def with_global_config(self, cfg: Optional[Dict[str, Any]] = None) -> 'BootstrapContextBuilder':
        self._guard_unbuilt('with_global_config')
        self._global_app_cfg = cfg or {}
        if 'bootstrap_strict_mode' in self._global_app_cfg:
            self._build_cfg.strict_mode = bool(self._global_app_cfg['bootstrap_strict_mode'])
        logger.debug('Global config set (%d keys, strict_mode=%s)', len(self._global_app_cfg), self._build_cfg.strict_mode)
        return self

    def with_build_options(self, *, strict_mode: Optional[bool] = None, enable_placeholders: Optional[bool] = None, validate_dependencies: Optional[bool] = None) -> 'BootstrapContextBuilder':
        self._guard_unbuilt('with_build_options')
        if strict_mode is not None:
            self._build_cfg.strict_mode = strict_mode
        if enable_placeholders is not None:
            self._build_cfg.enable_placeholders = enable_placeholders
        if validate_dependencies is not None:
            self._build_cfg.validate_dependencies = validate_dependencies
        logger.debug('Build options updated: %s', self._build_cfg.dict())
        return self

    async def build(self) -> BootstrapContext:
        if self._built:
            raise BootstrapContextBuildError('Builder already used – create a new instance')

        try:
            self._validate_required_fields()
            await self._create_support_components()
            
            if self._build_cfg.validate_dependencies:
                await self._validate_dependencies()

            context = self._make_context()
            self._validate_final_context(context)
            self._built = True

            logger.info("BootstrapContext built successfully (run_id='%s')", self.run_id)
            return context

        except (BootstrapContextBuildError, BootstrapValidationError):
            raise
        except Exception as exc:
            logger.error('Unexpected failure building context: %s', exc, exc_info=True)
            raise BootstrapContextBuildError(f'Context build failed: {exc}') from exc

    def _guard_unbuilt(self, method: str) -> None:
        if self._built:
            raise BootstrapContextBuildError(f'Cannot call {method} after build()')

    def _validate_required_fields(self) -> None:
        missing = []
        if self._config is None:
            missing.append('BootstrapConfig')
        if self._registry is None:
            missing.append('ComponentRegistry')
        if self._event_bus is None:
            missing.append('EventBusPort')

        if missing:
            raise BootstrapContextBuildError(f"Missing required components: {', '.join(missing)}")

    async def _create_support_components(self) -> None:
        try:
            self._registry_manager = RegistryManager(self._registry)
            self._health_reporter = HealthReporter(self._registry)
            self._signal_emitter = BootstrapSignalEmitter(self._event_bus, self.run_id)
            self._validation_store = BootstrapValidationData(self._global_app_cfg, self.run_id)
            logger.debug('Support components created (RegistryManager, HealthReporter, SignalEmitter, ValidationStore)')
        except Exception as exc:
            raise BootstrapContextBuildError(f'Failed to create support components: {exc}') from exc

    async def _validate_dependencies(self) -> None:
        try:
            if hasattr(self._registry, 'validate_health'):
                await self._registry.validate_health()
            
            # Skip health check for minimal event bus
            if (hasattr(self._event_bus, 'health_check') and 
                not self._event_bus.__class__.__name__ == 'MinimalEventBus'):
                await self._event_bus.health_check()
            
            logger.debug('Dependency health validation complete')
        except Exception as exc:
            if self._build_cfg.strict_mode:
                raise BootstrapContextBuildError(f'Dependency validation failed: {exc}') from exc
            logger.warning('Dependency validation failed (non‑strict mode): %s', exc)

    def _make_context(self) -> BootstrapContext:
        return BootstrapContext(
            config=self._config,
            run_id=self.run_id,
            registry=self._registry,
            registry_manager=self._registry_manager,
            health_reporter=self._health_reporter,
            signal_emitter=self._signal_emitter,
            global_app_config=self._global_app_cfg,
            validation_data_store=self._validation_store,
            interface_validator=self._interface_validator
        )

    def _validate_final_context(self, ctx: BootstrapContext) -> None:
        if ctx.run_id != self.run_id:
            raise BootstrapContextBuildError('Context run_id mismatch')
        if ctx.registry is None:
            raise BootstrapContextBuildError('Context missing registry')
        logger.debug('Final context validation passed')

    @classmethod
    async def create_default(cls, run_id: str, config: 'BootstrapConfig', global_config: Optional[Dict[str, Any]] = None) -> BootstrapContext:
        return await (cls(run_id)
                     .with_config(config)
                     .with_registry(config.existing_registry)
                     .with_event_bus(config.existing_event_bus)
                     .with_global_config(global_config)
                     .build())

async def create_bootstrap_context(
    run_id: str,
    config: 'BootstrapConfig',
    global_config: Optional[Dict[str, Any]] = None,
    registry: Optional[ComponentRegistry] = None,
    event_bus: Optional[EventBusPort] = None,
    **build_opts: Any
) -> BootstrapContext:
    try:
        builder = (BootstrapContextBuilder(run_id)
                  .with_config(config)
                  .with_registry(registry or config.existing_registry)
                  .with_event_bus(event_bus or config.existing_event_bus)
                  .with_global_config(global_config))
        
        if build_opts:
            builder = builder.with_build_options(**build_opts)
        
        return await builder.build()
    except Exception as exc:
        logger.error('create_bootstrap_context failed: %s', exc, exc_info=True)
        raise