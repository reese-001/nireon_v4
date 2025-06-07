# C:\Users\erees\Documents\development\nireon_v4\bootstrap\main.py
from __future__ import annotations
import asyncio
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union
import dataclasses
from pydantic import BaseModel, Field, ValidationError
from bootstrap.exceptions import BootstrapError, BootstrapValidationError, BootstrapTimeoutError
from bootstrap.phases.base_phase import BootstrapPhase, PhaseResult
from bootstrap.result_builder import BootstrapResult, BootstrapResultBuilder
from bootstrap.bootstrap_context_builder import create_bootstrap_context
from bootstrap.bootstrap_config import BootstrapConfig
from bootstrap.bootstrap_context import BootstrapContext
from configs.config_loader import ConfigLoader
from core.registry.component_registry import ComponentRegistry
from bootstrap.phases.abiogenesis_phase import AbiogenesisPhase
from bootstrap.phases.registry_setup_phase import RegistrySetupPhase
from bootstrap.phases.factory_setup_phase import FactorySetupPhase
from bootstrap.phases.manifest_phase import ManifestProcessingPhase
from bootstrap.phases.initialization_phase import ComponentInitializationPhase
from bootstrap.phases.validation_phase import InterfaceValidationPhase
from bootstrap.phases.rbac_phase import RBACSetupPhase
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
logger = logging.getLogger(__name__)
class BootstrapExecutionConfig(BaseModel):
    timeout_seconds: float = Field(default=300.0, ge=1.0, le=3600.0, description='Maximum time allowed for bootstrap execution')
    phase_timeout_seconds: float = Field(default=60.0, ge=1.0, le=300.0, description='Maximum time allowed per bootstrap phase')
    retry_on_failure: bool = Field(default=False, description='Whether to retry bootstrap on failure (non-strict mode only)')
    max_retries: int = Field(default=3, ge=0, le=10, description='Maximum number of retry attempts')
    enable_health_checks: bool = Field(default=True, description='Whether to perform health checks during bootstrap')
    class Config:
        extra = 'forbid'
        validate_assignment = True
class BootstrapOrchestrator:
    def __init__(self, config: BootstrapConfig, execution_config: Optional[BootstrapExecutionConfig]=None):
        self.config = config
        self.execution_config = execution_config or BootstrapExecutionConfig()
        self.run_id = self._generate_run_id()
        self.start_time: Optional[datetime] = None
        self._phases: Optional[List[BootstrapPhase]] = None
        logger.info(f'BootstrapOrchestrator initialized - run_id: {self.run_id}')
        logger.debug(f'Execution config: {self.execution_config.dict()}')
    async def execute_bootstrap(self) -> BootstrapResult:
        self.start_time = datetime.now(timezone.utc)
        logger.info('=== NIREON V4 System Bootstrap Starting (L0 Abiogenesis) ===')
        logger.info(f'Run ID: {self.run_id}')
        logger.info(f'Config Paths: {[str(p) for p in self.config.config_paths]}')
        logger.info(f"Environment: {self.config.env or 'default'}")
        try:
            bootstrap_task = self._execute_bootstrap_process()
            try:
                return await asyncio.wait_for(bootstrap_task, timeout=self.execution_config.timeout_seconds)
            except asyncio.TimeoutError:
                error_msg = f'Bootstrap timed out after {self.execution_config.timeout_seconds}s'
                logger.error(error_msg)
                raise BootstrapTimeoutError(error_msg)
        except Exception as e:
            return await self._handle_bootstrap_failure(e)
    async def _execute_bootstrap_process(self) -> BootstrapResult:
        global_config = await self._load_global_configuration()
        context = await self._create_bootstrap_context(global_config)
        logger.info(f'Effective Strict Mode: {context.strict_mode}')
        logger.info(f'Health Checks Enabled: {self.execution_config.enable_health_checks}')
        await self._signal_bootstrap_started(context)
        await self._execute_phases(context)
        if self.execution_config.enable_health_checks:
            await self._perform_final_health_checks(context)
        await self._signal_bootstrap_completion(context)
        return self._build_result(context) # Changed: direct call without await for build_result
    async def _load_global_configuration(self) -> Dict[str, Any]:
        try:
            logger.debug('Loading global configuration...')
            config_loader = ConfigLoader()
            global_config = await config_loader.load_global_config(env=self.config.env, provided_config=self.config.global_app_config)
            if not self.config.global_app_config:
                config_dict = dataclasses.asdict(self.config)
                config_dict['global_app_config'] = global_config
                self.config = BootstrapConfig(**config_dict)
            logger.info(f"Global configuration loaded for env: {self.config.env or 'default'}")
            logger.debug(f'Config keys: {list(global_config.keys())}')
            return global_config
        except Exception as e:
            logger.error(f'Failed to load global configuration: {e}')
            raise BootstrapError(f'Configuration loading failed: {e}') from e
    async def _create_bootstrap_context(self, global_config: Dict[str, Any]) -> BootstrapContext:
        try:
            logger.debug('Creating bootstrap context...')
            context = await create_bootstrap_context(run_id=self.run_id, config=self.config, global_config=global_config, strict_mode=self.config.effective_strict_mode, validate_dependencies=True)
            logger.info('Bootstrap context created successfully')
            return context
        except Exception as e:
            logger.error(f'Failed to create bootstrap context: {e}')
            raise BootstrapError(f'Context creation failed: {e}') from e
    async def _signal_bootstrap_started(self, context: BootstrapContext) -> None:
        try:
            await context.signal_emitter.emit_bootstrap_started()
            logger.debug('Bootstrap started signal emitted')
        except Exception as e:
            logger.warning(f'Failed to emit bootstrap started signal: {e}')
            if context.strict_mode:
                raise BootstrapError(f'Failed to emit bootstrap started signal: {e}') from e
    async def _execute_phases(self, context: BootstrapContext) -> None:
        phases = self._get_phases()
        total_phases = len(phases)
        logger.info(f'Executing {total_phases} bootstrap phases...')
        for i, phase in enumerate(phases, 1):
            phase_name = phase.__class__.__name__
            logger.info(f'Executing Phase {i}/{total_phases}: {phase_name}')
            try:
                phase_task = self._execute_single_phase(phase, context)
                result = await asyncio.wait_for(phase_task, timeout=self.execution_config.phase_timeout_seconds)
                if not result.success:
                    error_msg = f"Phase {phase_name} failed: {'; '.join(result.errors)}"
                    if context.strict_mode:
                        raise BootstrapError(error_msg)
                    else:
                        logger.warning(f'{error_msg} (continuing in non-strict mode)')
                else:
                    logger.info(f'✓ Phase {phase_name} completed successfully')
            except asyncio.TimeoutError:
                error_msg = f'Phase {phase_name} timed out after {self.execution_config.phase_timeout_seconds}s'
                logger.error(error_msg)
                if context.strict_mode:
                    raise BootstrapError(error_msg)
                else:
                    logger.warning(f'{error_msg} (continuing in non-strict mode)')
            except BootstrapError:
                raise
            except Exception as e:
                error_msg = f'Unexpected error in {phase_name}: {e}'
                logger.error(error_msg, exc_info=True)
                if context.strict_mode:
                    raise BootstrapError(error_msg) from e
                else:
                    logger.warning(f'{error_msg} (continuing in non-strict mode)')
    async def _execute_single_phase(self, phase: BootstrapPhase, context: BootstrapContext) -> 'PhaseResult':
        phase_name = phase.__class__.__name__
        try:
            if hasattr(phase, 'execute_with_hooks'):
                return await phase.execute_with_hooks(context)
            else:
                return await phase.execute(context)
        except Exception as e:
            logger.error(f'Phase {phase_name} execution failed: {e}', exc_info=True)
            from bootstrap.phases.base_phase import PhaseResult
            return PhaseResult(success=False, errors=[f'Phase execution failed: {e}'], phase_name=phase_name)
    async def _perform_final_health_checks(self, context: BootstrapContext) -> None:
        try:
            logger.info('Performing final health checks...')
            component_count = len(context.registry.list_components())
            if component_count == 0:
                logger.warning('No components registered during bootstrap')
            else:
                logger.info(f'Registry contains {component_count} components')
            if hasattr(context, 'health_reporter') and context.health_reporter:
                health_summary = context.health_reporter.generate_summary()
                logger.info(f'Health summary: {health_summary}')
            logger.info('Final health checks completed')
        except Exception as e:
            logger.warning(f'Health checks failed: {e}')
            if context.strict_mode:
                raise BootstrapError(f'Final health checks failed: {e}') from e
    async def _signal_bootstrap_completion(self, context: BootstrapContext) -> None:
        try:
            component_count = len(context.registry.list_components())
            duration_seconds = self._get_elapsed_seconds()
            await context.signal_emitter.emit_bootstrap_completed(component_count=component_count, duration_seconds=duration_seconds)
            logger.info(f'Bootstrap completion signaled - {component_count} components, {duration_seconds:.2f}s duration')
        except Exception as e:
            logger.warning(f'Failed to signal bootstrap completion: {e}')
            if context.strict_mode:
                raise BootstrapError(f'Failed to signal completion: {e}') from e
    
    def _build_result(self, context: BootstrapContext) -> BootstrapResult: # Changed: removed async
        try:
            logger.debug('Building bootstrap result...')
            builder = BootstrapResultBuilder(context)
            # Removed: builder.with_duration(self._get_elapsed_seconds())
            # Removed: builder.with_run_id(self.run_id)
            result = builder.build() # Changed: removed await
            if result.success:
                logger.info(f'✓ NIREON V4 Bootstrap Complete. Run ID: {result.run_id}. Components: {result.component_count}, Duration: {result.bootstrap_duration:.2f}s')
            else:
                logger.error(f'✗ NIREON V4 Bootstrap Failed. Run ID: {result.run_id}. Errors: {result.critical_failure_count}') # Using critical_failure_count for error summary
            return result
        except Exception as e:
            logger.error(f'Failed to build bootstrap result: {e}')
            raise BootstrapError(f'Result building failed: {e}') from e

    async def _handle_bootstrap_failure(self, error: Exception) -> BootstrapResult:
        logger.critical(f'Critical bootstrap failure: {error}', exc_info=True)
        effective_strict_mode = self.config.effective_strict_mode
        if effective_strict_mode:
            if isinstance(error, (BootstrapError, BootstrapTimeoutError)):
                raise
            raise BootstrapError(f'Bootstrap system failure: {error}') from error
        logger.warning('Attempting recovery in non-strict mode...')
        try:
            # _create_minimal_result is synchronous
            return self._create_minimal_result(error)
        except Exception as recovery_error:
            logger.error(f'Recovery failed: {recovery_error}')
            raise BootstrapError(f'Complete bootstrap failure: {error}') from error

    def _create_minimal_result(self, original_error: Exception) -> BootstrapResult: # Changed: removed async
        try:
            registry = self.config.existing_registry or ComponentRegistry()
            from bootstrap.validation_data import BootstrapValidationData
            validation_data = BootstrapValidationData(global_config=self.config.global_app_config or {}, run_id=self.run_id)
            # For minimal result, health_reporter will contain the error
            from bootstrap.health.reporter import HealthReporter, ComponentStatus
            health_reporter = HealthReporter(registry)
            health_reporter.add_phase_result(
                "BootstrapFailure",
                "failed",
                f"Bootstrap failed critically: {original_error}",
                errors=[str(original_error)]
            )
            health_reporter.mark_bootstrap_complete()


            return BootstrapResult(
                registry=registry,
                health_reporter=health_reporter, # Use a real health_reporter
                validation_data=validation_data,
                run_id=self.run_id,
                bootstrap_duration=self._get_elapsed_seconds(),
                global_config=self.config.global_app_config
            )
        except Exception as e:
            logger.error(f'Failed to create minimal result: {e}')
            # In this critical path, re-raise as a simple RuntimeError if BootstrapResult itself fails
            raise RuntimeError(f'Failed to create MINIMAL BootstrapResult: {e}') from e

    def _get_phases(self) -> List[BootstrapPhase]:
        if self._phases is None:
            self._phases = [AbiogenesisPhase(), RegistrySetupPhase(), FactorySetupPhase(), ManifestProcessingPhase(), ComponentInitializationPhase(), InterfaceValidationPhase(), RBACSetupPhase()]
        return self._phases
    def _generate_run_id(self) -> str:
        timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
        process_id = os.getpid()
        return f'bootstrap_run_{timestamp}_{process_id}'
    def _get_elapsed_seconds(self) -> float:
        if self.start_time is None:
            return 0.0
        return (datetime.now(timezone.utc) - self.start_time).total_seconds()
async def bootstrap_nireon_system(config_paths: Sequence[Union[str, Path]], **kwargs) -> BootstrapResult:
    try:
        config = BootstrapConfig.from_params(config_paths, **kwargs)
        execution_config_dict = {k: v for k, v in kwargs.items() if k in BootstrapExecutionConfig.__fields__}
        execution_config = None
        if execution_config_dict:
            execution_config = BootstrapExecutionConfig(**execution_config_dict)
        orchestrator = BootstrapOrchestrator(config, execution_config)
        return await orchestrator.execute_bootstrap()
    except ValidationError as e:
        raise BootstrapValidationError(f'Bootstrap configuration validation failed: {e}') from e
    except Exception as e:
        logger.error(f'Bootstrap system failure: {e}')
        raise BootstrapError(f'System bootstrap failed: {e}') from e
bootstrap = bootstrap_nireon_system
def bootstrap_sync(config_paths: Sequence[Union[str, Path]], **kwargs) -> BootstrapResult:
    logger.debug('Running V4 bootstrap in synchronous mode')
    try:
        try:
            loop = asyncio.get_running_loop()
            logger.warning('Event loop already running. Creating new loop for sync bootstrap.')
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                return new_loop.run_until_complete(bootstrap_nireon_system(config_paths, **kwargs))
            finally:
                new_loop.close()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(bootstrap_nireon_system(config_paths, **kwargs))
            finally:
                loop.close()
    except Exception as e:
        logger.error(f'Synchronous bootstrap failed: {e}')
        raise
__all__ = ['BootstrapExecutionConfig', 'BootstrapOrchestrator', 'bootstrap_nireon_system', 'bootstrap', 'bootstrap_sync']