# bootstrap.bootstrap.py
"""
Bootstrap Orchestrator V2 - Refactored for consistency and reliability.

This module provides the main orchestration for NIREON V4 bootstrap process
with improved error handling, consistent async patterns, and cleaner separation of concerns.
"""

from __future__ import annotations
import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from bootstrap.phases.base_phase import BootstrapPhase
from bootstrap.result_builder import BootstrapResult
from bootstrap.bootstrap_context_builder import BootstrapContextBuilder, create_bootstrap_context
from bootstrap.config.config_loader import ConfigLoader
from bootstrap.signals.bootstrap_signals import SYSTEM_BOOTSTRAPPED

from bootstrap.bootstrap_context import BootstrapContext
from bootstrap.bootstrap_config import BootstrapConfig


# Phase imports
from bootstrap.phases.abiogenesis_phase import AbiogenesisPhase
from bootstrap.phases.registry_setup_phase import RegistrySetupPhase
from bootstrap.phases.factory_setup_phase import FactorySetupPhase
from bootstrap.phases.manifest_phase import ManifestProcessingPhase
from bootstrap.phases.initialization_phase import ComponentInitializationPhase
from bootstrap.phases.validation_phase import InterfaceValidationPhase
from bootstrap.phases.rbac_phase import RBACSetupPhase

if TYPE_CHECKING:
    from core.registry import ComponentRegistry
    from application.ports.event_bus_port import EventBusPort

logger = logging.getLogger(__name__)





class BootstrapError(RuntimeError):
    """Exception raised when bootstrap fails."""
    pass


class BootstrapOrchestrator:
    """
    Main orchestrator for NIREON V4 bootstrap process.
    
    Manages the complete bootstrap lifecycle with proper error handling,
    consistent async patterns, and comprehensive logging.
    """
    
    def __init__(self, config: BootstrapConfig):
        self.config = config
        self.run_id = self._generate_run_id()
        self._phases: Optional[List[BootstrapPhase]] = None
        
        logger.info(f"BootstrapOrchestrator initialized - run_id: {self.run_id}")
    
    async def execute_bootstrap(self) -> BootstrapResult:
        """
        Execute the complete bootstrap process.
        
        Returns:
            BootstrapResult containing registry, health data, and metrics
            
        Raises:
            BootstrapError: If bootstrap fails in strict mode
        """
        logger.info('=== NIREON V4 System Bootstrap Starting (L0 Abiogenesis) ===')
        logger.info(f'Run ID: {self.run_id}')
        logger.info(f'Config Paths: {self.config.config_paths}')
        
        try:
            # Load configuration with proper async handling
            global_config = await self._load_global_configuration()
            
            # Create context with validation
            context = await self._create_bootstrap_context(global_config)
            
            logger.info(f'Effective Strict Mode: {context.strict_mode}')
            
            # Execute phases with error handling
            await self._execute_phases(context)
            
            # Signal completion
            await self._signal_bootstrap_completion(context)
            
            # Build final result
            return self._build_result(context)
            
        except Exception as e:
            return await self._handle_bootstrap_failure(e)
    
    async def _load_global_configuration(self) -> Dict[str, Any]:
        """Load global configuration with consistent error handling."""
        try:
            config_loader = ConfigLoader()
            global_config = await config_loader.load_global_config(
                env=self.config.env,
                provided_config=self.config.global_app_config
            )
            
            # Update config object for consistency
            if not self.config.global_app_config:
                self.config.global_app_config = global_config
            
            logger.info(f"Global configuration loaded for env: {self.config.env or 'default'}")
            return global_config
            
        except Exception as e:
            logger.error(f"Failed to load global configuration: {e}")
            raise BootstrapError(f"Configuration loading failed: {e}") from e
    
    async def _create_bootstrap_context(self, global_config: Dict[str, Any]) -> BootstrapContext:
        """Create bootstrap context with proper validation."""
        try:
            return create_bootstrap_context(
                run_id=self.run_id,
                config=self.config,
                global_config=global_config
            )
        except Exception as e:
            logger.error(f"Failed to create bootstrap context: {e}")
            raise BootstrapError(f"Context creation failed: {e}") from e
    
    async def _execute_phases(self, context: BootstrapContext) -> None:
        """Execute all bootstrap phases with proper error handling."""
        phases = self._get_phases()
        
        for i, phase in enumerate(phases, 1):
            phase_name = phase.__class__.__name__
            logger.info(f'Executing Phase {i}/{len(phases)}: {phase_name}')
            
            try:
                # Execute phase with hooks and error handling
                result = await phase.execute_with_hooks(context)
                
                if not result.success:
                    error_msg = f'Phase {phase_name} failed: {result.errors}'
                    
                    if context.strict_mode:
                        raise BootstrapError(error_msg)
                    else:
                        logger.warning(f'{error_msg} (continuing in non-strict mode)')
                        
                else:
                    logger.info(f'✓ Phase {phase_name} completed successfully')
                    
            except BootstrapError:
                # Re-raise bootstrap errors
                raise
            except Exception as e:
                error_msg = f'Unexpected error in {phase_name}: {e}'
                logger.error(error_msg, exc_info=True)
                
                if context.strict_mode:
                    raise BootstrapError(error_msg) from e
                else:
                    logger.warning(f'{error_msg} (continuing in non-strict mode)')
    
    async def _signal_bootstrap_completion(self, context: BootstrapContext) -> None:
        """Signal that bootstrap is complete."""
        try:
            component_count = len(context.registry.list_components())
            await context.signal_emitter.emit_system_bootstrapped(
                component_count=component_count,
                run_id=self.run_id
            )
            logger.info(f'Bootstrap completion signaled - {component_count} components')
        except Exception as e:
            logger.warning(f'Failed to signal bootstrap completion: {e}')
    
    def _build_result(self, context: BootstrapContext) -> BootstrapResult:
        """Build the final bootstrap result."""
        from bootstrap.result_builder import BootstrapResultBuilder
        
        builder = BootstrapResultBuilder(context)
        result = builder.build()
        
        if result.success:
            logger.info(f'✓ NIREON V4 Bootstrap Complete. Run ID: {result.run_id}. '
                       f'Components: {result.component_count}, Healthy: {result.healthy_component_count}.')
        else:
            logger.error(f'✗ NIREON V4 Bootstrap Failed. Run ID: {result.run_id}. '
                        f'Critical Failures: {result.critical_failure_count}.')
        
        return result
    
    async def _handle_bootstrap_failure(self, error: Exception) -> BootstrapResult:
        """Handle bootstrap failure with appropriate fallback."""
        logger.critical(f'Critical bootstrap failure: {error}', exc_info=True)
        
        effective_strict_mode = self.config.effective_strict_mode
        
        if effective_strict_mode:
            if isinstance(error, BootstrapError):
                raise
            raise BootstrapError(f'Bootstrap system failure: {error}') from error
        
        # Non-strict mode: create minimal result
        logger.warning('Continuing in non-strict mode despite bootstrap failure.')
        
        registry = self.config.existing_registry or ComponentRegistry()
        
        from bootstrap.result_builder import create_minimal_result
        result = create_minimal_result(registry, run_id=self.run_id)
        
        if hasattr(result, 'health_reporter') and result.health_reporter:
            result.health_reporter.add_phase_result(
                'OverallBootstrap', 
                'failed', 
                f'Critical failure: {error}', 
                errors=[str(error)]
            )
        
        return result
    
    def _get_phases(self) -> List[BootstrapPhase]:
        """Get the ordered list of bootstrap phases."""
        if self._phases is None:
            self._phases = [
                AbiogenesisPhase(),
                RegistrySetupPhase(),
                FactorySetupPhase(),
                ManifestProcessingPhase(),
                ComponentInitializationPhase(),
                InterfaceValidationPhase(),
                RBACSetupPhase()
            ]
        return self._phases
    
    def _generate_run_id(self) -> str:
        """Generate a unique run ID for this bootstrap."""
        timestamp = datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')
        return f"bootstrap_run_{timestamp}"


# Backwards compatibility - maintain existing API
async def bootstrap_nireon_system(
    config_paths: List[str | Path], 
    **kwargs
) -> BootstrapResult:
    """
    Bootstrap the NIREON system with given configuration.
    
    Args:
        config_paths: List of configuration file paths
        **kwargs: Additional configuration parameters
        
    Returns:
        BootstrapResult with registry and health information
    """
    config = BootstrapConfig.from_params(config_paths, **kwargs)
    orchestrator = BootstrapOrchestrator(config)
    return await orchestrator.execute_bootstrap()


# Alias for backwards compatibility
bootstrap = bootstrap_nireon_system


def bootstrap_sync(config_paths: List[str | Path], **kwargs) -> BootstrapResult:
    """Synchronous wrapper for bootstrap_nireon_system."""
    logger.debug('Running V4 bootstrap in synchronous mode')
    
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            logger.warning('Event loop already running. Creating new loop for sync bootstrap.')
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                result = new_loop.run_until_complete(bootstrap_nireon_system(config_paths, **kwargs))
                return result
            finally:
                new_loop.close()
                asyncio.set_event_loop(loop)
        else:
            return loop.run_until_complete(bootstrap_nireon_system(config_paths, **kwargs))
    except RuntimeError:
        # No event loop exists
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(bootstrap_nireon_system(config_paths, **kwargs))
        finally:
            loop.close()
            asyncio.set_event_loop(None)