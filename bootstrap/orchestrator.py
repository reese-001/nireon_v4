"""
V4 Bootstrap Orchestrator - Main controller for system initialization.

This orchestrator implements L0 Abiogenesis - bringing the NIREON system into existence
from configuration manifests following the V4 architecture principles.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.registry import ComponentRegistry
from application.ports.event_bus_port import EventBusPort
from signals.bootstrap_signals import SYSTEM_BOOTSTRAPPED  # Import constant, don't hard-code

from .config.config_loader import V4ConfigLoader
from .phases.abiogenesis_phase import AbiogenesisPhase
from .phases.registry_setup_phase import RegistrySetupPhase
from .phases.factory_setup_phase import FactorySetupPhase
from .phases.manifest_phase import ManifestProcessingPhase
from .phases.initialization_phase import ComponentInitializationPhase
from .phases.validation_phase import InterfaceValidationPhase
from .phases.rbac_phase import RBACSetupPhase
from .health.reporter import V4HealthReporter
from .registry.registry_manager import RegistryManager
from .signals.bootstrap_signals import BootstrapSignalEmitter

logger = logging.getLogger(__name__)

@dataclass
class BootstrapConfig:
    """V4 Bootstrap configuration following Configuration Guide hierarchy."""
    config_paths: List[Path]
    existing_registry: Optional[ComponentRegistry] = None
    existing_event_bus: Optional[EventBusPort] = None
    manifest_style: str = "auto"
    replay: bool = False
    env: Optional[str] = None
    global_app_config: Optional[Dict[str, Any]] = None
    strict_mode: bool = True

    @classmethod
    def from_params(cls, config_paths: List[str | Path], **kwargs) -> 'BootstrapConfig':
        """Create BootstrapConfig from bootstrap_nireon_system parameters."""
        return cls(
            config_paths=[Path(p) for p in config_paths],
            existing_registry=kwargs.get('existing_registry'),
            existing_event_bus=kwargs.get('existing_event_bus'),
            manifest_style=kwargs.get('manifest_style', 'auto'),
            replay=kwargs.get('replay', False),
            env=kwargs.get('env'),
            global_app_config=kwargs.get('global_app_config'),
            strict_mode=kwargs.get('strict_mode', True)
        )

@dataclass 
class BootstrapContext:
    """Shared context passed between bootstrap phases."""
    config: BootstrapConfig
    run_id: str
    registry: ComponentRegistry
    registry_manager: RegistryManager
    health_reporter: V4HealthReporter
    signal_emitter: BootstrapSignalEmitter
    global_app_config: Dict[str, Any]
    validation_data_store: Any  # BootstrapValidationData
    
    @property
    def strict_mode(self) -> bool:
        return self.global_app_config.get('bootstrap_strict_mode', True)

class BootstrapOrchestrator:
    """
    V4 Bootstrap Orchestrator - Coordinates system initialization phases.
    
    Implements L0 Abiogenesis by bringing NIREON components into existence
    through a structured phase-based approach following V4 architecture.
    """
    
    def __init__(self, config: BootstrapConfig):
        self.config = config
        self.run_id = f"bootstrap_run_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
        
    async def execute_bootstrap(self) -> 'BootstrapResult':
        """
        Execute the complete V4 bootstrap process.
        
        Returns:
            BootstrapResult with populated registry, health report, and validation data
        """
        logger.info("=== NIREON V4 System Bootstrap Starting (L0 Abiogenesis) ===")
        logger.info(f"Run ID: {self.run_id}")
        logger.info(f"Config Paths: {self.config.config_paths}")
        
        # Phase 0: Load global configuration
        config_loader = V4ConfigLoader()
        global_config = await config_loader.load_global_config(
            env=self.config.env,
            provided_config=self.config.global_app_config
        )
        
        # Initialize shared context
        context = await self._create_bootstrap_context(global_config)
        
        # Execute bootstrap phases in L0 Abiogenesis order
        phases = self._create_phases()
        
        for phase in phases:
            try:
                logger.info(f"Executing Phase: {phase.__class__.__name__}")
                result = await phase.execute(context)
                
                if not result.success and context.strict_mode:
                    raise BootstrapError(f"Phase {phase.__class__.__name__} failed in strict mode: {result.errors}")
                elif not result.success:
                    logger.warning(f"Phase {phase.__class__.__name__} failed but continuing in non-strict mode: {result.errors}")
                    
            except Exception as e:
                if context.strict_mode:
                    logger.error(f"Critical failure in {phase.__class__.__name__}: {e}")
                    raise
                logger.warning(f"Phase {phase.__class__.__name__} error (non-strict mode): {e}")
        
        # Emit system birth signal (L0 Abiogenesis complete)
        await context.signal_emitter.emit_system_bootstrapped(
            component_count=len(context.registry.list_components()),
            run_id=self.run_id
        )
        
        # Build final result
        from .result_builder import BootstrapResultBuilder
        return BootstrapResultBuilder(context).build()
    
    async def _create_bootstrap_context(self, global_config: Dict[str, Any]) -> BootstrapContext:
        """Create shared bootstrap context with V4 components."""
        registry = self.config.existing_registry or ComponentRegistry()
        registry_manager = RegistryManager(registry)
        health_reporter = V4HealthReporter(registry)
        
        # Create validation data store
        from .validation_data import BootstrapValidationData
        validation_data_store = BootstrapValidationData(global_config=global_config)
        
        # Signal emitter for bootstrap events
        event_bus = self.config.existing_event_bus
        signal_emitter = BootstrapSignalEmitter(event_bus, self.run_id)
        
        return BootstrapContext(
            config=self.config,
            run_id=self.run_id,
            registry=registry,
            registry_manager=registry_manager,
            health_reporter=health_reporter,
            signal_emitter=signal_emitter,
            global_app_config=global_config,
            validation_data_store=validation_data_store
        )
    
    def _create_phases(self) -> List['BootstrapPhase']:
        """Create ordered list of bootstrap phases for L0 Abiogenesis."""
        return [
            AbiogenesisPhase(),           # L0: System emergence
            RegistrySetupPhase(),         # Core registry setup
            FactorySetupPhase(),          # Initialize factories
            ManifestProcessingPhase(),    # Process manifests
            ComponentInitializationPhase(), # Initialize components
            InterfaceValidationPhase(),   # Validate interfaces
            RBACSetupPhase(),            # Enterprise RBAC policies
        ]

class BootstrapError(RuntimeError):
    """Bootstrap-specific error for V4 system initialization failures."""
    pass