# C:\Users\erees\Documents\development\nireon_v4\bootstrap\__init__.py
from __future__ import annotations

# Exceptions are fine as relative imports within the same package
from .exceptions import (
    BootstrapError, BootstrapValidationError, BootstrapTimeoutError,
    ComponentInstantiationError, ComponentInitializationError, ComponentValidationError,
    ManifestProcessingError, ConfigurationError, DependencyResolutionError,
    FactoryError, RegistryError, RBACError, PhaseExecutionError,
    BootstrapContextBuildError
)

# Main bootstrap orchestrator and config/context classes
from .main import (
    BootstrapOrchestrator, BootstrapConfig, BootstrapContext,
    bootstrap_nireon_system, bootstrap, bootstrap_sync
)

# Context builder
from .bootstrap_context_builder import (
    BootstrapContextBuilder, create_bootstrap_context
)

# Phase executor
from .bootstrap_phase_executor import (
    BootstrapPhaseExecutor, PhaseExecutionResult, PhaseExecutionSummary,
    execute_bootstrap_phases
)

# Result builder
from .result_builder import (
    BootstrapResult, BootstrapResultBuilder,
    build_result_from_context, create_minimal_result
)

# Validation data store
from .validation_data import (
    BootstrapValidationData, ComponentValidationData
)

# Health reporting
from .health.reporter import (
    HealthReporter, ComponentStatus, ComponentHealthRecord
)

# ConfigLoader - now imported from its new location
from configs.config_loader import ConfigLoader


__version__ = '4.0.0'
__author__ = 'NIREON V4 Bootstrap Team'
__description__ = 'L0 Abiogenesis – Bootstrap Infrastructure'
CURRENT_SCHEMA_VERSION = 'V4-alpha.1.0'

__all__ = [
    # Main entry points
    'bootstrap_nireon_system',
    'bootstrap',
    'bootstrap_sync',

    # Configuration and Context
    'BootstrapConfig',
    'BootstrapContext',
    'BootstrapContextBuilder',
    'create_bootstrap_context',

    # Orchestration and Execution
    'BootstrapOrchestrator',
    'BootstrapPhaseExecutor',
    'PhaseExecutionResult',
    'PhaseExecutionSummary',
    'execute_bootstrap_phases',

    # Results and Data
    'BootstrapResult',
    'BootstrapResultBuilder',
    'build_result_from_context',
    'create_minimal_result',
    'BootstrapValidationData',
    'ComponentValidationData',

    # Health and Monitoring
    'HealthReporter',
    'ComponentStatus',
    'ComponentHealthRecord',

    # Configuration
    'ConfigLoader', # Exporting the ConfigLoader from its new location

    # Exceptions
    'BootstrapError', 'BootstrapValidationError', 'BootstrapTimeoutError',
    'ComponentInstantiationError', 'ComponentInitializationError', 'ComponentValidationError',
    'ManifestProcessingError', 'ConfigurationError', 'DependencyResolutionError',
    'FactoryError', 'RegistryError', 'RBACError', 'PhaseExecutionError',
    'BootstrapContextBuildError',

    # Versioning
    'CURRENT_SCHEMA_VERSION',
    '__version__',
    '__author__',
    '__description__',
]