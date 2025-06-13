# nireon_v4/bootstrap/__init__.py
from __future__ import annotations
from __future__ import absolute_import

# Removed sys.modules monkey-patching for 'phases'
# Imports will now use fully qualified paths like 'bootstrap.phases.some_phase'

from .exceptions import *
from .core.main import BootstrapOrchestrator, bootstrap_nireon_system, bootstrap, bootstrap_sync
from .core.phase_executor import BootstrapPhaseExecutor, PhaseExecutionResult, PhaseExecutionSummary, execute_bootstrap_phases
from .context.bootstrap_context_builder import BootstrapContextBuilder, create_bootstrap_context
from .context.bootstrap_context import BootstrapContext
from .config.bootstrap_config import BootstrapConfig
from .result_builder import BootstrapResult, BootstrapResultBuilder, build_result_from_context, create_minimal_result
from .validation_data import BootstrapValidationData, ComponentValidationData
from .health.reporter import HealthReporter, ComponentStatus, ComponentHealthRecord
from configs.config_loader import ConfigLoader # Assuming this is correctly located

__version__ = '4.0.0'
__author__ = 'NIREON V4 Bootstrap Team'
__description__ = 'L0 Abiogenesis – Bootstrap Infrastructure'
CURRENT_SCHEMA_VERSION = 'V4-alpha.1.0'

__all__ = [
    'bootstrap_nireon_system', 'bootstrap', 'bootstrap_sync',
    'BootstrapConfig',
    'BootstrapContext', 'BootstrapContextBuilder', 'create_bootstrap_context',
    'BootstrapOrchestrator',
    'BootstrapPhaseExecutor', 'PhaseExecutionResult', 'PhaseExecutionSummary', 'execute_bootstrap_phases',
    'BootstrapResult', 'BootstrapResultBuilder', 'build_result_from_context', 'create_minimal_result',
    'BootstrapValidationData', 'ComponentValidationData',
    'HealthReporter', 'ComponentStatus', 'ComponentHealthRecord',
    'ConfigLoader',
    'CURRENT_SCHEMA_VERSION', '__version__', '__author__', '__description__',
    # Add exception classes if they are directly exposed and part of the public API
    # For example, if BootstrapError is meant to be caught by users:
    'BootstrapError', 'ComponentInstantiationError', 'ComponentInitializationError',
    'ComponentValidationError', 'ManifestProcessingError', 'ConfigurationError',
    'BootstrapTimeoutError', 'BootstrapValidationError', 'BootstrapContextBuildError',
    'DependencyResolutionError', 'FactoryError', 'StepCommandError', 'RegistryError',
    'RBACError', 'HealthReportingError', 'PhaseExecutionError'
]