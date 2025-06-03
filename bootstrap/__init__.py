"""
NIREON V4 Bootstrap System

This package provides the complete bootstrap infrastructure for NIREON V4 systems.
It implements L0 Abiogenesis - the emergence of epistemic capability from
configuration specifications into living, reasoning components.

Key Features:
- Phase-based bootstrap orchestration (7 phases from abiogenesis to validation)
- 6-layer configuration hierarchy (Runtime → Env Vars → Manifest → Env → Default → Python)
- Schema validation with JSON Schema enforcement
- Self-certification system for component health tracking
- Signal-driven architecture with 50+ bootstrap event types
- Comprehensive health reporting and validation tracking

Public API:
    bootstrap_nireon_system() - Main async bootstrap function
    bootstrap() - Convenience wrapper
    bootstrap_sync() - Synchronous bootstrap wrapper
    BootstrapResult - Result container with health and validation data
    BootstrapValidationData - Validation tracking throughout bootstrap

Example:
    >>> from bootstrap import bootstrap_nireon_system
    >>> result = await bootstrap_nireon_system(['config/manifests/standard.yaml'])
    >>> registry = result.registry
    >>> print(f"Bootstrap success: {result.success}")
"""

from .bootstrap import (
    bootstrap_nireon_system,
    bootstrap,
    bootstrap_sync,
    validate_bootstrap_config,
    smoke_test,
    smoke_test_sync,
    BootstrapResult,
    CURRENT_SCHEMA_VERSION
)

from .validation_data import (
    BootstrapValidationData,
    ComponentValidationData
)

from .bootstrap_helper.exceptions import (
    BootstrapError,
    ComponentInstantiationError,
    ComponentInitializationError,
    ComponentValidationError,
    ManifestProcessingError,
    ConfigurationError,
    DependencyResolutionError,
    FactoryError,
    RegistryError,
    RBACError,
    PhaseExecutionError
)

from .orchestrator import (
    BootstrapOrchestrator,
    BootstrapConfig,
    BootstrapContext
)

# Version and metadata
__version__ = "4.0.0"
__author__ = "NIREON V4 Bootstrap Team"
__description__ = "L0 Abiogenesis: Epistemic System Bootstrap Infrastructure"

# Public API exports
__all__ = [
    # Main bootstrap functions
    'bootstrap_nireon_system',
    'bootstrap',
    'bootstrap_sync',
    'validate_bootstrap_config',
    'smoke_test',
    'smoke_test_sync',
    
    # Result and data classes
    'BootstrapResult',
    'BootstrapValidationData',
    'ComponentValidationData',
    
    # Configuration and orchestration
    'BootstrapOrchestrator',
    'BootstrapConfig', 
    'BootstrapContext',
    
    # Exceptions
    'BootstrapError',
    'ComponentInstantiationError',
    'ComponentInitializationError',
    'ComponentValidationError',
    'ManifestProcessingError',
    'ConfigurationError',
    'DependencyResolutionError',
    'FactoryError',
    'RegistryError',
    'RBACError',
    'PhaseExecutionError',
    
    # Constants
    'CURRENT_SCHEMA_VERSION',
    
    # Metadata
    '__version__',
    '__author__',
    '__description__'
]


def get_version() -> str:
    """Get the bootstrap system version."""
    return __version__


def get_schema_version() -> str:
    """Get the current bootstrap schema version."""
    return CURRENT_SCHEMA_VERSION


def get_supported_manifest_types() -> list[str]:
    """Get list of supported manifest types."""
    return ['enhanced', 'simple']


def get_bootstrap_info() -> dict[str, any]:
    """
    Get comprehensive information about the bootstrap system.
    
    Returns:
        Dictionary with version, capabilities, and feature information
    """
    return {
        'version': __version__,
        'schema_version': CURRENT_SCHEMA_VERSION,
        'description': __description__,
        'author': __author__,
        'manifest_types': get_supported_manifest_types(),
        'features': {
            'phase_based_orchestration': True,
            'six_layer_config_hierarchy': True,
            'schema_validation': True,
            'self_certification': True,
            'signal_driven_architecture': True,
            'health_reporting': True,
            'rbac_support': True,
            'hot_reload_ready': True
        },
        'phases': [
            'AbiogenesisPhase',
            'RegistrySetupPhase', 
            'FactorySetupPhase',
            'ManifestProcessingPhase',
            'ComponentInitializationPhase',
            'InterfaceValidationPhase',
            'RBACSetupPhase'
        ],
        'api_functions': [
            'bootstrap_nireon_system',
            'bootstrap',
            'bootstrap_sync',
            'validate_bootstrap_config',
            'smoke_test'
        ]
    }


# Bootstrap system health check
def system_health_check() -> dict[str, any]:
    """
    Perform a system health check of the bootstrap infrastructure.
    
    Returns:
        Dictionary with health status and any detected issues
    """
    health = {
        'status': 'healthy',
        'issues': [],
        'checks_performed': []
    }
    
    try:
        # Check imports
        from .orchestrator import BootstrapOrchestrator
        from .phases.base_phase import BootstrapPhase
        from .processors.manifest_processor import ManifestProcessor
        health['checks_performed'].append('import_validation')
        
        # Check signal constants
        from signals.bootstrap_signals import ALL_BOOTSTRAP_SIGNALS
        if len(ALL_BOOTSTRAP_SIGNALS) < 30:  # Should have 50+ signals
            health['issues'].append('insufficient_bootstrap_signals')
        health['checks_performed'].append('signal_validation')
        
        # Check helper modules
        from .bootstrap_helper import (
            exceptions, utils, metadata, placeholders, 
            context_builder, health_reporter
        )
        health['checks_performed'].append('helper_validation')
        
        return health
        
    except ImportError as e:
        health['status'] = 'unhealthy'
        health['issues'].append(f'import_error: {e}')
        return health
    except Exception as e:
        health['status'] = 'degraded'
        health['issues'].append(f'health_check_error: {e}')
        return health