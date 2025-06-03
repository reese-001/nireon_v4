# Path: nireon_v4/bootstrap/__init__.py
from .bootstrap import (
    bootstrap_nireon_system, 
    bootstrap, 
    bootstrap_sync,
    validate_bootstrap_config,
    smoke_test,
    smoke_test_sync,
    create_test_bootstrap_config, # Added for testing convenience
    BootstrapResult, 
    CURRENT_SCHEMA_VERSION
)
from .validation_data import BootstrapValidationData, ComponentValidationData # V4 validation data
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
from .orchestrator import BootstrapOrchestrator, BootstrapConfig, BootstrapContext 

__version__ = '4.0.0' 
__author__ = 'NIREON V4 Bootstrap Team'
__description__ = 'L0 Abiogenesis: Epistemic System Bootstrap Infrastructure for NIREON V4'

__all__ = [
    # Core bootstrap functions
    'bootstrap_nireon_system',
    'bootstrap',
    'bootstrap_sync',
    'validate_bootstrap_config',
    'smoke_test',
    'smoke_test_sync',
    'create_test_bootstrap_config',
    
    # Core data structures
    'BootstrapResult', 
    'BootstrapValidationData', 
    'ComponentValidationData',
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
    
    # Constants and Info
    'CURRENT_SCHEMA_VERSION',
    '__version__',
    '__author__',
    '__description__',
    'get_version',
    'get_schema_version', # Using V4 specific schema versioning
    'get_supported_manifest_types',
    'get_bootstrap_info',
    'system_health_check' 
]

def get_version() -> str:
    """Returns the current version of the NIREON V4 bootstrap module."""
    return __version__

def get_schema_version() -> str:
    """
    Returns the schema version this bootstrap module is compatible with.
    This should align with manifest and configuration schema versions.
    (Ref: NIREON V4 API Governance.md, NIREON V4 Configuration Guide.md)
    """
    # Example: "1.0" if manifests are at version 1.0
    # For now, using the CURRENT_SCHEMA_VERSION from bootstrap.py which is more generic
    return CURRENT_SCHEMA_VERSION 

def get_supported_manifest_types() -> list[str]:
    """
    Returns a list of manifest types supported by this bootstrap version.
    (Ref: NIREON V4 Configuration Guide.md - Manifest Structure; bootstrap_helper/utils.py - detect_manifest_type)
    """
    return ['enhanced', 'simple'] 

def get_bootstrap_info() -> dict[str, any]:
    """Provides a dictionary of key information about the bootstrap module."""
    return {
        'version': __version__,
        'schema_version_compatibility': get_schema_version(),
        'description': __description__,
        'author': __author__,
        'supported_manifest_types': get_supported_manifest_types(),
        'core_features': {
            'phase_based_orchestration': True,
            'multi_layer_config_hierarchy': True, 
            'manifest_schema_validation': True, 
            'component_self_certification': True, 
            'event_signal_driven_init': True, 
            'comprehensive_health_reporting': True, 
            'rbac_setup_phase': True, 
            'hot_reload_support': False # Aspirational, currently False
        },
        'bootstrap_phases': [ 
            'AbiogenesisPhase',
            'RegistrySetupPhase',
            'FactorySetupPhase',
            'ManifestProcessingPhase',
            'ComponentInitializationPhase',
            'InterfaceValidationPhase',
            'RBACSetupPhase'
        ],
        'public_api_functions': [
            'bootstrap_nireon_system', 'bootstrap', 'bootstrap_sync',
            'validate_bootstrap_config', 'smoke_test', 'smoke_test_sync'
        ]
    }

def system_health_check() -> dict[str, any]:
    """
    Performs a basic internal health check of the bootstrap module's structure and dependencies.
    This is a lightweight check, not a full system bootstrap.
    """
    health = {'status': 'healthy', 'issues': [], 'checks_performed': []}
    try:
        # Check critical imports
        from .orchestrator import BootstrapOrchestrator
        from .phases.base_phase import BootstrapPhase
        from .processors.manifest_processor import ManifestProcessor
        from .config.config_loader import V4ConfigLoader
        health['checks_performed'].append('critical_module_imports')

        # Check for signal definitions
        from signals.bootstrap_signals import ALL_BOOTSTRAP_SIGNALS, SYSTEM_BOOTSTRAPPED
        if not ALL_BOOTSTRAP_SIGNALS or SYSTEM_BOOTSTRAPPED not in ALL_BOOTSTRAP_SIGNALS:
            health['issues'].append('Bootstrap signal definitions seem incomplete or missing key signals.')
        health['checks_performed'].append('bootstrap_signal_definitions')
        
        # Check key helper modules
        from .bootstrap_helper import exceptions, utils, metadata, placeholders, context_builder, health_reporter as v4_hr
        if not hasattr(v4_hr, 'V4HealthReporter'): # Check specific V4 class
             health['issues'].append('V4HealthReporter missing from bootstrap_helper.health_reporter.')
        health['checks_performed'].append('bootstrap_helper_modules')
        
        if health['issues']:
            health['status'] = 'degraded'
        
        return health
    except ImportError as e:
        health['status'] = 'unhealthy'
        health['issues'].append(f'Critical import error: {e}')
        return health
    except Exception as e:
        health['status'] = 'degraded'
        health['issues'].append(f'Unexpected health check error: {e}')
        return health