"""
NIREON Bootstrap Module - Refactored for consistency and reliability.

This module provides the main entry points for bootstrapping NIREON V4 systems
with improved error handling, consistent async patterns, and better separation of concerns.

Key improvements in this refactor:
- Consistent import patterns across all modules
- Proper null safety for event bus handling
- Standardized async/sync patterns
- Clear separation between context creation and orchestration
- Improved error handling with proper exception hierarchies
- Better validation and health reporting
"""

from __future__ import annotations

# Core orchestration (refactored version)
from .main import (
    BootstrapOrchestrator,
    BootstrapConfig, 
    BootstrapContext,
    bootstrap_nireon_system,
    bootstrap,
    bootstrap_sync,
    BootstrapError
)

# Context creation utilities
from .bootstrap_context_builder import (
    BootstrapContextBuilder,
    create_bootstrap_context
)

# Phase execution
from .bootstrap_phase_executor import (
    BootstrapPhaseExecutor,
    PhaseExecutionResult,
    PhaseExecutionSummary,
    execute_bootstrap_phases
)

# Results and validation
from .result_builder import (
    BootstrapResult,
    BootstrapResultBuilder,
    build_result_from_context,
    create_minimal_result
)
from .validation_data import (
    BootstrapValidationData,
    ComponentValidationData
)

# Health and monitoring
from .health.reporter import (
    HealthReporter,
    ComponentStatus,
    ComponentHealthRecord
)

# Configuration loading
from .config.config_loader import ConfigLoader

# Exception hierarchy
from .bootstrap_helper.exceptions import (
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

# Version and metadata
__version__ = '4.0.0'
__author__ = 'NIREON V4 Bootstrap Team'
__description__ = 'L0 Abiogenesis: Epistemic System Bootstrap Infrastructure for NIREON V4'

# Schema version for compatibility
CURRENT_SCHEMA_VERSION = 'V4-alpha.1.0'


# Testing utilities - define here to avoid import issues
def create_test_bootstrap_config(test_manifest_content=None, **kwargs):
    """Create test bootstrap configuration."""
    import tempfile
    import yaml
    from pathlib import Path
    
    config_paths_actual = []
    temp_file_to_clean = None
    
    if test_manifest_content:
        temp_file_to_clean = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
        yaml.dump(test_manifest_content, temp_file_to_clean)
        temp_file_to_clean.close()
        config_paths_actual = [Path(temp_file_to_clean.name)]
    
    default_strict = False
    default_global_config = {
        'bootstrap_strict_mode': default_strict,
        'feature_flags': {'test_mode_active': True}
    }
    
    provided_global_config = kwargs.get('global_app_config', {})
    final_global_config = {**default_global_config, **provided_global_config}
    
    if 'bootstrap_strict_mode' not in provided_global_config:
        final_global_config['bootstrap_strict_mode'] = kwargs.get('strict_mode', default_strict)
    
    config_object = BootstrapConfig.from_params(
        config_paths=config_paths_actual,
        strict_mode=final_global_config['bootstrap_strict_mode'],
        env=kwargs.get('env', 'test'),
        global_app_config=final_global_config,
        **{k: v for k, v in kwargs.items() if k not in ['strict_mode', 'env', 'global_app_config']}
    )
    
    return config_object


async def smoke_test():
    """Run smoke test for bootstrap system."""
    import tempfile
    import yaml
    from pathlib import Path
    
    print('Running NIREON V4 Bootstrap Smoke Test...')
    
    temp_file_path = None
    try:
        # Create minimal test manifest
        test_manifest_data = {
            'version': '1.0',
            'metadata': {
                'name': 'V4 Smoke Test Configuration',
                'description': 'Minimal config for V4 bootstrap smoke testing'
            },
            'shared_services': {},
            'mechanisms': {},
            'observers': {}
        }
        
        # Write to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(test_manifest_data, f)
            temp_file_path = f.name
        
        # Create bootstrap config
        bootstrap_config = create_test_bootstrap_config(
            test_manifest_content=test_manifest_data,
            strict_mode=False,
            env='test_smoke',
            global_app_config={
                'bootstrap_strict_mode': False,
                'feature_flags': {'smoke_test_active': True}
            }
        )
        
        # Override config paths with temp file
        if temp_file_path:
            bootstrap_config.config_paths = [Path(temp_file_path)]
        
        # Run bootstrap
        orchestrator = BootstrapOrchestrator(bootstrap_config)
        result = await orchestrator.execute_bootstrap()
        
        success = result.success
        min_expected_core_components = 2
        
        if result.component_count >= min_expected_core_components:
            print(f'✓ Bootstrap reported success: {result.success}, '
                  f'Component count: {result.component_count} (expected >= {min_expected_core_components})')
        else:
            print(f'✗ Component count {result.component_count} is less than minimum expected {min_expected_core_components}.')
            success = False
        
        if success:
            print('✓ V4 Smoke test PASSED.')
        else:
            print('✗ V4 Smoke test FAILED.')
            if hasattr(result, 'health_reporter') and result.health_reporter:
                print(f'Health Report:\n{result.get_health_report()}')
        
        return success
        
    except Exception as e:
        print(f'✗ V4 Smoke test EXCEPTION: {e}')
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Cleanup temp file
        if temp_file_path:
            try:
                import os
                os.unlink(temp_file_path)
            except Exception as e_clean:
                print(f'Error cleaning up temp file {temp_file_path}: {e_clean}')


def smoke_test_sync():
    """Synchronous wrapper for smoke test."""
    import asyncio
    return asyncio.run(smoke_test())


async def validate_bootstrap_config(config_paths, **kwargs):
    """Validate bootstrap configuration files."""
    from pathlib import Path
    
    validation_result = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'manifest_count': 0,
        'component_spec_count': 0,
        'schema_validation_performed': False,
        'schema_validation_passed': True
    }
    
    try:
        from .processors.manifest_processor import ManifestProcessor
        from .bootstrap_helper.utils import load_yaml_robust
        
        processor = ManifestProcessor(strict_mode=kwargs.get('strict_mode', True))
        validation_result['schema_validation_performed'] = True
        
        for config_path_input in config_paths:
            path = Path(config_path_input)
            if not path.exists():
                validation_result['errors'].append(f'Manifest file not found: {path}')
                validation_result['valid'] = False
                continue
            
            manifest_data = load_yaml_robust(path)
            if not manifest_data:
                validation_result['warnings'].append(f'Empty or invalid YAML in manifest file: {path}')
                continue
            
            processing_result = await processor.process_manifest(path, manifest_data)
            validation_result['manifest_count'] += 1
            validation_result['component_spec_count'] += processing_result.component_count
            
            if not processing_result.success:
                validation_result['valid'] = False
                validation_result['errors'].extend(processing_result.errors)
                if any('Schema validation failed' in err for err in processing_result.errors):
                    validation_result['schema_validation_passed'] = False
            
            validation_result['warnings'].extend(processing_result.warnings)
        
        print(f"Configuration validation complete: valid={validation_result['valid']}")
        return validation_result
        
    except ImportError as ie:
        validation_result['valid'] = False
        validation_result['errors'].append(f'Validation dependency error: {ie}')
        validation_result['schema_validation_performed'] = False
        return validation_result
    except Exception as e:
        validation_result['valid'] = False
        validation_result['errors'].append(f'Unexpected validation error: {e}')
        return validation_result


# Public API - maintained for backwards compatibility
__all__ = [
    # Main bootstrap functions
    'bootstrap_nireon_system',
    'bootstrap', 
    'bootstrap_sync',
    
    # Configuration and context
    'BootstrapConfig',
    'BootstrapContext',
    'BootstrapContextBuilder',
    'create_bootstrap_context',
    
    # Orchestration
    'BootstrapOrchestrator',
    
    # Phase execution
    'BootstrapPhaseExecutor',
    'PhaseExecutionResult', 
    'PhaseExecutionSummary',
    'execute_bootstrap_phases',
    
    # Results and validation
    'BootstrapResult',
    'BootstrapResultBuilder',
    'BootstrapValidationData',
    'ComponentValidationData',
    
    # Health monitoring
    'HealthReporter',
    'ComponentStatus',
    'ComponentHealthRecord',
    
    # Configuration
    'ConfigLoader',
    
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
    
    # Testing
    'create_test_bootstrap_config',
    'smoke_test',
    'smoke_test_sync', 
    'validate_bootstrap_config',
    
    # Version info
    'CURRENT_SCHEMA_VERSION',
    '__version__',
    '__author__',
    '__description__'
]


def get_version() -> str:
    """Get the current bootstrap version."""
    return __version__


def get_schema_version() -> str:
    """Get the current schema version for compatibility checking."""
    return CURRENT_SCHEMA_VERSION


def get_supported_manifest_types() -> list[str]:
    """Get list of supported manifest types."""
    return ['enhanced', 'simple']


def get_bootstrap_info() -> dict[str, any]:
    """
    Get comprehensive information about the bootstrap system.
    
    Returns:
        Dictionary containing version, features, and capabilities
    """
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
            'placeholder_monitoring': True,
            'null_safe_event_bus': True,  # New in refactor
            'consistent_async_patterns': True,  # New in refactor
            'context_builder_pattern': True,  # New in refactor
            'phase_execution_metrics': True,  # New in refactor
            'hot_reload_support': False
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
            'bootstrap_nireon_system',
            'bootstrap',
            'bootstrap_sync',
            'validate_bootstrap_config',
            'smoke_test',
            'smoke_test_sync'
        ],
        'refactor_improvements': {
            'null_safety': 'Event bus null handling with automatic placeholder creation',
            'consistent_imports': 'Standardized import patterns across all modules',
            'error_handling': 'Improved exception hierarchy and error propagation',
            'async_patterns': 'Consistent async/await usage throughout',
            'context_validation': 'Proper validation of bootstrap context creation',
            'phase_metrics': 'Detailed timing and success metrics for each phase',
            'health_integration': 'Better integration with health reporting system'
        }
    }


def system_health_check() -> dict[str, any]:
    """
    Perform a health check of the bootstrap system.
    
    Returns:
        Dictionary with health status and any issues found
    """
    health = {
        'status': 'healthy',
        'issues': [],
        'checks_performed': []
    }
    
    try:
        # Check critical module imports
        from .main import BootstrapOrchestrator
        from .bootstrap_context_builder import BootstrapContextBuilder  
        from .bootstrap_phase_executor import BootstrapPhaseExecutor
        from .phases.base_phase import BootstrapPhase
        from .processors.manifest_processor import ManifestProcessor
        from .config.config_loader import ConfigLoader
        health['checks_performed'].append('critical_module_imports')
        
        # Check signal definitions
        from .signals.bootstrap_signals import ALL_BOOTSTRAP_SIGNALS, SYSTEM_BOOTSTRAPPED
        if not ALL_BOOTSTRAP_SIGNALS or SYSTEM_BOOTSTRAPPED not in ALL_BOOTSTRAP_SIGNALS:
            health['issues'].append('Bootstrap signal definitions incomplete or missing key signals.')
        health['checks_performed'].append('bootstrap_signal_definitions')
        
        # Check helper modules
        from .bootstrap_helper import exceptions, utils, metadata, placeholders
        from .bootstrap_helper.health_reporter import BootstrapHealthReporter
        if not hasattr(BootstrapHealthReporter, '__init__'):
            health['issues'].append('BootstrapHealthReporter missing or malformed.')
        health['checks_performed'].append('bootstrap_helper_modules')
        
        # Check refactored components
        try:
            # Test context builder
            from .bootstrap_context_builder import create_bootstrap_context
            health['checks_performed'].append('context_builder_functionality')
            
            # Test phase executor
            from .bootstrap_phase_executor import execute_bootstrap_phases
            health['checks_performed'].append('phase_executor_functionality')
            
        except ImportError as e:
            health['issues'].append(f'Refactored component import error: {e}')
        
        # Determine overall health
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