# Path: nireon_v4/bootstrap/bootstrap.py
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# V4 imports
from core.registry import ComponentRegistry
from application.ports.event_bus_port import EventBusPort
from bootstrap.orchestrator import BootstrapOrchestrator, BootstrapConfig
from bootstrap.result_builder import BootstrapResult # V4 BootstrapResult
from bootstrap.validation_data import BootstrapValidationData # V4 ValidationData
from bootstrap.bootstrap_helper.exceptions import BootstrapError # V4 BootstrapError

# This should align with what's defined in NIREON V4 API Governance.md or similar
# For now, using a generic V4 version.
CURRENT_SCHEMA_VERSION = 'V4-alpha.1.0' # Placeholder for actual V4 schema version string

logger = logging.getLogger(__name__)

__all__ = [
    'bootstrap_nireon_system', 
    'bootstrap', 
    'bootstrap_sync',
    'BootstrapResult', 
    'BootstrapValidationData', # V4 version from bootstrap.validation_data
    'CURRENT_SCHEMA_VERSION',
    'BootstrapError',
    'validate_bootstrap_config',
    'smoke_test',
    'smoke_test_sync',
    'create_test_bootstrap_config' # Added for testing convenience
]


async def bootstrap_nireon_system(
    config_paths: List[Union[str, Path]], 
    *,
    existing_registry: Optional[ComponentRegistry] = None,
    existing_event_bus: Optional[EventBusPort] = None,
    manifest_style: str = 'auto', # 'auto', 'simple', 'enhanced'
    replay: bool = False, # For replaying events, not fully used in bootstrap
    env: Optional[str] = None,
    global_app_config: Optional[Dict[str, Any]] = None,
    strict_mode: bool = True # V4 default, can be overridden by global_app_config
) -> BootstrapResult:
    """
    Core V4 function to initialize the NIREON system.
    It orchestrates the bootstrap process through defined phases.
    """
    logger.info('=== NIREON V4 System Bootstrap Starting (Orchestrated) ===')
    logger.info(f'Run Environment (specified or default): {env or "default"}')
    logger.info(f'Effective Strict Mode: {strict_mode}') # This will be the input, global_app_config can override
    logger.info(f'Manifest Style: {manifest_style}')
    logger.info(f'Configuration Paths: {config_paths}')

    try:
        # BootstrapConfig will handle resolving strict_mode from global_app_config if provided
        bootstrap_config_obj = BootstrapConfig.from_params(
            config_paths=config_paths,
            existing_registry=existing_registry,
            existing_event_bus=existing_event_bus,
            manifest_style=manifest_style,
            replay=replay, # Replay for bootstrap context, not full event replay
            env=env,
            global_app_config=global_app_config, # Pass it here
            strict_mode=strict_mode # Initial strict_mode
        )
        
        orchestrator = BootstrapOrchestrator(bootstrap_config_obj)
        result: BootstrapResult = await orchestrator.execute_bootstrap() # This is V4 BootstrapResult

        if result.success:
            logger.info(
                f"✓ NIREON V4 Bootstrap Complete. Run ID: {result.run_id}. "
                f"Components: {result.component_count}, Healthy: {result.healthy_component_count}."
            )
        else:
            logger.error(
                f"✗ NIREON V4 Bootstrap Failed. Run ID: {result.run_id}. "
                f"Critical Failures: {result.critical_failure_count}."
            )
            if result.health_reporter: # V4HealthReporter
                 logger.error(f"Health Report Snippet:\n{result.get_health_report()[:1000]}")


        return result
    except Exception as e:
        logger.critical(f'Critical bootstrap failure: {e}', exc_info=True)
        # In strict mode, this exception would propagate. 
        # If not strict, or for a graceful exit, create a minimal failure result.
        effective_strict_mode = strict_mode
        if global_app_config and 'bootstrap_strict_mode' in global_app_config:
            effective_strict_mode = global_app_config['bootstrap_strict_mode']
        
        if effective_strict_mode:
            if isinstance(e, BootstrapError):
                raise
            raise BootstrapError(f'Bootstrap system failure: {e}') from e
        
        logger.warning('Continuing in non-strict mode despite bootstrap failure.')
        registry = existing_registry or ComponentRegistry()
        from bootstrap.result_builder import create_minimal_result # V4 minimal result
        # run_id needs to be defined here for the minimal result
        run_id = f"failed_bootstrap_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
        result = create_minimal_result(registry, run_id=run_id)
        if result.health_reporter:
            result.health_reporter.add_phase_result(
                "OverallBootstrap", "failed", f"Critical failure: {e}", errors=[str(e)]
            )
        return result

async def bootstrap(
    config_paths: List[Union[str, Path]], 
    **kwargs
) -> BootstrapResult: # V4 BootstrapResult
    """Convenience wrapper for bootstrap_nireon_system."""
    return await bootstrap_nireon_system(config_paths, **kwargs)

def bootstrap_sync(
    config_paths: List[Union[str, Path]], 
    **kwargs
) -> BootstrapResult: # V4 BootstrapResult
    """Synchronous version of the bootstrap function."""
    logger.debug('Running V4 bootstrap in synchronous mode')
    try:
        # Ensure an event loop is available for the current thread
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If a loop is already running (e.g. in Jupyter), create a new task
            # This is a bit complex; for simple CLI, new_event_loop is safer.
            # For library use where a loop might exist, this could be an option.
            # However, for now, let's stick to simpler new_event_loop approach.
            logger.warning("An event loop is already running. Creating a new one for sync bootstrap might cause issues.")
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            result = new_loop.run_until_complete(bootstrap_nireon_system(config_paths, **kwargs))
            new_loop.close()
            asyncio.set_event_loop(loop) # Restore original loop
            return result
        else:
            return loop.run_until_complete(bootstrap_nireon_system(config_paths, **kwargs))
    except RuntimeError: # No event loop in current thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(bootstrap_nireon_system(config_paths, **kwargs))
        finally:
            loop.close()
            # Reset the event loop for the current thread to None,
            # so a subsequent call to asyncio.get_event_loop() in the same thread
            # would create a new one or fail if not allowed.
            asyncio.set_event_loop(None) 

def create_test_bootstrap_config(
    test_manifest_content: Optional[Dict[str, Any]] = None, 
    **kwargs
) -> BootstrapConfig: # V4 BootstrapConfig
    """
    Helper to create a BootstrapConfig for testing, potentially with an in-memory manifest.
    """
    import tempfile
    import yaml # Ensure yaml is imported

    config_paths_actual: List[Path] = []
    temp_file_to_clean: Optional[tempfile.NamedTemporaryFile] = None

    if test_manifest_content:
        # Create a temporary file for the manifest content
        temp_file_to_clean = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
        yaml.dump(test_manifest_content, temp_file_to_clean)
        temp_file_to_clean.close() # Close it so it can be read by bootstrap
        config_paths_actual = [Path(temp_file_to_clean.name)]
    
    # Default to non-strict mode for tests unless overridden
    default_strict = False
    default_global_config = {'bootstrap_strict_mode': default_strict, 'feature_flags': {'test_mode_active': True}}

    # Merge provided global_app_config with defaults
    provided_global_config = kwargs.get('global_app_config', {})
    final_global_config = {**default_global_config, **provided_global_config}
    
    # Ensure 'bootstrap_strict_mode' is explicitly set if not in kwargs' global_app_config
    if 'bootstrap_strict_mode' not in provided_global_config:
        final_global_config['bootstrap_strict_mode'] = kwargs.get('strict_mode', default_strict)
    else: # If it was in provided_global_config, ensure strict_mode kwarg matches for clarity
        kwargs['strict_mode'] = final_global_config['bootstrap_strict_mode']


    config_object = BootstrapConfig.from_params(
        config_paths=config_paths_actual,
        # Use resolved strict_mode for the BootstrapConfig object
        strict_mode=final_global_config['bootstrap_strict_mode'], 
        env=kwargs.get('env', 'test'),
        global_app_config=final_global_config, # Pass the merged one
        **{k: v for k, v in kwargs.items() if k not in ['strict_mode', 'env', 'global_app_config']}
    )
    
    # Store temp file name for potential cleanup by caller if needed, or handle here
    # For simplicity, let's assume caller might want to inspect it or it's cleaned up by OS
    # If using `delete=True` with NamedTemporaryFile, it's auto-cleaned. `delete=False` requires manual.
    # This is tricky if bootstrap_nireon_system needs to open it by path.
    # For now, we leave temp_file_to_clean.name as the path. Caller should handle cleanup if delete=False.
    # An alternative is to pass content directly if bootstrap supports it.
    if temp_file_to_clean:
        logger.debug(f"Created temporary manifest for testing at: {temp_file_to_clean.name}")
        # To clean up: os.unlink(temp_file_to_clean.name) after bootstrap finishes

    return config_object


async def validate_bootstrap_config(
    config_paths: List[Union[str, Path]], 
    **kwargs
) -> Dict[str, Any]:
    """
    Validates manifest files and basic configuration without full component instantiation.
    """
    logger.info(f'Validating bootstrap configuration for paths: {config_paths}')
    validation_result: Dict[str, Any] = {
        'valid': True, 
        'errors': [], 
        'warnings': [],
        'manifest_count': 0,
        'component_spec_count': 0, # Number of component specifications found
        'schema_validation_performed': False, # Will be True if jsonschema is used
        'schema_validation_passed': True # Assumes pass unless errors found
    }
    
    try:
        from bootstrap.processors.manifest_processor import ManifestProcessor # V4 ManifestProcessor
        from bootstrap.bootstrap_helper.utils import load_yaml_robust # V4 util

        processor = ManifestProcessor(strict_mode=kwargs.get('strict_mode', True))
        validation_result['schema_validation_performed'] = True # Assuming jsonschema will be attempted

        for config_path_input in config_paths:
            path = Path(config_path_input)
            if not path.exists():
                validation_result['errors'].append(f'Manifest file not found: {path}')
                validation_result['valid'] = False
                continue
            
            manifest_data = load_yaml_robust(path)
            if not manifest_data:
                validation_result['warnings'].append(f'Empty or invalid YAML in manifest file: {path}')
                # Potentially not a critical error if other manifests exist and are valid
                continue
            
            # ManifestProcessor.process_manifest will perform schema validation if jsonschema is available
            # and parse component specs.
            processing_result = await processor.process_manifest(path, manifest_data)
            
            validation_result['manifest_count'] += 1
            validation_result['component_spec_count'] += processing_result.component_count
            
            if not processing_result.success:
                validation_result['valid'] = False
                validation_result['errors'].extend(processing_result.errors)
                if any("Schema validation failed" in err for err in processing_result.errors):
                     validation_result['schema_validation_passed'] = False
            validation_result['warnings'].extend(processing_result.warnings)

        logger.info(f"Configuration validation complete: valid={validation_result['valid']}")
        return validation_result

    except ImportError as ie:
        # If ManifestProcessor or its deps (like jsonschema if made mandatory) can't be imported
        logger.error(f'Configuration validation dependency error: {ie}', exc_info=True)
        validation_result['valid'] = False
        validation_result['errors'].append(f'Validation dependency error: {ie}')
        validation_result['schema_validation_performed'] = False # Schema validation couldn't run
        return validation_result
    except Exception as e:
        logger.error(f'Configuration validation failed with an unexpected error: {e}', exc_info=True)
        validation_result['valid'] = False
        validation_result['errors'].append(f'Unexpected validation error: {e}')
        return validation_result


async def smoke_test() -> bool:
    """Performs a basic smoke test of the V4 bootstrap system."""
    logger.info('--- Running NIREON V4 Bootstrap Smoke Test ---')
    temp_file_path = None
    try:
        # Minimal manifest content for the smoke test
        test_manifest_data = {
            'version': '1.0', # V4 manifest version
            'metadata': {
                'name': 'V4 Smoke Test Configuration',
                'description': 'Minimal config for V4 bootstrap smoke testing'
            },
            'shared_services': {}, # Empty but present section
            'mechanisms': {},      # Empty but present section
            'observers': {},       # Empty but present section
        }
        # Using the helper to create config with a temporary manifest file
        # The helper create_test_bootstrap_config handles temporary file creation.
        # We need to ensure cleanup.
        
        import tempfile
        import yaml
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(test_manifest_data, f)
            temp_file_path = f.name
        
        # Pass strict_mode=False for smoke test to ensure it attempts full run
        # and global_app_config to ensure bootstrap_strict_mode is also false.
        bootstrap_config = create_test_bootstrap_config(
            test_manifest_content=test_manifest_data, # This will be ignored if temp_file_path is used
            strict_mode=False, 
            env='test_smoke',
            global_app_config={'bootstrap_strict_mode': False, 
                               'feature_flags': {'smoke_test_active': True}}
        )
        # The create_test_bootstrap_config will create a temp file if content is passed.
        # We need to make sure the actual bootstrap_nireon_system uses this path.
        # Overriding config_paths in the created bootstrap_config object if it was created
        # by passing test_manifest_content to create_test_bootstrap_config
        if temp_file_path and hasattr(bootstrap_config, 'config_paths'):
             bootstrap_config.config_paths = [Path(temp_file_path)]


        orchestrator = BootstrapOrchestrator(bootstrap_config)
        result: BootstrapResult = await orchestrator.execute_bootstrap() # V4 result

        # Basic V4 checks
        success = result.success
        # Check if core services like ComponentRegistry, FeatureFlagsManager are there.
        # AbiogenesisPhase should register these.
        min_expected_core_components = 2 # ComponentRegistry itself, FeatureFlagsManager
        
        if result.component_count >= min_expected_core_components:
            logger.info(f"Smoke test: Bootstrap reported success: {result.success}, "
                        f"Component count: {result.component_count} (expected >= {min_expected_core_components})")
        else:
            logger.error(f"Smoke test: Component count {result.component_count} is less than "
                         f"minimum expected {min_expected_core_components}.")
            success = False # Override success if core components are missing

        if success:
            logger.info('✓ V4 Smoke test PASSED.')
        else:
            logger.error('✗ V4 Smoke test FAILED.')
            if result.health_reporter:
                logger.error(f"Health Report:\n{result.get_health_report()}")
        return success
    except Exception as e:
        logger.error(f'V4 Smoke test EXCEPTION: {e}', exc_info=True)
        return False
    finally:
        if temp_file_path:
            try:
                import os
                os.unlink(temp_file_path)
                logger.debug(f"Cleaned up temporary smoke test manifest: {temp_file_path}")
            except Exception as e_clean:
                logger.error(f"Error cleaning up smoke test manifest {temp_file_path}: {e_clean}")
        logger.info('--- End of NIREON V4 Bootstrap Smoke Test ---')


def smoke_test_sync() -> bool:
    """Synchronous version of the V4 smoke test."""
    return asyncio.run(smoke_test())


async def main_cli_entry(): # Renamed to avoid conflict if this file is run directly
    """CLI entry point for bootstrap.py (mainly for testing)."""
    import sys
    if len(sys.argv) < 2:
        print('Usage: python -m bootstrap.bootstrap <manifest_file1.yaml> [<manifest_file2.yaml> ...]')
        print('       python -m bootstrap.bootstrap --smoke-test')
        sys.exit(1)

    if sys.argv[1] == '--smoke-test':
        success = await smoke_test()
        sys.exit(0 if success else 1)

    config_paths_str: List[str] = sys.argv[1:]
    config_paths_path: List[Path] = [Path(p) for p in config_paths_str]
    
    try:
        # Example: Running with default strict_mode=True
        result: BootstrapResult = await bootstrap_nireon_system(config_paths_path, strict_mode=True)
        
        print(f'\nBootstrap Process Summary (Run ID: {result.run_id}):')
        print(f"  Overall Success: {result.success}")
        print(f"  Total Components in Registry: {result.component_count}")
        print(f"  Healthy/Operational Components: {result.healthy_component_count}")
        print(f"  Critical Failures: {result.critical_failure_count}")
        print(f"  Bootstrap Duration: {result.bootstrap_duration:.2f} seconds")
        
        if not result.success and result.health_reporter:
            print('\n--- Health Report Snippet ---')
            # Printing a snippet as the full report can be long
            report_lines = result.get_health_report().splitlines()
            for line in report_lines[:20]: # Print first 20 lines
                print(line)
            if len(report_lines) > 20:
                print("    ...")
            print('--- End Health Report Snippet ---')

        sys.exit(0 if result.success else 1)

    except BootstrapError as be:
        print(f'\nBOOTSTRAP ERROR: {be}')
        logger.error('Bootstrap process failed due to BootstrapError.', exc_info=True)
        sys.exit(2)
    except Exception as e:
        print(f'\nUNEXPECTED BOOTSTRAP FAILURE: {e}')
        logger.error('Bootstrap process failed due to an unexpected error.', exc_info=True)
        sys.exit(3)

if __name__ == '__main__':
    # Setup basic logging for direct script execution
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    asyncio.run(main_cli_entry())