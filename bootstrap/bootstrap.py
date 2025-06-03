"""
NIREON V4 Bootstrap System - Main Entry Point

This module provides the public API for bootstrapping NIREON V4 systems.
It coordinates the entire initialization process from configuration loading
through component instantiation and validation.

The bootstrap system implements L0 Abiogenesis - the emergence of epistemic
capability from configuration and manifest specifications.
"""

from __future__ import annotations
import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from core.registry import ComponentRegistry
from application.ports.event_bus_port import EventBusPort
from bootstrap.orchestrator import BootstrapOrchestrator, BootstrapConfig
from bootstrap.result_builder import BootstrapResult
from bootstrap.validation_data import BootstrapValidationData
from bootstrap.bootstrap_helper.exceptions import BootstrapError

# Current schema version for bootstrap validation
CURRENT_SCHEMA_VERSION = "4.0"

logger = logging.getLogger(__name__)

# Public API exports
__all__ = [
    'bootstrap_nireon_system',
    'bootstrap', 
    'BootstrapResult', 
    'BootstrapValidationData',
    'CURRENT_SCHEMA_VERSION',
    'BootstrapError'
]


async def bootstrap_nireon_system(
    config_paths: List[Union[str, Path]],
    *,
    existing_registry: Optional[ComponentRegistry] = None,
    existing_event_bus: Optional[EventBusPort] = None,
    manifest_style: str = 'auto',
    replay: bool = False,
    env: Optional[str] = None,
    global_app_config: Optional[Dict[str, Any]] = None,
    strict_mode: bool = True
) -> BootstrapResult:
    """
    Bootstrap the NIREON V4 system from configuration and manifests.
    
    This is the main entry point for initializing a complete NIREON system.
    It implements L0 Abiogenesis - the emergence of epistemic capability
    from static configuration into dynamic reasoning components.
    
    Args:
        config_paths: List of paths to manifest files to process
        existing_registry: Optional pre-initialized component registry
        existing_event_bus: Optional pre-initialized event bus
        manifest_style: Manifest type hint ('auto', 'enhanced', 'simple')
        replay: Whether to run in replay mode with deterministic seeds
        env: Environment name for configuration ('dev', 'prod', etc.)
        global_app_config: Override global configuration
        strict_mode: Whether to fail fast on errors (vs. resilient mode)
        
    Returns:
        BootstrapResult containing initialized registry, health report, and validation data
        
    Raises:
        BootstrapError: If critical initialization failures occur in strict mode
        
    Example:
        >>> # Basic bootstrap
        >>> result = await bootstrap_nireon_system(['config/manifests/standard.yaml'])
        >>> registry = result.registry
        >>> 
        >>> # Advanced bootstrap with custom config
        >>> result = await bootstrap_nireon_system(
        ...     config_paths=['manifests/custom.yaml'],
        ...     env='production',
        ...     strict_mode=True,
        ...     global_app_config={'feature_flags': {'enable_advanced_reasoning': True}}
        ... )
    """
    logger.info("=== NIREON V4 System Bootstrap Starting ===")
    logger.info(f"Config paths: {config_paths}")
    logger.info(f"Environment: {env or 'default'}")
    logger.info(f"Strict mode: {strict_mode}")
    
    try:
        # Create bootstrap configuration
        config = BootstrapConfig.from_params(
            config_paths=config_paths,
            existing_registry=existing_registry,
            existing_event_bus=existing_event_bus,
            manifest_style=manifest_style,
            replay=replay,
            env=env,
            global_app_config=global_app_config,
            strict_mode=strict_mode
        )
        
        # Create and execute orchestrator
        orchestrator = BootstrapOrchestrator(config)
        result = await orchestrator.execute_bootstrap()
        
        # Log completion
        if result.success:
            logger.info(
                f"✓ NIREON V4 Bootstrap Complete - {result.component_count} components, "
                f"{result.healthy_component_count} healthy"
            )
        else:
            logger.error(
                f"✗ NIREON V4 Bootstrap Failed - {result.critical_failure_count} critical failures"
            )
        
        return result
        
    except Exception as e:
        logger.error(f"Critical bootstrap failure: {e}", exc_info=True)
        
        if strict_mode:
            raise BootstrapError(f"Bootstrap system failure: {e}") from e
        
        # In non-strict mode, return a minimal result with the error
        logger.warning("Continuing in non-strict mode despite bootstrap failure")
        registry = existing_registry or ComponentRegistry()
        
        from bootstrap.result_builder import create_minimal_result
        result = create_minimal_result(registry, run_id="failed_bootstrap")
        return result


async def bootstrap(
    config_paths: List[Union[str, Path]],
    **kwargs
) -> BootstrapResult:
    """
    Convenience wrapper for bootstrap_nireon_system.
    
    This provides a shorter function name for the most common bootstrap use case.
    All arguments are passed through to bootstrap_nireon_system.
    
    Args:
        config_paths: List of manifest file paths
        **kwargs: Additional arguments passed to bootstrap_nireon_system
        
    Returns:
        BootstrapResult from the bootstrap process
        
    Example:
        >>> result = await bootstrap(['config/manifests/standard.yaml'])
        >>> print(f"Bootstrap success: {result.success}")
    """
    return await bootstrap_nireon_system(config_paths, **kwargs)


def bootstrap_sync(
    config_paths: List[Union[str, Path]],
    **kwargs
) -> BootstrapResult:
    """
    Synchronous wrapper for bootstrap process.
    
    This function handles the async event loop for callers who need
    a synchronous interface to the bootstrap process.
    
    Args:
        config_paths: List of manifest file paths
        **kwargs: Additional arguments passed to bootstrap_nireon_system
        
    Returns:
        BootstrapResult from the bootstrap process
        
    Example:
        >>> result = bootstrap_sync(['config/manifests/standard.yaml'])
        >>> registry = result.registry
    """
    logger.debug("Running bootstrap in synchronous mode")
    
    try:
        # Get or create event loop
        loop = asyncio.get_event_loop()
    except RuntimeError:
        # No event loop in current thread, create one
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    try:
        return loop.run_until_complete(bootstrap_nireon_system(config_paths, **kwargs))
    finally:
        # Clean up loop if we created it
        if not loop.is_running():
            loop.close()


def create_test_bootstrap_config(
    test_manifest: Optional[Dict[str, Any]] = None,
    **kwargs
) -> BootstrapConfig:
    """
    Create a bootstrap configuration for testing.
    
    Args:
        test_manifest: Optional test manifest data
        **kwargs: Additional config parameters
        
    Returns:
        BootstrapConfig suitable for testing
    """
    import tempfile
    import yaml
    
    # Create temporary manifest file if provided
    config_paths = []
    if test_manifest:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(test_manifest, f)
            config_paths = [f.name]
    
    return BootstrapConfig.from_params(
        config_paths=config_paths,
        strict_mode=kwargs.get('strict_mode', False),  # Default to non-strict for tests
        env=kwargs.get('env', 'test'),
        global_app_config=kwargs.get('global_app_config', {'bootstrap_strict_mode': False}),
        **{k: v for k, v in kwargs.items() if k not in ['strict_mode', 'env', 'global_app_config']}
    )


async def validate_bootstrap_config(
    config_paths: List[Union[str, Path]],
    **kwargs
) -> Dict[str, Any]:
    """
    Validate bootstrap configuration without performing full bootstrap.
    
    This function loads and validates configuration files and manifests
    without instantiating components, useful for configuration testing.
    
    Args:
        config_paths: List of manifest file paths to validate
        **kwargs: Additional arguments for validation context
        
    Returns:
        Dictionary with validation results
        
    Example:
        >>> validation_result = await validate_bootstrap_config(['manifests/test.yaml'])
        >>> if validation_result['valid']:
        ...     print("Configuration is valid")
    """
    logger.info(f"Validating bootstrap configuration: {config_paths}")
    
    validation_result = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'manifest_count': 0,
        'component_count': 0,
        'schema_validation': True
    }
    
    try:
        from bootstrap.processors.manifest_processor import ManifestProcessor
        from bootstrap.bootstrap_helper.utils import load_yaml_robust
        
        processor = ManifestProcessor(strict_mode=kwargs.get('strict_mode', True))
        
        for config_path in config_paths:
            path = Path(config_path)
            if not path.exists():
                validation_result['errors'].append(f"Manifest file not found: {path}")
                validation_result['valid'] = False
                continue
            
            manifest_data = load_yaml_robust(path)
            if not manifest_data:
                validation_result['warnings'].append(f"Empty manifest file: {path}")
                continue
            
            # Process manifest for validation
            result = await processor.process_manifest(path, manifest_data)
            validation_result['manifest_count'] += 1
            validation_result['component_count'] += len(result.components)
            
            if not result.success:
                validation_result['valid'] = False
                validation_result['errors'].extend(result.errors)
                validation_result['schema_validation'] = False
            
            validation_result['warnings'].extend(result.warnings)
        
        logger.info(f"Configuration validation complete: valid={validation_result['valid']}")
        return validation_result
        
    except Exception as e:
        logger.error(f"Configuration validation failed: {e}", exc_info=True)
        validation_result['valid'] = False
        validation_result['errors'].append(f"Validation error: {e}")
        return validation_result


# Smoke test function for quick verification
async def smoke_test() -> bool:
    """
    Perform a quick smoke test of the bootstrap system.
    
    Creates a minimal configuration and attempts bootstrap to verify
    that the system is working correctly.
    
    Returns:
        True if smoke test passes, False otherwise
    """
    logger.info("Running NIREON V4 bootstrap smoke test")
    
    try:
        # Create minimal test manifest
        test_manifest = {
            'version': '1.0',
            'metadata': {
                'name': 'Smoke Test Configuration',
                'description': 'Minimal config for smoke testing'
            },
            'shared_services': {},
            'mechanisms': {},
            'observers': {}
        }
        
        # Run bootstrap with test config
        config = create_test_bootstrap_config(test_manifest)
        orchestrator = BootstrapOrchestrator(config)
        result = await orchestrator.execute_bootstrap()
        
        success = result.success and result.component_count >= 0
        
        if success:
            logger.info("✓ Smoke test passed")
        else:
            logger.error("✗ Smoke test failed")
            logger.error(result.get_health_report())
        
        return success
        
    except Exception as e:
        logger.error(f"Smoke test exception: {e}", exc_info=True)
        return False


def smoke_test_sync() -> bool:
    """
    Synchronous version of smoke test.
    
    Returns:
        True if smoke test passes, False otherwise
    """
    return asyncio.run(smoke_test())


# Main entry point for CLI usage
async def main():
    """
    Main entry point for command-line bootstrap execution.
    
    This function provides a simple way to bootstrap NIREON from the command line
    using default configurations.
    """
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python -m bootstrap.bootstrap <manifest_file> [<manifest_file2> ...]")
        print("       python -m bootstrap.bootstrap --smoke-test")
        sys.exit(1)
    
    if sys.argv[1] == '--smoke-test':
        success = await smoke_test()
        sys.exit(0 if success else 1)
    
    # Bootstrap with provided manifest files
    config_paths = sys.argv[1:]
    
    try:
        result = await bootstrap_nireon_system(config_paths)
        
        print(f"Bootstrap completed: {result.success}")
        print(f"Components: {result.component_count}")
        print(f"Healthy: {result.healthy_component_count}")
        
        if not result.success:
            print("\nHealth Report:")
            print(result.get_health_report())
            sys.exit(1)
            
    except Exception as e:
        print(f"Bootstrap failed: {e}")
        logger.error("Bootstrap failed", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    # Set up basic logging for CLI usage
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    asyncio.run(main())