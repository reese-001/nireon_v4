#!/usr/bin/env python3
"""
Package-level CLI entry point for NIREON V4 Bootstrap.

This allows users to run: python -m bootstrap
"""

import asyncio
import sys
import logging
from pathlib import Path

# Configure logging for CLI usage
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


async def main_cli_entry():
    """CLI entry point for bootstrap system."""
    
    if len(sys.argv) < 2:
        print('Usage: python -m bootstrap <manifest_file1.yaml> [<manifest_file2.yaml> ...]')
        print('       python -m bootstrap --smoke-test')
        print('       python -m bootstrap --validate <manifest_file.yaml>')
        print('       python -m bootstrap --version')
        sys.exit(1)
    
    if sys.argv[1] == '--version':
        from . import get_version
        print(f"NIREON V4 Bootstrap {get_version()}")
        sys.exit(0)
    
    if sys.argv[1] == '--smoke-test':
        try:
            from .main import smoke_test
            success = await smoke_test()
            print("✓ Smoke test passed" if success else "✗ Smoke test failed")
            sys.exit(0 if success else 1)
        except ImportError:
            print("✗ Smoke test not available - check bootstrap implementation")
            sys.exit(1)
    
    if sys.argv[1] == '--validate':
        if len(sys.argv) < 3:
            print('Error: --validate requires a manifest file')
            sys.exit(1)
        
        try:
            from .main import validate_bootstrap_config
            result = await validate_bootstrap_config([sys.argv[2]])
            print(f"Validation: {'✓ PASSED' if result['valid'] else '✗ FAILED'}")
            
            if result['errors']:
                print("Errors:")
                for error in result['errors']:
                    print(f"  - {error}")
            
            if result['warnings']:
                print("Warnings:")
                for warning in result['warnings']:
                    print(f"  - {warning}")
            
            sys.exit(0 if result['valid'] else 1)
        except ImportError:
            print("✗ Validation not available - check bootstrap implementation")
            sys.exit(1)
    
    # Normal bootstrap execution
    config_paths = [Path(p) for p in sys.argv[1:]]
    
    # Validate config files exist
    for path in config_paths:
        if not path.exists():
            print(f"Error: Configuration file not found: {path}")
            sys.exit(1)
        if not path.suffix.lower() in ['.yaml', '.yml']:
            print(f"Warning: File {path} does not have .yaml/.yml extension")
    
    try:
        from .main import bootstrap_nireon_system
        
        logger.info(f"Starting bootstrap with {len(config_paths)} configuration files")
        
        result = await bootstrap_nireon_system(config_paths, strict_mode=True)
        
        print(f'\n=== Bootstrap Summary (Run ID: {result.run_id}) ===')
        print(f'Success: {result.success}')
        print(f'Components: {result.component_count}')
        print(f'Healthy: {result.healthy_component_count}')
        
        if hasattr(result, 'bootstrap_duration') and result.bootstrap_duration:
            print(f'Duration: {result.bootstrap_duration:.2f}s')
        
        if not result.success:
            print(f'Critical Failures: {result.critical_failure_count}')
            print('\n--- Health Report (excerpt) ---')
            health_report = result.get_health_report()
            # Show first 1000 characters of health report
            if len(health_report) > 1000:
                print(health_report[:1000] + '\n... (truncated)')
            else:
                print(health_report)
        
        sys.exit(0 if result.success else 1)
        
    except KeyboardInterrupt:
        print("\n✗ Bootstrap interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Bootstrap failed with exception: {e}", exc_info=True)
        print(f'\n✗ FATAL ERROR: {e}')
        sys.exit(2)


if __name__ == '__main__':
    try:
        asyncio.run(main_cli_entry())
    except KeyboardInterrupt:
        print("\nBootstrap interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)