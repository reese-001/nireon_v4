# nireon_v4/bootstrap/__main__.py
from __future__ import absolute_import # Moved to the top

import asyncio
import sys
import logging
from pathlib import Path


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def diagnose_registry(registry):
    """Print detailed information about what's in the registry"""
    print("\n" + "="*80)
    print("REGISTRY DIAGNOSTIC REPORT")
    print("="*80)
    
    # List all component IDs
    all_components = registry.list_components()
    print(f"\nTotal components registered: {len(all_components)}")
    print("\nAll registered component IDs:")
    for i, comp_id in enumerate(sorted(all_components), 1):
        print(f"  {i:3d}. {comp_id}")
    
    # Check for specific services we expect
    print("\n" + "-"*80)
    print("CHECKING FOR EXPECTED SERVICES:")
    print("-"*80)
    
    expected_services = [
        'llm_router_main',
        'LLMPort', 
        'event_bus_memory',
        'EventBusPort',
        'parameter_service_global',
        'ParameterService',
        'frame_factory_service',
        'FrameFactoryService',
        'budget_manager_inmemory',
        'BudgetManagerPort',
        'mechanism_gateway_main',
        'MechanismGatewayPort'
    ]
    
    for service_id in expected_services:
        try:
            component = registry.get(service_id)
            print(f"✓ {service_id:30} -> {type(component).__name__}")
            
            # Check if it has metadata
            try:
                metadata = registry.get_metadata(service_id)
                if metadata:
                    print(f"    Metadata ID: {metadata.id}, Category: {metadata.category}")
            except:
                if hasattr(component, 'metadata'):
                    meta = component.metadata
                    print(f"    Metadata ID: {meta.id}, Category: {meta.category}")
        except Exception as e:
            print(f"✗ {service_id:30} -> NOT FOUND ({str(e)[:50]}...)")
    
    # Check service instances by type
    print("\n" + "-"*80)
    print("CHECKING SERVICE INSTANCES BY TYPE:")
    print("-"*80)
    
    if hasattr(registry, 'get_service_instance'):
        try:
            from domain.ports.llm_port import LLMPort
            from domain.ports.event_bus_port import EventBusPort
            from domain.ports.embedding_port import EmbeddingPort
            from infrastructure.llm.parameter_service import ParameterService
            from application.services.frame_factory_service import FrameFactoryService
            from application.services.budget_manager import BudgetManagerPort
            
            service_types = [
                (LLMPort, "LLMPort"),
                (EventBusPort, "EventBusPort"), 
                (EmbeddingPort, "EmbeddingPort"),
                (ParameterService, "ParameterService"),
                (FrameFactoryService, "FrameFactoryService"),
                (BudgetManagerPort, "BudgetManagerPort")
            ]
            
            for service_type, type_name in service_types:
                try:
                    instance = registry.get_service_instance(service_type)
                    print(f"✓ {type_name:30} -> {type(instance).__name__}")
                    if hasattr(instance, 'component_id'):
                        print(f"    Component ID: {instance.component_id}")
                    elif hasattr(instance, 'metadata') and hasattr(instance.metadata, 'id'):
                        print(f"    Metadata ID: {instance.metadata.id}")
                except Exception as e:
                    print(f"✗ {type_name:30} -> NOT FOUND ({str(e)[:50]}...)")
        except ImportError as e:
            print(f"Could not import service types for type checking: {e}")
    
    # Look for components with 'llm' in their ID
    print("\n" + "-"*80)
    print("COMPONENTS WITH 'llm' IN ID:")
    print("-"*80)
    
    llm_components = [cid for cid in all_components if 'llm' in cid.lower()]
    if llm_components:
        for comp_id in llm_components:
            try:
                comp = registry.get(comp_id)
                print(f"  {comp_id} -> {type(comp).__name__}")
            except:
                pass
    else:
        print("  No components found with 'llm' in their ID")
    
    # Look for components with 'router' in their ID
    print("\n" + "-"*80)
    print("COMPONENTS WITH 'router' IN ID:")
    print("-"*80)
    
    router_components = [cid for cid in all_components if 'router' in cid.lower()]
    if router_components:
        for comp_id in router_components:
            try:
                comp = registry.get(comp_id)
                print(f"  {comp_id} -> {type(comp).__name__}")
            except:
                pass
    else:
        print("  No components found with 'router' in their ID")
    
    print("\n" + "="*80)
    print("END OF DIAGNOSTIC REPORT")
    print("="*80 + "\n")


async def main_cli_entry():
    if len(sys.argv) < 2:
        print('Usage: python -m bootstrap <manifest_file1.yaml> [<manifest_file2.yaml> ...]')
        print('       python -m bootstrap --smoke-test')
        print('       python -m bootstrap --validate <manifest_file.yaml>')
        print('       python -m bootstrap --diagnose <manifest_file.yaml>')
        print('       python -m bootstrap --version')
        sys.exit(1)

    if sys.argv[1] == '--version':
        from . import __version__ # Use __version__ directly
        print(f'NIREON V4 Bootstrap {__version__}')
        # Ensure get_version() is removed if it was defined elsewhere and no longer needed.
        sys.exit(0)

    if sys.argv[1] == '--smoke-test':
        try:
            from .core.main import smoke_test
            success = await smoke_test()
            print('✓ Smoke test passed' if success else '✗ Smoke test failed')
            sys.exit(0 if success else 1)
        except ImportError:
            logger.error("Smoke test function not found or import error.", exc_info=True)
            print('✗ Smoke test not available - check bootstrap implementation')
            sys.exit(1)
        except Exception as e:
            logger.error(f"Error during smoke test: {e}", exc_info=True)
            print(f'✗ Smoke test encountered an error: {e}')
            sys.exit(1)

    if sys.argv[1] == '--validate':
        if len(sys.argv) < 3:
            print('Error: --validate requires a manifest file')
            sys.exit(1)
        try:
            from .core.main import validate_bootstrap_config
            result = await validate_bootstrap_config([sys.argv[2]])
            print(f"Validation: {('✓ PASSED' if result['valid'] else '✗ FAILED')}")
            if result['errors']:
                print('Errors:')
                for error in result['errors']:
                    print(f'  - {error}')
            if result['warnings']:
                print('Warnings:')
                for warning in result['warnings']:
                    print(f'  - {warning}')
            sys.exit(0 if result['valid'] else 1)
        except ImportError:
            logger.error("validate_bootstrap_config function not found or import error.", exc_info=True)
            print('✗ Validation not available - check bootstrap implementation')
            sys.exit(1)
        except Exception as e:
            logger.error(f"Error during validation: {e}", exc_info=True)
            print(f'✗ Validation encountered an error: {e}')
            sys.exit(1)

    if sys.argv[1] == '--diagnose':
        if len(sys.argv) < 3:
            print('Error: --diagnose requires a manifest file')
            sys.exit(1)
        
        print('Running bootstrap with diagnostic mode...')
        config_paths = [Path(sys.argv[2])]
        
        # Validate the path
        for path in config_paths:
            if not path.exists():
                print(f'Error: Configuration file not found: {path}')
                sys.exit(1)
        
        try:
            from .core.main import bootstrap_nireon_system
            logger.info(f'Starting diagnostic bootstrap with {len(config_paths)} configuration file(s)')
            
            # Run bootstrap in non-strict mode for diagnostic
            result = await bootstrap_nireon_system(config_paths, strict_mode=False)
            
            # Print standard bootstrap summary
            print(f'\n=== Bootstrap Summary (Run ID: {result.run_id}) ===')
            print(f'Success: {result.success}')
            print(f'Components: {result.component_count}')
            print(f'Healthy: {result.healthy_component_count}')
            if hasattr(result, 'bootstrap_duration') and result.bootstrap_duration is not None:
                print(f'Duration: {result.bootstrap_duration:.2f}s')
            
            # Run the diagnostic
            if result.registry:
                diagnose_registry(result.registry)
            else:
                print('\n✗ No registry available for diagnostic')
                sys.exit(1)
            
            # If bootstrap failed, also show health report
            if not result.success:
                print('\n--- Health Report ---')
                health_report = result.get_health_report()
                print(health_report)
            
            sys.exit(0 if result.success else 1)
            
        except Exception as e:
            logger.error(f'Diagnostic bootstrap failed: {e}', exc_info=True)
            print(f'\n✗ DIAGNOSTIC ERROR: {e}')
            sys.exit(2)

    # Normal bootstrap mode
    config_paths = [Path(p) for p in sys.argv[1:]]
    for path in config_paths:
        if not path.exists():
            print(f'Error: Configuration file not found: {path}')
            sys.exit(1)
        if not path.suffix.lower() in ['.yaml', '.yml']:
            print(f'Warning: File {path} does not have .yaml/.yml extension, processing anyway.')

    try:
        from .core.main import bootstrap_nireon_system
        logger.info(f'Starting bootstrap with {len(config_paths)} configuration files: {config_paths}')
        result = await bootstrap_nireon_system(config_paths, strict_mode=True)
        
        print(f'\n=== Bootstrap Summary (Run ID: {result.run_id}) ===')
        print(f'Success: {result.success}')
        print(f'Components: {result.component_count}')
        print(f'Healthy: {result.healthy_component_count}')
        if hasattr(result, 'bootstrap_duration') and result.bootstrap_duration is not None:
            print(f'Duration: {result.bootstrap_duration:.2f}s')
        
        if not result.success:
            print(f'Critical Failures: {result.critical_failure_count}')
            print('\n--- Health Report (excerpt) ---')
            health_report = result.get_health_report() 
            if len(health_report) > 2000: 
                print(health_report[:2000] + '\n... (report truncated for brevity)')
            else:
                print(health_report)
        sys.exit(0 if result.success else 1)
    except KeyboardInterrupt:
        print('\n✗ Bootstrap interrupted by user')
        sys.exit(130) 
    except Exception as e:
        logger.error(f'Bootstrap failed with an unhandled exception: {e}', exc_info=True)
        print(f'\n✗ FATAL BOOTSTRAP ERROR: {e}')
        sys.exit(2) 

if __name__ == '__main__':
    try:
        asyncio.run(main_cli_entry())
    except KeyboardInterrupt:
        print('\nBootstrap process interrupted by user (top-level).')
        sys.exit(130)
    except Exception as e:
        print(f'Fatal error during bootstrap execution: {e}')
        logger.critical(f'Unhandled fatal error in __main__: {e}', exc_info=True)
        sys.exit(1)