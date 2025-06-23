#!/usr/bin/env python3
"""
Check if ReactorSetupPhase is working properly
"""
import asyncio
import sys
from pathlib import Path

# Find project root
def _find_project_root(markers=['bootstrap', 'domain', 'core', 'configs']):
    here = Path(__file__).resolve().parent
    for candidate in [here, *here.parents]:
        if all((candidate / m).is_dir() for m in markers):
            return candidate
    return None

PROJECT_ROOT = _find_project_root()
if PROJECT_ROOT is None:
    print('ERROR: Could not determine the NIREON V4 project root.')
    sys.exit(1)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Set up logging to see what's happening
import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)-30s - %(levelname)-8s - %(message)s')

# Patch the ReactorSetupPhase to add debugging
print("=== PATCHING REACTOR SETUP PHASE FOR DEBUGGING ===\n")

try:
    from bootstrap.phases.reactor_setup_phase import ReactorSetupPhase
    original_execute = ReactorSetupPhase.execute
    
    async def debug_execute(self, context):
        print("\n🔍 DEBUG: ReactorSetupPhase.execute() called!")
        print(f"   Context type: {type(context)}")
        print(f"   Has event_bus: {hasattr(context, 'event_bus')}")
        if hasattr(context, 'event_bus'):
            print(f"   Event bus type: {type(context.event_bus)}")
        
        # Call original
        result = await original_execute(self, context)
        
        print(f"\n🔍 DEBUG: ReactorSetupPhase.execute() completed!")
        print(f"   Result success: {result.success if hasattr(result, 'success') else 'N/A'}")
        print(f"   Result message: {result.message if hasattr(result, 'message') else 'N/A'}")
        
        return result
    
    ReactorSetupPhase.execute = debug_execute
    print("✅ Successfully patched ReactorSetupPhase for debugging\n")
    
except Exception as e:
    print(f"❌ Failed to patch ReactorSetupPhase: {e}\n")

# Now run a minimal bootstrap to see what happens
async def test_reactor_setup():
    print("=== RUNNING MINIMAL BOOTSTRAP TEST ===\n")
    
    from bootstrap import bootstrap_nireon_system
    
    # Use the standard manifest
    manifest_path = PROJECT_ROOT / 'configs' / 'manifests' / 'standard.yaml'
    
    print(f"Using manifest: {manifest_path}")
    print("Starting bootstrap...\n")
    
    result = await bootstrap_nireon_system(
        config_paths=[manifest_path],
        strict_mode=False
    )
    
    print(f"\n=== BOOTSTRAP RESULT ===")
    print(f"Success: {result.success}")
    print(f"Component count: {result.component_count}")
    
    # Check if reactor is listening
    if result.success:
        registry = result.registry
        event_bus = registry.get_service_instance('domain.ports.event_bus_port.EventBusPort')
        
        print(f"\n=== CHECKING EVENT BUS SUBSCRIPTIONS ===")
        if hasattr(event_bus, '_subscribers'):
            print(f"Event bus has subscribers attribute")
            for signal_type, subscribers in event_bus._subscribers.items():
                print(f"  {signal_type}: {len(subscribers)} subscriber(s)")
        else:
            print("Cannot inspect event bus subscribers")

if __name__ == '__main__':
    asyncio.run(test_reactor_setup())