import asyncio
import sys
import json
import logging
from pathlib import Path

# Setup path
sys.path.insert(0, str(Path(__file__).parent))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('test_with_frame')

# Constants
ROOT_FRAME_ID = 'F-ROOT-00000000-0000-0000-0000-000000000000'

async def test_with_frame():
    from bootstrap import bootstrap_nireon_system
    from signals.core import MathQuerySignal, MathResultSignal
    from core.registry.component_registry import ComponentRegistry
    from domain.ports.reactor_port import ReactorPort
    from domain.ports.event_bus_port import EventBusPort
    from application.services.frame_factory_service import FrameFactoryService
    from domain.context import NireonExecutionContext
    
    print('Bootstrapping system...')
    manifest = Path('configs/manifests/standard.yaml')
    boot_res = await bootstrap_nireon_system(
        config_paths=[manifest],
        strict_mode=False
    )
    
    if not boot_res.success:
        print('Bootstrap failed!')
        return
    
    print('Bootstrap complete!')
    
    # Get services
    registry = boot_res.registry
    reactor = registry.get_service_instance(ReactorPort)
    event_bus = registry.get_service_instance(EventBusPort)
    frame_factory = registry.get_service_instance(FrameFactoryService)
    
    # Create execution context
    context = NireonExecutionContext(
        run_id='test_run',
        component_id='test_runner',
        component_registry=registry,
        event_bus=event_bus,
        logger=logger
    )
    
    # Verify root frame exists (it should be created during bootstrap)
    print(f'Verifying root frame with ID: {ROOT_FRAME_ID}')
    root_frame = await frame_factory.get_frame_by_id(context, ROOT_FRAME_ID)
    
    if not root_frame:
        print('❌ Root frame not found! This should not happen after bootstrap.')
        return
    
    print(f'✅ Root frame found: {root_frame.id} (Status: {root_frame.status})')
    
    # Optional: Create a child frame for this specific test (but still use root frame for processing)
    test_frame = await frame_factory.create_frame(
        context,
        name='math_test_frame',
        owner_agent_id='test_runner',
        description='Frame for math operation testing',
        parent_frame_id=ROOT_FRAME_ID,  # Make it a child of root
        epistemic_goals=['TEST_MATH_OPERATIONS'],
        frame_type='epistemic'
    )
    print(f'Created test frame: {test_frame.id} (parent: {test_frame.parent_frame_id})')
    
    # Create signal - use ROOT_FRAME_ID for compatibility
    signal = MathQuerySignal(
        source_node_id='test_runner',
        natural_language_query='What is the derivative of x squared?',
        expression='x**2',
        operations=[{'type': 'differentiate', 'variable': 'x'}],
        payload={
            'metadata': 'Test with proper frame',
            'frame_id': ROOT_FRAME_ID  # Use root frame ID for processing
        }
    )
    
    print(f'Processing signal with root frame {ROOT_FRAME_ID}...')
    
    # Set up result handler
    result_future = asyncio.Future()
    
    def on_result(payload):
        if not result_future.done():
            print('Received MathResultSignal!')
            result_future.set_result(payload)
    
    event_bus.subscribe(MathResultSignal.__name__, on_result)
    
    # Process signal
    await reactor.process_signal(signal)
    
    try:
        result = await asyncio.wait_for(result_future, timeout=30.0)
        print('\n✅ SUCCESS!')
        print(f"Explanation: {result.get('explanation', 'No explanation')}")
        print(f"Details: {json.dumps(result.get('computation_details', {}), indent=2)}")
    except asyncio.TimeoutError:
        print('\n❌ Timed out waiting for result')
        print('The math computation likely succeeded, but the agent failed to publish the result.')
    finally:
        event_bus.unsubscribe(MathResultSignal.__name__, on_result)

if __name__ == '__main__':
    asyncio.run(test_with_frame())