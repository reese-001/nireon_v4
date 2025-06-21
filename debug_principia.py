import asyncio
import logging
import sys
import traceback
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('debug_test')

# Constants
ROOT_FRAME_ID = 'F-ROOT-00000000-0000-0000-0000-000000000000'

async def test_math_with_proper_context():
    """
    Main test function that demonstrates the proper setup for math operations.
    
    Key requirements for success:
    1. Bootstrap the system properly
    2. Use the ROOT frame (always available after bootstrap)
    3. Pass frame_id in the signal payload
    4. Ensure all services are properly initialized
    """
    # Setup path
    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))
    
    print('=== Math Agent Test with Proper Context ===')
    
    # Import after path setup
    from bootstrap import bootstrap_nireon_system
    from domain.cognitive_events import CognitiveEvent, LLMRequestPayload
    from domain.context import NireonExecutionContext
    from domain.epistemic_stage import EpistemicStage
    
    # Bootstrap the system
    print('Bootstrapping system...')
    manifest = Path('configs/manifests/standard.yaml')
    boot_res = await bootstrap_nireon_system(
        config_paths=[manifest],
        strict_mode=False
    )
    
    if not boot_res.success:
        print('Bootstrap failed!')
        return
    
    # Get all required services
    registry = boot_res.registry
    
    from domain.ports.mechanism_gateway_port import MechanismGatewayPort
    from domain.ports.event_bus_port import EventBusPort
    from application.services.frame_factory_service import FrameFactoryService
    
    gateway = registry.get_service_instance(MechanismGatewayPort)
    event_bus = registry.get_service_instance(EventBusPort)
    frame_factory = registry.get_service_instance(FrameFactoryService)
    
    # Create execution context
    context = NireonExecutionContext(
        run_id='debug_test',
        component_id='debug_runner',
        component_registry=registry,
        event_bus=event_bus,
        logger=logger
    )
    
    # Verify root frame exists
    root_frame = await frame_factory.get_frame_by_id(context, ROOT_FRAME_ID)
    print(f"Got root frame: {(root_frame.id if root_frame else 'None')}")
    
    if not root_frame:
        print('❌ CRITICAL: Root frame not found. Cannot proceed.')
        return
    
    # Test 1: Direct Math Computation via Gateway
    print('\n=== Test 1: Math Computation ===')
    math_event = CognitiveEvent(
        frame_id=ROOT_FRAME_ID,  # Use the root frame
        owning_agent_id='debug_test',
        service_call_type='MATH_COMPUTE',
        payload={
            'engine': 'sympy',
            'expression': 'x**2',
            'operations': [{'type': 'differentiate', 'variable': 'x'}]
        }
    )
    
    try:
        result = await gateway.process_cognitive_event(math_event, context)
        print(f'✅ Math result: {result}')
    except Exception as e:
        print(f'❌ Math computation failed: {e}')
        traceback.print_exc()
    
    # Test 2: LLM Test
    print('\n=== Test 2: LLM Test ===')
    llm_event = CognitiveEvent.for_llm_ask(
        frame_id=ROOT_FRAME_ID,  # Use the root frame
        owning_agent_id='debug_test',
        prompt='Please explain: The derivative of x² is 2x',
        stage=EpistemicStage.SYNTHESIS,
        role='math_explainer'
    )
    
    try:
        llm_result = await gateway.process_cognitive_event(llm_event, context)
        print(f'✅ LLM result: {llm_result}')
    except Exception as e:
        print(f'❌ LLM call failed: {e}')
        traceback.print_exc()
    
    # Test 3: Full Principia Agent Test via Reactor
    print('\n=== Test 3: Full Principia Agent Test ===')
    
    from signals.core import MathQuerySignal, MathResultSignal
    from domain.ports.reactor_port import ReactorPort
    
    reactor = registry.get_service_instance(ReactorPort)
    
    # Set up result handler
    result_received = asyncio.Event()
    final_result = None
    
    def on_result(payload):
        nonlocal final_result
        print('Received MathResultSignal!')
        final_result = payload
        result_received.set()
    
    event_bus.subscribe(MathResultSignal.__name__, on_result)
    
    # Create signal with frame_id in payload (critical!)
    signal = MathQuerySignal(
        source_node_id='debug_runner',
        natural_language_query='What is the derivative of x squared?',
        expression='x**2',
        operations=[{'type': 'differentiate', 'variable': 'x'}],
        payload={
            'frame_id': ROOT_FRAME_ID  # This is critical for the agent to work
        }
    )
    
    print('Processing math signal through reactor...')
    await reactor.process_signal(signal)
    
    try:
        await asyncio.wait_for(result_received.wait(), timeout=10.0)
        print(f'\n✅ SUCCESS! Got result:')
        if final_result:
            print(f"Explanation: {final_result.get('explanation', 'No explanation')}")
            print(f"Computation: {final_result.get('computation_details', {})}")
    except asyncio.TimeoutError:
        print('\n❌ Timed out waiting for MathResultSignal')
    finally:
        event_bus.unsubscribe(MathResultSignal.__name__, on_result)
    
    print('\n=== All Tests Complete ===')

if __name__ == '__main__':
    asyncio.run(test_math_with_proper_context())