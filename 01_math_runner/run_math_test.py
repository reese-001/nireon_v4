from __future__ import annotations
import argparse
import asyncio
import logging
import sys
import time
import json
from pathlib import Path
from typing import Optional, Dict, Any, List

# Project root finding
def _find_project_root(markers: list[str]=['bootstrap', 'domain', 'core', 'configs']) -> Path | None:
    here = Path(__file__).resolve().parent
    for candidate in [here, *here.parents]:
        if all(((candidate / m).is_dir() for m in markers)):
            return candidate
    return None

PROJECT_ROOT = _find_project_root()
if PROJECT_ROOT is None:
    print('ERROR: Could not determine the NIREON V4 project root.')
    sys.exit(1)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bootstrap import bootstrap_nireon_system
from signals.core import MathQuerySignal, MathResultSignal
from core.registry.component_registry import ComponentRegistry
from domain.ports.reactor_port import ReactorPort
from domain.ports.event_bus_port import EventBusPort
from domain.context import NireonExecutionContext
from application.services.frame_factory_service import FrameFactoryService

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)-20s - %(levelname)-8s - %(message)s'
)
logger = logging.getLogger('MathTestRunner')

# Constants
ROOT_FRAME_ID = 'F-ROOT-00000000-0000-0000-0000-000000000000'

async def wait_for_math_result(event_bus: EventBusPort, timeout: float) -> Optional[Dict[str, Any]]:
    """Wait for a MathResultSignal with timeout."""
    future = asyncio.get_running_loop().create_future()
    
    def callback(payload: Dict[str, Any]):
        if not future.done():
            logger.info('[wait_for_math_result] Received final MathResultSignal!')
            signal_payload = payload.get('payload', payload)
            future.set_result(signal_payload)
    
    signal_name = MathResultSignal.__name__
    event_bus.subscribe(signal_name, callback)
    
    try:
        logger.info(f"[wait_for_math_result] Waiting for '{signal_name}' for up to {timeout} seconds...")
        return await asyncio.wait_for(future, timeout)
    except asyncio.TimeoutError:
        logger.error(f"[wait_for_math_result] Timed out waiting for '{signal_name}'.")
        return None
    finally:
        event_bus.unsubscribe(signal_name, callback)

async def verify_root_frame(frame_factory: FrameFactoryService, context: NireonExecutionContext) -> bool:
    """Verify that the root frame exists and is active."""
    try:
        root_frame = await frame_factory.get_frame_by_id(context, ROOT_FRAME_ID)
        if root_frame:
            logger.info(f"Root frame verified: ID={root_frame.id}, Status={root_frame.status}")
            return root_frame.status == 'active'
        else:
            logger.error("Root frame not found!")
            return False
    except Exception as e:
        logger.error(f"Error verifying root frame: {e}")
        return False

async def main(query: str, expression: str, ops_filepath: str, timeout: float):
    logger.info('--- NIREON V4 Math Agent Test ---')
    logger.info('Bootstrapping NIREON V4 system...')
    
    # Bootstrap the system
    manifest = PROJECT_ROOT / 'configs' / 'manifests' / 'standard.yaml'
    boot_res = await bootstrap_nireon_system(
        config_paths=[manifest],
        strict_mode=False
    )
    
    if not boot_res.success:
        logger.error('Bootstrap failed. Aborting test.')
        print(boot_res.get_health_report())
        sys.exit(1)
    
    logger.info('Bootstrap complete. System is online.')
    
    # Get required services
    registry: ComponentRegistry = boot_res.registry
    reactor: ReactorPort = registry.get_service_instance(ReactorPort)
    event_bus: EventBusPort = registry.get_service_instance(EventBusPort)
    frame_factory: FrameFactoryService = registry.get_service_instance(FrameFactoryService)
    
    # Create execution context
    context = NireonExecutionContext(
        run_id='math_test_run',
        component_id='math_test_runner',
        component_registry=registry,
        event_bus=event_bus,
        logger=logger
    )
    
    # Verify root frame exists
    if not await verify_root_frame(frame_factory, context):
        logger.error('Root frame verification failed. Cannot proceed.')
        sys.exit(1)
    
    # Load operations file
    try:
        ops_path = Path(ops_filepath)
        if not ops_path.exists():
            logger.error(f'Operations file not found at: {ops_path.resolve()}')
            sys.exit(1)
        
        with open(ops_path, 'r') as f:
            operations = json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f'Invalid JSON in operations file {ops_filepath}: {e}')
        sys.exit(1)
    except Exception as e:
        logger.error(f'Error reading operations file {ops_filepath}: {e}')
        sys.exit(1)
    
    # Create math signal with root frame ID in payload
    math_signal = MathQuerySignal(
        source_node_id='MathTestRunner',
        natural_language_query=query,
        expression=expression,
        operations=operations,
        payload={
            'metadata': 'This signal was generated by the test runner.',
            'frame_id': ROOT_FRAME_ID  # Critical: include frame_id in payload
        }
    )
    
    logger.info(f'Constructed MathQuerySignal with data:\n{math_signal.model_dump_json(indent=2)}')
    
    # Set up result waiter
    waiter_task = asyncio.create_task(wait_for_math_result(event_bus, timeout=timeout))
    
    # Process the signal
    logger.info('Processing signal through the Reactor...')
    await reactor.process_signal(math_signal)
    
    # Wait for result
    final_result = await waiter_task
    
    # Display results
    logger.info('\n--- TEST COMPLETE ---')
    if final_result:
        logger.info('✅ Agent successfully completed the task and emitted a result signal.')
        print('\n' + '=' * 80)
        print('Final Explanation from Principia Agent:')
        print('=' * 80)
        print(final_result.get('explanation', 'No explanation found.'))
        print('\nComputation Details:')
        print(json.dumps(final_result.get('computation_details', {}), indent=2))
        print('=' * 80)
    else:
        logger.error('❌ Test failed. The agent did not produce a final result signal within the timeout.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Test the Principia Math Agent.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples of usage:

1. Simple derivative:
   python run_math_test.py --query "What is the derivative of x^6?" --expr "x**6" --ops derivative_ops.json

2. Integration:
   python run_math_test.py --query "Integrate x^6 from 0 to 1" --expr "x**6" --ops integral_ops.json

3. Complex example (default):
   python run_math_test.py --ops complex_ops.json
        '''
    )
    
    parser.add_argument('--query', default='What is the integral of x^6 * sin(x) from 0 to pi?',
                        help='The natural language question.')
    parser.add_argument('--expr', default='x**6 * sin(x)',
                        help='The mathematical expression for SymPy.')
    parser.add_argument('--ops', required=True,
                        help='Path to the JSON file containing the operations to perform.')
    parser.add_argument('--timeout', type=float, default=60.0,
                        help='Timeout in seconds to wait for the final result.')
    
    args = parser.parse_args()
    
    asyncio.run(main(args.query, args.expr, args.ops, args.timeout))