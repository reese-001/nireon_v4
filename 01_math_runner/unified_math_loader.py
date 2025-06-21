#!/usr/bin/env python3
"""
Unified Math Loader for NIREON V4
This is the primary entry point for running math operations through the Principia Agent.

Key design principles:
1. Always use the ROOT frame (ID: F-ROOT-00000000-0000-0000-0000-000000000000)
2. Pass frame_id in the signal payload
3. Verify all required services are available
4. Provide clear error messages and debugging info
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List

# Constants
ROOT_FRAME_ID = 'F-ROOT-00000000-0000-0000-0000-000000000000'
DEFAULT_TIMEOUT = 30.0

# Configure logging
def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)-20s - %(levelname)-8s - %(message)s'
    )
    return logging.getLogger('nireon_math')

# Find project root
def find_project_root() -> Path:
    """Find the NIREON project root by looking for marker directories."""
    markers = ['bootstrap', 'domain', 'core', 'configs']
    here = Path(__file__).resolve().parent
    
    for candidate in [here, *here.parents]:
        if all((candidate / m).is_dir() for m in markers):
            return candidate
    
    raise RuntimeError('Could not determine NIREON project root')

# Bootstrap the system
async def bootstrap_system(logger: logging.Logger) -> Dict[str, Any]:
    """Bootstrap the NIREON system and return essential services."""
    from bootstrap import bootstrap_nireon_system
    
    logger.info('Bootstrapping NIREON V4 system...')
    
    manifest = Path('configs/manifests/standard.yaml')
    boot_res = await bootstrap_nireon_system(
        config_paths=[manifest],
        strict_mode=False
    )
    
    if not boot_res.success:
        logger.error('Bootstrap failed!')
        if hasattr(boot_res, 'get_health_report'):
            print(boot_res.get_health_report())
        raise RuntimeError('System bootstrap failed')
    
    logger.info('Bootstrap complete. System is online.')
    
    # Get required services
    registry = boot_res.registry
    
    from domain.ports.reactor_port import ReactorPort
    from domain.ports.event_bus_port import EventBusPort
    from application.services.frame_factory_service import FrameFactoryService
    from domain.ports.mechanism_gateway_port import MechanismGatewayPort
    
    services = {
        'registry': registry,
        'reactor': registry.get_service_instance(ReactorPort),
        'event_bus': registry.get_service_instance(EventBusPort),
        'frame_factory': registry.get_service_instance(FrameFactoryService),
        'gateway': registry.get_service_instance(MechanismGatewayPort)
    }
    
    # Verify all services are available
    for name, service in services.items():
        if service is None:
            raise RuntimeError(f'Required service {name} not available')
    
    return services

# Verify root frame
async def verify_root_frame(services: Dict[str, Any], logger: logging.Logger) -> bool:
    """Verify the root frame exists and is active."""
    from domain.context import NireonExecutionContext
    
    context = NireonExecutionContext(
        run_id='verification',
        component_id='math_loader',
        component_registry=services['registry'],
        event_bus=services['event_bus'],
        logger=logger
    )
    
    frame_factory = services['frame_factory']
    root_frame = await frame_factory.get_frame_by_id(context, ROOT_FRAME_ID)
    
    if not root_frame:
        logger.error(f'Root frame {ROOT_FRAME_ID} not found!')
        return False
    
    if root_frame.status != 'active':
        logger.warning(f'Root frame status is {root_frame.status}, expected active')
    
    logger.info(f'Root frame verified: {root_frame.id} (Status: {root_frame.status})')
    return True

# Process math query
async def process_math_query(
    services: Dict[str, Any],
    query: str,
    expression: str,
    operations: List[Dict[str, Any]],
    timeout: float,
    logger: logging.Logger
) -> Optional[Dict[str, Any]]:
    """Process a math query through the Principia Agent."""
    from signals.core import MathQuerySignal, MathResultSignal
    from domain.context import NireonExecutionContext
    
    # Create execution context
    context = NireonExecutionContext(
        run_id='math_query',
        component_id='math_loader',
        component_registry=services['registry'],
        event_bus=services['event_bus'],
        logger=logger
    )
    
    # Create result handler
    result_future = asyncio.Future()
    
    def on_result(payload: Dict[str, Any]):
        if not result_future.done():
            logger.info('Received MathResultSignal')
            # The payload IS the signal data - don't try to extract nested 'payload'
            result_future.set_result(payload)
    
    # Subscribe to results
    event_bus = services['event_bus']
    event_bus.subscribe(MathResultSignal.__name__, on_result)
    
    try:
        # Create signal with frame_id in payload
        signal = MathQuerySignal(
            source_node_id='math_loader',
            natural_language_query=query,
            expression=expression,
            operations=operations,
            payload={
                'frame_id': ROOT_FRAME_ID,  # Critical!
                'metadata': 'Query from unified math loader'
            }
        )
        
        logger.info('Processing math signal...')
        logger.debug(f'Signal data: {signal.model_dump_json(indent=2)}')
        
        # Process through reactor
        reactor = services['reactor']
        await reactor.process_signal(signal)
        
        # Wait for result
        result = await asyncio.wait_for(result_future, timeout=timeout)
        return result
        
    except asyncio.TimeoutError:
        logger.error(f'Timed out after {timeout}s waiting for result')
        return None
    finally:
        event_bus.unsubscribe(MathResultSignal.__name__, on_result)

# Direct gateway test (optional)
async def test_gateway_directly(
    services: Dict[str, Any],
    expression: str,
    operations: List[Dict[str, Any]],
    logger: logging.Logger
) -> Optional[Dict[str, Any]]:
    """Test the math computation directly through the gateway (bypassing reactor)."""
    from domain.cognitive_events import CognitiveEvent
    from domain.context import NireonExecutionContext
    
    context = NireonExecutionContext(
        run_id='gateway_test',
        component_id='math_loader',
        component_registry=services['registry'],
        event_bus=services['event_bus'],
        logger=logger
    )
    
    event = CognitiveEvent(
        frame_id=ROOT_FRAME_ID,
        owning_agent_id='math_loader',
        service_call_type='MATH_COMPUTE',
        payload={
            'engine': 'sympy',
            'expression': expression,
            'operations': operations
        }
    )
    
    try:
        gateway = services['gateway']
        result = await gateway.process_cognitive_event(event, context)
        return result
    except Exception as e:
        logger.error(f'Gateway test failed: {e}')
        return None

# Main entry point
async def main():
    parser = argparse.ArgumentParser(
        description='Unified Math Loader for NIREON V4',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:

1. Simple derivative:
   python unified_math_loader.py --query "What is the derivative of x^2?" --expr "x**2" --ops '[{"type": "differentiate", "variable": "x"}]'

2. Integration with limits:
   python unified_math_loader.py --query "Integrate x^2 from 0 to 1" --expr "x**2" --ops '[{"type": "integrate", "variable": "x", "limits": [0, 1]}]'

3. Load operations from file:
   python unified_math_loader.py --query "Complex calculation" --expr "sin(x) * x**2" --ops-file operations.json

4. Test gateway directly (bypass reactor):
   python unified_math_loader.py --expr "x**3" --ops '[{"type": "differentiate", "variable": "x"}]' --gateway-only
        '''
    )
    
    parser.add_argument('--query', default='What is the derivative of x squared?',
                        help='Natural language query')
    parser.add_argument('--expr', default='x**2',
                        help='Mathematical expression')
    parser.add_argument('--ops', type=str,
                        help='Operations as JSON string')
    parser.add_argument('--ops-file', type=str,
                        help='Path to operations JSON file')
    parser.add_argument('--timeout', type=float, default=DEFAULT_TIMEOUT,
                        help='Timeout in seconds')
    parser.add_argument('--gateway-only', action='store_true',
                        help='Test gateway directly without reactor')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup
    logger = setup_logging(args.verbose)
    
    # Set up Python path
    try:
        project_root = find_project_root()
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        logger.debug(f'Project root: {project_root}')
    except RuntimeError as e:
        logger.error(str(e))
        sys.exit(1)
    
    # Load operations
    operations = []
    if args.ops:
        try:
            operations = json.loads(args.ops)
        except json.JSONDecodeError as e:
            logger.error(f'Invalid JSON in --ops: {e}')
            sys.exit(1)
    elif args.ops_file:
        try:
            with open(args.ops_file, 'r') as f:
                operations = json.load(f)
        except Exception as e:
            logger.error(f'Error loading operations file: {e}')
            sys.exit(1)
    else:
        # Default operation
        operations = [{'type': 'differentiate', 'variable': 'x'}]
    
    try:
        # Bootstrap system
        services = await bootstrap_system(logger)
        
        # Verify root frame
        if not await verify_root_frame(services, logger):
            logger.error('Root frame verification failed')
            sys.exit(1)
        
        # Run test
        if args.gateway_only:
            logger.info('=== Gateway Direct Test ===')
            result = await test_gateway_directly(services, args.expr, operations, logger)
            if result:
                print(f"\n✅ Gateway Result: {json.dumps(result, indent=2)}")
            else:
                print("\n❌ Gateway test failed")
        else:
            logger.info('=== Full Agent Test ===')
            result = await process_math_query(
                services, args.query, args.expr, operations, args.timeout, logger
            )
            
            if result:
                print("\n✅ SUCCESS!")
                print("=" * 80)
                print("Explanation:")
                print(result.get('explanation', 'No explanation provided'))
                print("\nComputation Details:")
                print(json.dumps(result.get('computation_details', {}), indent=2))
                print("=" * 80)
            else:
                print("\n❌ No result received")
                
    except Exception as e:
        logger.error(f'Fatal error: {e}', exc_info=True)
        sys.exit(1)

if __name__ == '__main__':
    asyncio.run(main())