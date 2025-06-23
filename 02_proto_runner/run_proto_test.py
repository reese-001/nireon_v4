# nireon_v4\02_proto_runner\run_proto_test.py
from __future__ import annotations
import argparse
import asyncio
import logging
import sys
import json
from pathlib import Path
from typing import Optional, Dict, Any

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
from signals.core import ProtoTaskSignal, ProtoResultSignal, ProtoErrorSignal, MathProtoResultSignal
from domain.ports.event_bus_port import EventBusPort
from domain.context import NireonExecutionContext
from core.base_component import NireonBaseComponent

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)-20s - %(levelname)-8s - %(message)s')
logger = logging.getLogger('ProtoTestRunner')

async def wait_for_proto_result(event_bus: EventBusPort, proto_id: str, timeout: float) -> Optional[Dict[str, Any]]:
    future = asyncio.get_running_loop().create_future()

    def callback(payload: Any):
        # LOGGING: Log every time the callback is triggered
        logger.info(f"[wait_for_proto_result] SUBSCRIBER CALLBACK TRIGGERED. Payload type: {type(payload).__name__}")
        
        signal_payload = None
        if hasattr(payload, 'signal_type'):
            signal_payload = payload
        elif isinstance(payload, dict):
            signal_payload = payload.get('payload', payload)

        if signal_payload is None:
            logger.warning("[wait_for_proto_result] Received a payload with no identifiable signal data.")
            return

        received_proto_id = getattr(signal_payload, 'proto_block_id', None) or (signal_payload.get('proto_block_id') if isinstance(signal_payload, dict) else None)
        signal_type = getattr(signal_payload, 'signal_type', 'unknown')

        # LOGGING: Log details of the received signal
        logger.info(f"[wait_for_proto_result] Received signal of type '{signal_type}' for proto_id '{received_proto_id}'.")

        if received_proto_id == proto_id and (not future.done()):
            logger.info(f"[wait_for_proto_result] MATCH FOUND! Received corresponding result/error signal for Proto ID '{proto_id}'!")
            if hasattr(signal_payload, 'model_dump'):
                future.set_result(signal_payload.model_dump())
            else:
                future.set_result(signal_payload)
    
    event_bus.subscribe(ProtoResultSignal.__name__, callback)
    event_bus.subscribe(MathProtoResultSignal.__name__, callback)
    event_bus.subscribe(ProtoErrorSignal.__name__, callback)

    try:
        logger.info(f"[wait_for_proto_result] Waiting for result for Proto ID '{proto_id}' for up to {timeout}s...")
        return await asyncio.wait_for(future, timeout)
    except asyncio.TimeoutError:
        logger.error(f"[wait_for_proto_result] Timed out waiting for result for Proto ID '{proto_id}'.")
        return None
    finally:
        event_bus.unsubscribe(ProtoResultSignal.__name__, callback)
        event_bus.unsubscribe(MathProtoResultSignal.__name__, callback)
        event_bus.unsubscribe(ProtoErrorSignal.__name__, callback)

async def main(proto_filepath: str, timeout: float, nl_request: Optional[str]=None):
    logger.info('--- NIREON V4 Proto Engine Test ---')
    logger.info('Bootstrapping NIREON V4 system...')
    manifest = PROJECT_ROOT / 'configs' / 'manifests' / 'standard.yaml'
    boot_res = await bootstrap_nireon_system(config_paths=[manifest], strict_mode=False)
    if not boot_res.success:
        logger.error('Bootstrap failed. Aborting test.')
        print(boot_res.get_health_report())
        sys.exit(1)

    logger.info('Bootstrap complete. System is online.')
    registry = boot_res.registry
    event_bus = registry.get_service_instance(EventBusPort)
    context = NireonExecutionContext(run_id='proto_test_run', component_id='proto_test_runner', component_registry=registry, event_bus=event_bus, logger=logger)

    proto_block = None
    if nl_request:
        logger.info(f'--- Testing AI-Authored Proto Workflow ---')
        logger.info(f"Using Natural Language Request: '{nl_request}'")
        proto_generator = registry.get('proto_generator_main')
        if not proto_generator:
            logger.error('ProtoGenerator not found in registry. Cannot process natural language request.')
            sys.exit(1)

        frame_factory = registry.get_service_instance('application.services.frame_factory_service.FrameFactoryService')
        frame = await frame_factory.create_frame(context, name='proto_gen_test_frame', owner_agent_id='test_runner', description='Frame for NL to Proto test')
        gen_context = context.with_metadata(current_frame_id=frame.id)

        logger.info('Calling ProtoGenerator to generate Proto YAML from NL request...')
        generator_result = await proto_generator.process({'natural_language_request': nl_request}, gen_context)

        if not generator_result.success:
            logger.error(f'ProtoGenerator failed: {generator_result.message}')
            if generator_result.output_data:
                logger.error(f"Raw LLM response:\n{generator_result.output_data.get('raw_response')}")
            sys.exit(1)

        proto_block = generator_result.output_data
        logger.info('✅ ProtoGenerator successfully created a Proto block.')
        logger.info(f"Generated Proto Block ID: {proto_block.get('id')}")
        logger.info(f"Generated Dialect (eidos): {proto_block.get('eidos')}")
    else:
        logger.info(f'--- Testing Human-Authored Proto Workflow ---')
        proto_path = Path(proto_filepath)
        if not proto_path.exists():
            logger.error(f'Proto file not found at: {proto_path.resolve()}')
            sys.exit(1)
        with open(proto_path, 'r') as f:
            from yaml import safe_load
            proto_block = safe_load(f)

    if not proto_block:
        logger.error('Could not load or generate a Proto block.')
        sys.exit(1)
    
    proto_id_to_track = proto_block.get('id')
    logger.info(f'Proceeding to execute Proto block with ID: {proto_id_to_track}')
    
    if not nl_request:
        frame_factory = registry.get_service_instance('application.services.frame_factory_service.FrameFactoryService')
        task_frame = await frame_factory.create_frame(context, name=f"proto_exec_{proto_block['id']}", owner_agent_id='test_runner', description=f"Frame for executing {proto_block['id']}")
        
        proto_task_signal = ProtoTaskSignal(
            source_node_id='ProtoTestRunner',
            proto_block=proto_block,
            dialect=proto_block.get('eidos', 'unknown'),
            context_tags={'frame_id': task_frame.id}
        )
        logger.info(f"Publishing ProtoTaskSignal for block '{proto_block['id']}'...")
        await asyncio.to_thread(event_bus.publish, ProtoTaskSignal.__name__, proto_task_signal)

    waiter_task = asyncio.create_task(wait_for_proto_result(event_bus, proto_id_to_track, timeout))
    
    final_result = await waiter_task

    logger.info('\n--- TEST COMPLETE ---')
    if final_result:
        if final_result.get('success', False):
            logger.info('✅ ProtoEngine successfully executed the task.')
            print('\n' + '=' * 80)
            print('Execution Result:')
            print('=' * 80)
            print(json.dumps(final_result.get('result'), indent=2))
            print('\nArtifacts:')
            print(json.dumps(final_result.get('artifacts', []), indent=2))
            print(f"\nExecution Time: {final_result.get('execution_time_sec', 0):.2f}s")
            print('=' * 80)
        else:
            logger.error('❌ ProtoEngine reported an execution failure.')
            print('\n' + '=' * 80)
            print('Failure Details:')
            print('=' * 80)
            print(f"Error Type: {final_result.get('error_type')}")
            print(f"Message: {final_result.get('error_message')}")
            print('=' * 80)
    else:
        logger.error('❌ Test failed. The ProtoEngine did not produce a result signal within the timeout.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Test the NIREON V4 ProtoEngine.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='\nExamples:\n\n'
               '1. Execute a math proto from a file:\n'
               '   python -m 02_proto_runner.run_proto_test --proto examples/math_proto_example.yaml\n\n'
               '2. Generate and execute a proto from a natural language request:\n'
               '   python -m 02_proto_runner.run_proto_test --nl "Plot a sine wave from 0 to 2*pi and save it as a PNG file."\n'
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--proto', dest='proto_filepath', help='Path to the YAML file containing the Proto block to execute.')
    group.add_argument('--nl', dest='nl_request', help='A natural language request to be converted into a Proto block by the ProtoGenerator.')
    parser.add_argument('--timeout', type=float, default=60.0, help='Timeout in seconds to wait for the final result.')
    args = parser.parse_args()

    # Ensure example file exists
    example_dir = PROJECT_ROOT / 'examples'
    example_dir.mkdir(exist_ok=True)
    example_file = example_dir / 'math_proto_example.yaml'
    if not example_file.exists():
        example_content = (
            'schema_version: proto/1.0\n'
            'id: proto_math_plot_sine_wave\n'
            'eidos: math\n'
            'description: "Generates and saves a plot of a sine wave."\n'
            'objective: "Visualize the sine function over one period."\n'
            'function_name: plot_sine_wave\n'
            'inputs:\n'
            '  output_filename: "sine_wave.png"\n'
            'code: |\n'
            '  import numpy as np\n'
            '  import matplotlib.pyplot as plt\n'
            '  def plot_sine_wave(output_filename: str):\n'
            '      x = np.linspace(0, 2 * np.pi, 400)\n'
            '      y = np.sin(x)\n'
            '      plt.figure(figsize=(10, 6))\n'
            '      plt.plot(x, y)\n'
            '      plt.title("Sine Wave from 0 to 2*pi")\n'
            '      plt.xlabel("Angle [rad]")\n'
            '      plt.ylabel("sin(x)")\n'
            '      plt.grid(True)\n'
            '      plt.savefig(output_filename)\n'
            '      plt.close() # Important to free memory\n'
            '      return {\n'
            '          "status": "success",\n'
            '          "message": f"Plot saved to {output_filename}",\n'
            '          "data_points": len(x)\n'
            '      }\n'
            'requirements: ["numpy", "matplotlib"]\n'
            'limits:\n'
            '  timeout_sec: 15\n'
            '  memory_mb: 256\n'
            '  allowed_imports: ["numpy", "matplotlib.pyplot"]\n'
        )
        example_file.write_text(example_content)
        print(f'Created example proto file at: {example_file}')
        
    asyncio.run(main(args.proto_filepath, args.timeout, args.nl_request))