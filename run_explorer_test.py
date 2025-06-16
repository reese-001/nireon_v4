from __future__ import annotations
import argparse
import asyncio
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

def _find_project_root(markers: list[str]=['bootstrap', 'domain', 'core', 'configs']) -> Optional[Path]:
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
from signals import SeedSignal, EpistemicSignal # Import EpistemicSignal for type hinting
from core.lifecycle import ComponentRegistryMissingError
from core.registry.component_registry import ComponentRegistry
from domain.ports.event_bus_port import EventBusPort
from domain.ports.idea_service_port import IdeaServicePort
from domain.ports.reactor_port import ReactorPort
from domain.context import NireonExecutionContext

_logging_fmt = '%(asctime)s - %(name)-30s - %(levelname)-8s - [%(component_id)s] - %(message)s'
logging.basicConfig(level=logging.INFO, format=_logging_fmt, datefmt='%H:%M:%S')
for quiet in ('httpx', 'httpcore', 'openai'):
    logging.getLogger(quiet).setLevel(logging.WARNING)

_default_record_factory = logging.getLogRecordFactory()
_current_cid = 'System'

def _record_factory(*args, **kw):
    rec = _default_record_factory(*args, **kw)
    rec.component_id = _current_cid
    return rec

logging.setLogRecordFactory(_record_factory)

def set_component(cid: str):
    global _current_cid
    _current_cid = cid


async def wait_for_signal(event_bus: EventBusPort, *, signal_name: str, timeout: float = 60.0):
    loop = asyncio.get_running_loop()
    fut: asyncio.Future[EpistemicSignal] = loop.create_future() # Hint that we expect a signal object

    def _cb(payload: EpistemicSignal): # The payload from our bus is the signal object
        if not fut.done():
            fut.set_result(payload)

    event_bus.subscribe(signal_name, _cb)
    try:
        return await asyncio.wait_for(fut, timeout)
    finally:
        # It's good practice to clean up the subscription, though not strictly necessary here.
        pass


async def main(iterations: int, timeout: float):
    set_component('MainRunner')
    log = logging.getLogger('MainRunner')

    log.info('Bootstrapping NIREON V4 via manifest…')
    manifest = PROJECT_ROOT / 'configs' / 'manifests' / 'standard.yaml'
    boot_res = await bootstrap_nireon_system(config_paths=[manifest], strict_mode=False)

    if not boot_res.success:
        log.error('Bootstrap failed – aborting.\n%s', boot_res.get_health_report())
        sys.exit(1)

    log.info('Bootstrap complete in %.2fs, %d components registered.', boot_res.bootstrap_duration,
             boot_res.component_count)

    registry: ComponentRegistry = boot_res.registry
    event_bus: EventBusPort = registry.get_service_instance(EventBusPort)
    idea_service: IdeaServicePort = registry.get_service_instance(IdeaServicePort)
    try:
        reactor: ReactorPort | None = registry.get_service_instance(ReactorPort)
    except ComponentRegistryMissingError:
        reactor = None

    if reactor is None:
        log.error('No Reactor component found – check your manifest / factory setup.')
        sys.exit(1)

    ctx = NireonExecutionContext(run_id=f'script_run_{int(time.time())}',
                                 component_registry=registry, event_bus=event_bus,
                                 config=boot_res.global_config,
                                 feature_flags=boot_res.global_config.get('feature_flags', {}))

    terminal_signal = 'GenerativeLoopFinishedSignal'

    for idx in range(1, iterations + 1):
        set_component('MainRunner')
        seed_text = 'A detective discovers a parallel universe hidden in a coffee shop.'
        seed_idea = idea_service.create_idea(text=seed_text, parent_id=None)
        log.info('[%d/%d] Seed idea persisted with ID %s', idx, iterations, seed_idea.idea_id)

        payload = {
            'seed_idea_id': seed_idea.idea_id,
            'text': seed_idea.text,
            'metadata': {
                'iteration': idx,
                'objective': 'Generate fun and creative story variations from this seed concept.'
            }
        }
        seed_signal = SeedSignal(source_node_id='MainRunner', payload=payload, run_id=ctx.run_id)

        waiter_task = asyncio.create_task(
            wait_for_signal(event_bus, signal_name=terminal_signal, timeout=timeout)
        )

        log.info('Processing signal %s for idea %s', seed_signal.signal_type, seed_idea.idea_id)
        await reactor.process_signal(seed_signal)

        try:
            finished_signal_object = await waiter_task
            # FIX: Access the 'payload' attribute of the signal object, which is a dictionary.
            finished_payload = finished_signal_object.payload
            log.info('🎉🎉🎉 Generative cycle %d finished successfully! – Reactor reported %s → %s 🎉🎉🎉',
                     idx, terminal_signal, finished_payload.get('status', 'N/A'))
        except asyncio.TimeoutError:
            log.warning('Timed out after %.1fs waiting for %s; continuing.', timeout, terminal_signal)
    log.info('All %d seed cycles submitted. Exiting.', iterations)


if __name__ == '__main__':
    if Path.cwd() != PROJECT_ROOT:
        os.chdir(PROJECT_ROOT)
        print(f"Changed CWD to project root: {PROJECT_ROOT}")

    parser = argparse.ArgumentParser(description='Kick‑off a Reactor‑driven idea evolution run.')
    parser.add_argument('--iterations', '-n', type=int, default=1,
                        help='How many distinct seed ideas to inject (default 1)')
    parser.add_argument('--timeout', '-t', type=float, default=120.0,
                        help='Seconds to wait for Reactor to finish each cycle')
    args = parser.parse_args()
    asyncio.run(main(args.iterations, args.timeout))