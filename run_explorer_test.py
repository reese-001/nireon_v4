from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import time
import json
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime

# --- Dynamic Path Setup ---
def _find_project_root(markers: list[str]=['bootstrap', 'domain', 'core', 'configs']) -> Optional[Path]:
    """Finds the project root by looking for key directory markers."""
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

# --- Nireon Module Imports ---
from bootstrap import bootstrap_nireon_system
from signals import SeedSignal, EpistemicSignal, IdeaGeneratedSignal, TrustAssessmentSignal
from core.lifecycle import ComponentRegistryMissingError
from core.registry.component_registry import ComponentRegistry
from domain.ports.event_bus_port import EventBusPort
from domain.ports.idea_service_port import IdeaServicePort
from domain.ports.reactor_port import ReactorPort
from domain.context import NireonExecutionContext

# --- Logging Configuration ---
_logging_fmt = '%(asctime)s - %(name)-30s - %(levelname)-8s - [%(component_id)s] - %(message)s'
logging.basicConfig(level=logging.INFO, format=_logging_fmt, datefmt='%H:%M:%S')
for quiet in ('httpx', 'httpcore', 'openai', 'huggingface_hub'):
    logging.getLogger(quiet).setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

_default_record_factory = logging.getLogRecordFactory()
_current_cid = 'System'

def _record_factory(*args, **kw):
    rec = _default_record_factory(*args, **kw)
    rec.component_id = _current_cid
    return rec
logging.setLogRecordFactory(_record_factory)

def set_component(cid: str):
    """Sets the component ID for the logging context."""
    global _current_cid
    _current_cid = cid

# --- Test Utilities ---

async def wait_for_signal(
    event_bus: EventBusPort, 
    *, 
    signal_name: str, 
    timeout: float=60.0, 
    terminal_status: str = "completed_one_branch"
):
    """
    Waits for a specific signal, but only completes if the signal's payload
    contains a status matching the terminal_status.
    """
    loop = asyncio.get_running_loop()
    fut: asyncio.Future[EpistemicSignal] = loop.create_future()

    def _cb(payload: Any):
        if fut.done():
            return
        
        reconstructed_signal = None
        if isinstance(payload, dict):
            # This handles signals that are dictionaries, common in event buses
            event_data = payload.get('event_data', payload)
            if 'signal_type' in event_data:
                try:
                    from signals import signal_class_map
                    signal_class = signal_class_map.get(event_data['signal_type'])
                    if signal_class:
                        reconstructed_signal = signal_class(**event_data)
                except Exception:
                    logger.error('Failed to reconstruct signal from dict', exc_info=True)
                    pass
        elif isinstance(payload, EpistemicSignal):
            reconstructed_signal = payload

        if reconstructed_signal:
            # THIS IS THE KEY CHANGE: Check the status in the payload
            signal_payload = getattr(reconstructed_signal, 'payload', {})
            current_status = signal_payload.get('status')
            logger.debug(f"[wait_for_signal] Received '{signal_name}' with status: '{current_status}'")
            if current_status == terminal_status:
                logger.info(f"[wait_for_signal] Terminal status '{terminal_status}' detected. Completing wait.")
                fut.set_result(reconstructed_signal)
            else:
                logger.info(f"[wait_for_signal] Intermediate status '{current_status}' detected. Continuing to wait.")

    event_bus.subscribe(signal_name, _cb)
    try:
        return await asyncio.wait_for(fut, timeout)
    finally:
        event_bus.unsubscribe(signal_name, _cb)

class ResultCapturer:
    """A helper class to capture and structure the results of the generative run."""
    def __init__(self, seed_idea_id: str, seed_text: str, event_bus: EventBusPort):
        self.event_bus = event_bus
        self.run_data: Dict[str, Any] = {
            'seed_idea': {
                'id': seed_idea_id,
                'text': seed_text,
                'trust_score': None,
                'is_stable': None,
                'variations': {}
            },
            'metadata': {
                'run_start_time': datetime.now().isoformat(),
                'run_end_time': None,
                'total_ideas': 1,
                'total_assessments': 0
            }
        }
        self.idea_map: Dict[str, Dict] = {seed_idea_id: self.run_data['seed_idea']['variations']}
        self._idea_to_parent_map: Dict[str, str] = {}

    def subscribe_to_signals(self):
        self.event_bus.subscribe(IdeaGeneratedSignal.__name__, self._handle_idea_generated)
        self.event_bus.subscribe(TrustAssessmentSignal.__name__, self._handle_trust_assessment)
        logger.info('[ResultCapturer] Subscribed to IdeaGenerated and TrustAssessment signals.')

    def _handle_idea_generated(self, payload: Dict[str, Any]):
        try:
            # Handle both direct signal objects and dictionary-based event payloads
            signal_payload = payload.get('payload', payload)
            parent_id = signal_payload.get('parent_id')
            idea_id = signal_payload.get('id') or payload.get('idea_id')

            if not parent_id or parent_id not in self.idea_map:
                logger.warning(f"[ResultCapturer] Received idea '{idea_id}' with an unknown parent '{parent_id}'. Cannot place in tree.")
                return

            new_idea_node = {
                'id': idea_id,
                'text': signal_payload.get('text') or payload.get('idea_content'),
                'source_mechanism': signal_payload.get('source_mechanism') or payload.get('source_node_id'),
                'trust_score': None,
                'is_stable': None,
                'variations': {}
            }
            self.idea_map[parent_id][idea_id] = new_idea_node
            self.idea_map[idea_id] = new_idea_node['variations']
            self._idea_to_parent_map[idea_id] = parent_id
            self.run_data['metadata']['total_ideas'] += 1
        except Exception as e:
            logger.error(f'[ResultCapturer] Error handling IdeaGeneratedSignal: {e}', exc_info=True)

    def _handle_trust_assessment(self, payload: Dict[str, Any]):
        try:
            idea_id = payload.get('target_id')
            trust_score = payload.get('trust_score')
            assessment_details = payload.get('payload', {}).get('assessment_details', {})
            is_stable = assessment_details.get('is_stable')
            
            node_to_update = self._find_idea_node(idea_id)
            if node_to_update:
                node_to_update['trust_score'] = trust_score
                node_to_update['is_stable'] = is_stable
                self.run_data['metadata']['total_assessments'] += 1
            else:
                logger.warning(f'[ResultCapturer] Could not find idea {idea_id} in the tree to update its assessment.')
        except Exception as e:
            logger.error(f'[ResultCapturer] Error handling TrustAssessmentSignal: {e}', exc_info=True)
            
    def _find_idea_node(self, idea_id: str) -> Optional[Dict]:
        if idea_id == self.run_data['seed_idea']['id']:
            return self.run_data['seed_idea']
        
        path = []
        curr_id = idea_id
        while curr_id in self._idea_to_parent_map:
            path.insert(0, curr_id)
            curr_id = self._idea_to_parent_map[curr_id]
        
        if curr_id != self.run_data['seed_idea']['id']:
             return None # Did not trace back to the root

        node = self.run_data['seed_idea']
        for step_id in path:
            node = node['variations'].get(step_id)
            if node is None:
                return None
        return node

    def save_results(self):
        self.run_data['metadata']['run_end_time'] = datetime.now().isoformat()
        runtime_dir = PROJECT_ROOT / 'runtime'
        runtime_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        filename = runtime_dir / f'idea_evolution_{timestamp}.json'
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.run_data, f, indent=2, ensure_ascii=False)
            logger.info(f'✅ Successfully saved evolution results to: {filename}')
        except Exception as e:
            logger.error(f'❌ Failed to save results to file: {e}')

# --- Main Execution Logic ---

async def main(iterations: int, timeout: float):
    set_component('MainRunner')
    logger.info('Bootstrapping NIREON V4 via manifest…')
    manifest = PROJECT_ROOT / 'configs' / 'manifests' / 'standard.yaml'
    boot_res = await bootstrap_nireon_system(config_paths=[manifest], strict_mode=False)

    if not boot_res.success:
        logger.error('Bootstrap failed – aborting.\n%s', boot_res.get_health_report())
        sys.exit(1)
    
    logger.info('Bootstrap complete in %.2fs, %d components registered.', boot_res.bootstrap_duration, boot_res.component_count)

    registry: ComponentRegistry = boot_res.registry
    event_bus: EventBusPort = registry.get_service_instance(EventBusPort)
    idea_service: IdeaServicePort = registry.get_service_instance(IdeaServicePort)
    try:
        reactor: ReactorPort | None = registry.get_service_instance(ReactorPort)
    except ComponentRegistryMissingError:
        reactor = None

    if reactor is None:
        logger.error('No Reactor component found – check your manifest / factory setup.')
        sys.exit(1)

    ctx = NireonExecutionContext(
        run_id=f'script_run_{int(time.time())}',
        component_registry=registry,
        event_bus=event_bus,
        config=boot_res.global_config,
        feature_flags=boot_res.global_config.get('feature_flags', {})
    )
    
    terminal_signal_name = 'GenerativeLoopFinishedSignal'
    terminal_signal_status = 'completed_one_branch'

    for idx in range(1, iterations + 1):
        set_component('MainRunner')
        seed_text = 'A radically new perspective on the proof that 2+2=4.'
        seed_idea = idea_service.create_idea(text=seed_text, parent_id=None, context=ctx)
        logger.info('[%d/%d] Seed idea persisted with ID %s', idx, iterations, seed_idea.idea_id)

        capturer = ResultCapturer(seed_idea.idea_id, seed_text, event_bus)
        capturer.subscribe_to_signals()
        
        objective = """
        Generate a *novel and previously unknown* formal proof of 2+2=4 based on von Neumann ordinals.

        **CRITICAL CONSTRAINTS:**
        1.  **AVOID** the standard, textbook presentation. Do not simply list the Peano axioms and apply them mechanically.
        2.  **INCORPORATE** a core concept or analogy from one of the following domains:
            - **Computer Science:** Type Theory, Lambda Calculus, or Cellular Automata.
            - **Physics:** Quantum Superposition or Spacetime Diagrams.
            - **Philosophy:** Mereology (the theory of parts and wholes).
        3.  The proof's logic MUST be fundamentally based on the chosen analogy, not just decorated with its terminology.
        4.  **MINIMIZE** reliance on traditional notations like 'S(n)' for successor. Find an alternative representation or concept.

        **GOAL:**
        The final proof should be both **rigorously correct** and **epistemologically surprising**. It should reveal a deeper, non-obvious connection between arithmetic and the chosen domain.

        **STRUCTURE:**
        1.  State the central analogy/concept.
        2.  Define numbers and operations in terms of this concept.
        3.  Present the step-by-step proof.
        4.  Conclude with the philosophical implications of this new perspective.
        """
        payload = {'seed_idea_id': seed_idea.idea_id, 'text': seed_idea.text, 'metadata': {'iteration': idx, 'objective': objective.strip()}}
        seed_signal = SeedSignal(source_node_id='MainRunner', payload=payload, run_id=ctx.run_id)

        waiter_task = asyncio.create_task(wait_for_signal(
            event_bus, 
            signal_name=terminal_signal_name, 
            timeout=timeout,
            terminal_status=terminal_signal_status
        ))

        logger.info('Processing signal %s for idea %s', seed_signal.signal_type, seed_idea.idea_id)
        await reactor.process_signal(seed_signal)

        try:
            finished_signal_object = await waiter_task
            finished_payload = finished_signal_object.payload
            logger.info('🎉🎉🎉 Generative cycle %d finished successfully! – Reactor reported %s → %s 🎉🎉🎉', idx, terminal_signal_name, finished_payload.get('status', 'N/A'))
        except asyncio.TimeoutError:
            logger.warning('Timed out after %.1fs waiting for terminal signal (status=%s); continuing.', timeout, terminal_signal_status)
        finally:
            capturer.save_results()

    logger.info('All %d seed cycles submitted. Exiting.', iterations)


if __name__ == '__main__':
    if Path.cwd() != PROJECT_ROOT:
        os.chdir(PROJECT_ROOT)
        print(f'Changed CWD to project root: {PROJECT_ROOT}')
    
    parser = argparse.ArgumentParser(description='Kick‑off a Reactor‑driven idea evolution run.')
    parser.add_argument('--iterations', '-n', type=int, default=1, help='How many distinct seed ideas to inject (default 1)')
    parser.add_argument('--timeout', '-t', type=float, default=120.0, help='Seconds to wait for Reactor to finish each cycle')
    args = parser.parse_args()
    
    asyncio.run(main(args.iterations, args.timeout))