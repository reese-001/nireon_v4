# C:\Users\erees\Documents\development\nireon_v4\run_explorer_test.py
# --- CORRECTED FILE CONTENT ---
from __future__ import annotations
import argparse
import asyncio
import logging
import os
import sys
import time
import json
from pathlib import Path
from typing import Optional, Dict, Any, List, Set
from datetime import datetime

# --- (Keep the _find_project_root and path setup as is) ---
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

# --- (The rest of the imports can stay the same) ---
from bootstrap import bootstrap_nireon_system
from signals import SeedSignal, EpistemicSignal, IdeaGeneratedSignal, TrustAssessmentSignal
from signals.core import ProtoResultSignal, MathProtoResultSignal, ProtoErrorSignal, ProtoTaskSignal
from core.lifecycle import ComponentRegistryMissingError
from core.registry.component_registry import ComponentRegistry
from domain.ports.event_bus_port import EventBusPort
from domain.ports.idea_service_port import IdeaServicePort
from domain.ports.reactor_port import ReactorPort
from domain.context import NireonExecutionContext

# --- (The logging setup can stay the same) ---
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
    global _current_cid
    _current_cid = cid

# --- (wait_for_signal and ResultCapturer can stay the same) ---
async def wait_for_signal(event_bus: EventBusPort, *, signal_name: str, timeout: float=60.0, condition: Optional[callable]=None):
    loop = asyncio.get_running_loop()
    fut: asyncio.Future[EpistemicSignal] = loop.create_future()
    def _cb(payload: Any):
        if fut.done():
            return
        reconstructed_signal = None
        if isinstance(payload, dict):
            event_data = payload.get('event_data', payload)
            if 'signal_type' in event_data:
                try:
                    from signals import signal_class_map
                    signal_class = signal_class_map.get(event_data['signal_type'])
                    if signal_class:
                        reconstructed_signal = signal_class(**event_data)
                except Exception:
                    logger.error('Failed to reconstruct signal from dict', exc_info=True)
                    return
        elif isinstance(payload, EpistemicSignal):
            reconstructed_signal = payload
        if reconstructed_signal:
            if condition:
                if condition(reconstructed_signal):
                    logger.info(f"[wait_for_signal] Condition met for '{signal_name}'. Completing wait.")
                    fut.set_result(reconstructed_signal)
            else:
                logger.info(f"[wait_for_signal] Received signal '{signal_name}'. Completing wait.")
                fut.set_result(reconstructed_signal)
    event_bus.subscribe(signal_name, _cb)
    try:
        return await asyncio.wait_for(fut, timeout)
    finally:
        event_bus.unsubscribe(signal_name, _cb)

class ResultCapturer:
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
        self.generated_idea_ids: Set[str] = set()  # Track all generated ideas
        self.assessed_idea_ids: Set[str] = set()    
        self.loop_finished_event = asyncio.Event()
        self.proto_task_detected = False
        self.proto_task_signal = None
        
    def subscribe_to_signals(self):
        self.event_bus.subscribe(IdeaGeneratedSignal.__name__, self._handle_idea_generated)
        self.event_bus.subscribe(TrustAssessmentSignal.__name__, self._handle_trust_assessment)
        self.event_bus.subscribe('ProtoTaskSignal', self._handle_proto_task)
        
        # ALSO listen for GenerativeLoopFinishedSignal to detect quantifier triggering
        self.event_bus.subscribe('GenerativeLoopFinishedSignal', self._handle_generative_loop_finished)
        
        logger.info('[ResultCapturer] Subscribed to IdeaGenerated, TrustAssessment, ProtoTask, and GenerativeLoopFinished signals.')
    
    def _handle_generative_loop_finished(self, raw_payload: Any):
        """Detect when QuantifierAgent completes (meaning it was triggered)."""
        try:
            # Handle both dict and signal object formats
            if isinstance(raw_payload, dict):
                signal_payload = raw_payload.get('payload', raw_payload)
                quantifier_triggered = signal_payload.get('quantifier_triggered', False)
            else:
                # It's a signal object - access attributes directly
                if hasattr(raw_payload, 'payload'):
                    quantifier_triggered = raw_payload.payload.get('quantifier_triggered', False) if isinstance(raw_payload.payload, dict) else False
                else:
                    # Try to access as attribute
                    quantifier_triggered = getattr(raw_payload, 'quantifier_triggered', False)
            
            if quantifier_triggered:
                logger.info(f"[ResultCapturer] QuantifierAgent was triggered! Proto generation initiated.")
                self.proto_task_detected = True
                
            # Mark that we received the signal regardless
            self.loop_finished_event.set()
            self.proto_task_signal = raw_payload
        except Exception as e:
            logger.error(f'[ResultCapturer] Error handling GenerativeLoopFinishedSignal: {e}', exc_info=True)
        
    def all_ideas_assessed(self) -> bool:
        """Check if all generated ideas have been assessed."""
        return len(self.generated_idea_ids) > 0 and self.generated_idea_ids == self.assessed_idea_ids
        
    def _handle_proto_task(self, raw_payload: Dict[str, Any]):
        """Detect when a ProtoTaskSignal is published (meaning QuantifierAgent was triggered)."""
        logger.info(f"[ResultCapturer] ProtoTaskSignal detected! QuantifierAgent has been triggered.")
        self.proto_task_detected = True
        self.proto_task_signal = raw_payload
            
    def _handle_idea_generated(self, payload: Any):
        try:
            # Handle both dict and signal object formats
            if hasattr(payload, 'payload'):
                # It's a signal object
                signal_payload = payload.payload if isinstance(payload.payload, dict) else {}
                parent_id = signal_payload.get('parent_id') or getattr(payload, 'parent_id', None)
                idea_id = signal_payload.get('id') or getattr(payload, 'idea_id', None) or getattr(payload, 'target_id', None)
                text = signal_payload.get('text') or getattr(payload, 'idea_content', None) or getattr(payload, 'text', None)
                source = signal_payload.get('source_mechanism') or getattr(payload, 'source_node_id', None)
            else:
                # It's a dict
                signal_payload = payload.get('payload', payload)
                parent_id = signal_payload.get('parent_id')
                idea_id = signal_payload.get('id') or payload.get('idea_id')
                text = signal_payload.get('text') or payload.get('idea_content')
                source = signal_payload.get('source_mechanism') or payload.get('source_node_id')
            
            # Track this generated idea
            if idea_id:
                self.generated_idea_ids.add(idea_id)
                logger.info(f"[ResultCapturer] Tracked generated idea: {idea_id}")
                
            if not parent_id:
                # If no parent, it might be a root variation of the seed
                parent_id = self.run_data['seed_idea']['id']
                logger.info(f"[ResultCapturer] No parent_id for idea '{idea_id}', assuming it's a child of seed")
                
            if parent_id not in self.idea_map:
                logger.warning(f"[ResultCapturer] Parent '{parent_id}' not in idea_map. Current map keys: {list(self.idea_map.keys())}")
                # Try to create a placeholder parent
                if parent_id != self.run_data['seed_idea']['id']:
                    self.idea_map[parent_id] = {}
                    logger.info(f"[ResultCapturer] Created placeholder for missing parent '{parent_id}'")
                
            new_idea_node = {
                'id': idea_id,
                'text': text or 'No text provided',
                'source_mechanism': source or 'unknown',
                'trust_score': None,
                'is_stable': None,
                'variations': {}
            }
            
            if parent_id in self.idea_map:
                self.idea_map[parent_id][idea_id] = new_idea_node
                self.idea_map[idea_id] = new_idea_node['variations']
                self._idea_to_parent_map[idea_id] = parent_id
                self.run_data['metadata']['total_ideas'] += 1
                logger.info(f"[ResultCapturer] Added idea '{idea_id}' under parent '{parent_id}'")
            else:
                logger.error(f"[ResultCapturer] Failed to add idea '{idea_id}' - parent '{parent_id}' not found")
                
        except Exception as e:
            logger.error(f'[ResultCapturer] Error handling IdeaGeneratedSignal: {e}', exc_info=True)
            
    def _handle_trust_assessment(self, payload: Dict[str, Any]):
        try:
            idea_id = payload.get('target_id') or payload.get('idea_id') # Handle both signal structures
            trust_score = payload.get('trust_score')
            
            # Track this assessment
            if idea_id:
                self.assessed_idea_ids.add(idea_id)
                
            # Handle flattened and nested payload structures
            if 'assessment_details' in payload.get('payload', {}):
                is_stable = payload.get('payload', {}).get('is_stable')
            else:
                 is_stable = payload.get('is_stable')

            node_to_update = self._find_idea_node(idea_id)
            if node_to_update:
                node_to_update['trust_score'] = trust_score
                node_to_update['is_stable'] = is_stable
                self.run_data['metadata']['total_assessments'] += 1
                
                # Log high trust scores
                if trust_score and trust_score > 6.0:
                    logger.info(f"[ResultCapturer] High trust score detected: Idea {idea_id} = {trust_score:.2f} (meets quantifier threshold)")
                    
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
            return None
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
    
    
def _handle_generative_loop_finished(self, raw_payload: Any):
    """Detect when QuantifierAgent completes (meaning it was triggered)."""
    try:
        # Handle both dict and signal object formats
        if isinstance(raw_payload, dict):
            signal_payload = raw_payload.get('payload', raw_payload)
            quantifier_triggered = signal_payload.get('quantifier_triggered', False)
        else:
            # It's a signal object - access attributes directly
            if hasattr(raw_payload, 'payload'):
                quantifier_triggered = raw_payload.payload.get('quantifier_triggered', False) if isinstance(raw_payload.payload, dict) else False
            else:
                # Try to access as attribute
                quantifier_triggered = getattr(raw_payload, 'quantifier_triggered', False)
        
        if quantifier_triggered:
            logger.info(f"[ResultCapturer] QuantifierAgent was triggered! Proto generation initiated.")
            self.proto_task_detected = True
            
        # Mark that we received the signal regardless
        self.loop_finished_event.set()
        self.proto_task_signal = raw_payload
    except Exception as e:
        logger.e
        
    def all_ideas_assessed(self) -> bool:
        """Check if all generated ideas have been assessed."""
        return len(self.generated_idea_ids) > 0 and self.generated_idea_ids == self.assessed_idea_ids
        
    def _handle_proto_task(self, raw_payload: Dict[str, Any]):
        """Detect when a ProtoTaskSignal is published (meaning QuantifierAgent was triggered)."""
        logger.info(f"[ResultCapturer] ProtoTaskSignal detected! QuantifierAgent has been triggered.")
        self.proto_task_detected = True
        self.proto_task_signal = raw_payload
            
    def _handle_idea_generated(self, payload: Any):
        try:
            # Handle both dict and signal object formats
            if hasattr(payload, 'payload'):
                # It's a signal object
                signal_payload = payload.payload if isinstance(payload.payload, dict) else {}
                parent_id = signal_payload.get('parent_id') or getattr(payload, 'parent_id', None)
                idea_id = signal_payload.get('id') or getattr(payload, 'idea_id', None) or getattr(payload, 'target_id', None)
                text = signal_payload.get('text') or getattr(payload, 'idea_content', None) or getattr(payload, 'text', None)
                source = signal_payload.get('source_mechanism') or getattr(payload, 'source_node_id', None)
            else:
                # It's a dict
                signal_payload = payload.get('payload', payload)
                parent_id = signal_payload.get('parent_id')
                idea_id = signal_payload.get('id') or payload.get('idea_id')
                text = signal_payload.get('text') or payload.get('idea_content')
                source = signal_payload.get('source_mechanism') or payload.get('source_node_id')
            
            # Track this generated idea
            if idea_id:
                self.generated_idea_ids.add(idea_id)
                logger.info(f"[ResultCapturer] Tracked generated idea: {idea_id}")
                
            if not parent_id:
                # If no parent, it might be a root variation of the seed
                parent_id = self.run_data['seed_idea']['id']
                logger.info(f"[ResultCapturer] No parent_id for idea '{idea_id}', assuming it's a child of seed")
                
            if parent_id not in self.idea_map:
                logger.warning(f"[ResultCapturer] Parent '{parent_id}' not in idea_map. Current map keys: {list(self.idea_map.keys())}")
                # Try to create a placeholder parent
                if parent_id != self.run_data['seed_idea']['id']:
                    self.idea_map[parent_id] = {}
                    logger.info(f"[ResultCapturer] Created placeholder for missing parent '{parent_id}'")
                
            new_idea_node = {
                'id': idea_id,
                'text': text or 'No text provided',
                'source_mechanism': source or 'unknown',
                'trust_score': None,
                'is_stable': None,
                'variations': {}
            }
            
            if parent_id in self.idea_map:
                self.idea_map[parent_id][idea_id] = new_idea_node
                self.idea_map[idea_id] = new_idea_node['variations']
                self._idea_to_parent_map[idea_id] = parent_id
                self.run_data['metadata']['total_ideas'] += 1
                logger.info(f"[ResultCapturer] Added idea '{idea_id}' under parent '{parent_id}'")
            else:
                logger.error(f"[ResultCapturer] Failed to add idea '{idea_id}' - parent '{parent_id}' not found")
                
        except Exception as e:
            logger.error(f'[ResultCapturer] Error handling IdeaGeneratedSignal: {e}', exc_info=True)
            
    def _handle_trust_assessment(self, payload: Dict[str, Any]):
        try:
            idea_id = payload.get('target_id') or payload.get('idea_id') # Handle both signal structures
            trust_score = payload.get('trust_score')
            
            # Track this assessment
            if idea_id:
                self.assessed_idea_ids.add(idea_id)
                
            # Handle flattened and nested payload structures
            if 'assessment_details' in payload.get('payload', {}):
                is_stable = payload.get('payload', {}).get('is_stable')
            else:
                 is_stable = payload.get('is_stable')

            node_to_update = self._find_idea_node(idea_id)
            if node_to_update:
                node_to_update['trust_score'] = trust_score
                node_to_update['is_stable'] = is_stable
                self.run_data['metadata']['total_assessments'] += 1
                
                # Log high trust scores
                if trust_score and trust_score > 6.0:
                    logger.info(f"[ResultCapturer] High trust score detected: Idea {idea_id} = {trust_score:.2f} (meets quantifier threshold)")
                    
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
            return None
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


async def main(iterations: int, timeout: float):
    set_component('MainRunner')
    logger.info('Bootstrapping NIREON V4 via manifest…')
    
    # MODIFICATION: Explicitly point to the correct manifest file.
    manifest_path = PROJECT_ROOT / 'configs' / 'manifests' / 'standard.yaml'
    if not manifest_path.exists():
        logger.error(f"FATAL: Manifest file not found at {manifest_path}")
        sys.exit(1)
        
    boot_res = await bootstrap_nireon_system(config_paths=[manifest_path], strict_mode=False)

    if not boot_res.success:
        logger.error('Bootstrap failed – aborting.\n%s', boot_res.get_health_report())
        sys.exit(1)
    
    logger.info('Bootstrap complete in %.2fs, %d components registered.', boot_res.bootstrap_duration, boot_res.component_count)
    
    registry: ComponentRegistry = boot_res.registry
    event_bus: EventBusPort = registry.get_service_instance(EventBusPort)
    idea_service: IdeaServicePort = registry.get_service_instance(IdeaServicePort)
    
    # FIX: Check the bootstrap result for reactor setup
    if hasattr(boot_res, 'reactor'):
        reactor = boot_res.reactor
        logger.info("✓ Found reactor from bootstrap result")
    else:
        # The reactor might have been setup but not exposed
        # Check if ReactorSetupPhase was successful
        logger.info("Checking bootstrap phases for reactor setup...")
        
        # Try to get event bus and manually process the signal
        reactor = None
        if event_bus:
            # The reactor is integrated with the event bus during ReactorSetupPhase
            # We can test if it's working by checking if rules are loaded
            logger.info("Event bus found, checking if reactor rules are active...")
            
            # Create a dummy signal to test
            test_signal = EpistemicSignal(signal_type='TEST_SIGNAL', source_node_id='test')
            
            # If reactor is setup, it should be listening on the event bus
            # For now, we'll proceed without direct reactor access
            # The rules should still fire when we publish signals
            reactor = None  # We'll use event bus publishing instead
            logger.info("Reactor appears to be integrated with event bus. Will proceed using event-based signaling.")
    
    if reactor is None and event_bus is None:
        logger.error('Neither Reactor nor EventBus found. Cannot proceed.')
        sys.exit(1)

    ctx = NireonExecutionContext(
        run_id=f'script_run_{int(time.time())}',
        component_registry=registry,
        event_bus=event_bus,
        config=boot_res.global_config,
        feature_flags=boot_res.global_config.get('feature_flags', {})
    )

    terminal_signal_name = 'GenerativeLoopFinishedSignal'
    
    for idx in range(1, iterations + 1):
        set_component('MainRunner')
        seed_text = 'How can a brick-and-mortar electronics retailer like Best Buy survive in an era of high tariffs and intense online competition?'
        objective = 'Generate innovative business strategies for Best Buy to adapt and thrive. Focus on supply chain resilience, in-store experience, and service-based revenue models.'
        seed_idea = idea_service.create_idea(text=seed_text, parent_id=None, context=ctx)
        
        logger.info('[%d/%d] Seed idea persisted with ID %s', idx, iterations, seed_idea.idea_id)
        
        capturer = ResultCapturer(seed_idea.idea_id, seed_text, event_bus)
        capturer.subscribe_to_signals()
        
        payload = {
            'seed_idea_id': seed_idea.idea_id,
            'text': seed_idea.text,
            'metadata': {'iteration': idx, 'objective': objective.strip(), 'depth': 0}
        }
        seed_signal = SeedSignal(source_node_id='MainRunner', payload=payload, run_id=ctx.run_id)

        logger.info('Processing signal %s for idea %s', seed_signal.signal_type, seed_idea.idea_id)
        
        # Instead of using reactor.process_signal, publish to event bus
        if reactor:
            await reactor.process_signal(seed_signal)
        else:
            # Publish directly to event bus - the reactor rules should pick it up
            logger.info("Publishing SeedSignal to event bus...")
            event_bus.publish(SeedSignal.__name__, seed_signal)
        try:
            logger.info('Waiting for the generative loop to finish (or timeout)...')
            await asyncio.wait_for(capturer.loop_finished_event.wait(), timeout=timeout)
            logger.info('✅ GenerativeLoopFinishedSignal received. Proceeding to analysis.')
        except asyncio.TimeoutError:
            logger.warning(f"Timed out after {timeout} seconds waiting for GenerativeLoopFinishedSignal.")
            logger.warning("This may mean the loop is stuck or taking too long. Check reactor rules for issues.")
               
        # Give a bit more time for the quantifier rule to trigger if applicable
        await asyncio.sleep(15)
        
        # Check if ProtoTaskSignal was detected (meaning QuantifierAgent was triggered)
        if capturer.proto_task_detected:  
            logger.info("🎉🎉🎉 QuantifierAgent Triggered! Proto task created. Waiting for proto execution result... 🎉🎉🎉")
            
            # Wait for the Proto execution to complete
            proto_result_event = asyncio.Event()
            proto_result = None
            
            def proto_callback(payload):
                nonlocal proto_result
                signal_data = payload.get('payload', payload) if isinstance(payload, dict) else payload
                logger.debug(f"[proto_callback] Received signal: {json.dumps(payload, indent=2) if isinstance(payload, dict) else payload}")
                
                if hasattr(signal_data, 'signal_type'):
                    if signal_data.signal_type in ['ProtoResultSignal', 'MathProtoResultSignal']:
                        logger.info(f"Proto execution completed! Proto ID: {getattr(signal_data, 'proto_block_id', 'unknown')}")
                        proto_result = signal_data
                        proto_result_event.set()
                    elif signal_data.signal_type == 'ProtoErrorSignal':
                        logger.error(f"Proto execution failed! Proto ID: {getattr(signal_data, 'proto_block_id', 'unknown')}")
                        proto_result = signal_data
                        proto_result_event.set()
            
            # Subscribe to Proto result signals
            event_bus.subscribe('ProtoResultSignal', proto_callback)
            event_bus.subscribe('MathProtoResultSignal', proto_callback)
            event_bus.subscribe('ProtoErrorSignal', proto_callback)
            
            try:
                # Wait up to 30 seconds for Proto execution
                await asyncio.wait_for(proto_result_event.wait(), timeout=30.0)
                
                if proto_result:
                    if hasattr(proto_result, 'success') and proto_result.success:
                        logger.info(f"✅ Proto execution successful! Artifacts: {getattr(proto_result, 'artifacts', [])}")
                        # List the artifacts directory
                        artifacts_dir = PROJECT_ROOT / 'runtime' / 'proto' / 'math' / 'artifacts'
                        if artifacts_dir.exists():
                            logger.info(f"Artifacts directory contents: {list(artifacts_dir.iterdir())}")
                    else:
                        logger.error(f"❌ Proto execution failed: {getattr(proto_result, 'error_message', 'Unknown error')}")
            except asyncio.TimeoutError:
                logger.warning("Timed out waiting for Proto execution result after 30 seconds")
            finally:
                event_bus.unsubscribe('ProtoResultSignal', proto_callback)
                event_bus.unsubscribe('MathProtoResultSignal', proto_callback)
                event_bus.unsubscribe('ProtoErrorSignal', proto_callback)
        else:
            # Log assessment summary
            logger.info('=' * 80)
            logger.info('ASSESSMENT SUMMARY:')
            logger.info(f'Generated ideas: {len(capturer.generated_idea_ids)}')
            logger.info(f'Assessed ideas: {len(capturer.assessed_idea_ids)}')
            
            # Show trust scores
            high_trust_count = 0
            for idea_id in capturer.assessed_idea_ids:
                node = capturer._find_idea_node(idea_id)
                if node:
                    trust_score = node.get('trust_score', 0)
                    if trust_score > 6.0:
                        high_trust_count += 1
                    logger.info(f"  Idea {idea_id}: trust_score={trust_score}, is_stable={node.get('is_stable', 'N/A')}")
                    
            if high_trust_count > 0:
                logger.warning(f'{high_trust_count} ideas had trust scores > 6.0 but QuantifierAgent was not triggered!')
                logger.info('Check that the quantifier rule in advanced.yaml is enabled and properly configured.')
            else:
                logger.info('No idea met the quantifier trigger criteria (trust_score > 6.0)')
            logger.info('=' * 80)
            
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