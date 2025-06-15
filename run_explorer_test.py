# run_explorer_test.py
import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional
import time
import numpy as np 
from domain.ports.idea_service_port import IdeaServicePort


def find_project_root(marker_dirs=['bootstrap', 'domain', 'core', 'configs']):
    """
    Finds the project root directory by looking for a set of marker directories.
    """
    current_dir = Path(__file__).resolve().parent
    paths_to_check = [current_dir, current_dir.parent, current_dir.parent.parent]
    for p_root in paths_to_check:
        if all(((p_root / marker).is_dir() for marker in marker_dirs)):
            return p_root
    
    if all((Path.cwd() / marker).is_dir() for marker in marker_dirs):
        return Path.cwd()
        
    return None

PROJECT_ROOT = find_project_root()
if PROJECT_ROOT is None:
    print('ERROR: Could not determine the NIREON V4 project root.')
    sys.exit(1)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bootstrap import bootstrap_nireon_system
from domain.context import NireonExecutionContext
from components.mechanisms.explorer.service import ExplorerMechanism
from components.mechanisms.sentinel.service import SentinelMechanism
from components.mechanisms.catalyst.service import CatalystMechanism
from domain.ideas.idea import Idea
from application.services.idea_service import IdeaService
from core.registry.component_registry import ComponentRegistry
from domain.ports.event_bus_port import EventBusPort
from domain.ports.embedding_port import EmbeddingPort # <-- IMPORT EMBEDDING PORT
from application.services.frame_factory_service import FrameFactoryService


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)-25s - %(levelname)-8s - [%(component_id)s] - %(message)s',
    datefmt='%H:%M:%S'
)
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)
logging.getLogger('openai').setLevel(logging.WARNING)

# Custom log record factory to inject component_id
_original_logrecord_factory = logging.getLogRecordFactory()
_component_id_for_log_context = 'System'

def record_factory(*args, **kwargs):
    record = _original_logrecord_factory(*args, **kwargs)
    global _component_id_for_log_context
    record.component_id = _component_id_for_log_context
    return record

logging.setLogRecordFactory(record_factory)

def set_current_log_component_id(component_id: str):
    """Sets the component ID for the current logging context."""
    global _component_id_for_log_context
    _component_id_for_log_context = component_id

async def main():
    # This list will be populated by our event handler
    newly_persisted_ideas: List[Idea] = []

    # --- Catalyst Setup ---
    # In V4, dimensions are defined by the embedding provider config. Let's get that.
    temp_bootstrap_result = await bootstrap_nireon_system(config_paths=[PROJECT_ROOT / 'configs' / 'manifests' / 'standard.yaml'], strict_mode=False)
    embedding_service_config = temp_bootstrap_result.global_config.get('embedding', {})
    vector_dims = embedding_service_config.get('dimensions', 384)
    logger = logging.getLogger(__name__)
    logger.info(f"Vector dimensions set to {vector_dims} from config.")
    
    cross_domain_vectors = {
        'science_fiction': np.random.rand(vector_dims).astype(np.float64),
        'ancient_history': np.random.rand(vector_dims).astype(np.float64),
        'culinary_arts': np.random.rand(vector_dims).astype(np.float64),
        'quantum_physics': np.random.rand(vector_dims).astype(np.float64),
    }
    logger.info(f"Created {len(cross_domain_vectors)} dummy cross-domain vectors for Catalyst.")
    # --- End Catalyst Setup ---

    def idea_signal_handler(payload: dict):
        nonlocal newly_persisted_ideas
        idea_data = payload.get('idea', payload)
        idea_id_for_log = idea_data.get('idea_id') or idea_data.get('id')
        logger.info(f"Handler received signal for idea '{idea_id_for_log}'")
        
        logger.info(f"Handler received signal for idea '{idea_data.get('idea_id') or idea_data.get('id')}'")
        
        idea_obj = idea_service.create_idea(
            text=idea_data['text'],
            parent_id=idea_data.get('parent_ids', [None])[0] if idea_data.get('parent_ids') else idea_data.get('seed_idea_id')
        )
        if 'vector' in idea_data and idea_data['vector']:
            from domain.embeddings.vector import Vector
            idea_obj.theta = Vector.from_list(idea_data['vector'])
        
        newly_persisted_ideas.append(idea_obj)

    set_current_log_component_id('MainRunner')
    logger = logging.getLogger(__name__)

    logger.info(f'Starting Nireon V4 Generative Loop Test from project root: {PROJECT_ROOT}')
    manifest_file = PROJECT_ROOT / 'configs' / 'manifests' / 'standard.yaml'
    
    logger.info('Bootstrapping NIREON system...')
    try:
        bootstrap_result = await bootstrap_nireon_system(config_paths=[manifest_file], strict_mode=False)
        if not bootstrap_result.success:
            logger.error('Bootstrap failed!\n' + bootstrap_result.get_health_report())
            return
        logger.info('Bootstrap successful!')
    except Exception as e:
        logger.error(f'Critical bootstrap failure: {e}', exc_info=True)
        return

    # Resolve components from the registry
    registry: ComponentRegistry = bootstrap_result.registry
    explorer: ExplorerMechanism = registry.get('explorer_instance_01')
    sentinel: SentinelMechanism = registry.get('sentinel_instance_01')
    catalyst: CatalystMechanism = registry.get('catalyst_instance_01')
    idea_service: IdeaService = registry.get_service_instance(IdeaServicePort)  # ✓ FIXED
    embed_port: EmbeddingPort = registry.get_service_instance(EmbeddingPort)    # ✓ Already correct
    event_bus: EventBusPort = registry.get_service_instance(EventBusPort)       # ✓ FIXED
    frame_factory: FrameFactoryService = registry.get('frame_factory_service')  # ✓ Already correct

    # Subscribe our handler to the event bus
    event_bus.subscribe('IdeaGeneratedSignal', idea_signal_handler)
    event_bus.subscribe('IdeaCatalyzedSignal', idea_signal_handler)
    logger.info("Main test runner subscribed to 'IdeaGeneratedSignal' and 'IdeaCatalyzedSignal'")

    # Set up execution context
    run_id = f'generative_loop_{int(time.time())}'
    main_logger = logging.getLogger(run_id)
    base_context = NireonExecutionContext(
        run_id=run_id,
        component_registry=registry,
        event_bus=event_bus,
        config=bootstrap_result.global_config,
        feature_flags=bootstrap_result.global_config.get('feature_flags', {}),
        logger=main_logger
    )

    # --- Generative Loop ---
    num_iterations = 1
    current_idea_text = 'A detective discovers a parallel universe hidden in a coffee shop.'
    current_idea_obj: Optional[Idea] = None
    all_ideas_in_thread: List[Idea] = []
    parent_iteration_frame_id: Optional[str] = None

    for i in range(num_iterations):
        print('\n' + '=' * 80)
        logger.info(f"Iteration {i + 1}/{num_iterations} | Current Seed: '{current_idea_text[:100]}...'")
        print('=' * 80)

        set_current_log_component_id('TestOrchestrator')
        iteration_frame = await frame_factory.create_frame(
            context=base_context,
            name=f"iteration_frame_{i+1}",
            description=f"Main frame for generative loop iteration {i+1}",
            owner_agent_id="test_orchestrator",
            parent_frame_id=parent_iteration_frame_id,
            resource_budget={'llm_calls': 25, 'event_publishes': 60}
        )
        logger.info(f"Created main iteration frame: {iteration_frame.id}")
        parent_iteration_frame_id = iteration_frame.id
        
        # 1. Create the seed idea and ENSURE IT HAS A VECTOR
        set_current_log_component_id('IdeaService')
        current_idea_obj = idea_service.create_idea(text=current_idea_text, parent_id=current_idea_obj.idea_id if current_idea_obj else None)
        logger.info(f'Created new seed idea with ID: {current_idea_obj.idea_id}')
        
        # ** FIX: REMOVED `await` from the synchronous `encode` call **
        set_current_log_component_id('EmbeddingService')
        current_idea_obj.theta = embed_port.encode(current_idea_obj.text)
        logger.info(f"Encoded vector for seed idea {current_idea_obj.idea_id}")
        all_ideas_in_thread.append(current_idea_obj)

        # 2. Generate new ideas using BOTH Explorer and Catalyst
        set_current_log_component_id('MechanismRunner')
        logger.info(f'Dispatching idea {current_idea_obj.idea_id} to Explorer and Catalyst.')
        
        explorer_context = base_context.with_component_scope(explorer.component_id).with_metadata(current_frame_id=iteration_frame.id)
        catalyst_context = base_context.with_component_scope(catalyst.component_id).with_metadata(current_frame_id=iteration_frame.id)

        generation_tasks = [
            explorer.process({'ideas': [current_idea_obj], 'objective': 'Evolve the story with a surprising twist.'}, explorer_context),
            catalyst.process({'ideas': [current_idea_obj], 'objective': 'Synthesize with a new domain.', 'cross_domain_vectors': cross_domain_vectors}, catalyst_context)
        ]
        
        results = await asyncio.gather(*generation_tasks, return_exceptions=True)

        if isinstance(results[0], Exception) or not results[0].success:
            logger.error(f'Exploration failed, ending loop. Reason: {results[0]}')
            await frame_factory.update_frame_status(base_context, iteration_frame.id, "error_explorer")
            break
        if isinstance(results[1], Exception) or not results[1].success:
            logger.error(f'Catalysis failed, ending loop. Reason: {results[1]}')
            await frame_factory.update_frame_status(base_context, iteration_frame.id, "error_catalyst")
            break
        
        await asyncio.sleep(0.2) 

        if not newly_persisted_ideas:
            logger.warning('Neither Explorer nor Catalyst generated new ideas. Ending loop.')
            await frame_factory.update_frame_status(base_context, iteration_frame.id, "completed_no_ideas")
            break
        
        logger.info(f'Explorer & Catalyst generated and handler persisted a total of {len(newly_persisted_ideas)} new variations.')
        variations_to_assess = newly_persisted_ideas.copy()
        all_ideas_in_thread.extend(variations_to_assess)
        newly_persisted_ideas.clear()  

        # 3. Use Sentinel to assess the generated variations
        assessed_variations = []
        for variation in variations_to_assess:
            set_current_log_component_id(sentinel.component_id)
            sentinel_context = base_context.with_component_scope(sentinel.component_id).with_metadata(current_frame_id=iteration_frame.id)
            
            logger.info(f"Sentinel assessing variation '{variation.text[:60]}...'")
            sentinel_result = await sentinel.process(
                {'target_idea_id': variation.idea_id, 'reference_ideas': all_ideas_in_thread}, 
                sentinel_context
            )
            if sentinel_result.success:
                assessment_data = sentinel_result.output_data
                assessed_variations.append(assessment_data)
                logger.info(f"  -> Assessment: Stable={assessment_data.get('is_stable')}, Trust={assessment_data.get('trust_score', 0):.2f}")
            else:
                logger.error(f'  -> Sentinel failed to assess variation {variation.idea_id}: {sentinel_result.message}')
        
        # 4. Select the best idea for the next iteration
        stable_and_good = [a for a in assessed_variations if a and a.get('is_stable') and a.get('trust_score', 0) > sentinel.trust_th]
        if not stable_and_good:
            logger.warning('No stable and high-trust ideas found in this iteration. Ending loop.')
            await frame_factory.update_frame_status(base_context, iteration_frame.id, "completed_no_selection")
            break
            
        best_variation = max(stable_and_good, key=lambda a: a.get('trust_score', 0))
        best_idea_obj = idea_service.get_idea(best_variation['idea_id'])
        logger.info(f"Selected best idea for next iteration (ID: {best_idea_obj.idea_id}, Trust: {best_variation.get('trust_score', 0):.2f})")
        current_idea_text = best_idea_obj.text
        current_idea_obj = best_idea_obj

        await frame_factory.update_frame_status(base_context, iteration_frame.id, "completed_ok")

    print('\n' + '=' * 80)
    logger.info('Generative loop finished.')
    print('=' * 80)

if __name__ == '__main__':
    if Path.cwd().name == 'nireon_v4' and (PROJECT_ROOT / 'configs').is_dir():
        pass
    elif (Path.cwd() / 'configs').is_dir():
        pass
    elif PROJECT_ROOT and (PROJECT_ROOT / 'configs').is_dir():
        os.chdir(PROJECT_ROOT)
        print(f'Changed CWD to detected project root: {Path.cwd()}')
    else:
        print("Warning: Could not robustly detect project root. Assuming current CWD is correct.")
        
    asyncio.run(main())