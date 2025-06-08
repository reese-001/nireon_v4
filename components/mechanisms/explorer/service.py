# nireon_v4/components/mechanisms/explorer/service.py
import logging
import uuid # For generating unique task IDs for frame names
from typing import Any, Dict, Optional, List # Added List

from core.base_component import NireonBaseComponent
from core.lifecycle import ComponentMetadata
from core.results import ProcessResult
from domain.context import NireonExecutionContext
from domain.epistemic_stage import EpistemicStage # For LLM CEs if used
from domain.cognitive_events import CognitiveEvent, LLMRequestPayload # For CEs
from domain.ports.mechanism_gateway_port import MechanismGatewayPort # New dependency
from application.services.frame_factory_service import FrameFactoryService # New dependency

from .config import ExplorerConfig

logger = logging.getLogger(__name__)

EXPLORER_METADATA = ComponentMetadata(
    id='explorer_default_from_code', # This might be overridden by manifest
    name='Explorer Mechanism', # More descriptive
    version='0.2.0', # Incremented version due to refactor
    category='mechanism',
    description='Explorer Mechanism for idea generation and systematic variation, using Frames and MechanismGateway.',
    epistemic_tags=['generator', 'variation', 'mutator', 'innovator'], # Added innovator
    capabilities={'generate_ideas', 'explore_variations', 'idea_mutation'},
    accepts=['SEED_SIGNAL', 'EXPLORATION_REQUEST'], # Can be dict with seed or just string
    produces=['IDEA_GENERATED', 'EXPLORATION_COMPLETE'], # To be published via Gateway
    requires_initialize=True
)

class ExplorerMechanism(NireonBaseComponent):
    ConfigModel = ExplorerConfig

    def __init__(self,
                 config: Dict[str, Any],
                 metadata_definition: ComponentMetadata,
                 # common_deps is kept for now, could hold other non-gateway utils like rng
                 # If Explorer *only* uses Gateway for external, common_deps might simplify
                 common_deps: Optional[Any] = None,
                 # New dependencies to be injected by bootstrap/factory:
                 gateway: Optional[MechanismGatewayPort] = None,
                 frame_factory: Optional[FrameFactoryService] = None # type: ignore
                 ):
        super().__init__(config=config, metadata_definition=metadata_definition)
        self.cfg: ExplorerConfig = ExplorerConfig(**self.config) # Config parsed by NireonBaseComponent
        
        self.common_deps = common_deps # Store if still needed
        self.gateway = gateway
        self.frame_factory = frame_factory
        
        self._exploration_count = 0
        # self._generated_ideas = [] # This was for internal tracking, might not be needed if ideas are published immediately

        logger.info(
            f"ExplorerMechanism '{self.component_id}' (instance of {metadata_definition.name} "
            f"v{metadata_definition.version}) created. Gateway and FrameFactory "
            f"{'provided' if gateway and frame_factory else 'MISSING (will fail if not injected by init)'}."
        )
        logger.debug(
            f"Explorer config: max_depth={self.cfg.max_depth}, "
            f"application_rate={self.cfg.application_rate}, strategy={self.cfg.exploration_strategy}"
        )

    async def _initialize_impl(self, context: NireonExecutionContext) -> None:
        context.logger.info(f"ExplorerMechanism '{self.component_id}' initializing. Max depth: {self.cfg.max_depth}")
        
        # Resolve dependencies from registry if not injected during __init__
        # This is a common pattern if dependencies are complex or from a central DI system.
        if not self.gateway:
            self.gateway = context.component_registry.get_service_instance(MechanismGatewayPort)
        if not self.frame_factory:
            self.frame_factory = context.component_registry.get_service_instance(FrameFactoryService)

        if not self.gateway:
            raise RuntimeError(f"ExplorerMechanism '{self.component_id}' requires MechanismGatewayPort, but it was not found/injected.")
        if not self.frame_factory:
            raise RuntimeError(f"ExplorerMechanism '{self.component_id}' requires FrameFactoryService, but it was not found/injected.")

        # The old checks for llm_port/embedding_port on common_deps are less relevant
        # as those services would be accessed via the gateway.
        # The gateway itself would handle missing underlying ports.
        
        self._init_exploration_strategies(context) # No change here
        context.logger.info(f"✓ ExplorerMechanism '{self.component_id}' initialized successfully with Gateway and FrameFactory.")

    def _init_exploration_strategies(self, context: NireonExecutionContext) -> None:
        # (No changes to this method's logic)
        strategy = self.cfg.exploration_strategy
        if strategy == 'depth_first':
            context.logger.debug('Using depth-first exploration strategy')
        elif strategy == 'breadth_first':
            context.logger.debug('Using breadth-first exploration strategy')
        elif strategy == 'random':
            context.logger.debug('Using random exploration strategy')
        else:
            context.logger.warning(f"Unknown exploration strategy '{strategy}', defaulting to depth-first")


    async def _process_impl(self, data: Any, context: NireonExecutionContext) -> ProcessResult:
        task_id = str(uuid.uuid4())[:8] # For unique frame name per process call
        frame_name = f"explorer_{self.component_id}_task_{task_id}"
        
        # 1. Acquire/Create a Frame
        try:
            # Using get_or_create_frame to potentially reuse frames if appropriate,
            # or create_frame for guaranteed new context. For exploration, new frame is often better.
            current_frame = await self.frame_factory.create_frame(
                context=context, # Pass current execution context
                name=frame_name,
                owner_agent_id=self.component_id,
                description=f"Exploration task based on input: {str(data)[:50]}...",
                epistemic_goals=["DIVERGENCE", "NOVELTY_SEEKING"],
                # llm_policy can be set here if explorer has specific LLM needs for this frame
                llm_policy=self.cfg.get_exploration_params().get("llm_policy_override", {}) 
            )
            current_frame_id = current_frame.id
            context.logger.info(f"ExplorerMechanism '{self.component_id}' processing in Frame '{current_frame_id}' for request: {str(data)[:100]}")
        except Exception as e:
            context.logger.error(f"Explorer '{self.component_id}' failed to acquire frame: {e}", exc_info=True)
            return ProcessResult(success=False, component_id=self.component_id, message=f'Frame acquisition failed: {e}', error_code='FRAME_ACQUISITION_ERROR')

        self._exploration_count += 1
        try:
            seed_idea_text = self._extract_seed_idea(data, context)
            
            # If Explorer uses LLM for variations (not in current example, but for future):
            # llm_payload = LLMRequestPayload(prompt=f"Generate variations for: {seed_idea_text}", ...)
            # ce_llm = CognitiveEvent.for_llm_ask(frame_id=current_frame_id, owning_agent_id=self.component_id, ...)
            # llm_response = await self.gateway.ask_llm(ce_llm)
            # variations_from_llm = parse_llm_response_for_variations(llm_response.text)

            variations_data = await self._generate_variations(seed_idea_text, context) # This is list of strings currently

            # 2. Publish IDEA_GENERATED events for each variation
            generated_idea_details = []
            for var_text in variations_data:
                # In a more complete system, these would be proper Idea objects
                idea_payload = {
                    "text": var_text, 
                    "source_mechanism": self.component_id,
                    "derivation_method": self.cfg.exploration_strategy,
                    "seed_idea": seed_idea_text,
                    "frame_id": current_frame_id
                }
                generated_idea_details.append({"text_preview": var_text[:50]+"..."})

                ce_idea_generated = CognitiveEvent(
                    frame_id=current_frame_id,
                    owning_agent_id=self.component_id,
                    service_call_type="EVENT_PUBLISH",
                    payload={"event_type": "IDEA_GENERATED", "event_data": idea_payload},
                    epistemic_intent="OUTPUT_GENERATED_IDEA"
                )
                await self.gateway.process_cognitive_event(ce_idea_generated)
            
            # 3. Publish EXPLORATION_COMPLETE event
            exploration_complete_payload = {
                "exploration_id": f'exp_{self.component_id}_{self._exploration_count}',
                "seed_idea": seed_idea_text,
                "variations_generated_count": len(variations_data),
                "exploration_strategy": self.cfg.exploration_strategy,
                "total_explorations_by_component": self._exploration_count,
                "frame_id": current_frame_id
            }
            ce_exploration_complete = CognitiveEvent(
                frame_id=current_frame_id,
                owning_agent_id=self.component_id,
                service_call_type="EVENT_PUBLISH",
                payload={"event_type": "EXPLORATION_COMPLETE", "event_data": exploration_complete_payload},
                epistemic_intent="SIGNAL_TASK_COMPLETION"
            )
            await self.gateway.process_cognitive_event(ce_exploration_complete)

            result_data = {
                **exploration_complete_payload, 
                "generated_idea_previews": generated_idea_details
            } # For the ProcessResult

            context.logger.info(f"✓ Explorer '{self.component_id}' generated and published {len(variations_data)} variations via Gateway.")
            
            # Mark frame as completed (optional, could be done by an orchestrator)
            await self.frame_factory.update_frame_status(context, current_frame_id, "completed")

            return ProcessResult(
                success=True, component_id=self.component_id, 
                output_data=result_data, 
                message=f'Generated and published {len(variations_data)} idea variations using {self.cfg.exploration_strategy} strategy within Frame {current_frame_id}.'
            )
        except Exception as e:
            context.logger.error(f"Explorer '{self.component_id}' processing failed in Frame '{current_frame_id}': {e}", exc_info=True)
            # Attempt to mark frame as error
            try:
                await self.frame_factory.update_frame_status(context, current_frame_id, "error")
            except Exception as frame_e:
                 context.logger.error(f"Failed to update frame '{current_frame_id}' status to error: {frame_e}")
            return ProcessResult(success=False, component_id=self.component_id, message=f'Exploration failed: {e}', error_code='EXPLORATION_ERROR')

    # _extract_seed_idea, _generate_variations, and its sub-methods (_depth_first_exploration etc.)
    # remain largely the same internally as they don't make external calls in the current example.
    # If they did (e.g., using LLM for variation), those calls would need to be wrapped in CEs.

    def _extract_seed_idea(self, data: Any, context: NireonExecutionContext) -> str:
        # (No changes to this method's logic)
        if isinstance(data, str):
            return data
        elif isinstance(data, dict):
            if 'text' in data:
                return data['text']
            elif 'seed' in data:
                return data['seed']
            elif 'idea' in data: # common key for Idea objects
                if isinstance(data['idea'], dict) and 'text' in data['idea']:
                    return data['idea']['text']
                elif isinstance(data['idea'], str):
                    return data['idea']
        default_seed = 'Explore new possibilities and generate innovative ideas'
        context.logger.debug(f'Using default seed idea for exploration: {default_seed}')
        return default_seed

    async def _generate_variations(self, seed_idea: str, context: NireonExecutionContext) -> list[str]:
        # (No changes to this method's logic, as it's programmatic)
        variations = []
        # ... (depth_first, breadth_first, random logic as before) ...
        try:
            if self.cfg.exploration_strategy == 'depth_first':
                variations = await self._depth_first_exploration(seed_idea, context)
            elif self.cfg.exploration_strategy == 'breadth_first':
                variations = await self._breadth_first_exploration(seed_idea, context)
            elif self.cfg.exploration_strategy == 'random':
                variations = await self._random_exploration(seed_idea, context)
            else: # Fallback to simple if strategy is unknown or for some error
                context.logger.warning(f"Unknown or default strategy '{self.cfg.exploration_strategy}' - using simple variations.")
                variations = await self._simple_variations(seed_idea, context)
        except Exception as e:
            context.logger.error(f'Variation generation failed: {e}')
            variations = await self._simple_variations(seed_idea, context) # Fallback
        return variations
        
    async def _depth_first_exploration(self, seed_idea: str, context: NireonExecutionContext) -> list[str]:
        # (No changes to this method's logic)
        variations = []
        current_depth = 0
        current_ideas = [seed_idea]
        max_vars_total = self.cfg.max_variations_per_level * self.cfg.max_depth # A soft cap

        while current_depth < self.cfg.max_depth and len(variations) < max_vars_total:
            next_level_ideas = []
            for idea_to_explore in current_ideas:
                if len(variations) >= max_vars_total: break
                # Generate up to max_variations_per_level from this idea
                for i in range(self.cfg.max_variations_per_level):
                    if len(variations) >= max_vars_total: break
                    new_variation = f'Depth {current_depth+1}, Var {i+1} from "{idea_to_explore[:20]}...": New angle on {idea_to_explore}'
                    variations.append(new_variation)
                    next_level_ideas.append(new_variation) # For next depth level
            current_ideas = next_level_ideas[:1] # True depth first only explores one path typically
            if not current_ideas: break # No path to continue
            current_depth += 1
        context.logger.debug(f'Depth-first exploration generated {len(variations)} variations up to depth {current_depth}')
        return variations

    async def _breadth_first_exploration(self, seed_idea: str, context: NireonExecutionContext) -> list[str]:
        # (No changes to this method's logic)
        variations = []
        queue = [(seed_idea, 0)] # (idea, depth)
        visited_ideas_count = 0
        max_vars_total = self.cfg.max_variations_per_level * self.cfg.max_depth

        while queue and len(variations) < max_vars_total:
            current_idea, depth = queue.pop(0)
            visited_ideas_count += 1
            if depth >= self.cfg.max_depth:
                continue

            for i in range(self.cfg.max_variations_per_level):
                if len(variations) >= max_vars_total: break
                new_variation = f'Breadth {depth+1}, Var {i+1} from "{current_idea[:20]}...": Expanded view of {current_idea}'
                variations.append(new_variation)
                queue.append((new_variation, depth + 1))
        
        context.logger.debug(f'Breadth-first exploration generated {len(variations)} variations from {visited_ideas_count} nodes.')
        return variations

    async def _random_exploration(self, seed_idea: str, context: NireonExecutionContext) -> list[str]:
        # (No changes to this method's logic)
        import random
        variation_templates = [
            'Random mutation: {seed} with unexpected twist', 
            'Chance combination: {seed} meets serendipity',
            'Stochastic enhancement: {seed} through probability',
            'Emergent property: {seed} with chaotic elements',
            'Quantum variation: {seed} in superposition'
        ]
        num_to_generate = min(self.cfg.max_variations_per_level * self.cfg.max_depth, self.cfg.max_variations_per_level * 2) # Cap random
        
        variations = []
        for _ in range(num_to_generate):
            template = random.choice(variation_templates)
            variations.append(template.format(seed=seed_idea))
            
        context.logger.debug(f'Random exploration generated {len(variations)} variations')
        return variations

    async def _simple_variations(self, seed_idea: str, context: NireonExecutionContext) -> list[str]:
        # (No changes to this method's logic)
        variations = [
            f'Basic variation 1: {seed_idea} with improvement',
            f'Basic variation 2: {seed_idea} with modification',
            f'Basic variation 3: Alternative to {seed_idea}'
        ]
        context.logger.debug(f'Simple exploration generated {len(variations)} variations')
        return variations[:self.cfg.max_variations_per_level]


    async def analyze(self, context: NireonExecutionContext):
        # (No changes to this method's logic, assuming it doesn't make external calls)
        from core.results import AnalysisResult
        # In a real scenario, self._generated_ideas would need to be persisted or managed differently
        # if this analyze method is called across multiple process calls.
        # For now, it reflects the state of the last process call if it stored ideas locally.
        # A better approach would be to query an IdeaRepository or aggregate from event logs.
        
        # This metric will be low if _generated_ideas is not persisted across calls
        # total_ideas_ever_generated = ... (this would need persistent tracking or event sourcing)
        
        metrics = {
            'total_explorations_this_instance': self._exploration_count,
            # 'total_ideas_generated_this_instance': len(self._generated_ideas), # No longer storing locally
            'exploration_strategy': self.cfg.exploration_strategy,
            'max_configured_depth': self.cfg.max_depth,
            'configured_application_rate': self.cfg.application_rate
        }
        return AnalysisResult(
            success=True, component_id=self.component_id, 
            metrics=metrics, confidence=0.8, 
            message=f'Explorer analysis: {self._exploration_count} explorations performed by this instance.'
        )

    async def health_check(self, context: NireonExecutionContext):
        # (No changes to this method's logic, assuming it doesn't make external calls)
        from core.results import ComponentHealth
        status = 'HEALTHY'
        messages = []
        if not self.is_initialized:
            status = 'UNHEALTHY' # NireonBaseComponent handles this
            messages.append('Explorer not initialized')
        elif not self.gateway or not self.frame_factory:
            status = 'DEGRADED'
            messages.append('Explorer missing critical dependencies (Gateway or FrameFactory).')
        
        if self.error_count > 0: # From NireonBaseComponent
            status = 'DEGRADED' if status == 'HEALTHY' else status
            messages.append(f'Explorer has encountered {self.error_count} errors.')
        
        if self._exploration_count == 0:
            messages.append('No explorations performed yet by this instance.')
        
        final_message = '; '.join(messages) if messages else 'Explorer operational with Gateway and FrameFactory.'
        return ComponentHealth(component_id=self.component_id, status=status, message=final_message)