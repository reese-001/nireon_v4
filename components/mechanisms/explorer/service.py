# nireon_v4\components\mechanisms\explorer\service.py
import logging
import uuid
from typing import Any, Dict, Optional, List
import random
import asyncio
import shortuuid
import hashlib
from datetime import datetime, timezone
import time
import json
from core.base_component import NireonBaseComponent
from core.lifecycle import ComponentMetadata
from core.results import ProcessResult, ComponentHealth, AnalysisResult, SystemSignal, AdaptationAction, AdaptationActionType, SignalType
from collections import deque, defaultdict
from domain.context import NireonExecutionContext
from domain.cognitive_events import CognitiveEvent, LLMRequestPayload
from domain.epistemic_stage import EpistemicStage
from domain.ports.mechanism_gateway_port import MechanismGatewayPort
from domain.ports.llm_port import LLMResponse
from application.services.frame_factory_service import FrameFactoryService, FrameNotFoundError
from domain.frames import Frame
from signals.core import IdeaGeneratedSignal, TrustAssessmentSignal
from signals.base import EpistemicSignal
from .config import ExplorerConfig
from .service_helpers.explorer_event_helper import ExplorerEventHelper
from .errors import ExplorerErrorCode

logger = logging.getLogger(__name__)

EXPLORER_METADATA = ComponentMetadata(
    id='explorer', name='Explorer Mechanism V4', version='1.6.0', category='mechanism',
    description='Explorer Mechanism for idea generation and systematic variation, using A-F-CE model (Frames and MechanismGateway).',
    epistemic_tags=['generator', 'variation', 'mutator', 'innovator', 'divergence', 'novelty_generation'],
    capabilities={'generate_ideas', 'explore_variations', 'idea_mutation', 'dynamic_frame_parameterization'},
    accepts=['SEED_SIGNAL', 'EXPLORATION_REQUEST'],
    produces=['IdeaGeneratedSignal', 'ExplorationCompleteSignal', 'FrameProcessingFailedSignal', 'TrustAssessmentSignal'],
    requires_initialize=True,
    dependencies={'MechanismGatewayPort': '>=1.0.0', 'FrameFactoryService': '*'}
)

class ExplorerMechanism(NireonBaseComponent):
    METADATA_DEFINITION = EXPLORER_METADATA
    ConfigModel = ExplorerConfig

    def __init__(self, config: Dict[str, Any], metadata_definition: ComponentMetadata, gateway: Optional[MechanismGatewayPort]=None, frame_factory: Optional[FrameFactoryService]=None):
        super().__init__(config=config, metadata_definition=metadata_definition or self.METADATA_DEFINITION)
        self.cfg: ExplorerConfig = ExplorerConfig(**self.config)
        self.gateway: Optional[MechanismGatewayPort] = gateway
        self.frame_factory: Optional[FrameFactoryService] = frame_factory
        self.event_helper: Optional[ExplorerEventHelper] = None
        self._exploration_count = 0
        self._rng_frame_specific: Optional[random.Random] = None
        self._pending_embedding_requests: Dict[str, Any] = {}
        self.last_n_frame_stats = deque(maxlen=10)
        self._assessment_events: Dict[str, asyncio.Event] = {}
        self._frame_assessment_trackers: Dict[str, Dict[str, asyncio.Event]] = defaultdict(dict)
        logger.info(f"ExplorerMechanism '{self.component_id}' (instance of {self.metadata.name} v{self.metadata.version}) created. Gateway and FrameFactory will be resolved during initialization.")
        logger.debug(f'Explorer initial config: {self.cfg.model_dump_json(indent=2)}')
        
    async def _initialize_impl(self, context: NireonExecutionContext) -> None:
        context.logger.info(f"ExplorerMechanism '{self.component_id}' initializing.")
        if not self.gateway:
            try:
                self.gateway = context.component_registry.get_service_instance(MechanismGatewayPort)
                context.logger.info(f"Resolved MechanismGatewayPort for '{self.component_id}'.")
            except Exception as e:
                context.logger.error(f"Failed to resolve MechanismGatewayPort for '{self.component_id}': {e}", exc_info=True)
                raise RuntimeError(f"ExplorerMechanism '{self.component_id}' requires MechanismGatewayPort.")

        if not self.frame_factory:
            try:
                self.frame_factory = context.component_registry.get_service_instance(FrameFactoryService)
                context.logger.info(f"Resolved FrameFactoryService for '{self.component_id}'.")
            except Exception as e:
                context.logger.error(f"Failed to resolve FrameFactoryService for '{self.component_id}': {e}", exc_info=True)
                raise RuntimeError(f"ExplorerMechanism '{self.component_id}' requires FrameFactoryService.")
        
        self.event_helper = ExplorerEventHelper(self.gateway, self.component_id, self.metadata.version, registry=context.component_registry)
        context.logger.info(f"Initialized ExplorerEventHelper for '{self.component_id}'.")
        
        if context.event_bus:
            context.event_bus.subscribe(TrustAssessmentSignal.__name__, self._handle_trust_assessment)
            context.logger.info(f"Explorer '{self.component_id}' subscribed to TrustAssessmentSignal.")

        self._init_exploration_strategies(context)
        context.logger.info(f"✓ ExplorerMechanism '{self.component_id}' initialized successfully.")
        
    async def _handle_trust_assessment(self, signal_payload: Dict[str, Any]):
        idea_id = signal_payload.get('target_id')
        frame_id = signal_payload.get('context_tags', {}).get('frame_id')
        
        if not idea_id or not frame_id:
            logger.debug(f'Received TrustAssessmentSignal without idea_id or frame_id in tags. Ignoring.')
            return

        if frame_id in self._frame_assessment_trackers and idea_id in self._frame_assessment_trackers[frame_id]:
            logger.info(f"Explorer received trust assessment for idea '{idea_id}' in frame '{frame_id}'. Marking as complete.")
            event = self._frame_assessment_trackers[frame_id][idea_id]
            if not event.is_set():
                event.set()

    def _init_exploration_strategies(self, context: NireonExecutionContext) -> None:
        strategy = self.cfg.exploration_strategy
        context.logger.debug(f"Explorer '{self.component_id}' configured with strategy: {strategy}")

    def _build_llm_prompt(self, seed_text: str, objective: str, attempt_details: Optional[Dict[str, Any]]=None) -> str:
        vector_distance_placeholder = attempt_details.get('vector_distance', 0.0) if attempt_details else 0.0
        template_vars = {
            'seed_idea_text': seed_text,
            'objective': objective or 'explore novel variations',
            'vector_distance': vector_distance_placeholder,
            'creativity_factor_desc': 'high' if self.cfg.creativity_factor > 0.7 else 'medium' if self.cfg.creativity_factor > 0.3 else 'low',
            'desired_length_min': self.cfg.minimum_idea_length,
            'desired_length_max': self.cfg.maximum_idea_length
        }
        prompt_template_to_use = self.cfg.default_prompt_template or "Generate a creative and divergent variation of the following idea: '{seed_idea_text}'. The overall objective is: {objective}. Aim for a {creativity_factor_desc} degree of novelty. The generated idea should be between {desired_length_min} and {desired_length_max} characters. Respond with ONLY the full text of the new, varied idea."
        try:
            return prompt_template_to_use.format(**template_vars)
        except KeyError as e:
            logger.warning(f'[{self.component_id}] Prompt template missing key: {e}. Using basic prompt.')
            return f'Generate a creative variation of: {seed_text}. Objective: {objective}.'

    def _add_audit_log(self, frame: Optional[Frame], event_type: str, summary: str, details: Optional[Dict[str, Any]]=None) -> None:
        if not frame or not hasattr(frame, 'context_tags') or not isinstance(frame.context_tags.get('audit_trail'), list):
            logger.warning(f'[{self.component_id}] Cannot add audit log: Frame or audit_trail invalid. Event: {event_type}')
            return
        
        entry = {'ts': datetime.now(timezone.utc).isoformat(), 'event_type': event_type, 'summary': summary}
        if details:
            entry['details'] = details
        
        audit_trail: List[Dict] = frame.context_tags['audit_trail']
        
        MAX_AUDIT_ENTRIES_IN_FRAME = 50
        MAX_AUDIT_STRING_LENGTH = 1024 * 10
        current_audit_size = sum(len(json.dumps(item)) for item in audit_trail)
        
        if len(audit_trail) >= MAX_AUDIT_ENTRIES_IN_FRAME or current_audit_size > MAX_AUDIT_STRING_LENGTH:
            logger.warning(f"Audit trail for frame {frame.id} near/exceeds limit ({len(audit_trail)} entries, {current_audit_size} bytes). Logging event '{event_type}' to standard logger instead of frame. Summary: {summary}. Details: {details}. TODO (Platform): Implement MechanismGateway.archive_frame_artifact for spill.")
            logger.info(f'[FrameAuditSpill][{frame.id}] Event: {event_type}, Summary: {summary}, Details: {details}')
        else:
            audit_trail.append(entry)
            
    async def _request_embedding_for_idea(self, frame: Frame, idea_text: str, idea_id: str, context: NireonExecutionContext) -> None:
        if not self.event_helper:
            context.logger.error(f'[{self.component_id}] Event helper not available. Cannot request embedding for idea {idea_id}.')
            return

        if len(self._pending_embedding_requests) >= self.cfg.max_pending_embedding_requests:
            context.logger.warning(f'[{self.component_id}] Max pending embedding requests ({self.cfg.max_pending_embedding_requests}) reached. Skipping embedding request for idea {idea_id} in frame {frame.id}.')
            self._add_audit_log(frame, 'EMBEDDING_REQUEST_SKIPPED', f'Max pending requests reached, skipped for idea {idea_id}.', {'target_idea_id': idea_id})
            return

        request_id = f'emb_req_{self.component_id}_{shortuuid.uuid()}'
        embedding_request_payload = {
            'request_id': request_id,
            'text_to_embed': idea_text,
            'target_artifact_id': idea_id,
            'request_timestamp_ms': int(time.time() * 1000),
            'embedding_vector_dtype': 'float32',
            'metadata': {**self.cfg.embedding_request_metadata, 'frame_id': frame.id, 'origin_component_id': self.component_id}
        }
        self._pending_embedding_requests[request_id] = {'idea_id': idea_id, 'frame_id': frame.id, 'text': idea_text, 'timestamp': time.time()}
        
        self._add_audit_log(frame, 'EMBEDDING_REQUESTED', f'Embedding requested for idea {idea_id}.', {'target_idea_id': idea_id, 'embedding_request_id': request_id})
        
        embedding_signal = EpistemicSignal(signal_type='EmbeddingRequestSignal', source_node_id=self.component_id, payload=embedding_request_payload, context_tags={'frame_id': frame.id})
        await self.event_helper.publish_signal(embedding_signal, context)
        context.logger.info(f"[{self.component_id}] Requested embedding for idea '{idea_id}' (request_id: {request_id}) in frame '{frame.id}'.")
    
    async def _process_impl(self, data: Any, context: NireonExecutionContext) -> ProcessResult:
        self._exploration_count += 1
        task_short_id = shortuuid.uuid()[:8]
        seed_input_text = ''
        seed_input_id = f'seed_{task_short_id}'
        
        # Extract depth from incoming data, default to 0 if not present
        current_depth = 0
        if isinstance(data, dict) and 'metadata' in data and isinstance(data['metadata'], dict):
            current_depth = data['metadata'].get('depth', 0)

        if isinstance(data, dict) and 'ideas' in data and data['ideas']:
            seed_idea_obj = data['ideas'][0]
            seed_input_text = seed_idea_obj.text
            seed_input_id = seed_idea_obj.idea_id
        elif isinstance(data, dict) and 'text' in data:
            seed_input_text = data['text']
            seed_input_id = data.get('id', seed_input_id)
        elif isinstance(data, str):
            seed_input_text = data
        else:
            context.logger.warning(f'Unrecognized input data structure for Explorer. Using default seed.')
            seed_input_text = f'default_seed_for_task_{task_short_id}'
        
        input_hash = hashlib.sha256(seed_input_text.encode()).hexdigest()[:12]
        objective_from_data = data.get('objective', 'Generate novel idea variations.') if isinstance(data, dict) else 'Generate novel idea variations.'
        
        frame_name = f'explorer_task_{task_short_id}_on_seed_{input_hash}'
        frame_description = f"Exploration task (Depth: {current_depth}) initiated by '{self.component_id}' for seed idea (hash: {input_hash}, preview: '{seed_input_text[:30]}...') using strategy '{self.cfg.exploration_strategy}'."
        parent_frame_id: Optional[str] = context.metadata.get('current_frame_id') if context.metadata else None
        
        if not parent_frame_id:
            context.logger.warning(f'parent_frame_id not found for Explorer task {task_short_id}. Creating top-level frame.')

        audit_trail_entries: List[Dict[str, Any]] = [{'ts': datetime.now(timezone.utc).isoformat(), 'event_type': 'EXPLORER_TASK_STARTED', 'summary': 'Explorer task processing initiated.'}]
        context_tags = {
            'explorer_version': self.metadata.version,
            'seed_idea_id': seed_input_id,
            'exploration_strategy_used': self.cfg.exploration_strategy,
            'initial_input_hash': input_hash,
            'depth': current_depth, # Pass depth into the frame
            'adaptive_parameters_log': [],
            'rng_seed_used': 'PENDING_FRAME_RNG_IMPL',
            'audit_trail': audit_trail_entries,
            'gateway_info': {
                'gateway_id': self.gateway.component_id if self.gateway else 'unknown',
                'gateway_version': self.gateway.metadata.version if self.gateway and hasattr(self.gateway, 'metadata') else 'unknown',
                'gateway_type': self.gateway.metadata.category if self.gateway and hasattr(self.gateway, 'metadata') else 'unknown'
            }
        }
        
        current_frame: Optional[Frame] = None
        frame_final_status: str = 'pending'

        try:
            context.logger.info(f"Requesting ExplorationFrame: Name='{frame_name}'")
            current_frame = await self.frame_factory.create_frame(
                context=context,
                name=frame_name,
                description=frame_description,
                owner_agent_id=self.component_id,
                parent_frame_id=parent_frame_id,
                epistemic_goals=['DIVERGENCE', 'NOVELTY_GENERATION', 'IDEA_VARIATION'],
                trust_basis={'seed_idea_trust': 0.75},
                resource_budget=self.cfg.default_resource_budget_for_exploration,
                context_tags=context_tags,
                initial_status='active'
            )
            context.logger.info(f'Successfully created ExplorationFrame ID: {current_frame.id}, Name: {current_frame.name}')
            self._add_audit_log(current_frame, 'FRAME_CREATED', f'Frame {current_frame.id} created successfully.')
            
            context.logger.info(f"[{self.component_id}] Starting exploration logic within Frame '{current_frame.id}'...")
            self._add_audit_log(current_frame, 'EXPLORATION_STARTED', 'Core exploration logic initiated.')

            llm_tasks = []
            generated_idea_ids = []
            num_variations_to_generate = self.cfg.max_variations_per_level
            
            for i in range(num_variations_to_generate):
                prompt = self._build_llm_prompt(seed_input_text, objective_from_data, {'attempt_index': i})
                llm_payload = LLMRequestPayload(prompt=prompt, stage=EpistemicStage.EXPLORATION, role='idea_generator', llm_settings={})
                
                ce_llm_request = CognitiveEvent(
                    frame_id=current_frame.id,
                    owning_agent_id=self.component_id,
                    service_call_type='LLM_ASK',
                    payload=llm_payload,
                    epistemic_intent='GENERATE_IDEA_VARIATION',
                    custom_metadata={'explorer_attempt': i + 1, 'seed_hash': input_hash, 'schema_version': 1}
                )
                llm_tasks.append(self.gateway.process_cognitive_event(ce_llm_request, context))
                self._add_audit_log(current_frame, 'LLM_CE_CREATED', f'LLM_ASK CE created for attempt {i + 1}.', {'ce_id_related': ce_llm_request.event_id})

            llm_responses = await asyncio.gather(*llm_tasks, return_exceptions=True)

            frame_tracker = self._frame_assessment_trackers[current_frame.id]
            for i, resp_or_exc in enumerate(llm_responses):
                if isinstance(resp_or_exc, LLMResponse) and resp_or_exc.text and not resp_or_exc.get('error'):
                    variation_text = resp_or_exc.text.strip()
                    if len(variation_text) >= self.cfg.minimum_idea_length:
                        # Pass depth to the new idea's metadata
                        new_idea_metadata = {'depth': current_depth + 1}
                        new_idea = self.event_helper.create_and_persist_idea(
                            text=variation_text, 
                            parent_id=seed_input_id, 
                            context=context,
                            metadata=new_idea_metadata
                        )
                        generated_idea_ids.append(new_idea.idea_id)
                        
                        assessment_event = asyncio.Event()
                        frame_tracker[new_idea.idea_id] = assessment_event

                        idea_signal_payload = {
                            'id': new_idea.idea_id,
                            'text': variation_text,
                            'parent_id': seed_input_id,
                            'source_mechanism': self.component_id,
                            'derivation_method': self.cfg.exploration_strategy,
                            'seed_idea_text_preview': seed_input_text[:50],
                            'frame_id': current_frame.id,
                            'llm_response_metadata': {k: v for k, v in resp_or_exc.items() if k != 'text'}
                        }
                        idea_signal = IdeaGeneratedSignal(
                            source_node_id=self.component_id,
                            idea_id=new_idea.idea_id,
                            idea_content=variation_text,
                            generation_method=self.cfg.exploration_strategy,
                            payload=idea_signal_payload,
                            context_tags={'frame_id': current_frame.id}
                        )
                        await self.event_helper.publish_signal(idea_signal, context)
                        self._add_audit_log(current_frame, 'IDEA_GENERATED', f'Variation {i + 1} generated and published.', {'idea_id': new_idea.idea_id})
                else:
                    context.logger.error(f'LLM call {i + 1} failed: {resp_or_exc}')

            if frame_tracker:
                assessment_tasks = [event.wait() for event in frame_tracker.values()]
                try:
                    assessment_timeout = 60.0
                    logger.info(f'Explorer waiting for {len(assessment_tasks)} assessments in frame {current_frame.id} (timeout: {assessment_timeout}s)...')
                    await asyncio.wait_for(asyncio.gather(*assessment_tasks), timeout=assessment_timeout)
                    logger.info(f'All {len(assessment_tasks)} assessments received for frame {current_frame.id}.')
                    self._add_audit_log(current_frame, 'ASSESSMENTS_COMPLETE', f'All {len(assessment_tasks)} generated ideas have been assessed.')
                except asyncio.TimeoutError:
                    logger.warning(f'Timeout waiting for assessments in frame {current_frame.id}. Continuing with partial data.')
                    self._add_audit_log(current_frame, 'ASSESSMENT_TIMEOUT', 'Timeout waiting for all idea assessments.')
            
            if current_frame.id in self._frame_assessment_trackers:
                del self._frame_assessment_trackers[current_frame.id]

            if not generated_idea_ids:
                frame_final_status = 'completed_degraded'
                message = f'Explorer task {task_short_id} completed with degradation in frame {current_frame.id}: No variations generated.'
            else:
                frame_final_status = 'completed_ok'
                message = f'Explorer task {task_short_id} completed successfully in frame {current_frame.id}.'

            exploration_complete_payload = {
                'exploration_id': f'exp_{self.component_id}_{self._exploration_count}_{task_short_id}',
                'seed_idea_id': seed_input_id,
                'variations_generated_count': len(generated_idea_ids),
                'exploration_strategy': self.cfg.exploration_strategy,
                'frame_id': current_frame.id,
                'status': frame_final_status
            }
            exploration_complete_signal = EpistemicSignal(signal_type='ExplorationCompleteSignal', source_node_id=self.component_id, payload=exploration_complete_payload, context_tags={'frame_id': current_frame.id})
            await self.event_helper.publish_signal(exploration_complete_signal, context)
            self._add_audit_log(current_frame, 'SIGNAL_PUBLISHED', 'ExplorationCompleteSignal published.')
            
            await self.frame_factory.update_frame_status(context, current_frame.id, frame_final_status)
            context.logger.info(f"ExplorationFrame '{current_frame.id}' final status: '{frame_final_status}'.")
            self._add_audit_log(current_frame, 'FRAME_STATUS_UPDATED', f'Frame status set to {frame_final_status}.')
            
            return ProcessResult(
                success=True,
                component_id=self.component_id,
                output_data=exploration_complete_payload,
                message=message
            )
        except Exception as e:
            context.logger.error(f'Explorer processing failed critically: {e}', exc_info=True)
            current_frame_id_for_error = current_frame.id if current_frame else None
            if current_frame:
                await self.frame_factory.update_frame_status(context, current_frame.id, 'error_internal')
            return ProcessResult(
                success=False,
                component_id=self.component_id,
                message=f'Explorer critical error: {e}',
                error_code=str(ExplorerErrorCode.EXPLORER_PROCESSING_ERROR),
                metadata={'frame_id': current_frame_id_for_error}
            )

    async def _generate_variations(self, seed_idea: str, context: NireonExecutionContext) -> List[str]:
        variations = []
        try:
            if self.cfg.exploration_strategy == 'depth_first':
                variations = await self._depth_first_exploration(seed_idea, context)
            elif self.cfg.exploration_strategy == 'breadth_first':
                variations = await self._breadth_first_exploration(seed_idea, context)
            elif self.cfg.exploration_strategy == 'random':
                variations = await self._random_exploration(seed_idea, context)
            else:
                context.logger.warning(f"Unknown or default strategy '{self.cfg.exploration_strategy}' - using simple variations.")
                variations = await self._simple_variations(seed_idea, context)
        except Exception as e:
            context.logger.error(f'Variation generation failed: {e}')
            variations = await self._simple_variations(seed_idea, context)
        return variations

    async def _depth_first_exploration(self, seed_idea: str, context: NireonExecutionContext) -> List[str]:
        variations = []
        current_depth = 0
        current_ideas = [seed_idea]
        max_vars_total = self.cfg.max_variations_per_level * self.cfg.max_depth
        while current_depth < self.cfg.max_depth and len(variations) < max_vars_total:
            next_level_ideas = []
            for idea_to_explore in current_ideas:
                if len(variations) >= max_vars_total: break
                for i in range(self.cfg.max_variations_per_level):
                    if len(variations) >= max_vars_total: break
                    new_variation = f'Depth {current_depth+1}, Var {i+1} from "{idea_to_explore[:20]}...": New angle on {idea_to_explore}'
                    variations.append(new_variation)
                    next_level_ideas.append(new_variation)
            current_ideas = next_level_ideas[:1] 
            if not current_ideas: break
            current_depth += 1
        context.logger.debug(f'Depth-first exploration generated {len(variations)} variations up to depth {current_depth}')
        return variations

    async def _breadth_first_exploration(self, seed_idea: str, context: NireonExecutionContext) -> List[str]:
        variations = []
        queue = [(seed_idea, 0)]
        visited_ideas_count = 0
        max_vars_total = self.cfg.max_variations_per_level * self.cfg.max_depth
        while queue and len(variations) < max_vars_total:
            current_idea, depth = queue.pop(0)
            visited_ideas_count += 1
            if depth >= self.cfg.max_depth: continue
            for i in range(self.cfg.max_variations_per_level):
                if len(variations) >= max_vars_total: break
                new_variation = f'Breadth {depth+1}, Var {i+1} from "{current_idea[:20]}...": Expanded view of {current_idea}'
                variations.append(new_variation)
                queue.append((new_variation, depth + 1))
        context.logger.debug(f'Breadth-first exploration generated {len(variations)} variations from {visited_ideas_count} nodes.')
        return variations

    async def _random_exploration(self, seed_idea: str, context: NireonExecutionContext) -> List[str]:
        import random
        variation_templates = [
            'Random mutation: {seed} with unexpected twist',
            'Chance combination: {seed} meets serendipity',
            'Stochastic enhancement: {seed} through probability',
            'Emergent property: {seed} with chaotic elements',
            'Quantum variation: {seed} in superposition'
        ]
        num_to_generate = min(self.cfg.max_variations_per_level * self.cfg.max_depth, self.cfg.max_variations_per_level * 2)
        variations = [random.choice(variation_templates).format(seed=seed_idea) for _ in range(num_to_generate)]
        context.logger.debug(f'Random exploration generated {len(variations)} variations')
        return variations
    
    async def _simple_variations(self, seed_idea: str, context: NireonExecutionContext) -> List[str]:
        variations = [
            f'Basic variation 1: {seed_idea} with improvement',
            f'Basic variation 2: {seed_idea} with modification',
            f'Alternative to {seed_idea}'
        ]
        context.logger.debug(f'Simple exploration generated {len(variations)} variations')
        return variations[:self.cfg.max_variations_per_level]

    async def analyze(self, context: NireonExecutionContext) -> AnalysisResult:
        context.logger.debug(f'[{self.component_id}] analyze() called.')
        metrics = {
            'total_explorations_by_instance': self._exploration_count,
            'configured_strategy': self.cfg.exploration_strategy,
            'recent_frames_analyzed': len(self.last_n_frame_stats),
            'current_divergence_strength_config': self.cfg.divergence_strength,
            'current_max_parallel_llm_calls_config': self.cfg.max_parallel_llm_calls_per_frame,
            'pending_embedding_requests': len(self._pending_embedding_requests)
        }
        insights = []
        anomalies = []
        recommendations = []
        trends = {}
        
        if not self.last_n_frame_stats:
            insights.append('No frame data available for detailed analysis yet.')
            confidence = 0.3
        else:
            successful_variations_generated = 0
            degraded_completions = 0
            error_completions = 0
            total_llm_calls_in_recent_frames = 0
            successful_llm_calls = 0
            failed_llm_calls = 0
            for stat in self.last_n_frame_stats:
                successful_variations_generated += stat.get('variations_generated', 0)
                total_llm_calls_in_recent_frames += stat.get('llm_calls_made', 0)
                successful_llm_calls += stat.get('llm_call_successes', 0)
                failed_llm_calls += stat.get('llm_call_failures', 0)
                if stat.get('status') == 'completed_degraded':
                    degraded_completions += 1
                elif str(stat.get('status')).startswith('error_'):
                    error_completions += 1
            
            metrics['avg_variations_per_recent_frame'] = successful_variations_generated / len(self.last_n_frame_stats)
            metrics['degraded_completion_rate_recent'] = degraded_completions / len(self.last_n_frame_stats)
            metrics['error_completion_rate_recent'] = error_completions / len(self.last_n_frame_stats)
            metrics['llm_success_rate'] = successful_llm_calls / total_llm_calls_in_recent_frames if total_llm_calls_in_recent_frames > 0 else 0
            metrics['variation_generation_efficiency'] = successful_variations_generated / total_llm_calls_in_recent_frames if total_llm_calls_in_recent_frames > 0 else 0

            if metrics['avg_variations_per_recent_frame'] < self.cfg.max_variations_per_level / 2:
                insights.append(f"Average variations per frame ({metrics['avg_variations_per_recent_frame']:.2f}) is low compared to target ({self.cfg.max_variations_per_level}).")
                recommendations.append('CONSIDER_INCREASING_DIVERGENCE_OR_REVISING_PROMPTS')
                trends['variation_generation'] = 'down'
            else:
                trends['variation_generation'] = 'stable'
            
            if metrics['error_completion_rate_recent'] > 0.2:
                insights.append(f"High error rate in recent frames ({metrics['error_completion_rate_recent']:.2%}).")
                anomalies.append({'metric': 'frame_error_rate', 'value': metrics['error_completion_rate_recent'], 'expected': 0.05, 'severity': 'high'})
                recommendations.append('INVESTIGATE_FRAME_ERRORS')
                trends['error_rate'] = 'up'
            else:
                trends['error_rate'] = 'stable'
            
            confidence = min(0.9, 0.3 + len(self.last_n_frame_stats) * 0.1)

        result = AnalysisResult(
            success=True,
            component_id=self.component_id,
            metrics=metrics,
            confidence=confidence,
            message=f'Analysis of {len(self.last_n_frame_stats)} recent frames complete. Insights: {len(insights)}, Anomalies: {len(anomalies)}.',
            insights=insights,
            recommendations=recommendations,
            anomalies=anomalies,
            trends=trends
        )
        result.metadata = {'internal_recommendations_for_adapt': recommendations}
        return result
        
    async def react(self, context: NireonExecutionContext) -> List[SystemSignal]:
        context.logger.debug(f"[{self.component_id}] react() called. Current signal in context: {(type(context.signal).__name__ if context.signal else 'None')}")
        return []

    async def adapt(self, context: NireonExecutionContext) -> List[AdaptationAction]:
        context.logger.debug(f'[{self.component_id}] adapt() called.')
        proposed_actions: List[AdaptationAction] = []
        analysis_result = await self.analyze(context)
        internal_recommendations = analysis_result.metadata.get('internal_recommendations_for_adapt', [])

        if 'CONSIDER_INCREASING_DIVERGENCE_OR_REVISING_PROMPTS' in internal_recommendations:
            current_strength = self.cfg.divergence_strength
            if current_strength < 0.8:
                new_strength = min(round(current_strength * 1.2, 3), 0.8)
                action = AdaptationAction(
                    action_type=AdaptationActionType.CONFIG_UPDATE,
                    component_id=self.component_id,
                    description=f'Propose increasing default divergence_strength from {current_strength} to {new_strength} due to consistently low average idea variations per frame.',
                    parameters={'config_key': 'divergence_strength', 'new_value': new_strength, 'old_value': current_strength},
                    priority=2,
                    estimated_impact='medium',
                    requires_approval=True
                )
                proposed_actions.append(action)
                context.logger.info(f'[{self.component_id}] Proposing adaptation: Increase divergence_strength to {new_strength}.')

        if not proposed_actions:
            context.logger.info(f'[{self.component_id}] No specific adaptations proposed based on current analysis.')
        
        return proposed_actions

    async def shutdown(self, context: NireonExecutionContext) -> None:
        context.logger.info(f"ExplorerMechanism '{self.component_id}' shutting down.")
        await super().shutdown(context)
        context.logger.info(f"ExplorerMechanism '{self.component_id}' shutdown complete.")

    async def health_check(self, context: NireonExecutionContext) -> ComponentHealth:
        context.logger.debug(f"ExplorerMechanism '{self.component_id}' health_check() called.")
        status = 'HEALTHY'
        messages = []
        details = {
            'explorations_performed': self._exploration_count,
            'configured_strategy': self.cfg.exploration_strategy
        }

        if not self.is_initialized:
            status = 'UNHEALTHY'
            messages.append('Explorer not initialized.')
        else:
            messages.append('Explorer initialized.')
            if self.gateway:
                messages.append('MechanismGatewayPort dependency resolved.')
            else:
                status = 'UNHEALTHY'
                messages.append('MechanismGatewayPort dependency NOT resolved.')
                details['gateway_status'] = 'MISSING'
            if self.frame_factory:
                messages.append('FrameFactoryService dependency resolved.')
            else:
                status = 'UNHEALTHY'
                messages.append('FrameFactoryService dependency NOT resolved.')
                details['frame_factory_status'] = 'MISSING'
        
        if self.error_count > 0:
            status = 'DEGRADED' if status == 'HEALTHY' else status
            messages.append(f'Explorer has encountered {self.error_count} errors during its lifetime.')
            details['error_count'] = self.error_count

        final_message = '; '.join(messages) if messages else 'Explorer operational.'
        
        return ComponentHealth(
            component_id=self.component_id,
            status=status,
            message=final_message,
            details=details
        )