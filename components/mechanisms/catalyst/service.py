# nireon_v4/components/mechanisms/catalyst/service.py
from __future__ import annotations
import asyncio
import logging
import random
import uuid
import dataclasses
from collections import deque
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
import shortuuid

from core.base_component import NireonBaseComponent
from core.lifecycle import ComponentMetadata
from core.results import ProcessResult, AnalysisResult, SystemSignal, AdaptationAction, AdaptationActionType
from domain.cognitive_events import CognitiveEvent, LLMRequestPayload
from domain.context import NireonExecutionContext
from domain.epistemic_stage import EpistemicStage
from domain.ideas.idea import Idea
from domain.embeddings.vector import Vector as DomainVector
from domain.ports.embedding_port import EmbeddingPort
from domain.ports.mechanism_gateway_port import MechanismGatewayPort
from application.services.frame_factory_service import FrameFactoryService, FrameNotFoundError
from components.mechanisms.catalyst.vector import VectorOperations
from components.mechanisms.catalyst.config import CatalystMechanismConfig
from components.mechanisms.catalyst.metadata import CATALYST_METADATA
from components.mechanisms.catalyst.service_helpers import CatalystEventHelper
from components.mechanisms.catalyst.prompt_builder import build_prompt_for_idea
from components.mechanisms.catalyst.adaptation import handle_duplication_detected, check_blend_cooldown, check_anti_constraints_expiry

logger = logging.getLogger(__name__)

class CatalystMechanism(NireonBaseComponent):
    """
    V4 implementation of the Catalyst Mechanism.
    Operates using the Agent -> Frame -> Cognitive Event model.
    """
    METADATA_DEFINITION = CATALYST_METADATA
    ConfigModel = CatalystMechanismConfig
    
    def __init__(
        self,
        config: Dict[str, Any],
        metadata_definition: ComponentMetadata,
        gateway: Optional[MechanismGatewayPort] = None,
        frame_factory: Optional[FrameFactoryService] = None,
    ):
        super().__init__(config=config, metadata_definition=metadata_definition or self.METADATA_DEFINITION)
        self.gateway = gateway
        self.frame_factory = frame_factory
        self.embed: Optional[EmbeddingPort] = None # Will be resolved in initialize
        self.catalyst_cfg = CatalystMechanismConfig(**self.config)
        self.event_helper: Optional[CatalystEventHelper] = None
        self._rng = random.Random()

        # Internal state
        self.cross_domain_vectors: Dict[str, np.ndarray] = {}
        self.base_blend: Tuple[float, float] = (self.catalyst_cfg.blend_low, self.catalyst_cfg.blend_high)
        self.current_blend: Tuple[float, float] = self.base_blend
        self.last_duplication_step: Optional[int] = None
        self.active_anti_constraints: List[str] = []
        self.anti_constraints_expiry: Optional[int] = None
        self.recent_semantic_distances: Deque[float] = deque(maxlen=50)

        logger.info(f"[{self.component_id}] V4 instance created. Gateway/FrameFactory to be resolved.")

    async def _initialize_impl(self, context: NireonExecutionContext) -> None:
        context.logger.info(f"[{self.component_id}] Initializing Catalyst V4.")
        if not self.gateway:
            self.gateway = context.component_registry.get_service_instance(MechanismGatewayPort)
        if not self.frame_factory:
            self.frame_factory = context.component_registry.get_service_instance(FrameFactoryService)
        self.embed = context.component_registry.get_service_instance(EmbeddingPort)
        self.event_helper = CatalystEventHelper(self.gateway, self.component_id, self.metadata.version)
        
        if not self.cross_domain_vectors:
             context.logger.warning(f"[{self.component_id}] No cross_domain_vectors loaded. Catalysis will not be effective.")
        
        if context.replay_mode and context.replay_seed is not None:
            self._rng.seed(context.replay_seed)
            
        context.logger.info(f"[{self.component_id}] V4 Initialization complete.")

    async def _process_impl(self, data: Any, context: NireonExecutionContext) -> ProcessResult:
        """
        Processes a catalysis request, which contains a list of ideas and an objective.
        """
        if not isinstance(data, dict) or 'ideas' not in data or 'objective' not in data:
            return ProcessResult(success=False, component_id=self.component_id, message="Input must be a dict with 'ideas' and 'objective'.", error_code='INVALID_INPUT')

        ideas_to_process: List[Idea] = data['ideas']
        objective: str = data['objective']
        self.cross_domain_vectors = data.get('cross_domain_vectors', self.cross_domain_vectors)

        if not self.cross_domain_vectors:
             return ProcessResult(success=False, component_id=self.component_id, message="No cross-domain vectors available for catalysis.", error_code='MISSING_VECTORS')

        task_id = shortuuid.uuid()[:8]
        frame_name = f"catalyst_task_{task_id}_on_{len(ideas_to_process)}_ideas"
        frame_description = f"Catalysis task for objective '{objective[:50]}...'"

        current_frame = await self.frame_factory.create_frame(
            context=context,
            name=frame_name,
            description=frame_description,
            owner_agent_id=self.component_id,
            epistemic_goals=['CROSS_POLLINATION', 'NOVELTY_SYNTHESIS'],
            llm_policy=self.catalyst_cfg.default_llm_policy_for_catalysis,
            resource_budget=self.catalyst_cfg.default_resource_budget_for_catalysis,
            context_tags={'catalyst_version': self.metadata.version, 'objective': objective}
        )

        try:
            cognitive_events: List[CognitiveEvent] = []
            ideas_for_catalysis = []
            for idea in ideas_to_process:
                if self._rng.random() < self.catalyst_cfg.application_rate:
                    ideas_for_catalysis.append(idea)

            context.logger.info(f"[{self.component_id}] Applying catalysis to {len(ideas_for_catalysis)}/{len(ideas_to_process)} ideas in Frame {current_frame.id}.")

            for idea in ideas_for_catalysis:
                if not self._ensure_idea_has_vector(idea, context):
                    context.logger.warning(f"Skipping idea {idea.idea_id}, could not ensure vector.")
                    continue
                
                domain, blend = self._choose_domain_and_blend()
                if not domain:
                    context.logger.warning(f"Skipping idea {idea.idea_id}, no valid domain found.")
                    continue
                
                prompt = build_prompt_for_idea(
                    idea=idea, domain=domain, blend=blend, objective=objective,
                    prompt_template=self.catalyst_cfg.prompt_template,
                    active_anti_constraints=self.active_anti_constraints,
                    anti_constraints_threshold=self.catalyst_cfg.anti_constraints_diversity_threshold,
                    recent_semantic_distances=self.recent_semantic_distances
                )
                
                llm_payload = LLMRequestPayload(prompt=prompt, stage=EpistemicStage.EXPLORATION, role="idea_synthesizer")
                ce = CognitiveEvent(
                    frame_id=current_frame.id,
                    owning_agent_id=self.component_id,
                    service_call_type='LLM_ASK',
                    payload=llm_payload,
                    epistemic_intent='CATALYZE_IDEA',
                    custom_metadata={'original_idea_id': idea.idea_id, 'domain': domain, 'blend_strength': blend}
                )
                cognitive_events.append(ce)

            llm_tasks = [self.gateway.process_cognitive_event(ce, context) for ce in cognitive_events]
            llm_responses = await asyncio.gather(*llm_tasks, return_exceptions=True)

            catalyzed_ideas_count = 0
            for i, resp_or_exc in enumerate(llm_responses):
                metadata_from_request = cognitive_events[i].custom_metadata
                original_idea_id = metadata_from_request['original_idea_id']
                original_idea = next((idea for idea in ideas_for_catalysis if idea.idea_id == original_idea_id), None)
                
                if isinstance(resp_or_exc, Exception) or (resp_or_exc and resp_or_exc.get('error')):
                    context.logger.error(f"Catalysis LLM call failed for idea {original_idea_id}: {resp_or_exc}")
                    continue
                
                if not original_idea:
                    context.logger.error(f"Could not find original idea {original_idea_id} for response. Skipping.")
                    continue

                new_idea_text = resp_or_exc.text.strip()
                if len(new_idea_text) < 10:
                    context.logger.warning(f"LLM output for idea {original_idea_id} is too short. Skipping.")
                    continue

                domain = metadata_from_request['domain']
                blend = metadata_from_request['blend_strength']

                new_idea_obj, sem_dist, vec_dist = await self._create_new_idea(original_idea, new_idea_text, domain, blend, context)
                self.recent_semantic_distances.append(sem_dist)
                
                await self.event_helper.publish_signal(
                    frame_id=current_frame.id,
                    signal_type_name='IdeaCatalyzedSignal',
                    signal_payload={
                        'idea': dataclasses.asdict(new_idea_obj),
                        'original_idea_id': metadata_from_request['original_idea_id'],
                        'domain_blended': domain,
                        'blend_strength': blend,
                        'semantic_distance': sem_dist,
                        'vector_distance': vec_dist
                    },
                    context=context,
                    epistemic_intent='OUTPUT_CATALYZED_IDEA'
                )
                catalyzed_ideas_count += 1
            
            final_status = 'completed_ok' if catalyzed_ideas_count > 0 else 'completed_degraded'
            message = f"Catalysis in frame {current_frame.id} complete. Generated {catalyzed_ideas_count} ideas."
            await self.frame_factory.update_frame_status(context, current_frame.id, final_status)

            return ProcessResult(success=True, component_id=self.component_id, output_data={'frame_id': current_frame.id, 'ideas_catalyzed': catalyzed_ideas_count}, message=message)

        except Exception as e:
            context.logger.error(f"Critical error in Catalyst frame {current_frame.id if current_frame else 'unknown'}: {e}", exc_info=True)
            if current_frame:
                await self.frame_factory.update_frame_status(context, current_frame.id, 'error_internal')
            return ProcessResult(success=False, component_id=self.component_id, message=f"Catalyst critical error: {e}", error_code='CATALYSIS_FRAME_ERROR')

    async def _create_new_idea(self, original_idea: Idea, new_text: str, domain: str, blend: float, ctx: NireonExecutionContext) -> Tuple[Idea, float, float]:
        """Creates a new Idea object and its associated vector."""
        if not hasattr(original_idea, 'theta') or original_idea.theta is None:
            raise ValueError(f"Original idea {original_idea.idea_id} is missing its vector (theta).")
            
        domain_vec_data = self.cross_domain_vectors[domain]
        new_vector_obj, vec_dist, sem_dist = VectorOperations.blend_vectors(original_idea.theta, domain_vec_data, blend)
        
        metadata = original_idea.metadata.copy()
        metadata.update({
            'stage': original_idea.metadata.get('stage', 'UNKNOWN'),
            'domain_blended': domain,
            'blend_strength': blend,
            'vector_distance_from_original': vec_dist,
            'semantic_distance_from_original': sem_dist,
            'mechanism': self.component_id,
            'mechanism_version': self.metadata.version,
            'anti_constraints_active': bool(self.active_anti_constraints)
        })
        
        new_idea = Idea.create(
            text=new_text,
            parent_id=original_idea.idea_id,
            step=ctx.step,
            method=self.component_id,
            metadata=metadata
        )
        new_idea.theta = new_vector_obj
        return new_idea, sem_dist, vec_dist

    def _ensure_idea_has_vector(self, idea: Idea, context: NireonExecutionContext) -> bool:
        """Ensures an idea has an embedding vector."""
        if isinstance(getattr(idea, 'theta', None), DomainVector) and idea.theta.data is not None:
            return True
        if not getattr(idea, 'text', None): return False
        try:
            idea.theta = self.embed.encode(idea.text)
            return True
        except Exception as e:
            context.logger.error(f"[{self.component_id}] Embedding failed for idea {idea.idea_id}: {e}", exc_info=True)
            return False

    def _choose_domain_and_blend(self) -> Tuple[Optional[str], float]:
        """Choose a domain and blend strength."""
        if not self.cross_domain_vectors:
            return (None, 0.0)
        valid_domains = [k for k, v in self.cross_domain_vectors.items() if v is not None]
        if not valid_domains:
            return (None, 0.0)
        domain = self._rng.choice(valid_domains)
        blend = self._rng.uniform(self.current_blend[0], self.current_blend[1])
        return (domain, blend)

    async def analyze(self, context: NireonExecutionContext) -> AnalysisResult:
        avg_sem_dist = sum(self.recent_semantic_distances) / len(self.recent_semantic_distances) if self.recent_semantic_distances else None
        metrics = {
            'current_blend': list(self.current_blend),
            'base_blend': list(self.base_blend),
            'application_rate': self.catalyst_cfg.application_rate,
            'duplication_check_enabled': self.catalyst_cfg.duplication_check_enabled,
            'active_anti_constraints_count': len(self.active_anti_constraints),
            'average_semantic_distance': avg_sem_dist,
        }
        return AnalysisResult(success=True, component_id=self.component_id, metrics=metrics, message="Catalyst state analysis.")

    async def react(self, context: NireonExecutionContext) -> List[SystemSignal]:
        """React to system state changes and perform adaptations."""
        self.current_blend, self.last_duplication_step = check_blend_cooldown(
            self.current_blend, self.base_blend, self.last_duplication_step,
            context.step, self.catalyst_cfg.duplication_cooldown_steps
        )
        self.active_anti_constraints, self.anti_constraints_expiry = check_anti_constraints_expiry(
            self.active_anti_constraints, self.anti_constraints_expiry, context.step
        )
        return []

    async def adapt(self, context: NireonExecutionContext) -> List[AdaptationAction]:
        actions = []
        analysis = await self.analyze(context)
        avg_sem_dist = analysis.metrics.get('average_semantic_distance')
        
        if avg_sem_dist is not None and avg_sem_dist < self.catalyst_cfg.anti_constraints_diversity_threshold:
             actions.append(AdaptationAction(
                action_type=AdaptationActionType.CONFIG_UPDATE,
                component_id=self.component_id,
                description="Average semantic distance is low, proposing to increase blend range to foster more novelty.",
                parameters={
                    'config_key': 'blend_low', 
                    'new_value': min(self.catalyst_cfg.blend_low + 0.05, self.catalyst_cfg.max_blend_low),
                    'old_value': self.catalyst_cfg.blend_low
                },
                requires_approval=True
            ))
        return actions

    def set_anti_constraints(self, constraints: List[str], expiry_step: Optional[int] = None, context: Optional[NireonExecutionContext] = None) -> None:
        if not self.catalyst_cfg.anti_constraints_enabled:
            logger.warning(f'[{self.component_id}] Cannot set anti-constraints: feature is disabled')
            return
        if not isinstance(constraints, list) or not all(isinstance(c, str) for c in constraints):
            logger.error(f'[{self.component_id}] Invalid anti-constraints: expected List[str]')
            return

        max_constraints = self.catalyst_cfg.anti_constraints_count
        if len(constraints) > max_constraints:
            logger.warning(f'[{self.component_id}] Limiting to {max_constraints} anti-constraints')
            constraints = constraints[:max_constraints]
        
        self.active_anti_constraints = constraints
        self.anti_constraints_expiry = expiry_step
        logger.info(f'[{self.component_id}] Set {len(constraints)} anti-constraints, expiring at step {expiry_step}')
        
        if self.event_helper and context:
            asyncio.create_task(self.event_helper.publish_signal(
                frame_id=context.metadata.get('current_frame_id', 'system_level'),
                signal_type_name='CatalystAdaptationSignal',
                signal_payload={'type': 'anti_constraints_set', 'constraints': constraints, 'expiry': expiry_step},
                context=context,
                epistemic_intent='UPDATE_ADAPTIVE_STATE'
            ))

    def clear_anti_constraints(self, context: Optional[NireonExecutionContext] = None) -> None:
        if self.active_anti_constraints:
            logger.info(f'[{self.component_id}] Cleared {len(self.active_anti_constraints)} anti-constraints')
            self.active_anti_constraints = []
            self.anti_constraints_expiry = None
            if self.event_helper and context:
                asyncio.create_task(self.event_helper.publish_signal(
                    frame_id=context.metadata.get('current_frame_id', 'system_level'),
                    signal_type_name='CatalystAdaptationSignal',
                    signal_payload={'type': 'anti_constraints_cleared'},
                    context=context,
                    epistemic_intent='UPDATE_ADAPTIVE_STATE'
                ))

    def _handle_duplication_detected(self, step: int) -> None:
        self.current_blend, self.last_duplication_step = handle_duplication_detected(
            current_blend=self.current_blend,
            max_blend_low=self.catalyst_cfg.max_blend_low,
            max_blend_high=self.catalyst_cfg.max_blend_high,
            duplication_aggressiveness=self.catalyst_cfg.duplication_aggressiveness,
            current_step=step
        )