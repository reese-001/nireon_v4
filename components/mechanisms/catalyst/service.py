# nireon_v4\components\mechanisms\catalyst\service.py
from __future__ import annotations
import asyncio
import logging
import random
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
from domain.ports.idea_service_port import IdeaServicePort
from domain.ports.mechanism_gateway_port import MechanismGatewayPort
from application.services.frame_factory_service import FrameFactoryService
from components.mechanisms.base import ProducerMechanism

from components.service_resolution_mixin import ServiceResolutionMixin
from .vector import VectorOperations
from .config import CatalystMechanismConfig
from .metadata import CATALYST_METADATA
from .service_helpers import CatalystEventHelper
from .prompt_builder import build_prompt_for_idea
from .adaptation import handle_duplication_detected, check_blend_cooldown, check_anti_constraints_expiry
from .processing import validate_cross_domain_vectors, select_ideas_for_catalysis, compute_blend_metrics, create_catalysis_summary

logger = logging.getLogger(__name__)

class CatalystMechanism(ProducerMechanism, ServiceResolutionMixin):
    """
    Catalyst Mechanism for cross-domain concept injection and creative synthesis.
    
    Required Services:
        - gateway (MechanismGatewayPort): For LLM communication
        - frame_factory (FrameFactoryService): For frame management
        - idea_service (IdeaServicePort): For idea management
        - embed (EmbeddingPort): For vector operations
    """
    
    METADATA_DEFINITION = CATALYST_METADATA
    ConfigModel = CatalystMechanismConfig

    def __init__(self, config: Dict[str, Any], metadata_definition: ComponentMetadata, 
                 gateway: Optional[MechanismGatewayPort] = None, 
                 frame_factory: Optional[FrameFactoryService] = None, 
                 idea_service: Optional[IdeaServicePort] = None):
        super().__init__(config=config, metadata_definition=metadata_definition or self.METADATA_DEFINITION)
        
        # Allow dependency injection through constructor (useful for testing)
        self.gateway = gateway
        self.frame_factory = frame_factory
        self.idea_service = idea_service
        self.embed: Optional[EmbeddingPort] = None
        
        self.catalyst_cfg = CatalystMechanismConfig(**self.config)
        self.event_helper: Optional[CatalystEventHelper] = None
        self._rng = random.Random()

        self.cross_domain_vectors: Dict[str, np.ndarray] = {}
        self.base_blend: Tuple[float, float] = (self.catalyst_cfg.blend_low, self.catalyst_cfg.blend_high)
        self.current_blend: Tuple[float, float] = self.base_blend
        self.last_duplication_step: Optional[int] = None
        self.active_anti_constraints: List[str] = []
        self.anti_constraints_expiry: Optional[int] = None

        self.recent_semantic_distances: Deque[float] = deque(maxlen=50)
        self._catalysis_stats = {
            'total_ideas_processed': 0,
            'total_ideas_catalyzed': 0,
            'domains_used': {},
            'avg_blend_strength': 0.0,
            'avg_semantic_distance': 0.0,
        }
        logger.info(f'[{self.component_id}] Catalyst V4 instance created. Application rate: {self.catalyst_cfg.application_rate:.0%}')

    async def _initialize_impl(self, context: NireonExecutionContext) -> None:
        """Initialize dependencies and validate configuration."""
        context.logger.info(f'[{self.component_id}] Initializing Catalyst V4.')
        
        # Resolve all required services using the mixin
        await self._resolve_all_dependencies(context)
        
        # Validate services are available
        self._validate_dependencies(context)
        
        # Initialize event helper with resolved services
        self.event_helper = CatalystEventHelper(
            self.gateway, 
            self.component_id, 
            self.metadata.version, 
            idea_service=self.idea_service
        )

        # Handle cross-domain vectors validation
        if self.cross_domain_vectors:
            is_valid, error_msg, normalized = validate_cross_domain_vectors(self.cross_domain_vectors)
            if not is_valid:
                context.logger.error(f'[{self.component_id}] Invalid cross_domain_vectors: {error_msg}')
                self.cross_domain_vectors = {}
            else:
                self.cross_domain_vectors = normalized
                context.logger.info(f'[{self.component_id}] Loaded {len(normalized)} normalized domain vectors')
        else:
            context.logger.warning(f'[{self.component_id}] No cross_domain_vectors loaded. Catalysis will not be effective.')

        # Handle replay mode
        if context.replay_mode and context.replay_seed is not None:
            self._rng.seed(context.replay_seed)
            
        context.logger.info(f'[{self.component_id}] V4 Initialization complete.')

    async def _resolve_all_dependencies(self, context: NireonExecutionContext) -> None:
        """Resolve all required dependencies using the ServiceResolutionMixin."""
        
        # Define service mappings - only include services not already injected
        service_map = {}
        
        # Only add to service_map if not already provided via constructor
        if not self.gateway:
            service_map['gateway'] = MechanismGatewayPort
        if not self.frame_factory:
            service_map['frame_factory'] = FrameFactoryService
        if not self.idea_service:
            service_map['idea_service'] = IdeaServicePort
        # embed is never injected via constructor, always resolve it
        service_map['embed'] = EmbeddingPort
        
        if service_map:
            try:
                # Resolve services using the mixin
                resolved_services = self.resolve_services(
                    context=context,
                    service_map=service_map,
                    raise_on_missing=True,  # Critical services must be available
                    log_resolution=True
                )
                
                context.logger.debug(
                    f"[{self.component_id}] Resolved {len(resolved_services)} services via mixin"
                )
                
            except RuntimeError as e:
                context.logger.error(f"[{self.component_id}] Failed to resolve dependencies: {e}")
                raise
        else:
            context.logger.debug(
                f"[{self.component_id}] All services were pre-injected, no resolution needed"
            )

    def _validate_dependencies(self, context: NireonExecutionContext) -> None:
        """Validate that all required dependencies are available."""
        
        required_services = ['gateway', 'frame_factory', 'idea_service', 'embed']
        
        # Use the mixin's validation method
        if not self.validate_required_services(required_services, context):
            # The mixin will have logged which services are missing
            missing = [s for s in required_services if not getattr(self, s, None)]
            raise RuntimeError(
                f"CatalystMechanism '{self.component_id}' missing critical dependencies: {', '.join(missing)}"
            )

    async def _process_impl(self, data: Any, context: NireonExecutionContext, **kwargs) -> ProcessResult:
        """Process incoming data for catalysis."""
        
        # Ensure services are still available (defensive check)
        if not self._ensure_services_available(context):
            return ProcessResult(
                success=False, 
                component_id=self.component_id, 
                message="Required services not available",
                error_code='MISSING_DEPENDENCIES'
            )
        
        if isinstance(data, dict) and 'target_idea_id' in data:
            # Handle single-idea trigger from Reactor
            return await self._process_single_idea_trigger(data, context, **kwargs)
        
        if not isinstance(data, dict) or 'ideas' not in data or 'objective' not in data:
            # Handle batch processing
            return ProcessResult(
                success=False, 
                component_id=self.component_id, 
                message="Input must be a dict with 'ideas' and 'objective' for batch mode.", 
                error_code='INVALID_BATCH_INPUT'
            )

        # Fallthrough to batch processing logic if structured correctly
        return await self._process_batch(data, context, **kwargs)

    def _ensure_services_available(self, context: NireonExecutionContext) -> bool:
        """
        Ensure all required services are available.
        This provides a runtime check and potential re-resolution attempt.
        """
        required_services = ['gateway', 'frame_factory', 'idea_service', 'embed']
        
        # Quick check if all services are already available
        if self.validate_required_services(required_services):
            return True
        
        # Attempt to re-resolve missing services
        context.logger.warning(
            f"[{self.component_id}] Some services missing at runtime, attempting re-resolution"
        )
        
        try:
            # Build service map only for missing services
            service_map = {}
            if not self.gateway:
                service_map['gateway'] = MechanismGatewayPort
            if not self.frame_factory:
                service_map['frame_factory'] = FrameFactoryService
            if not self.idea_service:
                service_map['idea_service'] = IdeaServicePort
            if not self.embed:
                service_map['embed'] = EmbeddingPort
            
            if service_map:
                self.resolve_services(
                    context=context,
                    service_map=service_map,
                    raise_on_missing=False,  # Don't raise, we'll check below
                    log_resolution=True
                )
            
            # Check again after resolution attempt
            return self.validate_required_services(required_services)
            
        except Exception as e:
            context.logger.error(f"[{self.component_id}] Failed to ensure services: {e}")
            return False

    async def _process_single_idea_trigger(self, data: Dict[str, Any], context: NireonExecutionContext, **kwargs) -> ProcessResult:
        """Process a single idea trigger."""
        target_idea_id = data.get('target_idea_id')
        if not self.idea_service:
            return ProcessResult(
                success=False, 
                component_id=self.component_id, 
                message="IdeaService not available to fetch idea.", 
                error_code='DEPENDENCY_MISSING'
            )

        target_idea = self.idea_service.get_by_id(target_idea_id)
        if not target_idea:
            return ProcessResult(
                success=False, 
                component_id=self.component_id, 
                message=f"Idea with ID '{target_idea_id}' not found.", 
                error_code='IDEA_NOT_FOUND'
            )

        context.logger.info(f"[{self.component_id}] Received single-idea trigger for '{target_idea_id}'.")
        # Wrap the single idea in a list to use the common batch processing logic
        batch_data = {
            'ideas': [target_idea],
            'objective': data.get('objective', 'Amplify and explore hybrid concepts from a high-trust idea.'),
            'cross_domain_vectors': data.get('cross_domain_vectors')
        }
        # Force application rate to 100% for this single triggered idea
        original_rate = self.catalyst_cfg.application_rate
        self.catalyst_cfg.application_rate = 1.0
        result = await self._process_batch(batch_data, context, **kwargs)
        self.catalyst_cfg.application_rate = original_rate # Restore original rate
        return result
    
    async def _process_batch(self, data: Dict[str, Any], context: NireonExecutionContext, **kwargs) -> ProcessResult:
        template_id = kwargs.get('template_id')
        if template_id:
            context.logger.info(f'[{self.component_id}] Batch process called with template_id: {template_id}')
        
        ideas_to_process: List[Idea] = data['ideas']
        objective: str = data['objective']
        new_vectors = data.get('cross_domain_vectors')
        
        if new_vectors:
            is_valid, error_msg, normalized = validate_cross_domain_vectors(new_vectors)
            if is_valid:
                self.cross_domain_vectors = normalized
                context.logger.info(f'[{self.component_id}] Updated cross-domain vectors')
        
        if not self.cross_domain_vectors:
            return ProcessResult(
                success=False,
                component_id=self.component_id,
                message='No cross-domain vectors available for catalysis.',
                error_code='MISSING_VECTORS'
            )
        
        task_id = shortuuid.uuid()[:8]
        frame_name = f'catalyst_task_{task_id}_on_{len(ideas_to_process)}_ideas'
        frame_description = f"Cross-domain catalysis for objective '{objective[:50]}...'"
        
        current_frame = await self.frame_factory.create_frame(
            context=context,
            name=frame_name,
            description=frame_description,
            owner_agent_id=self.component_id,
            epistemic_goals=['CROSS_POLLINATION', 'NOVELTY_SYNTHESIS', 'HYBRID_THINKING'],
            llm_policy=self.catalyst_cfg.default_llm_policy_for_catalysis,
            resource_budget=self.catalyst_cfg.default_resource_budget_for_catalysis,
            context_tags={
                'catalyst_version': self.metadata.version,
                'objective': objective,
                'blend_range': list(self.current_blend)
            }
        )
        
        try:
            ideas_for_catalysis = select_ideas_for_catalysis(
                ideas_to_process,
                self.catalyst_cfg.application_rate,
                self._rng,
                max_ideas=50
            )
            
            context.logger.info(
                f'[{self.component_id}] Applying catalysis to {len(ideas_for_catalysis)}/{len(ideas_to_process)} ideas in Frame {current_frame.id}.'
            )
            
            cognitive_events: List[CognitiveEvent] = []
            
            for idea in ideas_for_catalysis:
                if not self._ensure_idea_has_vector(idea, context):
                    context.logger.warning(f'Skipping idea {idea.idea_id}, could not ensure vector.')
                    continue
                
                domain, blend = self._choose_domain_and_blend()
                if not domain:
                    context.logger.warning(f'Skipping idea {idea.idea_id}, no valid domain found.')
                    continue
                
                prompt = build_prompt_for_idea(
                    idea=idea,
                    domain=domain,
                    blend=blend,
                    objective=objective,
                    prompt_template=self.catalyst_cfg.prompt_template,
                    active_anti_constraints=self.active_anti_constraints,
                    anti_constraints_threshold=self.catalyst_cfg.anti_constraints_diversity_threshold,
                    recent_semantic_distances=self.recent_semantic_distances
                )
                
                llm_payload = LLMRequestPayload(
                    prompt=prompt,
                    stage=EpistemicStage.EXPLORATION,
                    role='idea_synthesizer'
                )
                
                ce = CognitiveEvent(
                    frame_id=current_frame.id,
                    owning_agent_id=self.component_id,
                    service_call_type='LLM_ASK',
                    payload=llm_payload,
                    epistemic_intent='CATALYZE_IDEA',
                    custom_metadata={
                        'original_idea_id': idea.idea_id,
                        'domain': domain,
                        'blend_strength': blend
                    }
                )
                cognitive_events.append(ce)
            
            # Add concurrency control with semaphore
            MAX_CONCURRENT_LLM_CALLS = getattr(self.catalyst_cfg, 'max_concurrent_llm_calls', 5)
            semaphore = asyncio.Semaphore(MAX_CONCURRENT_LLM_CALLS)
            
            async def process_with_semaphore(ce: CognitiveEvent) -> Any:
                async with semaphore:
                    return await self.gateway.process_cognitive_event(ce, context)
            
            # Process LLM requests with controlled concurrency
            llm_tasks = [process_with_semaphore(ce) for ce in cognitive_events]
            llm_responses = await asyncio.gather(*llm_tasks, return_exceptions=True)
            
            # Process responses
            catalyzed_ideas_count = 0
            domain_usage = {}
            total_blend_strength = 0.0
            total_semantic_distance = 0.0
            
            for i, resp_or_exc in enumerate(llm_responses):
                metadata_from_request = cognitive_events[i].custom_metadata
                original_idea_id = metadata_from_request['original_idea_id']
                original_idea = next((idea for idea in ideas_for_catalysis if idea.idea_id == original_idea_id), None)
                
                if isinstance(resp_or_exc, Exception) or (resp_or_exc and resp_or_exc.get('error')):
                    context.logger.error(f'Catalysis LLM call failed for idea {original_idea_id}: {resp_or_exc}')
                    continue
                
                if not original_idea:
                    context.logger.error(f'Could not find original idea {original_idea_id} for response. Skipping.')
                    continue
                
                new_idea_text = resp_or_exc.text.strip()
                if len(new_idea_text) < 10:
                    context.logger.warning(f'LLM output for idea {original_idea_id} is too short. Skipping.')
                    continue
                
                domain = metadata_from_request['domain']
                blend = metadata_from_request['blend_strength']
                
                new_idea_obj, sem_dist, vec_dist, metrics = await self._create_new_idea(
                    original_idea, new_idea_text, domain, blend, context
                )
                
                self.recent_semantic_distances.append(sem_dist)
                domain_usage[domain] = domain_usage.get(domain, 0) + 1
                total_blend_strength += blend
                total_semantic_distance += sem_dist
                catalyzed_ideas_count += 1
                
                await self.event_helper.publish_signal(
                    frame_id=current_frame.id,
                    signal_type_name='IdeaCatalyzedSignal',
                    signal_payload={
                        'idea': dataclasses.asdict(new_idea_obj),
                        'original_idea_id': metadata_from_request['original_idea_id'],
                        'domain_blended': domain,
                        'blend_strength': blend,
                        'semantic_distance': sem_dist,
                        'vector_distance': vec_dist,
                        'metrics': metrics
                    },
                    context=context,
                    epistemic_intent='OUTPUT_CATALYZED_IDEA'
                )
            
            # Update statistics
            self._update_statistics(
                len(ideas_for_catalysis),
                catalyzed_ideas_count,
                domain_usage,
                total_blend_strength / catalyzed_ideas_count if catalyzed_ideas_count > 0 else 0,
                total_semantic_distance / catalyzed_ideas_count if catalyzed_ideas_count > 0 else 0
            )
            
            # Create summary
            summary = create_catalysis_summary(
                len(ideas_for_catalysis),
                catalyzed_ideas_count,
                domain_usage,
                self._catalysis_stats['avg_blend_strength'],
                self._catalysis_stats['avg_semantic_distance']
            )
            
            final_status = 'completed_ok' if catalyzed_ideas_count > 0 else 'completed_degraded'
            message = f'Catalysis in frame {current_frame.id} complete. Generated {catalyzed_ideas_count} hybrid ideas.'
            
            await self.frame_factory.update_frame_status(context, current_frame.id, final_status)
            
            return ProcessResult(
                success=True,
                component_id=self.component_id,
                output_data={
                    'frame_id': current_frame.id,
                    'ideas_catalyzed': catalyzed_ideas_count,
                    'summary': summary
                },
                message=message
            )
            
        except Exception as e:
            context.logger.error(
                f"Critical error in Catalyst frame {(current_frame.id if current_frame else 'unknown')}: {e}",
                exc_info=True
            )
            if current_frame:
                await self.frame_factory.update_frame_status(context, current_frame.id, 'error_internal')
            
            return ProcessResult(
                success=False,
                component_id=self.component_id,
                message=f'Catalyst critical error: {e}',
                error_code='CATALYSIS_FRAME_ERROR'
            )
    
    async def _create_new_idea(self, original_idea: Idea, new_text: str, domain: str, 
                               blend: float, ctx: NireonExecutionContext) -> Tuple[Idea, float, float, Dict[str, float]]:
        """Create a new catalyzed idea with blended vectors."""
        if not hasattr(original_idea, 'theta') or original_idea.theta is None:
            raise ValueError(f'Original idea {original_idea.idea_id} is missing its vector (theta).')

        domain_vec_data = self.cross_domain_vectors[domain]
        new_vector_obj, vec_dist, sem_dist = VectorOperations.blend_vectors(
            original_idea.theta, domain_vec_data, blend
        )
        metrics = compute_blend_metrics(original_idea.theta, new_vector_obj, domain_vec_data, blend)
        
        metadata = original_idea.metadata.copy()
        metadata.update({
            'stage': original_idea.metadata.get('stage', 'UNKNOWN'),
            'domain_blended': domain,
            'blend_strength': blend,
            'vector_distance_from_original': vec_dist,
            'semantic_distance_from_original': sem_dist,
            'mechanism': self.component_id,
            'mechanism_version': self.metadata.version,
            'anti_constraints_active': bool(self.active_anti_constraints),
            'catalyst_type': 'cross_domain_injection',
            'hybrid_metrics': metrics,
        })
        
        new_idea = self.event_helper.create_and_persist_idea(
            text=new_text,
            parent_id=original_idea.idea_id,
            context=ctx,
            metadata=metadata
        )
        new_idea.theta = new_vector_obj
        return new_idea, sem_dist, vec_dist, metrics

    def _ensure_idea_has_vector(self, idea: Idea, context: NireonExecutionContext) -> bool:
        """Ensure an idea has an embedding vector."""
        if isinstance(getattr(idea, 'theta', None), DomainVector) and idea.theta.data is not None:
            return True
        if not getattr(idea, 'text', None):
            return False
        
        try:
            idea.theta = self.embed.encode(idea.text)
            return True
        except Exception as e:
            context.logger.error(
                f'[{self.component_id}] Embedding failed for idea {idea.idea_id}: {e}', 
                exc_info=True
            )
            return False

    def _choose_domain_and_blend(self) -> Tuple[Optional[str], float]:
        """Choose a random domain and blend strength."""
        if not self.cross_domain_vectors:
            return None, 0.0

        valid_domains = [k for k, v in self.cross_domain_vectors.items() if v is not None]
        if not valid_domains:
            return None, 0.0

        domain = self._rng.choice(valid_domains)
        blend = self._rng.uniform(self.current_blend[0], self.current_blend[1])
        return domain, blend

    def _update_statistics(self, ideas_processed: int, ideas_catalyzed: int, 
                          domain_usage: Dict[str, int], avg_blend: float, avg_distance: float) -> None:
        """Update internal statistics."""
        self._catalysis_stats['total_ideas_processed'] += ideas_processed
        self._catalysis_stats['total_ideas_catalyzed'] += ideas_catalyzed
        for domain, count in domain_usage.items():
            current = self._catalysis_stats['domains_used'].get(domain, 0)
            self._catalysis_stats['domains_used'][domain] = current + count
        
        total_catalyzed = self._catalysis_stats['total_ideas_catalyzed']
        if total_catalyzed > 0:
            weight_new = ideas_catalyzed / total_catalyzed
            weight_old = 1.0 - weight_new
            self._catalysis_stats['avg_blend_strength'] = (
                weight_old * self._catalysis_stats['avg_blend_strength']
            ) + (weight_new * avg_blend)
            self._catalysis_stats['avg_semantic_distance'] = (
                weight_old * self._catalysis_stats['avg_semantic_distance']
            ) + (weight_new * avg_distance)

    async def analyze(self, context: NireonExecutionContext) -> AnalysisResult:
        """Analyze current state and performance."""
        avg_sem_dist = sum(self.recent_semantic_distances) / len(self.recent_semantic_distances) if self.recent_semantic_distances else None
        metrics = {
            'current_blend': list(self.current_blend),
            'base_blend': list(self.base_blend),
            'application_rate': self.catalyst_cfg.application_rate,
            'duplication_check_enabled': self.catalyst_cfg.duplication_check_enabled,
            'active_anti_constraints_count': len(self.active_anti_constraints),
            'average_semantic_distance': avg_sem_dist,
            'total_ideas_processed': self._catalysis_stats['total_ideas_processed'],
            'total_ideas_catalyzed': self._catalysis_stats['total_ideas_catalyzed'],
            'catalysis_success_rate': (
                self._catalysis_stats['total_ideas_catalyzed'] / self._catalysis_stats['total_ideas_processed'] 
                if self._catalysis_stats['total_ideas_processed'] > 0 else 0
            ),
            'most_used_domains': sorted(
                self._catalysis_stats['domains_used'].items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:5],
            'cross_domain_vectors_count': len(self.cross_domain_vectors),
            'services_available': self.validate_required_services(
                ['gateway', 'frame_factory', 'idea_service', 'embed']
            )
        }
        return AnalysisResult(
            success=True, 
            component_id=self.component_id, 
            metrics=metrics, 
            message='Catalyst state analysis complete.'
        )
        
    async def react(self, context: NireonExecutionContext) -> List[SystemSignal]:
        """React to system state and generate signals."""
        self.current_blend, self.last_duplication_step = check_blend_cooldown(
            self.current_blend, self.base_blend, self.last_duplication_step, 
            context.step, self.catalyst_cfg.duplication_cooldown_steps, self.event_helper
        )
        self.active_anti_constraints, self.anti_constraints_expiry = check_anti_constraints_expiry(
            self.active_anti_constraints, self.anti_constraints_expiry, 
            context.step, self.event_helper
        )
        
        signals = []
        if len(self.recent_semantic_distances) >= 20:
            avg_recent = sum(list(self.recent_semantic_distances)[-20:]) / 20
            if avg_recent < 0.1:
                signals.append(SystemSignal(
                    signal_type='LOW_CATALYST_DIVERSITY', 
                    source_component_id=self.component_id, 
                    payload={
                        'avg_semantic_distance': avg_recent, 
                        'measurement_window': 20
                    }
                ))
        return signals

    async def adapt(self, context: NireonExecutionContext) -> List[AdaptationAction]:
        """Propose adaptations based on performance."""
        actions = []
        analysis = await self.analyze(context)
        avg_sem_dist = analysis.metrics.get('average_semantic_distance')
        
        if avg_sem_dist is not None and avg_sem_dist < self.catalyst_cfg.anti_constraints_diversity_threshold:
            new_blend_low = min(self.catalyst_cfg.blend_low + 0.05, self.catalyst_cfg.max_blend_low)
            actions.append(AdaptationAction(
                action_type=AdaptationActionType.CONFIG_UPDATE,
                component_id=self.component_id,
                description='Average semantic distance is low, proposing to increase blend range to foster more novelty and interdisciplinary thinking.',
                parameters={
                    'config_key': 'blend_low', 
                    'new_value': new_blend_low, 
                    'old_value': self.catalyst_cfg.blend_low, 
                    'reason': 'low_diversity'
                },
                requires_approval=True
            ))

        if not self.catalyst_cfg.anti_constraints_enabled and avg_sem_dist is not None and avg_sem_dist < 0.1:
             actions.append(AdaptationAction(
                action_type=AdaptationActionType.CONFIG_UPDATE,
                component_id=self.component_id,
                description='Very low diversity detected. Proposing to enable anti-constraints to force more creative divergence.',
                parameters={
                    'config_key': 'anti_constraints_enabled', 
                    'new_value': True, 
                    'old_value': False, 
                    'reason': 'critical_low_diversity'
                },
                requires_approval=True
            ))
        return actions

    def set_anti_constraints(self, constraints: List[str], expiry_step: Optional[int] = None, 
                           context: Optional[NireonExecutionContext] = None) -> None:
        """Set anti-constraints to avoid certain themes."""
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
                signal_payload={
                    'type': 'anti_constraints_set', 
                    'constraints': constraints, 
                    'expiry': expiry_step
                },
                context=context,
                epistemic_intent='UPDATE_ADAPTIVE_STATE'
            ))

    def clear_anti_constraints(self, context: Optional[NireonExecutionContext] = None) -> None:
        """Clear all active anti-constraints."""
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

    def _handle_duplication_detected(self, step: int):
        """Handle detection of semantic duplication."""
        self.current_blend, self.last_duplication_step = handle_duplication_detected(
            current_blend=self.current_blend,
            max_blend_low=self.catalyst_cfg.max_blend_low,
            max_blend_high=self.catalyst_cfg.max_blend_high,
            duplication_aggressiveness=self.catalyst_cfg.duplication_aggressiveness,
            current_step=step,
            event_bus=self.event_helper
        )
        
    async def health_check(self, context: NireonExecutionContext) -> Dict[str, Any]:
        """Check component health and service status."""
        health = {
            'status': 'healthy', 
            'component_id': self.component_id, 
            'checks': {}
        }
        
        # Use the mixin to validate all required services
        required_services = ['gateway', 'frame_factory', 'embed', 'idea_service']
        service_health = self.validate_required_services(required_services, context)
        
        # Add individual service checks for detailed status
        health['checks']['gateway'] = 'ok' if self.gateway else 'missing'
        health['checks']['frame_factory'] = 'ok' if self.frame_factory else 'missing'
        health['checks']['embedding_port'] = 'ok' if self.embed else 'missing'
        health['checks']['idea_service'] = 'ok' if self.idea_service else 'missing'
        health['checks']['all_services_available'] = service_health

        # Check cross-domain vectors
        if not self.cross_domain_vectors:
            health['status'] = 'degraded'
            health['checks']['cross_domain_vectors'] = 'missing'
        else:
            health['checks']['cross_domain_vectors'] = f"ok ({len(self.cross_domain_vectors)} domains)"

        # Check performance metrics
        if self._catalysis_stats['total_ideas_processed'] > 0:
            success_rate = (
                self._catalysis_stats['total_ideas_catalyzed'] / 
                self._catalysis_stats['total_ideas_processed']
            )
            health['checks']['success_rate'] = f'{success_rate:.1%}'
            if success_rate < 0.5:
                health['status'] = 'degraded'
                health['message'] = 'Low catalysis success rate'

        # Determine overall health
        if not service_health:
            health['status'] = 'unhealthy'
            health['message'] = 'Required services not available'
        elif any(v != 'ok' and (not isinstance(v, str) or 'missing' in str(v)) for v in health['checks'].values()):
            health['status'] = 'unhealthy' if health['status'] != 'degraded' else health['status']

        return health