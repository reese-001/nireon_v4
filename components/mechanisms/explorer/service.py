# services.py - Explorer Mechanism
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import random
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Any, Dict, Final, List, Optional

import shortuuid
import numpy as np

from application.services.frame_factory_service import FrameFactoryService, FrameNotFoundError
from application.services.budget_manager import BudgetExceededError
from core.base_component import NireonBaseComponent
from core.lifecycle import ComponentMetadata
from core.results import (
    AdaptationAction,
    AdaptationActionType,
    AnalysisResult,
    ComponentHealth,
    ProcessResult,
    SystemSignal,
    SignalType,
)
from domain.context import NireonExecutionContext
from domain.cognitive_events import CognitiveEvent, LLMRequestPayload
from domain.epistemic_stage import EpistemicStage
from domain.frames import Frame
from domain.ideas.idea import Idea
from domain.ports.llm_port import LLMResponse
from domain.ports.mechanism_gateway_port import MechanismGatewayPort
from domain.ports.embedding_port import EmbeddingPort
from domain.embeddings.vector import Vector as DomainVector
from signals.base import EpistemicSignal
from signals.core import GenerativeLoopFinishedSignal, IdeaGeneratedSignal, TrustAssessmentSignal
from components.mechanisms.base import ProducerMechanism 
from components.service_resolution_mixin import ServiceResolutionMixin


from .config import ExplorerConfig
from .errors import ExplorerErrorCode
from .service_helpers.explorer_event_helper import ExplorerEventHelper

# --------------------------------------------------------------------------- #
#  module-level constants & logger
# --------------------------------------------------------------------------- #
logger: Final = logging.getLogger(__name__)

_EXPLORER_METADATA = ComponentMetadata(
    id="explorer",
    name="Explorer Mechanism V4",
    version="1.7.0",  # ↑ bumped for major enhancements
    category="mechanism",
    description=(
        "Explorer mechanism for idea generation and systematic variation, "
        "using A-F-CE model (Frames + MechanismGateway)."
    ),
    epistemic_tags=[
        "generator",
        "variation",
        "mutator",
        "innovator",
        "divergence",
        "novelty_generation",
    ],
    capabilities={
        "generate_ideas",
        "explore_variations",
        "idea_mutation",
        "dynamic_frame_parameterization",
        "semantic_exploration",
        "sentinel_feedback_loop",
    },
    accepts=["SEED_SIGNAL", "EXPLORATION_REQUEST"],
    produces=[
        "IdeaGeneratedSignal",
        "ExplorationCompleteSignal",
        "FrameProcessingFailedSignal",
        "TrustAssessmentSignal",
    ],
    requires_initialize=True,
    dependencies={"MechanismGatewayPort": ">=1.0.0", "FrameFactoryService": "*", "EmbeddingPort": "*"},
    interaction_pattern='producer'
)

# Audit-trail limits
_MAX_AUDIT_ENTRIES: Final = 50
_MAX_AUDIT_BYTES: Final = 10 * 1024  # ~10 KB

__all__ = ["ExplorerMechanism"]


class ExplorerMechanism(ProducerMechanism, ServiceResolutionMixin):
    """
    Explorer is a PRODUCER mechanism that generates new idea variations.
    It publishes IdeaGeneratedSignal directly via the gateway/event system.
    
    Required Services:
        - gateway (MechanismGatewayPort): For LLM communication and event publishing
        - frame_factory (FrameFactoryService): For frame management
        - embed (EmbeddingPort): For semantic exploration
    """

    METADATA_DEFINITION = _EXPLORER_METADATA
    ConfigModel = ExplorerConfig

    # --------------------------------------------------------------------- #
    #  Construction & initialisation
    # --------------------------------------------------------------------- #
    def __init__(
        self,
        config: Dict[str, Any],
        metadata_definition: ComponentMetadata | None,
        gateway: MechanismGatewayPort | None = None,
        frame_factory: FrameFactoryService | None = None,
        embed: EmbeddingPort | None = None,
    ) -> None:
        super().__init__(config=config, metadata_definition=metadata_definition or self.METADATA_DEFINITION)
        self.cfg: ExplorerConfig = ExplorerConfig(**self.config)
        
        # Allow dependency injection through constructor (useful for testing)
        self.gateway = gateway
        self.frame_factory = frame_factory
        self.embed = embed
        self.event_helper: Optional[ExplorerEventHelper] = None

        # Single instance RNG for deterministic behavior
        self._rng = random.Random()
        self._exploration_count = 0
        self._pending_embedding_requests: Dict[str, Any] = {}
        self._assessment_events: Dict[str, asyncio.Event] = {}
        self._frame_assessment_trackers: Dict[str, Dict[str, asyncio.Event]] = defaultdict(dict)
        self.last_n_frame_stats: deque = deque(maxlen=10)

        logger.info(
            "ExplorerMechanism '%s' (%s v%s) instantiated. Gateway & FrameFactory resolved at init().",
            self.component_id,
            self.metadata.name,
            self.metadata.version,
        )
        logger.debug("Explorer initial config:\n%s", self.cfg.model_dump_json(indent=2))

    async def _initialize_impl(self, context: NireonExecutionContext) -> None:
        """Initialize Explorer with dependency resolution."""
        context.logger.info("ExplorerMechanism '%s' initialising…", self.component_id)

        # Resolve all required services using the mixin
        await self._resolve_all_dependencies(context)
        
        # Validate dependencies
        self._validate_dependencies(context)

        # Helper wraps gateway + registry access
        self.event_helper = ExplorerEventHelper(
            self.gateway, self.component_id, self.metadata.version, registry=context.component_registry
        )
        context.logger.info("ExplorerEventHelper initialised for '%s'.", self.component_id)

        # Initialize exploration strategies
        self._init_exploration_strategies(context)
        
        # Seed RNG for deterministic behavior
        if context.replay_mode and context.replay_seed is not None:
            self._rng.seed(context.replay_seed)
            context.logger.info(f"Explorer RNG seeded with {context.replay_seed} for deterministic replay.")
        elif self.cfg.seed_randomness is not None:
            self._rng.seed(self.cfg.seed_randomness)
            context.logger.info(f"Explorer RNG seeded with {self.cfg.seed_randomness} from configuration.")
        
        context.logger.info("✓ ExplorerMechanism '%s' initialised.", self.component_id)

    async def _resolve_all_dependencies(self, context: NireonExecutionContext) -> None:
        """Resolve all required dependencies using the ServiceResolutionMixin."""
        
        # Build service map only for services not already injected
        service_map = {}
        
        if not self.gateway:
            service_map['gateway'] = MechanismGatewayPort
        if not self.frame_factory:
            service_map['frame_factory'] = FrameFactoryService
        if not self.embed:
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
        
        required_services = ['gateway', 'frame_factory', 'embed']
        
        # Use the mixin's validation method
        if not self.validate_required_services(required_services, context):
            # The mixin will have logged which services are missing
            missing = [s for s in required_services if not getattr(self, s, None)]
            raise RuntimeError(
                f"ExplorerMechanism '{self.component_id}' missing critical dependencies: {', '.join(missing)}"
            )

    def _ensure_services_available(self, context: NireonExecutionContext) -> bool:
        """
        Ensure required services are available at runtime.
        Can attempt re-resolution if services are missing.
        """
        required_services = ['gateway', 'frame_factory', 'embed']
        
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
            if not self.embed:
                service_map['embed'] = EmbeddingPort
            
            if service_map:
                self.resolve_services(
                    context=context,
                    service_map=service_map,
                    raise_on_missing=False,  # Don't raise, we'll check below
                    log_resolution=True
                )
                
                # Re-create event helper if gateway was re-resolved
                if self.gateway and not self.event_helper:
                    self.event_helper = ExplorerEventHelper(
                        self.gateway, self.component_id, self.metadata.version, 
                        registry=context.component_registry
                    )
            
            # Check again after resolution attempt
            return self.validate_required_services(required_services)
            
        except Exception as e:
            context.logger.error(f"[{self.component_id}] Failed to ensure services: {e}")
            return False

    # --------------------------------------------------------------------- #
    #  Internal helpers
    # --------------------------------------------------------------------- #
    def _init_exploration_strategies(self, context: NireonExecutionContext) -> None:
        context.logger.debug("Strategy for '%s': %s", self.component_id, self.cfg.exploration_strategy)

    # --------------------------- prompt builder -------------------------- #
    def _build_llm_prompt(
        self,
        seed_text: str,
        objective: str,
        attempt_details: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Return the generator prompt for a single variation attempt."""
        # Check if the seed text contains LaTeX-like patterns, indicating a formal idea
        is_formal_idea = '\\(' in seed_text or '\\[' in seed_text or '\\min' in seed_text or '\\ge' in seed_text

        if is_formal_idea:
            # Use a specialized prompt for formal/mathematical ideas
            template = self.cfg.default_prompt_template or """
            You are a creative mathematician. Your task is to generate a plausible and interesting variation of a formal mathematical or logical concept.
            Preserve the five-part template structure of the original idea if present.
            
            **Original Concept:**
            {seed_idea_text}
            
            **Overall Objective:**
            {objective}
            
            **Instructions:**
            1.  Identify a key component of the original concept to modify (e.g., the function, the set, the operation).
            2.  Create a novel variation by altering that component. For example, change "prime divisors" to "Fibonacci numbers" or "digit-sum" to "product of digits".
            3.  Re-compute or logically deduce the consequences of your change, filling out the 5-part structure accordingly.
            4.  The new idea must be a complete, self-contained concept.
            
            **New Varied Concept:**
            """
        else:
            # Use the general-purpose creative prompt for narrative ideas
            template = self.cfg.default_prompt_template or """
            Generate a creative and divergent variation of the following idea: '{seed_idea_text}'. 
            The overall objective is: {objective}. 
            Aim for a {creativity_factor_desc} degree of novelty. 
            The generated idea should be between {desired_length_min} and {desired_length_max} characters. 
            Respond with ONLY the full text of the new, varied idea.
            """

        creativity_desc = (
            "high" if self.cfg.creativity_factor > 0.7 else "medium" if self.cfg.creativity_factor > 0.3 else "low"
        )
        
        vars_ = {
            "seed_idea_text": seed_text,
            "objective": objective,
            "creativity_factor_desc": creativity_desc,
            "desired_length_min": self.cfg.minimum_idea_length,
            "desired_length_max": self.cfg.maximum_idea_length,
            "vector_distance": (attempt_details or {}).get("vector_distance", 0.0)
        }
        
        try:
            return template.format(**vars_).strip()
        except KeyError as exc:
            logger.warning("[%s] Prompt template missing key %s - falling back to basic prompt.", self.component_id, exc)
            return f"Generate a creative variation of: {seed_text}. Objective: {objective}."

    # ----------------------------- auditing ------------------------------ #
    def _add_audit_log(
        self,
        frame: Optional[Frame],
        event_type: str,
        summary: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Append an entry to the frame's audit trail, spilling to logger if full."""
        if not (frame and isinstance(frame.context_tags.get("audit_trail"), list)):
            logger.warning("[%s] Audit-trail unavailable for frame %s.", self.component_id, getattr(frame, "id", "?"))
            return

        audit_trail: List[Dict[str, Any]] = frame.context_tags["audit_trail"]
        current_bytes = sum(len(json.dumps(item)) for item in audit_trail)

        if len(audit_trail) >= _MAX_AUDIT_ENTRIES or current_bytes > _MAX_AUDIT_BYTES:
            logger.info(
                "[FrameAuditSpill][%s] %s - %s | details=%s",
                frame.id,
                event_type,
                summary,
                details,
            )
            return

        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "summary": summary,
            **({"details": details} if details else {}),
        }
        audit_trail.append(entry)

    # -------------------- embedding request helper ---------------------- #
    async def _request_embedding_for_idea(
        self, frame: Frame, idea_text: str, idea_id: str, context: NireonExecutionContext
    ) -> None:
        if self.event_helper is None:
            context.logger.error("[%s] EventHelper unavailable - cannot request embedding.", self.component_id)
            return
        if len(self._pending_embedding_requests) >= self.cfg.max_pending_embedding_requests:
            context.logger.warning(
                "[%s] Pending-embedding cap reached (%d) - skipping request for %s.",
                self.component_id,
                self.cfg.max_pending_embedding_requests,
                idea_id,
            )
            self._add_audit_log(
                frame,
                "EMBEDDING_REQUEST_SKIPPED",
                f"Max pending requests reached; skipped for {idea_id}.",
                {"target_idea_id": idea_id},
            )
            return

        request_id = f"emb_req_{self.component_id}_{shortuuid.uuid()}"
        embedding_payload = {
            "request_id": request_id,
            "text_to_embed": idea_text,
            "target_artifact_id": idea_id,
            "request_timestamp_ms": int(time.time() * 1000),
            "embedding_vector_dtype": "float32",
            "metadata": {
                **self.cfg.embedding_request_metadata,
                "frame_id": frame.id,
                "origin_component_id": self.component_id,
            },
        }
        self._pending_embedding_requests[request_id] = {
            "idea_id": idea_id,
            "frame_id": frame.id,
            "text": idea_text,
            "timestamp": time.time(),
        }
        self._add_audit_log(frame, "EMBEDDING_REQUESTED", f"Embedding requested for {idea_id}.", {"embedding_request_id": request_id})

        signal = EpistemicSignal(
            signal_type="EmbeddingRequestSignal",
            source_node_id=self.component_id,
            payload=embedding_payload,
            context_tags={"frame_id": frame.id},
        )
        await self.event_helper.publish_signal(signal, context)
        context.logger.info("[%s] Embedding requested for idea '%s' (request_id=%s).", self.component_id, idea_id, request_id)

    # --------------------------------------------------------------------- #
    #  Main process entry-point
    # --------------------------------------------------------------------- #
    async def _process_impl(self, data: Any, context: NireonExecutionContext) -> ProcessResult:
        if not self._ensure_services_available(context):
            return ProcessResult(success=False, component_id=self.component_id, message='Required services not available', error_code='MISSING_DEPENDENCIES')
        
        self._exploration_count += 1
        task_id = shortuuid.uuid()[:8]
        session_id = None
        planner_action = None
        parent_trust_score = None
        objective = None
        if isinstance(data, dict):
            session_id = data.get('session_id')
            if not session_id and 'metadata' in data:
                session_id = data['metadata'].get('session_id')
            if not session_id and 'payload' in data:
                session_id = data['payload'].get('session_id')
            planner_action = data.get('planner_action')
            parent_trust_score = data.get('parent_trust_score')
            objective = data.get('objective') or data.get('metadata', {}).get('objective') or (data.get('payload', {}).get('objective') if isinstance(data.get('payload'), dict) else None)
        
        if not objective:
            objective = 'Generate novel idea variations.'
        
        existing_idea_id = None
        if isinstance(data, dict):
            existing_idea_id = data.get('id') or data.get('seed_idea_id') or data.get('idea_id') or (data.get('payload', {}).get('seed_idea_id') if isinstance(data.get('payload'), dict) else None)
        
        seed_text, seed_id, current_depth = self._extract_seed_and_depth(data, context)
        context.logger.info(f'[{self.component_id}] Input data type: {type(data)}')
        if isinstance(data, dict):
            context.logger.info(f'[{self.component_id}] Input data keys: {list(data.keys())}')
            context.logger.info(f'[{self.component_id}] Input metadata: {data.get("metadata", "No metadata")}')
        is_formal = '\\(' in seed_text or '\\[' in seed_text or '\\min' in seed_text or ('\\ge' in seed_text)
        if isinstance(data, dict):
            input_meta = data.get('metadata', {})
            if 'is_formal' in input_meta:
                context.logger.info(f'[{self.component_id}] Overriding is_formal from metadata to {input_meta["is_formal"]}')
                is_formal = input_meta['is_formal']
        context.logger.info(f'[{self.component_id}] Idea classified as formal: {is_formal}')
        
        if existing_idea_id:
            context.logger.info(f'[{self.component_id}] Found existing idea ID: {existing_idea_id}')
            seed_id = existing_idea_id
            context.logger.info(f'[{self.component_id}] Using existing seed idea: {seed_id}')
        else:
            context.logger.debug(f'[{self.component_id}] No existing idea ID found, using generated/extracted seed_id: {seed_id}')
        
        # Create frame FIRST with seed_id in metadata
        frame_metadata = {'depth': current_depth, 'is_formal': is_formal, 'session_id': session_id, 'planner_action': planner_action, 'parent_trust_score': parent_trust_score, 'objective': objective, 'seed_id': seed_id, 'task_id': task_id}
        context.logger.info(f'[{self.component_id}] Creating frame with context_tags: {frame_metadata}')
        
        frame = await self.frame_factory.create_frame(context=context, name=f'explorer_task_{task_id}', owner_agent_id=self.component_id, description=f'Exploration frame for seed {seed_id[:8]} with objective: {objective[:50]}...', parent_frame_id=self._get_current_or_default_frame_id(context), context_tags=frame_metadata, resource_budget=self.cfg.default_resource_budget_for_exploration)
        frame.context_tags['audit_trail'] = []
        self._add_audit_log(frame, 'EXPLORATION_REQUESTED', f'Task {task_id} initiated.')
        
        # NEW: Set current_frame_id in context metadata for publishing
        context.metadata['current_frame_id'] = frame.id
        
        # Persist seed AFTER frame creation (now publish will have frame_id)
        seed_meta = {'objective': objective, 'depth': current_depth, 'is_formal': is_formal, 'is_seed': True}
        seed_idea = self.event_helper.create_and_persist_idea(
            text=seed_text,
            parent_id=None,
            context=context,
            metadata=seed_meta,
            idea_id=seed_id  # Explicitly pass the seed_id to use it as the idea_id
        )
        current_seed_id = seed_idea.idea_id  # This should now match seed_id
        context.logger.info(f'[{self.component_id}] Seed idea persisted with ID: {current_seed_id}')
        
        exploration_task = asyncio.create_task(self._run_exploration_in_background(frame=frame, seed_text=seed_text, seed_id=current_seed_id, objective=objective, context=context, session_id=session_id, planner_action=planner_action))
        frame.context_tags['exploration_task'] = exploration_task
        frame.context_tags['exploration_task_id'] = task_id
        
        return ProcessResult(success=True, component_id=self.component_id, message=f"Exploration task {task_id} launched for seed '{current_seed_id}' in frame {frame.id}", output_data={'task_id': task_id, 'frame_id': frame.id, 'seed_id': current_seed_id, 'session_id': session_id, 'depth': current_depth, 'completion_future': exploration_task})
    def _get_current_or_default_frame_id(self, context: NireonExecutionContext) -> Optional[str]:
        """Get the current frame ID from context or return None for root frame."""
        if context.metadata and 'frame_id' in context.metadata:
            return context.metadata['frame_id']
        return None  # Will create a root frame

    # ----------------------- seed extraction helper ---------------------- #
    @staticmethod
    def _extract_seed_and_depth(data: Any, ctx: NireonExecutionContext) -> tuple[str, str, int]:
        """
        Extracts the seed idea, its ID, and exploration depth from the input data.
        Establishes a clear order of precedence for finding the seed.
        """
        seed_text: str = ''
        seed_id: Optional[str] = None
        depth: int = 0
        
        if isinstance(data, dict):
            # Metadata is a primary source for depth and session info
            metadata = data.get('metadata', {})
            if isinstance(metadata, dict):
                depth = metadata.get('depth', 0)

            # Highest precedence: A single Idea object in a list
            ideas_list = data.get('ideas')
            if isinstance(ideas_list, list) and len(ideas_list) > 0:
                first_idea = ideas_list[0]
                if isinstance(first_idea, Idea):
                    seed_text = first_idea.text
                    seed_id = first_idea.idea_id
                    if len(ideas_list) > 1:
                        ctx.logger.warning(f"Multiple ideas provided; using the first one ('{seed_id}') as the seed.")

            # Next precedence: Explicit seed_idea_id or text in the main payload
            if not seed_text:
                seed_text = data.get('text') or data.get('seed_idea_text')
                seed_id = data.get('id') or data.get('seed_idea_id') or data.get('idea_id')

        elif isinstance(data, str):
            seed_text = data

        # If still no seed, create a default one
        if not seed_text:
            default_id = f'seed_{shortuuid.uuid()[:8]}'
            seed_text = f'default_seed_{default_id}'
            seed_id = default_id
            ctx.logger.warning('No valid seed idea found in input data. Using a default seed.')

        # Ensure seed_id is never None
        if seed_id is None:
            seed_id = f'seed_{hashlib.sha256(seed_text.encode()).hexdigest()[:12]}'

        return (seed_text, seed_id, depth)

    # --------------------------------------------------------------------- #
    #  Semantic exploration method
    # --------------------------------------------------------------------- #
    async def _semantic_exploration(self, frame: Frame, seed_text: str, objective: str, context: NireonExecutionContext) -> List[LLMResponse]:
        """
        Performs exploration by perturbing the seed idea's vector and using the new
        vector to guide LLM generation.
        """
        if not self.embed:
            context.logger.error(f"[{self.component_id}] Semantic exploration failed: EmbeddingPort is not available.")
            return []

        try:
            seed_vector = self.embed.encode(seed_text)
        except Exception as e:
            context.logger.error(f"[{self.component_id}] Could not get embedding for seed text: {e}")
            return []

        llm_tasks = []
        for i in range(self.cfg.max_variations_per_level):
            try:
                # 1. Create a random perturbation vector
                perturbation = np.random.standard_normal(seed_vector.dims)
                perturbation_unit_vector = perturbation / np.linalg.norm(perturbation)

                # 2. Add the scaled perturbation to the original vector
                perturbed_vector_data = seed_vector.data + perturbation_unit_vector * self.cfg.divergence_strength
                
                # 3. Normalize the new vector
                final_vector_data = perturbed_vector_data / np.linalg.norm(perturbed_vector_data)
                final_vector = DomainVector(final_vector_data)
                
                # 4. Craft a prompt that guides the LLM to "decode" the new semantic location
                # ... MODIFIED: Use the correct method to calculate semantic distance.
                semantic_distance = 1.0 - seed_vector.similarity(final_vector)
                
                prompt = (
                    f"You are a creative synthesizer. Your task is to generate a new idea that embodies a specific semantic concept.\n\n"
                    f"The original 'seed' idea is: '{seed_text}'\n\n"
                    f"We have semantically shifted this idea to a new conceptual location. This new location is related to the seed but has a conceptual distance of {semantic_distance:.3f} (where 1.0 is very different).\n\n"
                    f"The overall objective is: {objective}\n\n"
                    f"Your task: Write the text for the new idea that exists at this new conceptual point. Be creative and interpret what this semantic shift might imply. Respond with ONLY the full text of the new idea."
                )

                llm_payload = LLMRequestPayload(prompt=prompt, stage=EpistemicStage.EXPLORATION, role='semantic_synthesizer')
                ce = CognitiveEvent(
                    frame_id=frame.id, 
                    owning_agent_id=self.component_id, 
                    service_call_type='LLM_ASK', 
                    payload=llm_payload, 
                    epistemic_intent='GENERATE_SEMANTIC_VARIATION',
                    custom_metadata={'attempt': i + 1, 'semantic_distance_target': semantic_distance}
                )
                llm_tasks.append(self.gateway.process_cognitive_event(ce, context))
                self._add_audit_log(frame, 'SEMANTIC_LLM_CE_CREATED', f'Semantic LLM_ASK CE for attempt {i + 1}.', {'ce_id': ce.event_id})

            except BudgetExceededError as budget_exc:
                context.logger.warning(f"[{self.component_id}] Budget exceeded during semantic exploration. Stopping. Details: {budget_exc}")
                self._add_audit_log(frame, 'BUDGET_EXCEEDED', "Stopping semantic generation due to budget exhaustion.")
                break
            except Exception as e:
                context.logger.error(f"[{self.component_id}] Error creating semantic cognitive event: {e}", exc_info=True)

        # Gather all responses
        return await asyncio.gather(*llm_tasks, return_exceptions=True)

    # --------------------------------------------------------------------- #
    #  Background exploration loop
    # --------------------------------------------------------------------- #
    async def _run_exploration_in_background(
        self,
        frame: Frame,
        seed_text: str,
        seed_id: str,
        objective: str,
        context: NireonExecutionContext,
        session_id: Optional[str] = None,
        planner_action: Optional[str] = None,
    ) -> None:
        """
        Runs a multi-level exploration task in the background.

        This method now supports two primary modes based on configuration:
        1.  **Sentinel Feedback Loop (if enabled):** Generates variations, waits for
            Sentinel to assess them, and then selects the highest-scoring idea
            to seed the next level of exploration. This creates an intelligent,
            hill-climbing search.
        2.  **Open Loop (if disabled):** Generates variations and proceeds to the
            next level using the first-generated idea, without waiting for feedback.
            This is faster but less intelligent.
        """
        current_seed_text = seed_text
        current_seed_id = seed_id
        
        try:
            for depth in range(self.cfg.max_depth):
                context.logger.info(f"[{self.component_id}] Frame {frame.id}: Starting exploration at depth {depth+1} with seed '{current_seed_id}'.")
                self._add_audit_log(frame, 'EXPLORATION_LEVEL_START', f'Starting depth {depth+1}.', {'seed_id': current_seed_id})

                # --- Step 1: Generate Variations for the current level ---
                responses: List[Any]
                
                if self.cfg.is_semantic_enabled() and self.cfg.exploration_strategy != 'random':
                    context.logger.info(f"[{self.component_id}] Using Semantic Exploration strategy for frame {frame.id}.")
                    responses = await self._semantic_exploration(frame, current_seed_text, objective, context)
                else:
                    # Fallback to existing text-based generation
                    llm_tasks = []
                    for i in range(self.cfg.max_variations_per_level):
                        try:
                            prompt = self._build_llm_prompt(current_seed_text, objective, {'attempt_index': i})
                            llm_payload = LLMRequestPayload(prompt=prompt, stage=EpistemicStage.EXPLORATION, role='idea_generator', llm_settings=self.cfg.default_llm_policy_for_exploration)
                            ce = CognitiveEvent(
                                frame_id=frame.id, 
                                owning_agent_id=self.component_id, 
                                service_call_type='LLM_ASK', 
                                payload=llm_payload, 
                                epistemic_intent='GENERATE_IDEA_VARIATION', 
                                custom_metadata={'attempt': i + 1, 'seed_hash': hashlib.sha256(current_seed_text.encode()).hexdigest()[:12]}
                            )
                            llm_tasks.append(self.gateway.process_cognitive_event(ce, context))
                            self._add_audit_log(frame, 'LLM_CE_CREATED', f'LLM_ASK CE for attempt {i + 1}.', {'ce_id': ce.event_id})
                        except BudgetExceededError as budget_exc:
                            context.logger.warning(f"[{self.component_id}] Budget exceeded in frame {frame.id}. Stopping further generation. Details: {budget_exc}")
                            self._add_audit_log(frame, 'BUDGET_EXCEEDED', f"Stopping generation at attempt {i+1} due to budget exhaustion.")
                            break
                    responses = await asyncio.gather(*llm_tasks, return_exceptions=True)
                
                # --- Step 2: Persist new ideas and set up assessment tracking ---
                generated_ideas: Dict[str, Idea] = {}
                assessment_events: Dict[str, asyncio.Event] = {}

                for i, resp in enumerate(responses):
                    if isinstance(resp, LLMResponse) and resp.text and not resp.get('error'):
                        var_text = resp.text.strip()
                        if len(var_text) >= self.cfg.minimum_idea_length:
                            frame_depth = frame.context_tags.get("depth", 0)
                            new_depth = frame_depth + depth + 1
                            
                            new_meta = {"depth": new_depth, "session_id": session_id, "planner_action": planner_action,
                                        "is_formal": frame.context_tags.get('is_formal', False)}
                            idea = self.event_helper.create_and_persist_idea(var_text, parent_id=current_seed_id, context=context, metadata=new_meta)
                            generated_ideas[idea.idea_id] = idea

                            # Set up an event to wait for this idea's assessment
                            assessment_events[idea.idea_id] = asyncio.Event()
                            self._frame_assessment_trackers[frame.id][idea.idea_id] = assessment_events[idea.idea_id]
                            
                            idea_payload = {
                                'id': idea.idea_id, 'text': var_text, 'parent_id': current_seed_id,
                                'source_mechanism': self.component_id, 'derivation_method': self.cfg.exploration_strategy,
                                'seed_idea_text_preview': current_seed_text[:50], 'frame_id': frame.id,
                                'objective': objective, 'metadata': new_meta,
                                'llm_response_metadata': {k: v for k, v in resp.items() if k != 'text'}
                            }
                            signal = IdeaGeneratedSignal(
                                source_node_id=self.component_id, idea_id=idea.idea_id, idea_content=var_text,
                                generation_method=self.cfg.exploration_strategy, payload=idea_payload,
                                context_tags={"frame_id": frame.id}
                            )
                            await self.event_helper.publish_signal(signal, context)
                            self._add_audit_log(frame, "IDEA_GENERATED", f"Variation {i + 1} generated.", {"idea_id": idea.idea_id})

                if not generated_ideas:
                    context.logger.warning(f"[{self.component_id}] Frame {frame.id}: No valid variations generated at depth {depth+1}. Halting exploration.")
                    self._add_audit_log(frame, 'EXPLORATION_HALTED', f'No variations at depth {depth+1}.')
                    break

                # --- LOGIC BRANCH FOR SENTINEL FEEDBACK ---
                if not self.cfg.is_sentinel_feedback_enabled():
                    context.logger.info(f"[{self.component_id}] Frame {frame.id}: Sentinel feedback disabled. Selecting first variation as next seed.")
                    first_idea = list(generated_ideas.values())[0]
                    current_seed_text = first_idea.text
                    current_seed_id = first_idea.idea_id
                    continue # Move to the next depth level without waiting

                # --- Step 3: Wait for Sentinel assessments ---
                context.logger.info(f"[{self.component_id}] Frame {frame.id}: Waiting for assessments for {len(generated_ideas)} new ideas...")
                wait_tasks = [event.wait() for event in assessment_events.values()]
                try:
                    await asyncio.wait_for(asyncio.gather(*wait_tasks), timeout=self.cfg.exploration_timeout_seconds)
                except asyncio.TimeoutError:
                    context.logger.warning(f"[{self.component_id}] Frame {frame.id}: Timed out waiting for assessments after {self.cfg.exploration_timeout_seconds}s. Proceeding with received scores.")
                    self._add_audit_log(frame, 'ASSESSMENT_TIMEOUT', 'Proceeding with partial scores.')

                # --- Step 4: Select the best variation to be the next seed ---
                assessed_scores = frame.context_tags.get('variation_assessments', {})
                
                if not assessed_scores:
                    context.logger.warning(f"[{self.component_id}] Frame {frame.id}: No assessments received. Cannot continue depth-first search.")
                    self._add_audit_log(frame, 'EXPLORATION_HALTED', 'No assessments received from Sentinel.')
                    break

                best_idea_id: Optional[str] = None
                highest_score = -1.0
                for idea_id, score in assessed_scores.items():
                    if idea_id in generated_ideas and score > highest_score:
                        highest_score = score
                        best_idea_id = idea_id
                
                if best_idea_id is None:
                    context.logger.warning(f"[{self.component_id}] Frame {frame.id}: Could not determine a best idea from assessments. Halting.")
                    self._add_audit_log(frame, 'EXPLORATION_HALTED', 'Could not select a best idea.')
                    break
                    
                context.logger.info(f"[{self.component_id}] Frame {frame.id}: Best variation at depth {depth+1} is '{best_idea_id}' with score {highest_score:.2f}.")
                self._add_audit_log(frame, 'BEST_VARIATION_SELECTED', f'Best idea: {best_idea_id} (Score: {highest_score:.2f})', assessed_scores)
                
                # --- Step 5: Update seed for the next loop iteration ---
                current_seed_text = generated_ideas[best_idea_id].text
                current_seed_id = best_idea_id
            
            # --- After Loop: Finalize ---
            context.logger.info(f'[{self.component_id}] BG-TASK: Exploration loop finished for frame {frame.id}.')
            
            if self.event_helper:
                completion_signal = GenerativeLoopFinishedSignal(
                    source_node_id=self.component_id,
                    payload={
                        "status": "completed_ok", "frame_id": frame.id,
                        "generated_idea_count": len(frame.context_tags.get('variation_assessments', {})),
                        "source_idea_id": seed_id, "final_seed_id": current_seed_id
                    },
                    context_tags={"frame_id": frame.id}
                )
                await self.event_helper.publish_signal(completion_signal, context)

        except Exception as exc:
            context.logger.error("BG-TASK: Critical exploration error in frame %s: %s", frame.id, exc, exc_info=True)
            await self.frame_factory.update_frame_status(context, frame.id, "error_internal")
            self._add_audit_log(frame, "FRAME_ERROR", "Frame status set to error_internal due to an unhandled exception.")
        finally:
            # Clean up the tracker for this frame to prevent memory leaks
            if frame.id in self._frame_assessment_trackers:
                del self._frame_assessment_trackers[frame.id]

    # --------------------------------------------------------------------- #
    #  Variation helpers (no functional change, minor refactor)
    # --------------------------------------------------------------------- #
    
    def _extract_seed_idea(self, data: Any, context: NireonExecutionContext) -> str:
        if isinstance(data, str):
            return data
        elif isinstance(data, dict):
            if 'text' in data:
                return data['text']
            elif 'seed' in data:
                return data['seed']
            elif 'idea' in data:
                if isinstance(data['idea'], dict) and 'text' in data['idea']:
                    return data['idea']['text']
                elif isinstance(data['idea'], str):
                    return data['idea']
        default_seed = 'Explore new possibilities and generate innovative ideas'
        context.logger.debug(f'Using default seed idea for exploration: {default_seed}')
        return default_seed

    async def _generate_variations(self, seed_idea: str, context: NireonExecutionContext) -> List[str]:
        strat = self.cfg.exploration_strategy
        try:
            if strat == "depth_first":
                return await self._depth_first_exploration(seed_idea, context)
            if strat == "breadth_first":
                return await self._breadth_first_exploration(seed_idea, context)
            if strat == "random":
                return await self._random_exploration(seed_idea, context)
            context.logger.warning("Unknown strategy '%s'; falling back to _simple_variations.", strat)
        except Exception as exc:
            context.logger.error("Variation generation failed (%s). Falling back to simple variations.", exc)
        return await self._simple_variations(seed_idea, context)
        
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
            if depth >= self.cfg.max_depth:
                continue

            for i in range(self.cfg.max_variations_per_level):
                if len(variations) >= max_vars_total: break
                new_variation = f'Breadth {depth+1}, Var {i+1} from "{current_idea[:20]}...": Expanded view of {current_idea}'
                variations.append(new_variation)
                queue.append((new_variation, depth + 1))
        
        context.logger.debug(f'Breadth-first exploration generated {len(variations)} variations from {visited_ideas_count} nodes.')
        return variations

    async def _random_exploration(self, seed_idea: str, ctx: NireonExecutionContext) -> List[str]:
        templates = [
            "Random mutation: {seed} with unexpected twist",
            "Chance combination: {seed} meets serendipity",
            "Stochastic enhancement: {seed} through probability",
            "Emergent property: {seed} with chaotic elements",
            "Quantum variation: {seed} in superposition",
        ]
        n = min(self.cfg.max_variations_per_level * self.cfg.max_depth, self.cfg.max_variations_per_level * 2)
        vars_ = [self._rng.choice(templates).format(seed=seed_idea) for _ in range(n)]
        ctx.logger.debug("Random exploration generated %d variations.", len(vars_))
        return vars_

    async def _simple_variations(self, seed_idea: str, context: NireonExecutionContext) -> List[str]:
        variations = [
            f'Basic variation 1: {seed_idea} with improvement',
            f'Basic variation 2: {seed_idea} with modification',
            f'Basic variation 3: Alternative to {seed_idea}'
        ]
        context.logger.debug(f'Simple exploration generated {len(variations)} variations')
        return variations[:self.cfg.max_variations_per_level]

    async def analyze(self, context: NireonExecutionContext) -> AnalysisResult:
        """
        Analyze exploration effectiveness based on recent frame outcomes.
        """
        context.logger.debug(f"[{self.component_id}] analyze() called.")
        
        # Check service availability
        service_health = self.validate_required_services(['gateway', 'frame_factory', 'embed'], context)
        
        metrics = {
            'total_explorations_by_instance': self._exploration_count,
            'configured_strategy': self.cfg.exploration_strategy,
            'recent_frames_analyzed': len(self.last_n_frame_stats),
            'current_divergence_strength_config': self.cfg.divergence_strength,
            'current_max_parallel_llm_calls_config': self.cfg.max_parallel_llm_calls_per_frame,
            'pending_embedding_requests': len(self._pending_embedding_requests),
            'services_available': service_health,
            'semantic_exploration_enabled': self.cfg.enable_semantic_exploration,
            'sentinel_feedback_enabled': self.cfg.enable_sentinel_feedback
        }
        insights = []
        anomalies = []
        recommendations = []  
        trends = {}

        if not self.last_n_frame_stats:
            insights.append("No frame data available for detailed analysis yet.")
            confidence = 0.3
        else:
            successful_variations_generated = 0
            degraded_completions = 0
            error_completions = 0
            total_llm_calls_in_recent_frames = 0
            successful_llm_calls = 0
            failed_llm_calls = 0

            for stat in self.last_n_frame_stats:
                successful_variations_generated += stat.get("variations_generated", 0)
                total_llm_calls_in_recent_frames += stat.get("llm_calls_made", 0)
                successful_llm_calls += stat.get("llm_call_successes", 0)
                failed_llm_calls += stat.get("llm_call_failures", 0)
                
                if stat.get("status") == "completed_degraded":
                    degraded_completions += 1
                elif str(stat.get("status")).startswith("error_"):
                    error_completions += 1
            
            # Calculate metrics
            metrics['avg_variations_per_recent_frame'] = successful_variations_generated / len(self.last_n_frame_stats)
            metrics['degraded_completion_rate_recent'] = degraded_completions / len(self.last_n_frame_stats)
            metrics['error_completion_rate_recent'] = error_completions / len(self.last_n_frame_stats)
            metrics['llm_success_rate'] = successful_llm_calls / total_llm_calls_in_recent_frames if total_llm_calls_in_recent_frames > 0 else 0
            metrics['variation_generation_efficiency'] = successful_variations_generated / total_llm_calls_in_recent_frames if total_llm_calls_in_recent_frames > 0 else 0

            # Generate insights and recommendations
            if metrics['avg_variations_per_recent_frame'] < (self.cfg.max_variations_per_level / 2):
                insights.append(f"Average variations per frame ({metrics['avg_variations_per_recent_frame']:.2f}) is low compared to target ({self.cfg.max_variations_per_level}).")
                recommendations.append("CONSIDER_INCREASING_DIVERGENCE_OR_REVISING_PROMPTS")
                trends['variation_generation'] = 'down'
            else:
                trends['variation_generation'] = 'stable'
            
            if metrics['error_completion_rate_recent'] > 0.2:
                insights.append(f"High error rate in recent frames ({metrics['error_completion_rate_recent']:.2%}).")
                anomalies.append({
                    "metric": "frame_error_rate",
                    "value": metrics['error_completion_rate_recent'],
                    "expected": 0.05,
                    "severity": "high"
                })
                recommendations.append("INVESTIGATE_FRAME_ERRORS")
                trends['error_rate'] = 'up'
            else:
                trends['error_rate'] = 'stable'

            if metrics['degraded_completion_rate_recent'] > 0.3:
                insights.append(f"High rate of degraded completions ({metrics['degraded_completion_rate_recent']:.2%}). Exploration might not be effective.")
                recommendations.append("REVIEW_EXPLORATION_EFFECTIVENESS")
                anomalies.append({
                    "metric": "degraded_completion_rate",
                    "value": metrics['degraded_completion_rate_recent'],
                    "expected": 0.1,
                    "severity": "medium"
                })

            if metrics['llm_success_rate'] < 0.8:
                insights.append(f"LLM success rate ({metrics['llm_success_rate']:.2%}) is below expected threshold.")
                recommendations.append("REVIEW_LLM_CONFIGURATION")
                
            if metrics['variation_generation_efficiency'] < 0.5:
                insights.append(f"Low efficiency in variation generation ({metrics['variation_generation_efficiency']:.2f} variations per LLM call).")
                recommendations.append("OPTIMIZE_PROMPT_TEMPLATES")

            # Calculate confidence based on data quality
            confidence = min(0.9, 0.3 + (len(self.last_n_frame_stats) * 0.1))

        # Store recommendations in result
        result = AnalysisResult(
            success=True,
            component_id=self.component_id,
            metrics=metrics,
            confidence=confidence,
            message=f"Analysis of {len(self.last_n_frame_stats)} recent frames complete. Insights: {len(insights)}, Anomalies: {len(anomalies)}.",
            insights=insights,
            recommendations=recommendations,
            anomalies=anomalies,
            trends=trends
        )
        
        # Store internal recommendations separately for adapt() to use
        result.metadata = {"internal_recommendations_for_adapt": recommendations}
        
        return result

    async def react(self, context: NireonExecutionContext) -> List[SystemSignal]:
        """
        React to system signals, primarily EmbeddingComputedSignal and TrustAssessmentSignal.
        """
        context.logger.debug(f"[{self.component_id}] react() called. Current signal in context: {type(context.signal).__name__ if context.signal else 'None'}")
        
        # Handle TrustAssessmentSignal
        if isinstance(context.signal, TrustAssessmentSignal):
            assessment_payload = context.signal.payload
            idea_id = assessment_payload.get('idea_id')
            frame_id = context.signal.context_tags.get('frame_id')
            trust_score = assessment_payload.get('trust_score')

            if frame_id and idea_id and trust_score is not None:
                # Store the assessment in the frame's audit trail or a dedicated tag
                try:
                    frame = await self.frame_factory.get_frame_by_id(context, frame_id)
                    if frame:
                        if 'variation_assessments' not in frame.context_tags:
                            frame.context_tags['variation_assessments'] = {}
                        frame.context_tags['variation_assessments'][idea_id] = trust_score
                        self._add_audit_log(frame, 'ASSESSMENT_RECEIVED', f'Trust score {trust_score:.2f} for {idea_id}.')

                        # Trigger the event for the waiting exploration task
                        if frame_id in self._frame_assessment_trackers and idea_id in self._frame_assessment_trackers[frame_id]:
                            self._frame_assessment_trackers[frame_id][idea_id].set()
                            context.logger.info(f"[{self.component_id}] React: Set assessment event for {idea_id} in frame {frame_id}.")
                except FrameNotFoundError:
                    context.logger.warning(f"[{self.component_id}] React: Frame {frame_id} not found for assessment of {idea_id}.")
        
        emitted_signals: List[SystemSignal] = []

        # Handle EmbeddingComputedSignal
        if context.signal and getattr(context.signal, 'event_type', None) == "EmbeddingComputedSignal":
            signal_payload = getattr(context.signal, 'payload', {})
            request_id = signal_payload.get("request_id")

            if request_id and request_id in self._pending_embedding_requests:
                pending_req_data = self._pending_embedding_requests.pop(request_id)
                idea_id = pending_req_data["idea_id"]
                frame_id_for_idea = pending_req_data["frame_id"]
                
                # Try to get frame for audit logging
                frame_for_update = None
                try:
                    frame_for_update = await self.frame_factory.get_frame(context, frame_id_for_idea)
                except Exception as e:
                    context.logger.error(f"Frame {frame_id_for_idea} not found for embedding correlation of idea {idea_id}: {e}")

                if signal_payload.get("error_message"):
                    error_msg = signal_payload["error_message"]
                    context.logger.warning(
                        f"[{self.component_id}] Embedding computation failed for idea '{idea_id}' (request_id: {request_id}): {error_msg}"
                    )
                    if frame_for_update:
                        self._add_audit_log(frame_for_update, "EMBEDDING_COMPUTATION_FAILED", 
                                            f"Embedding failed for idea {idea_id}.", 
                                            {"target_idea_id": idea_id, "embedding_request_id": request_id, "error": error_msg})
                else:
                    embedding_vector = signal_payload.get("embedding_vector")
                    dims = signal_payload.get("embedding_dimensions")
                    context.logger.info(
                        f"[{self.component_id}] Received embedding for idea '{idea_id}' (request_id: {request_id}), dims: {dims}."
                    )
                    if frame_for_update:
                        self._add_audit_log(frame_for_update, "EMBEDDING_COMPUTED", 
                                            f"Embedding received for idea {idea_id}.", 
                                            {"target_idea_id": idea_id, "embedding_request_id": request_id, "dimensions": dims})
                        
                # Update frame data if we made changes
                if frame_for_update:
                    try:
                        frame_for_update.updated_ts = time.time()
                        context.logger.debug(f"Frame '{frame_for_update.id}' context_tags (audit log) updated in-memory after react.")
                    except Exception as e:
                        context.logger.error(f"Failed to update frame data for embedding event: {e}")

            elif request_id:
                context.logger.debug(f"[{self.component_id}] Received EmbeddingComputedSignal for unknown/timed-out request_id: {request_id}")

        # Check for timed-out embedding requests
        MAX_REQUEST_AGE = 300
        now = time.time()
        timed_out_requests_ids: List[str] = []
        aged_out_ids = []

        for req_id, req_data in self._pending_embedding_requests.items():
            request_age = now - req_data['timestamp']
            if request_age > MAX_REQUEST_AGE:
                aged_out_ids.append(req_id)

        # Clean up aged-out requests
        for req_id in aged_out_ids:
            req_data = self._pending_embedding_requests.pop(req_id, None)
            if req_data:
                context.logger.warning(
                    f"[{self.component_id}] Embedding request '{req_id}' aged out after {MAX_REQUEST_AGE}s"
                )
                

        for req_id in timed_out_requests_ids:
            if req_id in self._pending_embedding_requests:
                pending_req_data = self._pending_embedding_requests.pop(req_id)
                idea_id = pending_req_data["idea_id"]
                frame_id_for_idea = pending_req_data["frame_id"]
                context.logger.warning(
                    f"[{self.component_id}] Embedding request '{req_id}' for idea '{idea_id}' in frame '{frame_id_for_idea}' timed out."
                )

                # Try to get frame for audit logging
                frame_for_timeout_log = None
                try:
                    frame_for_timeout_log = await self.frame_factory.get_frame(context, frame_id_for_idea)
                    if frame_for_timeout_log:
                        self._add_audit_log(frame_for_timeout_log, "EMBEDDING_REQUEST_TIMEOUT",
                                            f"Embedding request for idea {idea_id} timed out.",
                                            {"target_idea_id": idea_id, "embedding_request_id": req_id})
                        frame_for_timeout_log.updated_ts = time.time()
                        context.logger.debug(f"Frame '{frame_for_timeout_log.id}' context_tags updated in-memory for timeout.")
                except Exception as e:
                    context.logger.error(f"Failed to update frame for timeout: {e}")

                # Emit a ProblemSignal as a SystemSignal
                problem_signal = SystemSignal(
                    signal_type=SignalType.WARNING,
                    component_id=self.component_id,
                    message=f"Embedding for idea '{idea_id}' timed out.",
                    payload={
                        "problem_type": "MissingEmbedding",
                        "component_id": self.component_id,
                        "details": f"Embedding for idea '{idea_id}' (request_id: {req_id}) in frame '{frame_id_for_idea}' was not received within timeout.",
                        "severity": "warning",
                        "related_artifact_id": idea_id,
                        "frame_id": frame_id_for_idea
                    }
                )
                emitted_signals.append(problem_signal)

        return emitted_signals

    async def adapt(self, context: NireonExecutionContext) -> List[AdaptationAction]:
        """
        Propose adaptations to own configuration based on analysis.
        """
        context.logger.debug(f"[{self.component_id}] adapt() called.")
        proposed_actions: List[AdaptationAction] = []

        # Call analyze to get current state and recommendations
        analysis_result = await self.analyze(context)
        internal_recommendations = analysis_result.metadata.get("internal_recommendations_for_adapt", [])

        if "CONSIDER_INCREASING_DIVERGENCE_OR_REVISING_PROMPTS" in internal_recommendations:
            # Propose increasing divergence_strength
            current_strength = self.cfg.divergence_strength
            if current_strength < 0.8:
                new_strength = min(round(current_strength * 1.2, 3), 0.8)
                action = AdaptationAction(
                    action_type=AdaptationActionType.CONFIG_UPDATE,
                    component_id=self.component_id,
                    description=(
                        f"Propose increasing default divergence_strength from {current_strength} to {new_strength} "
                        "due to consistently low average idea variations per frame."
                    ),
                    parameters={
                        "config_key": "divergence_strength",
                        "new_value": new_strength,
                        "old_value": current_strength
                    },
                    priority=2,
                    estimated_impact="medium",
                    requires_approval=True
                )
                proposed_actions.append(action)
                context.logger.info(f"[{self.component_id}] Proposing adaptation: Increase divergence_strength to {new_strength}.")

        if "REVIEW_LLM_CONFIGURATION" in internal_recommendations:
            # Propose adjusting LLM temperature if success rate is low
            current_temp = self.cfg.default_llm_policy_for_exploration.get("temperature", 0.7)
            if current_temp > 0.5:
                new_temp = round(current_temp * 0.9, 2)  # Reduce by 10%
                action = AdaptationAction(
                    action_type=AdaptationActionType.PARAMETER_ADJUST,
                    component_id=self.component_id,
                    description=(
                        f"Propose reducing LLM temperature from {current_temp} to {new_temp} "
                        "to improve response reliability."
                    ),
                    parameters={
                        "parameter_path": "default_llm_policy_for_exploration.temperature",
                        "new_value": new_temp,
                        "old_value": current_temp
                    },
                    priority=3,
                    estimated_impact="low",
                    requires_approval=True
                )
                proposed_actions.append(action)
                context.logger.info(f"[{self.component_id}] Proposing adaptation: Reduce LLM temperature to {new_temp}.")

        if "OPTIMIZE_PROMPT_TEMPLATES" in internal_recommendations:
            # Propose enabling more detailed prompts
            action = AdaptationAction(
                action_type=AdaptationActionType.BEHAVIOR_CHANGE,
                component_id=self.component_id,
                description="Propose switching to more detailed prompt templates to improve variation quality.",
                parameters={
                    "behavior": "prompt_template_strategy",
                    "change": "use_detailed_templates"
                },
                priority=2,
                estimated_impact="medium",
                requires_approval=True
            )
            proposed_actions.append(action)

        if "INVESTIGATE_FRAME_ERRORS" in internal_recommendations:
            # Propose temporary increased logging for diagnostics
            action = AdaptationAction(
                action_type=AdaptationActionType.PARAMETER_ADJUST,
                component_id=self.component_id,
                description="Temporarily increase logging verbosity due to high frame error rate.",
                parameters={
                    "parameter_name": "log_level",
                    "new_value": "DEBUG",
                    "duration_seconds": 3600  # 1 hour
                },
                priority=1,
                estimated_impact="low",
                requires_approval=False  # Logging changes might not need approval
            )
            proposed_actions.append(action)
            context.logger.warning(f"[{self.component_id}] High frame error rate detected. Proposing diagnostic logging increase.")

        # Check for resource constraints
        if analysis_result.metrics.get('error_completion_rate_recent', 0) > 0.1:
            # Check if errors are budget-related
            budget_errors = sum(1 for stat in self.last_n_frame_stats if stat.get("status") == "error_budget")
            if budget_errors > len(self.last_n_frame_stats) * 0.1:
                action = AdaptationAction(
                    action_type=AdaptationActionType.RESOURCE_ALLOCATION,
                    component_id=self.component_id,
                    description="Request increased LLM budget allocation due to frequent budget-related failures.",
                    parameters={
                        "resource_type": "llm_calls",
                        "current_budget": self.cfg.default_resource_budget_for_exploration.get("llm_calls", 10),
                        "requested_budget": self.cfg.default_resource_budget_for_exploration.get("llm_calls", 10) * 1.5
                    },
                    priority=4,
                    estimated_impact="high",
                    requires_approval=True
                )
                proposed_actions.append(action)

        if not proposed_actions:
            context.logger.info(f"[{self.component_id}] No specific adaptations proposed based on current analysis.")
            
        return proposed_actions
    
    

    

    async def shutdown(self, context: NireonExecutionContext) -> None:
        """
        Handle cleanup of resources.
        """
        context.logger.info(f"ExplorerMechanism '{self.component_id}' shutting down.")
        
        # Cancel any pending embedding requests
        if self._pending_embedding_requests:
            context.logger.info(
                f"[{self.component_id}] Cancelling {len(self._pending_embedding_requests)} pending embedding requests."
            )
            self._pending_embedding_requests.clear()
        
        # Clear assessment trackers
        if self._frame_assessment_trackers:
            context.logger.info(
                f"[{self.component_id}] Clearing assessment trackers for {len(self._frame_assessment_trackers)} frames."
            )
            self._frame_assessment_trackers.clear()
        
        await super().shutdown(context)
        context.logger.info(f"ExplorerMechanism '{self.component_id}' shutdown complete.")

    async def health_check(self, context: NireonExecutionContext) -> ComponentHealth:
        """
        Perform a health check for the Explorer mechanism.
        """
        context.logger.debug(f"ExplorerMechanism '{self.component_id}' health_check() called.")
        
        # Check service availability
        service_health = self.validate_required_services(['gateway', 'frame_factory', 'embed'], context)
        
        status = 'HEALTHY'
        messages = []
        details = {
            'explorations_performed': self._exploration_count,
            'configured_strategy': self.cfg.exploration_strategy,
            'pending_embedding_requests': len(self._pending_embedding_requests),
            'services_available': service_health,
            'semantic_exploration_enabled': self.cfg.enable_semantic_exploration,
            'sentinel_feedback_enabled': self.cfg.enable_sentinel_feedback
        }

        if not self.is_initialized:
            status = 'UNHEALTHY'
            messages.append('Explorer not initialized.')
        else:
            messages.append('Explorer initialized.')

            if not service_health:
                status = 'UNHEALTHY'
                messages.append('Required services not available.')
                # Check specific services
                if not self.gateway:
                    details['gateway_status'] = 'MISSING'
                if not self.frame_factory:
                    details['frame_factory_status'] = 'MISSING'
                if not self.embed:
                    details['embed_status'] = 'MISSING'
            else:
                messages.append('All required services available.')

        if self.error_count > 0:
            status = 'DEGRADED' if status == 'HEALTHY' else status
            messages.append(f'Explorer has encountered {self.error_count} errors during its lifetime.')
            details['error_count'] = self.error_count

        # Check if event helper is available
        if not self.event_helper:
            status = 'DEGRADED' if status == 'HEALTHY' else status
            messages.append('EventHelper not initialized.')
            details['event_helper_status'] = 'MISSING'

        final_message = '; '.join(messages) if messages else 'Explorer operational.'
        return ComponentHealth(
            component_id=self.component_id,
            status=status,
            message=final_message,
            details=details
        )