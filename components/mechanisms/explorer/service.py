# nireon/components/mechanisms/explorer/service.py
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
from collections import deque
from domain.context import NireonExecutionContext
from domain.cognitive_events import CognitiveEvent, LLMRequestPayload
from domain.epistemic_stage import EpistemicStage
from domain.ports.mechanism_gateway_port import MechanismGatewayPort
from domain.ports.llm_port import LLMResponse
from application.services.frame_factory_service import FrameFactoryService, FrameNotFoundError
from domain.frames import Frame

from .config import ExplorerConfig
from .service_helpers.explorer_event_helper import ExplorerEventHelper
from .errors import ExplorerErrorCode

logger = logging.getLogger(__name__)

EXPLORER_METADATA = ComponentMetadata(
    id='explorer',
    name='Explorer Mechanism V4',
    version='1.6.0',
    category='mechanism',
    description='Explorer Mechanism for idea generation and systematic variation, using A-F-CE model (Frames and MechanismGateway).',
    epistemic_tags=['generator', 'variation', 'mutator', 'innovator', 'divergence', 'novelty_generation'],
    capabilities={'generate_ideas', 'explore_variations', 'idea_mutation', 'dynamic_frame_parameterization'},
    accepts=['SEED_SIGNAL', 'EXPLORATION_REQUEST'],
    produces=['IdeaGeneratedSignal', 'ExplorationCompleteSignal', 'FrameProcessingFailedSignal'],
    requires_initialize=True,
    dependencies={
        'MechanismGatewayPort': '>=1.0.0',
        'FrameFactoryService': '*'
    }
)

class ExplorerMechanism(NireonBaseComponent):
    ConfigModel = ExplorerConfig

    def __init__(self,
                 config: Dict[str, Any],
                 metadata_definition: ComponentMetadata,
                 gateway: Optional[MechanismGatewayPort] = None,
                 frame_factory: Optional[FrameFactoryService] = None):
        super().__init__(config=config, metadata_definition=metadata_definition)
        self.cfg: ExplorerConfig = ExplorerConfig(**self.config)

        # Dependencies to be resolved in _initialize_impl
        self.gateway: Optional[MechanismGatewayPort] = gateway
        self.frame_factory: Optional[FrameFactoryService] = frame_factory
        self.event_helper: Optional[ExplorerEventHelper] = None

        self._exploration_count = 0
        self._rng_frame_specific: Optional[random.Random] = None
        self._pending_embedding_requests: Dict[str, Any] = {}
        
        # Store stats for the last N frames for analysis
        self.last_n_frame_stats = deque(maxlen=10)

        logger.info(f"ExplorerMechanism '{self.component_id}' (instance of {self.metadata.name} "
                    f"v{self.metadata.version}) created. Gateway and FrameFactory will be resolved during initialization.")
        logger.debug(f"Explorer initial config: {self.cfg.model_dump_json(indent=2)}")

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

        # Initialize ExplorerEventHelper once gateway is resolved
        self.event_helper = ExplorerEventHelper(self.gateway, self.component_id, self.metadata.version)
        context.logger.info(f"Initialized ExplorerEventHelper for '{self.component_id}'.")

        self._init_exploration_strategies(context)
        context.logger.info(f"✓ ExplorerMechanism '{self.component_id}' initialized successfully.")

    def _init_exploration_strategies(self, context: NireonExecutionContext) -> None:
        strategy = self.cfg.exploration_strategy
        context.logger.debug(f"Explorer '{self.component_id}' configured with strategy: {strategy}")

    def _build_llm_prompt(self, seed_text: str, objective: str, attempt_details: Optional[Dict[str, Any]] = None) -> str:
        """
        Builds a prompt for the LLM based on the configured template or a default.
        """
        vector_distance_placeholder = attempt_details.get("vector_distance", 0.0) if attempt_details else 0.0
        
        template_vars = {
            "seed_idea_text": seed_text,
            "objective": objective or "explore novel variations",
            "vector_distance": vector_distance_placeholder,
            "creativity_factor_desc": "high" if self.cfg.creativity_factor > 0.7 else "medium" if self.cfg.creativity_factor > 0.3 else "low",
            "desired_length_min": self.cfg.minimum_idea_length,
            "desired_length_max": self.cfg.maximum_idea_length,
        }
        
        prompt_template_to_use = self.cfg.default_prompt_template or \
            ("Generate a creative and divergent variation of the following idea: '{seed_idea_text}'. "
             "The overall objective is: {objective}. "
             "Aim for a {creativity_factor_desc} degree of novelty. "
             "The generated idea should be between {desired_length_min} and {desired_length_max} characters. "
             "Respond with ONLY the full text of the new, varied idea.")

        try:
            return prompt_template_to_use.format(**template_vars)
        except KeyError as e:
            logger.warning(f"[{self.component_id}] Prompt template missing key: {e}. Using basic prompt.")
            return f"Generate a creative variation of: {seed_text}. Objective: {objective}."

    def _add_audit_log(self, frame: Optional[Frame], event_type: str, summary: str, details: Optional[Dict[str, Any]] = None) -> None:
        """Helper to add structured logs to the frame's audit_trail."""
        if not frame or not hasattr(frame, 'context_tags') or not isinstance(frame.context_tags.get("audit_trail"), list):
            logger.warning(f"[{self.component_id}] Cannot add audit log: Frame or audit_trail invalid. Event: {event_type}")
            return

        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "summary": summary
        }
        if details:
            entry["details"] = details
        
        audit_trail: List[Dict] = frame.context_tags["audit_trail"]
        
        # DESIGN_AFCE.md Sec 3.1: Size Management & Spill Policy
        MAX_AUDIT_ENTRIES_IN_FRAME = 50
        MAX_AUDIT_STRING_LENGTH = 1024 * 10  # 10KB
        
        current_audit_size = sum(len(json.dumps(item)) for item in audit_trail)

        if len(audit_trail) >= MAX_AUDIT_ENTRIES_IN_FRAME or current_audit_size > MAX_AUDIT_STRING_LENGTH:
            logger.warning(
                f"Audit trail for frame {frame.id} near/exceeds limit ({len(audit_trail)} entries, {current_audit_size} bytes). "
                f"Logging event '{event_type}' to standard logger instead of frame. "
                f"Summary: {summary}. Details: {details}. "
                f"TODO (Platform): Implement MechanismGateway.archive_frame_artifact for spill."
            )
            logger.info(f"[FrameAuditSpill][{frame.id}] Event: {event_type}, Summary: {summary}, Details: {details}")
        else:
            audit_trail.append(entry)

    async def _request_embedding_for_idea(self, frame: Frame, idea_text: str, idea_id: str, context: NireonExecutionContext) -> None:
        """
        Emits an EmbeddingRequestSignal for a given idea.
        """
        if not self.event_helper:
            context.logger.error(f"[{self.component_id}] Event helper not available. Cannot request embedding for idea {idea_id}.")
            return

        if len(self._pending_embedding_requests) >= self.cfg.max_pending_embedding_requests:
            context.logger.warning(
                f"[{self.component_id}] Max pending embedding requests ({self.cfg.max_pending_embedding_requests}) reached. "
                f"Skipping embedding request for idea {idea_id} in frame {frame.id}."
            )
            self._add_audit_log(frame, "EMBEDDING_REQUEST_SKIPPED", 
                                f"Max pending requests reached, skipped for idea {idea_id}.",
                                {"target_idea_id": idea_id})
            return

        request_id = f"emb_req_{self.component_id}_{shortuuid.uuid()}"
        
        embedding_request_payload = {
            "request_id": request_id,
            "text_to_embed": idea_text,
            "target_artifact_id": idea_id,
            "request_timestamp_ms": int(time.time() * 1000),
            "embedding_vector_dtype": "float32",
            "metadata": {
                **self.cfg.embedding_request_metadata,
                "frame_id": frame.id,
                "origin_component_id": self.component_id
            }
        }

        self._pending_embedding_requests[request_id] = {
            "idea_id": idea_id,
            "frame_id": frame.id,
            "text": idea_text,
            "timestamp": time.time()
        }
        
        self._add_audit_log(frame, "EMBEDDING_REQUESTED", 
                            f"Embedding requested for idea {idea_id}.",
                            {"target_idea_id": idea_id, "embedding_request_id": request_id})

        await self.event_helper.publish_signal(
            frame_id=frame.id,
            signal_type_name="EmbeddingRequestSignal",
            signal_payload=embedding_request_payload,
            context=context,  # Pass context
            epistemic_intent="REQUEST_EMBEDDING_COMPUTATION"
        )
        context.logger.info(f"[{self.component_id}] Requested embedding for idea '{idea_id}' (request_id: {request_id}) in frame '{frame.id}'.")

    async def _process_impl(self, data: Any, context: NireonExecutionContext) -> ProcessResult:
        """
        Core processing logic for ExplorerMechanism using the A-F-CE model.
        1. Creates an ExplorationFrame.
        2. Executes exploration strategy (LLM calls via Gateway).
        3. Publishes results (Signals via Gateway).
        4. Closes the Frame.
        """
        self._exploration_count += 1
        task_short_id = shortuuid.uuid()[:8]
        
        # Extract seed information
        seed_input_text = ""
        seed_input_id = f"seed_{task_short_id}"
        if isinstance(data, str):
            seed_input_text = data
        elif isinstance(data, dict) and "text" in data:
            seed_input_text = data["text"]
            seed_input_id = data.get("id", seed_input_id)
        elif isinstance(data, dict) and "seed_idea" in data:
            seed_input_text = data["seed_idea"].get("text", f"default_seed_for_task_{task_short_id}")
            seed_input_id = data["seed_idea"].get("id", seed_input_id)
        else:
            seed_input_text = f"default_seed_for_task_{task_short_id}"
            context.logger.warning(f"Unrecognized input data structure for Explorer. Using default seed: {seed_input_text}")

        input_hash = hashlib.sha256(seed_input_text.encode()).hexdigest()[:12]
        objective_from_data = data.get("objective", "Generate novel idea variations.") if isinstance(data, dict) else "Generate novel idea variations."

        frame_name = f"explorer_task_{task_short_id}_on_seed_{input_hash}"
        frame_description = (
            f"Exploration task initiated by '{self.component_id}' (exploration #{self._exploration_count}) "
            f"for seed idea (hash: {input_hash}, preview: '{seed_input_text[:30]}...') "
            f"using strategy '{self.cfg.exploration_strategy}'."
        )

        # Resolve parent_frame_id
        parent_frame_id: Optional[str] = context.metadata.get('current_frame_id') if context.metadata else None
        if not parent_frame_id:
            context.logger.warning(
                f"parent_frame_id not found for Explorer task {task_short_id}. Creating top-level frame."
            )
        
        # Frame-specific RNG
        frame_rng: Optional[random.Random] = None

        # Initialize audit_trail and context_tags
        audit_trail_entries: List[Dict[str, Any]] = [{"ts": datetime.now(timezone.utc).isoformat(), "event_type": "EXPLORER_TASK_STARTED", "summary": "Explorer task processing initiated."}]
        context_tags = {
            "explorer_version": self.metadata.version,
            "seed_idea_id": seed_input_id,
            "exploration_strategy_used": self.cfg.exploration_strategy,
            "initial_input_hash": input_hash,
            "adaptive_parameters_log": [],
            "rng_seed_used": "PENDING_FRAME_RNG_IMPL",
            "audit_trail": audit_trail_entries
        }
        if self.gateway and hasattr(self.gateway, 'metadata'):
             context_tags["gateway_info"] = {
                "gateway_id": self.gateway.component_id,
                "gateway_version": self.gateway.metadata.version,
                "gateway_type": self.gateway.metadata.category
            }
        else:
            context_tags["gateway_info"] = {"gateway_id": "unknown", "gateway_version": "unknown", "gateway_type": "unknown"}

        current_frame: Optional[Frame] = None
        frame_final_status: str = "pending"  # Start with neutral status
        
        try:
            context.logger.info(f"Requesting ExplorationFrame: Name='{frame_name}'")
            current_frame = await self.frame_factory.create_frame(
                context=context,
                name=frame_name,
                description=frame_description,
                owner_agent_id=self.component_id,
                parent_frame_id=parent_frame_id,
                epistemic_goals=["DIVERGENCE", "NOVELTY_GENERATION", "IDEA_VARIATION"],
                trust_basis={"seed_idea_trust": 0.75},
                llm_policy=self.cfg.default_llm_policy_for_exploration,
                resource_budget=self.cfg.default_resource_budget_for_exploration,
                context_tags=context_tags,
                initial_status="active"
            )
            context.logger.info(f"Successfully created ExplorationFrame ID: {current_frame.id}, Name: {current_frame.name}")
            self._add_audit_log(current_frame, "FRAME_CREATED", f"Frame {current_frame.id} created successfully.")

            # Attempt to get frame-specific RNG
            if hasattr(current_frame, 'get_rng') and callable(current_frame.get_rng):
                frame_rng = current_frame.get_rng()
                context_tags["rng_seed_used"] = f"frame_rng_obtained_details_unavailable"
                # Fixed: Direct modification instead of update_frame_data
                if current_frame:
                    current_frame.context_tags = context_tags  # Update the in-memory frame object
                    current_frame.updated_ts = time.time()
                    context.logger.debug(f"Frame '{current_frame.id}' context_tags updated in-memory.")
                context.logger.info(f"Using frame-specific RNG for Frame {current_frame.id}")
            else:
                context.logger.warning(f"Frame object does not have get_rng(). Using component's default RNG. FrameID: {current_frame.id}.")
                seed_val = (context.replay_seed if context.replay_mode and context.replay_seed is not None else int(time.time()*1000)) + hash(current_frame.id)
                frame_rng = random.Random(seed_val)
                context_tags["rng_seed_used"] = f"fallback_rng_seeded_with_{seed_val}"
                # Fixed: Direct modification instead of update_frame_data
                if current_frame:
                    current_frame.context_tags = context_tags  # Update the in-memory frame object
                    current_frame.updated_ts = time.time()
                    context.logger.debug(f"Frame '{current_frame.id}' context_tags updated in-memory.")

            # --- Core Exploration Logic (LLM Calls) ---
            context.logger.info(f"[{self.component_id}] Starting exploration logic within Frame '{current_frame.id}'...")
            self._add_audit_log(current_frame, "EXPLORATION_STARTED", "Core exploration logic initiated.")
            
            generated_variations_texts: List[str] = []
            llm_tasks = []
            num_variations_to_generate = self.cfg.max_variations_per_level

            for i in range(num_variations_to_generate):
                prompt = self._build_llm_prompt(seed_input_text, objective_from_data, {"attempt_index": i})
                request_llm_settings = {**self.cfg.default_llm_policy_for_exploration, **(current_frame.llm_policy or {})}
                
                if not isinstance(request_llm_settings, dict):
                    logger.warning(f"LLM settings is not a dict: {type(request_llm_settings)}, using default")
                    request_llm_settings = {
                        'temperature': 0.8,
                        'max_tokens': 1500,
                        'model': 'gpt-4o-mini'
                    }

                llm_payload = LLMRequestPayload(
                    prompt=prompt,
                    stage=EpistemicStage.EXPLORATION,
                    role="idea_generator",
                    llm_settings=request_llm_settings,
                )
                
                ce_llm_request = CognitiveEvent(
                    frame_id=current_frame.id,
                    owning_agent_id=self.component_id,
                    service_call_type='LLM_ASK',
                    payload=llm_payload,
                    epistemic_intent="GENERATE_IDEA_VARIATION",
                    custom_metadata={
                        "explorer_attempt": i + 1,
                        "seed_hash": input_hash,
                        "schema_version": 1
                    },
                )
                llm_tasks.append(self.gateway.process_cognitive_event(ce_llm_request, context))
                self._add_audit_log(current_frame, "LLM_CE_CREATED", f"LLM_ASK CE created for attempt {i+1}.", {"ce_id_related": ce_llm_request.event_id})

            # Execute LLM calls concurrently with enhanced error handling
            # Fix for the original_ce_id issue in components/mechanisms/explorer/service.py
# Around lines 402-420, replace the problematic section with:

            llm_responses: List[Optional[LLMResponse]] = []
            budget_exceeded_hard = False
            try:
                raw_responses = await asyncio.gather(*llm_tasks, return_exceptions=True)
                for i, resp_or_exc in enumerate(raw_responses):
                    # Create a unique identifier for this LLM call
                    original_ce_id = f"{current_frame.id}_llm_{i+1}"
                    
                    if isinstance(resp_or_exc, Exception):
                        error_type_str = type(resp_or_exc).__name__
                        error_message_str = str(resp_or_exc)
                        context.logger.error(f"LLM call {i+1} (CE: {original_ce_id}) failed with exception. Type: {error_type_str}, Message: '{error_message_str}'")
                        self._add_audit_log(current_frame, "LLM_CALL_EXCEPTION", f"Attempt {i+1} raised exception.", 
                                            {"ce_id_related": original_ce_id, "error": error_message_str, "error_type": error_type_str})
                        llm_responses.append(None)
                        
                        if "budget" in error_message_str.lower():
                            frame_final_status = "error_budget"
                            self._add_audit_log(current_frame, "BUDGET_EXCEEDED_EXCEPTION", "Gateway/LLM raised budget-related error.", {"ce_id_related": original_ce_id})
                            budget_exceeded_hard = True
                            break
                            
                    elif isinstance(resp_or_exc, LLMResponse):
                        llm_responses.append(resp_or_exc)
                        if resp_or_exc.get("error"):
                            error_payload = resp_or_exc.get("error_payload", {})
                            error_code = error_payload.get("code", resp_or_exc.get("error_type", "LLM_RESPONSE_ERROR"))
                            error_message = error_payload.get("message", resp_or_exc.get("error", "Unknown LLM error in response"))
                            
                            context.logger.warning(f"LLM call {i+1} (CE: {original_ce_id}) returned an error: {error_code} - {error_message}")
                            self._add_audit_log(current_frame, "LLM_RESPONSE_ERROR", f"Attempt {i+1} error in LLM response.",
                                                {"ce_id_related": original_ce_id, "error_code": error_code, "llm_error": error_message})
                            
                            if "BUDGET_EXCEEDED" in error_code.upper():
                                frame_final_status = "error_budget"
                                if "HARD" in error_code.upper():
                                    context.logger.info(f"Hard budget limit reached for frame {current_frame.id}. Halting further LLM calls.")
                                    self._add_audit_log(current_frame, "BUDGET_EXCEEDED_HARD_STOP", "Halting LLM calls.", {"ce_id_related": original_ce_id})
                                    budget_exceeded_hard = True
                                    break
                        else:
                            self._add_audit_log(current_frame, "LLM_CALL_SUCCESS", f"Attempt {i+1} successful.", {"ce_id_related": original_ce_id})
                    else:
                        original_ce_id = f"{current_frame.id}_llm_{i+1}"
                        context.logger.error(f"LLM call {i+1} (CE: {original_ce_id}) returned unexpected type: {type(resp_or_exc)}")
                        self._add_audit_log(current_frame, "LLM_UNEXPECTED_RESPONSE", f"Attempt {i+1} unexpected response type.", 
                                            {"ce_id_related": original_ce_id, "response_type": str(type(resp_or_exc))})
                        llm_responses.append(None)


            except Exception as gather_exc:
                context.logger.error(f"Error during asyncio.gather for LLM calls: {gather_exc}", exc_info=True)
                llm_responses = [None] * len(llm_tasks)
                self._add_audit_log(current_frame, "LLM_BATCH_ERROR", f"asyncio.gather failed: {str(gather_exc)}")
                if frame_final_status not in ["error_budget"]:
                    frame_final_status = "error_internal"

            # Process responses and publish signals
            if self.event_helper and not budget_exceeded_hard:
                for i, llm_response in enumerate(llm_responses):
                    if llm_response and llm_response.text and not llm_response.get("error"):
                        variation_text = llm_response.text.strip()
                        context.logger.info(f"[Explorer] LLM Variation {i+1} for Frame {current_frame.id}: {variation_text}") # temp logging line
                        if len(variation_text) >= self.cfg.minimum_idea_length:
                            generated_variations_texts.append(variation_text)
                            
                            # Create a unique ID for this new idea variation
                            new_idea_id = f"idea_{current_frame.id}_{shortuuid.uuid()[:8]}"
                            
                            idea_payload = {
                                "id": new_idea_id,
                                "text": variation_text, 
                                "source_mechanism": self.component_id, 
                                "derivation_method": self.cfg.exploration_strategy, 
                                "seed_idea_text": seed_input_text,
                                "seed_idea_text_preview": seed_input_text[:50],
                                "frame_id": current_frame.id,
                                "llm_response_metadata": {k:v for k,v in llm_response.items() if k != "text"}
                            }
                            await self.event_helper.publish_signal(
                                frame_id=current_frame.id,
                                signal_type_name="IdeaGeneratedSignal",
                                signal_payload=idea_payload,
                                context=context,  # Pass context
                                epistemic_intent="OUTPUT_GENERATED_IDEA"
                            )
                            self._add_audit_log(current_frame, "IDEA_GENERATED", f"Variation {i+1} generated and published.", 
                                              {"variation_length": len(variation_text), "variation_preview": variation_text[:50] + "...", "idea_id": new_idea_id})
                            
                            # Request embedding if configured
                            if self.cfg.request_embeddings_for_variations:
                                await self._request_embedding_for_idea(current_frame, variation_text, new_idea_id, context)
                        else:
                            context.logger.warning(f"LLM generated text too short (attempt {i+1}): '{variation_text[:50]}...'")
                            self._add_audit_log(current_frame, "LLM_RESPONSE_FILTERED", "Generated text too short.", 
                                              {"attempt": i+1, "reason": "too_short", "length": len(variation_text)})
                    else:
                        context.logger.warning(f"Skipping idea generation for LLM attempt {i+1} due to error or empty response.")
                        self._add_audit_log(current_frame, "LLM_RESPONSE_SKIPPED_ERROR", f"Skipped idea from LLM attempt {i+1} due to error.", {})
            elif not self.event_helper:
                context.logger.warning("ExplorerEventHelper not available. Skipping signal publication for generated ideas.")
                self._add_audit_log(current_frame, "EVENT_HELPER_MISSING", "Cannot publish signals without EventHelper.")

            # Update audit trail in frame data
            # Fixed: Direct modification instead of update_frame_data
            if current_frame:
                current_frame.context_tags = context_tags  # Update the in-memory frame object
                current_frame.updated_ts = time.time()
                context.logger.debug(f"Frame '{current_frame.id}' context_tags updated in-memory before final status update.")

            # Determine final status and message
            success_flag = False
            
            # Determine the outcome first
            if not generated_variations_texts and num_variations_to_generate > 0:
                frame_final_status = 'completed_degraded'
                message = f'Explorer task {task_short_id} completed with degradation in frame {current_frame.id}: No variations generated.'
                success_flag = True
                self._add_audit_log(current_frame, 'EXPLORATION_DEGRADED', 'No variations generated.')
            elif len(generated_variations_texts) < num_variations_to_generate:
                frame_final_status = 'completed_degraded'
                message = f'Explorer task {task_short_id} completed with degradation in frame {current_frame.id}: Generated {len(generated_variations_texts)}/{num_variations_to_generate} variations.'
                success_flag = True
                self._add_audit_log(current_frame, 'EXPLORATION_PARTIAL', f'Generated {len(generated_variations_texts)}/{num_variations_to_generate} variations.')
            else:
                frame_final_status = 'completed_ok'
                message = f'Explorer task {task_short_id} completed successfully in frame {current_frame.id}.'
                success_flag = True
                self._add_audit_log(current_frame, 'EXPLORATION_SUCCESS', f'Generated {len(generated_variations_texts)} variations.')
            
            # Only check for errors if the status is still 'pending' or starts with 'error_'
            if frame_final_status == 'pending' or frame_final_status.startswith('error_'):
                # This means something went wrong earlier in the process
                if frame_final_status == 'pending':
                    frame_final_status = 'error_internal'  # Fallback if somehow we're still pending
                message = f"Explorer task {task_short_id} failed in frame {current_frame.id} due to: {frame_final_status.split('_', 1)[1]}."
                success_flag = False

            # Publish ExplorationCompleteSignal
            exploration_complete_payload = {
                'exploration_id': f'exp_{self.component_id}_{self._exploration_count}_{task_short_id}',
                'seed_idea_text_preview': seed_input_text[:100],
                'seed_idea_id': seed_input_id,
                'variations_generated_count': len(generated_variations_texts),
                'exploration_strategy': self.cfg.exploration_strategy,
                'total_explorations_by_component': self._exploration_count,
                'frame_id': current_frame.id,
                'status': frame_final_status
            }
            
            if self.event_helper:
                await self.event_helper.publish_signal(
                    frame_id=current_frame.id,
                    signal_type_name="ExplorationCompleteSignal",
                    signal_payload=exploration_complete_payload,
                    context=context,  # Pass context
                    epistemic_intent="SIGNAL_TASK_COMPLETION"
                )
                self._add_audit_log(current_frame, "SIGNAL_PUBLISHED", "ExplorationCompleteSignal published.")
            else:
                context.logger.warning("ExplorerEventHelper not available. Skipping ExplorationCompleteSignal publication.")

            # Update frame status
            await self.frame_factory.update_frame_status(context, current_frame.id, frame_final_status)
            context.logger.info(f"ExplorationFrame '{current_frame.id}' final status: '{frame_final_status}'.")
            self._add_audit_log(current_frame, "FRAME_STATUS_UPDATED", f"Frame status set to {frame_final_status}.")
            
            # Store frame statistics for analysis
            frame_stats = {
                "frame_id": current_frame.id,
                "status": frame_final_status,
                "variations_generated": len(generated_variations_texts),
                "llm_calls_made": len(llm_tasks),
                "seed_input_text_preview": seed_input_text[:50],
                "strategy_used": self.cfg.exploration_strategy,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "target_variations": num_variations_to_generate,
                "llm_call_successes": sum(1 for r in llm_responses if r and not r.get("error")),
                "llm_call_failures": sum(1 for r in llm_responses if not r or r.get("error")),
            }
            self.last_n_frame_stats.append(frame_stats)
            self._add_audit_log(current_frame, "FRAME_STATS_RECORDED", "Frame completion statistics recorded internally for analysis.")
            
            # Persist final audit trail
            # Fixed: Direct modification instead of update_frame_data
            if current_frame:
                current_frame.context_tags = context_tags  # Update the in-memory frame object
                current_frame.updated_ts = time.time()
                context.logger.debug(f"Frame '{current_frame.id}' context_tags updated in-memory with final audit trail.")
            
            return ProcessResult(
                success=success_flag,
                component_id=self.component_id,
                output_data={"frame_id": current_frame.id, **exploration_complete_payload},
                message=message,
                metadata={"frame_id": current_frame.id},
                error_code=str(ExplorerErrorCode.LLM_BUDGET_EXCEEDED) if frame_final_status == "error_budget" else None
            )

        except FrameNotFoundError as fnfe:
            context.logger.error(f"Frame operation failed: {fnfe}", exc_info=True)
            return ProcessResult(
                success=False, 
                component_id=self.component_id, 
                message=f"Frame error: {fnfe}", 
                error_code=str(ExplorerErrorCode.FRAME_CREATION_FAILED)
            )
        except Exception as e:
            context.logger.error(f"Explorer processing failed critically: {e}", exc_info=True)
            current_frame_id_for_error = current_frame.id if current_frame else None
            
            if current_frame:
                final_error_status_for_frame = "error_internal"
                error_code_to_use = ExplorerErrorCode.UNKNOWN_ERROR
                
                # Determine specific error type
                if "budget" in str(e).lower():
                    final_error_status_for_frame = "error_budget"
                    error_code_to_use = ExplorerErrorCode.LLM_BUDGET_EXCEEDED
                elif "timeout" in str(e).lower():
                    final_error_status_for_frame = "error_timeout"
                    error_code_to_use = ExplorerErrorCode.LLM_TIMEOUT
                elif "gateway" in str(e).lower():
                    error_code_to_use = ExplorerErrorCode.GATEWAY_UNAVAILABLE
                
                self._add_audit_log(current_frame, "EXPLORER_TASK_CRITICAL_ERROR", f"Unhandled exception: {str(e)}", {"error_type": type(e).__name__})
                
                try:
                    # Fixed: Direct modification instead of update_frame_data
                    if current_frame:  # current_frame might be None if frame creation failed
                        current_frame.context_tags = context_tags
                        current_frame.updated_ts = time.time()
                        context.logger.debug(f"Frame '{current_frame.id}' context_tags updated in-memory during error handling.")
                    await self.frame_factory.update_frame_status(context, current_frame.id, final_error_status_for_frame)
                    context.logger.info(f"ExplorationFrame '{current_frame.id}' status updated to '{final_error_status_for_frame}' due to critical error.")
                except Exception as frame_e:
                    context.logger.error(f"Failed to update frame '{current_frame.id}' status during critical error handling: {frame_e}", exc_info=True)
            
                if self.event_helper:
                    try:
                        await self.event_helper.publish_signal(
                            frame_id=current_frame.id,
                            signal_type_name="FrameProcessingFailedSignal",
                            signal_payload={
                                "frame_id": current_frame.id, 
                                "agent_id": self.component_id, 
                                "error_message": str(e), 
                                "error_type": type(e).__name__,
                                "error_code": str(error_code_to_use)
                            },
                            context=context,  # Pass context
                            epistemic_intent="SIGNAL_FAILURE"
                        )
                    except Exception as sig_e:
                        context.logger.error(f"Failed to publish FrameProcessingFailedSignal for frame {current_frame_id_for_error}: {sig_e}", exc_info=True)

            return ProcessResult(
                success=False, 
                component_id=self.component_id, 
                message=f"Explorer critical error: {e}", 
                error_code=str(error_code_to_use), 
                metadata={"frame_id": current_frame_id_for_error}
            )

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
            if depth >= self.cfg.max_depth:
                continue

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
        
        variations = []
        for _ in range(num_to_generate):
            template = random.choice(variation_templates)
            variations.append(template.format(seed=seed_idea))
            
        context.logger.debug(f'Random exploration generated {len(variations)} variations')
        return variations

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
        recommendations = []  # Internal recommendations for adapt()
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
        React to system signals, primarily EmbeddingComputedSignal.
        """
        context.logger.debug(f"[{self.component_id}] react() called. Current signal in context: {type(context.signal).__name__ if context.signal else 'None'}")
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
                        # TODO: Store the embedding for future use (diversity calculations, etc.)
                        # Options: Update idea artifact, store in frame context, emit another signal
                        
                # Update frame data if we made changes
                if frame_for_update:
                    try:
                        # Fixed: Direct modification instead of update_frame_data
                        frame_for_update.updated_ts = time.time()
                        context.logger.debug(f"Frame '{frame_for_update.id}' context_tags (audit log) updated in-memory after react.")
                    except Exception as e:
                        context.logger.error(f"Failed to update frame data for embedding event: {e}")

            elif request_id:
                context.logger.debug(f"[{self.component_id}] Received EmbeddingComputedSignal for unknown/timed-out request_id: {request_id}")

        # Check for timed-out embedding requests
        now = time.time()
        timed_out_requests_ids: List[str] = []
        for req_id, req_data in self._pending_embedding_requests.items():
            if now - req_data['timestamp'] > self.cfg.embedding_response_timeout_s:
                timed_out_requests_ids.append(req_id)

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
                        # Fixed: Direct modification instead of update_frame_data
                        frame_for_timeout_log.updated_ts = time.time()
                        context.logger.debug(f"Frame '{frame_for_timeout_log.id}' context_tags updated in-memory for timeout.")
                except Exception as e:
                    context.logger.error(f"Failed to update frame for timeout: {e}")

                # Emit a ProblemSignal as a SystemSignal
                from core.results import SignalType
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
        await super().shutdown(context)
        context.logger.info(f"ExplorerMechanism '{self.component_id}' shutdown complete.")

    async def health_check(self, context: NireonExecutionContext) -> ComponentHealth:
        """
        Perform a health check for the Explorer mechanism.
        """
        context.logger.debug(f"ExplorerMechanism '{self.component_id}' health_check() called.")
        status = 'HEALTHY'
        messages = []
        details = {
            'explorations_performed': self._exploration_count,
            'configured_strategy': self.cfg.exploration_strategy,
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