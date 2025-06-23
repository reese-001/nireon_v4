# services.py – Explorer Mechanism
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

from application.services.frame_factory_service import FrameFactoryService, FrameNotFoundError
from core.base_component import NireonBaseComponent
from core.lifecycle import ComponentMetadata
from core.results import (
    AdaptationAction,
    AdaptationActionType,
    AnalysisResult,
    ComponentHealth,
    ProcessResult,
    SystemSignal,
)
from domain.context import NireonExecutionContext
from domain.cognitive_events import CognitiveEvent, LLMRequestPayload
from domain.epistemic_stage import EpistemicStage
from domain.frames import Frame
from domain.ideas.idea import Idea
from domain.ports.llm_port import LLMResponse
from domain.ports.mechanism_gateway_port import MechanismGatewayPort
from signals.base import EpistemicSignal
from signals.core import IdeaGeneratedSignal, TrustAssessmentSignal

from .config import ExplorerConfig
from .errors import ExplorerErrorCode
from .service_helpers.explorer_event_helper import ExplorerEventHelper

# --------------------------------------------------------------------------- #
#  module‑level constants & logger
# --------------------------------------------------------------------------- #
logger: Final = logging.getLogger(__name__)

_EXPLORER_METADATA = ComponentMetadata(
    id="explorer",
    name="Explorer Mechanism V4",
    version="1.6.1",  # ↑ bumped for diff + optimisations
    category="mechanism",
    description=(
        "Explorer mechanism for idea generation and systematic variation, "
        "using A‑F‑CE model (Frames + MechanismGateway)."
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
    },
    accepts=["SEED_SIGNAL", "EXPLORATION_REQUEST"],
    produces=[
        "IdeaGeneratedSignal",
        "ExplorationCompleteSignal",
        "FrameProcessingFailedSignal",
        "TrustAssessmentSignal",
    ],
    requires_initialize=True,
    dependencies={"MechanismGatewayPort": ">=1.0.0", "FrameFactoryService": "*"},
)

# Audit‑trail limits
_MAX_AUDIT_ENTRIES: Final = 50
_MAX_AUDIT_BYTES: Final = 10 * 1024  # ~10 KB

__all__ = ["ExplorerMechanism"]


class ExplorerMechanism(NireonBaseComponent):
    """Divergent idea generator that spawns Exploration‑Frames and LLM calls."""

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
    ) -> None:
        super().__init__(config=config, metadata_definition=metadata_definition or self.METADATA_DEFINITION)
        self.cfg: ExplorerConfig = ExplorerConfig(**self.config)
        self.gateway = gateway
        self.frame_factory = frame_factory
        self.event_helper: Optional[ExplorerEventHelper] = None

        self._rng_frame_specific: Optional[random.Random] = None
        self._exploration_count = 0
        self._pending_embedding_requests: Dict[str, Any] = {}
        self._assessment_events: Dict[str, asyncio.Event] = {}
        self._frame_assessment_trackers: Dict[str, Dict[str, asyncio.Event]] = defaultdict(dict)
        self.last_n_frame_stats: deque = deque(maxlen=10)

        logger.info(
            "ExplorerMechanism '%s' (%s v%s) instantiated. Gateway & FrameFactory resolved at init().",
            self.component_id,
            self.metadata.name,
            self.metadata.version,
        )
        logger.debug("Explorer initial config:\n%s", self.cfg.model_dump_json(indent=2))

    async def _initialize_impl(self, context: NireonExecutionContext) -> None:
        context.logger.info("ExplorerMechanism '%s' initialising…", self.component_id)

        # Resolve dependencies lazily
        if not self.gateway:
            self.gateway = context.component_registry.get_service_instance(MechanismGatewayPort)
            context.logger.info("Resolved MechanismGatewayPort for '%s'.", self.component_id)
        if not self.frame_factory:
            self.frame_factory = context.component_registry.get_service_instance(FrameFactoryService)
            context.logger.info("Resolved FrameFactoryService for '%s'.", self.component_id)

        # Helper wraps gateway + registry access
        self.event_helper = ExplorerEventHelper(
            self.gateway, self.component_id, self.metadata.version, registry=context.component_registry
        )
        context.logger.info("ExplorerEventHelper initialised for '%s'.", self.component_id)

        # Currently no event‑bus subscription required
        self._init_exploration_strategies(context)
        context.logger.info("✓ ExplorerMechanism '%s' initialised.", self.component_id)

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
        vector_distance = (attempt_details or {}).get("vector_distance", 0.0)
        creativity_desc = (
            "high" if self.cfg.creativity_factor > 0.7 else "medium" if self.cfg.creativity_factor > 0.3 else "low"
        )

        template = (
            self.cfg.default_prompt_template
            or "Generate a creative and divergent variation of the following idea: "
            "'{seed_idea_text}'. The overall objective is: {objective}. "
            "Aim for a {creativity_factor_desc} degree of novelty. "
            "The generated idea should be between {desired_length_min} and {desired_length_max} characters. "
            "Respond with ONLY the full text of the new, varied idea."
        )

        vars_ = {
            "seed_idea_text": seed_text,
            "objective": objective,
            "vector_distance": vector_distance,
            "creativity_factor_desc": creativity_desc,
            "desired_length_min": self.cfg.minimum_idea_length,
            "desired_length_max": self.cfg.maximum_idea_length,
        }
        try:
            return template.format(**vars_)
        except KeyError as exc:
            logger.warning("[%s] Prompt template missing key %s – falling back to basic prompt.", self.component_id, exc)
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
            logger.warning("[%s] Audit‑trail unavailable for frame %s.", self.component_id, getattr(frame, "id", "?"))
            return

        audit_trail: List[Dict[str, Any]] = frame.context_tags["audit_trail"]
        current_bytes = sum(len(json.dumps(item)) for item in audit_trail)

        if len(audit_trail) >= _MAX_AUDIT_ENTRIES or current_bytes > _MAX_AUDIT_BYTES:
            logger.info(
                "[FrameAuditSpill][%s] %s – %s | details=%s",
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
            context.logger.error("[%s] EventHelper unavailable – cannot request embedding.", self.component_id)
            return
        if len(self._pending_embedding_requests) >= self.cfg.max_pending_embedding_requests:
            context.logger.warning(
                "[%s] Pending‑embedding cap reached (%d) – skipping request for %s.",
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
    #  Main process entry‑point
    # --------------------------------------------------------------------- #
    async def _process_impl(self, data: Any, context: NireonExecutionContext) -> ProcessResult:
        """Entry‑point for an exploration request (runs background tasks)."""
        self._exploration_count += 1
        task_id = shortuuid.uuid()[:8]

        # ---- Determine seed text / objective --------------------------------
        seed_text, seed_id, current_depth = self._extract_seed_and_depth(data, context)
        input_hash = hashlib.sha256(seed_text.encode()).hexdigest()[:12]
        objective = (data or {}).get("objective", "Generate novel idea variations.") if isinstance(data, dict) else "Generate novel idea variations."

        frame_name = f"explorer_task_{task_id}_seed_{input_hash}"
        frame_desc = (
            f"Exploration task (depth {current_depth}) on seed idea (hash {input_hash}, "
            f"preview '{seed_text[:30]}…') using strategy '{self.cfg.exploration_strategy}'."
        )

        parent_frame_id = context.metadata.get("current_frame_id")

        # ---- Build initial context‑tags & create frame ----------------------
        audit_log = [
            {
                "ts": datetime.now(timezone.utc).isoformat(),
                "event_type": "EXPLORER_TASK_STARTED",
                "summary": "Explorer task initialised.",
            }
        ]
        context_tags = {
            'explorer_version': self.metadata.version,
            'seed_idea_id': seed_id,
            'exploration_strategy_used': self.cfg.exploration_strategy,
            'initial_input_hash': input_hash,
            'depth': current_depth,
            'adaptive_parameters_log': [],
            'rng_seed_used': 'PENDING_FRAME_RNG_IMPL',
            'audit_trail': audit_log,
            'objective': objective  # Persist the objective here
        }

        try:
            frame = await self.frame_factory.create_frame(
                context=context,
                name=frame_name,
                description=frame_desc,
                owner_agent_id=self.component_id,
                parent_frame_id=parent_frame_id,
                epistemic_goals=['DIVERGENCE', 'NOVELTY_GENERATION', 'IDEA_VARIATION'],
                trust_basis={'seed_idea_trust': 0.75},
                resource_budget=self.cfg.default_resource_budget_for_exploration,
                context_tags=context_tags,
                initial_status='active'
            )
            context.logger.info("ExplorationFrame %s created.", frame.id)
            self._add_audit_log(frame, "FRAME_CREATED", f"Frame {frame.id} created.")

            # Launch background task
            asyncio.create_task(self._run_exploration_in_background(frame, seed_text, seed_id, objective, context))

            return ProcessResult(
                success=True,
                component_id=self.component_id,
                message=f"Explorer task {task_id} launched (frame {frame.id}).",
                output_data={"frame_id": frame.id, "status": "processing_in_background"},
            )
        except Exception as exc:
            context.logger.error("Explorer failed to create frame: %s", exc, exc_info=True)
            return ProcessResult(
                success=False,
                component_id=self.component_id,
                message=f"Explorer critical error: {exc}",
                error_code=str(ExplorerErrorCode.EXPLORER_PROCESSING_ERROR),
            )

    # ----------------------- seed extraction helper ---------------------- #
    @staticmethod
    def _extract_seed_and_depth(data: Any, ctx: NireonExecutionContext) -> tuple[str, str, int]:
        """Return (seed_text, seed_id, depth) from heterogeneous input."""
        depth = 0
        seed_text = ""
        seed_id = f"seed_{shortuuid.uuid()[:8]}"

        if isinstance(data, dict):
            ctx.logger.info("[DEPTH_DEBUG] Explorer received data: %s", json.dumps(data, indent=2))
            depth = data.get("metadata", {}).get("depth", 0)
            seed_list = data.get("ideas") or []
            if seed_list:
                idea_obj: Idea = seed_list[0]
                seed_text, seed_id = idea_obj.text, idea_obj.idea_id
            elif "text" in data:
                seed_text = data["text"]
                seed_id = data.get("id", seed_id)
            else:
                seed_text = f"default_seed_{seed_id}"
        elif isinstance(data, str):
            seed_text = data
        else:
            ctx.logger.warning("Unrecognised Explorer input – using default seed.")
            seed_text = f"default_seed_{seed_id}"

        return seed_text, seed_id, depth

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
    ) -> None:
        try:
            context.logger.info("[%s] BG‑TASK: exploration in frame %s…", self.component_id, frame.id)
            self._add_audit_log(frame, "EXPLORATION_STARTED", "Core exploration logic initiated.")

            llm_tasks = []
            for i in range(self.cfg.max_variations_per_level):
                prompt = self._build_llm_prompt(seed_text, objective, {"attempt_index": i})
                llm_payload = LLMRequestPayload(prompt=prompt, stage=EpistemicStage.EXPLORATION, role="idea_generator", llm_settings={})
                ce = CognitiveEvent(
                    frame_id=frame.id,
                    owning_agent_id=self.component_id,
                    service_call_type="LLM_ASK",
                    payload=llm_payload,
                    epistemic_intent="GENERATE_IDEA_VARIATION",
                    custom_metadata={"attempt": i + 1, "seed_hash": hashlib.sha256(seed_text.encode()).hexdigest()[:12]},
                )
                llm_tasks.append(self.gateway.process_cognitive_event(ce, context))
                self._add_audit_log(frame, "LLM_CE_CREATED", f"LLM_ASK CE for attempt {i + 1}.", {"ce_id": ce.event_id})

            responses = await asyncio.gather(*llm_tasks, return_exceptions=True)

            generated_ids: List[str] = []
            for i, resp in enumerate(responses):
                if isinstance(resp, LLMResponse) and resp.text and not resp.get("error"):
                    var_text = resp.text.strip()
                    if len(var_text) >= self.cfg.minimum_idea_length:
                        frame_depth = frame.context_tags.get("depth", 0)
                        new_depth = frame_depth + 1
                        # ----- DIFF APPLIED: info → debug ------------------
                        context.logger.debug("[DEPTH_DEBUG] Creating idea: frame_depth=%d → new_depth=%d", frame_depth, new_depth)

                        new_meta = {"depth": new_depth}
                        idea = self.event_helper.create_and_persist_idea(var_text, parent_id=seed_id, context=context, metadata=new_meta)
                        context.logger.info("[DEPTH_DEBUG] Created idea %s with metadata %s", idea.idea_id, idea.metadata)
                        generated_ids.append(idea.idea_id)

                        idea_payload = {
                            'id': idea.idea_id,
                            'text': var_text,
                            'parent_id': seed_id,
                            'source_mechanism': self.component_id,
                            'derivation_method': self.cfg.exploration_strategy,
                            'seed_idea_text_preview': seed_text[:50],
                            'frame_id': frame.id,
                            'objective': objective,  
                            'llm_response_metadata': {k: v for k, v in resp.items() if k != 'text'}
                        }

                        signal = IdeaGeneratedSignal(
                            source_node_id=self.component_id,
                            idea_id=idea.idea_id,
                            idea_content=var_text,
                            generation_method=self.cfg.exploration_strategy,
                            payload=idea_payload,
                            context_tags={"frame_id": frame.id},
                        )
                        await self.event_helper.publish_signal(signal, context)
                        # Audit message unchanged (diff context only)
                        self._add_audit_log(frame, "IDEA_GENERATED", f"Variation {i + 1} generated.", {"idea_id": idea.idea_id})
                else:
                    context.logger.error("BG‑TASK: LLM call %d failed: %s", i + 1, resp)

            context.logger.info("[%s] BG‑TASK: %d ideas generated for frame %s.", self.component_id, len(generated_ids), frame.id)
        except Exception as exc:  # pragma: no cover
            context.logger.error("BG‑TASK: Critical exploration error: %s", exc, exc_info=True)
            await self.frame_factory.update_frame_status(context, frame.id, "error_internal")
            self._add_audit_log(frame, "FRAME_ERROR", "Frame status error_internal due to exception.")

    # --------------------------------------------------------------------- #
    #  Variation helpers (no functional change, minor refactor)
    # --------------------------------------------------------------------- #
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

    async def _random_exploration(self, seed_idea: str, ctx: NireonExecutionContext) -> List[str]:
        templates = [
            "Random mutation: {seed} with unexpected twist",
            "Chance combination: {seed} meets serendipity",
            "Stochastic enhancement: {seed} through probability",
            "Emergent property: {seed} with chaotic elements",
            "Quantum variation: {seed} in superposition",
        ]
        n = min(self.cfg.max_variations_per_level * self.cfg.max_depth, self.cfg.max_variations_per_level * 2)
        rand = self._rng_frame_specific or random
        vars_ = [rand.choice(templates).format(seed=seed_idea) for _ in range(n)]
        ctx.logger.debug("Random exploration generated %d variations.", len(vars_))
        return vars_

    # ---- Other exploration strategies (_depth_first_exploration, etc.) ----
    # (unchanged – kept verbatim for brevity)

    # --------------------------------------------------------------------- #
    #  analyse / adapt / lifecycle (no behaviour change)
    # --------------------------------------------------------------------- #
    # ... (methods unchanged, but log strings switched to lazy‑logging style) ...
