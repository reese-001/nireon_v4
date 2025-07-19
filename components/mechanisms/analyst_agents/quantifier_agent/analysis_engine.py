"""analysis_engine.py
======================
Core logic that turns an *idea* (free‑form text) into an actionable work order
for the **QuantifierAgent**.

Overview
--------
The :class:`QuantificationAnalysisEngine` orchestrates an LLM‑centric workflow:

1. **Viability gate** – quick, cheap prompt to decide if any quantitative work is
   appropriate at all (skipped when running in *single_call* mode).
2. **Comprehensive analysis** – richer prompt asking the LLM to choose between
   *quantification* and *formalisation* and to emit a structured YAML block.
3. **Parsing** – response is parsed via the generic :pymod:`components.common.llm_response_parser`.
4. **Post‑processing** – library extraction, Mermaid detection/generation, and
   construction of an :class:`AnalysisResult` returned to the caller.

Key improvements vs the previous revision
----------------------------------------
* Added full module & class‑level docstrings for maintainability.
* Switched :class:`AnalysisResult` to a `@dataclass` for clarity / defaults.
* Captures `llm_latency_ms` (if present on the :class:`LLMResponse`).
* Consistent logging prefixes and structured reasons on rejection.
* Mermaid helper methods are now *private* (prefixed with `_`).
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from domain.context import NireonExecutionContext
from domain.ports.llm_port import LLMResponse
from .prompts import QuantifierPrompts
from .config import QuantifierConfig
from components.common.llm_response_parser import (
    BooleanFieldExtractor,
    FieldSpec,
    NumericFieldExtractor,
    ParserFactory,
    TextFieldExtractor,
)

__all__: List[str] = [
    "AnalysisResult",
    "QuantificationAnalysisEngine",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DTO – analysis result
# ---------------------------------------------------------------------------
@dataclass
class AnalysisResult:
    """Outcome of the analysis step returned to *QuantifierAgent*."""

    viable: bool
    approach: str = ""
    implementation_request: str = ""
    libraries: List[str] = field(default_factory=list)
    use_mermaid: bool = False
    mermaid_content: str = ""
    confidence: float = 0.0
    analysis_type: str = "quantification"  # or "formalization"
    reasoning: str = ""


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------
class QuantificationAnalysisEngine:  # pylint: disable=too-many-instance-attributes
    """Run prompt+parse loop to transform an *idea* into a work order."""

    def __init__(self, config: QuantifierConfig) -> None:  # noqa: D401
        self.config = config
        self.prompts = QuantifierPrompts()

        # Build parsers (reuse factory helpers)
        self.viability_parser, self.viability_specs = ParserFactory.create_viability_parser()
        self.comprehensive_parser, self.comprehensive_specs = self._create_comprehensive_parser()

    # ------------------------------------------------------------------
    # Public entry‐point
    # ------------------------------------------------------------------
    async def analyze_idea(self, idea_text: str, gateway, context: NireonExecutionContext) -> Optional[AnalysisResult]:  # noqa: D401,E501,ANN001
        """Top‑level orchestration method."""
        if self.config.llm_approach == "single_call":
            return await self._single_call_analysis(idea_text, gateway, context)
        return await self._iterative_analysis(idea_text, gateway, context)

    # ------------------------------------------------------------------
    # Parser construction
    # ------------------------------------------------------------------
    def _create_comprehensive_parser(self) -> Tuple[Any, List[FieldSpec]]:  # noqa: ANN401
        from components.common.llm_response_parser import LLMResponseParser  # local import to avoid cycle

        parser = LLMResponseParser()
        specs: List[FieldSpec] = [
            FieldSpec("analysis_type", TextFieldExtractor("analysis_type"), default="quantification", required=True),
            FieldSpec("viable", BooleanFieldExtractor("viability"), default=False, required=True),
            FieldSpec("reasoning", TextFieldExtractor("reasoning", min_length=10), default="", required=False),
            FieldSpec("implementation", TextFieldExtractor("implementation_request", min_length=50, multiline=True), default="", required=False),
            FieldSpec("confidence", NumericFieldExtractor("confidence", min_val=0.0, max_val=1.0), default=0.5, required=False),
        ]
        return parser, specs

    # ------------------------------------------------------------------
    # Single‑call path
    # ------------------------------------------------------------------
    async def _single_call_analysis(self, idea_text: str, gateway, context: NireonExecutionContext) -> Optional[AnalysisResult]:  # noqa: ANN001,E501
        logger.info("[analysis] single‑call mode for idea: %.80s", idea_text)
        prompt = self.prompts.comprehensive_analysis(idea_text, self.config.available_libraries)
        response = await self._call_llm(gateway, prompt, "comprehensive_analyst", context)
        if not response:
            return None
        return await self._parse_comprehensive_response(response, idea_text, gateway, context)

    # ------------------------------------------------------------------
    # Iterative path (cheap viability gate first)
    # ------------------------------------------------------------------
    async def _iterative_analysis(self, idea_text: str, gateway, context: NireonExecutionContext) -> Optional[AnalysisResult]:  # noqa: ANN001,E501
        logger.info("[analysis] iterative mode for idea: %.80s", idea_text)

        # 1) quick gate
        v_prompt = self.prompts.viability_quick_check(idea_text)
        v_resp = await self._call_llm(gateway, v_prompt, "viability_checker", context)
        if not v_resp:
            return None

        v_result = self.viability_parser.parse(v_resp.text, self.viability_specs, "quantifier_agent")
        if not v_result.data["viable"]:
            logger.info("[analysis] quick‑gate deemed idea non‑viable")
            return AnalysisResult(False, confidence=v_result.data.get("confidence", 0.8))

        # 2) full analysis
        c_prompt = self.prompts.comprehensive_analysis(idea_text, self.config.available_libraries)
        c_resp = await self._call_llm(gateway, c_prompt, "detailed_analyst", context)
        if not c_resp:
            return None
        return await self._parse_comprehensive_response(c_resp, idea_text, gateway, context)

    # ------------------------------------------------------------------
    # Response parsing & post‑processing
    # ------------------------------------------------------------------
    async def _parse_comprehensive_response(self, llm_resp: LLMResponse, idea_text: str, gateway, context: NireonExecutionContext) -> Optional[AnalysisResult]:  # noqa: ANN001,E501
        resp_text = llm_resp.text
        result = self.comprehensive_parser.parse(resp_text, self.comprehensive_specs, "quantifier_agent")
        for warn in result.warnings:
            logger.warning("[analysis] parse warning: %s", warn)

        viable: bool = result.data.get("viable", False)
        confidence: float = result.data.get("confidence", 0.9 if result.is_success else 0.3)
        analysis_type: str = result.data.get("analysis_type", "quantification")
        reasoning: str = result.data.get("reasoning", "")

        # Non‑viable path --------------------------------------------------
        if not viable:
            logger.info("[analysis] idea not viable – %s", self._extract_rejection_reason(resp_text))
            return AnalysisResult(False, confidence=confidence, analysis_type=analysis_type, reasoning=reasoning)

        implementation = result.data["implementation"]

        # Formalisation path ----------------------------------------------
        if analysis_type == "formalization":
            return AnalysisResult(True, analysis_type=analysis_type, implementation_request=implementation, confidence=confidence, reasoning=reasoning)

        # Quantification path ---------------------------------------------
        approach = reasoning
        if self._should_use_mermaid(resp_text, approach) and self.config.enable_mermaid_output:
            mermaid = await self._generate_mermaid_diagram(idea_text, approach, gateway, context)
            if mermaid:
                proto_req = self._create_mermaid_proto_request(idea_text, mermaid)
                return AnalysisResult(True, approach, proto_req, use_mermaid=True, mermaid_content=mermaid, confidence=confidence, analysis_type=analysis_type, reasoning=reasoning)

        if not implementation or len(implementation) < self.config.min_request_length:
            logger.warning("[analysis] implementation request too short (%s chars)", len(implementation))
            return AnalysisResult(False, confidence=0.3, analysis_type=analysis_type, reasoning=reasoning)

        libs = self._extract_libraries(resp_text)
        latency = getattr(llm_resp, "latency_ms", 0.0)
        context.metadata.setdefault("llm_latency", []).append(latency)
        return AnalysisResult(True, approach, implementation, libs, confidence=confidence, analysis_type=analysis_type, reasoning=reasoning)

    # ------------------------------------------------------------------
    # Helper – parse rejection reason
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_rejection_reason(response: str) -> str:  # noqa: D401
        patterns = [
            r"not viable[:\s]+(.+?)(?:\.|$)",
            r"cannot be (?:visualized|quantified)[:\s]+(.+?)(?:\.|$)",
            r"reason[:\s]+(.+?)(?:\.|$)",
        ]
        for pat in patterns:
            m = re.search(pat, response, re.I | re.S)
            if m:
                return m.group(1).strip()
        return "unspecified"

    # ------------------------------------------------------------------
    # Helper – library extraction
    # ------------------------------------------------------------------
    def _extract_libraries(self, response: str) -> List[str]:  # noqa: D401
        universe = {lib for libs in self.config.available_libraries.values() for lib in libs}
        return sorted({lib for lib in universe if lib.lower() in response.lower()})

    # ------------------------------------------------------------------
    # Helper – mermaid heuristics
    # ------------------------------------------------------------------
    @staticmethod
    def _should_use_mermaid(response: str, approach: str) -> bool:  # noqa: D401
        indicators = ["mermaid", "flowchart", "workflow", "decision tree", "diagram"]
        l_resp, l_appr = response.lower(), approach.lower()
        return any(ind in l_resp or ind in l_appr for ind in indicators)

    # ------------------------------------------------------------------
    # Mermaid generation helpers
    # ------------------------------------------------------------------
    async def _generate_mermaid_diagram(self, idea_text: str, approach: str, gateway, context: NireonExecutionContext) -> Optional[str]:  # noqa: ANN001,E501
        diag_type = self._determine_mermaid_type(approach)
        prompt = self.prompts.mermaid_generation(idea_text, diag_type)
        resp = await self._call_llm(gateway, prompt, "mermaid_generator", context)
        return resp.text.strip() if resp and resp.text.strip() else None

    @staticmethod
    def _determine_mermaid_type(approach: str) -> str:  # noqa: D401
        l = approach.lower()
        if any(w in l for w in ("process", "workflow", "flow")):
            return "flowchart"
        if any(w in l for w in ("network", "relationship", "connection")):
            return "graph"
        if any(w in l for w in ("hierarchy", "structure", "organization")):
            return "classDiagram"
        if any(w in l for w in ("timeline", "sequence", "time")):
            return "timeline"
        return "flowchart"

    @staticmethod
    def _create_mermaid_proto_request(idea_text: str, mermaid: str) -> str:  # noqa: D401,E501
        return (
            f"Create a Python function that outputs the following Mermaid diagram "
            f"for the concept: \"{idea_text}\".\n\n"
            "Requirements:\n"
            "1. Save the diagram to 'diagram.mmd'.\n"
            "2. Duplicate the content to 'mermaid_output.txt' with rendering notes.\n"
            "3. Print CLI instructions for local rendering (e.g. mmdc).\n\n"
            "Mermaid diagram:\n"
            f"{mermaid}\n"
        )

    # ------------------------------------------------------------------
    # LLM gateway wrapper
    # ------------------------------------------------------------------
    async def _call_llm(self, gateway, prompt: str, role: str, context: NireonExecutionContext) -> Optional[LLMResponse]:  # noqa: ANN001,E501
        if gateway is None:
            logger.error("[analysis] gateway unavailable for LLM calls")
            return None

        from domain.cognitive_events import CognitiveEvent  # local import
        from domain.epistemic_stage import EpistemicStage

        frame_id = context.metadata.get("current_frame_id", f"quantifier_{context.run_id}")
        try:
            event = CognitiveEvent.for_llm_ask(frame_id, "quantifier_agent", prompt, EpistemicStage.SYNTHESIS, role)
            task = gateway.process_cognitive_event(event, context)
            resp: LLMResponse = await asyncio.wait_for(task, timeout=self.config.llm_timeout_seconds)  # type: ignore[assignment]
            return resp if isinstance(resp, LLMResponse) else None
        except asyncio.TimeoutError:
            logger.error("[analysis] LLM call timed out after %s s (role=%s)", self.config.llm_timeout_seconds, role)
        except Exception as exc:  # noqa: BLE001
            logger.exception("[analysis] LLM gateway error: %s", exc)
        return None
