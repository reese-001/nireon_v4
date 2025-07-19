# stage_evaluation_service.py
from __future__ import annotations

import logging
import textwrap
from typing import Any, Dict, Mapping, Optional

from core.base_component import NireonBaseComponent
from core.lifecycle import ComponentMetadata
from core.results import ProcessResult
from domain.context import NireonExecutionContext
from domain.epistemic_stage import EpistemicStage

__all__ = ["StageEvaluationService"]

logger = logging.getLogger(__name__)

###############################################################################
# Component metadata
###############################################################################
STAGE_EVALUATION_SERVICE_METADATA = ComponentMetadata(
    id="stage_evaluation_service",
    name="Stage Evaluation Service",
    version="1.1.0",
    category="service_core",
    description=(
        "Resolves stage-specific assessment parameters and synthesises the LLM "
        "prompt used by evaluation mechanisms (e.g. Sentinel)."
    ),
    epistemic_tags=["evaluator", "contextualizer", "parameter_resolver"],
    requires_initialize=True,
)

###############################################################################
# Default/fallback parameters
###############################################################################
DEFAULT_PARAMS: Dict[str, float] = {
    "alignment_weight": 0.4,
    "feasibility_weight": 0.3,
    "novelty_weight": 0.3,
    "trust_threshold": 5.0,
    "min_axis_score": 4.0,
}

###############################################################################
# Prompt fragments
###############################################################################
_STAGE_FOCUS = {
    EpistemicStage.EXPLORATION: "initial exploration and creativity",
    EpistemicStage.DEFAULT: "balanced exploration and refinement",
}.get  # type: ignore[attr-defined]

_PROMPT_TEMPLATE = textwrap.dedent(
    """
    Evaluate the following idea against the stated objective.

    • **Current stage:** “{stage}” → focus on {stage_focus}.
    • **Scoring:** Provide 1-10 floats for:
        - Alignment (align_score)  
        - Feasibility (feas_score)

    Return **only** a valid JSON object with:
    {{
      "align_score": <float>,
      "feas_score": <float>,
      "explanation": "<≤150 characters>"
    }}

    **Objective**
    \"\"\"{objective}\"\"\"

    **Idea**
    \"\"\"{idea}\"\"\"
    """
).strip()


class StageEvaluationService(NireonBaseComponent):
    """
    Library-style component.
    Resolves assessment weights/thresholds and builds the evaluation prompt.
    """

    def __init__(
        self,
        config: Optional[Mapping[str, Any]] = None,
        metadata_definition: Optional[ComponentMetadata] = None,
    ) -> None:
        super().__init__(
            config=dict(config or {}),
            metadata_definition=metadata_definition or STAGE_EVALUATION_SERVICE_METADATA,
        )
        self._stage_overrides: Dict[str, Dict[str, Any]] = self.config.get(
            "stage_evaluation_configs", {}
        )
        logger.info(
            "[%s] loaded %d stage-specific override sets",
            self.component_id,
            len(self._stage_overrides),
        )

    # --------------------------------------------------------------------- #
    #  lifecycle hooks
    # --------------------------------------------------------------------- #
    async def _process_impl(
        self, data: Any, context: NireonExecutionContext
    ) -> ProcessResult:  # noqa: D401
        """
        This service is query-only; direct processing is unsupported.
        """
        return ProcessResult(
            success=False,
            component_id=self.component_id,
            message="Direct `process()` not supported - call helper methods instead.",
            error_code="METHOD_NOT_SUPPORTED",
        )

    # --------------------------------------------------------------------- #
    #  public helpers
    # --------------------------------------------------------------------- #
    def get_assessment_parameters(
        self,
        stage: Optional[EpistemicStage],
        context: NireonExecutionContext,
        base_params: Mapping[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """
        Merge global defaults → mechanism-supplied base_params → stage overrides.
        """
        stage_key = (stage or EpistemicStage.DEFAULT).value
        overrides = self._stage_overrides.get(stage_key, {})

        final: Dict[str, Any] = {
            **DEFAULT_PARAMS,  # global fallbacks
            **(base_params or {}),
            **overrides,  # highest precedence
        }

        # Sanity check: weights should sum to ≈ 1.0 (±0.01).
        weights = [
            final.get("alignment_weight", 0.0),
            final.get("feasibility_weight", 0.0),
            final.get("novelty_weight", 0.0),
        ]
        if abs(sum(weights) - 1.0) > 0.01:
            context.logger.warning(
                "[%s] assessment weights sum to %.3f (expected 1.0); caller should verify.",
                self.component_id,
                sum(weights),
            )

        context.logger.debug(
            "[%s] resolved parameters for stage '%s': %s", self.component_id, stage_key, final
        )
        return final

    def build_stage_assessment_prompt(
        self,
        idea_text: str,
        objective: str,
        stage: EpistemicStage,
        context: NireonExecutionContext,
    ) -> str:
        """
        Construct the LLM prompt used by Sentinel / other mechanisms.
        """
        prompt = _PROMPT_TEMPLATE.format(
            stage=stage.value,
            stage_focus=_STAGE_FOCUS(stage, "refinement and feasibility"),
            idea=idea_text,
            objective=objective,
        )
        context.logger.debug(
            "[%s] built assessment prompt for stage '%s' (%d chars)",
            self.component_id,
            stage.value,
            len(prompt),
        )
        return prompt
