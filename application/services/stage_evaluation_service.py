# nireon_v4\application\services\stage_evaluation_service.py
from __future__ import annotations
import logging
import textwrap
from typing import Any, Dict, Optional
from core.base_component import NireonBaseComponent
from core.lifecycle import ComponentMetadata
from core.results import ProcessResult
from domain.context import NireonExecutionContext
from domain.epistemic_stage import EpistemicStage
# --- REMOVED IMPORTS from components.mechanisms.sentinel ---

logger = logging.getLogger(__name__)

STAGE_EVALUATION_SERVICE_METADATA = ComponentMetadata(id='stage_evaluation_service', name='Stage Evaluation Service', version='1.0.0', category='service_core', description='Provides stage-specific parameters and prompts for idea evaluation.', epistemic_tags=['evaluator', 'contextualizer', 'parameter_resolver'], requires_initialize=False)

# --- ADDED LOCAL DEFAULTS to decouple from Sentinel ---
DEFAULT_ALIGNMENT_WEIGHT = 0.4
DEFAULT_FEASIBILITY_WEIGHT = 0.3
DEFAULT_NOVELTY_WEIGHT = 0.3
DEFAULT_TRUST_THRESHOLD = 5.0
DEFAULT_MIN_AXIS_SCORE = 4.0

class StageEvaluationService(NireonBaseComponent):
    def __init__(self, config: Optional[Dict[str, Any]]=None, metadata_definition: Optional[ComponentMetadata]=None):
        super().__init__(config=config or {}, metadata_definition=metadata_definition or STAGE_EVALUATION_SERVICE_METADATA)
        self.stage_configs = self.config.get('stage_evaluation_configs', {})
        logger.info(f'[{self.component_id}] initialized with {len(self.stage_configs)} stage-specific configs.')

    async def _process_impl(self, data: Any, context: NireonExecutionContext) -> ProcessResult:
        return ProcessResult(success=False, component_id=self.component_id, message='StageEvaluationService is a library-like service and does not implement a generic process method.', error_code='METHOD_NOT_SUPPORTED')

    def get_assessment_parameters(self, stage: Optional[EpistemicStage], context: NireonExecutionContext) -> Dict[str, Any]:
        stage_key = stage.value if stage else EpistemicStage.DEFAULT.value
        stage_config = self.stage_configs.get(stage_key, {})
        # --- MODIFIED to use local defaults ---
        defaults = {
            'alignment_weight': DEFAULT_ALIGNMENT_WEIGHT,
            'feasibility_weight': DEFAULT_FEASIBILITY_WEIGHT,
            'novelty_weight': DEFAULT_NOVELTY_WEIGHT,
            'trust_threshold': DEFAULT_TRUST_THRESHOLD,
            'min_axis_score': DEFAULT_MIN_AXIS_SCORE
        }
        params = {**defaults, **stage_config}
        context.logger.debug(f"[{self.component_id}] Resolved assessment parameters for stage '{stage_key}': {params}")
        return params

    def build_stage_assessment_prompt(self, idea_text: str, objective: str, stage: EpistemicStage, context: NireonExecutionContext) -> str:
        prompt = textwrap.dedent(f'''
            Evaluate the following idea based on the stated objective and its current epistemic stage.
            The current stage is '{stage.value}', which implies a focus on {('initial exploration and creativity' if stage == EpistemicStage.EXPLORATION else 'refinement and feasibility')}.

            Provide scores on a 1-10 scale for the following axes:
            - Alignment: How well does the idea align with the objective?
            - Feasibility: How practical and achievable is the idea, considering the current stage?

            Return your evaluation *only* as a single, valid JSON object with the following keys:
            {{
              "align_score": <1-10 float>,
              "feas_score": <1-10 float>,
              "explanation": "<brief rationale for your scores>"
            }}

            Objective: "{objective}"

            Idea to Evaluate:
            """
            {idea_text}
            """
        ''').strip()
        context.logger.debug(f"[{self.component_id}] Built assessment prompt for stage '{stage.value}'.")
        return prompt

__all__ = ['StageEvaluationService']