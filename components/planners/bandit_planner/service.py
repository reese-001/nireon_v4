# nireon_v4/components/planners/bandit_planner/service.py
import logging
import random
import asyncio
from pathlib import Path
from typing import Any, Dict

from core.base_component import NireonBaseComponent
from core.lifecycle import ComponentMetadata
from core.results import ProcessResult
from domain.context import NireonExecutionContext
from domain.ports.event_bus_port import EventBusPort
from signals.core import SeedSignal, PlanNextStepSignal
from .config import BanditPlannerConfig
from .metadata import BANDIT_PLANNER_METADATA

logger = logging.getLogger(__name__)

class BanditPlanner(NireonBaseComponent):
    METADATA_DEFINITION = BANDIT_PLANNER_METADATA
    ConfigModel = BanditPlannerConfig

    def __init__(self, config: Dict[str, Any], metadata_definition: ComponentMetadata):
        super().__init__(config, metadata_definition)
        self.cfg: BanditPlannerConfig = self.ConfigModel(**self.config)
        self.event_bus: EventBusPort | None = None
        self.mab = None
        self.rng = random.Random()
        self._model_last_loaded_time = 0.0

    async def _initialize_impl(self, context: NireonExecutionContext) -> None:
        self.event_bus = context.component_registry.get_service_instance(EventBusPort)
        self._load_model(context)

        # FIX (from review): Start a background task for model hot-reloading.
        asyncio.create_task(self._model_watcher(context))

    async def _model_watcher(self, context: NireonExecutionContext):
        """Periodically checks if the model file has been updated and reloads it."""
        while True:
            await asyncio.sleep(300) # Check every 5 minutes
            try:
                model_path = Path(self.cfg.model_path)
                if model_path.exists():
                    mod_time = model_path.stat().st_mtime
                    if mod_time > self._model_last_loaded_time:
                        context.logger.info(f"[{self.component_id}] New model file detected. Reloading...")
                        self._load_model(context)
            except Exception as e:
                context.logger.error(f"[{self.component_id}] Error in model watcher: {e}")

    def _load_model(self, context: NireonExecutionContext):
        model_path = Path(self.cfg.model_path)
        if model_path.exists():
            try:
                from mabwiser.mab import ContextualMAB
                self.mab = ContextualMAB.load(str(model_path))
                self._model_last_loaded_time = model_path.stat().st_mtime
                context.logger.info(f"[{self.component_id}] Loaded bandit model from {model_path}")
            except Exception as e:
                context.logger.error(f"[{self.component_id}] Failed to load model: {e}. Will use default action.")
                self.mab = None
        else:
            context.logger.warning(f"[{self.component_id}] Bandit model not found at {model_path}. Will use default action.")
            self.mab = None

    async def _process_impl(self, data: Dict[str, Any], context: NireonExecutionContext) -> ProcessResult:
        try:
            # Reconstruct the signal from the dictionary provided by the reactor rule
            signal = PlanNextStepSignal(**data)
        except Exception as e:
            msg = f"Failed to create PlanNextStepSignal from input data: {e}. Data: {data}"
            logger.error(f'[{self.component_id}] {msg}')
            return ProcessResult(success=False, component_id=self.component_id, message=msg, error_code="INVALID_INPUT_DATA")

        chosen_action = self.cfg.default_action
        context_features = [signal.current_trust_score, float(signal.current_depth)]

        # Epsilon is now read directly from the (potentially updated) config.
        current_epsilon = self.cfg.exploration_epsilon
        
        if self.mab and self.rng.random() > current_epsilon:
            try:
                prediction = self.mab.predict([context_features])
                chosen_action = prediction[0]
                context.logger.info(f"[{self.component_id}] Bandit chose action '{chosen_action}' based on context.")
            except Exception as e:
                context.logger.error(f"[{self.component_id}] Bandit prediction failed: {e}. Falling back to default.")
        else:
            chosen_action = self.rng.choice(self.cfg.available_actions)
            context.logger.info(f"[{self.component_id}] Epsilon-greedy exploration chose random action: '{chosen_action}'")
        
        # FIX: The planner's "action" is to emit a signal to trigger a mechanism.
        # This now uses the configurable map from the YAML file.
        target_component_id = self.cfg.action_to_mechanism_map.get(chosen_action, self.cfg.action_to_mechanism_map.get(self.cfg.default_action))
        
        # We re-use SeedSignal as a generic entry point for the generative loop.
        # It's crucial to pass metadata forward for the trace assembler.
        # When emitting the next SeedSignal, use the reconstructed signal's data
        next_signal = SeedSignal(
            source_node_id=self.component_id,
            seed_content=signal.current_idea_text,
            # The payload is now a flat dictionary, which is much cleaner.
            payload={
                'seed_idea_id': signal.current_idea_id,
                'text': signal.current_idea_text,
                'objective': signal.objective,
                'depth': signal.current_depth,
                'planner_action': chosen_action,
                'target_component_id': target_component_id,
                'session_id': signal.session_id,
                'parent_trust_score': signal.current_trust_score,
                'frame_id': context.metadata.get('current_frame_id')
            }
        )
        

       
        if self.event_bus:
            self.event_bus.publish(next_signal.signal_type, next_signal)
            message = f"Planned next action '{chosen_action}' -> '{target_component_id}' and emitted SeedSignal."
            return ProcessResult(success=True, component_id=self.component_id, message=message, output_data={'chosen_action': chosen_action})
        else:
            return ProcessResult(success=False, component_id=self.component_id, message='EventBus not available.')