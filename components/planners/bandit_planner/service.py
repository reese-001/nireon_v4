# nireon_v4/components/planners/bandit_planner/service.py
import logging
import random
import asyncio
from pathlib import Path
from typing import Any, Dict, Optional

from core.base_component import NireonBaseComponent
from core.lifecycle import ComponentMetadata
from core.results import ProcessResult, ComponentHealth, AnalysisResult
from domain.context import NireonExecutionContext
from domain.ports.event_bus_port import EventBusPort
from components.service_resolution_mixin import ServiceResolutionMixin
from signals.core import SeedSignal, PlanNextStepSignal
from .config import BanditPlannerConfig
from .metadata import BANDIT_PLANNER_METADATA

logger = logging.getLogger(__name__)

class BanditPlanner(NireonBaseComponent, ServiceResolutionMixin):
    """
    Bandit-based Epistemic Planner that uses a contextual bandit model to choose next actions.
    
    Required Services:
        - event_bus (EventBusPort): For publishing signals to trigger next mechanisms
        
    The planner loads a pre-trained contextual bandit model and uses it to select
    the next action (mechanism) based on the current context (trust score, depth).
    """
    
    METADATA_DEFINITION = BANDIT_PLANNER_METADATA
    ConfigModel = BanditPlannerConfig

    def __init__(self, config: Dict[str, Any], metadata_definition: ComponentMetadata):
        super().__init__(config, metadata_definition)
        self.cfg: BanditPlannerConfig = self.ConfigModel(**self.config)
        self.event_bus: EventBusPort | None = None
        self.mab = None
        self.rng = random.Random()
        self._model_last_loaded_time = 0.0
        
        # Metrics for monitoring
        self._total_decisions = 0
        self._bandit_decisions = 0
        self._random_decisions = 0
        self._fallback_decisions = 0
        self._model_reload_count = 0
        self._last_chosen_action = None
        
        # Background task handle
        self._model_watcher_task: Optional[asyncio.Task] = None

    async def _initialize_impl(self, context: NireonExecutionContext) -> None:
        """Initialize the planner with service resolution and model loading."""
        context.logger.info(f"[{self.component_id}] Initializing BanditPlanner")
        
        # Resolve required services using the mixin
        await self._resolve_all_dependencies(context)
        
        # Validate dependencies
        self._validate_dependencies(context)
        
        # Load the bandit model
        self._load_model(context)

        # Start a background task for model hot-reloading
        self._model_watcher_task = asyncio.create_task(self._model_watcher(context))
        
        context.logger.info(f"[{self.component_id}] BanditPlanner initialized successfully")

    async def _resolve_all_dependencies(self, context: NireonExecutionContext) -> None:
        """Resolve all required dependencies using the ServiceResolutionMixin."""
        
        # Define required services
        service_map = {
            'event_bus': EventBusPort
        }
        
        try:
            # Resolve services using the mixin
            resolved_services = self.resolve_services(
                context=context,
                service_map=service_map,
                raise_on_missing=True,
                log_resolution=True
            )
            
            context.logger.debug(
                f"[{self.component_id}] Resolved {len(resolved_services)} services"
            )
            
        except RuntimeError as e:
            context.logger.error(f"[{self.component_id}] Failed to resolve dependencies: {e}")
            raise

    def _validate_dependencies(self, context: NireonExecutionContext) -> None:
        """Validate that all required dependencies are available."""
        
        required_services = ['event_bus']
        
        # Use the mixin's validation method
        if not self.validate_required_services(required_services, context):
            raise RuntimeError(
                f"BanditPlanner '{self.component_id}' missing required EventBusPort"
            )

    def _ensure_services_available(self, context: NireonExecutionContext) -> bool:
        """
        Ensure required services are available at runtime.
        Can attempt re-resolution if services are missing.
        """
        if self.event_bus:
            return True
            
        # Attempt to re-resolve
        context.logger.warning(
            f"[{self.component_id}] EventBus not available, attempting re-resolution"
        )
        
        try:
            self.resolve_services(
                context=context,
                service_map={'event_bus': EventBusPort},
                raise_on_missing=False,
                log_resolution=True
            )
            return self.event_bus is not None
            
        except Exception as e:
            context.logger.error(f"[{self.component_id}] Failed to ensure services: {e}")
            return False

    async def _model_watcher(self, context: NireonExecutionContext):
        """Periodically checks if the model file has been updated and reloads it."""
        while True:
            await asyncio.sleep(300)  # Check every 5 minutes
            try:
                model_path = Path(self.cfg.model_path)
                if model_path.exists():
                    mod_time = model_path.stat().st_mtime
                    if mod_time > self._model_last_loaded_time:
                        context.logger.info(f"[{self.component_id}] New model file detected. Reloading...")
                        self._load_model(context)
                        self._model_reload_count += 1
            except asyncio.CancelledError:
                # Task is being cancelled during shutdown
                break
            except Exception as e:
                context.logger.error(f"[{self.component_id}] Error in model watcher: {e}")

    def _load_model(self, context: NireonExecutionContext):
        """Load the contextual bandit model from disk."""
        model_path = Path(self.cfg.model_path)
        if model_path.exists():
            try:
                from mabwiser.mab import ContextualMAB
                self.mab = ContextualMAB.load(str(model_path))
                self._model_last_loaded_time = model_path.stat().st_mtime
                context.logger.info(f"[{self.component_id}] Loaded bandit model from {model_path}")
            except ImportError:
                context.logger.error(
                    f"[{self.component_id}] mabwiser not installed. Cannot load bandit model."
                )
                self.mab = None
            except Exception as e:
                context.logger.error(
                    f"[{self.component_id}] Failed to load model: {e}. Will use default action."
                )
                self.mab = None
        else:
            context.logger.warning(
                f"[{self.component_id}] Bandit model not found at {model_path}. Will use default action."
            )
            self.mab = None

    async def _process_impl(self, data: Dict[str, Any], context: NireonExecutionContext) -> ProcessResult:
        """Process a planning request and emit the next action signal."""
        
        # Ensure services are available
        if not self._ensure_services_available(context):
            return ProcessResult(
                success=False,
                component_id=self.component_id,
                message="EventBus not available",
                error_code="MISSING_DEPENDENCY"
            )
        
        try:
            # Reconstruct the signal from the dictionary provided by the reactor rule
            signal = PlanNextStepSignal(**data)
        except Exception as e:
            msg = f"Failed to create PlanNextStepSignal from input data: {e}. Data: {data}"
            logger.error(f'[{self.component_id}] {msg}')
            return ProcessResult(
                success=False, 
                component_id=self.component_id, 
                message=msg, 
                error_code="INVALID_INPUT_DATA"
            )

        # Increment total decisions counter
        self._total_decisions += 1

        chosen_action = self.cfg.default_action
        context_features = [signal.current_trust_score, float(signal.current_depth)]

        # Epsilon is now read directly from the (potentially updated) config
        current_epsilon = self.cfg.exploration_epsilon
        
        # Decide whether to exploit (use model) or explore (random)
        if self.mab and self.rng.random() > current_epsilon:
            try:
                prediction = self.mab.predict([context_features])
                chosen_action = prediction[0]
                self._bandit_decisions += 1
                context.logger.info(
                    f"[{self.component_id}] Bandit chose action '{chosen_action}' "
                    f"based on context (trust={signal.current_trust_score:.2f}, depth={signal.current_depth})"
                )
            except Exception as e:
                context.logger.error(
                    f"[{self.component_id}] Bandit prediction failed: {e}. Falling back to default."
                )
                self._fallback_decisions += 1
        else:
            # Exploration: choose random action
            chosen_action = self.rng.choice(self.cfg.available_actions)
            self._random_decisions += 1
            context.logger.info(
                f"[{self.component_id}] Epsilon-greedy exploration chose random action: '{chosen_action}'"
            )
        
        # Store last chosen action for monitoring
        self._last_chosen_action = chosen_action
        
        # Map the abstract action to a concrete mechanism component
        target_component_id = self.cfg.action_to_mechanism_map.get(
            chosen_action, 
            self.cfg.action_to_mechanism_map.get(self.cfg.default_action)
        )
        
        # Create the next signal to trigger the chosen mechanism
        next_signal = SeedSignal(
            source_node_id=self.component_id,
            seed_content=signal.current_idea_text,
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
        
        # Publish the signal
        if self.event_bus:
            self.event_bus.publish(next_signal.signal_type, next_signal)
            message = f"Planned next action '{chosen_action}' -> '{target_component_id}' and emitted SeedSignal."
            return ProcessResult(
                success=True, 
                component_id=self.component_id, 
                message=message, 
                output_data={'chosen_action': chosen_action, 'target_component': target_component_id}
            )
        else:
            return ProcessResult(
                success=False, 
                component_id=self.component_id, 
                message='EventBus not available.',
                error_code='EVENT_BUS_MISSING'
            )

    async def analyze(self, context: NireonExecutionContext) -> AnalysisResult:
        """Analyze the planner's performance and decision patterns."""
        
        # Calculate decision type breakdown
        exploitation_rate = 0.0
        exploration_rate = 0.0
        fallback_rate = 0.0
        
        if self._total_decisions > 0:
            exploitation_rate = self._bandit_decisions / self._total_decisions
            exploration_rate = self._random_decisions / self._total_decisions
            fallback_rate = self._fallback_decisions / self._total_decisions
        
        # Check service availability
        service_health = self.validate_required_services(['event_bus'], context)
        
        # Calculate action distribution
        action_distribution = {}
        if hasattr(self, '_action_history'):
            for action in self._action_history:
                action_distribution[action] = action_distribution.get(action, 0) + 1
        
        findings = {
            'total_decisions': self._total_decisions,
            'exploitation_rate': exploitation_rate,
            'exploration_rate': exploration_rate,
            'fallback_rate': fallback_rate,
            'current_epsilon': self.cfg.exploration_epsilon,
            'model_loaded': self.mab is not None,
            'model_reload_count': self._model_reload_count,
            'last_chosen_action': self._last_chosen_action,
            'service_health': service_health,
            'action_distribution': action_distribution
        }
        
        recommendations = []
        insights = []
        
        # Generate insights
        if self.mab is None:
            insights.append("Bandit model not loaded - using random/default actions only")
            recommendations.append(f"Check model file at {self.cfg.model_path}")
        
        if exploitation_rate < 0.5 and self._total_decisions > 100:
            insights.append("Low exploitation rate - mostly exploring")
            recommendations.append("Consider reducing epsilon for more exploitation")
        
        if fallback_rate > 0.1:
            insights.append("High fallback rate indicates model prediction issues")
            recommendations.append("Review model compatibility and input features")
        
        return AnalysisResult(
            success=True,
            component_id=self.component_id,
            message=f"Analyzed {self._total_decisions} planning decisions",
            findings=findings,
            recommendations=recommendations,
            insights=insights
        )

    async def health_check(self, context: NireonExecutionContext) -> ComponentHealth:
        """Check the health of the planner component."""
        
        # Check service availability
        service_available = self.event_bus is not None
        
        # Check model status
        model_status = "loaded" if self.mab else "not_loaded"
        model_path_exists = Path(self.cfg.model_path).exists()
        
        # Check background task
        watcher_running = (
            self._model_watcher_task is not None and 
            not self._model_watcher_task.done()
        )
        
        # Determine overall health
        if not service_available:
            status = "unhealthy"
            message = "EventBus service not available"
        elif not model_path_exists:
            status = "degraded"
            message = f"Model file not found at {self.cfg.model_path}"
        elif self.mab is None:
            status = "degraded"
            message = "Model failed to load - using fallback behavior"
        else:
            status = "healthy"
            message = "All systems operational"
        
        metrics = {
            'total_decisions': self._total_decisions,
            'model_status': model_status,
            'model_path_exists': model_path_exists,
            'model_reload_count': self._model_reload_count,
            'watcher_task_running': watcher_running,
            'service_available': service_available,
            'current_epsilon': self.cfg.exploration_epsilon,
            'last_action': self._last_chosen_action
        }
        
        return ComponentHealth(
            component_id=self.component_id,
            status=status,
            message=message,
            metrics=metrics
        )

    async def shutdown(self, context: NireonExecutionContext) -> None:
        """Clean shutdown with proper cleanup."""
        context.logger.info(f"[{self.component_id}] Beginning BanditPlanner shutdown...")
        
        # Cancel the model watcher task
        if self._model_watcher_task and not self._model_watcher_task.done():
            context.logger.debug(f"[{self.component_id}] Cancelling model watcher task...")
            self._model_watcher_task.cancel()
            try:
                await self._model_watcher_task
            except asyncio.CancelledError:
                pass
        
        # Log final metrics
        context.logger.info(
            f"[{self.component_id}] Shutdown complete. "
            f"Total decisions: {self._total_decisions}, "
            f"Model reloads: {self._model_reload_count}"
        )
        
        # Call parent shutdown
        await super().shutdown(context)