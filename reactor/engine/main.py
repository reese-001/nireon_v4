from __future__ import annotations
import logging
from typing import List, TYPE_CHECKING, Optional
from reactor.protocols import ReactorRule
from reactor.models import RuleContext, TriggerComponentAction, EmitSignalAction, Action
from core.registry import ComponentRegistry
from core.lifecycle import ComponentRegistryMissingError
if TYPE_CHECKING:
    from bootstrap.signals import EpistemicSignal

logger = logging.getLogger(__name__)

class MainReactorEngine:
    def __init__(
        self, 
        registry: ComponentRegistry, 
        rules: Optional[List[ReactorRule]] = None, 
        max_recursion_depth: int = 10
    ):
        self.registry = registry
        self.rules: List[ReactorRule] = rules or []
        self.max_recursion_depth = max_recursion_depth
        logger.info(
            f'MainReactorEngine initialized with {len(self.rules)} rules '
            f'(Max Recursion: {self.max_recursion_depth}).'
        )
    
    def add_rule(self, rule: ReactorRule) -> None:
        """Add a new rule to the reactor engine."""
        self.rules.append(rule)
        logger.debug(f"Rule '{rule.rule_id}' added to ReactorEngine.")
    
    async def process_signal(self, signal: 'EpistemicSignal', current_depth: int = 0) -> None:
        """Process a signal through all matching rules."""
        if current_depth >= self.max_recursion_depth:
            logger.error(
                f"Recursion depth limit ({self.max_recursion_depth}) exceeded. "
                f"Stopping signal chain for signal type '{signal.signal_type}'. "
                f"This prevents an infinite loop."
            )
            return
        
        run_id = getattr(signal, 'run_id', 'unknown_run')
        context = RuleContext(
            signal=signal,
            run_id=run_id,
            component_registry=self.registry,
            logger=logger,
            recursion_depth=current_depth
        )
        
        logger.debug(f'[Depth:{current_depth}] Processing signal: {signal.signal_type}')
        
        actions_to_execute: List[Action] = []
        
        # Evaluate all rules and collect actions
        for rule in self.rules:
            try:
                if await rule.matches(signal, context):
                    logger.info(f"Rule '{rule.rule_id}' MATCHED signal '{signal.signal_type}'.")
                    new_actions = await rule.execute(signal, context)
                    actions_to_execute.extend(new_actions)
                else:
                    logger.debug(f"Rule '{rule.rule_id}' did not match.")
            except Exception as e:
                logger.error(
                    f"Error evaluating rule '{rule.rule_id}' for signal '{signal.signal_type}': {e}",
                    exc_info=True
                )
        
        if not actions_to_execute:
            logger.debug(f"No actions triggered for signal '{signal.signal_type}'.")
            return
        
        await self._execute_actions(actions_to_execute, context)
    
    async def _execute_actions(self, actions: List[Action], context: RuleContext) -> None:
        """Execute a list of actions."""
        for action in actions:
            try:
                if isinstance(action, TriggerComponentAction):
                    await self._handle_trigger_component(action, context)
                elif isinstance(action, EmitSignalAction):
                    await self._handle_emit_signal(action, context)
                else:
                    logger.warning(f'Unknown action type received: {type(action).__name__}')
            except Exception as e:
                logger.error(f"Failed to execute action {type(action).__name__}: {e}", exc_info=True)
    
    async def _handle_trigger_component(self, action: TriggerComponentAction, context: RuleContext) -> None:
        logger.info(f"Executing TriggerComponentAction for component '{action.component_id}' (Template: {action.template_id})")
        try:
            component = self.registry.get(action.component_id)
            if not component:
                raise ComponentRegistryMissingError(f"Component '{action.component_id}' not found in registry.")
            if not hasattr(component, 'process') or not callable(getattr(component, 'process')):
                raise AttributeError(f"Component '{action.component_id}' has no callable 'process' method.")

            # Prepare the arguments for the process call
            process_kwargs = {
                'data': action.input_data,
                'context': context
            }
            # Add template_id if it exists in the action
            if action.template_id:
                process_kwargs['template_id'] = action.template_id

            await component.process(**process_kwargs)
        except Exception as e:
            logger.error(f"Failed to execute TriggerComponentAction for '{action.component_id}': {e}", exc_info=True)
    
    async def _handle_emit_signal(
        self, 
        action: EmitSignalAction, 
        context: RuleContext
    ) -> None:
        """Handle EmitSignalAction by creating and processing a new signal."""
        logger.info(
            f"Executing EmitSignalAction: emitting signal type '{action.signal_type}' "
            f"with recursion depth {context.recursion_depth + 1}"
        )
        
        try:
            # Import here to avoid circular imports
            from signals.base import EpistemicSignal
            
            # Create the new signal
            new_signal = EpistemicSignal(
                signal_type=action.signal_type,
                source_node_id=action.source_node_id_override or context.signal.source_node_id,
                payload=action.payload
            )
            
            # Process the new signal recursively (with increased depth)
            await self.process_signal(new_signal, current_depth=context.recursion_depth + 1)
            
        except Exception as e:
            logger.error(
                f"Failed to emit signal '{action.signal_type}': {e}",
                exc_info=True
            )