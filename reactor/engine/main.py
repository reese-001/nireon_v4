# nireon_v4/reactor/engine/main.py
from __future__ import annotations
import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Final, Iterable, List, Optional, TYPE_CHECKING
from core.lifecycle import ComponentRegistryMissingError
from core.registry import ComponentRegistry
from domain.context import NireonExecutionContext
from domain.ports.event_bus_port import EventBusPort
# FIX: Use the private _signal_class_map for programmatic instantiation
from signals import _signal_class_map, EpistemicSignal

from ..models import Action, EmitSignalAction, RuleContext, TriggerComponentAction
from ..protocols import ReactorRule

if TYPE_CHECKING:
    from signals.base import EpistemicSignal

logger = logging.getLogger(__name__)
_RECURSION_LIMIT: Final = 25


class MainReactorEngine:
    def __init__(self, registry: ComponentRegistry, *, rules: Optional[Iterable[ReactorRule]]=None, max_recursion_depth: int=10, loop: Optional[asyncio.AbstractEventLoop]=None) -> None:
        self.registry = registry
        # Sort rules by priority during initialization
        self.rules: List[ReactorRule] = sorted(list(rules) if rules else [], key=lambda r: getattr(r, 'priority', 100))
        self.max_recursion_depth = min(max_recursion_depth, _RECURSION_LIMIT)
        self._loop = loop or asyncio.get_event_loop()
        try:
            self._event_bus: Optional[EventBusPort] = self.registry.get_service_instance(EventBusPort)
        except ComponentRegistryMissingError:
            self._event_bus = None
            logger.warning('Reactor could not find an EventBusPort – emitted signals will not be broadcast.')
        logger.info('MainReactorEngine ready – %d rule(s) sorted by priority, max_depth=%d', len(self.rules), self.max_recursion_depth)

    def add_rule(self, rule: ReactorRule) -> None:
        self.rules.append(rule)
        self.rules.sort(key=lambda r: getattr(r, 'priority', 100))
        logger.debug("Rule '%s' added and rules re-sorted (total=%d)", rule.rule_id, len(self.rules))

    async def process_signal(self, signal: 'EpistemicSignal', *, _depth: int=0) -> None:
        logger.debug("REACTOR RECEIVED SIGNAL:\n%s", signal.model_dump_json(indent=2))
        if _depth >= self.max_recursion_depth:
            logger.error('Recursion limit (%d) exceeded for signal %s', self.max_recursion_depth, signal.signal_type)
            return

        run_id = getattr(signal, 'run_id', 'unknown_run')
        context = RuleContext(
            signal=signal,
            run_id=run_id,
            component_registry=self.registry,
            logger=logger,
            recursion_depth=_depth
        )
        logger.info('[Depth:%d] Processing signal %s from %s', _depth, signal.signal_type, signal.source_node_id)

        matched_actions: List[Action] = []
        for rule in self.rules:
            try:
                # Use a helper to encapsulate matching logic and add detailed logging
                is_match, reason = await self._evaluate_rule(rule, signal, context)
                if is_match:
                    logger.info("✅ Rule '%s' MATCHED signal '%s'.", rule.rule_id, signal.signal_type)
                    actions = await rule.execute(signal, context)
                    matched_actions.extend(actions)
                else:
                    # This debug log is invaluable for seeing why a rule didn't fire
                    logger.debug("  Rule '%s' skipped for signal '%s'. Reason: %s", rule.rule_id, signal.signal_type, reason)

            except Exception as exc:
                logger.exception("Rule '%s' failed during evaluation: %s", rule.rule_id, exc)

        if matched_actions:
            logger.info("Found %d action(s) to execute from matched rules.", len(matched_actions))
            await self._execute_actions(matched_actions, context)
        else:
            logger.debug('No rules matched signal %s', signal.signal_type)

    async def _evaluate_rule(self, rule: ReactorRule, signal: 'EpistemicSignal', context: 'RuleContext') -> tuple[bool, str]:
        """Evaluates a rule against a signal, providing a reason for failure."""
        if not hasattr(rule, 'signal_type') or signal.signal_type != rule.signal_type:
            return False, f"Signal type mismatch (Rule wants '{getattr(rule, 'signal_type', 'N/A')}', got '{signal.signal_type}')"
        
        # This assumes rules now have a `conditions` attribute, which our YAML-loaded rules do.
        if not hasattr(rule, 'conditions'):
             # Fallback for simple rule types that don't use the condition list structure
            if await rule.matches(signal, context):
                return True, "Simple match successful"
            else:
                return False, "Simple match failed"

        for i, condition in enumerate(rule.conditions):
            cond_type = condition.get('type')
            if cond_type == 'signal_type_match':
                continue # Already checked this

            if cond_type == 'payload_expression':
                expression = condition.get('expression', 'False')
                try:
                    # Re-use the evaluation logic from ConditionalRule for consistency
                    from ..expressions.rel_engine import RELEngine
                    rel_engine = RELEngine()
                    # Build a rich context for the expression evaluator
                    rel_context = {
                        'signal': signal, 'context': context, 'payload': signal.payload,
                        'metadata': getattr(signal, 'metadata', {}),
                        'now': datetime.now(timezone.utc),
                        'trust_score': getattr(signal, 'trust_score', None),
                        'novelty_score': getattr(signal, 'novelty_score', None),
                        'exists': lambda x: x is not None,
                        'lower': lambda s: s.lower() if isinstance(s, str) else s
                    }
                    result = bool(rel_engine.evaluate(expression, rel_context))
                    if not result:
                        return False, f"Condition #{i+1} (expression) failed: '{expression}' evaluated to False."
                except Exception as e:
                    return False, f"Condition #{i+1} (expression) failed with error: {e}"
            else:
                return False, f"Unknown condition type '{cond_type}' in rule '{rule.rule_id}'"

        return True, "All conditions met"

    async def _execute_actions(self, actions: List[Action], context: RuleContext) -> None:
        for act in actions:
            if isinstance(act, TriggerComponentAction):
                await self._handle_trigger_component(act, context)
            elif isinstance(act, EmitSignalAction):
                await self._handle_emit_signal(act, context)
            else:
                logger.warning('Unknown Action type: %s', type(act).__name__)

    async def _handle_trigger_component(self, action: TriggerComponentAction, context: RuleContext) -> None:
        comp_id = action.component_id
        logger.info('Triggering component %s (template=%s)', comp_id, action.template_id)
        try:
            component = self.registry.get(comp_id)
        except ComponentRegistryMissingError as exc:
            logger.error("Component '%s' not found: %s", comp_id, exc)
            return
        if not callable(getattr(component, 'process', None)):
            logger.error("Component %s lacks an async 'process' method", comp_id)
            return

        exec_ctx = NireonExecutionContext(
            run_id=context.run_id,
            component_id=comp_id,
            component_registry=self.registry,
            event_bus=self._event_bus,
            logger=context.logger,
            metadata={
                'source_rule_id': context.signal.signal_id,
                'triggering_signal_type': context.signal.signal_type
            }
        )

        kwargs = {
            'data': action.input_data,
            'context': exec_ctx,
        }
        if action.template_id:
            kwargs['template_id'] = action.template_id

        try:
            await component.process(**kwargs)
        except Exception as exc:
            logger.exception('TriggerComponentAction failed: %s', exc)

    async def _handle_emit_signal(self, action: EmitSignalAction, context: RuleContext) -> None:
        logger.info('Emitting signal %s (depth %d)', action.signal_type, context.recursion_depth + 1)
        
        # --- START OF FIX ---
        
        # 1. Look up the specific signal class (e.g., PlanNextStepSignal) from the map.
        # Fallback to the generic EpistemicSignal if not found.
        SignalClass = _signal_class_map.get(action.signal_type, EpistemicSignal)
        
        # 2. Prepare the data for the signal's constructor.
        constructor_data = action.payload.copy()
        constructor_data['signal_type'] = action.signal_type
        
        # 3. Add the required source_node_id, using an override if the rule specifies one.
        constructor_data['source_node_id'] = action.source_node_id_override or context.signal.source_node_id
        
 
        
        try:
            # The 'constructor_data' now directly contains the fields needed by the specific signal class.
            new_sig = SignalClass(**constructor_data)
        except Exception as e:
            logger.error(f"Failed to construct signal '{action.signal_type}': {e}", exc_info=True)
            logger.error(f"Constructor data that failed validation: {constructor_data}")
            return
        
        # --- END OF FIX ---

        if self._event_bus:
            # Publishing now uses the created signal object to ensure consistency
            self._event_bus.publish(new_sig.signal_type, new_sig)
        else:
            logger.debug('Event bus unavailable – signal not broadcast externally')
        # Recurse with the newly created and validated signal object
        await self.process_signal(new_sig, _depth=context.recursion_depth + 1)