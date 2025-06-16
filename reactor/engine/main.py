from __future__ import annotations
import asyncio
import logging
from collections import deque
from typing import List, Optional, TYPE_CHECKING, Final, Iterable

from reactor.protocols import ReactorRule
from reactor.models import RuleContext, TriggerComponentAction, EmitSignalAction, Action
from core.registry import ComponentRegistry
from core.lifecycle import ComponentRegistryMissingError
from domain.context import NireonExecutionContext
from domain.ports.event_bus_port import EventBusPort

if TYPE_CHECKING:
    from signals.base import EpistemicSignal

logger = logging.getLogger(__name__)

_RECURSION_LIMIT: Final[int] = 25

class MainReactorEngine:

    def __init__(self, registry: ComponentRegistry, *, rules: Optional[Iterable[ReactorRule]] = None, max_recursion_depth: int = 10, loop: Optional[asyncio.AbstractEventLoop] = None) -> None:
        self.registry = registry
        self.rules: list[ReactorRule] = list(rules) if rules else []
        self.max_recursion_depth: int = min(max_recursion_depth, _RECURSION_LIMIT)
        self._loop = loop or asyncio.get_running_loop()
        # FIX: Get a reference to the event bus during initialization
        self._event_bus: Optional[EventBusPort] = None
        try:
            self._event_bus = self.registry.get_service_instance(EventBusPort)
        except ComponentRegistryMissingError:
            logger.warning("Reactor could not find an EventBusPort in the registry. Emitted signals will not be broadcast.")
        
        logger.info('MainReactorEngine ready – %d rule(s), max_depth=%d', len(self.rules), self.max_recursion_depth)

    def add_rule(self, rule: ReactorRule) -> None:
        self.rules.append(rule)
        logger.debug("Rule '%s' added (total=%d)", rule.rule_id, len(self.rules))

    async def process_signal(self, signal: 'EpistemicSignal', *, _depth: int = 0) -> None:
        if _depth >= self.max_recursion_depth:
            logger.error('Recursion limit (%d) exceeded for signal %s – chain halted', self.max_recursion_depth, signal.signal_type)
            return

        run_id: str = getattr(signal, 'run_id', 'unknown_run')
        context = RuleContext(signal=signal, run_id=run_id, component_registry=self.registry,
                              logger=logger, recursion_depth=_depth)

        logger.debug('[Depth:%d] Processing signal %s', _depth, signal.signal_type)

        matching_actions: list[Action] = []
        for rule in self.rules:
            try:
                if await rule.matches(signal, context):
                    logger.info('Rule %s matched signal %s', rule.rule_id, signal.signal_type)
                    acts = await rule.execute(signal, context)
                    matching_actions.extend(acts)
                else:
                    logger.debug('Rule %s skipped for signal %s', rule.rule_id, signal.signal_type)
            except Exception as exc:
                logger.exception('Rule %s failed: %s', rule.rule_id, exc)

        if matching_actions:
            await self._execute_actions(matching_actions, context)
        else:
            logger.debug('No actions produced for %s', signal.signal_type)

    async def _execute_actions(self, actions: List[Action], context: RuleContext) -> None:
        for action in actions:
            if isinstance(action, TriggerComponentAction):
                await self._handle_trigger_component(action, context)
            elif isinstance(action, EmitSignalAction):
                await self._handle_emit_signal(action, context)
            else:
                logger.warning('Unknown Action type: %s', type(action).__name__)

    async def _handle_trigger_component(self, action: TriggerComponentAction, context: RuleContext) -> None:
        component_id_to_trigger = action.component_id
        logger.info('Rule requests triggering component %s (template=%s)', component_id_to_trigger, action.template_id)
        component = None
        try:
            component = self.registry.get(component_id_to_trigger)
        except ComponentRegistryMissingError as e:
            logger.warning("Component '%s' not found. Attempting to resolve common aliases.", component_id_to_trigger)
            potential_aliases = []
            if component_id_to_trigger.endswith("_primary") or component_id_to_trigger.endswith("_main") or component_id_to_trigger.endswith("_default"):
                 potential_aliases.append(component_id_to_trigger.rsplit('_', 1)[0])

            for alias in potential_aliases:
                try:
                    component = self.registry.get(alias)
                    if component:
                        logger.info("ALIAS RESOLVED: Found component '%s' for rule target '%s'. Proceeding.", alias, component_id_to_trigger)
                        component_id_to_trigger = alias
                        break
                except ComponentRegistryMissingError:
                    continue
            
            if not component:
                logger.error("TriggerComponentAction failed: %s", e, exc_info=True)
                return

        try:
            if component is None:
                raise ComponentRegistryMissingError(component_id_to_trigger)

            if not callable(getattr(component, 'process', None)):
                raise AttributeError(f"Component {component_id_to_trigger!r} lacks a 'process' coroutine")

            exec_context = NireonExecutionContext(
                run_id=context.run_id,
                component_id=component_id_to_trigger,
                component_registry=self.registry,
                event_bus=self.registry.get_service_instance(EventBusPort),
                logger=context.logger,
                metadata={'source_rule_id': context.signal.source_node_id, 'triggering_signal_type': context.signal.signal_type}
            )

            kwargs = {'data': action.input_data, 'context': exec_context}
            if action.template_id:
                kwargs['template_id'] = action.template_id

            await component.process(**kwargs)

        except Exception as exc:
            logger.exception('TriggerComponentAction failed: %s', exc)

    async def _handle_emit_signal(self, action: EmitSignalAction, context: RuleContext) -> None:
        logger.info('Emitting signal %s (depth=%d)', action.signal_type, context.recursion_depth + 1)
        try:
            from signals.base import EpistemicSignal
            new_signal = EpistemicSignal(
                signal_type=action.signal_type,
                source_node_id=action.source_node_id_override or context.signal.source_node_id,
                payload=action.payload
            )
            
            # FIX: Publish the signal to the external event bus so listeners (like the test runner) can hear it.
            if self._event_bus:
                logger.debug(f"Publishing emitted signal '{new_signal.signal_type}' to external event bus.")
                self._event_bus.publish(new_signal.signal_type, new_signal)
            else:
                logger.warning(f"Cannot publish emitted signal '{new_signal.signal_type}' externally: event bus not available.")

            # Also process it internally for chaining rules.
            await self.process_signal(new_signal, _depth=context.recursion_depth + 1)

        except Exception as exc:
            logger.exception('EmitSignalAction failed: %s', exc)