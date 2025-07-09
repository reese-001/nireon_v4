from __future__ import annotations
import asyncio
import hashlib
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Final, Iterable, List, Optional, TYPE_CHECKING
from core.lifecycle import ComponentRegistryMissingError
from core.registry import ComponentRegistry
from core.results import ProcessResult
from domain.context import NireonExecutionContext
from domain.ports.event_bus_port import EventBusPort
from signals import _signal_class_map, EpistemicSignal
from signals.core import TrustAssessmentSignal, ProtoTaskSignal, GenerativeLoopFinishedSignal, BranchCompletionStatus

from ..models import Action, EmitSignalAction, RuleContext, TriggerComponentAction
from ..protocols import ReactorRule

if TYPE_CHECKING:
    from signals.base import EpistemicSignal

logger = logging.getLogger(__name__)
_RECURSION_LIMIT: Final = 25


class MainReactorEngine:
    def __init__(self, registry: ComponentRegistry, *, rules: Optional[Iterable[ReactorRule]]=None, max_recursion_depth: int=10, loop: Optional[asyncio.AbstractEventLoop]=None) -> None:
        self.registry = registry
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
        logger.debug('REACTOR RECEIVED SIGNAL:\n%s', signal.model_dump_json(indent=2))

        # Add diagnostic for IdeaGeneratedSignal
        if signal.signal_type == 'IdeaGeneratedSignal' and _depth == 0:
            self.diagnose_rules_for_signal(signal)
        
        if _depth >= self.max_recursion_depth:
            logger.error('Recursion limit (%d) exceeded for signal %s', self.max_recursion_depth, signal.signal_type)
            return
        
        run_id = getattr(signal, 'run_id', 'unknown_run')
        context = RuleContext(signal=signal, run_id=run_id, component_registry=self.registry, logger=logger, recursion_depth=_depth)
        
        logger.info('[Depth:%d] Processing signal %s from %s', _depth, signal.signal_type, signal.source_node_id)
        
        matched_actions: List[Action] = []
        
        for rule in self.rules:
            try:
                is_match, reason = await self._evaluate_rule(rule, signal, context)
                if is_match:
                    logger.info("✅ Rule '%s' MATCHED signal '%s'.", rule.rule_id, signal.signal_type)
                    actions = await rule.execute(signal, context)
                    matched_actions.extend(actions)
                else:
                    logger.debug("  Rule '%s' skipped for signal '%s'. Reason: %s", rule.rule_id, signal.signal_type, reason)
            except Exception as exc:
                logger.exception("Rule '%s' failed during evaluation: %s", rule.rule_id, exc)
        
        if matched_actions:
            logger.info('Found %d action(s) to execute from matched rules.', len(matched_actions))
            await self._execute_actions(matched_actions, context)
        else:
            logger.debug('No rules matched signal %s', signal.signal_type)

    async def _evaluate_rule(self, rule: ReactorRule, signal: 'EpistemicSignal', context: 'RuleContext') -> tuple[bool, str]:
        """Evaluates a rule and returns a tuple of (bool, reason_string)."""
        # First, delegate to the rule's matches() method if available
        # This allows rules to implement their own matching logic and logging
        try:
            matches = await rule.matches(signal, context)
            if matches:
                return (True, 'Rule matches() returned True')
            else:
                # Try to provide a more specific reason if we can
                if hasattr(rule, 'signal_type') and signal.signal_type != rule.signal_type:
                    return (False, f"Signal type mismatch (Rule wants '{getattr(rule, 'signal_type', 'N/A')}', got '{signal.signal_type}')")
                return (False, 'Rule matches() returned False')
        except AttributeError:
            # If the rule doesn't have a matches() method, fall back to our internal logic
            logger.debug("Rule %s doesn't have matches() method, using internal evaluation", getattr(rule, 'rule_id', 'unknown'))
        
        # Fallback evaluation for rules without matches() method
        # First, check if the rule is interested in this signal type at all.
        if not hasattr(rule, 'signal_type') or signal.signal_type != rule.signal_type:
            return (False, f"Signal type mismatch (Rule wants '{getattr(rule, 'signal_type', 'N/A')}', got '{signal.signal_type}')")
        
        # If it's a simple rule without conditions, assume it matches
        if not hasattr(rule, 'conditions'):
            return (True, 'Simple rule without conditions')
        
        # For conditional rules, check each condition.
        for i, condition in enumerate(rule.conditions):
            cond_type = condition.get('type')
            
            if cond_type == 'signal_type_match':
                continue  # Already checked this
            
            if cond_type == 'payload_expression':
                expression = condition.get('expression', 'False')
                try:
                    from ..expressions.rel_engine import RELEngine
                    rel_engine = RELEngine()
                    
                    # Build context for REL evaluation
                    rel_context = {
                        'signal': signal, 
                        'context': context, 
                        'payload': signal.payload,
                        'metadata': getattr(signal, 'metadata', {}),
                        'now': datetime.now(timezone.utc),
                        'trust_score': getattr(signal, 'trust_score', None),
                        'novelty_score': getattr(signal, 'novelty_score', None),
                        'exists': lambda x: x is not None,
                        'lower': lambda s: s.lower() if isinstance(s, str) else s
                    }
                    
                    result = bool(rel_engine.evaluate(expression, rel_context))
                    
                    if not result:
                        return (False, f"Condition #{i + 1} (expression) failed: '{expression}' evaluated to False.")
                except Exception as e:
                    logger.error("REL evaluation failed for expression: %s", expression)
                    logger.error("Error details: %s", str(e), exc_info=True)
                    return (False, f'Condition #{i + 1} (expression) failed with error: {e}')
            else:
                return (False, f"Unknown condition type '{cond_type}' in rule '{rule.rule_id}'")
        
        return (True, 'All conditions met')

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
        logger.debug('=' * 80)
        logger.debug(f'REACTOR TRIGGERING COMPONENT: {comp_id}')
        logger.debug(f'  Template: {action.template_id}')
        logger.debug(f'  Input data: {action.input_data}')
        logger.debug('=' * 80)
        
        try:
            component = self.registry.get(comp_id)
            logger.info(f'  Found component: {type(component).__name__}')
        except ComponentRegistryMissingError as exc:
            logger.error(f"Component '{comp_id}' not found: {exc}")
            return
        
        if not callable(getattr(component, 'process', None)):
            logger.error(f"Component {comp_id} lacks an async 'process' method")
            return
        
        # Build execution context
        exec_metadata = {
            'source_rule_id': context.signal.signal_id,
            'triggering_signal_type': context.signal.signal_type
        }
        
        source_signal_frame_id = getattr(context.signal, 'context_tags', {}).get('frame_id')
        if source_signal_frame_id:
            exec_metadata['current_frame_id'] = source_signal_frame_id
            logger.debug(f"Propagating frame_id '{source_signal_frame_id}' to component '{comp_id}'.")
        
        exec_ctx = NireonExecutionContext(
            run_id=context.run_id,
            component_id=comp_id,
            component_registry=self.registry,
            event_bus=self._event_bus,
            logger=context.logger,
            metadata=exec_metadata
        )
        
        kwargs = {'data': action.input_data, 'context': exec_ctx}
        if action.template_id:
            kwargs['template_id'] = action.template_id
        
        pattern = getattr(component.metadata, 'interaction_pattern', 'unknown')
        logger.info(f'  Component pattern: {pattern}')
        
        if pattern == 'processor':
            logger.info(f"  >>> CALLING PROCESSOR {comp_id}.process() and awaiting result...")
            try:
                result = await component.process(**kwargs)
                logger.info(f"  <<< PROCESSOR {comp_id} RETURNED")
                logger.info(f"      Result type: {type(result).__name__}")
                logger.info(f"      Success: {getattr(result, 'success', 'N/A')}")
                
                if isinstance(result, ProcessResult):
                    logger.info(f"      Has output_data: {result.output_data is not None}")
                    if result.output_data and isinstance(result.output_data, dict):
                        logger.info(f"      Output type: {result.output_data.get('type')}")
                    
                    # Promote to signal
                    logger.info(f"  >>> PROMOTING ProcessResult to signal...")
                    await self._promote_result_to_signal(result, context.signal, context)
                    logger.info(f"  <<< PROMOTION COMPLETE")
                else:
                    logger.warning(f"Processor '{comp_id}' did not return a ProcessResult: {type(result)}")
            except Exception as exc:
                logger.exception(f'Processor {comp_id} failed: {exc}')
        else:
            logger.debug(f"Triggering {('Producer' if pattern == 'producer' else 'Component')} '{comp_id}' (fire-and-forget).")
            try:
                asyncio.create_task(component.process(**kwargs))
            except Exception as exc:
                logger.exception('Producer TriggerComponentAction failed: %s', exc)
        
        logger.info('=' * 80)

    async def _promote_result_to_signal(self, pr: ProcessResult, parent: 'EpistemicSignal', context: RuleContext) -> None:
        self._debug_processor_result(pr, parent)
        if not pr.success:
            logger.warning(f'Processor {pr.component_id} returned unsuccessful result. Not promoting. Message: {pr.message}')
            return
        
        data = pr.output_data
        if not isinstance(data, dict) or 'type' not in data:
            logger.warning(f"Processor {pr.component_id} returned invalid output_data (not a dict or missing 'type'). Not promoting.")
            return
            
        output_type = data.get('type')
        new_signal = None

        if output_type == 'trust_assessment':
            idea_id = data.get('idea_id')
            trust_score = data.get('trust_score')
            if idea_id is None or trust_score is None:
                logger.error(f'Cannot promote trust assessment - missing required fields. idea_id={idea_id}, trust_score={trust_score}')
                return
            
            axis_scores = data.get('axis_scores', {})
            novel_score_data = axis_scores.get('novel', {})
            novelty_score = None
            if isinstance(novel_score_data, dict) and 'score' in novel_score_data:
                novelty_score = float(novel_score_data['score']) / 10.0
            elif isinstance(novel_score_data, (int, float)):
                novelty_score = float(novel_score_data) / 10.0

            id_str = f'{idea_id}|{pr.component_id}|assessment'
            sid = f'sig_tas_{hashlib.sha256(id_str.encode()).hexdigest()[:24]}'

            signal_payload = {
                'idea_id': idea_id,
                'idea_text': data.get('idea_text', ''),
                'is_stable': data.get('is_stable', False),
                'rejection_reason': data.get('rejection_reason'),
                'assessment_details': data.get('assessment_details', {}),
                'idea_metadata': data.get('metadata', {}),
                '_assessment_id': idea_id,
                '_axis_count': len(axis_scores),
            }
            
            new_signal = TrustAssessmentSignal(
                signal_id=sid,
                source_node_id=pr.component_id,
                target_id=idea_id,
                target_type='Idea',
                trust_score=trust_score,
                novelty_score=novelty_score,
                assessment_rationale=data.get('rejection_reason'),
                payload=signal_payload,
                parent_signal_ids=[parent.signal_id],
                context_tags=getattr(parent, 'context_tags', {})
            )

            # --- THIS IS THE CORRECTED LOGGING STATEMENT ---
            # We format the novelty_score string BEFORE putting it in the f-string.
            novelty_str = f"{novelty_score:.2f}" if novelty_score is not None else "N/A"
            logger.info(f"Created TrustAssessmentSignal for idea '{idea_id}' with trust_score={trust_score:.2f}, novelty_score={novelty_str}")
            # --- END OF CORRECTION ---

        elif output_type in ['quantification_complete', 'quantification_triggered']:
            signal_data = self._build_completion_payload(data)
            
            constructor_args = {
                'source_node_id': pr.component_id,
                'payload': signal_data.get('payload', {}),
                'parent_signal_ids': [parent.signal_id],
                'context_tags': getattr(parent, 'context_tags', {})
            }

            specifics = signal_data.get('specific_fields', {})
            if 'completion_status' in specifics:
                constructor_args['completion_status'] = specifics['completion_status']
            if 'completion_reason' in specifics:
                constructor_args['completion_reason'] = specifics['completion_reason']

            try:
                new_signal = GenerativeLoopFinishedSignal(**constructor_args)
            except Exception as e:
                logger.error(f"FATAL: Failed to construct GenerativeLoopFinishedSignal: {e}", exc_info=True)
                logger.error(f"Constructor data that failed validation: {constructor_args}")
                new_signal = None

        elif output_type == 'proto_task':
            new_signal = ProtoTaskSignal(source_node_id=pr.component_id, payload=data, parent_signal_ids=[parent.signal_id])

        if new_signal:
            if self._event_bus:
                logger.info(f"Promoting ProcessResult from '{pr.component_id}' to {new_signal.signal_type} '{new_signal.signal_id}'")
                self._event_bus.publish(new_signal.signal_type, new_signal)
                await self.process_signal(new_signal, _depth=context.recursion_depth + 1)
            else:
                logger.warning(f'Event bus not available. Cannot publish promoted signal {new_signal.signal_type}.')
        else:
            logger.debug(f"No signal promotion for output type '{output_type}' from processor '{pr.component_id}'")

    def _build_completion_payload(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build payload for GenerativeLoopFinishedSignal from processor output."""
        assessment_details = data.get('assessment_details', {})
        metadata = assessment_details.get('metadata', {})
        
        # The old payload part
        payload = {
            'status': 'completed_one_branch',
            'final_idea_id': data.get('idea_id', assessment_details.get('idea_id')),
            'final_trust_score': data.get('trust_score', assessment_details.get('trust_score')),
            'final_depth': metadata.get('depth', 0),
            'quantifier_triggered': data.get('quantified', False)  # Keep for backward compatibility
        }
        
        # The new, specific fields
        specific_fields = {
            'completion_status': data.get('completion_status', 'continue'),  # Default to continue
            'completion_reason': data.get('completion_reason')
        }
        
        return {'payload': payload, 'specific_fields': specific_fields}

    async def _handle_emit_signal(self, action: EmitSignalAction, context: RuleContext) -> None:
        logger.info('Emitting signal %s (depth %d)', action.signal_type, context.recursion_depth + 1)
        
        SignalClass = _signal_class_map.get(action.signal_type, EpistemicSignal)
        
        constructor_data = action.payload.copy()
        constructor_data['signal_type'] = action.signal_type
        constructor_data['source_node_id'] = action.source_node_id_override or context.signal.source_node_id
        
        try:
            new_sig = SignalClass(**constructor_data)
        except Exception as e:
            logger.error(f"Failed to construct signal '{action.signal_type}': {e}", exc_info=True)
            logger.error(f"Constructor data that failed validation: {constructor_data}")
            return

        if self._event_bus:
            self._event_bus.publish(new_sig.signal_type, new_sig)
        else:
            logger.debug('Event bus unavailable – signal not broadcast externally')
            
        await self.process_signal(new_sig, _depth=context.recursion_depth + 1)

    def _debug_processor_result(self, pr: ProcessResult, parent: 'EpistemicSignal') -> None:
        """Debug helper to trace processor results"""
        logger.info("=" * 80)
        logger.info(f"REACTOR RECEIVED ProcessResult from '{pr.component_id}'")
        logger.info(f"  Success: {pr.success}")
        logger.info(f"  Message: {pr.message}")
        
        if pr.output_data:
            logger.debug(f"  Output data type: {type(pr.output_data)}")
            if isinstance(pr.output_data, dict):
                logger.debug(f"  Output data keys: {list(pr.output_data.keys())}")
                output_type = pr.output_data.get('type')
                logger.debug(f"  Output type: '{output_type}'")
                
                if output_type == 'trust_assessment':
                    logger.debug("  TRUST ASSESSMENT DATA:")
                    logger.debug(f"    idea_id: {pr.output_data.get('idea_id')}")
                    logger.debug(f"    trust_score: {pr.output_data.get('trust_score')}")
                    logger.debug(f"    is_stable: {pr.output_data.get('is_stable')}")
                    logger.debug(f"    idea_text length: {len(pr.output_data.get('idea_text', ''))}")
                    logger.debug(f"    axis_scores: {pr.output_data.get('axis_scores', {})}")
        else:
            logger.info("  Output data: None")
        
        logger.debug(f"  Parent signal: {parent.signal_type} (id={parent.signal_id})")
        logger.debug("=" * 80)

    def diagnose_rules_for_signal(self, signal: 'EpistemicSignal') -> None:
        """Diagnostic method to show which rules match a signal"""
        logger.debug("=" * 80)
        logger.debug(f"RULE DIAGNOSTIC for signal type: {signal.signal_type}")
        logger.debug(f"Total rules loaded: {len(self.rules)}")
        logger.debug("=" * 80)
        
        for i, rule in enumerate(self.rules):
            logger.debug(f"\nRule {i+1}:")
            logger.debug(f"  ID: {rule.rule_id}")
            logger.debug(f"  Priority: {getattr(rule, 'priority', 'N/A')}")
            logger.debug(f"  Signal type: {getattr(rule, 'signal_type', 'N/A')}")
            
            # Check if signal types match
            rule_signal_type = getattr(rule, 'signal_type', None)
            if rule_signal_type:
                if rule_signal_type == signal.signal_type:
                    logger.debug(f"  ✓ Signal type MATCHES")
                else:
                    logger.debug(f"  ✗ Signal type mismatch (wants '{rule_signal_type}')")
            
            # Show conditions
            if hasattr(rule, 'conditions'):
                logger.debug(f"  Conditions: {len(rule.conditions)}")
                for j, cond in enumerate(rule.conditions):
                    logger.info(f"    {j+1}. Type: {cond.get('type')}")
                    if cond.get('type') == 'payload_expression':
                        logger.info(f"       Expression: {cond.get('expression')}")
            
            # Show actions
            if hasattr(rule, 'actions_on_match'):
                logger.debug(f"  Actions: {len(rule.actions_on_match)}")
                for j, act in enumerate(rule.actions_on_match):
                    logger.debug(f"    {j+1}. Type: {act.get('type')}")
                    if act.get('type') == 'trigger_component':
                        logger.debug(f"       Component: {act.get('component_id')}")
        
        logger.info("=" * 80)