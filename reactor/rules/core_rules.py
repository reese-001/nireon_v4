# nireon_v4\reactor\rules\core_rules.py
from __future__ import annotations
import logging
import re
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, TYPE_CHECKING
from ..protocols import ReactorRule
from ..models import TriggerComponentAction, EmitSignalAction, Action
from ..expressions.rel_engine import RELEngine
if TYPE_CHECKING:
    from signals.base import EpistemicSignal
    from ..models import RuleContext
logger = logging.getLogger(__name__)


class ConditionalRule(ReactorRule):
    def __init__(self, *,
                 rule_id: str,
                 signal_type: str,
                 conditions: Optional[List[Dict[str, Any]]] = None,
                 actions_on_match: Optional[List[Dict[str, Any]]] = None,
                 priority: int = 100,
                 namespace: str = 'core') -> None:
        self.rule_id = rule_id
        self.signal_type = signal_type
        self.conditions = conditions or []
        self.actions_on_match = actions_on_match or []
        self.priority = priority
        self.namespace = namespace
        self._rel = RELEngine()

    async def matches(self, signal: 'EpistemicSignal', context: 'RuleContext') -> bool:
        if signal.signal_type != self.signal_type:
            return False

        for cond in self.conditions:
            if cond.get('type') == 'signal_type_match':
                continue
            if not await self._evaluate_condition(cond, signal, context):
                return False

        return True

    async def execute(self, signal: 'EpistemicSignal', context: 'RuleContext') -> List[Action]:
        return [
            a for cfg in self.actions_on_match if (
                a := self._create_action(cfg, signal, context)
            )
        ]

    async def _evaluate_condition(
        self, condition: Dict[str, Any], signal: 'EpistemicSignal', context: 'RuleContext'
    ) -> bool:
        match condition.get('type'):
            case 'payload_expression':
                expression = condition.get('expression', '')
                if not expression:
                    logger.warning('Rule %s has empty expression', self.rule_id)
                    return False

                try:
                    ctx = {
                        'signal': signal,
                        'context': context,
                        'payload': signal.payload,
                        'metadata': getattr(signal, 'metadata', {}),
                        'now': datetime.now(timezone.utc),
                        'trust_score': getattr(signal, 'trust_score', None),
                        'novelty_score': getattr(signal, 'novelty_score', None),
                        'confidence': getattr(signal, 'confidence', None),
                        'source_node_id': signal.source_node_id,
                        'signal_id': signal.signal_id,
                        'parent_signal_ids': signal.parent_signal_ids,
                    }
                    return bool(self._rel.evaluate(expression, ctx))
                except Exception as exc:
                    logger.exception('Rule %s REL error: %s', self.rule_id, exc)
                    return False
            case _:
                logger.warning('Rule %s has unknown condition type %s', self.rule_id, condition.get('type'))
                return False

    def _create_action(self, cfg: Dict[str, Any], signal: 'EpistemicSignal', context: 'RuleContext') -> Optional[Action]:
        act_type = cfg.get('type')
        if act_type == 'trigger_component':
            data = self._build_input_data(cfg, signal)
            return TriggerComponentAction(
                component_id=cfg['component_id'],
                template_id=cfg.get('template_id'),
                input_data=data
            )
        if act_type == 'emit_signal':
            payload = self._process_payload_template(cfg.get('payload', {}), signal)
            return EmitSignalAction(
                signal_type=cfg['signal_type'],
                payload=payload,
                source_node_id_override=cfg.get('source_node_id_override')
            )

        logger.warning('Rule %s unknown action type %s', self.rule_id, act_type)
        return None

    def _build_input_data(self, cfg: Dict[str, Any], signal: 'EpistemicSignal') -> Dict[str, Any]:
        data = dict(cfg.get('input_data', {}))
        for target, source in cfg.get('input_data_mapping', {}).items():
            # If the source is a string, treat it as a path to resolve
            if isinstance(source, str):
                resolved = self._resolve_path(source, signal)
                if resolved is not None:
                    data[target] = resolved
            # Otherwise, pass the literal value (e.g., a dict, number)
            else:
                data[target] = source
        return data

    @staticmethod
    def _resolve_path(path: str, signal: 'EpistemicSignal') -> Any:
        obj: Any = signal
        for part in path.split("."):
            if part == 'payload':
                obj = signal.payload
            elif part == 'signal':
                obj = signal
            elif isinstance(obj, dict) and part in obj:
                obj = obj[part]
            elif hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                logger.debug('Path %s could not be resolved (part=%s)', path, part)
                return None
        return obj

    def _process_payload_template(self, template: Dict[str, Any], signal: 'EpistemicSignal') -> Dict[str, Any]:
        pattern = re.compile(r'\{\{\s*(.+?)\s*\}\}')
        
        def substitute(val: Any) -> Any:
            if isinstance(val, str):
                # This is a simple substitution. A full template engine (like Jinja2) would be better for expressions.
                # For now, we handle simple additions like {{ a + 1 }}
                def repl(match):
                    expr = match.group(1).strip()
                    if '+' in expr:
                        parts = [p.strip() for p in expr.split('+')]
                        try:
                            resolved_vals = [self._resolve_path(p, signal) for p in parts]
                            if all(isinstance(v, (int, float)) for v in resolved_vals):
                                return str(sum(resolved_vals))
                        except Exception:
                            pass # Fallback to simple path resolution
                    
                    resolved = self._resolve_path(expr, signal)
                    return str(resolved) if resolved is not None else ''

                return pattern.sub(repl, val)

            if isinstance(val, dict):
                return {k: substitute(v) for k, v in val.items()}
            return val
        
        return {k: substitute(v) for k, v in template.items()}

class SignalTypeMatchRule(ReactorRule):
    def __init__(self,
                 rule_id: str,
                 signal_type_to_match: str,
                 component_id_to_trigger: str,
                 *,
                 input_data: Optional[Dict[str, Any]] = None) -> None:
        self.rule_id = rule_id
        self.signal_type_to_match = signal_type_to_match
        self.component_id_to_trigger = component_id_to_trigger
        self.input_data = input_data or {}
        self.priority = 100
        self.namespace = 'core'

    async def matches(self, signal: 'EpistemicSignal', context: 'RuleContext') -> bool:
        return signal.signal_type == self.signal_type_to_match

    async def execute(self, signal: 'EpistemicSignal', context: 'RuleContext') -> List[Action]:
        # The input data for this simple rule is static, so no need for mapping
        # It's passed directly to the component.
        return [
            TriggerComponentAction(
                component_id=self.component_id_to_trigger,
                input_data=self.input_data
            )
        ]