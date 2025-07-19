# core_rules.py
from __future__ import annotations
import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from .expressions.rel_engine import RELEngine
from .models import EmitSignalAction, TriggerComponentAction, Action
from .protocols import ReactorRule

if TYPE_CHECKING:
    from signals.base import EpistemicSignal
    from .models import RuleContext

logger = logging.getLogger(__name__)

_PATTERN = re.compile('\\{\\{\\s*(.+?)\\s*\\}\\}')

def _resolve_path(path: str, signal: 'EpistemicSignal') -> Any:
    obj: Any = signal
    for part in path.split('.'):
        if part == 'payload':
            # Handle both dict and object payloads
            obj = getattr(signal, 'payload', None)
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

def _substitute(template: Any, signal: 'EpistemicSignal') -> Any:
    if isinstance(template, str):
        stripped = template.strip()
        if _PATTERN.fullmatch(stripped):
            expr = stripped[2:-2].strip()
            return _resolve_path(expr, signal)
        def repl(match: re.Match[str]) -> str:
            expr = match.group(1).strip()
            # Simple math support for expressions like {{ value + 1 }}
            if '+' in expr:
                parts = [p.strip() for p in expr.split('+')]
                try:
                    vals = [_resolve_path(p, signal) for p in parts]
                    if all((isinstance(v, (int, float)) for v in vals)):
                        return str(sum(vals))
                except Exception:
                    pass
            resolved = _resolve_path(expr, signal)
            return str(resolved) if resolved is not None else ''

        return _PATTERN.sub(repl, template)

    if isinstance(template, dict):
        return {k: _substitute(v, signal) for k, v in template.items()}
    
    return template


class ConditionalRule(ReactorRule):
    def __init__(self, *, rule_id: str, signal_type: str, conditions: Optional[List[Dict[str, Any]]]=None, actions_on_match: Optional[List[Dict[str, Any]]]=None, priority: int=100, namespace: str='core') -> None:
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

        for i, cond in enumerate(self.conditions):
            # The 'signal_type_match' is implicitly handled above, so we can skip it here.
            if cond.get('type') == 'signal_type_match':
                continue
            
            if not await self._evaluate_condition(cond, signal, context):
                # For debugging: log which condition failed
                # logger.debug(f"Rule '{self.rule_id}' condition #{i+1} failed: {cond}")
                return False
        
        return True

    async def execute(self, signal: 'EpistemicSignal', context: 'RuleContext') -> List[Action]:
        return [act for cfg in self.actions_on_match if (act := self._create_action(cfg, signal, context))]

    async def _evaluate_condition(self, condition: Dict[str, Any], signal: 'EpistemicSignal', context: 'RuleContext') -> bool:
        if condition.get('type') != 'payload_expression':
            logger.warning('Rule %s has unknown condition type %s', self.rule_id, condition.get('type'))
            return False

        expression = condition.get('expression', '')
        if not expression:
            logger.warning('Rule %s has empty expression', self.rule_id)
            return False
            
        # Prepare a safe context for expression evaluation
        payload_for_context = signal.payload
        if hasattr(signal.payload, 'model_dump'):
            payload_for_context = signal.payload.model_dump()
        elif hasattr(signal.payload, 'dict'):
             payload_for_context = signal.payload.dict()
        
        # Wrapper to allow attribute access on dicts
        class PayloadWrapper:
            def __init__(self, data):
                self._data = data
                for key, value in data.items():
                    setattr(self, key, value)
            def get(self, key, default=None):
                return self._data.get(key, default)

        wrapped_payload = payload_for_context
        if isinstance(payload_for_context, dict):
             wrapped_payload = PayloadWrapper(payload_for_context)

        ctx = {
            'signal': signal,
            'context': context,
            'payload': wrapped_payload,
            'metadata': getattr(signal, 'metadata', {}),
            'now': datetime.now(timezone.utc),
            'trust_score': getattr(signal, 'trust_score', None),
            'novelty_score': getattr(signal, 'novelty_score', None),
            'confidence': getattr(signal, 'confidence', None),
            'source_node_id': signal.source_node_id,
            'signal_id': signal.signal_id,
            'parent_signal_ids': signal.parent_signal_ids,
            'exists': lambda x: x is not None,
            'lower': lambda s: s.lower() if isinstance(s, str) else s,
        }

        try:
            result = bool(self._rel.evaluate(expression, ctx))
            return result
        except Exception as exc:
            logger.exception('Rule %s REL error: %s', self.rule_id, exc)
            return False

    def _create_action(self, cfg: Dict[str, Any], signal: 'EpistemicSignal', _: 'RuleContext') -> Optional[Action]:
        act_type = cfg.get('type')
        if act_type == 'trigger_component':
            component_id = cfg['component_id']
            if isinstance(component_id, str) and '{{' in component_id:
                component_id = _substitute(component_id, signal)
            
            template_id = cfg.get('template_id')
            if template_id and isinstance(template_id, str) and ('{{' in template_id):
                template_id = _substitute(template_id, signal)

            data = self._build_input_data(cfg, signal)
            return TriggerComponentAction(component_id=component_id, template_id=template_id, input_data=data)
        
        if act_type == 'emit_signal':
            if 'payload' in cfg:
                payload = _substitute(cfg.get('payload', {}), signal)
            else:
                # Fallback for old rule format
                payload = {}
                payload_keys = set(cfg.keys()) - {'type', 'signal_type', 'source_node_id_override'}
                for key in payload_keys:
                    payload[key] = _substitute(cfg.get(key), signal)
            
            source_override = cfg.get('source_node_id_override')
            if source_override and isinstance(source_override, str) and ('{{' in source_override):
                source_override = _substitute(source_override, signal)
            
            return EmitSignalAction(signal_type=cfg['signal_type'], payload=payload, source_node_id_override=source_override)
        
        logger.warning('Rule %s unknown action type %s', self.rule_id, act_type)
        return None

    def _build_input_data(self, cfg: Dict[str, Any], signal: 'EpistemicSignal') -> Dict[str, Any]:
        data = {}

        # Robust template handling
        if 'input_data' in cfg:
            raw_input = cfg['input_data']
            if isinstance(raw_input, str):
                substituted = _substitute(raw_input, signal)
            else:
                substituted = raw_input
            if isinstance(substituted, dict):
                data.update(substituted)
            elif isinstance(substituted, str):
                data['payload'] = substituted  # Wrap strings to avoid dict() error
            else:
                data['input'] = substituted
        
        # Handle the new 'input_data_mapping' which is more explicit
        for target, source in cfg.get('input_data_mapping', {}).items():
            # Special case: if the source is the literal string 'signal',
            # pass the entire signal object.
            if source == 'signal':
                data[target] = signal
                continue

            # Existing logic for path resolution
            if isinstance(source, str):
                resolved = _resolve_path(source, signal)
                if resolved is not None:
                    data[target] = resolved
            else:
                # Pass through non-string values directly (e.g., booleans, numbers)
                data[target] = source
        
        # Backwards compatibility for the flawed `input_data` with templates
        # This part has the bug. We keep it for old rules but the new mapping is preferred.
        for key, value in data.items():
            if isinstance(value, str) and '{{' in value:
                # This logic is flawed because it doesn't handle passing the whole object.
                # The fix above in `input_data_mapping` is the correct way forward.
                data[key] = _substitute(value, signal)

        return data

class SignalTypeMatchRule(ReactorRule):
    def __init__(self, rule_id: str, signal_type_to_match: str, component_id_to_trigger: str, *, input_data: Optional[Dict[str, Any]]=None) -> None:
        self.rule_id = rule_id
        self.signal_type_to_match = signal_type_to_match
        self.component_id_to_trigger = component_id_to_trigger
        self.input_data = input_data or {}
        self.priority = 100
        self.namespace = 'core'

    async def matches(self, signal: 'EpistemicSignal', _: 'RuleContext') -> bool:
        return signal.signal_type == self.signal_type_to_match

    async def execute(self, signal: 'EpistemicSignal', _: 'RuleContext') -> List[Action]:
        return [TriggerComponentAction(component_id=self.component_id_to_trigger, input_data=self.input_data)]