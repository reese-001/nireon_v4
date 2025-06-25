# reactor/core_rules.py
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


# --------------------------------------------------------------------- #
#  Helper functions
# --------------------------------------------------------------------- #
_PATTERN = re.compile(r"\{\{\s*(.+?)\s*\}\}")


def _resolve_path(path: str, signal: "EpistemicSignal") -> Any:
    obj: Any = signal
    for part in path.split("."):
        if part == "payload":
            obj = signal.payload
        elif part == "signal":
            obj = signal
        elif isinstance(obj, dict) and part in obj:
            obj = obj[part]
        elif hasattr(obj, part):
            obj = getattr(obj, part)
        else:
            logger.debug("Path %s could not be resolved (part=%s)", path, part)
            return None
    return obj


def _substitute(template: Any, signal: "EpistemicSignal") -> Any:
    """Recursively substitute {{ path }} placeholders inside *template*."""
    if isinstance(template, str):

        def repl(match: re.Match[str]) -> str:
            expr = match.group(1).strip()
            if "+" in expr:  # simple summing expression  a + b
                parts = [p.strip() for p in expr.split("+")]
                try:
                    vals = [_resolve_path(p, signal) for p in parts]
                    if all(isinstance(v, (int, float)) for v in vals):
                        return str(sum(vals))
                except Exception:  # pragma: no cover
                    pass
            resolved = _resolve_path(expr, signal)
            return str(resolved) if resolved is not None else ""

        return _PATTERN.sub(repl, template)

    if isinstance(template, dict):
        return {k: _substitute(v, signal) for k, v in template.items()}

    return template


# --------------------------------------------------------------------- #
#  Core conditional rule
# --------------------------------------------------------------------- #
class ConditionalRule(ReactorRule):
    """Rule that matches a signal + optional REL conditions and fires actions."""

    def __init__(
        self,
        *,
        rule_id: str,
        signal_type: str,
        conditions: Optional[List[Dict[str, Any]]] = None,
        actions_on_match: Optional[List[Dict[str, Any]]] = None,
        priority: int = 100,
        namespace: str = "core",
    ) -> None:
        self.rule_id = rule_id
        self.signal_type = signal_type
        self.conditions = conditions or []
        self.actions_on_match = actions_on_match or []
        self.priority = priority
        self.namespace = namespace
        self._rel = RELEngine()

    # --------------- Protocol methods ---------------- #
    async def matches(self, signal: "EpistemicSignal", context: "RuleContext") -> bool:
        if signal.signal_type != self.signal_type:
            return False
        for cond in self.conditions:
            # Implicit signal_type_match already handled above
            if cond.get("type") == "signal_type_match":
                continue
            if not await self._evaluate_condition(cond, signal, context):
                return False
        return True

    async def execute(
        self, signal: "EpistemicSignal", context: "RuleContext"
    ) -> List[Action]:
        return [
            act
            for cfg in self.actions_on_match
            if (act := self._create_action(cfg, signal, context))
        ]

    # --------------- Internal helpers --------------- #
    async def _evaluate_condition(
        self,
        condition: Dict[str, Any],
        signal: "EpistemicSignal",
        context: "RuleContext",
    ) -> bool:
        if condition.get("type") != "payload_expression":
            logger.warning(
                "Rule %s has unknown condition type %s",
                self.rule_id,
                condition.get("type"),
            )
            return False

        expression = condition.get("expression", "")
        if not expression:
            logger.warning("Rule %s has empty expression", self.rule_id)
            return False

        ctx = {
            "signal": signal,
            "context": context,
            "payload": signal.payload,
            "metadata": getattr(signal, "metadata", {}),
            "now": datetime.now(timezone.utc),
            "trust_score": getattr(signal, "trust_score", None),
            "novelty_score": getattr(signal, "novelty_score", None),
            "confidence": getattr(signal, "confidence", None),
            "source_node_id": signal.source_node_id,
            "signal_id": signal.signal_id,
            "parent_signal_ids": signal.parent_signal_ids,
        }

        try:
            return bool(self._rel.evaluate(expression, ctx))
        except Exception as exc:
            logger.exception("Rule %s REL error: %s", self.rule_id, exc)
            return False

    # ----------------- action builders -------------- #
    def _create_action(self, cfg: Dict[str, Any], signal: 'EpistemicSignal', _: 'RuleContext') -> Optional[Action]:
        act_type = cfg.get('type')
        if act_type == 'trigger_component':
            # Fix: Substitute template variables in component_id
            component_id = cfg['component_id']
            if isinstance(component_id, str) and '{{' in component_id:
                component_id = _substitute(component_id, signal)
            
            # Also substitute template_id if present
            template_id = cfg.get('template_id')
            if template_id and isinstance(template_id, str) and '{{' in template_id:
                template_id = _substitute(template_id, signal)
            
            data = self._build_input_data(cfg, signal)
            return TriggerComponentAction(
                component_id=component_id, 
                template_id=template_id, 
                input_data=data
            )
        if act_type == 'emit_signal':
            if 'payload' in cfg:
                payload = _substitute(cfg.get('payload', {}), signal)
            else:
                payload = {}
                payload_keys = set(cfg.keys()) - {'type', 'signal_type', 'source_node_id_override'}
                for key in payload_keys:
                    payload[key] = _substitute(cfg.get(key), signal)
            
            # Also substitute source_node_id_override if it contains templates
            source_override = cfg.get('source_node_id_override')
            if source_override and isinstance(source_override, str) and '{{' in source_override:
                source_override = _substitute(source_override, signal)
                
            return EmitSignalAction(
                signal_type=cfg['signal_type'], 
                payload=payload, 
                source_node_id_override=source_override
            )
        logger.warning('Rule %s unknown action type %s', self.rule_id, act_type)
        return None

    # ------------------------------------------------ #
    def _build_input_data(self, cfg: Dict[str, Any], signal: "EpistemicSignal") -> Dict[str, Any]:
        data = dict(cfg.get("input_data", {}))
        for target, source in cfg.get("input_data_mapping", {}).items():
            if isinstance(source, str):
                resolved = _resolve_path(source, signal)
                if resolved is not None:
                    data[target] = resolved
            else:
                data[target] = source
        return data


# --------------------------------------------------------------------- #
#  Very simple rule variant
# --------------------------------------------------------------------- #
class SignalTypeMatchRule(ReactorRule):
    """Rule that triggers a component whenever *signal_type_to_match* is seen."""

    def __init__(
        self,
        rule_id: str,
        signal_type_to_match: str,
        component_id_to_trigger: str,
        *,
        input_data: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.rule_id = rule_id
        self.signal_type_to_match = signal_type_to_match
        self.component_id_to_trigger = component_id_to_trigger
        self.input_data = input_data or {}
        self.priority = 100
        self.namespace = "core"

    async def matches(self, signal: "EpistemicSignal", _: "RuleContext") -> bool:
        return signal.signal_type == self.signal_type_to_match

    async def execute(
        self, signal: "EpistemicSignal", _: "RuleContext"
    ) -> List[Action]:
        return [
            TriggerComponentAction(
                component_id=self.component_id_to_trigger, input_data=self.input_data
            )
        ]
