# FILE: nireon_v4/reactor/rules/core_rules.py

from __future__ import annotations
from typing import List, Dict, Any, Optional, TYPE_CHECKING
import logging
from datetime import datetime, timezone

from ..protocols import ReactorRule
from ..models import TriggerComponentAction, EmitSignalAction, Action
from ..expressions.rel_engine import RELEngine

if TYPE_CHECKING:
    from signals import EpistemicSignal
    from ..models import RuleContext

logger = logging.getLogger(__name__)

class ConditionalRule(ReactorRule):
    """
    A declarative-style rule that evaluates a list of conditions
    before executing a list of configured actions.
    This acts as the Python representation of a YAML rule.
    """
    def __init__(
        self,
        rule_id: str,
        signal_type: str,
        conditions: Optional[List[Dict[str, Any]]] = None,
        actions_on_match: Optional[List[Dict[str, Any]]] = None,
        priority: int = 100,
        namespace: str = 'core'
    ):
        self.rule_id = rule_id
        self.signal_type = signal_type
        self.conditions = conditions or []
        self.actions_on_match = actions_on_match or []
        self.priority = priority
        self.namespace = namespace
        self._rel_engine = RELEngine()  # Initialize REL engine
    
    async def matches(self, signal: 'EpistemicSignal', context: 'RuleContext') -> bool:
        """Check if signal matches the primary type and all other conditions."""
        # The primary signal_type match is handled here for efficiency.
        if signal.signal_type != self.signal_type:
            return False
        
        # Evaluate all other conditions. We filter out the primary type match.
        for condition in self.conditions:
            if condition.get('type') == 'signal_type_match':
                continue # Already checked
            if not await self._evaluate_condition(condition, signal, context):
                return False
        
        return True
    
    async def execute(self, signal: 'EpistemicSignal', context: 'RuleContext') -> List[Action]:
        """Execute configured actions."""
        actions: List[Action] = []
        for action_config in self.actions_on_match:
            action = self._create_action(action_config, signal, context)
            if action:
                actions.append(action)
        return actions
    
    async def _evaluate_condition(self, condition: Dict[str, Any], signal: 'EpistemicSignal', context: 'RuleContext') -> bool:
        """
        Evaluates a single condition.
        Supports:
        - payload_expression: Uses REL (Rule Expression Language) for complex expressions
        """
        condition_type = condition.get('type')
        
        if condition_type == 'payload_expression':
            expression = condition.get('expression', '')
            if not expression:
                logger.warning(f"Rule '{self.rule_id}' has empty payload_expression")
                return False
            
            try:
                # Build evaluation context with all available data
                eval_context = {
                    'signal': signal,
                    'context': context,
                    'payload': signal.payload,
                    'metadata': getattr(signal, 'metadata', {}),
                    'now': datetime.now(timezone.utc),
                    # Add signal attributes for easier access
                    'trust_score': getattr(signal, 'trust_score', None),
                    'novelty_score': getattr(signal, 'novelty_score', None),
                    'confidence': getattr(signal, 'confidence', None),
                    'source_node_id': signal.source_node_id,
                    'signal_id': signal.signal_id,
                    'parent_signal_ids': signal.parent_signal_ids,
                }
                
                # Evaluate the expression
                result = self._rel_engine.evaluate(expression, eval_context)
                
                # Log evaluation for debugging
                logger.debug(f"Rule '{self.rule_id}' evaluated expression '{expression}' -> {result}")
                
                # Ensure boolean result
                return bool(result)
                
            except Exception as e:
                logger.error(f"Error evaluating REL expression in rule '{self.rule_id}': {e}")
                # In case of error, we fail closed (condition doesn't match)
                return False
        
        # For any other unknown condition type, log warning and fail closed
        logger.warning(f"Unknown condition type '{condition_type}' in rule '{self.rule_id}'")
        return False
    
    def _create_action(self, action_config: Dict[str, Any], signal: 'EpistemicSignal', context: 'RuleContext') -> Optional[Action]:
        """Create an action object from its dictionary configuration."""
        action_type = action_config.get('type')
        
        if action_type == 'trigger_component':
            # Handle both static input_data and dynamic input_data_mapping
            input_data = action_config.get('input_data', {}).copy()
            
            # Apply any dynamic mappings
            input_data_mapping = action_config.get('input_data_mapping', {})
            for target_key, source_path in input_data_mapping.items():
                try:
                    # Simple path resolution (could be enhanced)
                    value = self._resolve_path(source_path, signal)
                    if value is not None:
                        input_data[target_key] = value
                except Exception as e:
                    logger.warning(f"Failed to resolve mapping '{source_path}' -> '{target_key}': {e}")
            
            return TriggerComponentAction(
                component_id=action_config.get('component_id', ''),
                template_id=action_config.get('template_id'),
                input_data=input_data
            )
            
        elif action_type == 'emit_signal':
            # Process payload to substitute any template variables
            payload = self._process_payload_template(
                action_config.get('payload', {}), 
                signal
            )
            
            return EmitSignalAction(
                signal_type=action_config.get('signal_type', ''),
                payload=payload,
                source_node_id_override=action_config.get('source_node_id_override')
            )
        
        logger.warning(f"Unknown action type '{action_type}' in rule '{self.rule_id}'")
        return None
    
    def _resolve_path(self, path: str, signal: 'EpistemicSignal') -> Any:
        """
        Resolve a dot-notation path against the signal.
        Examples: "payload.idea_content", "signal.trust_score"
        """
        parts = path.split('.')
        obj = signal
        
        for part in parts:
            if part == 'payload':
                obj = signal.payload
            elif part == 'signal':
                obj = signal
            elif hasattr(obj, part):
                obj = getattr(obj, part)
            elif isinstance(obj, dict) and part in obj:
                obj = obj[part]
            else:
                logger.debug(f"Could not resolve path '{path}' at part '{part}'")
                return None
        
        return obj
    
    def _process_payload_template(self, payload: Dict[str, Any], signal: 'EpistemicSignal') -> Dict[str, Any]:
        """
        Process payload dictionary to replace template variables.
        Supports: {{ signal.field_name }} syntax
        """
        import re
        
        processed = {}
        template_pattern = re.compile(r'\{\{\s*(.+?)\s*\}\}')
        
        for key, value in payload.items():
            if isinstance(value, str):
                # Look for template variables
                matches = template_pattern.findall(value)
                for match in matches:
                    resolved = self._resolve_path(match, signal)
                    if resolved is not None:
                        value = value.replace(f'{{{{ {match} }}}}', str(resolved))
                processed[key] = value
            elif isinstance(value, dict):
                # Recursively process nested dictionaries
                processed[key] = self._process_payload_template(value, signal)
            else:
                processed[key] = value
        
        return processed


class SignalTypeMatchRule(ReactorRule):
    """
    Simple rule that matches a specific signal type and triggers a component.
    This is a convenience class for backwards compatibility.
    """
    def __init__(
        self, 
        rule_id: str, 
        signal_type_to_match: str, 
        component_id_to_trigger: str,
        input_data: Optional[Dict[str, Any]] = None
    ):
        self.rule_id = rule_id
        self.signal_type_to_match = signal_type_to_match
        self.component_id_to_trigger = component_id_to_trigger
        self.input_data = input_data or {}
        self.priority = 100  # Default priority
        self.namespace = 'core'  # Default namespace
    
    async def matches(self, signal: 'EpistemicSignal', context: 'RuleContext') -> bool:
        """Check if the signal type matches."""
        return signal.signal_type == self.signal_type_to_match
    
    async def execute(self, signal: 'EpistemicSignal', context: 'RuleContext') -> List[Action]:
        """Return action to trigger the configured component."""
        return [
            TriggerComponentAction(
                component_id=self.component_id_to_trigger,
                input_data=self.input_data
            )
        ]