"""
Debug Helpers Module
Utilities for debugging and inspecting the NIREON system.
Located at: ./00_explorer_runner/debug_helpers.py
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from core.registry.component_registry import ComponentRegistry
from domain.ports.event_bus_port import EventBusPort


class DebugInspector:
    """Inspector for debugging NIREON components and configuration."""
    
    def __init__(self, registry: ComponentRegistry, event_bus: EventBusPort, logger: logging.Logger):
        self.registry = registry
        self.event_bus = event_bus
        self.logger = logger
        
    def check_reactor_rules(self):
        """Check and report on reactor rules configuration."""
        self.logger.info("=" * 80)
        self.logger.info("REACTOR RULES DEBUG CHECK")
        self.logger.info("=" * 80)
        
        try:
            from reactor.engine.base import ReactorEngine
            reactor_instance = self.registry.get_service_instance(ReactorEngine)
            
            if not reactor_instance or not hasattr(reactor_instance, 'rules'):
                self.logger.error("❌ Could not access reactor or reactor.rules!")
                return
                
            rules = reactor_instance.rules
            self.logger.info(f"✅ Total rules loaded: {len(rules)}")
            
            # Group rules by namespace
            namespace_count = {}
            for rule in rules:
                ns = getattr(rule, 'namespace', 'unknown')
                namespace_count[ns] = namespace_count.get(ns, 0) + 1
            
            self.logger.info("\nRules by namespace:")
            for ns, count in sorted(namespace_count.items()):
                self.logger.info(f"  - {ns}: {count} rules")
            
            # Check specific rule types
            self._check_specific_rules(rules)
            
            # Check critical components
            self._check_critical_components()
            
        except Exception as e:
            self.logger.error(f"Error checking reactor rules: {e}", exc_info=True)
        
        self.logger.info("=" * 80)
    
    def check_quantifier_setup(self):
        """Check quantifier agent and related rules."""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("QUANTIFIER CONFIGURATION CHECK")
        self.logger.info("=" * 80)
        
        try:
            from reactor.engine.base import ReactorEngine
            reactor_instance = self.registry.get_service_instance(ReactorEngine)
            
            if reactor_instance and hasattr(reactor_instance, 'rules'):
                # Find quantifier rules
                quantifier_rules = [
                    r for r in reactor_instance.rules 
                    if 'quantifier' in getattr(r, 'rule_id', '').lower()
                ]
                
                self.logger.info(f"Found {len(quantifier_rules)} quantifier-related rules:")
                
                for rule in quantifier_rules:
                    self._inspect_rule(rule)
            
            # Check quantifier agent
            self._check_component('quantifier_agent_primary', 'QuantifierAgent')
            
        except Exception as e:
            self.logger.error(f"Error checking quantifier setup: {e}", exc_info=True)
        
        self.logger.info("=" * 80 + "\n")
    
    def setup_signal_flow_tracking(self):
        """Setup comprehensive signal flow tracking."""
        self.logger.info("🔍 Setting up signal flow tracking...")
        
        signals_to_track = [
            'SeedSignal',
            'IdeaGeneratedSignal',
            'TrustAssessmentSignal',
            'PlanNextStepSignal',
            'ProtoTaskSignal',
            'GenerativeLoopFinishedSignal',
            'TraceEmittedSignal'
        ]
        
        for signal_name in signals_to_track:
            self.event_bus.subscribe(
                signal_name,
                self._create_signal_tracker(signal_name)
            )
        
        self.logger.info(f"✅ Tracking {len(signals_to_track)} signal types")
    
    def _check_specific_rules(self, rules: List[Any]):
        """Check for specific important rules."""
        # Check idea processing rules
        idea_rules = [r for r in rules if getattr(r, 'signal_type', '') == 'IdeaGeneratedSignal']
        self.logger.info(f"\nRules processing IdeaGeneratedSignal: {len(idea_rules)}")
        
        for rule in idea_rules[:3]:  # Show first 3
            self.logger.info(f"  - {rule.rule_id}")
            if hasattr(rule, 'actions_on_match'):
                for action in rule.actions_on_match[:2]:  # Show first 2 actions
                    self.logger.info(f"    → {action.get('type', 'unknown')}: {action.get('component_id', 'N/A')}")
        
        # Check trust assessment rules
        trust_rules = [r for r in rules if getattr(r, 'signal_type', '') == 'TrustAssessmentSignal']
        self.logger.info(f"\nRules processing TrustAssessmentSignal: {len(trust_rules)}")
    
    def _check_critical_components(self):
        self.logger.info('\nChecking critical components:')
        critical_components = [
            ('sentinel_instance_01', 'SentinelMechanism'),
            ('active_planner', 'BanditPlanner'),
            ('quantifier_agent_primary', 'QuantifierAgent'),
            ('proto_engine_math', 'ProtoEngine') 
        ]
        for component_id, component_type in critical_components:
            self._check_component(component_id, component_type)
    
    def _check_component(self, component_id: str, component_type: str):
        """Check if a component exists in the registry."""
        try:
            component = self.registry.get(component_id)
            if component:
                actual_type = type(component).__name__
                self.logger.info(f"  ✅ {component_id} found (type: {actual_type})")
            else:
                self.logger.error(f"  ❌ {component_id} is None!")
        except Exception as e:
            self.logger.error(f"  ❌ {component_id} not found: {e}")
    
    def _inspect_rule(self, rule: Any):
        """Inspect a single rule in detail."""
        self.logger.info(f"\nRule: {rule.rule_id}")
        self.logger.info(f"  - Signal type: {getattr(rule, 'signal_type', 'unknown')}")
        self.logger.info(f"  - Priority: {getattr(rule, 'priority', 'unknown')}")
        self.logger.info(f"  - Enabled: {getattr(rule, 'enabled', True)}")
        
        if hasattr(rule, 'conditions'):
            self.logger.info(f"  - Conditions: {len(rule.conditions)}")
            for i, cond in enumerate(rule.conditions[:2]):  # Show first 2
                if cond.get('type') == 'payload_expression':
                    expr = cond.get('expression', 'N/A')
                    if len(expr) > 80:
                        expr = expr[:80] + "..."
                    self.logger.info(f"    Condition {i}: {expr}")
        
        if hasattr(rule, 'actions_on_match'):
            self.logger.info(f"  - Actions: {len(rule.actions_on_match)}")
            for j, action in enumerate(rule.actions_on_match[:3]):  # Show first 3
                self.logger.info(f"    Action {j}: {action.get('type')} → {action.get('component_id', 'N/A')}")
    
    def _create_signal_tracker(self, signal_type: str):
        """Create a signal tracking callback."""
        def tracker(payload):
            timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
            self.logger.debug(f"[{timestamp}] Signal: {signal_type}")
            
            # Extract key information
            if isinstance(payload, dict):
                session_id = self._extract_session_id(payload)
                if session_id:
                    self.logger.debug(f"  └─ Session: {session_id}")
                    
                # Log specific details for important signals
                if signal_type == 'TrustAssessmentSignal':
                    trust_score = payload.get('trust_score', 'N/A')
                    self.logger.debug(f"  └─ Trust Score: {trust_score}")
                elif signal_type == 'IdeaGeneratedSignal':
                    idea_id = payload.get('id') or payload.get('idea_id', 'N/A')
                    self.logger.debug(f"  └─ Idea ID: {idea_id}")
        
        return tracker
    
    def _extract_session_id(self, payload: Dict[str, Any]) -> Optional[str]:
        """Extract session ID from various payload structures."""
        # Try different locations
        if 'session_id' in payload:
            return payload['session_id']
        elif 'payload' in payload and isinstance(payload['payload'], dict):
            return payload['payload'].get('session_id')
        elif 'assessment_details' in payload:
            return payload['assessment_details'].get('metadata', {}).get('session_id')
        return None


class SignalFlowVisualizer:
    """Visualize signal flow in real-time."""
    
    def __init__(self, event_bus: EventBusPort):
        self.event_bus = event_bus
        self.signal_counts = {}
        self.signal_flow = []
        
    def start_tracking(self):
        """Start tracking signal flow."""
        signals = [
            'SeedSignal', 'IdeaGeneratedSignal', 'TrustAssessmentSignal',
            'ProtoTaskSignal', 'GenerativeLoopFinishedSignal'
        ]
        
        for signal in signals:
            self.event_bus.subscribe(signal, self._track_signal(signal))
    
    def _track_signal(self, signal_type: str):
        """Create a tracking callback for a signal type."""
        def callback(payload):
            self.signal_counts[signal_type] = self.signal_counts.get(signal_type, 0) + 1
            self.signal_flow.append({
                'timestamp': datetime.now(),
                'signal_type': signal_type,
                'count': self.signal_counts[signal_type]
            })
            
            # Print live update
            self._print_flow_update(signal_type)
        
        return callback
    
    def _print_flow_update(self, latest_signal: str):
        """Print a visual update of signal flow."""
        # Create a simple ASCII visualization
        symbols = {
            'SeedSignal': '🌱',
            'IdeaGeneratedSignal': '💡',
            'TrustAssessmentSignal': '⚖️',
            'ProtoTaskSignal': '🔧',
            'GenerativeLoopFinishedSignal': '🏁'
        }
        
        line = f"[{datetime.now().strftime('%H:%M:%S')}] "
        line += symbols.get(latest_signal, '❓')
        line += f" {latest_signal}"
        
        # Add counts
        if self.signal_counts:
            counts = [f"{k}:{v}" for k, v in sorted(self.signal_counts.items())]
            line += f" | Totals: {', '.join(counts)}"
        
        print(f"\r{line}", end='', flush=True)