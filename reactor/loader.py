# FILE: nireon_v4/reactor/loader.py

import yaml
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

# Use relative imports for sibling modules
from .protocols import ReactorRule
from .rules.core_rules import ConditionalRule

logger = logging.getLogger(__name__)

class RuleLoader:
    """
    Loads and instantiates Reactor rules from YAML configuration files.
    """
    def __init__(self):
        self.loaded_rules: List[ReactorRule] = []

    def load_rules_from_directory(self, directory_path: Path) -> List[ReactorRule]:
        """
        Scans a directory for .yaml or .yml files and loads all rules.
        """
        if not directory_path.is_dir():
            logger.warning(f"Rule directory not found: {directory_path}")
            return []
        
        logger.info(f"Scanning for rules in: {directory_path}")
        
        # Load from all YAML files
        for filepath in directory_path.glob('*.yaml'):
            self._load_rules_from_file(filepath)
        for filepath in directory_path.glob('*.yml'):
            self._load_rules_from_file(filepath)
        
        # Sort rules by priority (lower priority number = higher precedence)
        self.loaded_rules.sort(key=lambda r: getattr(r, 'priority', 100))
        
        logger.info(f"Total rules loaded and sorted by priority: {len(self.loaded_rules)}")
        return self.loaded_rules

    def _load_rules_from_file(self, filepath: Path):
        """Loads and parses a single YAML rule file."""
        try:
            with open(filepath, 'r') as f:
                data = yaml.safe_load(f)

            if not data or 'rules' not in data:
                logger.warning(f"Skipping file {filepath}: No 'rules' key found.")
                return
            
            # Check version if present
            version = data.get('version', '1.0')
            if version != '1.0':
                logger.warning(f"Rule file {filepath} uses version {version}, expected 1.0")

            for rule_config in data['rules']:
                if not rule_config.get('enabled', True):
                    logger.info(f"Skipping disabled rule: {rule_config.get('id', 'N/A')}")
                    continue
                
                # Validate rule configuration
                if not self._validate_rule_config(rule_config):
                    logger.error(f"Skipping invalid rule config: {rule_config.get('id', 'N/A')}")
                    continue

                # Create rule from configuration
                rule = self._create_rule_from_config(rule_config)
                if rule:
                    self.loaded_rules.append(rule)
                    logger.debug(f"Loaded rule '{rule.rule_id}' from {filepath.name}")

        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML file {filepath}: {e}")
        except Exception as e:
            logger.error(f"Error loading rules from {filepath}: {e}", exc_info=True)

    def _validate_rule_config(self, config: Dict[str, Any]) -> bool:
        """Validates that a rule's configuration is well-formed."""
        rule_id = config.get('id', 'N/A')
        
        # Check required fields
        if not rule_id or rule_id == 'N/A':
            logger.error("Rule missing 'id' field")
            return False
        
        if not config.get('conditions'):
            logger.error(f"Rule '{rule_id}' missing 'conditions' field")
            return False
        
        if not config.get('actions'):
            logger.error(f"Rule '{rule_id}' missing 'actions' field")
            return False
        
        # Validate each action
        for action in config.get('actions', []):
            action_type = action.get('type')
            if not action_type:
                logger.error(f"Rule '{rule_id}' has an action missing a 'type'.")
                return False
            
            if action_type == 'trigger_component' and not action.get('component_id'):
                logger.error(f"Rule '{rule_id}' has a 'trigger_component' action missing a 'component_id'.")
                return False
            
            if action_type == 'emit_signal' and not action.get('signal_type'):
                logger.error(f"Rule '{rule_id}' has an 'emit_signal' action missing a 'signal_type'.")
                return False
        
        return True

    def _create_rule_from_config(self, config: Dict[str, Any]) -> Optional[ReactorRule]:
        """
        Instantiates a rule object from its dictionary configuration.
        This is the factory part of the loader.
        """
        rule_id = config.get('id')
        if not rule_id:
            logger.error(f"Rule missing 'id' in config: {config}")
            return None

        # Extract the signal_type from the conditions
        signal_type = "UNKNOWN"
        for cond in config.get('conditions', []):
            if cond.get('type') == 'signal_type_match':
                signal_type = cond.get('signal_type', 'UNKNOWN')
                break
        
        if signal_type == "UNKNOWN":
            logger.error(f"Rule '{rule_id}' has no 'signal_type_match' condition. Cannot create rule.")
            return None

        # Create ConditionalRule instance
        rule = ConditionalRule(
            rule_id=rule_id,
            signal_type=signal_type,
            conditions=config.get('conditions', []),
            actions_on_match=config.get('actions', []),
            priority=config.get('priority', 100),
            namespace=config.get('namespace', 'core')
        )
        
        logger.debug(
            f"Successfully created rule object for '{rule.rule_id}' "
            f"(namespace: {rule.namespace}, priority: {rule.priority})"
        )
        return rule