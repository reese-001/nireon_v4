# reactor/loader.py
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .core_rules import ConditionalRule
from .protocols import ReactorRule

logger = logging.getLogger(__name__)


class RuleLoader:
    """Load YAML rule‑files from a directory tree."""

    def __init__(self) -> None:
        self.loaded_rules: List[ReactorRule] = []

    # ------------------------------------------------------------------ #
    def load_rules_from_directory(self, directory_path: Path) -> List[ReactorRule]:
        if not directory_path.is_dir():
            logger.warning("Rule directory not found: %s", directory_path)
            return []

        logger.info("Loading rules from %s", directory_path)
        for fp in directory_path.rglob("*"):
            if fp.suffix.lower() not in (".yml", ".yaml", ".txt"):
                continue
            self._load_rules_from_file(fp)

        self.loaded_rules.sort(key=lambda r: (getattr(r, "priority", 100), r.rule_id))
        logger.info("%d rule(s) loaded", len(self.loaded_rules))
        return self.loaded_rules

    # ------------------------------------------------------------------ #
    def _load_rules_from_file(self, filepath: Path) -> None:
        logger.info(f"Loading rules from file: {filepath}")  # You already added this
        try:
            data = yaml.safe_load(filepath.read_text())
            if not data or 'rules' not in data:
                logger.debug("No 'rules' key in %s – skipped", filepath.name)
                return
            if (ver := data.get('version', '1.0')) != '1.0':
                logger.warning("Unexpected version '%s' in %s", ver, filepath.name)
            
            # Add this to track rules being loaded
            file_rules = []
            
            for cfg in data['rules']:
                if not cfg.get('enabled', True):
                    logger.debug("Rule %s disabled - skipping", cfg.get('id', 'unknown'))
                    continue
                if not self._validate_rule_config(cfg):
                    logger.debug("Rule %s failed validation - skipping", cfg.get('id', 'unknown'))
                    continue
                rule = self._create_rule_from_config(cfg)
                if rule:
                    # Check for duplicates
                    existing_rule_ids = [r.rule_id for r in self.loaded_rules]
                    if rule.rule_id in existing_rule_ids:
                        logger.warning("DUPLICATE RULE ID '%s' from %s - will overwrite previous", 
                                    rule.rule_id, filepath.name)
                    
                    self.loaded_rules.append(rule)
                    file_rules.append(rule.rule_id)
                    logger.debug('Rule %s loaded from %s', rule.rule_id, filepath.name)
            
            # Log summary for this file
            logger.info("Loaded %d rules from %s: %s", len(file_rules), filepath.name, file_rules)
            
        except yaml.YAMLError as exc:
            logger.error('YAML error in %s: %s', filepath, exc)
        except Exception as exc:
            logger.exception('Failed to load %s: %s', filepath, exc)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _validate_rule_config(cfg: Dict[str, Any]) -> bool:
        rule_id = cfg.get("id")
        if not rule_id:
            logger.error("Rule missing 'id': %s", cfg)
            return False

        for key in ("conditions", "actions"):
            if not cfg.get(key):
                logger.error("Rule %s missing '%s'", rule_id, key)
                return False

        for act in cfg["actions"]:
            act_type = act.get("type")
            if not act_type:
                logger.error("Rule %s has action without 'type'", rule_id)
                return False
            match act_type:
                case "trigger_component":
                    if not act.get("component_id"):
                        logger.error("Rule %s trigger_component lacks component_id", rule_id)
                        return False
                case "emit_signal":
                    if not act.get("signal_type"):
                        logger.error("Rule %s emit_signal lacks signal_type", rule_id)
                        return False
                case _:
                    logger.warning("Rule %s unknown action type %s", rule_id, act_type)
        return True

    # ------------------------------------------------------------------ #
    @staticmethod
    def _create_rule_from_config(cfg: Dict[str, Any]) -> Optional[ReactorRule]:
        signal_type = next(
            (c.get("signal_type") for c in cfg["conditions"] if c.get("type") == "signal_type_match"),
            None,
        )
        if signal_type is None:
            logger.error("Rule %s lacks signal_type_match condition", cfg["id"])
            return None
        return ConditionalRule(
            rule_id=cfg["id"],
            signal_type=signal_type,
            conditions=cfg["conditions"],
            actions_on_match=cfg["actions"],
            priority=cfg.get("priority", 100),
            namespace=cfg.get("namespace", "core"),
        )
