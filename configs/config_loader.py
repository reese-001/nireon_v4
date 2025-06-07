# C:\Users\erees\Documents\development\nireon_v4\configs\config_loader.py
from __future__ import annotations
import copy
import json
import logging
import os
import re
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from configs.config_utils import ConfigMerger # Assuming this import is still valid relative to the new location
logger = logging.getLogger(__name__)
DEFAULT_CONFIG: Dict[str, Any] = {'env': 'default', 'feature_flags': {}, 'storage': {'lineage_db_path': 'runtime/lineage_v4.db', 'ideas_path': 'runtime/ideas_v4.json', 'facts_path': 'runtime/world_facts_v4.json'}, 'logging': {'prompt_response_logger': {'output_dir': 'runtime/llm_logs_v4'}, 'pipeline_event_logger': {'output_dir': 'runtime/pipeline_event_logs_v4'}}}

class ConfigLoader:
    def __init__(self):
        # Adjust package_root if necessary, assuming 'configs' is one level down from project root
        self.package_root = Path(__file__).resolve().parents[1]
    async def load_global_config(self, env: Optional[str]=None, provided_config: Optional[Dict[str, Any]]=None) -> Dict[str, Any]:
        if provided_config is not None:
            logger.info('Using provided global configuration')
            return self._expand_environment_variables(provided_config)
        env = env or 'default'
        logger.info(f'Loading global configuration for environment: {env}')
        try:
            config = copy.deepcopy(DEFAULT_CONFIG)
            config['env'] = env
            config = await self._load_global_app_configs(config, env)
            config = await self._load_llm_configs(config, env)
            config = self._expand_environment_variables(config)
            config = await self._apply_enhancements(config, env)
            self._validate_required_config(config, env)
            logger.info(f"✓ Global configuration loaded for env='{env}'")
            logger.debug(f'Config keys: {list(config.keys())}')
            return config
        except Exception as e:
            logger.error(f"Failed to load global configuration for env '{env}': {e}")
            raise
    async def load_component_config(self, component_spec: Dict[str, Any], component_id: str, global_config: Dict[str, Any]) -> Dict[str, Any]:
        logger.debug(f'Loading component config for: {component_id}')
        config = {}
        default_config = await self._load_default_component_config(component_spec, component_id)
        config = ConfigMerger.merge(config, default_config, f'{component_id}_defaults')
        env = global_config.get('env', 'default')
        env_config = await self._load_env_component_config(component_spec, component_id, env)
        config = ConfigMerger.merge(config, env_config, f'{component_id}_env_{env}')
        manifest_inline_config = component_spec.get('inline_config', {})
        config_override = component_spec.get('config_override', {})
        if manifest_inline_config:
            config = ConfigMerger.merge(config, manifest_inline_config, f'{component_id}_inline')
        if config_override:
            config = ConfigMerger.merge(config, config_override, f'{component_id}_override')
        config = self._expand_environment_variables(config)
        logger.debug(f'✓ Component config resolved for {component_id}: {list(config.keys())}')
        return config
    async def _load_global_app_configs(self, config: Dict[str, Any], env: str) -> Dict[str, Any]:
        config_files_to_load = []
        default_global_path = self.package_root / 'configs' / 'default' / 'global_app_config.yaml'
        config_files_to_load.append(('DEFAULT_GLOBAL_APP_CONFIG', default_global_path))
        if env != 'default':
            env_global_path = self.package_root / 'configs' / env / 'global_app_config.yaml'
            config_files_to_load.append((f'ENV_GLOBAL_APP_CONFIG ({env})', env_global_path))
        for label, path in config_files_to_load:
            if path.exists():
                try:
                    with open(path, 'r', encoding='utf-8') as fh:
                        content = yaml.safe_load(fh) or {}
                    if isinstance(content, dict):
                        config = ConfigMerger.merge(config, content, label)
                        logger.info(f'Loaded and merged {label}: {path}')
                    else:
                        logger.warning(f'{label} is not a dict: {path}')
                except Exception as exc:
                    logger.error(f'Error loading {label} from {path}: {exc}', exc_info=True)
            elif 'ENV_GLOBAL' in label:
                logger.warning(f'{label} not found: {path}')
        return config
    async def _load_llm_configs(self, config: Dict[str, Any], env: str) -> Dict[str, Any]:
        llm_config_from_yaml: Dict[str, Any] = {}
        llm_config_paths_to_check = []
        default_llm_path = self.package_root / 'configs' / 'default' / 'llm_config.yaml'
        llm_config_paths_to_check.append(('DEFAULT_LLM_CONFIG', default_llm_path))
        if env != 'default':
            env_llm_path = self.package_root / 'configs' / env / 'llm_config.yaml'
            llm_config_paths_to_check.append((f'ENV_LLM_CONFIG ({env})', env_llm_path))
        for label, path in llm_config_paths_to_check:
            if path.exists():
                try:
                    with open(path, 'r', encoding='utf-8') as fh:
                        content = yaml.safe_load(fh) or {}
                    if isinstance(content, dict):
                        llm_config_from_yaml = ConfigMerger.merge(llm_config_from_yaml, content, f'llm_config_merge: {label}')
                        logger.info(f'Loaded and merged LLM config from {label}: {path}')
                    else:
                        logger.warning(f'{label} at {path} is not a dict, skipping.')
                except Exception as exc:
                    logger.error(f'Error loading LLM config from {label} at {path}: {exc}', exc_info=True)
            elif 'ENV_LLM' in label:
                logger.debug(f'{label} not found at {path}. Using defaults or previously loaded LLM config.')
        if not llm_config_from_yaml:
            if 'llm' in config and config['llm']:
                logger.info("Using 'llm' section defined directly in global_app_config.yaml as llm_config.yaml was not found or empty.")
            else:
                raise RuntimeError(f"LLM configuration section is completely empty after attempting to load files for env '{env}'. Cannot proceed.")
        elif 'llm' not in config or not config['llm']:
            config['llm'] = llm_config_from_yaml
        else:
            config['llm'] = ConfigMerger.merge(config['llm'], llm_config_from_yaml, 'global_llm_with_specific_llm_config')
        return config
    async def _apply_enhancements(self, base_config: Dict[str, Any], env: str) -> Dict[str, Any]:
        enhanced = base_config.copy()
        if 'bootstrap_strict_mode' not in enhanced:
            logger.warning('Bootstrap strict mode not defined in any config, defaulting to True (system hardcoded).')
            enhanced['bootstrap_strict_mode'] = True
        if 'feature_flags' not in enhanced:
            logger.warning('Feature flags section not defined in any config, defaulting to empty.')
            enhanced['feature_flags'] = {}
        default_flags = {'enable_schema_validation': True, 'enable_self_certification': True}
        for flag, default_val in default_flags.items():
            if flag not in enhanced['feature_flags']:
                logger.info(f"Feature flag '{flag}' not found in config, applying system default: {default_val}")
                enhanced['feature_flags'][flag] = default_val
        return enhanced
    async def _load_default_component_config(self, component_spec: Dict[str, Any], component_id: str) -> Dict[str, Any]:
        config_path_template = component_spec.get('config')
        if not config_path_template:
            return {}
        config_path = config_path_template.replace('{id}', component_id)
        full_path = self.package_root / config_path # Assumes config_path is relative to package_root
        if not full_path.exists():
            logger.debug(f'No default config found: {full_path}')
            return {}
        try:
            with open(full_path, 'r', encoding='utf-8') as fh:
                content = yaml.safe_load(fh) or {}
            logger.debug(f'Loaded default component config: {full_path}')
            return content if isinstance(content, dict) else {}
        except Exception as e:
            logger.error(f'Error loading default component config from {full_path}: {e}')
            return {}
    async def _load_env_component_config(self, component_spec: Dict[str, Any], component_id: str, env: str) -> Dict[str, Any]:
        if env == 'default':
            return {}
        config_path_template = component_spec.get('config')
        if not config_path_template:
            return {}
        # This path logic might need to be more robust if 'default' isn't always part of the template path
        config_path = config_path_template.replace('default', env).replace('{id}', component_id)
        full_path = self.package_root / config_path # Assumes config_path is relative to package_root
        if not full_path.exists():
            logger.debug(f'No env config found: {full_path}')
            return {}
        try:
            with open(full_path, 'r', encoding='utf-8') as fh:
                content = yaml.safe_load(fh) or {}
            logger.debug(f'Loaded env component config: {full_path}')
            return content if isinstance(content, dict) else {}
        except Exception as e:
            logger.error(f'Error loading env component config from {full_path}: {e}')
            return {}
    def _expand_environment_variables(self, config: Any) -> Any:
        if isinstance(config, dict):
            return {key: self._expand_environment_variables(value) for key, value in config.items()}
        elif isinstance(config, list):
            return [self._expand_environment_variables(item) for item in config]
        elif isinstance(config, str):
            return self._expand_env_var_string(config)
        else:
            return config
    def _expand_env_var_string(self, value_str: str) -> str:
        if not isinstance(value_str, str):
            return value_str
        original_value = value_str
        # Regex for ${VAR:-default}
        def repl_default(match):
            var_name = match.group(1)
            default_val = match.group(2)
            env_val = os.getenv(var_name)
            result = env_val if env_val is not None else default_val
            logger.debug(f"Expanding ${{{var_name}:-{default_val}}} -> '{result}'")
            return result
        expanded_str = re.sub('\\$\\{([A-Za-z_][A-Za-z0-9_]*):-(.*?)\\}', repl_default, value_str)
        # Standard environment variable expansion for $VAR or ${VAR}
        expanded_str = os.path.expandvars(expanded_str)
        if original_value != expanded_str:
            logger.debug(f"Environment expansion: '{original_value}' -> '{expanded_str}'")
        # Check for unresolved ${...} patterns specifically, as os.path.expandvars might leave them if malformed or var not found.
        elif '${' in original_value and '}' in original_value: # Simplified check
            logger.warning(f"Unresolved environment variable: '{original_value}'")
        return expanded_str
    def _validate_required_config(self, config: Dict[str, Any], env: str) -> None:
        required_keys = ['env', 'feature_flags', 'bootstrap_strict_mode']
        missing_keys = [key for key in required_keys if key not in config]
        if missing_keys:
            raise ValueError(f"Missing required config keys for env '{env}': {missing_keys}")
        if 'llm' in config:
            if not isinstance(config['llm'], dict):
                raise ValueError('LLM configuration must be a dictionary')
            if 'models' not in config['llm'] or not config['llm']['models']:
                raise ValueError("LLM configuration must include 'models' dictionary")
            default_model_key = config['llm'].get('default')
            if not default_model_key:
                raise ValueError(f"No 'default' LLM model key specified in LLM configuration for env '{env}'.")
            if default_model_key not in config['llm']['models']:
                # Check if it's a route
                is_valid_route_to_model = False
                if 'routes' in config['llm'] and isinstance(config['llm']['routes'], dict):
                    model_key_from_route = config['llm']['routes'].get(default_model_key)
                    if model_key_from_route and model_key_from_route in config['llm']['models']:
                        is_valid_route_to_model = True
                if not is_valid_route_to_model:
                    available_models = list(config['llm']['models'].keys())
                    raise ValueError(f"Default LLM key '{default_model_key}' not found in 'llm.models' keys or as a valid 'llm.routes' entry for env '{env}'. Available models: {available_models}")
        logger.debug(f"✓ Configuration validation passed for env '{env}'")