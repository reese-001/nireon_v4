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
        # This variable will store the merged content from *inside* the 'llm' key
        # of all llm_config.yaml files encountered.
        effective_llm_config_data: Dict[str, Any] = {}

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
                        # content_from_yaml_file will be like {'llm': { actual_configs }}
                        content_from_yaml_file = yaml.safe_load(fh) or {}
                    
                    if isinstance(content_from_yaml_file, dict):
                        # Extract the actual LLM configuration data, which is expected
                        # to be under the 'llm' key in the YAML file.
                        current_file_llm_data_part = content_from_yaml_file.get('llm', {})
                        if not isinstance(current_file_llm_data_part, dict):
                            logger.warning(f"Content under 'llm' key in {label} ({path}) is not a dictionary. Skipping this part.")
                            current_file_llm_data_part = {}

                        # Merge the data from this file into our accumulating effective_llm_config_data
                        effective_llm_config_data = ConfigMerger.merge(
                            effective_llm_config_data,
                            current_file_llm_data_part, # Merge the content *inside* 'llm'
                            f'llm_config_content_merge: {label}'
                        )
                        logger.info(f'Loaded and merged LLM config content from {label}: {path}')
                    else:
                        logger.warning(f'{label} at {path} is not a dict, skipping.')
                except Exception as exc:
                    logger.error(f'Error loading LLM config from {label} at {path}: {exc}', exc_info=True)
            elif 'ENV_LLM' in label: # Only log debug if an environment-specific file is missing
                logger.debug(f'{label} not found at {path}. Using defaults or previously loaded LLM config.')

        # Now, merge the accumulated effective_llm_config_data into the main config's 'llm' section.
        # The main config['llm'] might have been initialized by global_app_config.yaml (e.g., with 'parameters').
        if not effective_llm_config_data:
            # This means no llm_config.yaml was found, or they were empty/invalid under 'llm' key.
            # We rely on global_app_config.yaml potentially having a complete 'llm' section.
            if 'llm' not in config or not config['llm']:
                # This is a critical failure: no 'llm' section from global_app_config
                # AND no valid content from any llm_config.yaml files.
                raise RuntimeError(f"LLM configuration section ('llm') is completely empty after attempting to load files for env '{env}'. Cannot proceed.")
            else:
                # global_app_config.yaml has an 'llm' section, and no llm_config.yaml provided meaningful content.
                # The 'llm' section in 'config' is already what it should be.
                logger.info("Using 'llm' section as defined in global_app_config.yaml, as no overriding llm_config.yaml content was found.")
        else: # We have valid content from llm_config.yaml files (in effective_llm_config_data)
            if 'llm' not in config: # If global_app_config.yaml had NO 'llm' section at all
                config['llm'] = effective_llm_config_data
            else: # Both global_app_config.yaml (e.g. 'parameters') and llm_config.yaml files contributed.
                  # config['llm'] already exists (possibly from global_app_config)
                config['llm'] = ConfigMerger.merge(
                    config.get('llm', {}), # Use existing config['llm'] as base
                    effective_llm_config_data, 
                    'global_llm_with_specific_llm_config_content'
                )
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

    def _get_config_path(self, filename: str, env: str) -> Optional[Path]:
        """
        Constructs the path to a configuration file, checking env-specific then default.
        Returns the Path object if the file exists, otherwise None.
        """
        # Try environment-specific path first
        if env != 'default':
            env_path = self.package_root / 'configs' / env / filename
            if env_path.exists():
                logger.debug(f"Found config file '{filename}' in env '{env}' directory: {env_path}")
                return env_path

        # Fallback to default path
        default_path = self.package_root / 'configs' / 'default' / filename
        if default_path.exists():
            logger.debug(f"Found config file '{filename}' in 'default' directory: {default_path}")
            return default_path
        
        logger.debug(f"Config file '{filename}' not found in env '{env}' or 'default' directories.")
        return None

    def _validate_required_config(self, config: Dict[str, Any], env: str) -> None:
        """Validate that required configuration sections exist."""
        # Skip validation if llm config hasn't been loaded yet
        if 'llm' not in config:
            logger.warning(f"LLM config not yet loaded during validation for env '{env}' - skipping validation")
            return
            
        llm_config = config.get('llm', {})
        
        # Only validate if llm config is present and is a dict
        if isinstance(llm_config, dict) and llm_config:
            if 'models' not in llm_config:
                # If 'models' is missing from the llm_config (which should be fully merged by now),
                # it's an error in the YAML content itself. _load_llm_configs already handles
                # cases where llm_config.yaml might be missing entirely.
                raise ValueError(
                    f"LLM configuration (config['llm']) must include a 'models' dictionary. "
                    f"Current keys in config['llm']: {list(llm_config.keys())}. "
                    "This indicates an issue with the content of your llm_config.yaml or global_app_config.yaml."
                )
        else:
            # This case should ideally be caught by _load_llm_configs raising a RuntimeError
            # if the 'llm' section is completely empty after attempting to load files.
            logger.warning(f"LLM configuration (config['llm']) is missing, not a dictionary, or empty for env '{env}'. This might be an issue.")