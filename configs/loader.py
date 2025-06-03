import json
import logging
import os
import re
import copy
from pathlib import Path
from typing import Any, Dict
import yaml
from .config_utils import ConfigMerger

logger = logging.getLogger(__name__)

# Minimized DEFAULT_CONFIG - only structure and absolute fallbacks
DEFAULT_CONFIG: Dict[str, Any] = {
    'env': 'default',
    'feature_flags': {},  # Empty - let YAML define the flags
    'storage': {
        'lineage_db_path': 'runtime/lineage_v4.db',
        'ideas_path': 'runtime/ideas_v4.json', 
        'facts_path': 'runtime/world_facts_v4.json'
    },
    'logging': {
        'prompt_response_logger': {'output_dir': 'runtime/llm_logs_v4'},
        'pipeline_event_logger': {'output_dir': 'runtime/pipeline_event_logs_v4'}
    }
    # Removed bootstrap_strict_mode and reactor_rules_module - let YAML control these
}

def _expand_env_var_string(value_str: str) -> str:
    if not isinstance(value_str, str):
        return value_str
    
    original_value = value_str
    
    def repl_default(match_default):
        var_name = match_default.group(1)
        default_val = match_default.group(2)
        env_val = os.getenv(var_name)
        result = env_val if env_val is not None else default_val
        logger.debug(f"Expanding ${{{var_name}:-{default_val}}} -> '{result}' (env_val={env_val})")
        return result
    
    expanded_str = re.sub('\\$\\{([\\w_]+):-?([^}]*)\\}', repl_default, value_str)
    expanded_str = os.path.expandvars(expanded_str)
    
    if original_value != expanded_str:
        logger.debug(f"Expanded env vars in string: '{original_value}' -> '{expanded_str}'")
    elif '${' in original_value and '}' in original_value:
        logger.warning(f"Environment variable placeholder not expanded: '{original_value}' - check if env var is set or syntax is correct")
    
    return expanded_str

def _expand_env_vars_in_dict(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in config_dict.items():
        config_dict[key] = _expand_env_vars_in_item(value)
    return config_dict

def _expand_env_vars_in_item(item: Any) -> Any:
    if isinstance(item, dict):
        return _expand_env_vars_in_dict(item)
    elif isinstance(item, list):
        return [_expand_env_vars_in_item(i) for i in item]
    elif isinstance(item, str):
        return _expand_env_var_string(item)
    return item

def load_config(env: str = 'default') -> Dict[str, Any]:
    package_root = Path(__file__).resolve().parents[1]
    config = copy.deepcopy(DEFAULT_CONFIG)
    config['env'] = env
    
    logger.debug(f"Initialized config with minimal DEFAULT_CONFIG for env '{env}'. Package root: {package_root}")
    
    # Load global_app_config.yaml files with YAML precedence
    config_files_to_load = []
    default_global_path = package_root / 'configs' / 'default' / 'global_app_config.yaml'
    config_files_to_load.append(('DEFAULT_GLOBAL_APP_CONFIG', default_global_path))
    
    if env != 'default':
        env_global_path = package_root / 'configs' / env / 'global_app_config.yaml'
        config_files_to_load.append((f'ENV_GLOBAL_APP_CONFIG ({env})', env_global_path))
    
    for label, path in config_files_to_load:
        if path.exists():
            try:
                with open(path, 'r', encoding='utf-8') as fh:
                    content = yaml.safe_load(fh) or {}
                if isinstance(content, dict):
                    # YAML overrides Python defaults
                    config = ConfigMerger.merge(config, content, label)
                    logger.info(f'Loaded and merged {label}: {path}')
                else:
                    logger.warning(f'{label} is not a dict: {path}')
            except Exception as exc:
                logger.error(f'Error loading {label} from {path}: {exc}', exc_info=True)
        elif 'ENV_GLOBAL' in label:
            logger.warning(f'{label} not found: {path}')
    
    # Load and merge LLM configuration
    llm_config_from_yaml: Dict[str, Any] = {}
    llm_config_paths_to_check = []
    default_llm_path = package_root / 'configs' / 'default' / 'llm_config.yaml'
    llm_config_paths_to_check.append(('DEFAULT_LLM_CONFIG', default_llm_path))
    
    if env != 'default':
        env_llm_path = package_root / 'configs' / env / 'llm_config.yaml'
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
    
    # Apply environment variable expansion to final LLM config
    if llm_config_from_yaml:
        logger.debug(f'LLM config BEFORE env var expansion: {json.dumps(llm_config_from_yaml, indent=2, default=str)}')
        llm_config_from_yaml = _expand_env_vars_in_dict(llm_config_from_yaml)
        logger.info(f'LLM config after env var expansion completed')
        logger.debug(f'LLM config AFTER env var expansion: {json.dumps(llm_config_from_yaml, indent=2, default=str)}')
    
    # Set final LLM configuration
    if not llm_config_from_yaml:
        if 'llm' in config and config['llm']:
            logger.info("Using 'llm' section defined directly in global_app_config.yaml as llm_config.yaml was not found or empty.")
            config['llm'] = _expand_env_vars_in_dict(config['llm'])
        else:
            raise RuntimeError(f"LLM configuration section is completely empty after attempting to load files for env '{env}'. Cannot proceed.")
    elif 'llm' not in config or not config['llm']:
        config['llm'] = llm_config_from_yaml
    else:
        config['llm'] = ConfigMerger.merge(config['llm'], llm_config_from_yaml, 'global_llm_with_specific_llm_config')
        config['llm'] = _expand_env_vars_in_dict(config['llm'])
    
    # Validate core LLM configuration
    if not isinstance(config.get('llm'), dict) or not isinstance(config['llm'].get('models'), dict) or not config['llm']['models']:
        raise RuntimeError(f"Missing or empty 'models' dictionary in the final LLM configuration for env '{env}'. Valid LLM configuration with defined models is required.")
    
    default_model_key = config['llm'].get('default')
    if not default_model_key:
        raise RuntimeError(f"No 'default' LLM model key specified in LLM configuration for env '{env}'.")
    
    if default_model_key not in config['llm']['models']:
        # Check for routing
        is_valid_route_to_model = False
        if 'routes' in config['llm'] and isinstance(config['llm']['routes'], dict):
            model_key_from_route = config['llm']['routes'].get(default_model_key)
            if model_key_from_route and model_key_from_route in config['llm']['models']:
                is_valid_route_to_model = True
        
        if not is_valid_route_to_model:
            raise RuntimeError(f"Default LLM key '{default_model_key}' (from 'llm.default') not found in 'llm.models' keys or as a valid 'llm.routes' entry for env '{env}'. Available models: {list(config['llm']['models'].keys())}")
    
    logger.info(f"Config loaded for env='{env}'. Top-level keys: {list(config.keys())}")
    logger.info(f"LLM configuration active. Default model key: '{config['llm'].get('default', 'N/A')}'")
    
    return config