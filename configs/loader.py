# Adapted from nireon_staging/nireon/configs/loader.py
import json
import logging
import os
import re
import copy # V4: Ensure deepcopy for default config
from pathlib import Path
from typing import Any, Dict

import yaml

# V4: Use V4 ConfigMerger
from .config_utils import ConfigMerger 

logger = logging.getLogger(__name__)

# V4: Define a baseline default config. This can be expanded.
DEFAULT_CONFIG: Dict[str, Any] = {
    "env": "default",
    "feature_flags": {
        "debug_mode": False,
    },
    "storage": { # V4: As per V3 structure
        "lineage_db_path": "runtime/lineage_v4.db", # V4 specific path
        "ideas_path": "runtime/ideas_v4.json",
        "facts_path": "runtime/world_facts_v4.json",
    },
    "logging": { # V4: As per V3 structure
         "prompt_response_logger": {"output_dir": "runtime/llm_logs_v4"},
         "pipeline_event_logger": {"output_dir": "runtime/pipeline_event_logs_v4"}
    },
    "bootstrap_strict_mode": False, # V4: Keep this crucial flag
    "reactor_rules_module": "nireon_v4.application.orchestration.reactor_rules.default_rules" # V4 path
}

def _expand_env_var_string(value_str: str) -> str:
    if not isinstance(value_str, str):
        return value_str # type: ignore

    original_value = value_str

    # Regex for ${VAR:-default} or ${VAR}
    def repl_default(match_default):
        var_name = match_default.group(1)
        default_val = match_default.group(2) # Can be empty if :- is not used
        env_val = os.getenv(var_name)
        result = env_val if env_val is not None else default_val
        logger.debug(f"Expanding ${{{var_name}:-{default_val}}} -> '{result}' (env_val={env_val})")
        return result

    # V4: Updated regex to handle optional default value more robustly
    expanded_str = re.sub(r'\$\{([\w_]+):-?([^}]*)\}', repl_default, value_str)
    # Fallback for simple ${VAR}
    expanded_str = os.path.expandvars(expanded_str)

    if original_value != expanded_str:
        logger.debug(f"Expanded env vars in string: '{original_value}' -> '{expanded_str}'")
    elif '${' in original_value and '}' in original_value: # Check if any unexpanded ${VAR} syntax remains
        logger.warning(f"Environment variable placeholder not expanded: '{original_value}' - check if env var is set or syntax is correct (e.g. ${{VAR:-default_val}} or ${{VAR}})")
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

def load_config(env: str = "default") -> Dict[str, Any]:
    # V4: Determine package root based on this file's location
    # Assumes loader.py is in nireon_v4/configs/
    package_root = Path(__file__).resolve().parents[1] 
    
    config = copy.deepcopy(DEFAULT_CONFIG) # Start with hardcoded defaults
    config["env"] = env # Set the current environment
    logger.debug(f"Initialized config with DEFAULT_CONFIG for env '{env}'. Package root: {package_root}")

    # Define paths for global_app_config.yaml
    config_files_to_load = []
    default_global_path = package_root / "configs" / "default" / "global_app_config.yaml"
    config_files_to_load.append(("DEFAULT_GLOBAL_APP_CONFIG", default_global_path))
    
    if env != "default":
        env_global_path = package_root / "configs" / env / "global_app_config.yaml"
        config_files_to_load.append((f"ENV_GLOBAL_APP_CONFIG ({env})", env_global_path))

    for label, path in config_files_to_load:
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    content = yaml.safe_load(fh) or {}
                if isinstance(content, dict):
                    config = ConfigMerger.merge(config, content, label)
                    logger.info(f"Loaded {label}: {path}")
                else:
                    logger.warning(f"{label} is not a dict: {path}")
            except Exception as exc:
                logger.error(f"Error loading {label} from {path}: {exc}", exc_info=True)
        elif "ENV_GLOBAL" in label: # Only warn if env-specific is missing
             logger.warning(f"{label} not found: {path}")


    # Load and merge LLM configurations
    llm_config_from_yaml: Dict[str, Any] = {}
    llm_config_paths_to_check = []
    default_llm_path = package_root / "configs" / "default" / "llm_config.yaml"
    llm_config_paths_to_check.append(("DEFAULT_LLM_CONFIG", default_llm_path))

    if env != "default":
        env_llm_path = package_root / "configs" / env / "llm_config.yaml"
        llm_config_paths_to_check.append((f"ENV_LLM_CONFIG ({env})", env_llm_path))

    for label, path in llm_config_paths_to_check:
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    content = yaml.safe_load(fh) or {}
                if isinstance(content, dict):
                    llm_config_from_yaml = ConfigMerger.merge(llm_config_from_yaml, content, f"llm_config_merge: {label}")
                    logger.info(f"Loaded and merged LLM config from {label}: {path}")
                else:
                    logger.warning(f"{label} at {path} is not a dict, skipping.")
            except Exception as exc:
                logger.error(f"Error loading LLM config from {label} at {path}: {exc}", exc_info=True)
        elif "ENV_LLM" in label:
             logger.warning(f"{label} not found at {path}. Using defaults or previously loaded LLM config.")
    
    # Expand environment variables in the LLM config part
    if llm_config_from_yaml:
        logger.debug(f"LLM config BEFORE env var expansion: {json.dumps(llm_config_from_yaml, indent=2, default=str)}")
        llm_config_from_yaml = _expand_env_vars_in_dict(llm_config_from_yaml) # Applied to the LLM part
        logger.info(f"LLM config after env var expansion completed")
        logger.debug(f"LLM config AFTER env var expansion: {json.dumps(llm_config_from_yaml, indent=2, default=str)}")
    else:
        logger.warning("llm_config_from_yaml is empty before environment variable expansion step.")


    # V4: Basic validation of LLM config structure
    if not llm_config_from_yaml:
        # If global_app_config *itself* defined an llm section, use that, else error.
        if 'llm' in config and config['llm']:
            logger.info("Using 'llm' section defined directly in global_app_config.yaml as llm_config.yaml was not found or empty.")
            config['llm'] = _expand_env_vars_in_dict(config['llm']) # Expand here if taken from global
        else:
            raise RuntimeError(f"LLM configuration section is completely empty after attempting to load files for env '{env}'. Cannot proceed.")
    elif 'llm' not in config or not config['llm']: # if llm_config.yaml was loaded, but global_app_config has no llm section
        config['llm'] = llm_config_from_yaml # Assign loaded llm_config to the main config
    else: # Both global_app_config.llm and llm_config.yaml have content, merge them
        config['llm'] = ConfigMerger.merge(config['llm'], llm_config_from_yaml, "global_llm_with_specific_llm_config")
        config['llm'] = _expand_env_vars_in_dict(config['llm']) # Ensure expansion on final merged LLM config

    # Final check on the structure of config['llm']
    if not isinstance(config.get('llm'), dict) or \
       not isinstance(config['llm'].get('models'), dict) or \
       not config['llm']['models']:
        raise RuntimeError(
            f"Missing or empty 'models' dictionary in the final LLM configuration for env '{env}'. "
            "Valid LLM configuration with defined models is required."
        )
    
    # Validate default model existence
    default_model_key = config['llm'].get('default')
    if not default_model_key:
        raise RuntimeError(f"No 'default' LLM model key specified in LLM configuration for env '{env}'.")
    
    # Check if default key is in models or is a valid route
    if default_model_key not in config['llm']['models']:
        is_valid_route_to_model = False
        if 'routes' in config['llm'] and isinstance(config['llm']['routes'], dict):
            model_key_from_route = config['llm']['routes'].get(default_model_key)
            if model_key_from_route and model_key_from_route in config['llm']['models']:
                is_valid_route_to_model = True
        if not is_valid_route_to_model:
            raise RuntimeError(
                f"Default LLM key '{default_model_key}' (from 'llm.default') not found in 'llm.models' keys "
                f"or as a valid 'llm.routes' entry for env '{env}'. "
                f"Available models: {list(config['llm']['models'].keys())}"
            )
            
    logger.info(f"Config loaded for env='{env}'. Top-level keys: {list(config.keys())}")
    logger.info(f"LLM configuration active. Default model key: '{config['llm'].get('default', 'N/A')}'")
    return config