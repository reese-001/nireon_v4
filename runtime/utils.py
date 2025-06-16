# C:\Users\erees\Documents\development\nireon\runtime\utils.py
import importlib
import logging
from pathlib import Path
from typing import Any, Dict, Mapping, Union
import yaml
logger = logging.getLogger(__name__)

def import_by_path(path: str, suppress_expected_errors: bool=True) -> Any:
    if not isinstance(path, str):
        raise TypeError(f'Import path must be a string, got {type(path)}')
    if ':' in path:
        module_name, attr_name = path.split(':', 1)
    elif '.' in path:
        module_name, attr_name = path.rsplit('.', 1)
        if not module_name and attr_name: # Handles cases like ".ClassName" if such were allowed (they are not here)
            raise ValueError(f"Relative import path '{path}' not supported. Provide full path.")
    else:
        # This case implies 'path' is a module name, and we'd need a convention for the attribute.
        # Or, it's just an attribute in the current module if this util was used differently.
        # Given the context, it's likely an error or underspecified.
        logger.error(f"Import path '{path}' is ambiguous. Use 'pkg.mod:Class' or 'pkg.mod.func'.")
        raise ValueError(f"Import path '{path}' is ambiguous. Use 'pkg.mod:Class' or 'pkg.mod.func'.")

    if not module_name or not attr_name:
        logger.error(f"Could not parse module and attribute from path '{path}'. Module: '{module_name}', Attribute: '{attr_name}'.")
        raise ValueError(f'Invalid import path format: {path}. Could not determine module and attribute.')

    log_level_if_not_found = logging.DEBUG if suppress_expected_errors else logging.ERROR
    exc_log_level_if_not_found = logging.DEBUG if suppress_expected_errors else logging.ERROR

    try:
        module = importlib.import_module(module_name)
        logger.debug(f'Successfully imported module: {module_name}')
    except ImportError as e:
        logger.log(log_level_if_not_found, f"Failed to import module '{module_name}' from path '{path}': {e}", exc_info=(log_level_if_not_found >= logging.ERROR))
        raise ImportError(f"Could not import module '{module_name}': {e}") from e
    try:
        attribute = getattr(module, attr_name)
        logger.debug(f"Successfully retrieved attribute '{attr_name}' from module '{module_name}'.")
        return attribute
    except AttributeError as e:
        logger.log(log_level_if_not_found, f"Attribute '{attr_name}' not found in module '{module_name}' (from path '{path}'): {e}", exc_info=(log_level_if_not_found >= logging.ERROR))
        raise AttributeError(f"Attribute '{attr_name}' not found in module '{module_name}': {e}") from e

def load_yaml_robust(path: Union[str, Path, None]) -> Dict[str, Any]:
    if not path:
        return {}

    path_str_representation = str(path)
    # Handling specific placeholder path often seen in templates
    if path_str_representation == 'nireon/configs/default/components/{id}.yaml':
        logger.debug(f"Path '{path_str_representation}' is the known placeholder for default component config. Returning empty config.")
        return {}
    elif '{id}' in path_str_representation or '{ID}' in path_str_representation: # General placeholder check
        logger.debug(f"Path '{path_str_representation}' contains placeholder tokens. Assuming default component config. Returning empty config.")
        return {}

    actual_path = Path(path)
    if not actual_path.exists():
        logger.warning(f"YAML file not found: '{actual_path}'. Returning empty config.")
        return {}
    try:
        with open(actual_path, 'r', encoding='utf-8') as fh:
            data = yaml.safe_load(fh) or {}
        if not isinstance(data, Mapping): # Ensure top-level is a dictionary
            logger.error(f"Top-level YAML object in '{actual_path}' must be a mapping/dictionary. Found {type(data)}. Returning empty config.")
            return {}
        return data
    except yaml.YAMLError as exc:
        logger.error(f"Error parsing YAML file '{actual_path}': {exc}", exc_info=True)
        return {}
    except Exception as exc: # Catch other potential errors like file permission issues
        logger.error(f"Unexpected error loading YAML file '{actual_path}': {exc}", exc_info=True)
        return {}

def detect_manifest_type(config_data: Dict[str, Any], style_hint: str='auto') -> str:
    if style_hint != 'auto':
        return style_hint

    v4_enhanced_markers = {'version', 'metadata', 'shared_services', 'mechanisms', 'observers', 'managers', 'composites', 'orchestration_commands'}
    # Check for V4 enhanced manifest
    if 'version' in config_data and 'metadata' in config_data and \
       any(marker in config_data for marker in v4_enhanced_markers - {'version', 'metadata'}):
        return 'enhanced'

    # Check for V4 simple manifest (might be 'components' list or old 'nireon_components' structure)
    if 'components' in config_data or \
       ('nireon_components' in config_data and 'mechanisms' in config_data.get('nireon_components', {})):
        return 'simple'

    logger.debug("Could not definitively detect manifest type, defaulting to 'enhanced' for V4.")
    return 'enhanced' # Default for V4 if unclear

def validate_config_hierarchy(config_dict: Dict[str, Any], config_name: str) -> bool:
    if not isinstance(config_dict, dict):
        logger.error(f"Configuration '{config_name}' must be a dictionary, got {type(config_dict)}")
        return False
    # Example check: look for unresolved environment variables
    if any(key.startswith('$') for key in config_dict.keys()):
        logger.warning(f"Configuration '{config_name}' contains keys starting with '$' - these may be unresolved environment variables")
    return True

def normalize_path(path: Union[str, Path], base_path: Union[str, Path, None] = None) -> Path:
    path_obj = Path(path)
    if not path_obj.is_absolute() and base_path:
        path_obj = Path(base_path) / path_obj
    return path_obj.resolve()

def safe_filename(name: str, max_length: int = 255) -> str:
    safe_chars = []
    for char in name:
        if char.isalnum() or char in '-_':
            safe_chars.append(char)
        else:
            safe_chars.append('_')
    safe_name = ''.join(safe_chars)

    if len(safe_name) > max_length:
        # Truncate and add ellipsis
        safe_name = safe_name[:max_length - 3] + '...'
    return safe_name

def deep_merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge_dicts(result[key], value)
        else:
            result[key] = value
    return result

def extract_class_name(class_path: str) -> str:
    if ':' in class_path:
        return class_path.split(':', 1)[1]
    elif '.' in class_path:
        return class_path.split('.')[-1]
    else:
        return class_path

def is_valid_component_id(component_id: str) -> bool:
    if not component_id or not isinstance(component_id, str):
        return False
    # Basic check: starts with a letter or underscore, followed by alphanumeric or _-
    if not (component_id[0].isalpha() or component_id[0] == '_'):
        return False
    return all(char.isalnum() or char in '_-' for char in component_id)