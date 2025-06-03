from __future__ import annotations
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

from configs.loader import load_config
from configs.config_utils import ConfigMerger

logger = logging.getLogger(__name__)

class ConfigLoader:
    def __init__(self):
        self.package_root = Path(__file__).resolve().parents[2]

    async def load_global_config(self, env: Optional[str] = None, provided_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if provided_config is not None:
            logger.info('Using provided global configuration')
            return self._expand_environment_variables(provided_config)
        
        env = env or 'default'
        logger.info(f'Loading  global configuration for environment: {env}')
        
        try:
            # Use the configs.loader.load_config() function as primary source
            base_config = load_config(env=env)
            
            # Apply  enhancements - but remove hardcoded defaults
            enhanced_config = await self._apply__enhancements(base_config, env)
            
            # Final environment variable expansion
            final_config = self._expand_environment_variables(enhanced_config)
            
            # Validate required configuration
            self._validate_required_config(final_config, env)
            
            logger.info(f"✓  global configuration loaded for env='{env}'")
            logger.debug(f'Config keys: {list(final_config.keys())}')
            
            return final_config
            
        except Exception as e:
            logger.error(f"Failed to load  global configuration for env '{env}': {e}")
            raise

    async def load_component_config(self, component_spec: Dict[str, Any], component_id: str, global_config: Dict[str, Any]) -> Dict[str, Any]:
        logger.debug(f'Loading  component config for: {component_id}')
        
        config = {}
        
        # Layer 1: Pydantic defaults (handled by component instantiator)
        # Layer 2: Default YAML config
        default_config = await self._load_default_component_config(component_spec, component_id)
        config = ConfigMerger.merge(config, default_config, f'{component_id}_defaults')
        
        # Layer 3: Environment-specific YAML config
        env = global_config.get('env', 'default')
        env_config = await self._load_env_component_config(component_spec, component_id, env)
        config = ConfigMerger.merge(config, env_config, f'{component_id}_env_{env}')
        
        # Layer 4 & 5: Manifest config and config_override
        manifest_config = component_spec.get('config', {})
        config_override = component_spec.get('config_override', {})
        merged_manifest = ConfigMerger.merge(manifest_config, config_override, f'{component_id}_manifest')
        config = ConfigMerger.merge(config, merged_manifest, f'{component_id}_with_manifest')
        
        # Layer 6: Environment variable expansion
        config = self._expand_environment_variables(config)
        
        logger.debug(f'✓ Component config resolved for {component_id}: {list(config.keys())}')
        return config

    async def _apply__enhancements(self, base_config: Dict[str, Any], env: str) -> Dict[str, Any]:
        """Apply  enhancements but remove hardcoded defaults - let YAML control these"""
        enhanced = base_config.copy()
        
        # Check if critical keys are present after YAML loading
        if 'bootstrap_strict_mode' not in enhanced:
            logger.warning("Bootstrap strict mode not defined in any config, defaulting to True (system hardcoded).")
            enhanced['bootstrap_strict_mode'] = True  # Last resort default
            
        if 'feature_flags' not in enhanced:
            logger.warning("Feature flags section not defined in any config, defaulting to empty.")
            enhanced['feature_flags'] = {}
        
        # For specific  flags, if they MUST exist, check and add last-resort defaults if absent
        # This is less ideal than them being defined in global_app_config.yaml
        default__flags = {
            'enable_schema_validation': True,  # System considers this critical
            'enable_self_certification': True,
        }
        
        for flag, default_val in default__flags.items():
            if flag not in enhanced['feature_flags']:
                logger.info(f"Feature flag '{flag}' not found in config, applying system default: {default_val}")
                enhanced['feature_flags'][flag] = default_val
                
        return enhanced

    async def _load_default_component_config(self, component_spec: Dict[str, Any], component_id: str) -> Dict[str, Any]:
        config_path_template = component_spec.get('config')
        if not config_path_template:
            return {}
        
        # Replace {id} with actual component_id
        config_path = config_path_template.replace('{id}', component_id)
        full_path = self.package_root / config_path
        
        if not full_path.exists():
            logger.debug(f'No default config found: {full_path}')
            return {}
        
        from bootstrap.bootstrap_helper.utils import load_yaml_robust
        return load_yaml_robust(full_path)

    async def _load_env_component_config(self, component_spec: Dict[str, Any], component_id: str, env: str) -> Dict[str, Any]:
        if env == 'default':
            return {}
        
        config_path_template = component_spec.get('config')
        if not config_path_template:
            return {}
        
        # Replace 'default' with env and {id} with component_id
        config_path = config_path_template.replace('default', env).replace('{id}', component_id)
        full_path = self.package_root / config_path
        
        if not full_path.exists():
            logger.debug(f'No env config found: {full_path}')
            return {}
        
        from bootstrap.bootstrap_helper.utils import load_yaml_robust
        return load_yaml_robust(full_path)

    def _expand_environment_variables(self, config: Any) -> Any:
        """Recursively expand environment variables in configuration"""
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
        
        # Handle ${VAR:-default} syntax
        def repl_default(match):
            var_name = match.group(1)
            default_val = match.group(2)
            env_val = os.getenv(var_name)
            result = env_val if env_val is not None else default_val
            logger.debug(f"Expanding ${{{var_name}:-{default_val}}} -> '{result}'")
            return result
        
        # Apply custom expansion first
        expanded_str = re.sub('\\$\\{([A-Za-z_][A-Za-z0-9_]*):-(.*?)\\}', repl_default, value_str)
        
        # Apply standard expansion
        expanded_str = os.path.expandvars(expanded_str)
        
        if original_value != expanded_str:
            logger.debug(f"Environment expansion: '{original_value}' -> '{expanded_str}'")
        elif '${' in original_value and '}' in original_value:
            logger.warning(f"Unresolved environment variable: '{original_value}'")
        
        return expanded_str

    def _validate_required_config(self, config: Dict[str, Any], env: str) -> None:
        """Validate that required configuration keys are present"""
        required_keys = ['env', 'feature_flags', 'bootstrap_strict_mode']
        missing_keys = [key for key in required_keys if key not in config]
        
        if missing_keys:
            raise ValueError(f"Missing required  config keys for env '{env}': {missing_keys}")
        
        # Validate LLM configuration if present
        if 'llm' in config:
            if not isinstance(config['llm'], dict):
                raise ValueError('LLM configuration must be a dictionary')
            if 'models' not in config['llm'] or not config['llm']['models']:
                raise ValueError("LLM configuration must include 'models' dictionary")
        
        logger.debug(f"✓  configuration validation passed for env '{env}'")