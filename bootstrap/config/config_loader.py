"""
V4 Configuration Loader - Implements 6-layer configuration hierarchy.

Follows the exact precedence order from NIREON V4 Configuration Guide:
1. Runtime Adaptations (highest)
2. Environment Variables  
3. Manifest Overrides
4. Environment Configs
5. Default Configs
6. Python Defaults (lowest)
"""
from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

from configs.loader import load_config
from configs.config_utils import ConfigMerger

logger = logging.getLogger(__name__)

class V4ConfigLoader:
    """
    V4 Configuration Loader implementing the 6-layer hierarchy.
    
    Handles environment variable expansion, config merging, and 
    validation according to V4 Configuration Guide specifications.
    """
    
    def __init__(self):
        self.package_root = Path(__file__).resolve().parents[2]  # nireon_v4 root
        
    async def load_global_config(
        self, 
        env: Optional[str] = None,
        provided_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Load global application configuration following V4 hierarchy.
        
        Args:
            env: Environment name (default, dev, prod, etc.)
            provided_config: Pre-provided config to use instead of loading
            
        Returns:
            Merged configuration dict following 6-layer hierarchy
        """
        if provided_config is not None:
            logger.info("Using provided global configuration")
            return self._expand_environment_variables(provided_config)
        
        env = env or "default"
        logger.info(f"Loading V4 global configuration for environment: {env}")
        
        try:
            # Use existing V4 config loader as foundation (layer 5-6)
            base_config = load_config(env=env)
            
            # Apply V4-specific enhancements
            enhanced_config = await self._apply_v4_enhancements(base_config, env)
            
            # Layer 2: Environment variable expansion
            final_config = self._expand_environment_variables(enhanced_config)
            
            self._validate_required_config(final_config, env)
            
            logger.info(f"✓ V4 global configuration loaded for env='{env}'")
            logger.debug(f"Config keys: {list(final_config.keys())}")
            
            return final_config
            
        except Exception as e:
            logger.error(f"Failed to load V4 global configuration for env '{env}': {e}")
            raise
    
    async def load_component_config(
        self,
        component_spec: Dict[str, Any],
        component_id: str,
        global_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Load component configuration following V4 6-layer hierarchy.
        
        Args:
            component_spec: Component specification from manifest
            component_id: Unique component identifier  
            global_config: Global application configuration
            
        Returns:
            Resolved component configuration
        """
        logger.debug(f"Loading V4 component config for: {component_id}")
        
        # Layer 6: Python defaults (handled by Pydantic)
        config = {}
        
        # Layer 5: Default configs
        default_config = await self._load_default_component_config(component_spec, component_id)
        config = ConfigMerger.merge(config, default_config, f"{component_id}_defaults")
        
        # Layer 4: Environment configs  
        env = global_config.get('env', 'default')
        env_config = await self._load_env_component_config(component_spec, component_id, env)
        config = ConfigMerger.merge(config, env_config, f"{component_id}_env_{env}")
        
        # Layer 3: Manifest overrides
        manifest_config = component_spec.get('config', {})
        config_override = component_spec.get('config_override', {})
        merged_manifest = ConfigMerger.merge(manifest_config, config_override, f"{component_id}_manifest")
        config = ConfigMerger.merge(config, merged_manifest, f"{component_id}_with_manifest")
        
        # Layer 2: Environment variables
        config = self._expand_environment_variables(config)
        
        # Layer 1: Runtime adaptations (handled by component adaptation system)
        
        logger.debug(f"✓ Component config resolved for {component_id}: {list(config.keys())}")
        return config
    
    async def _apply_v4_enhancements(self, base_config: Dict[str, Any], env: str) -> Dict[str, Any]:
        """Apply V4-specific configuration enhancements."""
        enhanced = base_config.copy()
        
        # Ensure V4 bootstrap settings
        if 'bootstrap_strict_mode' not in enhanced:
            enhanced['bootstrap_strict_mode'] = env != 'dev'  # Strict except in dev
            
        # Ensure V4 feature flags structure
        if 'feature_flags' not in enhanced:
            enhanced['feature_flags'] = {}
            
        # Add V4-specific feature flags
        v4_flags = {
            'enable_schema_validation': env in ['prod', 'staging'],
            'enable_self_certification': True,
            'enable_rbac_bootstrap': env == 'prod',
            'enable_component_tracing': env == 'dev'
        }
        
        for flag, default_value in v4_flags.items():
            if flag not in enhanced['feature_flags']:
                enhanced['feature_flags'][flag] = default_value
                
        return enhanced
    
    async def _load_default_component_config(
        self, 
        component_spec: Dict[str, Any], 
        component_id: str
    ) -> Dict[str, Any]:
        """Load default component configuration from configs/default/."""
        config_path_template = component_spec.get('config')
        if not config_path_template:
            return {}
            
        # Handle {id} substitution
        config_path = config_path_template.replace('{id}', component_id)
        full_path = self.package_root / config_path
        
        if not full_path.exists():
            logger.debug(f"No default config found: {full_path}")
            return {}
            
        from bootstrap.bootstrap_helper.utils import load_yaml_robust
        return load_yaml_robust(full_path)
    
    async def _load_env_component_config(
        self, 
        component_spec: Dict[str, Any], 
        component_id: str,
        env: str
    ) -> Dict[str, Any]:
        """Load environment-specific component configuration."""
        if env == 'default':
            return {}
            
        config_path_template = component_spec.get('config')
        if not config_path_template:
            return {}
            
        # Replace default with env and {id} with component_id
        config_path = config_path_template.replace('default', env).replace('{id}', component_id)
        full_path = self.package_root / config_path
        
        if not full_path.exists():
            logger.debug(f"No env config found: {full_path}")
            return {}
            
        from bootstrap.bootstrap_helper.utils import load_yaml_robust
        return load_yaml_robust(full_path)
    
    def _expand_environment_variables(self, config: Any) -> Any:
        """
        Expand environment variables following V4 Configuration Guide patterns.
        
        Supports:
        - ${VAR_NAME} - required variable
        - ${VAR_NAME:-default} - variable with default
        - Recursive expansion in nested structures
        """
        if isinstance(config, dict):
            return {key: self._expand_environment_variables(value) for key, value in config.items()}
        elif isinstance(config, list):
            return [self._expand_environment_variables(item) for item in config]
        elif isinstance(config, str):
            return self._expand_env_var_string(config)
        else:
            return config
    
    def _expand_env_var_string(self, value_str: str) -> str:
        """Expand environment variables in a string."""
        if not isinstance(value_str, str):
            return value_str
            
        original_value = value_str
        
        # Pattern: ${VAR_NAME:-default_value}
        def repl_default(match):
            var_name = match.group(1)
            default_val = match.group(2)
            env_val = os.getenv(var_name)
            result = env_val if env_val is not None else default_val
            logger.debug(f"Expanding ${{{var_name}:-{default_val}}} -> '{result}'")
            return result
        
        # Handle ${VAR:-default} pattern
        expanded_str = re.sub(r'\$\{([A-Za-z_][A-Za-z0-9_]*):-(.*?)\}', repl_default, value_str)
        
        # Handle simple ${VAR} pattern
        expanded_str = os.path.expandvars(expanded_str)
        
        if original_value != expanded_str:
            logger.debug(f"Environment expansion: '{original_value}' -> '{expanded_str}'")
        elif '${' in original_value and '}' in original_value:
            logger.warning(f"Unresolved environment variable: '{original_value}'")
            
        return expanded_str
    
    def _validate_required_config(self, config: Dict[str, Any], env: str) -> None:
        """Validate that required V4 configuration is present."""
        required_keys = ['env', 'feature_flags', 'bootstrap_strict_mode']
        
        missing_keys = [key for key in required_keys if key not in config]
        if missing_keys:
            raise ValueError(f"Missing required V4 config keys for env '{env}': {missing_keys}")
        
        # Validate LLM configuration if present
        if 'llm' in config:
            if not isinstance(config['llm'], dict):
                raise ValueError("LLM configuration must be a dictionary")
            if 'models' not in config['llm'] or not config['llm']['models']:
                raise ValueError("LLM configuration must include 'models' dictionary")
                
        logger.debug(f"✓ V4 configuration validation passed for env '{env}'")