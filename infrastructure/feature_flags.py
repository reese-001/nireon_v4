# Adapted from nireon_staging/nireon/infrastructure/feature_flags.py
import logging
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

class FeatureFlagsManager:
    def __init__(self, config: Dict[str, Any] = None):
        self._flags: Dict[str, bool] = {}
        self._descriptions: Dict[str, str] = {}
        self._registered_flags: Set[str] = set() # Keep track of explicitly registered flags

        if config:
            for flag_name, value in config.items():
                if isinstance(value, bool):
                    self._flags[flag_name] = value
                elif isinstance(value, dict) and 'enabled' in value: # V3: Support dict config
                    self._flags[flag_name] = bool(value['enabled'])
                    if 'description' in value:
                        self._descriptions[flag_name] = str(value['description'])
        
        # Register some common flags expected by Nireon components or bootstrap itself
        # These act as defaults if not in config.
        self.register_flag("sentinel_enable_progression_adjustment", default_value=False, description="Enables progression bonus in Sentinel mechanism")
        self.register_flag("sentinel_enable_edge_trust", default_value=False, description="Enables edge trust calculations in Sentinel (if Idea supports graph structure)")
        self.register_flag("enable_exploration", default_value=True, description="Enables the Explorer mechanism to generate idea variations")
        self.register_flag("enable_catalyst", default_value=True, description="Enables the Catalyst mechanism for cross-domain idea blending")
        self.register_flag("catalyst_anti_constraints", default_value=False, description="Enables anti-constraint functionality in Catalyst") # From V3 Catalyst
        self.register_flag("catalyst_duplication_check", default_value=False, description="Enables duplication detection and adaptation in Catalyst") # From V3 Catalyst
        
        logger.info(f"FeatureFlagsManager initialized with {len(self._flags)} flags from config and defaults.")

    def register_flag(self, flag_name: str, default_value: bool = False, description: Optional[str] = None) -> None:
        self._registered_flags.add(flag_name)
        if flag_name not in self._flags: # Only set default if not already loaded from config
            self._flags[flag_name] = default_value
        if description:
            self._descriptions[flag_name] = description
        logger.debug(f"Registered feature flag: {flag_name} (default: {default_value})")

    def is_enabled(self, flag_name: str, default: Optional[bool] = None) -> bool:
        if flag_name in self._flags:
            return self._flags[flag_name]
        
        if default is not None: # If a specific default is passed by the caller
            logger.warning(f"Unregistered feature flag: {flag_name}, using provided default: {default}")
            return default
        
        # Fallback if flag is completely unknown and no default provided at call site
        logger.warning(f"Unregistered feature flag: {flag_name}, defaulting to False")
        return False

    def set_flag(self, flag_name: str, value: bool) -> None:
        if flag_name not in self._registered_flags:
            logger.warning(f"Setting unregistered feature flag: {flag_name}")
            self._registered_flags.add(flag_name) # Add it to registered if set at runtime
        self._flags[flag_name] = bool(value) # Ensure it's a boolean
        logger.info(f"Feature flag {flag_name} set to {value}")

    def get_all_flags(self) -> Dict[str, bool]:
        return dict(self._flags) # Return a copy

    def get_flag_description(self, flag_name: str) -> Optional[str]:
        return self._descriptions.get(flag_name)

    def get_registered_flags(self) -> List[Dict[str, Any]]:
        # Provide a structured list of registered flags and their states
        result = []
        for flag_name in sorted(self._registered_flags): # Iterate over explicitly registered ones
            flag_info = {
                "name": flag_name,
                "enabled": self._flags.get(flag_name, False) # Get current value
            }
            if flag_name in self._descriptions:
                flag_info["description"] = self._descriptions[flag_name]
            result.append(flag_info)
        return result

# Global function for registration, similar to V3's style, but likely not primary API.
# The manager instance is preferred.
def register_flag(flag_name: str, default_value: bool = False, description: Optional[str]=None) -> None:
    # This global function would typically interact with a singleton FeatureFlagsManager instance
    # For now, it's more of a placeholder or for contexts where a global manager is assumed.
    # In V4 bootstrap, we create and use an instance directly.
    logger.info(f"Feature flag registration requested: {flag_name} (default: {default_value})")
    logger.info(f"Description: {description}")