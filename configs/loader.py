"""
DEPRECATED: This module has been deprecated in favor of bootstrap.config.config_loader.

Please update your imports to use:
    from bootstrap.config.config_loader import ConfigLoader

This file will be removed in a future version.
"""

import warnings
from configs.config_loader import ConfigLoader as NewConfigLoader

# Issue deprecation warning
warnings.warn(
    "configs.loader is deprecated. Use bootstrap.config.config_loader.ConfigLoader instead.",
    DeprecationWarning,
    stacklevel=2
)

# Provide backward compatibility for a transition period
def load_config(env: str = 'default'):
    """Deprecated: Use ConfigLoader().load_global_config() instead"""
    warnings.warn(
        "load_config() is deprecated. Use ConfigLoader().load_global_config() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    import asyncio
    loader = NewConfigLoader()
    return asyncio.run(loader.load_global_config(env=env))