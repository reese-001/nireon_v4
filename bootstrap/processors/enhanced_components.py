# nireon_v4/bootstrap/processors/enhanced_components.py
from __future__ import annotations
from __future__ import absolute_import

# This module is now significantly reduced as its core functionalities
# have been moved to bootstrap.utils.component_utils.

# It might still be imported by other processor modules if they expect
# these names to be available from this specific path.
# For now, we re-export the public versions from the utils module
# to maintain compatibility for any internal imports within the processors package
# that might have been using `from .enhanced_components import ...`.

import logging
from typing import Any, Dict, Type, Optional # Keep necessary typings

from core.base_component import NireonBaseComponent
from core.lifecycle import ComponentMetadata

# Import and re-export the consolidated helper functions
from bootstrap.utils.component_utils import (
    get_pydantic_defaults as _get_pydantic_defaults, #
    create_component_instance as _create_component_instance,
    inject_dependencies as _inject_dependencies,
    validate_component_interfaces as _validate_component_interfaces,
    configure_component_logging as _configure_component_logging,
    prepare_component_metadata as _prepare_component_metadata,
)

logger = logging.getLogger(__name__)

# Expose them with their original "private-like" names if other modules in this package
# were importing them that way. This helps minimize changes in other processor files.
# If no other processor files use these directly, this __all__ can be empty or removed.
__all__ = [
    '_get_pydantic_defaults',
    '_create_component_instance',
    '_inject_dependencies',
    '_validate_component_interfaces',
    '_configure_component_logging',
    '_prepare_component_metadata',
]

# The actual implementations are now in bootstrap.utils.component_utils.
# This file serves as a compatibility layer if other modules in the
# 'processors' package were importing these helpers directly from here.
# Ideally, those other modules would eventually be updated to import
# from bootstrap.utils.component_utils directly.