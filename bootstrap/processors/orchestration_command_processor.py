"""Main component processor module - imports and re-exports all processor functions for backward compatibility."""

# Import all processor functions
from .simple_component_processor import (
    process_simple_component,
    _build_component_metadata
)
from .shared_service_processor import (
    instantiate_shared_service,
    _safe_register_service_instance_with_port
)
from .orchestration_command_processor import (
    register_orchestration_command
)

# Re-export all functions for backward compatibility
__all__ = [
    'process_simple_component',
    '_build_component_metadata',
    'instantiate_shared_service',
    '_safe_register_service_instance_with_port',
    'register_orchestration_command'
]