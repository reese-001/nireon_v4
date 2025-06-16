# Re-export functions from component_processor to maintain compatibility
from .component_processor import (
    process_simple_component, 
    _build_component_metadata,
    instantiate_shared_service, 
    _safe_register_service_instance_with_port,
    register_orchestration_command
)

__all__ = [
    'process_simple_component', 
    '_build_component_metadata',
    'instantiate_shared_service', 
    '_safe_register_service_instance_with_port',
    'register_orchestration_command'
]