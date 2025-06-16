"""
Metadata Module
==============

Provides default metadata definitions for V4 components and factory-based component creation.
Maintains backward compatibility with V3 factory keys while supporting V4 metadata definitions.
"""

from __future__ import annotations
import logging
from typing import Dict, Optional
from core.lifecycle import ComponentMetadata

logger = logging.getLogger(__name__)

# Default metadata mapping for V3 factory keys (backward compatibility)
DEFAULT_FACTORY_METADATA: Dict[str, ComponentMetadata] = {
    'explorer': ComponentMetadata(
        id='explorer',
        name='Explorer Mechanism V4',
        version='4.0.0',
        category='mechanism',
        description='Advanced exploration mechanism with semantic enhancement',
        capabilities=['exploration', 'idea_generation', 'semantic_analysis'],
        epistemic_tags=['generator', 'mutator', 'explorer'],
        dependencies={
            'MechanismGatewayPort': '*',
            'FrameFactoryService': '*'
        },
        requires_initialize=True
    ),
    
    'critic': ComponentMetadata(
        id='critic',
        name='Critic Mechanism V4',
        version='4.0.0',
        category='mechanism',
        description='Advanced criticism and evaluation mechanism',
        capabilities=['criticism', 'evaluation', 'quality_assessment'],
        epistemic_tags=['evaluator', 'critic', 'quality_controller'],
        dependencies={
            'MechanismGatewayPort': '*',
            'FrameFactoryService': '*'
        },
        requires_initialize=True
    ),
    
    'synthesizer': ComponentMetadata(
        id='synthesizer',
        name='Synthesizer Mechanism V4',
        version='4.0.0',
        category='mechanism',
        description='Advanced synthesis and integration mechanism',
        capabilities=['synthesis', 'integration', 'combination'],
        epistemic_tags=['synthesizer', 'integrator', 'combiner'],
        dependencies={
            'MechanismGatewayPort': '*',
            'FrameFactoryService': '*'
        },
        requires_initialize=True
    ),
    
    'orchestrator': ComponentMetadata(
        id='orchestrator',
        name='Orchestrator Mechanism V4',
        version='4.0.0',
        category='mechanism',
        description='Advanced orchestration and coordination mechanism',
        capabilities=['orchestration', 'coordination', 'workflow_management'],
        epistemic_tags=['orchestrator', 'coordinator', 'workflow_manager'],
        dependencies={
            'MechanismGatewayPort': '*',
            'FrameFactoryService': '*'
        },
        requires_initialize=True
    ),
    
    'analyzer': ComponentMetadata(
        id='analyzer',
        name='Analyzer Mechanism V4',
        version='4.0.0',
        category='mechanism',
        description='Advanced analysis and insight generation mechanism',
        capabilities=['analysis', 'insight_generation', 'pattern_recognition'],
        epistemic_tags=['analyzer', 'pattern_recognizer', 'insight_generator'],
        dependencies={
            'MechanismGatewayPort': '*',
            'FrameFactoryService': '*'
        },
        requires_initialize=True
    )
}

# Shared service metadata
DEFAULT_SERVICE_METADATA: Dict[str, ComponentMetadata] = {
    'event_bus_memory': ComponentMetadata(
        id='event_bus_memory',
        name='Memory Event Bus',
        version='4.0.0',
        category='shared_service',
        description='In-memory event bus implementation',
        capabilities=['event_publishing', 'event_subscription', 'event_history'],
        epistemic_tags=['communication', 'event_handling'],
        requires_initialize=True
    ),
    
    'parameter_service_global': ComponentMetadata(
        id='parameter_service_global',
        name='Global Parameter Service',
        version='4.0.0',
        category='shared_service',
        description='Global parameter management service for LLM configurations',
        capabilities=['parameter_management', 'configuration_serving'],
        epistemic_tags=['configuration', 'parameter_management'],
        requires_initialize=False
    ),
    
    'frame_factory_service': ComponentMetadata(
        id='frame_factory_service',
        name='Frame Factory Service',
        version='4.0.0',
        category='shared_service',
        description='Service for creating and managing execution frames',
        capabilities=['frame_creation', 'frame_management', 'lifecycle_management'],
        epistemic_tags=['frame_management', 'lifecycle'],
        requires_initialize=True
    ),
    
    'budget_manager_inmemory': ComponentMetadata(
        id='budget_manager_inmemory',
        name='In-Memory Budget Manager',
        version='4.0.0',
        category='shared_service',
        description='In-memory budget management service',
        capabilities=['budget_management', 'resource_tracking', 'quota_enforcement'],
        epistemic_tags=['resource_management', 'budget_control'],
        requires_initialize=True
    ),
    
    'llm_router_main': ComponentMetadata(
        id='llm_router_main',
        name='Main LLM Router',
        version='4.0.0',
        category='shared_service',
        description='Main LLM routing service with circuit breakers and health checks',
        capabilities=['llm_routing', 'circuit_breaking', 'health_monitoring'],
        epistemic_tags=['llm_management', 'routing', 'resilience'],
        requires_initialize=True
    )
}

# Composite metadata
DEFAULT_COMPOSITE_METADATA: Dict[str, ComponentMetadata] = {
    'mechanism_gateway_main': ComponentMetadata(
        id='mechanism_gateway_main',
        name='Main Mechanism Gateway',
        version='4.1.0',
        category='composite',
        description='Unified entry-point for external service calls made by NIREON mechanisms',
        capabilities=['llm_routing', 'event_publishing', 'budget_enforcement', 'dependency_resolution'],
        epistemic_tags=['gateway', 'service_orchestration', 'dependency_management'],
        dependencies={
            'LLMPort': '*',
            'parameter_service_global': '>=1.0.0',
            'frame_factory_service': '*',
            'BudgetManagerPort': '*',
            'EventBusPort': '*'  # Optional but recommended
        },
        requires_initialize=True
    )
}


def get_default_metadata(factory_key: str) -> Optional[ComponentMetadata]:
    """
    Get default metadata for a factory key or component identifier.
    
    Args:
        factory_key: The factory key or component identifier
        
    Returns:
        ComponentMetadata if found, None otherwise
    """
    if not factory_key:
        return None
        
    # Check factory metadata first
    if factory_key in DEFAULT_FACTORY_METADATA:
        logger.debug(f"Found factory metadata for key: {factory_key}")
        return DEFAULT_FACTORY_METADATA[factory_key]
    
    # Check service metadata
    if factory_key in DEFAULT_SERVICE_METADATA:
        logger.debug(f"Found service metadata for key: {factory_key}")
        return DEFAULT_SERVICE_METADATA[factory_key]
    
    # Check composite metadata
    if factory_key in DEFAULT_COMPOSITE_METADATA:
        logger.debug(f"Found composite metadata for key: {factory_key}")
        return DEFAULT_COMPOSITE_METADATA[factory_key]
    
    logger.debug(f"No default metadata found for key: {factory_key}")
    return None


def register_factory_metadata(factory_key: str, metadata: ComponentMetadata) -> None:
    """
    Register metadata for a factory key.
    
    Args:
        factory_key: The factory key
        metadata: The metadata to register
    """
    DEFAULT_FACTORY_METADATA[factory_key] = metadata
    logger.info(f"Registered factory metadata for key: {factory_key}")


def register_service_metadata(service_key: str, metadata: ComponentMetadata) -> None:
    """
    Register metadata for a service key.
    
    Args:
        service_key: The service key
        metadata: The metadata to register
    """
    DEFAULT_SERVICE_METADATA[service_key] = metadata
    logger.info(f"Registered service metadata for key: {service_key}")


def create_minimal_metadata(
    component_id: str,
    component_name: Optional[str] = None,
    category: str = 'unknown',
    version: str = '1.0.0'
) -> ComponentMetadata:
    """
    Create minimal metadata for a component.
    
    Args:
        component_id: The component ID
        component_name: Optional component name (defaults to component_id)
        category: The component category
        version: The component version
        
    Returns:
        ComponentMetadata instance
    """
    return ComponentMetadata(
        id=component_id,
        name=component_name or component_id,
        version=version,
        category=category,
        description=f"Minimal metadata for {component_id}",
        requires_initialize=True
    )


def update_metadata_for_instance(
    base_metadata: ComponentMetadata,
    instance_id: str,
    overrides: Optional[Dict[str, any]] = None
) -> ComponentMetadata:
    """
    Update metadata for a specific instance.
    
    Args:
        base_metadata: The base metadata
        instance_id: The instance ID
        overrides: Optional metadata overrides
        
    Returns:
        Updated ComponentMetadata instance
    """
    import dataclasses
    
    # Start with base metadata
    metadata_dict = dataclasses.asdict(base_metadata)
    
    # Update ID
    metadata_dict['id'] = instance_id
    
    # Apply overrides
    if overrides:
        for key, value in overrides.items():
            if key in metadata_dict:
                metadata_dict[key] = value
    
    return ComponentMetadata(**metadata_dict)


def get_all_factory_keys() -> list[str]:
    """Get all registered factory keys."""
    return list(DEFAULT_FACTORY_METADATA.keys())


def get_all_service_keys() -> list[str]:
    """Get all registered service keys."""
    return list(DEFAULT_SERVICE_METADATA.keys())


def get_all_composite_keys() -> list[str]:
    """Get all registered composite keys."""
    return list(DEFAULT_COMPOSITE_METADATA.keys())