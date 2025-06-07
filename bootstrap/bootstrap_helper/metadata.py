"""
Metadata utilities and default component metadata definitions for NIREON V4 bootstrap.

This module provides utilities for creating component metadata and defines
default metadata for built-in NIREON components.
"""

from typing import Dict, Optional
from core.lifecycle import ComponentMetadata


# Default metadata for Explorer mechanism
EXPLORER_METADATA_DEFAULT = ComponentMetadata(
    id='explorer_mechanism_default',
    name='ExplorerMechanism',
    version='1.0.0',
    category='mechanism',
    epistemic_tags=['mutator', 'innovator'],
    description='Explores idea space through systematic variation and mutation',
    capabilities={'generate_ideas', 'explore_variations', 'idea_mutation'},
    accepts=['SEED_SIGNAL', 'EXPLORATION_REQUEST'],
    produces=['IDEA_GENERATED', 'EXPLORATION_COMPLETE']
)


# Default metadata for Catalyst mechanism
CATALYST_METADATA_DEFAULT = ComponentMetadata(
    id='catalyst_mechanism_default',
    name='CatalystMechanism',
    version='1.0.0',
    category='mechanism',
    epistemic_tags=['synthesizer', 'cross_pollinator'],
    description='Catalyzes idea evolution through cross-domain synthesis and amplification',
    capabilities={'synthesize_ideas', 'cross_domain_blending', 'idea_amplification'},
    accepts=['IDEA_GENERATED', 'SYNTHESIS_REQUEST'],
    produces=['SYNTHESIZED_IDEA', 'CATALYST_COMPLETE']
)


# Default metadata for Sentinel mechanism
SENTINEL_METADATA_DEFAULT = ComponentMetadata(
    id='sentinel_mechanism_default',
    name='SentinelMechanism',
    version='1.0.0',
    category='mechanism',
    epistemic_tags=['evaluator', 'gatekeeper'],
    description='Evaluates and guards idea quality through trust assessment and validation',
    capabilities={'evaluate_trust', 'assess_quality', 'guard_boundaries'},
    accepts=['IDEA_GENERATED', 'TRUST_EVALUATION_REQUEST'],
    produces=['TRUST_ASSESSMENT', 'QUALITY_EVALUATION']
)


# Default metadata for Adversarial Critic
ADVERSARIAL_CRITIC_METADATA_DEFAULT = ComponentMetadata(
    id='adversarial_critic_default',
    name='AdversarialCritic',
    version='1.0.0',
    category='observer',
    epistemic_tags=['critic', 'challenger', 'red_team'],
    description='Provides adversarial critique and challenges to strengthen ideas',
    capabilities={'adversarial_critique', 'challenge_assumptions', 'red_team_analysis'},
    accepts=['IDEA_GENERATED', 'CRITIQUE_REQUEST'],
    produces=['ADVERSARIAL_CRITIQUE', 'CHALLENGE_RESULT']
)


# Default metadata for Lineage Tracker
LINEAGE_TRACKER_METADATA_DEFAULT = ComponentMetadata(
    id='lineage_tracker_default',
    name='LineageTracker',
    version='1.0.0',
    category='observer',
    epistemic_tags=['historian', 'tracer', 'genealogist'],
    description='Tracks and maintains idea evolution lineage and relationships',
    capabilities={'track_lineage', 'maintain_genealogy', 'evolution_history'},
    accepts=['IDEA_GENERATED', 'LINEAGE_UPDATE'],
    produces=['LINEAGE_RECORDED', 'GENEALOGY_UPDATE']
)


# Default metadata for Trust Evaluator
TRUST_EVALUATOR_METADATA_DEFAULT = ComponentMetadata(
    id='trust_evaluator_default',
    name='TrustEvaluator',
    version='1.0.0',
    category='observer',
    epistemic_tags=['evaluator', 'assessor', 'trust_calculator'],
    description='Evaluates and assigns trust scores to ideas and components',
    capabilities={'calculate_trust', 'assess_reliability', 'trust_propagation'},
    accepts=['IDEA_GENERATED', 'TRUST_CALCULATION_REQUEST'],
    produces=['TRUST_SCORE', 'TRUST_ASSESSMENT']
)


# Mapping of component factory keys to their default metadata
DEFAULT_COMPONENT_METADATA_MAP: Dict[str, ComponentMetadata] = {
    'explorer_mechanism': EXPLORER_METADATA_DEFAULT,
    'catalyst_mechanism': CATALYST_METADATA_DEFAULT,
    'sentinel_mechanism': SENTINEL_METADATA_DEFAULT,
    'adversarial_critic': ADVERSARIAL_CRITIC_METADATA_DEFAULT,
    'lineage_tracker': LINEAGE_TRACKER_METADATA_DEFAULT,
    'trust_evaluator': TRUST_EVALUATOR_METADATA_DEFAULT,
}


def get_default_metadata(factory_key_or_class_name: str) -> Optional[ComponentMetadata]:
    """
    Get default metadata for a component by factory key or class name.
    
    Args:
        factory_key_or_class_name: Factory key or class name to look up
        
    Returns:
        ComponentMetadata if found, None otherwise
    """
    return DEFAULT_COMPONENT_METADATA_MAP.get(factory_key_or_class_name)


def create_service_metadata(
    service_id: str,
    service_name: str,
    category: str = 'service',
    description: Optional[str] = None,
    requires_initialize: bool = False
) -> ComponentMetadata:
    """
    Create metadata for a service component.
    
    Args:
        service_id: Unique service identifier
        service_name: Human-readable service name
        category: Service category (default: 'service')
        description: Optional description
        requires_initialize: Whether service needs initialization
        
    Returns:
        ComponentMetadata for the service
    """
    return ComponentMetadata(
        id=service_id,
        name=service_name,
        version='1.0.0',
        category=category,
        description=description or f'Bootstrap-created {service_name}',
        epistemic_tags=[],
        requires_initialize=requires_initialize
    )


def create_mechanism_metadata(
    mechanism_id: str,
    mechanism_name: str,
    epistemic_tags: list[str],
    description: str,
    capabilities: set[str] = None,
    accepts: list[str] = None,
    produces: list[str] = None,
    version: str = '1.0.0'
) -> ComponentMetadata:
    """
    Create metadata for a mechanism component.
    
    Args:
        mechanism_id: Unique mechanism identifier
        mechanism_name: Human-readable mechanism name
        epistemic_tags: List of epistemic tags
        description: Mechanism description
        capabilities: Set of capabilities (optional)
        accepts: List of accepted signal types (optional)
        produces: List of produced signal types (optional)
        version: Version string (default: '1.0.0')
        
    Returns:
        ComponentMetadata for the mechanism
    """
    return ComponentMetadata(
        id=mechanism_id,
        name=mechanism_name,
        version=version,
        category='mechanism',
        description=description,
        epistemic_tags=epistemic_tags,
        capabilities=capabilities or set(),
        accepts=accepts or [],
        produces=produces or [],
        requires_initialize=True
    )


def create_observer_metadata(
    observer_id: str,
    observer_name: str,
    epistemic_tags: list[str],
    description: str,
    capabilities: set[str] = None,
    accepts: list[str] = None,
    produces: list[str] = None,
    version: str = '1.0.0'
) -> ComponentMetadata:
    """
    Create metadata for an observer component.
    
    Args:
        observer_id: Unique observer identifier
        observer_name: Human-readable observer name
        epistemic_tags: List of epistemic tags
        description: Observer description
        capabilities: Set of capabilities (optional)
        accepts: List of accepted signal types (optional)
        produces: List of produced signal types (optional)
        version: Version string (default: '1.0.0')
        
    Returns:
        ComponentMetadata for the observer
    """
    return ComponentMetadata(
        id=observer_id,
        name=observer_name,
        version=version,
        category='observer',
        description=description,
        epistemic_tags=epistemic_tags,
        capabilities=capabilities or set(),
        accepts=accepts or [],
        produces=produces or [],
        requires_initialize=True
    )


def create_manager_metadata(
    manager_id: str,
    manager_name: str,
    description: str,
    capabilities: set[str] = None,
    version: str = '1.0.0'
) -> ComponentMetadata:
    """
    Create metadata for a manager component.
    
    Args:
        manager_id: Unique manager identifier
        manager_name: Human-readable manager name
        description: Manager description
        capabilities: Set of capabilities (optional)
        version: Version string (default: '1.0.0')
        
    Returns:
        ComponentMetadata for the manager
    """
    return ComponentMetadata(
        id=manager_id,
        name=manager_name,
        version=version,
        category='manager',
        description=description,
        epistemic_tags=['coordinator', 'orchestrator'],
        capabilities=capabilities or set(),
        requires_initialize=True
    )


def create_orchestration_command_metadata(
    command_id: str,
    command_name: str,
    description: str,
    version: str = '1.0.0'
) -> ComponentMetadata:
    """
    Create metadata for an orchestration command.
    
    Args:
        command_id: Unique command identifier
        command_name: Human-readable command name
        description: Command description
        version: Version string (default: '1.0.0')
        
    Returns:
        ComponentMetadata for the orchestration command
    """
    return ComponentMetadata(
        id=command_id,
        name=command_name,
        version=version,
        category='orchestration_command',
        description=description,
        epistemic_tags=['command', 'orchestration'],
        requires_initialize=False  # Commands typically don't need initialization
    )


def validate_metadata_consistency(metadata: ComponentMetadata, component_id: str) -> list[str]:
    """
    Validate that component metadata is consistent and complete.
    
    Args:
        metadata: ComponentMetadata to validate
        component_id: Expected component ID
        
    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []
    
    if metadata.id != component_id:
        errors.append(f"Metadata ID '{metadata.id}' does not match component ID '{component_id}'")
    
    if not metadata.name:
        errors.append("Component name cannot be empty")
    
    if not metadata.version:
        errors.append("Component version cannot be empty")
    
    if not metadata.category:
        errors.append("Component category cannot be empty")
    
    # Category-specific validations
    if metadata.category == 'mechanism' and 'mutator' not in metadata.epistemic_tags and 'transformer' not in metadata.epistemic_tags:
        errors.append("Mechanism components should have 'mutator' or 'transformer' epistemic tags")
    
    if metadata.category == 'observer' and 'evaluator' not in metadata.epistemic_tags and 'analyzer' not in metadata.epistemic_tags:
        errors.append("Observer components should have 'evaluator' or 'analyzer' epistemic tags")
    
    return errors