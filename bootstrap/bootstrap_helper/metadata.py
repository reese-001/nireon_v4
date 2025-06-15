# nireon_v4/bootstrap/bootstrap_helper/metadata.py

from typing import Dict, Optional
from core.lifecycle import ComponentMetadata


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


DEFAULT_COMPONENT_METADATA_MAP: Dict[str, ComponentMetadata] = {
    'explorer_mechanism': EXPLORER_METADATA_DEFAULT,
    'catalyst_mechanism': CATALYST_METADATA_DEFAULT,
    'sentinel_mechanism': SENTINEL_METADATA_DEFAULT,
    'adversarial_critic': ADVERSARIAL_CRITIC_METADATA_DEFAULT,
    'lineage_tracker': LINEAGE_TRACKER_METADATA_DEFAULT,
    'trust_evaluator': TRUST_EVALUATOR_METADATA_DEFAULT,
}

def get_default_metadata(factory_key_or_class_name: str) -> Optional[ComponentMetadata]:
    return DEFAULT_COMPONENT_METADATA_MAP.get(factory_key_or_class_name)

def create_service_metadata(service_id: str, service_name: str, category: str = 'service', description: Optional[str] = None, requires_initialize: bool = False) -> ComponentMetadata:
    return ComponentMetadata(
        id=service_id,
        name=service_name,
        version='1.0.0',
        category=category,
        description=description or f'Bootstrap-created {service_name}',
        epistemic_tags=[],
        requires_initialize=requires_initialize
    )

def create_mechanism_metadata(mechanism_id: str, mechanism_name: str, epistemic_tags: list[str], description: str, capabilities: set[str] = None, accepts: list[str] = None, produces: list[str] = None, version: str = '1.0.0') -> ComponentMetadata:
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

def create_observer_metadata(observer_id: str, observer_name: str, epistemic_tags: list[str], description: str, capabilities: set[str] = None, accepts: list[str] = None, produces: list[str] = None, version: str = '1.0.0') -> ComponentMetadata:
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

def create_manager_metadata(manager_id: str, manager_name: str, description: str, capabilities: set[str] = None, version: str = '1.0.0') -> ComponentMetadata:
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

def validate_metadata_consistency(metadata: ComponentMetadata, component_id: str) -> list[str]:
    errors = []
    if metadata.id != component_id:
        errors.append(f"Metadata ID '{metadata.id}' does not match component ID '{component_id}'")
    if not metadata.name:
        errors.append('Component name cannot be empty')
    if not metadata.version:
        errors.append('Component version cannot be empty')
    if not metadata.category:
        errors.append('Component category cannot be empty')
    if metadata.category == 'mechanism' and 'mutator' not in metadata.epistemic_tags and 'transformer' not in metadata.epistemic_tags:
        errors.append("Mechanism components should have 'mutator' or 'transformer' epistemic tags")
    if metadata.category == 'observer' and 'evaluator' not in metadata.epistemic_tags and 'analyzer' not in metadata.epistemic_tags:
        errors.append("Observer components should have 'evaluator' or 'analyzer' epistemic tags")
    return errors

def create_orchestration_command_metadata(command_id: str, command_name: str, description: str, version: str = '1.0.0') -> ComponentMetadata:
    return ComponentMetadata(
        id=command_id, name=command_name, version=version, category='orchestration_command',
        description=description, epistemic_tags=['command', 'orchestration'], requires_initialize=False
    )