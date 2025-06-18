# nireon_v4/components/mechanisms/catalyst/metadata.py
"""Metadata definition for the Catalyst mechanism."""
from datetime import datetime, timezone
from core.lifecycle import ComponentMetadata

CATALYST_METADATA = ComponentMetadata(
    id='catalyst_mechanism_v4',
    name='Catalyst Mechanism V4',
    version='4.0.0',
    category='mechanism',
    subcategory='generative_blending',
    description=(
        'Cross-Domain Concept Injection for Creative Synthesis. '
        'Blends agent ideas with pre-encoded domain vectors to produce '
        'hybrid representations that span disciplinary boundaries, '
        'enabling interdisciplinary exploration and creative breakthroughs.'
    ),
    capabilities={
        'cross_domain_injection',
        'configurable_blending',
        'anti_constraints',
        'duplication_detection',
        'frame_based_processing',
        'semantic_preservation',
        'hybrid_thinking',
        'interdisciplinary_synthesis'
    },
    invariants=[
        'Blend strength parameters must be within [0.0, 1.0] and low < high.',
        'Requires cross_domain_vectors for effective operation.',
        'Preserves original semantic direction while adding external influence.',
        'Normalized vectors maintained throughout blending process.'
    ],
    accepts=['CATALYSIS_REQUEST', 'CROSS_DOMAIN_INJECTION_REQUEST'],
    produces=[
        'IdeaCatalyzedSignal',
        'CatalysisCompleteSignal',
        'CatalysisFailedSignal',
        'CrossDomainBlendSignal',
        'HybridConceptGeneratedSignal'
    ],
    author='Nireon Team',
    created_at=datetime(2024, 5, 21, tzinfo=timezone.utc),
    epistemic_tags=[
        'synthesizer',
        'mutator',
        'diversity_enhancer',
        'cross_pollinator',
        'interdisciplinary_bridge',
        'creative_engine',
        'hybrid_generator'
    ],
    requires_initialize=True,
    dependencies={
        'MechanismGatewayPort': '>=1.0.0',
        'FrameFactoryService': '*',
        'EmbeddingPort': '*',
        'LLMPort': '*'  # For NL regeneration after blending
    }
)
