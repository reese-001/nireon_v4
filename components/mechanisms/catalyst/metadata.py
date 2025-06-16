# nireon_v4/components/mechanisms/catalyst/metadata.py

from datetime import datetime, timezone
from core.lifecycle import ComponentMetadata # Use actual V4 path

CATALYST_METADATA = ComponentMetadata(
    id='catalyst_mechanism_v4',
    name='Catalyst Mechanism V4',
    version='4.0.0',
    category='mechanism',
    subcategory='generative_blending',
    description='Blends ideas with cross-domain vectors for novelty and creative synthesis using the A->F->CE model.',
    capabilities={
        'configurable_blending',
        'anti_constraints',
        'duplication_detection',
        'frame_based_processing'
    },
    invariants=[
        'Blend strength parameters must be within [0.0, 1.0] and low < high.',
        'Requires cross_domain_vectors for effective operation.'
    ],
    accepts=['CATALYSIS_REQUEST'],
    produces=['IdeaCatalyzedSignal', 'CatalysisCompleteSignal', 'CatalysisFailedSignal'],
    author='Nireon Team',
    created_at=datetime(2024, 5, 21, tzinfo=timezone.utc),
    epistemic_tags=['synthesizer', 'mutator', 'diversity_enhancer', 'cross_pollinator'],
    requires_initialize=True,
    dependencies={'MechanismGatewayPort': '>=1.0.0', 'FrameFactoryService': '*', 'EmbeddingPort': '*'}
)