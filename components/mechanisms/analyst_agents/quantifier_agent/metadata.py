"""
Metadata definition for the QuantifierAgent.
"""

from typing import Final
from core.lifecycle import ComponentMetadata

QUANTIFIER_METADATA: Final = ComponentMetadata(
    id='quantifier_agent_primary',
    name='Quantifier Agent',
    version='3.0.0',
    category='mechanism',
    subcategory='analyst',
    description='Converts qualitative ideas into quantitative analyses using curated visualization libraries',
    epistemic_tags=['analyzer', 'translator', 'modeler', 'quantifier', 'visualizer'],
    accepts=['Dict[str, Any]'],  # As a processor, it accepts data dicts
    produces=['ProcessResult'],  # As a processor, it produces ProcessResult
    requires_initialize=True,
    dependencies={
        'ProtoGenerator': '*', 
        'IdeaService': '*', 
        'MechanismGatewayPort': '*',
        'FrameFactoryService': '*'
    },
    interaction_pattern='processor'
)