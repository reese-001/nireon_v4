from datetime import datetime, timezone
from core.lifecycle import ComponentMetadata
from core.results import ProcessResult, AnalysisResult

SENTINEL_METADATA = ComponentMetadata(
    id='sentinel_mechanism_default', # This is the class/template ID
    name='SentinelMechanism',
    version='1.1.0', # Version bumped to reflect refactoring
    category='mechanism',
    subcategory='evaluation',
    description='Evaluates ideas on alignment, feasibility, and novelty axes to ensure quality.',
    capabilities={'multi_axis_evaluation', 'threshold_based_filtering', 'weighted_scoring', 'adaptive_thresholds', 'llm_integration'},
    invariants=[
        'Evaluation weights must sum to 1.0.',
        'All axis scores must be between 1.0 and 10.0.',
        'Trust threshold must be between 0.0 and 10.0.',
        'Requires idea text for evaluation.'
    ],
    accepts=['Dict[str, Any]'],
    produces=[ProcessResult.__name__, AnalysisResult.__name__],
    author='Nireon Team',
    created_at=datetime(2023, 10, 20, tzinfo=timezone.utc),
    epistemic_tags=['evaluator', 'quality_control', 'gatekeeper', 'critic'],
    requires_initialize=True,
    dependencies={
        'MechanismGatewayPort': '*',
        'EmbeddingPort': '*',
        'IdeaService': '*',
        'EventBusPort': '>=1.0.0'
    },
    interaction_pattern='processor'
)