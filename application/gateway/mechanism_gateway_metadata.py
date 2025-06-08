# nireon_v4/application/gateway/mechanism_gateway_metadata.py
from core.lifecycle import ComponentMetadata

MECHANISM_GATEWAY_METADATA = ComponentMetadata(
    id="mechanism_gateway",
    name="Mechanism Gateway",
    version="0.1.0",
    category="service_gateway",
    description="Unified façade for mechanism-initiated cognitive events and service interactions.",
    epistemic_tags=['router', 'policy_enforcer', 'contextualizer', 'facade'],
    capabilities={'route_cognitive_events', 'apply_frame_policies', 'log_episodes', 'manage_service_interaction'},
    accepts=['COGNITIVE_EVENT_LLM_ASK', 'COGNITIVE_EVENT_PUBLISH'],
    produces=['GATEWAY_LLM_EPISODE_COMPLETED', 'GATEWAY_EVENT_EPISODE_COMPLETED'],
    requires_initialize=True,  # Changed to True - ComponentInitializationPhase will handle initialization
    # Add proper dependencies that match actual component IDs
    dependencies={
        "LLMPort": "*",
        "parameter_service_global": ">=1.0.0",
        "frame_factory_service": "*"
    }
)