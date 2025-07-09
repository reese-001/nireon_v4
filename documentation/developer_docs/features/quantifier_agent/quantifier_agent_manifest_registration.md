# Add to configs/manifests/standard.yaml under the 'mechanisms' section:

mechanisms:
  # ... other mechanisms ...
  
  quantifier_agent_primary:
    enabled: true
    class: "components.mechanisms.analyst_agents.quantifier_agent.service:QuantifierAgent"
    metadata_definition: "components.mechanisms.analyst_agents.quantifier_agent.metadata:QUANTIFIER_METADATA"
    config: "configs/default/mechanisms/{id}.yaml"
    config_override:
      llm_approach: "single_call"
      max_visualizations: 1
      enable_mermaid_output: true
      viability_threshold: 0.7
      
# Also ensure ProtoGenerator is registered if not already:
shared_services:
  # ... other services ...
  
  proto_generator_main:
    enabled: true
    preload: false
    class: "proto_generator.service:ProtoGenerator"
    metadata_definition: "proto_generator.metadata:PROTO_GENERATOR_METADATA"
    config:
      max_retries: 2
      timeout_seconds: 30