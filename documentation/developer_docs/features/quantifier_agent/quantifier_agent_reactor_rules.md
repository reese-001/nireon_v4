# Add to configs/reactor/rules/advanced.yaml

- id: "route_high_trust_to_quantifier"
  description: "When a high-trust idea is stable, send it to the QuantifierAgent."
  namespace: "synthesis_loop"
  priority: 40
  enabled: true
  conditions:
    - type: "signal_type_match"
      signal_type: "TrustAssessmentSignal"
    - type: "payload_expression"
      expression: "payload.is_stable == True and payload.trust_score > 7.0"
  actions:
    - type: "trigger_component"
      component_id: "quantifier_agent_primary"
      input_data_mapping:
        idea_id: "payload.idea_id"
        idea_text: "payload.idea_text"
        assessment_details: "payload"

# Alternative rule for specific domains
- id: "route_business_idea_to_quantifier"
  description: "Route business-related stable ideas to quantification."
  namespace: "domain_specific"
  priority: 45
  enabled: true
  conditions:
    - type: "signal_type_match"
      signal_type: "TrustAssessmentSignal"
    - type: "payload_expression"
      expression: |
        payload.is_stable == True and 
        payload.trust_score > 6.0 and
        ('business' in lower(payload.idea_text) or 
         'market' in lower(payload.idea_text) or
         'revenue' in lower(payload.idea_text))
  actions:
    - type: "trigger_component"
      component_id: "quantifier_agent_primary"
      input_data_mapping:
        idea_id: "payload.idea_id"
        idea_text: "payload.idea_text"
        assessment_details: "payload"

# Rule for handling ProtoTaskSignal from QuantifierAgent
- id: "quantifier_proto_to_engine"
  description: "Route ProtoTaskSignals from QuantifierAgent to ProtoGateway"
  namespace: "proto_execution"
  priority: 20
  enabled: true
  conditions:
    - type: "signal_type_match"
      signal_type: "ProtoTaskSignal"
    - type: "payload_expression"
      expression: "signal.source_node_id == 'quantifier_agent_primary'"
  actions:
    - type: "trigger_component"
      component_id: "proto_gateway_main"
      input_data_mapping:
        proto_block: "payload.proto_block"
        request_id: "signal.signal_id"