# In service.py, add to dependency resolution:

def _resolve_dependencies(self, context: NireonExecutionContext) -> None:
    """Resolve required dependencies from the component registry."""
    
    try:
        from proto_generator.service import ProtoGenerator
        from application.services.idea_service import IdeaService
        from domain.ports.mechanism_gateway_port import MechanismGatewayPort
        from domain.ports.event_bus_port import EventBusPort  # Add this import
        
        if not self.proto_generator:
            self.proto_generator = context.component_registry.get('proto_generator_main')
        if not self.idea_service:
            self.idea_service = context.component_registry.get_service_instance(IdeaService)
        if not self.gateway:
            self.gateway = context.component_registry.get_service_instance(MechanismGatewayPort)
        if not self.event_bus:
            self.event_bus = context.component_registry.get_service_instance(EventBusPort)  # Add this
            
    except Exception as e:
        logger.error(f"Failed to resolve dependencies: {e}")
        raise

# Then in the class __init__:
def __init__(self, config: Dict[str, Any], metadata_definition=None) -> None:
    super().__init__(config, metadata_definition or self.METADATA_DEFINITION)
    
    # Initialize configuration
    self.cfg: QuantifierConfig = self.ConfigModel(**self.config)
    
    # Initialize analysis engine
    self.analysis_engine = QuantificationAnalysisEngine(self.cfg)
    
    # Dependencies (resolved during initialization)
    self.proto_generator = None
    self.idea_service = None
    self.gateway = None
    self.event_bus = None  # Add this

# Update signal emission to use the injected event bus:
async def _handle_non_quantifiable_idea(self, data: Dict[str, Any], 
                                      context: NireonExecutionContext) -> ProcessResult:
    """Handle ideas that cannot be quantified."""
    
    context.logger.info(f"[{self.component_id}] Idea not suitable for quantification")
    
    # Build completion payload
    assessment_details = data.get('assessment_details', {})
    completion_payload = self._build_completion_payload(assessment_details, quantifier_triggered=False)
    
    # Emit completion signal using injected event bus
    completion_signal = GenerativeLoopFinishedSignal(
        source_node_id=self.component_id,
        payload=completion_payload
    )
    
    if self.event_bus:  # Use self.event_bus instead of context.event_bus
        # The EventBusPort.publish is synchronous in the interface
        self.event_bus.publish(completion_signal.signal_type, completion_signal)
            
    return self._create_success_result(
        'Idea was not suitable for quantitative analysis - loop completed gracefully'
    )