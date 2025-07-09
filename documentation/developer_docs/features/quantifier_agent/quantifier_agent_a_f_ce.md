# Updated service.py with proper Frame management

async def _process_impl(self, data: Any, context: NireonExecutionContext) -> ProcessResult:
    """Enhanced process implementation with proper Frame management."""
    
    logger.info('=== QUANTIFIER AGENT PROCESSING START ===')
    
    # Extract and validate input
    idea_id = data.get('idea_id')
    idea_text = data.get('idea_text')
    
    if not self._validate_input_data(idea_id, idea_text):
        return self._create_error_result("Missing required input: 'idea_id' or 'idea_text'")

    # CRITICAL: Create a proper Frame for this quantification task
    frame = None
    try:
        # Get FrameFactoryService from registry
        frame_factory = context.component_registry.get_service_instance('FrameFactoryService')
        if not frame_factory:
            logger.error("FrameFactoryService not available")
            return self._create_error_result("FrameFactoryService not available")
            
        # Create a Frame with proper configuration
        frame = await frame_factory.create_frame(
            agent_id=self.component_id,
            frame_type='QUANTIFICATION_ANALYSIS',
            objective=f"Quantify and visualize idea: {idea_text[:50]}...",
            resource_limit=1000,  # Adjust based on your needs
            metadata={
                'idea_id': idea_id,
                'session_id': data.get('assessment_details', {}).get('metadata', {}).get('session_id'),
                'parent_frame_id': context.metadata.get('current_frame_id'),
                'trust_score': data.get('assessment_details', {}).get('trust_score', 0.0)
            }
        )
        
        # Update context with the new frame
        context.metadata['current_frame_id'] = frame.id
        
    except Exception as e:
        logger.error(f"Failed to create Frame: {e}")
        return self._create_error_result(f"Frame creation failed: {e}")

    context.logger.info(f"[{self.component_id}] Analyzing idea '{idea_id}' in Frame '{frame.id}'")

    # Perform quantification analysis with proper frame context
    analysis_result = await self.analysis_engine.analyze_idea(idea_text, self.gateway, context)
    
    if not analysis_result or not analysis_result.viable:
        # Idea cannot be quantified - complete the loop gracefully
        return await self._handle_non_quantifiable_idea(data, context)

    # Generate Proto task for quantifiable idea
    return await self._trigger_proto_generation(analysis_result, idea_id, data, context)

# Also update dependency resolution to include FrameFactoryService
def _resolve_dependencies(self, context: NireonExecutionContext) -> None:
    """Resolve required dependencies from the component registry."""
    
    try:
        from proto_generator.service import ProtoGenerator
        from application.services.idea_service import IdeaService
        from domain.ports.mechanism_gateway_port import MechanismGatewayPort
        from domain.ports.event_bus_port import EventBusPort
        from application.services.frame_factory_service import FrameFactoryService
        
        if not self.proto_generator:
            self.proto_generator = context.component_registry.get('proto_generator_main')
        if not self.idea_service:
            self.idea_service = context.component_registry.get_service_instance(IdeaService)
        if not self.gateway:
            self.gateway = context.component_registry.get_service_instance(MechanismGatewayPort)
        if not self.event_bus:
            self.event_bus = context.component_registry.get_service_instance(EventBusPort)
        if not self.frame_factory:
            self.frame_factory = context.component_registry.get_service_instance(FrameFactoryService)
            
    except Exception as e:
        logger.error(f"Failed to resolve dependencies: {e}")
        raise