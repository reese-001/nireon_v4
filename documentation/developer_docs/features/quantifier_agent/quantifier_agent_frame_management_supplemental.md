async def _process_impl(self, data: Any, context: NireonExecutionContext) -> ProcessResult:
    """Enhanced process implementation with proper Frame management."""
    
    logger.info('=== QUANTIFIER AGENT PROCESSING START ===')
    
    # Extract and validate input
    idea_id = data.get('idea_id')
    idea_text = data.get('idea_text')
    
    if not self._validate_input_data(idea_id, idea_text):
        return self._create_error_result("Missing required input: 'idea_id' or 'idea_text'")

    # IMPORTANT: Create a proper Frame for this quantification task
    try:
        frame_factory = context.component_registry.get_service_instance('FrameFactoryService')
        frame = await frame_factory.create_frame(
            agent_id=self.component_id,
            frame_type='QUANTIFICATION_ANALYSIS',
            objective=f"Quantify and visualize idea: {idea_text[:50]}...",
            resource_limit=1000,  # Adjust based on your needs
            metadata={
                'idea_id': idea_id,
                'session_id': data.get('assessment_details', {}).get('metadata', {}).get('session_id'),
                'parent_frame_id': context.metadata.get('current_frame_id')
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
    
    # ... rest of the method remains the same ...