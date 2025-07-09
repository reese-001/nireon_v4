"""
Main QuantifierAgent service implementation with ServiceResolutionMixin.
"""

from __future__ import annotations
import asyncio
import logging
from typing import Any, Dict, Optional, Final
from core.base_component import NireonBaseComponent
from core.results import ProcessResult, ComponentHealth
from domain.context import NireonExecutionContext
from components.mechanisms.base import ProcessorMechanism

from components.service_resolution_mixin import ServiceResolutionMixin
from .config import QuantifierConfig
from .metadata import QUANTIFIER_METADATA
from .analysis_engine import QuantificationAnalysisEngine

__all__ = ['QuantifierAgent']
logger = logging.getLogger(__name__)

class QuantifierAgent(ProcessorMechanism, ServiceResolutionMixin):
    """
    Quantifier is a PROCESSOR mechanism that analyzes ideas for quantification potential.
    It returns analysis results that the Reactor can promote to appropriate signals.
    
    Required Services:
        - proto_generator (ProtoGenerator): For generating Proto blocks
        - idea_service (IdeaService): For idea management
        - gateway (MechanismGatewayPort): For LLM communication
        - frame_factory (FrameFactoryService): For frame management
    """
    
    METADATA_DEFINITION = QUANTIFIER_METADATA
    ConfigModel = QuantifierConfig

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
        self.frame_factory = None
        
        # Metrics
        self._ideas_processed = 0
        self._success_rate = 0.0
        self._avg_llm_latency = 0.0
        self._proto_gen_rate = 0.0
        self._mermaid_usage_rate = 0.0

    async def _initialize_impl(self, context: NireonExecutionContext) -> None:
        """Initialize dependencies and validate configuration."""
        
        # Use the mixin to resolve all dependencies at once
        await self._resolve_all_dependencies(context)
        
        # Validate that all required services are present
        self._validate_dependencies()
        
        # Log configuration
        self._log_configuration(context)
        
        context.logger.info(f"QuantifierAgent '{self.component_id}' initialized successfully.")

    async def _resolve_all_dependencies(self, context: NireonExecutionContext) -> None:
        """Resolve all required dependencies using the ServiceResolutionMixin."""
        
        # Import types here to avoid circular dependencies
        from proto_generator.service import ProtoGenerator
        from application.services.idea_service import IdeaService
        from domain.ports.mechanism_gateway_port import MechanismGatewayPort
        from application.services.frame_factory_service import FrameFactoryService
        
        # Define service mappings
        service_map = {
            'proto_generator': 'proto_generator_main',  # String-based lookup for specific component ID
            'idea_service': IdeaService,                # Type-based lookup
            'gateway': MechanismGatewayPort,           # Type-based lookup
            'frame_factory': FrameFactoryService       # Type-based lookup
        }
        
        try:
            # Resolve all services at once using the mixin
            resolved_services = self.resolve_services(
                context=context,
                service_map=service_map,
                raise_on_missing=True,  # Will raise if any service is missing
                log_resolution=True     # Will log each successful resolution
            )
            
            context.logger.debug(
                f"[{self.component_id}] Successfully resolved {len(resolved_services)} services"
            )
            
        except RuntimeError as e:
            # The mixin will provide a detailed error message about which services failed
            logger.error(f"[{self.component_id}] Failed to resolve dependencies: {e}")
            raise

    def _validate_dependencies(self) -> None:
        """Validate that all required dependencies are available using the mixin."""
        
        required_services = ['proto_generator', 'idea_service', 'gateway', 'frame_factory']
        
        # Use the mixin's validation method
        if not self.validate_required_services(required_services):
            # The mixin will have already logged which services are missing
            missing = [s for s in required_services if not getattr(self, s, None)]
            raise RuntimeError(
                f"QuantifierAgent '{self.component_id}' missing dependencies: {', '.join(missing)}"
            )

    def _log_configuration(self, context: NireonExecutionContext) -> None:
        """Log current configuration for debugging."""
        
        context.logger.info(f"QuantifierAgent configuration:")
        context.logger.info(f"  - LLM approach: {self.cfg.llm_approach}")
        context.logger.info(f"  - Max visualizations: {self.cfg.max_visualizations}")
        context.logger.info(f"  - Mermaid enabled: {self.cfg.enable_mermaid_output}")
        context.logger.info(f"  - Available libraries: {len(sum(self.cfg.available_libraries.values(), []))}")

    async def _process_impl(self, data: Any, context: NireonExecutionContext) -> ProcessResult:
        """
        Process implementation that returns quantification results without publishing signals.
        The Reactor will convert this data into appropriate signals.
        """
        
        logger.info('=== QUANTIFIER AGENT PROCESSING START ===')
        logger.debug(f"Input data keys: {(list(data.keys()) if isinstance(data, dict) else 'Not a dict')}")
        
        # --- NEW: VALIDATION BLOCK ---
        is_valid, error_msg = self._validate_input_data(data)
        if not is_valid:
            logger.error(f"[{self.component_id}] Invalid input data: {error_msg}")
            # Return a failed ProcessResult to immediately halt this branch of logic.
            return ProcessResult(
                success=False,
                component_id=self.component_id,
                message=error_msg,
                error_code='INVALID_INPUT'
            )
        
        idea_id = data.get('idea_id')
        idea_text = data.get('idea_text')
        # --- END: VALIDATION BLOCK ---

        # CRITICAL: Create a proper Frame for this quantification task
        frame = None
        try:
            # Get FrameFactoryService from registry
            if not self.frame_factory:
                logger.error("FrameFactoryService not available")
                return ProcessResult(
                    success=False,
                    component_id=self.component_id,
                    message="FrameFactoryService not available",
                    error_code='MISSING_DEPENDENCY'
                )
                
            # Create a Frame with proper configuration
            frame = await self.frame_factory.create_frame(
                context=context,
                name=f"quantification_analysis_{idea_id}",
                owner_agent_id=self.component_id,
                description=f"Quantify and visualize idea: {idea_text[:50]}...",
                parent_frame_id=context.metadata.get('current_frame_id'),
                epistemic_goals=["quantify", "visualize", "analyze"],
                resource_budget={
                    'max_llm_calls': 3,
                    'max_compute_time': 30,
                    'max_memory_mb': 512
                },
                context_tags={
                    'idea_id': idea_id,
                    'session_id': data.get('assessment_details', {}).get('metadata', {}).get('session_id'),
                    'trust_score': data.get('assessment_details', {}).get('trust_score', 0.0),
                    'analysis_type': 'quantification'
                }
            )
            
            # Update context with the new frame
            context.metadata['current_frame_id'] = frame.id
            
        except Exception as e:
            logger.error(f"Failed to create Frame: {e}")
            return ProcessResult(
                success=False,
                component_id=self.component_id,
                message=f"Frame creation failed: {e}",
                error_code='FRAME_CREATION_ERROR'
            )

        context.logger.info(f"[{self.component_id}] Analyzing idea '{idea_id}' in Frame '{frame.id}'")

        # Perform quantification analysis with proper frame context
        analysis_result = await self.analysis_engine.analyze_idea(idea_text, self.gateway, context)
        
        self._ideas_processed += 1
        
        if not analysis_result or not analysis_result.viable:
            # Idea cannot be quantified - return completion data
            return ProcessResult(
                success=True,
                component_id=self.component_id,
                message='Idea was not suitable for quantitative analysis',
                output_data={
                    'type': 'quantification_complete',
                    'idea_id': idea_id,
                    'quantified': False,
                    'reason': 'not_viable_for_quantification',
                    'assessment_details': data.get('assessment_details', {}),
                    'confidence': getattr(analysis_result, 'confidence', 0.0) if analysis_result else 0.0,
                    'frame_id': frame.id if frame else None,
                    # NEW: Use enum for completion status
                    'completion_status': 'terminal_no_op',  # String value of enum
                    'completion_reason': 'Idea not viable for quantitative analysis'
                }
            )

        # Generate Proto task for quantifiable idea
        return await self._trigger_proto_generation(analysis_result, idea_id, data, context, frame)

    async def _trigger_proto_generation(self, analysis_result, idea_id: str, 
                                      original_data: Dict[str, Any], 
                                      context: NireonExecutionContext,
                                      frame: Any) -> ProcessResult:
        """Trigger Proto block generation for quantifiable ideas."""
        
        context.logger.info(f"[{self.component_id}] Triggering ProtoGenerator for idea '{idea_id}'")
        context.logger.debug(f"[{self.component_id}] Implementation approach: {analysis_result.approach}")
        
        try:
            # Call ProtoGenerator
            generator_result = await self.proto_generator.process(
                {'natural_language_request': analysis_result.implementation_request},
                context
            )
            
            if generator_result.success:
                self._proto_gen_rate = (self._proto_gen_rate * (self._ideas_processed - 1) + 1.0) / self._ideas_processed
                
                # Return success data for Reactor to handle
                return ProcessResult(
                    success=True,
                    component_id=self.component_id,
                    message=f"Successfully triggered quantitative analysis for idea '{idea_id}'",
                    output_data={
                        'type': 'quantification_triggered',
                        'idea_id': idea_id,
                        'quantified': True,
                        'proto_generation_result': generator_result.output_data,
                        'analysis_approach': analysis_result.approach,
                        'libraries_used': analysis_result.libraries,
                        'uses_mermaid': analysis_result.use_mermaid,
                        'confidence': analysis_result.confidence,
                        'assessment_details': original_data.get('assessment_details', {}),
                        'frame_id': frame.id if frame else None,
                        # NEW: Use enum for completion status
                        'completion_status': 'terminal_success',  # String value of enum
                        'completion_reason': 'Quantification completed successfully'
                    }
                )
            else:
                # ProtoGenerator failed - return completion data
                context.logger.warning(f"ProtoGenerator failed: {generator_result.message}")
                return ProcessResult(
                    success=True,
                    component_id=self.component_id,
                    message=f'Proto generation failed for idea {idea_id}',
                    output_data={
                        'type': 'quantification_complete',
                        'idea_id': idea_id,
                        'quantified': False,
                        'reason': 'proto_generation_failed',
                        'error': generator_result.message,
                        'assessment_details': original_data.get('assessment_details', {}),
                        'frame_id': frame.id if frame else None,
                        # NEW: Use enum for completion status
                        'completion_status': 'terminal_failure',  # String value of enum
                        'completion_reason': f'Proto generation failed: {generator_result.message}'
                    }
                )
                
        except Exception as exc:
            logger.exception(f'Error triggering proto generation for idea {idea_id}')
            return ProcessResult(
                success=False,
                component_id=self.component_id,
                message=f'Proto generation failed: {exc}',
                error_code='PROTO_GENERATION_ERROR'
            )

    async def health_check(self, context: NireonExecutionContext) -> ComponentHealth:
        """Report component health and metrics."""
        
        # Check service health
        service_health = self.validate_required_services(
            ['proto_generator', 'idea_service', 'gateway', 'frame_factory'],
            context
        )
        
        # Determine overall health status
        if not service_health:
            status = "unhealthy"
        elif self._ideas_processed > 0 and self._success_rate < 0.5:
            status = "degraded"
        else:
            status = "healthy"
        
        return ComponentHealth(
            component_id=self.component_id,
            status=status,
            metrics={
                "ideas_processed": self._ideas_processed,
                "quantification_success_rate": self._success_rate,
                "average_llm_latency_ms": self._avg_llm_latency,
                "proto_generation_rate": self._proto_gen_rate,
                "mermaid_usage_rate": self._mermaid_usage_rate,
                "all_services_available": service_health
            }
        )

    # --- NEW: VALIDATION METHOD ---
    @staticmethod
    def _validate_input_data(data: Any) -> tuple[bool, str]:
        """Validates that the input data contains all necessary fields."""
        if not isinstance(data, dict):
            return False, "Input data is not a dictionary."
            
        required_fields = {
            'idea_id': str,
            'idea_text': str,
            # 'objective': str, # Could also be made mandatory
        }
        
        missing = [field for field in required_fields if field not in data]
        if missing:
            return False, f"Missing required input fields: {', '.join(missing)}"
            
        for field, expected_type in required_fields.items():
            if not isinstance(data[field], expected_type):
                return False, f"Field '{field}' has wrong type. Expected {expected_type.__name__}, got {type(data[field]).__name__}."
            if expected_type is str and not data[field]:
                return False, f"Field '{field}' is present but empty."
                
        return True, ""

    # Optional: Add a method for lazy service resolution if needed
    def _ensure_services_available(self, context: NireonExecutionContext) -> bool:
        """
        Ensure all required services are available, attempting to resolve them if not.
        This can be used for lazy resolution or re-resolution after failures.
        """
        
        # Check if all services are already available
        required_services = ['proto_generator', 'idea_service', 'gateway', 'frame_factory']
        if self.validate_required_services(required_services):
            return True
        
        # Attempt to re-resolve missing services
        try:
            # Import types
            from proto_generator.service import ProtoGenerator
            from application.services.idea_service import IdeaService
            from domain.ports.mechanism_gateway_port import MechanismGatewayPort
            from application.services.frame_factory_service import FrameFactoryService
            
            # Build service map only for missing services
            service_map = {}
            if not self.proto_generator:
                service_map['proto_generator'] = 'proto_generator_main'
            if not self.idea_service:
                service_map['idea_service'] = IdeaService
            if not self.gateway:
                service_map['gateway'] = MechanismGatewayPort
            if not self.frame_factory:
                service_map['frame_factory'] = FrameFactoryService
            
            if service_map:
                self.resolve_services(
                    context=context,
                    service_map=service_map,
                    raise_on_missing=False,
                    log_resolution=True
                )
            
            # Check again
            return self.validate_required_services(required_services)
            
        except Exception as e:
            context.logger.error(f"[{self.component_id}] Failed to ensure services: {e}")
            return False