"""
Abiogenesis Phase - L0 System Emergence from Configuration.

Implements the foundational L0 Abiogenesis layer by bringing the most essential
system components into existence from unstructured configuration origins.
"""
from __future__ import annotations

import logging
from typing import Any, Dict

from .base_phase import BootstrapPhase, PhaseResult
from infrastructure.feature_flags import FeatureFlagsManager
from bootstrap.bootstrap_helper.placeholders import (
    PlaceholderLLMPortImpl,
    PlaceholderEmbeddingPortImpl, 
    PlaceholderEventBusImpl,
    PlaceholderIdeaRepositoryImpl
)
from application.ports.llm_port import LLMPort
from application.ports.embedding_port import EmbeddingPort
from application.ports.event_bus_port import EventBusPort
from application.ports.idea_repository_port import IdeaRepositoryPort
from application.services.idea_service import IdeaService

logger = logging.getLogger(__name__)

class AbiogenesisPhase(BootstrapPhase):
    """
    L0 Abiogenesis Phase - System emergence from configuration origin.
    
    This phase represents the fundamental emergence of the NIREON system
    from unstructured configuration, bringing essential services into
    existence to enable all subsequent component instantiation.
    
    Responsibilities:
    - Register FeatureFlagsManager for system-wide feature control
    - Establish core service ports with placeholders initially
    - Set up fundamental IdeaService for epistemic operations
    - Register ComponentRegistry with itself for self-reference
    - Prepare foundation for all other bootstrap phases
    """
    
    async def execute(self, context) -> PhaseResult:
        """
        Execute L0 Abiogenesis - bring core system services into existence.
        
        This is the moment of system emergence from configuration chaos
        into structured, discoverable services.
        """
        logger.info("Executing L0 Abiogenesis - System emergence from configuration origin")
        
        errors = []
        warnings = []
        created_services = []
        
        try:
            # Step 1: Emerge FeatureFlagsManager (system control emergence)
            await self._emerge_feature_flags_manager(context, created_services, errors)
            
            # Step 2: Self-register ComponentRegistry (reflexive emergence)
            await self._emerge_component_registry(context, created_services, errors)
            
            # Step 3: Emerge core service ports (interface emergence)
            await self._emerge_core_ports(context, created_services, errors, warnings)
            
            # Step 4: Emerge IdeaService (epistemic capability emergence)
            await self._emerge_idea_service(context, created_services, errors)
            
            # Emit L0 emergence signal
            await self._emit_abiogenesis_signal(context, created_services)
            
            success = len(errors) == 0
            message = f"L0 Abiogenesis complete - {len(created_services)} essential services emerged"
            
            return PhaseResult(
                success=success,
                message=message,
                errors=errors,
                warnings=warnings,
                metadata={
                    'services_emerged': created_services,
                    'l0_abiogenesis': True,
                    'emergence_count': len(created_services)
                }
            )
            
        except Exception as e:
            error_msg = f"Critical failure during L0 Abiogenesis: {e}"
            logger.error(error_msg, exc_info=True)
            return PhaseResult.failure_result(
                message="L0 Abiogenesis failed",
                errors=[error_msg],
                warnings=warnings,
                metadata={'abiogenesis_failed': True}
            )
    
    async def _emerge_feature_flags_manager(
        self, 
        context, 
        created_services: list, 
        errors: list
    ) -> None:
        """Emerge FeatureFlagsManager for system-wide feature control."""
        try:
            # Check if already exists
            try:
                existing_ff = context.registry.get_service_instance(FeatureFlagsManager)
                logger.info("FeatureFlagsManager already exists in registry")
                return
            except:
                pass  # Not found, need to create
            
            # Create FeatureFlagsManager from global config
            feature_flags_config = context.global_app_config.get('feature_flags', {})
            ff_manager = FeatureFlagsManager(feature_flags_config)
            
            # Register with certification
            context.registry_manager.register_service_with_certification(
                service_type=FeatureFlagsManager,
                instance=ff_manager,
                service_id="FeatureFlagsManager",
                category="core_service",
                description="System-wide feature flag management for adaptive behavior",
                requires_initialize=False
            )
            
            created_services.append("FeatureFlagsManager")
            logger.info("✓ FeatureFlagsManager emerged with system feature control")
            
        except Exception as e:
            error_msg = f"Failed to emerge FeatureFlagsManager: {e}"
            errors.append(error_msg)
            logger.error(error_msg)
    
    async def _emerge_component_registry(
        self, 
        context, 
        created_services: list, 
        errors: list
    ) -> None:
        """Register ComponentRegistry with itself (reflexive emergence)."""
        try:
            # Self-register the registry for reflexive discovery
            context.registry_manager.register_service_with_certification(
                service_type=type(context.registry),
                instance=context.registry,
                service_id="component_registry",
                category="core_service", 
                description="Central component registry enabling system self-awareness",
                requires_initialize=False
            )
            
            created_services.append("ComponentRegistry")
            logger.info("✓ ComponentRegistry achieved reflexive self-emergence")
            
        except Exception as e:
            error_msg = f"Failed to register ComponentRegistry reflexively: {e}"
            errors.append(error_msg)
            logger.error(error_msg)
    
    async def _emerge_core_ports(
        self, 
        context, 
        created_services: list, 
        errors: list,
        warnings: list
    ) -> None:
        """Emerge core service ports with placeholder implementations."""
        
        port_configs = [
            (LLMPort, PlaceholderLLMPortImpl, "LLMPort", "Language model interface for epistemic reasoning"),
            (EmbeddingPort, PlaceholderEmbeddingPortImpl, "EmbeddingPort", "Vector embedding interface for semantic operations"),
            (EventBusPort, PlaceholderEventBusImpl, "EventBusPort", "Event publication and subscription for system communication"),
            (IdeaRepositoryPort, PlaceholderIdeaRepositoryImpl, "IdeaRepositoryPort", "Idea storage and retrieval for epistemic persistence")
        ]
        
        for port_type, placeholder_impl, service_id, description in port_configs:
            try:
                await self._emerge_service_port(
                    context, port_type, placeholder_impl, service_id, description, created_services
                )
            except Exception as e:
                error_msg = f"Failed to emerge {service_id}: {e}"
                if context.strict_mode:
                    errors.append(error_msg)
                else:
                    warnings.append(error_msg)
                logger.error(error_msg)
    
    async def _emerge_service_port(
        self,
        context,
        port_type,
        placeholder_impl,
        service_id: str,
        description: str,
        created_services: list
    ) -> None:
        """Emerge a specific service port."""
        # Check if external instance provided
        if service_id == "EventBusPort" and context.config.existing_event_bus:
            instance = context.config.existing_event_bus
            logger.info(f"Using provided {service_id} instance")
        else:
            # Use placeholder implementation
            instance = placeholder_impl()
            logger.info(f"Created placeholder {service_id}")
        
        # Register with certification
        context.registry_manager.register_service_with_certification(
            service_type=port_type,
            instance=instance,
            service_id=service_id,
            category="port_service",
            description=description,
            requires_initialize=False
        )
        
        created_services.append(service_id)
        logger.debug(f"✓ {service_id} emerged as system interface")
    
    async def _emerge_idea_service(
        self, 
        context, 
        created_services: list, 
        errors: list
    ) -> None:
        """Emerge IdeaService for epistemic operations."""
        try:
            # Check if already exists
            try:
                existing_idea_service = context.registry.get_service_instance(IdeaService)
                logger.info("IdeaService already exists in registry")
                return
            except:
                pass  # Not found, need to create
            
            # Get required dependencies
            idea_repo = context.registry.get_service_instance(IdeaRepositoryPort)
            event_bus = context.registry.get_service_instance(EventBusPort)
            
            # Create IdeaService
            idea_service = IdeaService(repository=idea_repo, event_bus=event_bus)
            
            # Register with certification
            context.registry_manager.register_service_with_certification(
                service_type=IdeaService,
                instance=idea_service,
                service_id="IdeaService",
                category="domain_service",
                description="Core epistemic service for idea creation, storage, and evolution",
                requires_initialize=False
            )
            
            created_services.append("IdeaService") 
            logger.info("✓ IdeaService emerged with epistemic capabilities")
            
        except Exception as e:
            error_msg = f"Failed to emerge IdeaService: {e}"
            errors.append(error_msg)
            logger.error(error_msg)
    
    async def _emit_abiogenesis_signal(self, context, created_services: list) -> None:
        """Emit L0 Abiogenesis completion signal."""
        try:
            from signals.bootstrap_signals import L0_ABIOGENESIS_COMPLETE
            
            await context.signal_emitter.emit_signal(
                signal_type=L0_ABIOGENESIS_COMPLETE,
                payload={
                    'services_emerged': created_services,
                    'emergence_count': len(created_services),
                    'l0_complete': True,
                    'run_id': context.run_id
                }
            )
            
        except Exception as e:
            # Non-critical - don't fail the phase
            logger.warning(f"Failed to emit L0 Abiogenesis signal: {e}")
    
    def should_skip_phase(self, context) -> tuple[bool, str]:
        """L0 Abiogenesis is never skipped - it's the foundation of existence."""
        return False, ""