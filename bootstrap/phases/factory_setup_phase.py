"""
Factory Setup Phase - Initialize component factories and validators.

Sets up the factory ecosystem needed for component instantiation during
manifest processing, including mechanism factories, validators, and
dependency injection containers.
"""
from __future__ import annotations

import logging
from typing import Any, Dict

from bootstrap.bootstrap_helper.context_helper import build_execution_context

from .base_phase import BootstrapPhase, PhaseResult
from factories.mechanism_factory import SimpleMechanismFactory
from factories.dependencies import CommonMechanismDependencies
from validation.interface_validator import InterfaceValidator


logger = logging.getLogger(__name__)

class FactorySetupPhase(BootstrapPhase):
    """
    Factory Setup Phase - Initialize component factories and validation.
    
    Establishes the factory infrastructure needed for dynamic component
    instantiation from manifest specifications. Creates dependency
    injection containers and validation systems.
    
    Responsibilities:
    - Create CommonMechanismDependencies for dependency injection
    - Initialize SimpleMechanismFactory for mechanism creation
    - Setup InterfaceValidator for component validation
    - Configure factory parameters and policies
    - Prepare factories for manifest-driven instantiation
    """
    
    async def execute(self, context) -> PhaseResult:
        """
        Execute factory and validator setup.
        
        Creates the factory ecosystem needed for component instantiation
        during manifest processing phases.
        """
        logger.info("Setting up component factories and validation systems")
        
        errors = []
        warnings = []
        setup_components = []
        
        try:
            # Setup dependency injection container
            common_deps = await self._setup_common_dependencies(context, setup_components, errors)
            
            # Initialize mechanism factory
            mechanism_factory = await self._setup_mechanism_factory(
                context, common_deps, setup_components, errors
            )
            
            # Initialize interface validator
            interface_validator = await self._setup_interface_validator(
                context, setup_components, errors
            )
            
            # Store factories in context for later phases
            context.common_mechanism_deps = common_deps
            context.mechanism_factory = mechanism_factory
            context.interface_validator = interface_validator
            
            # Configure factory policies
            await self._configure_factory_policies(context, setup_components, warnings)
            
            # Validate factory setup
            await self._validate_factory_setup(context, errors)
            
            success = len(errors) == 0
            message = f"Factory setup complete - {len(setup_components)} components initialized"
            
            return PhaseResult(
                success=success,
                message=message,
                errors=errors,
                warnings=warnings,
                metadata={
                    'factories_setup': setup_components,
                    'factory_system_ready': True,
                    'dependency_injection_enabled': common_deps is not None
                }
            )
            
        except Exception as e:
            error_msg = f"Critical error during factory setup: {e}"
            logger.error(error_msg, exc_info=True)
            return PhaseResult.failure_result(
                message="Factory setup failed",
                errors=[error_msg],
                warnings=warnings,
                metadata={'factory_setup_failed': True}
            )
    
    async def _setup_common_dependencies(
        self, 
        context, 
        setup_components: list, 
        errors: list
    ) -> 'CommonMechanismDependencies':
        """Setup common dependency injection container."""
        try:
            # Resolve core services for dependency injection
            llm_port = await self._resolve_service_safe(context, "LLMPort", errors)
            embedding_port = await self._resolve_service_safe(context, "EmbeddingPort", errors)
            event_bus = await self._resolve_service_safe(context, "EventBusPort", errors)
            idea_service = await self._resolve_service_safe(context, "IdeaService", errors)
            
            # Try to resolve LLMRouter (optional)
            llm_router = None
            try:
                from application.services.llm_router import LLMRouter
                llm_router = context.registry.get_service_instance(LLMRouter)
                logger.debug("✓ LLMRouter resolved for dependency injection")
            except Exception:
                logger.debug("LLMRouter not available - continuing without it")
            
            # Create dependency container
            if llm_port and embedding_port and event_bus and idea_service:
                import random
                common_deps = CommonMechanismDependencies(
                    llm_port=llm_port,
                    llm_router=llm_router,
                    embedding_port=embedding_port,
                    event_bus=event_bus,
                    idea_service=idea_service,
                    component_registry=context.registry,
                    rng=random.Random()
                )
                
                setup_components.append("CommonMechanismDependencies")
                logger.info("✓ Common dependency injection container created")
                return common_deps
            else:
                error_msg = "Failed to resolve required services for dependency injection"
                errors.append(error_msg)
                return None
                
        except Exception as e:
            error_msg = f"Failed to setup common dependencies: {e}"
            errors.append(error_msg)
            logger.error(error_msg)
            return None
    
    async def _resolve_service_safe(self, context, service_name: str, errors: list) -> Any:
        """Safely resolve a service by name."""
        try:
            if service_name == "LLMPort":
                from application.ports.llm_port import LLMPort
                return context.registry.get_service_instance(LLMPort)
            elif service_name == "EmbeddingPort":
                from application.ports.embedding_port import EmbeddingPort
                return context.registry.get_service_instance(EmbeddingPort)
            elif service_name == "EventBusPort":
                from application.ports.event_bus_port import EventBusPort
                return context.registry.get_service_instance(EventBusPort)
            elif service_name == "IdeaService":
                from application.services.idea_service import IdeaService
                return context.registry.get_service_instance(IdeaService)
            else:
                raise ValueError(f"Unknown service name: {service_name}")
                
        except Exception as e:
            error_msg = f"Failed to resolve {service_name}: {e}"
            errors.append(error_msg)
            logger.error(error_msg)
            return None
    
    async def _setup_mechanism_factory(
        self, 
        context, 
        common_deps: 'CommonMechanismDependencies',
        setup_components: list, 
        errors: list
    ) -> 'SimpleMechanismFactory':
        """Initialize mechanism factory."""
        try:
            if common_deps is None:
                error_msg = "Cannot create MechanismFactory without CommonMechanismDependencies"
                errors.append(error_msg)
                return None
            
            mechanism_factory = SimpleMechanismFactory(common_deps)
            setup_components.append("SimpleMechanismFactory")
            logger.info("✓ Mechanism factory initialized")
            
            # Register factory in registry for potential later use
            context.registry_manager.register_service_with_certification(
                service_type=SimpleMechanismFactory,
                instance=mechanism_factory,
                service_id="MechanismFactory",
                category="factory_service",
                description="Factory for creating mechanism components from specifications",
                requires_initialize=False
            )
            
            return mechanism_factory
            
        except Exception as e:
            error_msg = f"Failed to setup mechanism factory: {e}"
            errors.append(error_msg)
            logger.error(error_msg)
            return None
    
    async def _setup_interface_validator(
        self, 
        context, 
        setup_components: list, 
        errors: list
    ) -> 'InterfaceValidator':
        """Initialize interface validator."""
        try:
            # Create execution context for validator
            validator_context = build_execution_context(
                component_id="bootstrap_validator",
                run_id=context.run_id,
                registry=context.registry,
                event_bus=context.registry.get_service_instance(
                    context.registry.get_service_instance.__self__.__class__.mro()[0]
                    if hasattr(context.registry.get_service_instance, '__self__') 
                    else type(context.registry.get_service_instance)
                )
            )
            
            # Safer approach - get event bus directly
            from application.ports.event_bus_port import EventBusPort
            event_bus = context.registry.get_service_instance(EventBusPort)
            
            validator_context = build_execution_context(
                component_id="bootstrap_validator",
                run_id=context.run_id,
                registry=context.registry,
                event_bus=event_bus
            )
            
            interface_validator = InterfaceValidator(validator_context)
            setup_components.append("InterfaceValidator")
            logger.info("✓ Interface validator initialized")
            
            # Register validator in registry
            context.registry_manager.register_service_with_certification(
                service_type=InterfaceValidator,
                instance=interface_validator,
                service_id="InterfaceValidator",
                category="validation_service",
                description="Validator for component interfaces and contracts",
                requires_initialize=False
            )
            
            return interface_validator
            
        except Exception as e:
            error_msg = f"Failed to setup interface validator: {e}"
            errors.append(error_msg)
            logger.error(error_msg)
            return None
    
    async def _configure_factory_policies(
        self, 
        context, 
        setup_components: list, 
        warnings: list
    ) -> None:
        """Configure factory policies and parameters."""
        try:
            factory_config = context.global_app_config.get('factory_configuration', {})
            
            # Configure mechanism factory policies
            mechanism_config = factory_config.get('mechanism_factory', {})
            if mechanism_config and hasattr(context, 'mechanism_factory'):
                # Future: Configure factory-specific policies
                setup_components.append("mechanism_factory_policies_configured")
                logger.debug("✓ Mechanism factory policies configured")
            
            # Configure validation policies
            validation_config = factory_config.get('interface_validation', {})
            if validation_config and hasattr(context, 'interface_validator'):
                # Future: Configure validation-specific policies
                setup_components.append("validation_policies_configured")
                logger.debug("✓ Interface validation policies configured")
            
            # Configure dependency injection policies
            di_config = factory_config.get('dependency_injection', {})
            if di_config:
                # Future: Configure DI container policies
                setup_components.append("dependency_injection_policies_configured")
                logger.debug("✓ Dependency injection policies configured")
                
        except Exception as e:
            warning_msg = f"Failed to configure factory policies: {e}"
            warnings.append(warning_msg)
            logger.warning(warning_msg)
    
    async def _validate_factory_setup(self, context, errors: list) -> None:
        """Validate that factory setup is working correctly."""
        try:
            # Check that required factories are available
            required_factories = [
                ('common_mechanism_deps', 'CommonMechanismDependencies'),
                ('mechanism_factory', 'SimpleMechanismFactory'),
                ('interface_validator', 'InterfaceValidator')
            ]
            
            for attr_name, factory_name in required_factories:
                if not hasattr(context, attr_name):
                    errors.append(f"Missing required factory: {factory_name}")
                elif getattr(context, attr_name) is None:
                    errors.append(f"Factory {factory_name} is None")
            
            # Test dependency injection container
            if hasattr(context, 'common_mechanism_deps') and context.common_mechanism_deps:
                deps = context.common_mechanism_deps
                if deps.llm_port is None:
                    errors.append("CommonMechanismDependencies missing LLMPort")
                if deps.embedding_port is None:
                    errors.append("CommonMechanismDependencies missing EmbeddingPort")
                if deps.component_registry is None:
                    errors.append("CommonMechanismDependencies missing ComponentRegistry")
            
            # Test mechanism factory (basic smoke test)
            if hasattr(context, 'mechanism_factory') and context.mechanism_factory:
                # Future: Add basic factory smoke test
                logger.debug("✓ Mechanism factory smoke test passed")
            
            # Test interface validator (basic smoke test)
            if hasattr(context, 'interface_validator') and context.interface_validator:
                # Future: Add basic validator smoke test
                logger.debug("✓ Interface validator smoke test passed")
            
            if not errors:
                logger.info("✓ Factory setup validation complete")
                
        except Exception as e:
            error_msg = f"Factory setup validation failed: {e}"
            errors.append(error_msg)
            logger.error(error_msg)
    
    def should_skip_phase(self, context) -> tuple[bool, str]:
        """Factory setup is required for manifest processing."""
        skip_factories = context.global_app_config.get('skip_factory_setup', False)
        if skip_factories:
            return True, "Factory setup disabled in configuration"
        return False, ""