# nireon_v4/bootstrap/phases/factory_setup_phase.py
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

# Nireon Core/Domain imports
from core.lifecycle import ComponentMetadata
from domain.ports.llm_port import LLMPort
from domain.ports.embedding_port import EmbeddingPort
from domain.ports.event_bus_port import EventBusPort
from domain.ports.idea_service_port import IdeaServicePort
from domain.ports.mechanism_gateway_port import MechanismGatewayPort

# Nireon Application imports
from application.services.idea_service import IdeaService
from application.services.frame_factory_service import FrameFactoryService, FRAME_FACTORY_SERVICE_METADATA
from application.gateway.mechanism_gateway import MechanismGateway
from application.gateway.mechanism_gateway_metadata import MECHANISM_GATEWAY_METADATA # Import the metadata

# Nireon Infrastructure imports
from infrastructure.llm.parameter_service import ParameterService
from infrastructure.llm.router import LLMRouter

# Nireon Factories & Bootstrap specific imports
from .base_phase import BootstrapPhase, PhaseResult
from factories.mechanism_factory import SimpleMechanismFactory
from factories.dependencies import CommonMechanismDependencies
from bootstrap.validators.interface_validator import InterfaceValidator
from bootstrap.bootstrap_helper.context_helper import build_execution_context

logger = logging.getLogger(__name__)

# Define metadata for ParameterService if it's to be registered independently
PARAMETER_SERVICE_METADATA = ComponentMetadata(
    id='parameter_service_global',
    name='Global LLM Parameter Service',
    version='1.0.0',
    category='service_core',
    description='Centralized service for LLM parameter resolution.',
    requires_initialize=True  # Changed to True for consistency
)


class FactorySetupPhase(BootstrapPhase):
    """
    Bootstrap phase responsible for setting up core factories, services, and gateways.
    
    This phase sets up:
    - CommonMechanismDependencies
    - SimpleMechanismFactory  
    - InterfaceValidator
    - ParameterService (for LLM parameter management)
    - FrameFactoryService (for frame lifecycle management)
    - MechanismGateway (unified facade for mechanism interactions)
    
    Note: This phase creates and registers components. The ComponentInitializationPhase
    is responsible for calling initialize() on components where requires_initialize=True.
    """

    async def execute(self, context) -> PhaseResult:
        logger.info('Executing V4 Factory Setup Phase - setting up core factories, services, and gateways')
        errors = []
        warnings = []
        setup_components = []

        try:
            # 1. Setup CommonMechanismDependencies
            common_deps = await self._setup_common_dependencies(context, setup_components, errors, warnings)
            if common_deps is None:
                errors.append("Failed to establish CommonMechanismDependencies, cannot proceed with factory setup.")
                context.common_mechanism_deps = None
                if context.strict_mode:
                    return PhaseResult.failure_result(
                        message="Critical failure: CommonMechanismDependencies setup.", 
                        errors=errors, warnings=warnings
                    )
                else:
                    logger.error("Strict mode OFF: Proceeding without CommonMechanismDependencies. Some factories may fail.")
            else:
                # Ensure context.common_mechanism_deps is available if other parts of bootstrap expect it
                context.common_mechanism_deps = common_deps

            # 2. Setup SimpleMechanismFactory
            mechanism_factory = await self._setup_mechanism_factory(context, common_deps, setup_components, errors, warnings)
            if mechanism_factory:
                if not hasattr(context, 'mechanism_factory'):
                    context.mechanism_factory = mechanism_factory
            else:
                # Always set the attribute to None if setup failed
                context.mechanism_factory = None
                if common_deps is not None and context.strict_mode:
                    return PhaseResult.failure_result(
                        message="MechanismFactory setup failed.", 
                        errors=errors, warnings=warnings
                    )

            # 3. Setup InterfaceValidator
            interface_validator = await self._setup_interface_validator(context, setup_components, errors, warnings)
            if interface_validator:
                context.interface_validator = interface_validator
                logger.info("✓ InterfaceValidator successfully set on context")
            else:
                # Always set the attribute to None if setup failed, so validation can handle it properly
                context.interface_validator = None
                logger.error("✗ InterfaceValidator setup failed - set to None on context")
                if context.strict_mode:
                    logger.error("✗ Strict mode: returning failure due to InterfaceValidator setup failure")
                    return PhaseResult.failure_result(
                        message="InterfaceValidator setup failed in strict mode.", 
                        errors=errors, warnings=warnings
                    )
                else:
                    logger.warning("⚠ Non-strict mode: continuing despite InterfaceValidator failure")

            # 4. Setup ParameterService (no longer initializing here)
            parameter_service_instance = await self._setup_parameter_service(context, setup_components, errors, warnings)
            if parameter_service_instance is None and context.strict_mode:
                errors.append('ParameterService setup failed, which is critical for MechanismGateway.')

            # 5. Setup FrameFactoryService (no longer initializing here)
            frame_factory_service = await self._setup_frame_factory_service(context, setup_components, errors, warnings)
            if frame_factory_service is None and context.strict_mode:
                return PhaseResult.failure_result(
                    message="FrameFactoryService setup failed.", 
                    errors=errors, warnings=warnings
                )

            # 6. Setup MechanismGateway (no longer initializing here)
            if frame_factory_service:
                await self._setup_mechanism_gateway(context, frame_factory_service, setup_components, errors, warnings)
            elif common_deps is not None:
                error_msg = "MechanismGateway setup skipped due to FrameFactoryService failure."
                if context.strict_mode:
                    errors.append(error_msg)
                else:
                    warnings.append(error_msg)
                logger.warning(error_msg)

            # 7. Configure policies and validate setup
            await self._configure_factory_policies(context, setup_components, warnings)
            await self._validate_factory_setup(context, errors)

            success = len(errors) == 0 or (not context.strict_mode and not any("Critical failure" in e for e in errors))
            message = f'Factory setup phase complete - {len(setup_components)} core components/services processed.'
            if errors:
                message += f" Encountered {len(errors)} errors."
            if warnings:
                message += f" Encountered {len(warnings)} warnings."

            return PhaseResult(
                success=success,
                message=message,
                errors=errors,
                warnings=warnings,
                metadata={
                    'factories_setup': setup_components,
                    'factory_system_ready': success,
                    'dependency_injection_enabled': common_deps is not None
                }
            )

        except Exception as e:
            error_msg = f'Critical unhandled error during factory setup phase: {e}'
            logger.error(error_msg, exc_info=True)
            return PhaseResult.failure_result(
                message='Factory setup phase failed critically',
                errors=[error_msg, *errors],
                warnings=warnings,
                metadata={'factory_setup_failed': True}
            )

    async def _setup_common_dependencies(self, context, setup_components: list, errors: list, warnings: list) -> Optional[CommonMechanismDependencies]:
        """Setup CommonMechanismDependencies with core services."""
        try:
            llm_port = await self._resolve_service_safe(context, LLMPort, "LLMPort", errors, warnings, is_critical=True)
            embedding_port = await self._resolve_service_safe(context, EmbeddingPort, "EmbeddingPort", errors, warnings, is_critical=True)
            event_bus = await self._resolve_service_safe(context, EventBusPort, "EventBusPort", errors, warnings, is_critical=False)
            idea_service = await self._resolve_service_safe(context, IdeaServicePort, "IdeaServicePort", errors, warnings, is_critical=True)
            llm_router = llm_port

            if llm_port and embedding_port and idea_service:
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
                setup_components.append('CommonMechanismDependencies')
                logger.info('✓ CommonMechanismDependencies created with core services.')
                return common_deps
            else:
                logger.error("One or more critical services for CommonMechanismDependencies could not be resolved.")
                return None

        except Exception as e:
            error_msg = f'Failed to setup CommonMechanismDependencies: {e}'
            errors.append(error_msg)
            logger.error(error_msg, exc_info=True)
            return None

    async def _resolve_service_safe(self, context, service_type: type, service_name_for_log: str, errors: list, warnings: list, is_critical: bool = True) -> Optional[Any]:
        """Safely resolve a service from the registry with proper error handling."""
        try:
            instance = context.registry.get_service_instance(service_type)
            logger.info(f"✓ Resolved service '{service_name_for_log}' (Type: {service_type.__name__}) from registry.")
            return instance
        except Exception as e:
            msg = f"Failed to resolve service '{service_name_for_log}' (Type: {service_type.__name__}): {e}"
            if is_critical:
                errors.append(msg)
                logger.error(msg)
            else:
                warnings.append(msg)
                logger.warning(msg)
            return None

    async def _setup_mechanism_factory(self, context, common_deps: Optional[CommonMechanismDependencies], setup_components: list, errors: list, warnings: list) -> Optional[SimpleMechanismFactory]:
        """Setup SimpleMechanismFactory with mechanism type mappings."""
        if common_deps is None:
            msg = 'Cannot create SimpleMechanismFactory without CommonMechanismDependencies.'
            if context.strict_mode:
                errors.append(msg)
            else:
                warnings.append(msg)
            logger.warning(msg + " (Skipping SimpleMechanismFactory setup)")
            return None

        try:
            mechanism_factory = SimpleMechanismFactory(common_deps)
            mechanism_mappings = context.global_app_config.get('mechanism_mappings', {})
            for factory_key, class_path in mechanism_mappings.items():
                mechanism_factory.register_mechanism_type(factory_key, class_path)
                logger.info(f'Registered custom mechanism type in SimpleMechanismFactory: {factory_key} -> {class_path}')

            setup_components.append('SimpleMechanismFactory')
            logger.info('✓ SimpleMechanismFactory initialized.')

            if hasattr(context, 'registry_manager'):
                context.registry_manager.register_service_with_certification(
                    service_type=SimpleMechanismFactory,
                    instance=mechanism_factory,
                    service_id='SimpleMechanismFactory',
                    category='factory_service',
                    description='Factory for creating mechanism components from specifications.',
                    requires_initialize=False
                )
            else:
                context.registry.register_service_instance(SimpleMechanismFactory, mechanism_factory)
            return mechanism_factory

        except Exception as e:
            error_msg = f'Failed to setup SimpleMechanismFactory: {e}'
            errors.append(error_msg)
            logger.error(error_msg, exc_info=True)
            return None

    async def _setup_interface_validator(self, context, setup_components: list, errors: list, warnings: list) -> Optional[InterfaceValidator]:
        """Setup InterfaceValidator for component interface validation."""
        logger.info("Attempting to set up InterfaceValidator...")
        
        event_bus = await self._resolve_service_safe(
            context, EventBusPort, "EventBusPort (for InterfaceValidator)", 
            errors, warnings, is_critical=False
        )
        if event_bus is None:
            logger.warning("InterfaceValidator will be created without EventBusPort.")

        try:
            # Check if required dependencies exist
            if not hasattr(context, 'registry') or context.registry is None:
                error_msg = "Cannot create InterfaceValidator: context.registry is missing"
                errors.append(error_msg)
                logger.error(error_msg)
                return None
                
            if not hasattr(context, 'run_id'):
                logger.warning("context.run_id is missing, using default")
                context.run_id = 'bootstrap_default'
            
            validator_exec_context = build_execution_context(
                component_id='bootstrap_validator',
                run_id=context.run_id,
                registry=context.registry,
                event_bus=event_bus
            )
            interface_validator = InterfaceValidator(validator_exec_context)
            setup_components.append('InterfaceValidator')
            logger.info('✓ InterfaceValidator initialized.')

            if hasattr(context, 'registry_manager') and context.registry_manager is not None:
                context.registry_manager.register_service_with_certification(
                    service_type=InterfaceValidator,
                    instance=interface_validator,
                    service_id='InterfaceValidator',
                    category='validation_service',
                    description='Validator for component interfaces and contracts.',
                    requires_initialize=False
                )
                logger.info('✓ InterfaceValidator registered with registry_manager.')
            else:
                logger.warning("registry_manager not available, using direct registry registration")
                context.registry.register_service_instance(InterfaceValidator, interface_validator)
                logger.info('✓ InterfaceValidator registered with direct registry.')
                
            return interface_validator

        except Exception as e:
            error_msg = f'Failed to setup InterfaceValidator: {e}'
            errors.append(error_msg)
            logger.error(error_msg, exc_info=True)
            return None

    async def _setup_parameter_service(self, context, setup_components: list, errors: list, warnings: list) -> Optional[ParameterService]:
        """Setup ParameterService for centralized LLM parameter management.
        
        Note: Creates and registers the service but does NOT initialize it.
        Initialization will be handled by ComponentInitializationPhase.
        """
        logger.info("Attempting to set up ParameterService...")
        try:
            # Config for ParameterService could come from global_app_config or a dedicated section
            param_service_config = context.global_app_config.get('llm', {}).get('parameters', {})

            # Ensure that param_service_config is a dictionary
            if not isinstance(param_service_config, dict):
                logger.warning(f"ParameterService config is not a dictionary (type: {type(param_service_config)}), using empty config.")
                param_service_config = {}

            instance = ParameterService(config=param_service_config, metadata_definition=PARAMETER_SERVICE_METADATA)

            # Register canonically with metadata
            context.registry.register(instance, PARAMETER_SERVICE_METADATA)
            
            # Also register by type for service lookups
            context.registry.register_service_instance(ParameterService, instance)
            
            setup_components.append(PARAMETER_SERVICE_METADATA.id)
            logger.info(f"✓ {PARAMETER_SERVICE_METADATA.name} created and registered (initialization deferred).")
            return instance

        except Exception as e:
            error_msg = f"Failed to setup ParameterService: {e}"
            errors.append(error_msg)
            logger.error(error_msg, exc_info=True)
            return None

    async def _setup_frame_factory_service(self, context, setup_components: list, errors: list, warnings: list) -> Optional[FrameFactoryService]:
        """Setup FrameFactoryService for frame lifecycle management.
        
        Note: Creates and registers the service but does NOT initialize it.
        Initialization will be handled by ComponentInitializationPhase.
        """
        logger.info("Attempting to set up FrameFactoryService...")
        try:
            frame_factory_config = context.global_app_config.get('frame_factory_config', {})
            frame_factory = FrameFactoryService(
                config=frame_factory_config,
                metadata_definition=FRAME_FACTORY_SERVICE_METADATA
            )
            
            # Register canonically with metadata
            context.registry.register(frame_factory, FRAME_FACTORY_SERVICE_METADATA)
            
            # Also register by type
            context.registry.register_service_instance(FrameFactoryService, frame_factory)
            
            setup_components.append(FRAME_FACTORY_SERVICE_METADATA.id)
            logger.info(f"✓ {FRAME_FACTORY_SERVICE_METADATA.name} created and registered (initialization deferred).")
            return frame_factory

        except Exception as e:
            error_msg = f'Failed to setup FrameFactoryService: {e}'
            errors.append(error_msg)
            logger.error(error_msg, exc_info=True)
            return None

    async def _setup_mechanism_gateway(
        self, context, frame_factory_service: FrameFactoryService,
        setup_components: list, errors: list, warnings: list
    ) -> Optional[MechanismGateway]:
        """Setup MechanismGateway as unified facade for mechanism interactions.
        
        Note: Creates and registers the gateway but does NOT initialize it.
        Initialization will be handled by ComponentInitializationPhase because
        MECHANISM_GATEWAY_METADATA.requires_initialize = True.
        """
        logger.info("Attempting to set up MechanismGateway...")
        try:
            llm_router = await self._resolve_service_safe(
                context, LLMPort, "LLMPort (for MechanismGateway)", 
                errors, warnings, is_critical=True
            )
            param_service = await self._resolve_service_safe(
                context, ParameterService, "ParameterService (for MechanismGateway)", 
                errors, warnings, is_critical=True
            )
            event_bus = await self._resolve_service_safe(
                context, EventBusPort, "EventBusPort (for MechanismGateway)", 
                errors, warnings, is_critical=False
            )

            if not llm_router or not param_service:
                error_msg = "MechanismGateway setup failed due to missing critical dependencies (LLMRouter or ParameterService)."
                errors.append(error_msg)
                logger.error(error_msg)
                return None

            gateway_config = context.global_app_config.get('mechanism_gateway_config', {})
            
            # Create the gateway instance
            gateway = MechanismGateway(
                llm_router=llm_router,
                parameter_service=param_service,
                frame_factory=frame_factory_service,
                event_bus=event_bus,
                config=gateway_config,
                metadata_definition=MECHANISM_GATEWAY_METADATA
            )
            
            # Step 1: Register canonically using the instance's metadata
            # NireonBaseComponent sets gateway.metadata from the metadata_definition parameter
            context.registry.register(gateway, gateway.metadata)
            logger.info(f"✓ {gateway.metadata.name} (ID: {gateway.component_id}) canonically registered.")
            
            # Step 2: Register alias for type-based lookup
            context.registry.register_service_instance(MechanismGatewayPort, gateway)
            logger.info(f"✓ {gateway.metadata.name} also registered as MechanismGatewayPort.")
            
            # Note: We do NOT call gateway.initialize() here.
            # ComponentInitializationPhase will handle it because gateway.metadata.requires_initialize = True
            
            setup_components.append(gateway.component_id)
            return gateway

        except Exception as e:
            error_msg = f'Failed to setup MechanismGateway: {e}'
            errors.append(error_msg)
            logger.error(error_msg, exc_info=True)
            return None

    async def _configure_factory_policies(self, context, setup_components: list, warnings: list) -> None:
        """Configure factory-specific policies and settings."""
        logger.debug("Executing _configure_factory_policies (no specific policies for new services in this version).")
        # Future: Add any factory-specific configuration here
        pass

    async def _validate_factory_setup(self, context, errors: list) -> None:
        """Validate that all factory components are properly set up and accessible."""
        try:
            # Check for attributes directly on the context object
            required_context_attrs = [
                ('common_mechanism_deps', 'CommonMechanismDependencies'),
                ('mechanism_factory', 'SimpleMechanismFactory'),
                ('interface_validator', 'InterfaceValidator')
            ]
            
            for attr_name, component_name in required_context_attrs:
                if not hasattr(context, attr_name) or getattr(context, attr_name) is None:
                    msg = f'Context is missing required factory/service attribute: {attr_name} ({component_name})'
                    # Only treat as error if common_deps were expected to be there (i.e., setup was successful)
                    if context.strict_mode and hasattr(context, 'common_mechanism_deps') and context.common_mechanism_deps:
                        errors.append(msg)
                    else:
                        logger.warning(msg + ' (Skipping related validation or non-strict mode)')

            # Check if services are available in the registry
            registry_services_to_check = [
                (ParameterService, "ParameterService"),
                (FrameFactoryService, "FrameFactoryService"),
                (MechanismGatewayPort, "MechanismGateway (via Port)")
            ]

            for service_type, service_name in registry_services_to_check:
                try:
                    context.registry.get_service_instance(service_type)
                    logger.debug(f"✓ {service_name} found in registry.")
                except Exception:
                    errors.append(f"{service_name} not found in registry after setup.")

            # Validate CommonMechanismDependencies if it exists
            if hasattr(context, 'common_mechanism_deps') and context.common_mechanism_deps:
                deps = context.common_mechanism_deps
                if deps.llm_port is None:
                    errors.append('CommonMechanismDependencies missing LLMPort')
                if deps.embedding_port is None:
                    errors.append('CommonMechanismDependencies missing EmbeddingPort')
                if deps.component_registry is None:
                    errors.append('CommonMechanismDependencies missing ComponentRegistry')

            if not errors:
                logger.info('✓ Factory setup validation complete, including new services.')
            else:
                logger.warning(f'Factory setup validation found {len(errors)} issues.')

        except Exception as e:
            error_msg = f'Factory setup validation itself failed: {e}'
            errors.append(error_msg)
            logger.error(error_msg, exc_info=True)

    def should_skip_phase(self, context) -> tuple[bool, str]:
        """Check if this phase should be skipped based on configuration."""
        skip_factories = context.global_app_config.get('skip_factory_setup', False)
        if skip_factories:
            return (True, 'Factory setup phase explicitly disabled in configuration.')
        return (False, '')