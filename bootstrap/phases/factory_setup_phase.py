# nireon_v4/bootstrap/phases/factory_setup_phase.py
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone

# Nireon Core/Domain imports
from core.lifecycle import ComponentMetadata
from core.results import (
    ProcessResult, SystemSignal, SignalType, 
    ResultCollector, create_success_result, create_error_result
)
from domain.ports.llm_port import LLMPort
from domain.ports.embedding_port import EmbeddingPort
from domain.ports.event_bus_port import EventBusPort
from domain.ports.idea_service_port import IdeaServicePort
from domain.ports.mechanism_gateway_port import MechanismGatewayPort
from domain.context import NireonExecutionContext

# Nireon Application imports
from application.services.idea_service import IdeaService
from application.services.frame_factory_service import FrameFactoryService, FRAME_FACTORY_SERVICE_METADATA
from application.services.budget_manager import BudgetManagerPort, InMemoryBudgetManager  # Make sure InMemoryBudgetManager is imported
from application.gateway.mechanism_gateway import MechanismGateway
from application.gateway.mechanism_gateway_metadata import MECHANISM_GATEWAY_METADATA

# Nireon Infrastructure imports
from infrastructure.llm.parameter_service import ParameterService
from infrastructure.llm.router import LLMRouter

# Nireon Factories & Bootstrap specific imports
from .base_phase import BootstrapPhase, PhaseResult
from factories.mechanism_factory import SimpleMechanismFactory
from factories.dependencies import CommonMechanismDependencies
from bootstrap.validators.interface_validator import InterfaceValidator
from bootstrap.bootstrap_helper.context_helper import build_execution_context

from core.lifecycle import ComponentRegistryMissingError
logger = logging.getLogger(__name__)

# Enhanced metadata for ParameterService
PARAMETER_SERVICE_METADATA = ComponentMetadata(
    id='parameter_service_global',
    name='Global LLM Parameter Service',
    version='1.0.0',
    category='service_core',
    description='Centralized service for LLM parameter resolution.',
    requires_initialize=True,
    capabilities={'parameter_resolution', 'config_management', 'stage_routing'},
    produces=['parameter_set', 'routing_config']
)


class FactorySetupPhase(BootstrapPhase):
    """Enhanced Factory Setup Phase with result tracking and dependency validation."""

    def __init__(self):
        super().__init__()
        self.result_collector = ResultCollector()
        self.phase_start_time = None

    async def execute(self, context) -> PhaseResult:
        logger.info('Executing Enhanced V4 Factory Setup Phase')
        self.phase_start_time = datetime.now(timezone.utc)
        
        # Track phase correlation
        phase_correlation_id = f"factory_setup_{self.phase_start_time.isoformat()}"
        
        setup_components = []

        try:
            # 1. Setup CommonMechanismDependencies
            deps_result = await self._setup_common_dependencies(context, phase_correlation_id)
            self.result_collector.add(deps_result)
            
            if deps_result.success:
                context.common_mechanism_deps = deps_result.output_data
                setup_components.append('CommonMechanismDependencies')
            else:
                context.common_mechanism_deps = None
                if context.strict_mode:
                    return self._create_phase_result()

            # 2. Setup SimpleMechanismFactory
            factory_result = await self._setup_mechanism_factory(
                context, context.common_mechanism_deps, phase_correlation_id
            )
            self.result_collector.add(factory_result)
            
            if factory_result.success:
                context.mechanism_factory = factory_result.output_data
                setup_components.append('SimpleMechanismFactory')
            else:
                context.mechanism_factory = None
                if context.strict_mode and context.common_mechanism_deps:
                    return self._create_phase_result()

            # 3. Setup InterfaceValidator
            validator_result = await self._setup_interface_validator(context, phase_correlation_id)
            self.result_collector.add(validator_result)
            
            if validator_result.success:
                context.interface_validator = validator_result.output_data
                setup_components.append('InterfaceValidator')
            else:
                context.interface_validator = None
                if context.strict_mode:
                    return self._create_phase_result()

            # 4. Setup ParameterService
            param_result = await self._setup_parameter_service(context, phase_correlation_id)
            self.result_collector.add(param_result)
            
            if not param_result.success and context.strict_mode:
                return self._create_phase_result()
            elif param_result.success:
                setup_components.append(PARAMETER_SERVICE_METADATA.id)

            # 5. Setup FrameFactoryService
            frame_result = await self._setup_frame_factory_service(context, phase_correlation_id)
            self.result_collector.add(frame_result)
            
            if not frame_result.success and context.strict_mode:
                return self._create_phase_result()
            elif frame_result.success:
                setup_components.append(FRAME_FACTORY_SERVICE_METADATA.id)

            # ***** NEW: Setup Budget Manager BEFORE Mechanism Gateway *****
            budget_manager_result = await self._setup_budget_manager(context, phase_correlation_id)
            self.result_collector.add(budget_manager_result)
            if not budget_manager_result.success and context.strict_mode:
                return self._create_phase_result()
            elif budget_manager_result.success:
                # Ensure the BudgetManager instance is available on the context if needed later,
                # or rely on registry.get_service_instance(BudgetManagerPort)
                # For MechanismGateway, it will pull from the registry.
                setup_components.append('BudgetManagerPort_Instance')

            # 6. Setup MechanismGateway (NOW it can be created)
            if frame_result.success and budget_manager_result.success:  # Check budget_manager_result too
                # The gateway will fetch FrameFactoryService and BudgetManagerPort from the registry
                gateway_result = await self._setup_mechanism_gateway(
                    context,  # Pass the whole context
                    phase_correlation_id
                )
                self.result_collector.add(gateway_result)
                
                if gateway_result.success:
                    setup_components.append(MECHANISM_GATEWAY_METADATA.id)
            else:
                logger.warning("MechanismGateway setup skipped due to FrameFactoryService or BudgetManager failure")

            # 7. Validate setup and check dependencies
            validation_result = await self._validate_factory_setup(context, phase_correlation_id)
            self.result_collector.add(validation_result)

            return self._create_phase_result(setup_components)

        except Exception as e:
            error_result = create_error_result("factory_setup_phase", e)
            error_result.set_correlation_id(phase_correlation_id)
            self.result_collector.add(error_result)
            
            # Emit critical signal
            critical_signal = SystemSignal(
                signal_type=SignalType.CRITICAL,
                component_id="factory_setup_phase",
                message=f"Critical failure in factory setup: {e}",
                payload={"exception_type": type(e).__name__},
                requires_acknowledgment=True
            )
            
            if hasattr(context, 'event_bus') and context.event_bus:
                await context.event_bus.publish(
                    critical_signal.to_event()["event_type"],
                    critical_signal.to_event()
                )
            
            return self._create_phase_result()

    async def _setup_common_dependencies(self, context, correlation_id: str) -> ProcessResult:
        """Setup CommonMechanismDependencies with result tracking."""
        try:
            # Resolve services
            services = {}
            service_checks = [
                (LLMPort, "LLMPort", True),
                (EmbeddingPort, "EmbeddingPort", True),
                (EventBusPort, "EventBusPort", False),
                (IdeaServicePort, "IdeaServicePort", True)
            ]
            
            for service_type, name, is_critical in service_checks:
                try:
                    instance = context.registry.get_service_instance(service_type)
                    services[name] = instance
                    logger.info(f"✓ Resolved {name}")
                except Exception as e:
                    if is_critical:
                        error_result = create_error_result(
                            "common_dependencies",
                            RuntimeError(f"Failed to resolve critical service {name}: {e}")
                        )
                        error_result.set_correlation_id(correlation_id)
                        return error_result
                    else:
                        services[name] = None
                        logger.warning(f"Optional service {name} not available: {e}")

            # Create dependencies object
            import random
            common_deps = CommonMechanismDependencies(
                llm_port=services["LLMPort"],
                llm_router=services["LLMPort"],  # Same as llm_port in this context
                embedding_port=services["EmbeddingPort"],
                event_bus=services.get("EventBusPort"),
                idea_service=services["IdeaServicePort"],
                component_registry=context.registry,
                rng=random.Random()
            )
            
            result = create_success_result(
                "common_dependencies",
                "CommonMechanismDependencies created successfully",
                output_data=common_deps
            )
            result.set_correlation_id(correlation_id)
            result.add_metric("services_resolved", len([s for s in services.values() if s]))
            
            logger.info('✓ CommonMechanismDependencies created with core services')
            return result

        except Exception as e:
            error_result = create_error_result("common_dependencies", e)
            error_result.set_correlation_id(correlation_id)
            return error_result

    async def _setup_mechanism_factory(
        self, context, common_deps: Optional[CommonMechanismDependencies], 
        correlation_id: str
    ) -> ProcessResult:
        """Setup SimpleMechanismFactory with enhanced tracking."""
        if common_deps is None:
            return create_error_result(
                "mechanism_factory",
                RuntimeError("Cannot create SimpleMechanismFactory without CommonMechanismDependencies")
            )

        try:
            mechanism_factory = SimpleMechanismFactory(common_deps)
            
            # Register custom mechanism types
            mechanism_mappings = context.global_app_config.get('mechanism_mappings', {})
            for factory_key, class_path in mechanism_mappings.items():
                mechanism_factory.register_mechanism_type(factory_key, class_path)
                logger.info(f'Registered mechanism type: {factory_key} -> {class_path}')

            # Register in registry
            if hasattr(context, 'registry_manager'):
                context.registry_manager.register_service_with_certification(
                    service_type=SimpleMechanismFactory,
                    instance=mechanism_factory,
                    service_id='SimpleMechanismFactory',
                    category='factory_service',
                    description='Factory for creating mechanism components.',
                    requires_initialize=False
                )
            else:
                context.registry.register_service_instance(SimpleMechanismFactory, mechanism_factory)

            result = create_success_result(
                "mechanism_factory",
                "SimpleMechanismFactory initialized successfully",
                output_data=mechanism_factory
            )
            result.set_correlation_id(correlation_id)
            result.add_metric("mechanism_types_registered", len(mechanism_mappings))
            
            logger.info('✓ SimpleMechanismFactory initialized')
            return result

        except Exception as e:
            error_result = create_error_result("mechanism_factory", e)
            error_result.set_correlation_id(correlation_id)
            return error_result

    async def _setup_interface_validator(self, context, correlation_id: str) -> ProcessResult:
        """Setup InterfaceValidator with enhanced error handling."""
        try:
            # Try to get event bus (optional)
            event_bus = None
            try:
                event_bus = context.registry.get_service_instance(EventBusPort)
            except Exception:
                logger.warning("EventBus not available for InterfaceValidator")

            # Build execution context
            if not hasattr(context, 'run_id'):
                context.run_id = 'bootstrap_default'
            
            validator_exec_context = build_execution_context(
                component_id='bootstrap_validator',
                run_id=context.run_id,
                registry=context.registry,
                event_bus=event_bus
            )
            
            interface_validator = InterfaceValidator(validator_exec_context)
            
            # Register validator
            if hasattr(context, 'registry_manager') and context.registry_manager:
                context.registry_manager.register_service_with_certification(
                    service_type=InterfaceValidator,
                    instance=interface_validator,
                    service_id='InterfaceValidator',
                    category='validation_service',
                    description='Validator for component interfaces.',
                    requires_initialize=False
                )
            else:
                context.registry.register_service_instance(InterfaceValidator, interface_validator)

            result = create_success_result(
                "interface_validator",
                "InterfaceValidator initialized successfully",
                output_data=interface_validator
            )
            result.set_correlation_id(correlation_id)
            
            logger.info('✓ InterfaceValidator initialized')
            return result

        except Exception as e:
            error_result = create_error_result("interface_validator", e)
            error_result.set_correlation_id(correlation_id)
            return error_result

    async def _setup_parameter_service(self, context, correlation_id: str) -> ProcessResult:
        """Setup ParameterService with enhanced metadata."""
        try:
            param_service_config = context.global_app_config.get('llm', {}).get('parameters', {})
            
            if not isinstance(param_service_config, dict):
                logger.warning(f"Invalid ParameterService config type: {type(param_service_config)}")
                param_service_config = {}

            # Fix: Use the actual registered component IDs
            PARAMETER_SERVICE_METADATA.dependencies = {
                "ComponentRegistry": "*"  # Changed from "component_registry" to match actual ID
            }

            instance = ParameterService(
                config=param_service_config, 
                metadata_definition=PARAMETER_SERVICE_METADATA
            )

            # Register with enhanced metadata
            context.registry.register(instance, PARAMETER_SERVICE_METADATA)
            context.registry.register_service_instance(ParameterService, instance)
            
            result = create_success_result(
                PARAMETER_SERVICE_METADATA.id,
                f"{PARAMETER_SERVICE_METADATA.name} created and registered",
                output_data=instance
            )
            result.set_correlation_id(correlation_id)
            
            logger.info(f"✓ {PARAMETER_SERVICE_METADATA.name} created and registered")
            return result

        except Exception as e:
            error_result = create_error_result(PARAMETER_SERVICE_METADATA.id, e)
            error_result.set_correlation_id(correlation_id)
            return error_result

    async def _setup_frame_factory_service(self, context, correlation_id: str) -> ProcessResult:
        """Setup FrameFactoryService with enhanced tracking."""
        try:
            frame_factory_config = context.global_app_config.get('frame_factory_config', {})
            
            # Fix: Use the actual registered component IDs
            FRAME_FACTORY_SERVICE_METADATA.dependencies = {
                "ComponentRegistry": "*",  # Changed from "component_registry"
                "EventBusPort": "*"        # Changed from "event_bus"
            }
            
            frame_factory = FrameFactoryService(
                config=frame_factory_config,
                metadata_definition=FRAME_FACTORY_SERVICE_METADATA
            )
            
            # Register with enhanced metadata
            context.registry.register(frame_factory, FRAME_FACTORY_SERVICE_METADATA)
            context.registry.register_service_instance(FrameFactoryService, frame_factory)
            
            result = create_success_result(
                FRAME_FACTORY_SERVICE_METADATA.id,
                f"{FRAME_FACTORY_SERVICE_METADATA.name} created and registered",
                output_data=frame_factory
            )
            result.set_correlation_id(correlation_id)
            
            logger.info(f"✓ {FRAME_FACTORY_SERVICE_METADATA.name} created and registered")
            return result

        except Exception as e:
            error_result = create_error_result(FRAME_FACTORY_SERVICE_METADATA.id, e)
            error_result.set_correlation_id(correlation_id)
            return error_result

    # ***** NEW METHOD: _setup_budget_manager *****
    async def _setup_budget_manager(self, context, correlation_id: str) -> ProcessResult:
        BUDGET_MANAGER_METADATA = ComponentMetadata(
            id='budget_manager_inmemory_instance',  # Or use manifest ID if defined there
            name='InMemoryBudgetManager Instance',
            version='1.0.0',
            category='service_core',
            description='In-memory budget manager for resource tracking.',
            requires_initialize=False  # Typically simple managers might not need async init
        )
        try:
            # Check if already registered by manifest (good practice)
            try:
                existing_bm = context.registry.get_service_instance(BudgetManagerPort)
                logger.info(f"BudgetManagerPort already registered (likely by manifest): {type(existing_bm).__name__}")
                return create_success_result(BUDGET_MANAGER_METADATA.id, 'BudgetManagerPort already registered.', output_data=existing_bm)
            except ComponentRegistryMissingError:
                logger.info("BudgetManagerPort not found in registry. Creating default InMemoryBudgetManager.")

            budget_manager_config = context.global_app_config.get('budget_manager_config', {})
            budget_manager = InMemoryBudgetManager(config=budget_manager_config)

            # Register the instance with the port type and a specific metadata
            context.registry.register(budget_manager, BUDGET_MANAGER_METADATA)
            context.registry.register_service_instance(BudgetManagerPort, budget_manager)

            result = create_success_result(BUDGET_MANAGER_METADATA.id, f'{BUDGET_MANAGER_METADATA.name} created and registered', output_data=budget_manager)
            result.set_correlation_id(correlation_id)
            logger.info(f'✓ {BUDGET_MANAGER_METADATA.name} created and registered by FactorySetupPhase')
            return result
        except Exception as e:
            error_result = create_error_result(BUDGET_MANAGER_METADATA.id, e)
            error_result.set_correlation_id(correlation_id)
            return error_result

    # Modify _setup_mechanism_gateway to fetch BudgetManagerPort from registry
    async def _setup_mechanism_gateway(self, context: NireonExecutionContext, correlation_id: str) -> ProcessResult:
        """Setup MechanismGateway with dependency validation."""
        try:
            deps = {}
            # Required dependencies that MechanismGateway constructor expects by name
            # The constructor will look these up in the registry if not passed directly.
            # We will rely on the constructor's internal lookup via context.component_registry
            # for most of these, but specifically pass BudgetManagerPort for clarity here.

            llm_router_instance = context.registry.get_service_instance(LLMPort)
            param_service_instance = context.registry.get_service_instance(ParameterService)
            frame_factory_instance = context.registry.get_service_instance(FrameFactoryService)
            budget_manager_instance = context.registry.get_service_instance(BudgetManagerPort)  # Key change

            event_bus_instance = None
            try:
                event_bus_instance = context.registry.get_service_instance(EventBusPort)
            except Exception:
                logger.warning('EventBus not available for MechanismGateway during FactorySetupPhase instantiation')

            gateway_config = context.global_app_config.get('mechanism_gateway_config', {})

            # Ensure metadata definition is correct and has the new dependency
            gateway_metadata_def = MECHANISM_GATEWAY_METADATA
            gateway_metadata_def.dependencies['BudgetManagerPort'] = '*'  # Add if not already there

            gateway = MechanismGateway(
                llm_router=llm_router_instance,
                parameter_service=param_service_instance,
                frame_factory=frame_factory_instance,
                budget_manager=budget_manager_instance,  # Explicitly pass it
                event_bus=event_bus_instance,
                config=gateway_config,
                metadata_definition=gateway_metadata_def
            )

            # Register the gateway instance
            context.registry.register(gateway, gateway.metadata)  # Use metadata from the instance
            context.registry.register_service_instance(MechanismGatewayPort, gateway)

            result = create_success_result(gateway.component_id, f'{gateway.metadata.name} created and registered', output_data=gateway)
            result.set_correlation_id(correlation_id)
            result.add_metric('dependencies_resolved', 4 + (1 if event_bus_instance else 0))
            logger.info(f'✓ {gateway.metadata.name} created and registered by FactorySetupPhase')
            return result
        except Exception as e:
            error_result = create_error_result(MECHANISM_GATEWAY_METADATA.id, e)  # Use the canonical ID for error
            error_result.set_correlation_id(correlation_id)
            return error_result

    async def _validate_factory_setup(self, context, correlation_id: str) -> ProcessResult:
        """Enhanced validation with dependency checking."""
        issues = []
        
        try:
            # Check context attributes
            required_attrs = [
                ('common_mechanism_deps', 'CommonMechanismDependencies'),
                ('mechanism_factory', 'SimpleMechanismFactory'),
                ('interface_validator', 'InterfaceValidator')
            ]
            
            for attr_name, component_name in required_attrs:
                if not hasattr(context, attr_name) or getattr(context, attr_name) is None:
                    issue = f"Missing required attribute: {attr_name} ({component_name})"
                    if context.strict_mode:
                        issues.append(issue)
                        logger.error(f"Validation issue: {issue}")
                    else:
                        logger.warning(f"Warning: {issue}")

            # Check registry services
            registry_checks = [
                (ParameterService, "ParameterService"),
                (FrameFactoryService, "FrameFactoryService"),
                (MechanismGatewayPort, "MechanismGateway"),
                (BudgetManagerPort, "BudgetManager")  # Added budget manager check
            ]

            for service_type, service_name in registry_checks:
                try:
                    context.registry.get_service_instance(service_type)
                    logger.debug(f"✓ {service_name} found in registry")
                except Exception as e:
                    issue = f"{service_name} not found in registry: {str(e)}"
                    issues.append(issue)
                    logger.error(f"Validation issue: {issue}")

            # Check dependency conflicts
            if hasattr(context.registry, 'check_dependency_conflicts'):
                conflicts = context.registry.check_dependency_conflicts()
                if conflicts:
                    for conflict in conflicts:
                        issues.append(f"Dependency conflict: {conflict}")
                        logger.error(f"Validation issue: Dependency conflict: {conflict}")
            else:
                logger.debug("Registry does not support dependency conflict checking")

            # Log summary
            if issues:
                logger.error(f"Factory validation found {len(issues)} issues:")
                for i, issue in enumerate(issues, 1):
                    logger.error(f"  {i}. {issue}")
                
                result = ProcessResult(
                    success=False,
                    component_id="factory_validation",
                    message=f"Validation found {len(issues)} issues",
                    error_code="VALIDATION_FAILED",
                    metadata={"issues": issues}
                )
            else:
                logger.info("✓ Factory setup validation passed - all checks successful")
                result = create_success_result(
                    "factory_validation",
                    "Factory setup validation passed"
                )
                
            result.set_correlation_id(correlation_id)
            return result

        except Exception as e:
            logger.error(f"Error during validation: {e}", exc_info=True)
            error_result = create_error_result("factory_validation", e)
            error_result.set_correlation_id(correlation_id)
            return error_result

    def _create_phase_result(self, setup_components: Optional[List[str]] = None) -> PhaseResult:
        """Create phase result from collected results."""
        failures = self.result_collector.get_failures()
        critical_signals = self.result_collector.get_critical_signals()
        
        success = len(failures) == 0 and len(critical_signals) == 0
        
        # Calculate phase duration
        phase_duration = (datetime.now(timezone.utc) - self.phase_start_time).total_seconds()
        
        # Build message
        if success:
            message = f"Factory setup completed successfully - {len(setup_components or [])} components"
        else:
            message = f"Factory setup failed - {len(failures)} failures, {len(critical_signals)} critical issues"
        
        # Extract errors and warnings
        errors = [f"{r.component_id}: {r.message}" for r in failures]
        warnings = []
        
        # Add unacknowledged signals as warnings
        for signal in self.result_collector.get_by_type(SystemSignal):
            if signal.needs_acknowledgment:
                warnings.append(f"{signal.component_id}: {signal.message}")
        
        return PhaseResult(
            success=success,
            message=message,
            errors=errors,
            warnings=warnings,
            metadata={
                'factories_setup': setup_components or [],
                'factory_system_ready': success,
                'phase_duration_seconds': phase_duration,
                'total_results_collected': len(self.result_collector.results),
                'dependency_conflicts': len(self.result_collector.get_by_component("factory_validation"))
            }
        )

    def should_skip_phase(self, context) -> tuple[bool, str]:
        """Check if this phase should be skipped."""
        skip_factories = context.global_app_config.get('skip_factory_setup', False)
        if skip_factories:
            return (True, 'Factory setup phase explicitly disabled in configuration.')
        return (False, '')