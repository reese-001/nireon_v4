# nireon_v4\bootstrap\phases\factory_setup_phase.py
from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone
from core.lifecycle import ComponentMetadata
from core.results import ProcessResult, SystemSignal, SignalType, ResultCollector, create_success_result, create_error_result
from domain.ports.llm_port import LLMPort
from domain.ports.embedding_port import EmbeddingPort
from domain.ports.event_bus_port import EventBusPort
from domain.ports.idea_service_port import IdeaServicePort
from domain.ports.mechanism_gateway_port import MechanismGatewayPort
from domain.context import NireonExecutionContext
from application.services.idea_service import IdeaService
from application.services.frame_factory_service import FrameFactoryService, FRAME_FACTORY_SERVICE_METADATA
from application.services.budget_manager import BudgetManagerPort, InMemoryBudgetManager
# --- THIS IS THE CRITICAL SECTION TO VERIFY ---
from infrastructure.gateway.mechanism_gateway import MechanismGateway
from infrastructure.gateway.mechanism_gateway_metadata import MECHANISM_GATEWAY_METADATA
# ----------------------------------------------
from infrastructure.llm.parameter_service import ParameterService
from infrastructure.llm.router import LLMRouter
from .base_phase import BootstrapPhase, PhaseResult
from factories.mechanism_factory import SimpleMechanismFactory
from factories.dependencies import CommonMechanismDependencies
from bootstrap.validators.interface_validator import InterfaceValidator
from bootstrap.bootstrap_helper.context_helper import build_execution_context, create_context_builder, SimpleConfigProvider
from core.lifecycle import ComponentRegistryMissingError
from bootstrap.processors.service_resolver import _safe_register_service_instance
from application.services.budget_manager import BUDGET_MANAGER_METADATA as BUDGET_MANAGER_CLASS_METADATA
logger = logging.getLogger(__name__)
PARAMETER_SERVICE_METADATA = ComponentMetadata(id='parameter_service_global', name='Global LLM Parameter Service', version='1.0.0', category='service_core', description='Centralized service for LLM parameter resolution.', requires_initialize=True, capabilities={'parameter_resolution', 'config_management', 'stage_routing'}, produces=['parameter_set', 'routing_config'])
class FactorySetupPhase(BootstrapPhase):
    def __init__(self):
        super().__init__()
        self.result_collector = ResultCollector()
        self.phase_start_time = None
        self.config_provider: Optional[SimpleConfigProvider] = None
        self.phase_contexts: Dict[str, Any] = {}
    async def execute(self, context) -> PhaseResult:
        logger.info('Executing Enhanced V4 Factory Setup Phase with V2 Context Integration')
        self.phase_start_time = datetime.now(timezone.utc)
        self._initialize_enhanced_config(context)
        phase_correlation_id = f'factory_setup_{self.phase_start_time.isoformat()}'
        setup_components = []
        try:
            deps_result = await self._setup_common_dependencies(context, phase_correlation_id)
            self.result_collector.add(deps_result)
            if deps_result.success:
                context.common_mechanism_deps = deps_result.output_data
                setup_components.append('CommonMechanismDependencies')
            else:
                context.common_mechanism_deps = None
                if context.strict_mode:
                    return self._create_phase_result()
            factory_result = await self._setup_mechanism_factory(context, context.common_mechanism_deps, phase_correlation_id)
            self.result_collector.add(factory_result)
            if factory_result.success:
                context.mechanism_factory = factory_result.output_data
                setup_components.append('SimpleMechanismFactory')
            else:
                context.mechanism_factory = None
                if context.strict_mode and context.common_mechanism_deps:
                    return self._create_phase_result()
            validator_result = await self._setup_interface_validator(context, phase_correlation_id)
            self.result_collector.add(validator_result)
            if validator_result.success:
                context.interface_validator = validator_result.output_data
                setup_components.append('InterfaceValidator')
            else:
                context.interface_validator = None
                if context.strict_mode:
                    return self._create_phase_result()
            param_result = await self._setup_parameter_service(context, phase_correlation_id)
            self.result_collector.add(param_result)
            if not param_result.success and context.strict_mode:
                return self._create_phase_result()
            elif param_result.success:
                setup_components.append(PARAMETER_SERVICE_METADATA.id)
            frame_result = await self._setup_frame_factory_service(context, phase_correlation_id)
            self.result_collector.add(frame_result)
            if not frame_result.success and context.strict_mode:
                return self._create_phase_result()
            elif frame_result.success:
                setup_components.append(FRAME_FACTORY_SERVICE_METADATA.id)
            budget_manager_result = await self._setup_budget_manager(context, phase_correlation_id)
            self.result_collector.add(budget_manager_result)
            if not budget_manager_result.success and context.strict_mode:
                return self._create_phase_result()
            elif budget_manager_result.success:
                setup_components.append('BudgetManagerPort_Instance')
            if frame_result.success and budget_manager_result.success:
                gateway_result = await self._setup_mechanism_gateway(context, phase_correlation_id)
                self.result_collector.add(gateway_result)
                if gateway_result.success:
                    setup_components.append(MECHANISM_GATEWAY_METADATA.id)
            else:
                logger.warning('MechanismGateway setup skipped due to FrameFactoryService or BudgetManager failure')
            validation_result = await self._validate_factory_setup(context, phase_correlation_id)
            self.result_collector.add(validation_result)
            return self._create_phase_result(setup_components)
        except Exception as e:
            error_result = create_error_result('factory_setup_phase', e)
            error_result.set_correlation_id(phase_correlation_id)
            self.result_collector.add(error_result)
            critical_signal = SystemSignal(signal_type=SignalType.CRITICAL, component_id='factory_setup_phase', message=f'Critical failure in factory setup: {e}', payload={'exception_type': type(e).__name__}, requires_acknowledgment=True)
            if hasattr(context, 'event_bus') and context.event_bus:
                await context.event_bus.publish(critical_signal.to_event()['event_type'], critical_signal.to_event())
            return self._create_phase_result()
    def _initialize_enhanced_config(self, context) -> None:
        try:
            factory_config = context.global_app_config.get('factory_setup', {})
            llm_config = context.global_app_config.get('llm', {})
            frame_factory_config = context.global_app_config.get('frame_factory_config', {})
            mechanism_gateway_config = context.global_app_config.get('mechanism_gateway_config', {})
            enhanced_config = {**{f'factory.{k}': v for k, v in factory_config.items()}, **{f'llm.{k}': v for k, v in llm_config.items()}, **{f'frame_factory.{k}': v for k, v in frame_factory_config.items()}, **{f'mechanism_gateway.{k}': v for k, v in mechanism_gateway_config.items()}}
            self.config_provider = SimpleConfigProvider(enhanced_config)
            logger.debug(f'Factory setup enhanced config provider initialized with {len(enhanced_config)} configuration entries')
        except Exception as e:
            logger.warning(f'Failed to initialize factory setup enhanced config provider: {e}')
            self.config_provider = None
    async def _setup_common_dependencies(self, context, correlation_id: str) -> ProcessResult:
        try:
            deps_context = self._create_component_context(context, 'common_dependencies', {'setup_step': 'dependency_resolution'})
            self.phase_contexts['common_dependencies'] = deps_context
            services = {}
            service_checks = [(LLMPort, 'LLMPort', True), (EmbeddingPort, 'EmbeddingPort', True), (EventBusPort, 'EventBusPort', False), (IdeaServicePort, 'IdeaServicePort', True)]
            for service_type, name, is_critical in service_checks:
                try:
                    instance = context.registry.get_service_instance(service_type)
                    services[name] = instance
                    logger.info(f'✓ Resolved {name}')
                except Exception as e:
                    if is_critical:
                        error_result = create_error_result('common_dependencies', RuntimeError(f'Failed to resolve critical service {name}: {e}'))
                        error_result.set_correlation_id(correlation_id)
                        return error_result
                    else:
                        services[name] = None
                        logger.warning(f'Optional service {name} not available: {e}')
            import random
            common_deps = CommonMechanismDependencies(llm_port=services['LLMPort'], llm_router=services['LLMPort'], embedding_port=services['EmbeddingPort'], event_bus=services.get('EventBusPort'), idea_service=services['IdeaServicePort'], component_registry=context.registry, rng=random.Random())
            result = create_success_result('common_dependencies', 'CommonMechanismDependencies created successfully with V2 context', output_data=common_deps)
            result.set_correlation_id(correlation_id)
            result.add_metric('services_resolved', len([s for s in services.values() if s]))
            result.add_metric('v2_context_enabled', deps_context is not None)
            logger.info('✓ CommonMechanismDependencies created with core services and V2 context')
            return result
        except Exception as e:
            error_result = create_error_result('common_dependencies', e)
            error_result.set_correlation_id(correlation_id)
            return error_result
    async def _setup_mechanism_factory(self, context, common_deps: Optional[CommonMechanismDependencies], correlation_id: str) -> ProcessResult:
        if common_deps is None:
            return create_error_result('mechanism_factory', RuntimeError('Cannot create SimpleMechanismFactory without CommonMechanismDependencies'))
        try:
            factory_context = self._create_component_context(context, 'mechanism_factory', {'setup_step': 'factory_creation', 'dependencies_available': True})
            self.phase_contexts['mechanism_factory'] = factory_context
            mechanism_factory = SimpleMechanismFactory(common_deps)
            if self.config_provider:
                mechanism_mappings = self.config_provider.get_config('factory', 'mechanism_mappings', {})
            else:
                mechanism_mappings = context.global_app_config.get('mechanism_mappings', {})
            for factory_key, class_path in mechanism_mappings.items():
                mechanism_factory.register_mechanism_type(factory_key, class_path)
                logger.info(f'Registered mechanism type: {factory_key} -> {class_path}')
            if hasattr(context, 'registry_manager'):
                context.registry_manager.register_service_with_certification(service_type=SimpleMechanismFactory, instance=mechanism_factory, service_id='SimpleMechanismFactory', category='factory_service', description='Factory for creating mechanism components (V2 enhanced)', requires_initialize=False)
            else:
                context.registry.register_service_instance(SimpleMechanismFactory, mechanism_factory)
            result = create_success_result('mechanism_factory', 'SimpleMechanismFactory initialized successfully with V2 context', output_data=mechanism_factory)
            result.set_correlation_id(correlation_id)
            result.add_metric('mechanism_types_registered', len(mechanism_mappings))
            result.add_metric('v2_context_enabled', factory_context is not None)
            logger.info('✓ SimpleMechanismFactory initialized with V2 context integration')
            return result
        except Exception as e:
            error_result = create_error_result('mechanism_factory', e)
            error_result.set_correlation_id(correlation_id)
            return error_result
    async def _setup_interface_validator(self, context, correlation_id: str) -> ProcessResult:
        try:
            event_bus = None
            try:
                event_bus = context.registry.get_service_instance(EventBusPort)
            except Exception:
                logger.warning('EventBus not available for InterfaceValidator')
            if not hasattr(context, 'run_id'):
                context.run_id = 'bootstrap_default'
            validator_exec_context = build_execution_context(component_id='bootstrap_validator', run_id=context.run_id, registry=context.registry, event_bus=event_bus, feature_flags=getattr(context, 'feature_flags', {}), metadata={'validator_setup': True, 'v2_context': True})
            interface_validator = InterfaceValidator(validator_exec_context)
            validator_context = self._create_component_context(context, 'interface_validator', {'setup_step': 'validator_creation', 'execution_context_created': True})
            self.phase_contexts['interface_validator'] = validator_context
            if hasattr(context, 'registry_manager') and context.registry_manager:
                context.registry_manager.register_service_with_certification(service_type=InterfaceValidator, instance=interface_validator, service_id='InterfaceValidator', category='validation_service', description='Validator for component interfaces (V2 enhanced)', requires_initialize=False)
            else:
                context.registry.register_service_instance(InterfaceValidator, interface_validator)
            result = create_success_result('interface_validator', 'InterfaceValidator initialized successfully with V2 context', output_data=interface_validator)
            result.set_correlation_id(correlation_id)
            result.add_metric('v2_context_enabled', validator_context is not None)
            result.add_metric('execution_context_created', True)
            logger.info('✓ InterfaceValidator initialized with V2 context integration')
            return result
        except Exception as e:
            error_result = create_error_result('interface_validator', e)
            error_result.set_correlation_id(correlation_id)
            return error_result
    async def _setup_parameter_service(self, context, correlation_id: str) -> ProcessResult:
        try:
            if self.config_provider:
                param_service_config = self.config_provider.get_config('llm', 'parameters', {})
            else:
                param_service_config = context.global_app_config.get('llm', {}).get('parameters', {})
            if not isinstance(param_service_config, dict):
                logger.warning(f'Invalid ParameterService config type: {type(param_service_config)}')
                param_service_config = {}
            PARAMETER_SERVICE_METADATA.dependencies = {'ComponentRegistry': '*'}
            instance = ParameterService(config=param_service_config, metadata_definition=PARAMETER_SERVICE_METADATA)
            param_context = self._create_component_context(context, 'parameter_service', {'setup_step': 'service_creation', 'config_provider_used': self.config_provider is not None})
            self.phase_contexts['parameter_service'] = param_context
            _safe_register_service_instance(registry=context.registry, service_protocol_type=ParameterService, instance=instance, service_id_for_meta=PARAMETER_SERVICE_METADATA.id, category_for_meta=PARAMETER_SERVICE_METADATA.category, description_for_meta=PARAMETER_SERVICE_METADATA.description, requires_initialize_override=PARAMETER_SERVICE_METADATA.requires_initialize)
            if hasattr(context, 'validation_data_store') and context.validation_data_store:
                context.validation_data_store.store_component_data(component_id=PARAMETER_SERVICE_METADATA.id, original_metadata=PARAMETER_SERVICE_METADATA, resolved_config=param_service_config, manifest_spec={'class': f'{instance.__class__.__module__}:{instance.__class__.__name__}', 'config': param_service_config})
            result = create_success_result(PARAMETER_SERVICE_METADATA.id, f'{PARAMETER_SERVICE_METADATA.name} created and registered with V2 context', output_data=instance)
            result.set_correlation_id(correlation_id)
            result.add_metric('v2_context_enabled', param_context is not None)
            logger.info(f'✓ {PARAMETER_SERVICE_METADATA.name} created and registered with V2 integration')
            return result
        except Exception as e:
            error_result = create_error_result(PARAMETER_SERVICE_METADATA.id, e)
            error_result.set_correlation_id(correlation_id)
            return error_result
    async def _setup_frame_factory_service(self, context, correlation_id: str) -> ProcessResult:
        try:
            if self.config_provider:
                frame_factory_config = self.config_provider.get_config('frame_factory', 'config', {})
            else:
                frame_factory_config = context.global_app_config.get('frame_factory_config', {})
            FRAME_FACTORY_SERVICE_METADATA.dependencies = {'ComponentRegistry': '*', 'EventBusPort': '*'}
            frame_factory = FrameFactoryService(config=frame_factory_config, metadata_definition=FRAME_FACTORY_SERVICE_METADATA)
            frame_context = self._create_component_context(context, 'frame_factory_service', {'setup_step': 'service_creation', 'config_provider_used': self.config_provider is not None})
            self.phase_contexts['frame_factory_service'] = frame_context
            _safe_register_service_instance(registry=context.registry, service_protocol_type=FrameFactoryService, instance=frame_factory, service_id_for_meta=FRAME_FACTORY_SERVICE_METADATA.id, category_for_meta=FRAME_FACTORY_SERVICE_METADATA.category, description_for_meta=FRAME_FACTORY_SERVICE_METADATA.description, requires_initialize_override=FRAME_FACTORY_SERVICE_METADATA.requires_initialize)
            if hasattr(context, 'validation_data_store') and context.validation_data_store:
                context.validation_data_store.store_component_data(component_id=FRAME_FACTORY_SERVICE_METADATA.id, original_metadata=FRAME_FACTORY_SERVICE_METADATA, resolved_config=frame_factory_config, manifest_spec={'class': f'{frame_factory.__class__.__module__}:{frame_factory.__class__.__name__}', 'config': frame_factory_config})
            result = create_success_result(FRAME_FACTORY_SERVICE_METADATA.id, f'{FRAME_FACTORY_SERVICE_METADATA.name} created and registered with V2 context', output_data=frame_factory)
            result.set_correlation_id(correlation_id)
            result.add_metric('v2_context_enabled', frame_context is not None)
            logger.info(f'✓ {FRAME_FACTORY_SERVICE_METADATA.name} created and registered with V2 integration')
            return result
        except Exception as e:
            error_result = create_error_result(FRAME_FACTORY_SERVICE_METADATA.id, e)
            error_result.set_correlation_id(correlation_id)
            return error_result
    async def _setup_budget_manager(self, context, correlation_id: str) -> ProcessResult:
        try:
            try:
                existing_bm = context.registry.get_service_instance(BudgetManagerPort)
                logger.info(f'BudgetManagerPort already registered (likely by manifest): {type(existing_bm).__name__}')
                return create_success_result(BUDGET_MANAGER_CLASS_METADATA.id, 'BudgetManagerPort already registered.', output_data=existing_bm)
            except ComponentRegistryMissingError:
                logger.info('BudgetManagerPort not found in registry. Creating default InMemoryBudgetManager.')
            if self.config_provider:
                budget_manager_config = self.config_provider.get_config('budget_manager', 'config', {})
            else:
                budget_manager_config = context.global_app_config.get('budget_manager_config', {})
            budget_manager = InMemoryBudgetManager(config=budget_manager_config, metadata_definition=BUDGET_MANAGER_CLASS_METADATA)
            budget_context = self._create_component_context(context, 'budget_manager', {'setup_step': 'manager_creation', 'config_provider_used': self.config_provider is not None})
            self.phase_contexts['budget_manager'] = budget_context
            _safe_register_service_instance(registry=context.registry, service_protocol_type=BudgetManagerPort, instance=budget_manager, service_id_for_meta=BUDGET_MANAGER_CLASS_METADATA.id, category_for_meta=BUDGET_MANAGER_CLASS_METADATA.category, description_for_meta=BUDGET_MANAGER_CLASS_METADATA.description, requires_initialize_override=BUDGET_MANAGER_CLASS_METADATA.requires_initialize)
            logger.info(f'✓ {BUDGET_MANAGER_CLASS_METADATA.name} created and registered with V2 integration')
            if hasattr(context, 'validation_data_store') and context.validation_data_store:
                context.validation_data_store.store_component_data(component_id=BUDGET_MANAGER_CLASS_METADATA.id, original_metadata=BUDGET_MANAGER_CLASS_METADATA, resolved_config=budget_manager_config, manifest_spec={'class': f'{budget_manager.__class__.__module__}:{budget_manager.__class__.__name__}', 'config': budget_manager_config})
            result = create_success_result(BUDGET_MANAGER_CLASS_METADATA.id, f'{BUDGET_MANAGER_CLASS_METADATA.name} created and registered with V2 context', output_data=budget_manager)
            result.set_correlation_id(correlation_id)
            result.add_metric('v2_context_enabled', budget_context is not None)
            return result
        except Exception as e:
            error_result = create_error_result(BUDGET_MANAGER_CLASS_METADATA.id, e)
            error_result.set_correlation_id(correlation_id)
            return error_result
    async def _setup_mechanism_gateway(self, context: NireonExecutionContext, correlation_id: str) -> ProcessResult:
        try:
            try:
                _ = context.registry.get('mechanism_gateway_main')
                logger.info(f"MechanismGateway 'mechanism_gateway_main' found in registry (likely from manifest). Skipping factory phase creation of 'mechanism_gateway'.")
                return create_success_result(MECHANISM_GATEWAY_METADATA.id, "Manifest-defined MechanismGateway 'mechanism_gateway_main' takes precedence. Factory creation skipped.", output_data=None)
            except ComponentRegistryMissingError:
                logger.info(f"Manifest-defined 'mechanism_gateway_main' not found. Proceeding with factory phase creation of 'mechanism_gateway'.")
            except Exception as e_check:
                logger.warning(f"Error checking for existing 'mechanism_gateway_main': {e_check}. Proceeding with factory creation.")
        except Exception as e:
            pass
        try:
            llm_router_instance = context.registry.get_service_instance(LLMPort)
            param_service_instance = context.registry.get_service_instance(ParameterService)
            frame_factory_instance = context.registry.get_service_instance(FrameFactoryService)
            budget_manager_instance = context.registry.get_service_instance(BudgetManagerPort)
            event_bus_instance = None
            try:
                event_bus_instance = context.registry.get_service_instance(EventBusPort)
            except Exception:
                logger.warning('EventBus not available for MechanismGateway during FactorySetupPhase instantiation')
            if self.config_provider:
                gateway_config = self.config_provider.get_config('mechanism_gateway', 'config', {})
            else:
                gateway_config = context.global_app_config.get('mechanism_gateway_config', {})
            gateway_context = self._create_component_context(context, 'mechanism_gateway', {'setup_step': 'gateway_creation', 'dependencies_resolved': 4 + (1 if event_bus_instance else 0), 'config_provider_used': self.config_provider is not None})
            self.phase_contexts['mechanism_gateway'] = gateway_context
            gateway_metadata_def = MECHANISM_GATEWAY_METADATA
            gateway_metadata_def.dependencies['BudgetManagerPort'] = '*'
            gateway = MechanismGateway(llm_router=llm_router_instance, parameter_service=param_service_instance, frame_factory=frame_factory_instance, budget_manager=budget_manager_instance, event_bus=event_bus_instance, config=gateway_config, metadata_definition=gateway_metadata_def)
            _safe_register_service_instance(registry=context.registry, service_protocol_type=MechanismGatewayPort, instance=gateway, service_id_for_meta=gateway.component_id, category_for_meta=gateway.metadata.category, description_for_meta=gateway.metadata.description, requires_initialize_override=gateway.metadata.requires_initialize)
            if hasattr(context, 'validation_data_store') and context.validation_data_store:
                context.validation_data_store.store_component_data(component_id=gateway.component_id, original_metadata=gateway.metadata, resolved_config=gateway_config, manifest_spec={'class': f'{gateway.__class__.__module__}:{gateway.__class__.__name__}', 'config': gateway_config})
            result = create_success_result(gateway.component_id, f'{gateway.metadata.name} created and registered with V2 context', output_data=gateway)
            result.set_correlation_id(correlation_id)
            result.add_metric('dependencies_resolved', 4 + (1 if event_bus_instance else 0))
            result.add_metric('v2_context_enabled', gateway_context is not None)
            logger.info(f'✓ {gateway.metadata.name} created and registered with V2 integration')
            return result
        except Exception as e:
            error_result = create_error_result(MECHANISM_GATEWAY_METADATA.id, e)
            error_result.set_correlation_id(correlation_id)
            return error_result
    async def _validate_factory_setup(self, context, correlation_id: str) -> ProcessResult:
        issues = []
        try:
            validation_context = self._create_component_context(context, 'factory_validation', {'setup_step': 'validation', 'phase_contexts_count': len(self.phase_contexts)})
            required_attrs = [('common_mechanism_deps', 'CommonMechanismDependencies'), ('mechanism_factory', 'SimpleMechanismFactory'), ('interface_validator', 'InterfaceValidator')]
            for attr_name, component_name in required_attrs:
                if not hasattr(context, attr_name) or getattr(context, attr_name) is None:
                    issue = f'Missing required attribute: {attr_name} ({component_name})'
                    if context.strict_mode:
                        issues.append(issue)
                        logger.error(f'Validation issue: {issue}')
                    else:
                        logger.warning(f'Warning: {issue}')
            registry_checks = [(ParameterService, 'ParameterService'), (FrameFactoryService, 'FrameFactoryService'), (MechanismGatewayPort, 'MechanismGateway'), (BudgetManagerPort, 'BudgetManager')]
            for service_type, service_name in registry_checks:
                try:
                    context.registry.get_service_instance(service_type)
                    logger.debug(f'✓ {service_name} found in registry')
                except Exception as e:
                    issue = f'{service_name} not found in registry: {str(e)}'
                    issues.append(issue)
                    logger.error(f'Validation issue: {issue}')
            v2_validation_issues = self._validate_v2_integration()
            if v2_validation_issues:
                issues.extend(v2_validation_issues)
            if hasattr(context.registry, 'check_dependency_conflicts'):
                conflicts = context.registry.check_dependency_conflicts()
                if conflicts:
                    for conflict in conflicts:
                        issues.append(f'Dependency conflict: {conflict}')
                        logger.error(f'Validation issue: Dependency conflict: {conflict}')
            else:
                logger.debug('Registry does not support dependency conflict checking')
            if issues:
                logger.error(f'Factory validation found {len(issues)} issues:')
                for i, issue in enumerate(issues, 1):
                    logger.error(f'  {i}. {issue}')
                result = ProcessResult(success=False, component_id='factory_validation', message=f'Validation found {len(issues)} issues', error_code='VALIDATION_FAILED', metadata={'issues': issues, 'v2_context_enabled': validation_context is not None, 'phase_contexts_created': len(self.phase_contexts)})
            else:
                logger.info('✓ Factory setup validation passed - all checks successful with V2 integration')
                result = create_success_result('factory_validation', 'Factory setup validation passed with V2 context integration')
                result.add_metric('v2_context_enabled', validation_context is not None)
                result.add_metric('phase_contexts_created', len(self.phase_contexts))
            result.set_correlation_id(correlation_id)
            return result
        except Exception as e:
            logger.error(f'Error during validation: {e}', exc_info=True)
            error_result = create_error_result('factory_validation', e)
            error_result.set_correlation_id(correlation_id)
            return error_result
    def _create_component_context(self, base_context, component_id: str, metadata: Dict[str, Any]):
        try:
            context_builder = create_context_builder(component_id=f'factory_{component_id}', run_id=f'{base_context.run_id}_factory')
            if hasattr(base_context, 'registry') and base_context.registry is not None:
                context_builder.with_registry(base_context.registry)
            else:
                logger.warning(f'No registry available for V2 context creation for {component_id}')
                return None
            if hasattr(base_context, 'event_bus') and base_context.event_bus is not None:
                context_builder.with_event_bus(base_context.event_bus)
            else:
                from bootstrap.bootstrap_helper.placeholders import PlaceholderEventBusImpl
                placeholder_bus = PlaceholderEventBusImpl()
                context_builder.with_event_bus(placeholder_bus)
            enhanced_metadata = {**metadata, 'factory_phase': True, 'v2_context': True, 'config_provider_available': self.config_provider is not None}
            context_builder.with_metadata(**enhanced_metadata)
            if hasattr(base_context, 'feature_flags') and base_context.feature_flags:
                context_builder.with_feature_flags(base_context.feature_flags)
            return context_builder.build()
        except ImportError as e:
            logger.debug(f'V2 context helper not available for {component_id}: {e}')
            return None
        except AttributeError as e:
            logger.warning(f'V2 context builder missing attribute for {component_id}: {e}')
            return None
        except Exception as e:
            logger.warning(f'Failed to create V2 context for {component_id}: {e}')
            return None
    def _validate_v2_integration(self) -> List[str]:
        issues = []
        try:
            if self.config_provider is None:
                issues.append('V2 config provider not initialized (non-critical)')
            expected_contexts = ['common_dependencies', 'mechanism_factory', 'interface_validator']
            missing_contexts = []
            for expected in expected_contexts:
                if expected not in self.phase_contexts:
                    missing_contexts.append(expected)
                elif self.phase_contexts[expected] is None:
                    missing_contexts.append(f'{expected} (failed to create)')
            if missing_contexts:
                logger.warning(f'V2 integration issues (non-critical): Missing contexts for {missing_contexts}')
            working_contexts = 0
            for context_name, phase_context in self.phase_contexts.items():
                if phase_context is not None and hasattr(phase_context, 'metadata'):
                    working_contexts += 1
            if working_contexts > 0:
                logger.debug(f'V2 integration partially working: {working_contexts} contexts created successfully')
            else:
                logger.warning('V2 integration not working - falling back to V1 compatibility mode')
        except Exception as e:
            logger.warning(f'V2 integration validation failed (non-critical): {e}')
        return issues
    def _create_phase_result(self, setup_components: Optional[List[str]]=None) -> PhaseResult:
        failures = self.result_collector.get_failures()
        critical_signals = self.result_collector.get_critical_signals()
        success = len(failures) == 0 and len(critical_signals) == 0
        phase_duration = (datetime.now(timezone.utc) - self.phase_start_time).total_seconds()
        if success:
            message = f'Factory setup completed successfully with V2 integration - {len(setup_components or [])} components'
        else:
            message = f'Factory setup failed - {len(failures)} failures, {len(critical_signals)} critical issues'
        errors = [f'{r.component_id}: {r.message}' for r in failures]
        warnings = []
        for signal in self.result_collector.get_by_type(SystemSignal):
            if signal.needs_acknowledgment:
                warnings.append(f'{signal.component_id}: {signal.message}')
        return PhaseResult(success=success, message=message, errors=errors, warnings=warnings, metadata={'factories_setup': setup_components or [], 'factory_system_ready': success, 'phase_duration_seconds': phase_duration, 'total_results_collected': len(self.result_collector.results), 'dependency_conflicts': len(self.result_collector.get_by_component('factory_validation')), 'v2_integration': True, 'config_provider_enabled': self.config_provider is not None, 'phase_contexts_created': len(self.phase_contexts), 'v2_context_support': self.supports_v2_context()})
    def should_skip_phase(self, context) -> tuple[bool, str]:
        if self.config_provider:
            skip_factories = self.config_provider.get_config('factory', 'skip_factory_setup', False)
        else:
            skip_factories = context.global_app_config.get('skip_factory_setup', False)
        if skip_factories:
            return (True, 'Factory setup phase explicitly disabled in configuration.')
        return (False, '')