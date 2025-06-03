from __future__ import annotations
import logging
from typing import Any, Dict

from .base_phase import BootstrapPhase, PhaseResult
from infrastructure.feature_flags import FeatureFlagsManager
from bootstrap.bootstrap_helper.placeholders import (
    PlaceholderLLMPortImpl, PlaceholderEmbeddingPortImpl, 
    PlaceholderEventBusImpl, PlaceholderIdeaRepositoryImpl
)
from bootstrap.bootstrap_helper.utils import import_by_path
from application.ports.llm_port import LLMPort
from application.ports.embedding_port import EmbeddingPort
from application.ports.event_bus_port import EventBusPort
from application.ports.idea_repository_port import IdeaRepositoryPort
from application.services.idea_service import IdeaService

logger = logging.getLogger(__name__)

class AbiogenesisPhase(BootstrapPhase):
    async def execute(self, context) -> PhaseResult:
        logger.info('Executing L0 Abiogenesis - System emergence from configuration origin')
        
        errors = []
        warnings = []
        created_services = []
        
        try:
            # Emerge feature flags manager
            await self._emerge_feature_flags_manager(context, created_services, errors)
            
            # Emerge component registry (self-registration)
            await self._emerge_component_registry(context, created_services, errors)
            
            # Emerge core ports using YAML-driven approach
            await self._emerge_core_ports(context, created_services, errors, warnings)
            
            # Emerge idea service
            await self._emerge_idea_service(context, created_services, errors)
            
            # Emit abiogenesis signal
            await self._emit_abiogenesis_signal(context, created_services)
            
            success = len(errors) == 0
            message = f'L0 Abiogenesis complete - {len(created_services)} essential services emerged'
            
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
            error_msg = f'Critical failure during L0 Abiogenesis: {e}'
            logger.error(error_msg, exc_info=True)
            return PhaseResult.failure_result(
                message='L0 Abiogenesis failed',
                errors=[error_msg],
                warnings=warnings,
                metadata={'abiogenesis_failed': True}
            )

    async def _emerge_feature_flags_manager(self, context, created_services: list, errors: list) -> None:
        try:
            # Check if already exists
            try:
                existing_ff = context.registry.get_service_instance(FeatureFlagsManager)
                logger.info('FeatureFlagsManager already exists in registry')
                return
            except:
                pass
            
            # Create from configuration
            feature_flags_config = context.global_app_config.get('feature_flags', {})
            ff_manager = FeatureFlagsManager(feature_flags_config)
            
            # Register with certification
            context.registry_manager.register_service_with_certification(
                service_type=FeatureFlagsManager,
                instance=ff_manager,
                service_id='FeatureFlagsManager',
                category='core_service',
                description='System-wide feature flag management for adaptive behavior',
                requires_initialize=False
            )
            
            created_services.append('FeatureFlagsManager')
            logger.info('✓ FeatureFlagsManager emerged with system feature control')
            
        except Exception as e:
            error_msg = f'Failed to emerge FeatureFlagsManager: {e}'
            errors.append(error_msg)
            logger.error(error_msg)

    async def _emerge_component_registry(self, context, created_services: list, errors: list) -> None:
        try:
            # Self-register the registry for reflexive access
            context.registry_manager.register_service_with_certification(
                service_type=type(context.registry),
                instance=context.registry,
                service_id='component_registry',
                category='core_service',
                description='Central component registry enabling system self-awareness',
                requires_initialize=False
            )
            
            created_services.append('ComponentRegistry')
            logger.info('✓ ComponentRegistry achieved reflexive self-emergence')
            
        except Exception as e:
            error_msg = f'Failed to register ComponentRegistry reflexively: {e}'
            errors.append(error_msg)
            logger.error(error_msg)

    async def _emerge_core_ports(self, context, created_services: list, errors: list, warnings: list) -> None:
        """Emerge core ports using YAML-driven service instantiation"""
        
        # Get shared services from manifest (if loaded into global config)
        manifest_shared_services = context.global_app_config.get('shared_services', {})
        
        port_configs = [
            (LLMPort, PlaceholderLLMPortImpl, 'LLMPort', 'Language model interface for epistemic reasoning'),
            (EmbeddingPort, PlaceholderEmbeddingPortImpl, 'EmbeddingPort', 'Vector embedding interface for semantic operations'),
            (EventBusPort, PlaceholderEventBusImpl, 'EventBusPort', 'Event publication and subscription for system communication'),
            (IdeaRepositoryPort, PlaceholderIdeaRepositoryImpl, 'IdeaRepositoryPort', 'Idea storage and retrieval for epistemic persistence')
        ]
        
        for port_type, placeholder_impl, service_id, description in port_configs:
            try:
                await self._emerge_service_port(
                    context, port_type, placeholder_impl, service_id, description, 
                    created_services, manifest_shared_services
                )
            except Exception as e:
                error_msg = f'Failed to emerge {service_id}: {e}'
                if context.strict_mode:
                    errors.append(error_msg)
                else:
                    warnings.append(error_msg)
                logger.error(error_msg)

    async def _emerge_service_port(self, context, port_type, placeholder_impl, service_id: str, 
                                   description: str, created_services: list, 
                                   manifest_shared_services: dict) -> None:
        """Emerge a single service port, checking manifest first"""
        
        # Check if using provided event bus
        if service_id == 'EventBusPort' and context.config.existing_event_bus:
            instance = context.config.existing_event_bus
            logger.info(f'Using provided {service_id} instance')
        else:
            # Check if manifest defines this service
            service_spec = manifest_shared_services.get(service_id)
            
            if service_spec and service_spec.get('enabled', True):
                # Try to instantiate from manifest specification
                class_path = service_spec.get('class')
                if class_path:
                    try:
                        logger.info(f'Creating {service_id} from manifest: {class_path}')
                        service_class = import_by_path(class_path)
                        
                        # Try to instantiate (basic approach for now)
                        try:
                            instance = service_class()
                        except TypeError:
                            # Try with empty config
                            instance = service_class({})
                        
                        logger.info(f'✓ {service_id} emerged from manifest specification')
                        
                    except Exception as e:
                        logger.warning(f'Failed to instantiate {service_id} from manifest ({class_path}): {e}. Falling back to placeholder.')
                        instance = placeholder_impl()
                        
                else:
                    logger.warning(f'Manifest entry for {service_id} missing class path. Using placeholder.')
                    instance = placeholder_impl()
                    
            elif service_spec and not service_spec.get('enabled', True):
                logger.info(f'Service {service_id} disabled in manifest. Skipping.')
                return
                
            else:
                # No manifest entry, use placeholder
                instance = placeholder_impl()
                logger.info(f'Created placeholder {service_id} (no manifest entry)')

        # Register the service
        context.registry_manager.register_service_with_certification(
            service_type=port_type,
            instance=instance,
            service_id=service_id,
            category='port_service',
            description=description,
            requires_initialize=False
        )
        
        created_services.append(service_id)
        logger.debug(f'✓ {service_id} emerged as system interface')

    async def _emerge_idea_service(self, context, created_services: list, errors: list) -> None:
        try:
            # Check if already exists
            try:
                existing_idea_service = context.registry.get_service_instance(IdeaService)
                logger.info('IdeaService already exists in registry')
                return
            except:
                pass
            
            # Get dependencies
            idea_repo = context.registry.get_service_instance(IdeaRepositoryPort)
            event_bus = context.registry.get_service_instance(EventBusPort)
            
            # Create IdeaService
            idea_service = IdeaService(repository=idea_repo, event_bus=event_bus)
            
            # Register with certification
            context.registry_manager.register_service_with_certification(
                service_type=IdeaService,
                instance=idea_service,
                service_id='IdeaService',
                category='domain_service',
                description='Core epistemic service for idea creation, storage, and evolution',
                requires_initialize=False
            )
            
            created_services.append('IdeaService')
            logger.info('✓ IdeaService emerged with epistemic capabilities')
            
        except Exception as e:
            error_msg = f'Failed to emerge IdeaService: {e}'
            errors.append(error_msg)
            logger.error(error_msg)

    async def _emit_abiogenesis_signal(self, context, created_services: list) -> None:
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
            logger.warning(f'Failed to emit L0 Abiogenesis signal: {e}')

    def should_skip_phase(self, context) -> tuple[bool, str]:
        return (False, '')