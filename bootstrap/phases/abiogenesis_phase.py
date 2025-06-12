from __future__ import annotations
import logging
from typing import Any, Dict, List, Tuple, Optional
from enum import Enum
from datetime import datetime, timezone

from .base_phase import BootstrapPhase, PhaseResult

logger = logging.getLogger(__name__)

class ServiceResolutionMode(Enum):
    STRICT = 'strict'
    PRODUCTION = 'production'
    DEVELOPMENT = 'development'
    TESTING = 'testing'

class ServiceImplementationType(Enum):
    REAL_IMPLEMENTATION = 'real'
    PLACEHOLDER = 'placeholder' 
    PROVIDED_INSTANCE = 'provided'
    MANIFEST_CONFIGURED = 'manifest'

class ServiceResolutionResult:
    def __init__(self, service_id: str, implementation_type: ServiceImplementationType, 
                 instance: Any, class_path: Optional[str] = None, 
                 fallback_reason: Optional[str] = None):
        self.service_id = service_id
        self.implementation_type = implementation_type
        self.instance = instance
        self.class_path = class_path
        self.fallback_reason = fallback_reason
        self.resolved_at = datetime.now(timezone.utc)

    def is_placeholder(self) -> bool:
        return self.implementation_type == ServiceImplementationType.PLACEHOLDER

    def to_dict(self) -> Dict[str, Any]:
        return {
            'service_id': self.service_id,
            'implementation_type': self.implementation_type.value,
            'class_name': self.instance.__class__.__name__,
            'class_path': self.class_path,
            'fallback_reason': self.fallback_reason,
            'resolved_at': self.resolved_at.isoformat(),
            'is_placeholder': self.is_placeholder()
        }

class AbiogenesisPhase(BootstrapPhase):
    def __init__(self):
        super().__init__()
        self.service_resolutions: List[ServiceResolutionResult] = []
        self.resolution_mode: ServiceResolutionMode = ServiceResolutionMode.DEVELOPMENT
        self.config_provider: Optional[Any] = None

    async def execute(self, context) -> PhaseResult:
        logger.info('Executing Enhanced L0 Abiogenesis with V2 context integration')
        
        self._initialize_enhanced_config(context)
        self.resolution_mode = self._determine_resolution_mode(context)
        
        logger.info(f'Service resolution mode: {self.resolution_mode.value}')
        
        errors = []
        warnings = []
        created_services = []

        try:
            await self._emerge_feature_flags_manager(context, created_services, errors)
            await self._emerge_component_registry(context, created_services, errors)
            await self._emerge_core_ports_enhanced(context, created_services, errors, warnings)
            await self._emerge_idea_service(context, created_services, errors)
            await self._verify_service_emergence_with_contexts(context, created_services, warnings)
            
            placeholder_analysis = self._analyze_placeholder_usage()
            warnings.extend(placeholder_analysis['warnings'])
            errors.extend(placeholder_analysis['errors'])
            
            await self._emit_enhanced_abiogenesis_signal(context, created_services)
            
            success = len(errors) == 0
            message = self._create_completion_message(created_services, placeholder_analysis)
            
            return PhaseResult(
                success=success,
                message=message,
                errors=errors,
                warnings=warnings,
                metadata={
                    'services_emerged': created_services,
                    'l0_abiogenesis': True,
                    'emergence_count': len(created_services),
                    'resolution_mode': self.resolution_mode.value,
                    'service_resolutions': [sr.to_dict() for sr in self.service_resolutions],
                    'placeholder_count': len([sr for sr in self.service_resolutions if sr.is_placeholder()]),
                    'real_service_count': len([sr for sr in self.service_resolutions if not sr.is_placeholder()]),
                    'config_provider_enabled': self.config_provider is not None,
                    'v2_integration': True
                }
            )
            
        except Exception as e:
            error_msg = f'Critical failure during Enhanced L0 Abiogenesis: {e}'
            logger.error(error_msg, exc_info=True)
            
            return PhaseResult.failure_result(
                message='Enhanced L0 Abiogenesis failed',
                errors=[error_msg],
                warnings=warnings,
                metadata={
                    'abiogenesis_failed': True,
                    'resolution_mode': self.resolution_mode.value,
                    'partial_resolutions': [sr.to_dict() for sr in self.service_resolutions],
                    'v2_integration': True
                }
            )

    def _initialize_enhanced_config(self, context) -> None:
        try:
            # Import here to avoid circular imports
            from bootstrap.bootstrap_helper.context_helper import SimpleConfigProvider
            
            feature_flags = context.global_app_config.get('feature_flags', {})
            abiogenesis_config = context.global_app_config.get('abiogenesis', {})
            
            enhanced_config = {
                **feature_flags,
                **{f'abiogenesis.{k}': v for k, v in abiogenesis_config.items()}
            }
            
            self.config_provider = SimpleConfigProvider(enhanced_config)
            logger.debug(f'Enhanced config provider initialized with {len(enhanced_config)} configuration entries')
            
        except Exception as e:
            logger.warning(f'Failed to initialize enhanced config provider: {e}')
            self.config_provider = None

    def _determine_resolution_mode(self, context) -> ServiceResolutionMode:
        if self.config_provider:
            explicit_mode = self.config_provider.get_config('abiogenesis', 'resolution_mode')
        else:
            explicit_mode = context.global_app_config.get('abiogenesis', {}).get('resolution_mode')
        
        if explicit_mode:
            try:
                return ServiceResolutionMode(explicit_mode)
            except ValueError:
                logger.warning(f'Invalid resolution mode "{explicit_mode}", falling back to environment detection')
        
        env = context.global_app_config.get('env', 'development').lower()
        if env in ['prod', 'production']:
            return ServiceResolutionMode.PRODUCTION
        elif env in ['test', 'testing', 'pytest']:
            return ServiceResolutionMode.TESTING
        elif env in ['strict']:
            return ServiceResolutionMode.STRICT
        else:
            return ServiceResolutionMode.DEVELOPMENT

    async def _emerge_feature_flags_manager(self, context, created_services: list, errors: list) -> None:
        try:
            # Import here to avoid circular imports
            from bootstrap.bootstrap_helper.feature_flags import FeatureFlagsManager
            
            # Check if already exists
            try:
                existing_ff = context.registry.get_service_instance(FeatureFlagsManager)
                logger.info('FeatureFlagsManager already exists in registry')
                return
            except:
                pass
            
            if self.config_provider:
                feature_flags_config = self.config_provider.get_all_config()
                feature_flags_config = {k: v for k, v in feature_flags_config.items() 
                                      if not k.startswith('abiogenesis.')}
            else:
                feature_flags_config = context.global_app_config.get('feature_flags', {})
            
            ff_manager = FeatureFlagsManager(feature_flags_config)
            
            context.registry_manager.register_service_with_certification(
                service_type=FeatureFlagsManager,
                instance=ff_manager,
                service_id='FeatureFlagsManager',
                category='core_service',
                description='System-wide feature flag management for adaptive behavior (V2 enhanced)',
                requires_initialize=False
            )
            
            created_services.append('FeatureFlagsManager')
            logger.info('✓ FeatureFlagsManager emerged with enhanced V2 configuration support')
            
        except Exception as e:
            error_msg = f'Failed to emerge FeatureFlagsManager: {e}'
            errors.append(error_msg)
            logger.error(error_msg)

    async def _emerge_component_registry(self, context, created_services: list, errors: list) -> None:
        try:
            context.registry_manager.register_service_with_certification(
                service_type=type(context.registry),
                instance=context.registry,
                service_id='ComponentRegistry',
                category='core_service',
                description='Central component registry enabling system self-awareness (V2 enhanced)',
                requires_initialize=False
            )
            
            created_services.append('ComponentRegistry')
            logger.info('✓ ComponentRegistry achieved reflexive self-emergence with V2 integration')
            
        except Exception as e:
            error_msg = f'Failed to register ComponentRegistry reflexively: {e}'
            errors.append(error_msg)
            logger.error(error_msg)

    async def _emerge_core_ports_enhanced(self, context, created_services: list, errors: list, warnings: list) -> None:
        # Import here to avoid circular imports
        try:
            from domain.ports.llm_port import LLMPort
            from domain.ports.embedding_port import EmbeddingPort
            from domain.ports.event_bus_port import EventBusPort
            from domain.ports.idea_repository_port import IdeaRepositoryPort
            from bootstrap.bootstrap_helper.placeholders import (
                PlaceholderLLMPortImpl, PlaceholderEmbeddingPortImpl,
                PlaceholderEventBusImpl, PlaceholderIdeaRepositoryImpl
            )
        except ImportError as e:
            logger.error(f"Failed to import port types or placeholders: {e}")
            # Create minimal placeholders
            errors.append(f"Failed to import required port types: {e}")
            return
        
        manifest_shared_services = context.global_app_config.get('shared_services', {})
        
        port_configs = [
            (LLMPort, PlaceholderLLMPortImpl, 'LLMPort', 
             'Language model interface for epistemic reasoning'),
            (EmbeddingPort, PlaceholderEmbeddingPortImpl, 'EmbeddingPort', 
             'Vector embedding interface for semantic operations'),
            (EventBusPort, PlaceholderEventBusImpl, 'EventBusPort', 
             'Event publication and subscription for system communication'),
            (IdeaRepositoryPort, PlaceholderIdeaRepositoryImpl, 'IdeaRepositoryPort', 
             'Idea storage and retrieval for epistemic persistence')
        ]
        
        for port_type, placeholder_impl, service_id, description in port_configs:
            try:
                resolution_result = await self._emerge_service_port_enhanced(
                    context, port_type, placeholder_impl, service_id, 
                    description, manifest_shared_services
                )
                
                if resolution_result:
                    self.service_resolutions.append(resolution_result)
                    created_services.append(service_id)
                    
                    if resolution_result.is_placeholder():
                        logger.warning(f'⚠️  {service_id} resolved to PLACEHOLDER: {resolution_result.fallback_reason}')
                    else:
                        logger.info(f"✓ {service_id} resolved to real implementation: {resolution_result.class_path or 'provided instance'}")
                        
            except Exception as e:
                error_msg = f'Failed to emerge {service_id}: {e}'
                if self.resolution_mode == ServiceResolutionMode.STRICT:
                    errors.append(error_msg)
                else:
                    warnings.append(error_msg)
                logger.error(error_msg)

    async def _emerge_service_port_enhanced(self, context, port_type, placeholder_impl, 
                                          service_id: str, description: str, 
                                          manifest_shared_services: dict) -> Optional[ServiceResolutionResult]:
        
        # Special handling for EventBusPort if provided
        if service_id == 'EventBusPort' and context.config.existing_event_bus:
            instance = context.config.existing_event_bus
            result = ServiceResolutionResult(
                service_id=service_id,
                implementation_type=ServiceImplementationType.PROVIDED_INSTANCE,
                instance=instance,
                class_path=f'{instance.__class__.__module__}.{instance.__class__.__name__}'
            )
            await self._register_service_instance(context, port_type, instance, service_id, description)
            return result
        
        # Check manifest for service specification
        service_spec = manifest_shared_services.get(service_id)
        if service_spec and service_spec.get('enabled', True):
            class_path = service_spec.get('class')
            if class_path:
                try:
                    logger.debug(f'Attempting to create {service_id} from manifest: {class_path}')
                    
                    # Import here to avoid circular imports
                    from runtime.utils import import_by_path
                    service_class = import_by_path(class_path)
                    instance = self._instantiate_service_class(service_class, service_spec.get('config', {}))
                    
                    result = ServiceResolutionResult(
                        service_id=service_id,
                        implementation_type=ServiceImplementationType.MANIFEST_CONFIGURED,
                        instance=instance,
                        class_path=class_path
                    )
                    
                    await self._register_service_instance(context, port_type, instance, service_id, description)
                    return result
                    
                except Exception as e:
                    fallback_reason = f'Failed to instantiate from manifest ({class_path}): {e}'
                    logger.warning(fallback_reason)
                    
                    if self.resolution_mode == ServiceResolutionMode.STRICT:
                        raise RuntimeError(f'Strict mode: Cannot fallback to placeholder for {service_id}. {fallback_reason}')
                    
                    return await self._create_placeholder_service(
                        context, port_type, placeholder_impl, service_id, description, fallback_reason
                    )
            else:
                fallback_reason = f'Manifest entry for {service_id} missing class path'
                logger.warning(fallback_reason)
                
                if self.resolution_mode == ServiceResolutionMode.STRICT:
                    raise RuntimeError(f'Strict mode: {fallback_reason}')
                    
                return await self._create_placeholder_service(
                    context, port_type, placeholder_impl, service_id, description, fallback_reason
                )
        
        elif service_spec and not service_spec.get('enabled', True):
            logger.info(f'Service {service_id} disabled in manifest. Skipping.')
            return None
        else:
            fallback_reason = f'No manifest entry found for {service_id}'
            
            if self.resolution_mode == ServiceResolutionMode.STRICT:
                raise RuntimeError(f'Strict mode: {fallback_reason}')
                
            return await self._create_placeholder_service(
                context, port_type, placeholder_impl, service_id, description, fallback_reason
            )

    async def _create_placeholder_service(self, context, port_type, placeholder_impl, 
                                        service_id: str, description: str, 
                                        fallback_reason: str) -> ServiceResolutionResult:
        instance = placeholder_impl()
        
        result = ServiceResolutionResult(
            service_id=service_id,
            implementation_type=ServiceImplementationType.PLACEHOLDER,
            instance=instance,
            class_path=f'{instance.__class__.__module__}.{instance.__class__.__name__}',
            fallback_reason=fallback_reason
        )
        
        await self._register_service_instance(context, port_type, instance, service_id, f'{description} [PLACEHOLDER]')
        return result

    def _instantiate_service_class(self, service_class, service_config: Dict[str, Any] = None) -> Any:
        config = service_config or {}
        
        # Try different constructor patterns
        try:
            return service_class(config=config)
        except TypeError:
            try:
                return service_class(cfg=config)
            except TypeError:
                try:
                    return service_class(**config)
                except TypeError:
                    try:
                        return service_class()
                    except TypeError:
                        return service_class(config)

    async def _register_service_instance(self, context, port_type, instance, service_id: str, description: str):
        context.registry_manager.register_service_with_certification(
            service_type=port_type,
            instance=instance,
            service_id=service_id,
            category='port_service',
            description=description,
            requires_initialize=False
        )

    async def _emerge_idea_service(self, context, created_services: list, errors: list) -> None:
        try:
            # Import here to avoid circular imports
            from application.services.idea_service import IdeaService
            
            # Check if already exists
            try:
                existing_idea_service = context.registry.get_service_instance(IdeaService)
                logger.info('IdeaService already exists in registry')
                return
            except:
                pass
            
            # Get dependencies
            try:
                from domain.ports.idea_repository_port import IdeaRepositoryPort
                from domain.ports.event_bus_port import EventBusPort
                
                idea_repo = context.registry.get_service_instance(IdeaRepositoryPort)
                event_bus = context.registry.get_service_instance(EventBusPort)
                
                idea_service = IdeaService(repository=idea_repo, event_bus=event_bus)
                
                context.registry_manager.register_service_with_certification(
                    service_type=IdeaService,
                    instance=idea_service,
                    service_id='IdeaService',
                    category='domain_service',
                    description='Core epistemic service for idea creation, storage, and evolution (V2 enhanced)',
                    requires_initialize=False
                )
                
                created_services.append('IdeaService')
                logger.info('✓ IdeaService emerged with enhanced V2 epistemic capabilities')
                
            except Exception as e:
                error_msg = f'Failed to get dependencies for IdeaService: {e}'
                errors.append(error_msg)
                logger.error(error_msg)
                
        except Exception as e:
            error_msg = f'Failed to emerge IdeaService: {e}'
            errors.append(error_msg)
            logger.error(error_msg)

    async def _verify_service_emergence_with_contexts(self, context, created_services: list, warnings: list) -> None:
        try:
            logger.debug('Verifying service emergence with V2 context isolation')
            
            for service_id in created_services:
                try:
                    # Import here to avoid circular imports
                    from bootstrap.bootstrap_helper.context_helper import create_minimal_context
                    
                    verification_context = create_minimal_context(
                        component_id=f'verify_{service_id}',
                        run_id=f'{context.run_id}_verification'
                    )
                    
                    service_instance = context.registry.get(service_id)
                    if service_instance is None:
                        warnings.append(f'Service {service_id} verification failed: service not found')
                    else:
                        logger.debug(f'✓ Service {service_id} verified in isolated context')
                        
                except Exception as e:
                    warnings.append(f'Service {service_id} verification failed: {e}')
                    logger.warning(f'Service verification failed for {service_id}: {e}')
                    
        except Exception as e:
            warnings.append(f'Service verification process failed: {e}')
            logger.warning(f'Service verification process failed: {e}')

    def _analyze_placeholder_usage(self) -> Dict[str, List[str]]:
        analysis = {'warnings': [], 'errors': []}
        
        placeholder_services = [sr for sr in self.service_resolutions if sr.is_placeholder()]
        real_services = [sr for sr in self.service_resolutions if not sr.is_placeholder()]
        
        if not placeholder_services:
            logger.info('✅ No placeholder services detected - all services using real implementations')
            return analysis
        
        placeholder_names = [ps.service_id for ps in placeholder_services]
        placeholder_summary = f"Placeholder services: {', '.join(placeholder_names)}"
        
        if self.resolution_mode == ServiceResolutionMode.STRICT:
            analysis['errors'].append(f'STRICT MODE VIOLATION: {placeholder_summary}')
        elif self.resolution_mode == ServiceResolutionMode.PRODUCTION:
            analysis['warnings'].append(f'PRODUCTION WARNING: {placeholder_summary}')
            analysis['warnings'].append('🚨 Using placeholder services in production may indicate misconfiguration')
            for ps in placeholder_services:
                analysis['warnings'].append(f'   - {ps.service_id}: {ps.fallback_reason}')
        elif self.resolution_mode == ServiceResolutionMode.DEVELOPMENT:
            analysis['warnings'].append(f'Development mode: {placeholder_summary}')
        
        logger.info(f'Service Resolution Summary (V2 Enhanced):')
        logger.info(f'  Real implementations: {len(real_services)}')
        logger.info(f'  Placeholder services: {len(placeholder_services)}')
        logger.info(f'  Resolution mode: {self.resolution_mode.value}')
        logger.info(f"  Config provider: {'enabled' if self.config_provider else 'disabled'}")
        
        for sr in self.service_resolutions:
            status = '🔴 PLACEHOLDER' if sr.is_placeholder() else '✅ REAL'
            logger.info(f'  {sr.service_id}: {status} ({sr.instance.__class__.__name__})')
        
        return analysis

    def _create_completion_message(self, created_services: List[str], placeholder_analysis: Dict[str, List[str]]) -> str:
        base_message = f'Enhanced L0 Abiogenesis V2 complete - {len(created_services)} services emerged'
        
        placeholder_count = len([sr for sr in self.service_resolutions if sr.is_placeholder()])
        real_count = len([sr for sr in self.service_resolutions if not sr.is_placeholder()])
        
        if placeholder_count == 0:
            return f'{base_message} (all real implementations)'
        else:
            status = 'with warnings' if placeholder_analysis['warnings'] else 'with placeholders'
            return f'{base_message} ({real_count} real, {placeholder_count} placeholder) {status}'

    async def _emit_enhanced_abiogenesis_signal(self, context, created_services: list) -> None:
        try:
            from bootstrap.signals.bootstrap_signals import L0_ABIOGENESIS_COMPLETE
            
            payload = {
                'services_emerged': created_services,
                'emergence_count': len(created_services),
                'l0_complete': True,
                'run_id': context.run_id,
                'resolution_mode': self.resolution_mode.value,
                'service_resolutions': [sr.to_dict() for sr in self.service_resolutions],
                'placeholder_count': len([sr for sr in self.service_resolutions if sr.is_placeholder()]),
                'real_service_count': len([sr for sr in self.service_resolutions if not sr.is_placeholder()]),
                'placeholder_services': [sr.service_id for sr in self.service_resolutions if sr.is_placeholder()],
                'v2_integration': True,
                'config_provider_enabled': self.config_provider is not None
            }
            
            await context.signal_emitter.emit_signal(signal_type=L0_ABIOGENESIS_COMPLETE, payload=payload)
            
        except Exception as e:
            logger.warning(f'Failed to emit enhanced L0 Abiogenesis signal: {e}')

    def should_skip_phase(self, context) -> tuple[bool, str]:
        return (False, '')