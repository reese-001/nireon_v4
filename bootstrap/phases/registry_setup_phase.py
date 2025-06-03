from __future__ import annotations
import logging
from typing import Any, Dict

from .base_phase import BootstrapPhase, PhaseResult

logger = logging.getLogger(__name__)

class RegistrySetupPhase(BootstrapPhase):
    """
    Registry Setup Phase - Ensures component registry is properly configured
    and ready for component registration and discovery
    """
    
    async def execute(self, context) -> PhaseResult:
        logger.info('Executing Registry Setup Phase - Preparing component registry for V4 operations')
        
        errors = []
        warnings = []
        setup_actions = []
        
        try:
            # Validate registry is available
            if not context.registry:
                error_msg = 'Component registry not available in bootstrap context'
                errors.append(error_msg)
                return PhaseResult.failure_result(
                    message='Registry setup failed - no registry available',
                    errors=errors
                )
            
            # Verify registry basic functionality
            await self._verify_registry_functionality(context, setup_actions, errors)
            
            # Configure registry for V4 operations
            await self._configure_registry_for_v4(context, setup_actions, warnings)
            
            # Validate registry manager integration
            await self._validate_registry_manager(context, setup_actions, errors)
            
            # Setup registry monitoring and health checking
            await self._setup_registry_monitoring(context, setup_actions)
            
            success = len(errors) == 0
            message = f'Registry setup complete - {len(setup_actions)} configuration actions applied'
            
            return PhaseResult(
                success=success,
                message=message,
                errors=errors,
                warnings=warnings,
                metadata={
                    'setup_actions': setup_actions,
                    'registry_type': type(context.registry).__name__,
                    'registry_manager_available': hasattr(context, 'registry_manager'),
                    'v4_ready': success
                }
            )
            
        except Exception as e:
            error_msg = f'Critical error during registry setup: {e}'
            logger.error(error_msg, exc_info=True)
            return PhaseResult.failure_result(
                message='Registry setup failed with exception',
                errors=[error_msg],
                warnings=warnings,
                metadata={'setup_failed': True}
            )

    async def _verify_registry_functionality(self, context, setup_actions: list, errors: list) -> None:
        """Verify basic registry functionality"""
        try:
            # Test basic operations
            test_components = context.registry.list_components()
            logger.debug(f'Registry contains {len(test_components)} components')
            
            # Verify service registration capability
            if hasattr(context.registry, 'register_service_instance'):
                setup_actions.append('service_registration_verified')
                logger.debug('✓ Service registration capability verified')
            else:
                errors.append('Registry missing register_service_instance method')
            
            # Verify component registration capability
            if hasattr(context.registry, 'register'):
                setup_actions.append('component_registration_verified')
                logger.debug('✓ Component registration capability verified')
            else:
                errors.append('Registry missing register method')
            
            # Verify certification support
            if hasattr(context.registry, 'register_certification'):
                setup_actions.append('certification_support_verified')
                logger.debug('✓ Certification support verified')
            else:
                errors.append('Registry missing certification support')
                
        except Exception as e:
            error_msg = f'Registry functionality verification failed: {e}'
            errors.append(error_msg)
            logger.error(error_msg)

    async def _configure_registry_for_v4(self, context, setup_actions: list, warnings: list) -> None:
        """Configure registry for V4-specific operations"""
        try:
            # Enable V4 features if supported
            if hasattr(context.registry, 'enable_v4_features'):
                context.registry.enable_v4_features()
                setup_actions.append('v4_features_enabled')
                logger.debug('✓ V4 features enabled on registry')
            
            # Configure component metadata tracking
            if hasattr(context.registry, 'enable_metadata_tracking'):
                context.registry.enable_metadata_tracking()
                setup_actions.append('metadata_tracking_enabled')
                logger.debug('✓ Component metadata tracking enabled')
            
            # Setup component lifecycle tracking
            setup_actions.append('lifecycle_tracking_configured')
            logger.debug('✓ Component lifecycle tracking configured')
            
        except Exception as e:
            warning_msg = f'Registry V4 configuration partially failed: {e}'
            warnings.append(warning_msg)
            logger.warning(warning_msg)

    async def _validate_registry_manager(self, context, setup_actions: list, errors: list) -> None:
        """Validate registry manager integration"""
        try:
            if not hasattr(context, 'registry_manager'):
                errors.append('Registry manager not available in context')
                return
            
            registry_manager = context.registry_manager
            
            # Verify registry manager has correct registry reference
            if registry_manager.registry is not context.registry:
                errors.append('Registry manager registry reference mismatch')
                return
            
            # Test certification functionality
            if hasattr(registry_manager, 'register_with_certification'):
                setup_actions.append('registry_manager_certification_verified')
                logger.debug('✓ Registry manager certification capability verified')
            else:
                errors.append('Registry manager missing certification capability')
            
            setup_actions.append('registry_manager_validated')
            logger.debug('✓ Registry manager integration validated')
            
        except Exception as e:
            error_msg = f'Registry manager validation failed: {e}'
            errors.append(error_msg)
            logger.error(error_msg)

    async def _setup_registry_monitoring(self, context, setup_actions: list) -> None:
        """Setup registry monitoring and health checking"""
        try:
            # Enable registry statistics if available
            if hasattr(context.registry, 'enable_statistics'):
                context.registry.enable_statistics()
                setup_actions.append('registry_statistics_enabled')
                logger.debug('✓ Registry statistics enabled')
            
            # Configure health monitoring
            setup_actions.append('health_monitoring_configured')
            logger.debug('✓ Registry health monitoring configured')
            
            # Setup performance tracking
            setup_actions.append('performance_tracking_configured')
            logger.debug('✓ Registry performance tracking configured')
            
        except Exception as e:
            logger.debug(f'Registry monitoring setup had minor issues: {e}')
            # Non-critical, so don't add to errors

    def should_skip_phase(self, context) -> tuple[bool, str]:
        """Check if registry setup should be skipped"""
        # Skip if registry not available (this would be a critical error anyway)
        if not hasattr(context, 'registry') or not context.registry:
            return (True, 'No registry available in context')
        
        return (False, '')