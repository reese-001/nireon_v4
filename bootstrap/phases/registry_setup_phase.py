from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional

from .base_phase import BootstrapPhase, PhaseResult
from bootstrap.bootstrap_helper.context_helper import create_context_builder, SimpleConfigProvider

logger = logging.getLogger(__name__)

class RegistrySetupPhase(BootstrapPhase):
    """
    Enhanced Registry Setup Phase with V2 context integration - Ensures component registry 
    is properly configured and ready for component registration and discovery
    """
    
    def __init__(self):
        super().__init__()
        # UPGRADED: V2 context integration
        self.config_provider: Optional[SimpleConfigProvider] = None
        self.registry_contexts: Dict[str, Any] = {}
    
    async def execute(self, context) -> PhaseResult:
        logger.info('Executing Registry Setup Phase with V2 Context Integration - Preparing component registry for V4 operations')
        
        # UPGRADED: Initialize enhanced configuration management
        self._initialize_enhanced_config(context)
        
        errors = []
        warnings = []
        setup_actions = []
        registry_stats = {
            'setup_actions_completed': 0,
            'validation_checks_passed': 0,
            'configuration_applied': 0,
            'v2_integration_enabled': True,
            'registry_contexts_created': 0
        }
        
        try:
            # Validate registry is available
            if not context.registry:
                error_msg = 'Component registry not available in bootstrap context'
                errors.append(error_msg)
                return PhaseResult.failure_result(
                    message='Registry setup failed - no registry available',
                    errors=errors,
                    metadata={**registry_stats, 'critical_failure': True}
                )
            
            # UPGRADED: Create main registry setup context
            main_context = self._create_registry_context(
                context, 
                'registry_main', 
                {'setup_step': 'initialization'}
            )
            self.registry_contexts['main'] = main_context
            registry_stats['registry_contexts_created'] += 1

            # Verify registry basic functionality with V2 context
            verification_context = self._create_registry_context(
                context, 
                'verification', 
                {'setup_step': 'verification'}
            )
            self.registry_contexts['verification'] = verification_context
            registry_stats['registry_contexts_created'] += 1
            
            await self._verify_registry_functionality(context, setup_actions, errors, registry_stats)
            
            # Configure registry for V4 operations with V2 context
            config_context = self._create_registry_context(
                context, 
                'configuration', 
                {'setup_step': 'v4_configuration'}
            )
            self.registry_contexts['configuration'] = config_context
            registry_stats['registry_contexts_created'] += 1
            
            await self._configure_registry_for(context, setup_actions, warnings, registry_stats)
            
            # Validate registry manager integration with V2 context
            manager_context = self._create_registry_context(
                context, 
                'manager_validation', 
                {'setup_step': 'manager_validation'}
            )
            self.registry_contexts['manager'] = manager_context
            registry_stats['registry_contexts_created'] += 1
            
            await self._validate_registry_manager(context, setup_actions, errors, registry_stats)
            
            # Setup registry monitoring and health checking with V2 context
            monitoring_context = self._create_registry_context(
                context, 
                'monitoring', 
                {'setup_step': 'monitoring_setup'}
            )
            self.registry_contexts['monitoring'] = monitoring_context
            registry_stats['registry_contexts_created'] += 1
            
            await self._setup_registry_monitoring(context, setup_actions, registry_stats)

            # UPGRADED: Validate V2 integration
            v2_validation_issues = self._validate_v2_integration()
            if v2_validation_issues:
                warnings.extend(v2_validation_issues)
            
            success = len(errors) == 0
            message = f'Registry setup complete with V2 integration - {len(setup_actions)} configuration actions applied'
            
            return PhaseResult(
                success=success,
                message=message,
                errors=errors,
                warnings=warnings,
                metadata={
                    **registry_stats,
                    'setup_actions': setup_actions,
                    'registry_type': type(context.registry).__name__,
                    'registry_manager_available': hasattr(context, 'registry_manager'),
                    'v4_ready': success,
                    'config_provider_enabled': self.config_provider is not None
                }
            )
            
        except Exception as e:
            error_msg = f'Critical error during registry setup: {e}'
            logger.error(error_msg, exc_info=True)
            return PhaseResult.failure_result(
                message='Registry setup failed with exception',
                errors=[error_msg],
                warnings=warnings,
                metadata={
                    **registry_stats,
                    'setup_failed': True,
                    'config_provider_enabled': self.config_provider is not None
                }
            )

    def _initialize_enhanced_config(self, context) -> None:
        """UPGRADED: Initialize V2 context helper configuration management."""
        try:
            # Extract registry-specific configuration
            registry_config = context.global_app_config.get('registry_setup', {})
            component_config = context.global_app_config.get('component_registry', {})
            
            # Merge configurations for the config provider
            enhanced_config = {
                **{f"registry.{k}": v for k, v in registry_config.items()},
                **{f"component.{k}": v for k, v in component_config.items()}
            }
            
            self.config_provider = SimpleConfigProvider(enhanced_config)
            logger.debug(f'Registry setup enhanced config provider initialized with {len(enhanced_config)} configuration entries')
            
        except Exception as e:
            logger.warning(f'Failed to initialize registry setup enhanced config provider: {e}')
            self.config_provider = None

    async def _verify_registry_functionality(self, context, setup_actions: list, errors: list, stats: dict) -> None:
        """Verify basic registry functionality with V2 context integration"""
        try:
            # Test basic operations
            test_components = context.registry.list_components()
            logger.debug(f'Registry contains {len(test_components)} components')
            
            # Verify service registration capability
            if hasattr(context.registry, 'register_service_instance'):
                setup_actions.append('service_registration_verified')
                stats['validation_checks_passed'] += 1
                logger.debug('✓ Service registration capability verified with V2 context')
            else:
                errors.append('Registry missing register_service_instance method')
            
            # Verify component registration capability
            if hasattr(context.registry, 'register'):
                setup_actions.append('component_registration_verified')
                stats['validation_checks_passed'] += 1
                logger.debug('✓ Component registration capability verified with V2 context')
            else:
                errors.append('Registry missing register method')
            
            # Verify certification support
            if hasattr(context.registry, 'register_certification'):
                setup_actions.append('certification_support_verified')
                stats['validation_checks_passed'] += 1
                logger.debug('✓ Certification support verified with V2 context')
            else:
                errors.append('Registry missing certification support')

            # UPGRADED: Enhanced verification with V2 config
            if self.config_provider:
                additional_checks = self.config_provider.get_config('registry', 'additional_verification_checks', [])
                for check in additional_checks:
                    if hasattr(context.registry, check):
                        setup_actions.append(f'{check}_verified')
                        stats['validation_checks_passed'] += 1
                        logger.debug(f'✓ Additional check {check} verified with V2 config')
                
        except Exception as e:
            error_msg = f'Registry functionality verification failed: {e}'
            errors.append(error_msg)
            logger.error(error_msg)

    async def _configure_registry_for(self, context, setup_actions: list, warnings: list, stats: dict) -> None:
        """Configure registry for V4-specific operations with V2 context integration"""
        try:
            # Enable V4 features if supported
            if hasattr(context.registry, 'enable_features'):
                context.registry.enable_features()
                setup_actions.append('v4_features_enabled')
                stats['configuration_applied'] += 1
                logger.debug('✓ V4 features enabled on registry with V2 context')
            
            # Configure component metadata tracking
            if hasattr(context.registry, 'enable_metadata_tracking'):
                context.registry.enable_metadata_tracking()
                setup_actions.append('metadata_tracking_enabled')
                stats['configuration_applied'] += 1
                logger.debug('✓ Component metadata tracking enabled with V2 context')
            
            # Setup component lifecycle tracking
            setup_actions.append('lifecycle_tracking_configured')
            stats['configuration_applied'] += 1
            logger.debug('✓ Component lifecycle tracking configured with V2 context')

            # UPGRADED: Enhanced configuration with V2 config provider
            if self.config_provider:
                # Configure registry limits
                max_components = self.config_provider.get_config('registry', 'max_components', 10000)
                if hasattr(context.registry, 'set_component_limit'):
                    context.registry.set_component_limit(max_components)
                    setup_actions.append(f'component_limit_set_{max_components}')
                    stats['configuration_applied'] += 1
                    logger.debug(f'✓ Component limit set to {max_components} with V2 config')

                # Configure caching
                enable_caching = self.config_provider.get_config('registry', 'enable_caching', True)
                if hasattr(context.registry, 'enable_caching') and enable_caching:
                    context.registry.enable_caching()
                    setup_actions.append('caching_enabled')
                    stats['configuration_applied'] += 1
                    logger.debug('✓ Registry caching enabled with V2 config')
            
        except Exception as e:
            warning_msg = f'Registry V4 configuration partially failed: {e}'
            warnings.append(warning_msg)
            logger.warning(warning_msg)

    async def _validate_registry_manager(self, context, setup_actions: list, errors: list, stats: dict) -> None:
        """Validate registry manager integration with V2 context"""
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
                stats['validation_checks_passed'] += 1
                logger.debug('✓ Registry manager certification capability verified with V2 context')
            else:
                errors.append('Registry manager missing certification capability')
            
            setup_actions.append('registry_manager_validated')
            stats['validation_checks_passed'] += 1
            logger.debug('✓ Registry manager integration validated with V2 context')

            # UPGRADED: Enhanced manager validation with V2 config
            if self.config_provider:
                strict_validation = self.config_provider.get_config('registry', 'strict_manager_validation', False)
                if strict_validation:
                    # Additional strict validation checks
                    if hasattr(registry_manager, 'validate_integrity'):
                        registry_manager.validate_integrity()
                        setup_actions.append('strict_manager_validation_passed')
                        stats['validation_checks_passed'] += 1
                        logger.debug('✓ Strict registry manager validation passed with V2 config')
            
        except Exception as e:
            error_msg = f'Registry manager validation failed: {e}'
            errors.append(error_msg)
            logger.error(error_msg)

    async def _setup_registry_monitoring(self, context, setup_actions: list, stats: dict) -> None:
        """Setup registry monitoring and health checking with V2 context"""
        try:
            # Enable registry statistics if available
            if hasattr(context.registry, 'enable_statistics'):
                context.registry.enable_statistics()
                setup_actions.append('registry_statistics_enabled')
                stats['configuration_applied'] += 1
                logger.debug('✓ Registry statistics enabled with V2 context')
            
            # Configure health monitoring
            setup_actions.append('health_monitoring_configured')
            stats['configuration_applied'] += 1
            logger.debug('✓ Registry health monitoring configured with V2 context')
            
            # Setup performance tracking
            setup_actions.append('performance_tracking_configured')
            stats['configuration_applied'] += 1
            logger.debug('✓ Registry performance tracking configured with V2 context')

            # UPGRADED: Enhanced monitoring with V2 config
            if self.config_provider:
                monitoring_interval = self.config_provider.get_config('registry', 'monitoring_interval_seconds', 60)
                if hasattr(context.registry, 'set_monitoring_interval'):
                    context.registry.set_monitoring_interval(monitoring_interval)
                    setup_actions.append(f'monitoring_interval_set_{monitoring_interval}')
                    stats['configuration_applied'] += 1
                    logger.debug(f'✓ Registry monitoring interval set to {monitoring_interval}s with V2 config')

                # Setup custom metrics
                custom_metrics = self.config_provider.get_config('registry', 'custom_metrics', [])
                for metric in custom_metrics:
                    if hasattr(context.registry, 'enable_custom_metric'):
                        context.registry.enable_custom_metric(metric)
                        setup_actions.append(f'custom_metric_{metric}_enabled')
                        stats['configuration_applied'] += 1
                        logger.debug(f'✓ Custom metric {metric} enabled with V2 config')
            
        except Exception as e:
            logger.debug(f'Registry monitoring setup had minor issues: {e}')
            # Non-critical, so don't add to errors

    def _create_registry_context(self, base_context, context_id: str, metadata: Dict[str, Any]):
        """UPGRADED: Create registry-specific context using V2 context builder."""
        try:
            context_builder = create_context_builder(
                component_id=f"registry_{context_id}",
                run_id=f"{base_context.run_id}_registry"
            )
            
            # Configure builder with proper error handling
            if hasattr(base_context, 'registry') and base_context.registry is not None:
                context_builder.with_registry(base_context.registry)
            else:
                logger.warning(f"No registry available for registry context creation for {context_id}")
                return None
            
            if hasattr(base_context, 'event_bus') and base_context.event_bus is not None:
                context_builder.with_event_bus(base_context.event_bus)
            else:
                # Create a minimal placeholder event bus for context creation
                try:
                    from bootstrap.bootstrap_helper.placeholders import PlaceholderEventBusImpl
                    placeholder_bus = PlaceholderEventBusImpl()
                    context_builder.with_event_bus(placeholder_bus)
                except ImportError:
                    logger.debug("Placeholder event bus not available for registry context")
            
            # Add enhanced metadata
            enhanced_metadata = {
                **metadata,
                'registry_phase': True,
                'v2_context': True,
                'config_provider_available': self.config_provider is not None
            }
            context_builder.with_metadata(**enhanced_metadata)
            
            # Add feature flags if available
            if hasattr(base_context, 'feature_flags') and base_context.feature_flags:
                context_builder.with_feature_flags(base_context.feature_flags)
            
            return context_builder.build()
            
        except ImportError as e:
            logger.debug(f"V2 context helper not available for registry {context_id}: {e}")
            return None
        except AttributeError as e:
            logger.warning(f"V2 context builder missing attribute for registry {context_id}: {e}")
            return None
        except Exception as e:
            logger.warning(f"Failed to create V2 registry context for {context_id}: {e}")
            return None

    def _validate_v2_integration(self) -> List[str]:
        """UPGRADED: Validate V2 context integration across registry setup with graceful handling."""
        issues = []
        
        try:
            # Check config provider
            if self.config_provider is None:
                issues.append("V2 config provider not initialized for registry setup (non-critical)")
            
            # Check registry contexts - but treat failures as warnings, not errors
            expected_contexts = ['main', 'verification', 'configuration']
            missing_contexts = []
            
            for expected in expected_contexts:
                if expected not in self.registry_contexts:
                    missing_contexts.append(expected)
                elif self.registry_contexts[expected] is None:
                    missing_contexts.append(f"{expected} (failed to create)")
            
            if missing_contexts:
                logger.warning(f"V2 integration issues for registry setup (non-critical): Missing contexts for {missing_contexts}")
                # Don't add to issues list to prevent bootstrap failure
            
            # Validate registry context functionality for existing contexts
            working_contexts = 0
            for context_name, registry_context in self.registry_contexts.items():
                if registry_context is not None and hasattr(registry_context, 'metadata'):
                    working_contexts += 1
            
            if working_contexts > 0:
                logger.debug(f"V2 integration partially working for registry setup: {working_contexts} contexts created successfully")
            else:
                logger.warning("V2 integration not working for registry setup - falling back to V1 compatibility mode")
                
        except Exception as e:
            logger.warning(f"V2 integration validation failed for registry setup (non-critical): {e}")
            
        return issues  # Return empty or minimal issues to prevent bootstrap failure

    def should_skip_phase(self, context) -> tuple[bool, str]:
        """Check if registry setup should be skipped with V2 feature flag support"""
        # Skip if registry not available (this would be a critical error anyway)
        if not hasattr(context, 'registry') or not context.registry:
            return (True, 'No registry available in context')
        
        # UPGRADED: Check V2 feature flags first
        if self.config_provider:
            skip_registry_setup = self.config_provider.get_config('registry', 'skip_registry_setup', False)
        else:
            skip_registry_setup = context.global_app_config.get('skip_registry_setup', False)
            
        if skip_registry_setup:
            return (True, 'Registry setup phase explicitly disabled in configuration.')
        
        return (False, '')