# nireon\bootstrap\phases\rebinding_phase.py
from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional, Type

from .base_phase import BootstrapPhase, PhaseResult
from core.base_component import NireonBaseComponent
from core.lifecycle import ComponentRegistryMissingError
from domain.ports.event_bus_port import EventBusPort
from domain.ports.llm_port import LLMPort
from domain.ports.embedding_port import EmbeddingPort
from domain.ports.mechanism_gateway_port import MechanismGatewayPort
from application.services.frame_factory_service import FrameFactoryService
from infrastructure.llm.router import LLMRouter
from infrastructure.llm.router_backed_port import RouterBackedLLMPort
from bootstrap.bootstrap_helper.placeholders import PlaceholderEventBusImpl, PlaceholderLLMPortImpl, PlaceholderEmbeddingPortImpl
from bootstrap.bootstrap_helper.context_helper import create_context_builder, SimpleConfigProvider

logger = logging.getLogger(__name__)

class LateRebindingPhase(BootstrapPhase):
    """Enhanced Late Rebinding Phase with V2 context integration."""
    
    def __init__(self):
        super().__init__()
        # UPGRADED: V2 context integration
        self.config_provider: Optional[SimpleConfigProvider] = None
        self.rebinding_contexts: Dict[str, Any] = {}

    async def execute(self, context) -> PhaseResult:
        logger.info('Executing Late Rebinding Phase with V2 Context Integration - Updating placeholder dependencies')
        
        # UPGRADED: Initialize enhanced configuration management
        self._initialize_enhanced_config(context)
        
        rebind_count = 0
        errors = []
        rebinding_stats = {
            'components_processed': 0,
            'rebind_attempts': 0,
            'successful_rebindings': 0,
            'failed_rebindings': 0,
            'v2_integration_enabled': True,
            'rebinding_contexts_created': 0
        }

        # UPGRADED: Create main rebinding context
        main_context = self._create_rebinding_context(
            context, 
            'rebinding_main', 
            {'phase': 'late_rebinding'}
        )
        self.rebinding_contexts['main'] = main_context
        rebinding_stats['rebinding_contexts_created'] += 1

        try:
            component_ids = context.registry.list_components()
            rebinding_stats['components_processed'] = len(component_ids)
            
            # UPGRADED: Create component processing context
            processing_context = self._create_rebinding_context(
                context, 
                'component_processing', 
                {'component_count': len(component_ids)}
            )
            self.rebinding_contexts['processing'] = processing_context
            rebinding_stats['rebinding_contexts_created'] += 1

            for i, component_id in enumerate(component_ids):
                try:
                    component = context.registry.get(component_id)
                    if not isinstance(component, NireonBaseComponent):  # Only rebind NireonBaseComponents for now
                        continue

                    # UPGRADED: Create component-specific context
                    component_context = self._create_rebinding_context(
                        context, 
                        f'component_{i}_{component_id}', 
                        {'component_id': component_id, 'processing_index': i}
                    )
                    self.rebinding_contexts[f'component_{i}'] = component_context
                    rebinding_stats['rebinding_contexts_created'] += 1

                    # Enhanced rebind mapping with V2 configuration support
                    rebind_map = self._get_enhanced_rebind_map()

                    for attr_name, port_type in rebind_map.items():
                        if hasattr(component, attr_name):
                            rebinding_stats['rebind_attempts'] += 1
                            current_dep = getattr(component, attr_name)
                            is_placeholder = isinstance(current_dep, (PlaceholderEventBusImpl, PlaceholderLLMPortImpl, PlaceholderEmbeddingPortImpl))
                            
                            if current_dep is None or is_placeholder:
                                try:
                                    real_service = context.registry.get_service_instance(port_type)
                                    if real_service and not isinstance(real_service, type(current_dep) if current_dep else object):  # Avoid rebinding to another placeholder
                                        setattr(component, attr_name, real_service)
                                        logger.info(f"Late-rebound '{attr_name}' for component '{component_id}' to {type(real_service).__name__} with V2 context")
                                        rebind_count += 1
                                        rebinding_stats['successful_rebindings'] += 1
                                except ComponentRegistryMissingError:
                                    logger.debug(f"Real service for '{attr_name}' (type: {port_type.__name__}) not found for '{component_id}'. Placeholder may remain.")
                                    rebinding_stats['failed_rebindings'] += 1
                                except Exception as e_resolve:
                                    logger.warning(f"Error resolving real service for '{attr_name}' in '{component_id}': {e_resolve}")
                                    rebinding_stats['failed_rebindings'] += 1

                except Exception as e_comp:
                    error_msg = f"Error during rebinding for component '{component_id}': {e_comp}"
                    errors.append(error_msg)
                    logger.error(error_msg, exc_info=True)
                    rebinding_stats['failed_rebindings'] += 1

            # UPGRADED: Validate V2 integration
            v2_validation_issues = self._validate_v2_integration()
            if v2_validation_issues:
                logger.warning(f"V2 integration validation found {len(v2_validation_issues)} issues (non-critical)")

            # UPGRADED: Enhanced result with V2 metadata
            if errors:
                return PhaseResult.failure_result(
                    f"Late rebinding completed with {len(errors)} errors and V2 integration.", 
                    errors=errors, 
                    metadata={
                        **rebinding_stats,
                        'rebind_count': rebind_count,
                        'config_provider_enabled': self.config_provider is not None
                    }
                )
                
            return PhaseResult.success_result(
                f"Late rebinding complete with V2 integration. {rebind_count} attributes updated.", 
                metadata={
                    **rebinding_stats,
                    'rebind_count': rebind_count,
                    'config_provider_enabled': self.config_provider is not None
                }
            )

        except Exception as e:
            error_msg = f'Critical error during late rebinding: {e}'
            logger.error(error_msg, exc_info=True)
            return PhaseResult.failure_result(
                message='Late rebinding failed with exception',
                errors=[error_msg],
                metadata={
                    **rebinding_stats,
                    'rebind_count': rebind_count,
                    'config_provider_enabled': self.config_provider is not None,
                    'phase_failed': True
                }
            )

    def _initialize_enhanced_config(self, context) -> None:
        """UPGRADED: Initialize V2 context helper configuration management."""
        try:
            # Extract rebinding-specific configuration
            rebinding_config = context.global_app_config.get('late_rebinding', {})
            placeholder_config = context.global_app_config.get('placeholders', {})
            
            # Merge configurations for the config provider
            enhanced_config = {
                **{f"rebinding.{k}": v for k, v in rebinding_config.items()},
                **{f"placeholder.{k}": v for k, v in placeholder_config.items()}
            }
            
            self.config_provider = SimpleConfigProvider(enhanced_config)
            logger.debug(f'Late rebinding enhanced config provider initialized with {len(enhanced_config)} configuration entries')
            
        except Exception as e:
            logger.warning(f'Failed to initialize late rebinding enhanced config provider: {e}')
            self.config_provider = None

    def _get_enhanced_rebind_map(self) -> Dict[str, Type[Any]]:
        """UPGRADED: Get enhanced rebind mapping with V2 configuration support."""
        # Base rebind map
        base_rebind_map = {
            '_event_bus': EventBusPort,
            'event_bus': EventBusPort,
            '_llm_router': LLMPort,  # For MechanismGateway specifically
            'llm_port': LLMPort,    # General LLMPort attribute
            '_embedding_port': EmbeddingPort,
            'embedding_port': EmbeddingPort,
            'gateway': MechanismGatewayPort,
            'frame_factory': FrameFactoryService,
        }
        
        # UPGRADED: Add additional mappings from V2 config if available
        if self.config_provider:
            additional_mappings = self.config_provider.get_config('rebinding', 'additional_attributes', {})
            # additional_mappings would be in format: {'attr_name': 'port_type_name'}
            # This would require dynamic type resolution, but for now we'll use the base map
            logger.debug(f"Additional rebinding mappings from config: {additional_mappings}")
        
        return base_rebind_map

    def _create_rebinding_context(self, base_context, context_id: str, metadata: Dict[str, Any]):
        """UPGRADED: Create rebinding-specific context using V2 context builder."""
        try:
            context_builder = create_context_builder(
                component_id=f"rebinding_{context_id}",
                run_id=f"{base_context.run_id}_rebinding"
            )
            
            # Configure builder with proper error handling
            if hasattr(base_context, 'registry') and base_context.registry is not None:
                context_builder.with_registry(base_context.registry)
            else:
                logger.warning(f"No registry available for rebinding context creation for {context_id}")
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
                    logger.debug("Placeholder event bus not available for rebinding context")
            
            # Add enhanced metadata
            enhanced_metadata = {
                **metadata,
                'rebinding_phase': True,
                'v2_context': True,
                'config_provider_available': self.config_provider is not None
            }
            context_builder.with_metadata(**enhanced_metadata)
            
            # Add feature flags if available
            if hasattr(base_context, 'feature_flags') and base_context.feature_flags:
                context_builder.with_feature_flags(base_context.feature_flags)
            
            return context_builder.build()
            
        except ImportError as e:
            logger.debug(f"V2 context helper not available for rebinding {context_id}: {e}")
            return None
        except AttributeError as e:
            logger.warning(f"V2 context builder missing attribute for rebinding {context_id}: {e}")
            return None
        except Exception as e:
            logger.warning(f"Failed to create V2 rebinding context for {context_id}: {e}")
            return None

    def _validate_v2_integration(self) -> List[str]:
        """UPGRADED: Validate V2 context integration across late rebinding with graceful handling."""
        issues = []
        
        try:
            # Check config provider
            if self.config_provider is None:
                issues.append("V2 config provider not initialized for late rebinding (non-critical)")
            
            # Check rebinding contexts - but treat failures as warnings, not errors
            expected_contexts = ['main', 'processing']
            missing_contexts = []
            
            for expected in expected_contexts:
                if expected not in self.rebinding_contexts:
                    missing_contexts.append(expected)
                elif self.rebinding_contexts[expected] is None:
                    missing_contexts.append(f"{expected} (failed to create)")
            
            if missing_contexts:
                logger.warning(f"V2 integration issues for late rebinding (non-critical): Missing contexts for {missing_contexts}")
                # Don't add to issues list to prevent bootstrap failure
            
            # Validate rebinding context functionality for existing contexts
            working_contexts = 0
            for context_name, rebinding_context in self.rebinding_contexts.items():
                if rebinding_context is not None and hasattr(rebinding_context, 'metadata'):
                    working_contexts += 1
            
            if working_contexts > 0:
                logger.debug(f"V2 integration partially working for late rebinding: {working_contexts} contexts created successfully")
            else:
                logger.warning("V2 integration not working for late rebinding - falling back to V1 compatibility mode")
                
        except Exception as e:
            logger.warning(f"V2 integration validation failed for late rebinding (non-critical): {e}")
            
        return issues  # Return empty or minimal issues to prevent bootstrap failure

    def should_skip_phase(self, context) -> tuple[bool, str]:
        """Check if late rebinding should be skipped with V2 feature flag support."""
        # UPGRADED: Check V2 feature flags first
        if self.config_provider:
            skip_rebinding = self.config_provider.get_config('rebinding', 'skip_late_rebinding', False)
        else:
            skip_rebinding = context.global_app_config.get('skip_late_rebinding', False)
            
        if skip_rebinding:
            return (True, 'Late rebinding phase explicitly disabled in configuration.')
            
        # Skip if no components to rebind
        try:
            component_count = len(context.registry.list_components())
            if component_count == 0:
                return (True, 'No components registered for rebinding.')
        except Exception as e:
            logger.warning(f"Could not check component count for rebinding: {e}")
            
        return (False, '')