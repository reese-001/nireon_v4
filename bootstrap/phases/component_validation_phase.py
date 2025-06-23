from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional

from domain.context import NireonExecutionContext
from bootstrap.context.bootstrap_context import BootstrapContext
from bootstrap.validation_data import ComponentValidationData
from .base_phase import BootstrapPhase, PhaseResult
from bootstrap.bootstrap_helper.context_helper import build_validation_context
from core.base_component import NireonBaseComponent
from core.lifecycle import ComponentMetadata, ComponentRegistryMissingError

logger = logging.getLogger(__name__)

class InterfaceValidationPhase(BootstrapPhase):
    """Phase to validate component interfaces and contracts."""

    async def execute(self, context: BootstrapContext) -> PhaseResult:
        logger.info('Executing Interface Validation Phase - Validating component interfaces and contracts')
        
        errors = []
        warnings = []
        validation_stats = {
            'total_components': 0,
            'components_validated': 0,
            'validation_passed': 0,
            'validation_failed': 0,
            'validation_skipped': 0,
            'interface_checks': 0,
            'metadata_checks': 0,
            'contract_violations': 0
        }
        
        try:
            # Check if interface validator is available
            if not hasattr(context, 'interface_validator') or context.interface_validator is None:
                warning_msg = 'Interface validator not available on BootstrapContext - skipping detailed validation'
                warnings.append(warning_msg)
                logger.warning(warning_msg)
                return PhaseResult.success_result(
                    message='Interface validation skipped - no validator available on BootstrapContext',
                    warnings=warnings,
                    metadata=validation_stats
                )
            
            all_component_ids = context.registry.list_components()
            validation_stats['total_components'] = len(all_component_ids)
            
            if not all_component_ids:
                logger.info('No components registered for validation')
                return PhaseResult.success_result(
                    message='No components to validate',
                    metadata=validation_stats
                )
            
            logger.info(f'Validating {len(all_component_ids)} registered components')
            
            # Filter out type-based keys and validate only actual component instances
            actual_components = []
            for component_id in all_component_ids:
                if self._should_validate_component(component_id, context):
                    actual_components.append(component_id)
                else:
                    validation_stats['validation_skipped'] += 1
                    logger.debug(f'Skipping validation for type-based key: {component_id}')
            
            logger.info(f'Filtered to {len(actual_components)} actual components for validation')
            
            # Validate each actual component
            for component_id in actual_components:
                try:
                    # Get validation data from store
                    validation_data_obj = None
                    if hasattr(context, 'validation_data_store') and context.validation_data_store:
                        validation_data_obj = context.validation_data_store.get_validation_data_for_component(component_id)
                    
                    # Create validation context
                    validation_run_context = build_validation_context(
                        component_id=component_id,
                        base_context=context.with_component_scope(component_id),
                        validation_data=validation_data_obj.__dict__ if validation_data_obj else {}
                    )
                    
                    await self._validate_single_component(
                        component_id, context, validation_run_context, 
                        validation_stats, errors, warnings
                    )
                    validation_stats['components_validated'] += 1
                    
                except Exception as e:
                    validation_stats['validation_failed'] += 1
                    error_msg = f'Critical error validating component {component_id}: {e}'
                    if context.strict_mode:
                        errors.append(error_msg)
                        logger.error(error_msg)
                    else:
                        warnings.append(error_msg)
                        logger.warning(error_msg)
            
            # Perform cross-component validation
            await self._perform_cross_component_validation(context, validation_stats, errors, warnings)
            
            # Update health reporter
            await self._update_health_reporter_with_validation_results(context, validation_stats)
            
            # Emit validation signals
            await self._emit_validation_signals(context, validation_stats)
            
            # Determine success
            success = len(errors) == 0 or not context.strict_mode
            message = f"Interface validation complete - {validation_stats['validation_passed']}/{validation_stats['components_validated']} components passed validation"
            
            return PhaseResult(
                success=success,
                message=message,
                errors=errors,
                warnings=warnings,
                metadata=validation_stats
            )
            
        except Exception as e:
            error_msg = f'Critical error during interface validation: {e}'
            logger.error(error_msg, exc_info=True)
            return PhaseResult.failure_result(
                message='Interface validation failed',
                errors=[error_msg],
                warnings=warnings,
                metadata=validation_stats
            )
    
    def _should_validate_component(self, component_id: str, context: BootstrapContext) -> bool:
        """Determine if a component should be validated based on its ID and characteristics."""
        
        # Skip type-based keys (they contain dots and uppercase class names)
        if self._is_type_based_key(component_id):
            return False
        
        # Try to get the component and its metadata
        try:
            component = context.registry.get(component_id)
            if component is None:
                logger.debug(f'Component {component_id} is None, skipping validation')
                return False
            
            # Only validate NireonBaseComponent instances
            if not isinstance(component, NireonBaseComponent):
                logger.debug(f'Component {component_id} is not a NireonBaseComponent, skipping validation')
                return False
            
            # Try to get metadata to ensure it's a real component
            try:
                metadata = context.registry.get_metadata(component_id)
                return metadata is not None
            except ComponentRegistryMissingError:
                # This might be an alias or type-based key
                logger.debug(f'No metadata found for {component_id}, treating as type-based key')
                return False
                
        except Exception as e:
            logger.debug(f'Error accessing component {component_id}: {e}')
            return False
        
        return True
    
    def _is_type_based_key(self, component_id: str) -> bool:
        """Check if a component ID appears to be a type-based registry key."""
        # Type-based keys typically contain module paths with dots
        # and class names, e.g., 'domain.ports.llm_port.LLMPort'
        if '.' in component_id:
            segments = component_id.split('.')
            # Check if any segment starts with uppercase (likely a class name)
            if any(segment and segment[0].isupper() for segment in segments):
                return True
        
        # Also check for common type patterns
        type_patterns = [
            'Port', 'Service', 'Manager', 'Factory', 'Validator', 
            'Gateway', 'Router', 'Adapter', 'Handler'
        ]
        
        for pattern in type_patterns:
            if component_id.endswith(pattern) and '.' in component_id:
                return True
        
        return False

    async def _validate_single_component(
        self, 
        component_id: str, 
        bootstrap_context: BootstrapContext,
        validation_exec_context: NireonExecutionContext, 
        validation_stats: dict, 
        errors: list, 
        warnings: list
    ) -> None:
        """Validate a single component."""
        
        logger.debug(f'Validating component: {component_id}')
        
        try:
            # Get component and metadata
            component = bootstrap_context.registry.get(component_id)
            metadata = await self._get_component_metadata_safe(component_id, component, bootstrap_context)
            
            if metadata is None:
                validation_stats['validation_skipped'] += 1
                warning_msg = f'Skipping validation for {component_id} - no metadata available'
                warnings.append(warning_msg)
                logger.debug(warning_msg)
                return
            
            # Get validation data from store
            validation_data_obj = None
            if hasattr(bootstrap_context, 'validation_data_store') and bootstrap_context.validation_data_store:
                validation_data_obj = bootstrap_context.validation_data_store.get_validation_data_for_component(component_id)
            
            # Debug logging for frame_factory_service
            if component_id.lower() == 'frame_factory_service':
                logger.debug(f"CRITICAL DEBUG for {component_id}: validation_data_obj is {'NOT None' if validation_data_obj else 'None'}")
                if validation_data_obj:
                    logger.debug(f"CRITICAL DEBUG for {component_id}: validation_data_obj.original_metadata.epistemic_tags = {validation_data_obj.original_metadata.epistemic_tags}")
                    logger.debug(f"CRITICAL DEBUG for {component_id}: actual_runtime_metadata.epistemic_tags = {metadata.epistemic_tags}")
            
            # Validate the component
            if isinstance(component, NireonBaseComponent):
                validation_errors = await self._validate_nireon_component(
                    component, metadata, validation_exec_context, 
                    validation_data_obj, validation_stats
                )
            else:
                validation_errors = await self._validate_custom_component(
                    component, metadata, validation_exec_context, validation_stats
                )
            
            # Handle validation results
            if validation_errors:
                validation_stats['validation_failed'] += 1
                validation_stats['contract_violations'] += len(validation_errors)
                error_msg = f'Component {component_id} failed validation: {validation_errors}'
                if bootstrap_context.strict_mode:
                    errors.append(error_msg)
                    logger.error(error_msg)
                else:
                    warnings.append(error_msg)
                    logger.warning(error_msg)
            else:
                validation_stats['validation_passed'] += 1
                logger.debug(f'✓ Component {component_id} passed validation')
                
        except Exception as e:
            validation_stats['validation_failed'] += 1
            error_msg = f'Exception during validation of {component_id}: {e}'
            logger.error(error_msg, exc_info=True)
            if bootstrap_context.strict_mode:
                errors.append(error_msg)
            else:
                warnings.append(error_msg)

    async def _get_component_metadata_safe(self, component_id: str, component: Any, context) -> Optional[ComponentMetadata]:
        """Safely get component metadata with fallback strategies."""
        
        # Try direct metadata lookup
        try:
            return context.registry.get_metadata(component_id)
        except ComponentRegistryMissingError:
            pass
        
        # Try to find metadata using matching strategies
        try:
            available_metadata_keys = []
            for existing_id in context.registry.list_components():
                try:
                    context.registry.get_metadata(existing_id)
                    available_metadata_keys.append(existing_id)
                except ComponentRegistryMissingError:
                    continue
            
            # Try simple name matching
            simple_component_name = component_id.split('.')[-1]
            for metadata_key in available_metadata_keys:
                if (metadata_key == simple_component_name or 
                    metadata_key.endswith(simple_component_name) or 
                    simple_component_name.endswith(metadata_key)):
                    logger.debug(f'Found metadata for {component_id} using key {metadata_key}')
                    return context.registry.get_metadata(metadata_key)
                    
        except Exception as e:
            logger.debug(f'Error in metadata key matching for {component_id}: {e}')
        
        # Try to get metadata from component instance
        try:
            if hasattr(component, 'metadata') and isinstance(component.metadata, ComponentMetadata):
                logger.debug(f'Using instance metadata for {component_id}')
                return component.metadata
        except Exception as e:
            logger.debug(f'Error getting instance metadata for {component_id}: {e}')
        
        # Create fallback metadata for non-NireonBaseComponent services
        try:
            if not isinstance(component, NireonBaseComponent):
                from bootstrap.bootstrap_helper.metadata import create_service_metadata
                fallback_metadata = create_service_metadata(
                    service_id=component_id,
                    service_name=component_id.split('.')[-1],
                    category='service',
                    description=f'Fallback metadata for {component_id}',
                    requires_initialize=False
                )
                logger.debug(f'Created fallback metadata for service {component_id}')
                return fallback_metadata
        except Exception as e:
            logger.debug(f'Error creating fallback metadata for {component_id}: {e}')
        
        logger.warning(f'Could not resolve metadata for component {component_id}')
        return None

    async def _validate_nireon_component(
        self, 
        component: NireonBaseComponent, 
        actual_runtime_metadata: ComponentMetadata,
        validation_context: NireonExecutionContext, 
        validation_data_from_store: Optional[ComponentValidationData],
        validation_stats: dict
    ) -> List[str]:
        """Validate a NireonBaseComponent instance."""
        
        validation_stats['interface_checks'] += 1
        validation_stats['metadata_checks'] += 1
        
        component_id_for_log = actual_runtime_metadata.id
        errors: List[str] = []
        
        # Determine expected metadata for comparison
        if validation_data_from_store:
            expected_metadata_for_comparison = validation_data_from_store.original_metadata
            resolved_config_for_comparison = validation_data_from_store.resolved_config
            manifest_spec_for_comparison = validation_data_from_store.manifest_spec
            logger.debug(f'[{component_id_for_log}] Using manifest-derived data for validation. Expected tags from store: {expected_metadata_for_comparison.epistemic_tags}')
        else:
            expected_metadata_for_comparison = actual_runtime_metadata
            resolved_config_for_comparison = component.config
            manifest_spec_for_comparison = {}
            logger.debug(f'[{component_id_for_log}] No manifest-derived data found. Using actual runtime metadata as expected for validation. Expected tags from runtime: {expected_metadata_for_comparison.epistemic_tags}')
        
        # Use interface validator if available
        try:
            interface_validator = validation_context.interface_validator
            if not interface_validator:
                return ['Interface validator not available on execution context for detailed validation']
            
            validation_errors_from_validator = await interface_validator.validate_component(
                instance=component,
                expected_metadata=expected_metadata_for_comparison,
                context=validation_context,
                yaml_config_at_instantiation=resolved_config_for_comparison,
                actual_runtime_metadata=actual_runtime_metadata,
                manifest_spec=manifest_spec_for_comparison
            )
            
            errors.extend(validation_errors_from_validator)
            return errors
            
        except Exception as e:
            logger.error(f'Error during NireonBaseComponent validation for {component_id_for_log}: {e}', exc_info=True)
            return [f'Validation error: {e}']

    async def _validate_custom_component(
        self, 
        component, 
        metadata, 
        validation_context: NireonExecutionContext, 
        validation_stats: dict
    ) -> List[str]:
        """Validate a custom (non-NireonBaseComponent) component."""
        
        validation_stats['interface_checks'] += 1
        validation_errors = []
        
        try:
            # Basic validation for different component categories
            if metadata.category == 'service' or metadata.category == 'shared_service':
                if component is None:
                    validation_errors.append(f'Service component is None')
            
            elif metadata.category in ['mechanism', 'observer', 'manager']:
                required_methods = ['initialize', 'process']
                for method_name in required_methods:
                    if not hasattr(component, method_name):
                        validation_errors.append(f'Component missing required method: {method_name}')
            
            elif metadata.category == 'composite':
                if not hasattr(component, 'initialize'):
                    validation_errors.append(f'Composite component missing initialize method')
            
            elif metadata.category == 'port_service':
                if component is None:
                    validation_errors.append(f'Port service component is None')
            
            # Check metadata consistency
            if hasattr(component, 'metadata'):
                if component.metadata.id != metadata.id:
                    validation_errors.append(f'Component metadata ID mismatch: {component.metadata.id} != {metadata.id}')
            
            return validation_errors
            
        except Exception as e:
            logger.error(f'Error during custom component validation: {e}')
            return [f'Custom validation error: {e}']

    async def _perform_cross_component_validation(self, context, validation_stats: dict, errors: list, warnings: list) -> None:
        """Perform cross-component validation checks."""
        
        try:
            logger.debug('Performing cross-component validation')
            
            # Validate component dependencies
            await self._validate_component_dependencies(context, validation_stats, warnings)
            
            # Validate interface compatibility
            await self._validate_interface_compatibility(context, validation_stats, warnings)
            
            # Validate epistemic consistency
            await self._validate_epistemic_consistency(context, validation_stats, warnings)
            
        except Exception as e:
            error_msg = f'Cross-component validation failed: {e}'
            logger.error(error_msg)
            if context.strict_mode:
                errors.append(error_msg)
            else:
                warnings.append(error_msg)

    async def _validate_component_dependencies(self, context, validation_stats: dict, warnings: list) -> None:
        """Validate that core service dependencies are available."""
        
        try:
            core_services = ['EventBusPort', 'IdeaService', 'ComponentRegistry']
            missing_services = []
            
            for service_name in core_services:
                try:
                    service = context.registry.get(service_name)
                    if service is None:
                        missing_services.append(service_name)
                except:
                    missing_services.append(service_name)
            
            if missing_services:
                warning_msg = f'Missing core services: {missing_services}'
                warnings.append(warning_msg)
                logger.warning(warning_msg)
                
        except Exception as e:
            logger.debug(f'Dependency validation had issues: {e}')

    async def _validate_interface_compatibility(self, context, validation_stats: dict, warnings: list) -> None:
        """Validate interface compatibility between components."""
        
        try:
            producers = {}
            consumers = {}
            
            # Only check actual components, not type-based keys
            for component_id in context.registry.list_components():
                if not self._should_validate_component(component_id, context):
                    continue
                    
                try:
                    metadata = await self._get_component_metadata_safe(component_id, None, context)
                    if not metadata:
                        continue
                    
                    # Track what signals each component produces/consumes
                    for produced_signal in metadata.produces:
                        if produced_signal not in producers:
                            producers[produced_signal] = []
                        producers[produced_signal].append(component_id)
                    
                    for accepted_signal in metadata.accepts:
                        if accepted_signal not in consumers:
                            consumers[accepted_signal] = []
                        consumers[accepted_signal].append(component_id)
                        
                except Exception as e:
                    logger.debug(f'Error checking interfaces for {component_id}: {e}')
            
            # Find orphaned consumers (signals consumed but not produced)
            orphaned_consumers = set(consumers.keys()) - set(producers.keys())
            if orphaned_consumers:
                warning_msg = f'Signals consumed but not produced: {orphaned_consumers}'
                warnings.append(warning_msg)
                logger.debug(warning_msg)
                
        except Exception as e:
            logger.debug(f'Interface compatibility validation had issues: {e}')

    async def _validate_epistemic_consistency(self, context, validation_stats: dict, warnings: list) -> None:
        """Validate epistemic role consistency across the system."""
        
        try:
            tag_distribution = {}
            
            # Only check actual components, not type-based keys
            for component_id in context.registry.list_components():
                if not self._should_validate_component(component_id, context):
                    continue
                    
                try:
                    metadata = await self._get_component_metadata_safe(component_id, None, context)
                    if not metadata:
                        continue
                    
                    for tag in metadata.epistemic_tags:
                        if tag not in tag_distribution:
                            tag_distribution[tag] = 0
                        tag_distribution[tag] += 1
                        
                except Exception as e:
                    logger.debug(f'Error checking epistemic tags for {component_id}: {e}')
            
            # Check for missing key epistemic roles
            key_roles = ['generator', 'evaluator', 'mutator', 'synthesizer']
            missing_roles = [role for role in key_roles if role not in tag_distribution]
            
            if missing_roles:
                warning_msg = f'Missing epistemic roles: {missing_roles}'
                warnings.append(warning_msg)
                logger.debug(warning_msg)
                
        except Exception as e:
            logger.debug(f'Epistemic consistency validation had issues: {e}')

    async def _update_health_reporter_with_validation_results(self, context, validation_stats: dict) -> None:
        """Update health reporter with validation results."""
        
        try:
            if hasattr(context, 'health_reporter'):
                context.health_reporter.add_phase_result(
                    phase_name='InterfaceValidationPhase',
                    status='completed' if validation_stats['validation_failed'] == 0 else 'failed',
                    message=f"Validated {validation_stats['components_validated']} components",
                    metadata=validation_stats
                )
        except Exception as e:
            logger.warning(f'Failed to update health reporter with validation results: {e}')

    async def _emit_validation_signals(self, context, validation_stats: dict) -> None:
        """Emit validation completion signals."""
        
        try:
            if hasattr(context, 'signal_emitter'):
                from bootstrap.signals.bootstrap_signals import VALIDATION_PHASE_COMPLETE
                await context.signal_emitter.emit_signal(
                    VALIDATION_PHASE_COMPLETE,
                    {
                        'validation_stats': validation_stats,
                        'phase': 'InterfaceValidationPhase',
                        'success': validation_stats['validation_failed'] == 0
                    }
                )
        except Exception as e:
            logger.warning(f'Failed to emit validation signals: {e}')

    def should_skip_phase(self, context) -> tuple[bool, str]:
        """Determine if this phase should be skipped."""
        
        # Skip if schema validation is disabled
        skip_validation = context.global_app_config.get('feature_flags', {}).get('enable_schema_validation', True) == False
        if skip_validation:
            return (True, 'Interface validation disabled in configuration')
        
        # Skip if no registry available
        if not hasattr(context, 'registry') or not context.registry:
            return (True, 'No component registry available')
        
        return (False, '')