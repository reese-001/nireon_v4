from __future__ import annotations
import logging
from typing import Any, Dict, List

from .base_phase import BootstrapPhase, PhaseResult
from bootstrap.bootstrap_helper.context_helper import build_validation_context
from application.components.base import NireonBaseComponent

logger = logging.getLogger(__name__)

class InterfaceValidationPhase(BootstrapPhase):
    """
    Interface Validation Phase - Validates component interfaces, contracts,
    and metadata consistency across all registered components
    """
    
    async def execute(self, context) -> PhaseResult:
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
            if not hasattr(context, 'interface_validator') or not context.interface_validator:
                warning_msg = 'Interface validator not available - skipping detailed validation'
                warnings.append(warning_msg)
                logger.warning(warning_msg)
                return PhaseResult.success_result(
                    message='Interface validation skipped - no validator available',
                    warnings=warnings,
                    metadata=validation_stats
                )
            
            # Get all registered components
            all_component_ids = context.registry.list_components()
            validation_stats['total_components'] = len(all_component_ids)
            
            if not all_component_ids:
                logger.info('No components registered for validation')
                return PhaseResult.success_result(
                    message='No components to validate',
                    metadata=validation_stats
                )
            
            logger.info(f'Validating {len(all_component_ids)} registered components')
            
            # Validate each component
            for component_id in all_component_ids:
                try:
                    await self._validate_single_component(
                        component_id, context, validation_stats, errors, warnings
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
            
            # Perform cross-component validations
            await self._perform_cross_component_validation(context, validation_stats, errors, warnings)
            
            # Update health reporter with validation results
            await self._update_health_reporter_with_validation_results(context, validation_stats)
            
            # Emit validation signals
            await self._emit_validation_signals(context, validation_stats)
            
            success = len(errors) == 0 or not context.strict_mode
            message = f'Interface validation complete - {validation_stats["validation_passed"]}/{validation_stats["components_validated"]} components passed validation'
            
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

    async def _validate_single_component(self, component_id: str, context, 
                                       validation_stats: dict, errors: list, warnings: list) -> None:
        """Validate a single component's interfaces and contracts"""
        logger.debug(f'Validating component: {component_id}')
        
        try:
            # Get component and metadata
            component = context.registry.get(component_id)
            metadata = context.registry.get_metadata(component_id)
            
            # Get validation data if available
            validation_data = None
            if hasattr(context, 'validation_data_store') and context.validation_data_store:
                validation_data = context.validation_data_store.get_validation_data_for_component(component_id)
            
            # Skip validation for non-NireonBaseComponent instances if they don't have validate method
            if not isinstance(component, NireonBaseComponent) and not hasattr(component, 'validate'):
                validation_stats['validation_skipped'] += 1
                logger.debug(f'Skipping validation for {component_id} - not a validatable component')
                return
            
            # Prepare validation context
            validation_context = build_validation_context(component_id, context, validation_data)
            
            # Perform interface validation using the interface validator
            if isinstance(component, NireonBaseComponent):
                validation_errors = await self._validate_nireon_component(
                    component, metadata, validation_context, validation_data, validation_stats
                )
            else:
                validation_errors = await self._validate_custom_component(
                    component, metadata, validation_context, validation_stats
                )
            
            # Process validation results
            if validation_errors:
                validation_stats['validation_failed'] += 1
                validation_stats['contract_violations'] += len(validation_errors)
                
                error_msg = f'Component {component_id} failed validation: {validation_errors}'
                if context.strict_mode:
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
            if context.strict_mode:
                errors.append(error_msg)
            else:
                warnings.append(error_msg)

    async def _validate_nireon_component(self, component: NireonBaseComponent, metadata, 
                                       validation_context, validation_data, validation_stats: dict) -> List[str]:
        """Validate a NireonBaseComponent using the interface validator"""
        validation_stats['interface_checks'] += 1
        validation_stats['metadata_checks'] += 1
        
        try:
            # Extract validation parameters
            expected_metadata = metadata
            resolved_config = {}
            manifest_spec = {}
            
            if validation_data:
                expected_metadata = validation_data.original_metadata
                resolved_config = validation_data.resolved_config
                manifest_spec = validation_data.manifest_spec
            
            # Use interface validator
            interface_validator = validation_context.registry.get_service_instance(type(validation_context.interface_validator))
            
            validation_errors = await interface_validator.validate_component(
                instance=component,
                expected_metadata=expected_metadata,
                context=validation_context,
                yaml_config_at_instantiation=resolved_config,
                actual_runtime_metadata=component.metadata,
                manifest_spec=manifest_spec
            )
            
            return validation_errors
            
        except Exception as e:
            logger.error(f'Error during NireonBaseComponent validation: {e}')
            return [f'Validation error: {e}']

    async def _validate_custom_component(self, component, metadata, validation_context, 
                                       validation_stats: dict) -> List[str]:
        """Validate a custom component that's not a NireonBaseComponent"""
        validation_stats['interface_checks'] += 1
        
        validation_errors = []
        
        try:
            # Check if component has required methods for its category
            if metadata.category == 'service':
                # Services should be callable or have specific service methods
                if not callable(component) and not hasattr(component, 'process'):
                    validation_errors.append(f'Service component should be callable or have process method')
            
            elif metadata.category in ['mechanism', 'observer', 'manager']:
                # These should ideally be NireonBaseComponent, but we can check basic interface
                required_methods = ['initialize', 'process']
                for method_name in required_methods:
                    if not hasattr(component, method_name):
                        validation_errors.append(f'Component missing required method: {method_name}')
            
            # Validate metadata consistency
            if hasattr(component, 'metadata'):
                if component.metadata.id != metadata.id:
                    validation_errors.append(f'Component metadata ID mismatch: {component.metadata.id} != {metadata.id}')
            
            return validation_errors
            
        except Exception as e:
            logger.error(f'Error during custom component validation: {e}')
            return [f'Custom validation error: {e}']

    async def _perform_cross_component_validation(self, context, validation_stats: dict, 
                                                errors: list, warnings: list) -> None:
        """Perform validations that span multiple components"""
        try:
            logger.debug('Performing cross-component validation')
            
            # Validate component dependencies
            await self._validate_component_dependencies(context, validation_stats, warnings)
            
            # Validate interface compatibility
            await self._validate_interface_compatibility(context, validation_stats, warnings)
            
            # Validate epistemic tag consistency
            await self._validate_epistemic_consistency(context, validation_stats, warnings)
            
        except Exception as e:
            error_msg = f'Cross-component validation failed: {e}'
            logger.error(error_msg)
            if context.strict_mode:
                errors.append(error_msg)
            else:
                warnings.append(error_msg)

    async def _validate_component_dependencies(self, context, validation_stats: dict, warnings: list) -> None:
        """Validate that component dependencies are satisfied"""
        try:
            # Check that core services are available
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
        """Validate interface compatibility between components"""
        try:
            # Get all components that produce and consume specific signals/data
            producers = {}
            consumers = {}
            
            for component_id in context.registry.list_components():
                try:
                    metadata = context.registry.get_metadata(component_id)
                    
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
            
            # Check for unmatched interfaces
            orphaned_consumers = set(consumers.keys()) - set(producers.keys())
            if orphaned_consumers:
                warning_msg = f'Signals consumed but not produced: {orphaned_consumers}'
                warnings.append(warning_msg)
                logger.debug(warning_msg)
                
        except Exception as e:
            logger.debug(f'Interface compatibility validation had issues: {e}')

    async def _validate_epistemic_consistency(self, context, validation_stats: dict, warnings: list) -> None:
        """Validate epistemic tag consistency and coverage"""
        try:
            tag_distribution = {}
            
            for component_id in context.registry.list_components():
                try:
                    metadata = context.registry.get_metadata(component_id)
                    for tag in metadata.epistemic_tags:
                        if tag not in tag_distribution:
                            tag_distribution[tag] = 0
                        tag_distribution[tag] += 1
                except Exception as e:
                    logger.debug(f'Error checking epistemic tags for {component_id}: {e}')
            
            # Check for adequate coverage of key epistemic roles
            key_roles = ['generator', 'evaluator', 'mutator', 'synthesizer']
            missing_roles = [role for role in key_roles if role not in tag_distribution]
            
            if missing_roles:
                warning_msg = f'Missing epistemic roles: {missing_roles}'
                warnings.append(warning_msg)
                logger.debug(warning_msg)
                
        except Exception as e:
            logger.debug(f'Epistemic consistency validation had issues: {e}')

    async def _update_health_reporter_with_validation_results(self, context, validation_stats: dict) -> None:
        """Update health reporter with validation results"""
        try:
            if hasattr(context, 'health_reporter'):
                # Add validation phase result
                context.health_reporter.add_phase_result(
                    phase_name='InterfaceValidationPhase',
                    status='completed' if validation_stats['validation_failed'] == 0 else 'failed',
                    message=f"Validated {validation_stats['components_validated']} components",
                    metadata=validation_stats
                )
        except Exception as e:
            logger.warning(f'Failed to update health reporter with validation results: {e}')

    async def _emit_validation_signals(self, context, validation_stats: dict) -> None:
        """Emit validation completion signals"""
        try:
            if hasattr(context, 'signal_emitter'):
                from signals.bootstrap_signals import VALIDATION_PHASE_COMPLETE
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
        """Check if validation phase should be skipped"""
        # Skip if validation disabled
        skip_validation = context.global_app_config.get('feature_flags', {}).get('enable_schema_validation', True) == False
        if skip_validation:
            return (True, 'Interface validation disabled in configuration')
        
        # Skip if no components to validate
        if not hasattr(context, 'registry') or not context.registry:
            return (True, 'No component registry available')
        
        return (False, '')