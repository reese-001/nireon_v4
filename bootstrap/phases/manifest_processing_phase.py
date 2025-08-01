from __future__ import annotations
import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .base_phase import BootstrapPhase, PhaseResult
from runtime.utils import detect_manifest_type, load_yaml_robust
from bootstrap.processors.manifest_processor import ManifestProcessor, ComponentSpec, ManifestProcessingResult
from bootstrap.bootstrap_helper.context_helper import create_context_builder, SimpleConfigProvider

logger = logging.getLogger(__name__)


class ManifestProcessingPhase(BootstrapPhase):

    def __init__(self):
        super().__init__()
        self.config_provider: Optional[SimpleConfigProvider] = None
        self.processing_contexts: Dict[str, Any] = {}

    async def execute(self, context) -> PhaseResult:
        logger.info('Executing Manifest Processing Phase with V2 Context Integration - Loading and processing component manifests')
        self._initialize_enhanced_config(context)
        errors = []
        warnings = []
        processing_stats = {
            'manifests_processed': 0,
            'components_discovered': 0,
            'components_instantiated': 0,
            'instantiation_failures': 0,
            'v2_integration_enabled': True,
            'processing_contexts_created': 0
        }
        try:
            manifest_paths = self._get_manifest_paths(context)
            if not manifest_paths:
                return PhaseResult.success_result(message='No manifest files found to process', metadata=processing_stats)

            processor = ManifestProcessor(strict_mode=context.strict_mode)
            processing_context = self._create_processing_context(context, 'manifest_processing', {'manifest_count': len(manifest_paths)})
            self.processing_contexts['main'] = processing_context
            processing_stats['processing_contexts_created'] += 1

            await self._process_manifests(manifest_paths, processor, context, processing_stats, errors, warnings)

            v2_validation_issues = self._validate_v2_integration()
            if v2_validation_issues:
                warnings.extend(v2_validation_issues)

            await self._emit_manifest_signals(context, processing_stats)
            
            success = len(errors) == 0 or not context.strict_mode
            message = f"Manifest processing complete with V2 integration - {processing_stats['manifests_processed']} manifests, {processing_stats['components_instantiated']}/{processing_stats['components_discovered']} components instantiated"

            return PhaseResult(success=success, message=message, errors=errors, warnings=warnings, metadata=processing_stats)

        except Exception as e:
            error_msg = f'Critical error during manifest processing: {e}'
            logger.error(error_msg, exc_info=True)
            return PhaseResult.failure_result(
                message='Manifest processing failed',
                errors=[error_msg],
                warnings=warnings,
                metadata=processing_stats
            )

    async def _process_manifests(
        self,
        manifest_paths: List[Path],
        processor: ManifestProcessor,
        context,
        stats: dict,
        errors: list,
        warnings: list
    ) -> None:
        from bootstrap.processors.component_processor import instantiate_shared_service, process_simple_component, register_orchestration_command
        
        for i, manifest_path in enumerate(manifest_paths):
            try:
                manifest_context = self._create_processing_context(context, f'manifest_{i}_{manifest_path.stem}', {'manifest_path': str(manifest_path), 'processing_index': i})
                self.processing_contexts[f'manifest_{i}'] = manifest_context
                stats['processing_contexts_created'] += 1
                
                logger.info(f'Processing manifest with V4 direct approach: {manifest_path}')
                manifest_data = load_yaml_robust(manifest_path)
                if not manifest_data:
                    warning_msg = f'Manifest file is empty or invalid: {manifest_path}'
                    warnings.append(warning_msg)
                    logger.warning(warning_msg)
                    continue

                current_env = context.global_app_config.get('environment', 'development')
                if 'environment_overrides' in manifest_data and current_env in manifest_data['environment_overrides']:
                    env_overrides = manifest_data['environment_overrides'][current_env]
                    self._apply_environment_overrides(manifest_data, env_overrides)
                
                # --- Shared Services ---
                shared_services = manifest_data.get('shared_services', {})
                for service_key, service_spec in shared_services.items():
                    if service_spec.get('enabled', True):
                        try:
                            await instantiate_shared_service(service_key, service_spec, context)
                            stats['components_instantiated'] += 1
                            logger.debug(f'✓ Instantiated shared service: {service_key}')
                        except Exception as e:
                            error_msg = f'Failed to instantiate shared service {service_key}: {e}'
                            errors.append(error_msg)
                            stats['instantiation_failures'] += 1
                            logger.error(error_msg, exc_info=True)
                
                # *** START OF FIX ***
                # Added logic to process the 'proto_engines' section
                proto_engines = manifest_data.get('proto_engines', {})
                for engine_id, engine_spec in proto_engines.items():
                    if engine_spec.get('enabled', True):
                        try:
                            # We can re-use the same instantiation logic as for shared_services
                            await instantiate_shared_service(engine_id, engine_spec, context)
                            stats['components_instantiated'] += 1
                            logger.debug(f'✓ Instantiated proto engine: {engine_id}')
                        except Exception as e:
                            error_msg = f'Failed to instantiate proto engine {engine_id}: {e}'
                            errors.append(error_msg)
                            stats['instantiation_failures'] += 1
                            logger.error(error_msg, exc_info=True)
                # *** END OF FIX ***

                # --- Composites ---
                composites = manifest_data.get('composites', {})
                for comp_id, comp_spec in composites.items():
                    if comp_spec.get('enabled', True):
                        try:
                            simple_comp_def = {
                                'component_id': comp_id,
                                'class': comp_spec.get('class'),
                                'type': 'composite',
                                'config': comp_spec.get('config', {}),
                                'config_override': comp_spec.get('config_override', {}),
                                'enabled': comp_spec.get('enabled', True),
                                'metadata_definition': comp_spec.get('metadata_definition'),
                                'epistemic_tags': comp_spec.get('epistemic_tags', []),
                            }
                            await process_simple_component(simple_comp_def, context.registry, getattr(context, 'mechanism_factory', None), context.health_reporter, context.run_id, context.global_app_config, getattr(context, 'validation_data_store', None))
                            stats['components_instantiated'] += 1
                            logger.debug(f'✓ Instantiated composite: {comp_id}')
                        except Exception as e:
                            error_msg = f'Failed to instantiate composite {comp_id}: {e}'
                            errors.append(error_msg)
                            stats['instantiation_failures'] += 1
                            logger.error(error_msg, exc_info=True)
                
                # --- Mechanisms ---
                mechanisms = manifest_data.get('mechanisms', {})
                for mech_id, mech_spec in mechanisms.items():
                    if mech_spec.get('enabled', True):
                        try:
                            simple_comp_def = {
                                'component_id': mech_id,
                                'class': mech_spec.get('class'),
                                'type': 'mechanism',
                                'config': mech_spec.get('config', {}),
                                'config_override': mech_spec.get('config_override', {}),
                                'enabled': mech_spec.get('enabled', True),
                                'metadata_definition': mech_spec.get('metadata_definition'),
                                'epistemic_tags': mech_spec.get('epistemic_tags', []),
                            }
                            await process_simple_component(simple_comp_def, context.registry, getattr(context, 'mechanism_factory', None), context.health_reporter, context.run_id, context.global_app_config, getattr(context, 'validation_data_store', None))
                            stats['components_instantiated'] += 1
                            logger.debug(f'✓ Instantiated mechanism: {mech_id}')
                        except Exception as e:
                            error_msg = f'Failed to instantiate mechanism {mech_id}: {e}'
                            errors.append(error_msg)
                            stats['instantiation_failures'] += 1
                            logger.error(error_msg, exc_info=True)
                
                # --- Planners ---
                planners = manifest_data.get('planners', {})
                for planner_id, planner_spec in planners.items():
                    if planner_spec.get('enabled', True):
                        try:
                            # Planners can be treated like other simple components
                            simple_comp_def = {
                                'component_id': planner_id,
                                'class': planner_spec.get('class'),
                                'type': 'planner', # Important for metadata
                                'config': planner_spec.get('config', {}),
                                'config_override': planner_spec.get('config_override', {}),
                                'enabled': planner_spec.get('enabled', True),
                                'metadata_definition': planner_spec.get('metadata_definition'),
                                'epistemic_tags': planner_spec.get('epistemic_tags', [])
                            }
                            await process_simple_component(simple_comp_def, context.registry, getattr(context, 'mechanism_factory', None), context.health_reporter, context.run_id, context.global_app_config, getattr(context, 'validation_data_store', None))
                            stats['components_instantiated'] += 1
                            logger.debug(f'✓ Instantiated planner: {planner_id}')
                        except Exception as e:
                            error_msg = f'Failed to instantiate planner {planner_id}: {e}'
                            errors.append(error_msg)
                            stats['instantiation_failures'] += 1
                            logger.error(error_msg, exc_info=True)
                
                # --- Observers ---
                observers = manifest_data.get('observers', {})
                for obs_id, obs_spec in observers.items():
                    if obs_spec.get('enabled', True):
                        try:
                            # Create a definition that process_simple_component can understand
                            simple_comp_def = {
                                'component_id': obs_id,
                                'class': obs_spec.get('class'),
                                'type': 'observer', # This sets the category
                                'config': obs_spec.get('config', {}),
                                'config_override': obs_spec.get('config_override', {}),
                                'enabled': obs_spec.get('enabled', True),
                                'metadata_definition': obs_spec.get('metadata_definition'),
                                'epistemic_tags': obs_spec.get('epistemic_tags', [])
                            }
                            # Use the existing helper function to process it
                            await process_simple_component(
                                simple_comp_def,
                                context.registry,
                                getattr(context, 'mechanism_factory', None),
                                context.health_reporter,
                                context.run_id,
                                context.global_app_config,
                                getattr(context, 'validation_data_store', None)
                            )
                            stats['components_instantiated'] += 1
                            logger.debug(f'✓ Instantiated observer: {obs_id}')
                        except Exception as e:
                            error_msg = f'Failed to instantiate observer {obs_id}: {e}'
                            errors.append(error_msg)
                            stats['instantiation_failures'] += 1
                            logger.error(error_msg, exc_info=True)
                
                # --- Orchestration Commands ---
                orch_commands = manifest_data.get('orchestration_commands', {})
                for cmd_id, cmd_spec in orch_commands.items():
                    if cmd_spec.get('enabled', True):
                        try:
                            await register_orchestration_command(cmd_id, cmd_spec, context.registry, context.health_reporter, context.global_app_config)
                            stats['components_instantiated'] += 1
                            logger.debug(f'✓ Registered orchestration command: {cmd_id}')
                        except Exception as e:
                            error_msg = f'Failed to register orchestration command {cmd_id}: {e}'
                            errors.append(error_msg)
                            stats['instantiation_failures'] += 1
                            logger.error(error_msg, exc_info=True)
                
                stats['manifests_processed'] += 1
                stats['components_discovered'] += (
                    len(shared_services) + 
                    len(proto_engines) + 
                    len(composites) + 
                    len(mechanisms) + 
                    len(planners) + 
                    len(observers) + 
                    len(manifest_data.get('managers', {})) + 
                    len(orch_commands)
                )

            except Exception as e:
                error_msg = f'Exception processing manifest {manifest_path}: {e}'
                errors.append(error_msg)
                logger.error(error_msg, exc_info=True)

    def _apply_environment_overrides(self, manifest_data: Dict[str, Any], env_overrides: Dict[str, Any]) -> None:
        for section_name, section_overrides in env_overrides.items():
            if section_name in manifest_data:
                if isinstance(manifest_data[section_name], dict) and isinstance(section_overrides, dict):
                    for key, override_value in section_overrides.items():
                        if key in manifest_data[section_name]:
                            if isinstance(manifest_data[section_name][key], dict) and isinstance(override_value, dict):
                                if 'config_override' in override_value:
                                    if 'config_override' not in manifest_data[section_name][key]:
                                        manifest_data[section_name][key]['config_override'] = {}
                                    manifest_data[section_name][key]['config_override'].update(override_value['config_override'])
                                    override_value_copy = override_value.copy()
                                    del override_value_copy['config_override']
                                    manifest_data[section_name][key].update(override_value_copy)
                                else:
                                    manifest_data[section_name][key].update(override_value)
                            else:
                                manifest_data[section_name][key] = override_value
                        else:
                            manifest_data[section_name][key] = override_value
                else:
                    manifest_data[section_name] = section_overrides
    
    def _initialize_enhanced_config(self, context) -> None:
        try:
            manifest_config = context.global_app_config.get('manifest_processing', {})
            component_config = context.global_app_config.get('component_instantiation', {})
            enhanced_config = {
                **{f'manifest.{k}': v for k, v in manifest_config.items()},
                **{f'component.{k}': v for k, v in component_config.items()}
            }
            self.config_provider = SimpleConfigProvider(enhanced_config)
            logger.debug(f'Manifest processing enhanced config provider initialized with {len(enhanced_config)} configuration entries')
        except Exception as e:
            logger.warning(f'Failed to initialize manifest processing enhanced config provider: {e}')
            self.config_provider = None

    def _get_manifest_paths(self, context) -> List[Path]:
        manifest_paths = []
        if self.config_provider:
            additional_paths = self.config_provider.get_config('manifest', 'additional_paths', [])
            for path_str in additional_paths:
                additional_path = Path(path_str)
                if not additional_path.is_absolute():
                    package_root = Path(__file__).resolve().parents[2]
                    additional_path = package_root / additional_path
                if additional_path.exists() and additional_path.suffix in ['.yaml', '.yml']:
                    manifest_paths.append(additional_path)

        if hasattr(context.config, 'config_paths'):
            for path in context.config.config_paths:
                if path.exists() and path.suffix in ['.yaml', '.yml']:
                    manifest_paths.append(path)

        if manifest_paths:
            logger.info(f'Found {len(manifest_paths)} manifest files to process with V2 integration')
            for path in manifest_paths:
                logger.debug(f'  - {path}')
        else:
            logger.warning('No manifest files found in configuration paths')
        return manifest_paths

    def _create_processing_context(self, base_context, context_id: str, metadata: Dict[str, Any]):
        try:
            context_builder = create_context_builder(component_id=f'manifest_{context_id}', run_id=f'{base_context.run_id}_manifest')
            if hasattr(base_context, 'registry'):
                context_builder.with_registry(base_context.registry)
            if hasattr(base_context, 'event_bus'):
                context_builder.with_event_bus(base_context.event_bus)
            
            enhanced_metadata = {
                **metadata,
                'manifest_phase': True,
                'v2_context': True,
                'config_provider_available': self.config_provider is not None
            }
            context_builder.with_metadata(**enhanced_metadata)
            
            if hasattr(base_context, 'feature_flags'):
                context_builder.with_feature_flags(base_context.feature_flags)
            return context_builder.build()
        except Exception as e:
            logger.warning(f'Failed to create V2 processing context for {context_id}: {e}')
            return None

    def _validate_v2_integration(self) -> List[str]:
        issues = []
        try:
            if self.config_provider is None:
                issues.append('V2 config provider not initialized for manifest processing')
            expected_contexts = ['main']
            for expected in expected_contexts:
                if expected not in self.processing_contexts:
                    issues.append(f'Missing V2 processing context for: {expected}')
            
            for context_name, processing_context in self.processing_contexts.items():
                if processing_context is None:
                    issues.append(f'V2 processing context for {context_name} is None')
                elif not hasattr(processing_context, 'metadata'):
                    issues.append(f'V2 processing context for {context_name} missing metadata')

            if issues:
                logger.warning(f'V2 integration validation for manifest processing found {len(issues)} issues')
            else:
                logger.debug('V2 integration validation for manifest processing passed')

        except Exception as e:
            issues.append(f'V2 integration validation failed for manifest processing: {e}')
        return issues

    async def _emit_manifest_signals(self, context, stats: dict) -> None:
        try:
            if hasattr(context, 'signal_emitter'):
                from bootstrap.signals.bootstrap_signals import MANIFEST_PROCESSING_COMPLETE
                enhanced_stats = {
                    **stats,
                    'v2_context_integration': True,
                    'config_provider_enabled': self.config_provider is not None,
                    'processing_contexts_created': len(self.processing_contexts),
                    'phase': 'ManifestProcessingPhase'
                }
                await context.signal_emitter.emit_signal(MANIFEST_PROCESSING_COMPLETE, enhanced_stats)
                logger.debug('Emitted manifest processing signal with V2 integration metadata')
        except Exception as e:
            logger.warning(f'Failed to emit manifest processing signals: {e}')
    
    def should_skip_phase(self, context) -> tuple[bool, str]:
        if not hasattr(context.config, 'config_paths') or not context.config.config_paths:
            return (True, 'No manifest configuration paths provided')
        
        if self.config_provider:
            skip_manifest = self.config_provider.get_config('manifest', 'skip_manifest_processing', False)
        else:
            skip_manifest = context.global_app_config.get('skip_manifest_processing', False)
        
        if skip_manifest:
            return (True, 'Manifest processing disabled in configuration')
        return (False, '')