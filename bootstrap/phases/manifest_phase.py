# C:\Users\erees\Documents\development\nireon_v4\bootstrap\phases\manifest_phase.py
from __future__ import annotations
import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .base_phase import BootstrapPhase, PhaseResult
from runtime.utils import detect_manifest_type, load_yaml_robust
from bootstrap.processors.manifest_processor import ManifestProcessor, ComponentSpec, ManifestProcessingResult
from bootstrap.processors.component_processor import ComponentInstantiator

logger = logging.getLogger(__name__)


class ManifestProcessingPhase(BootstrapPhase):
    async def execute(self, context) -> PhaseResult:
        logger.info('Executing Manifest Processing Phase - Loading and processing component manifests')
        errors = []
        warnings = []
        processing_stats = {
            'manifests_processed': 0,
            'components_discovered': 0,
            'components_instantiated': 0,
            'instantiation_failures': 0
        }

        try:
            manifest_paths = self._get_manifest_paths(context)
            if not manifest_paths:
                return PhaseResult.success_result(
                    message='No manifest files found to process',
                    metadata=processing_stats
                )

            # Use the dedicated ManifestProcessor from processors module
            processor = ManifestProcessor(strict_mode=context.strict_mode)
            
            instantiator = ComponentInstantiator(
                mechanism_factory=getattr(context, 'mechanism_factory', None),
                interface_validator=getattr(context, 'interface_validator', None),
                registry_manager=context.registry_manager,
                global_app_config=context.global_app_config
            )

            # Process all manifests
            all_component_specs = []
            for manifest_path in manifest_paths:
                try:
                    manifest_result = await self._process_single_manifest(
                        manifest_path, processor, context, processing_stats, errors, warnings
                    )
                    if manifest_result:
                        all_component_specs.extend(manifest_result.components)
                except Exception as e:
                    error_msg = f'Failed to process manifest {manifest_path}: {e}'
                    errors.append(error_msg)
                    logger.error(error_msg, exc_info=True)

            processing_stats['components_discovered'] = len(all_component_specs)

            # Instantiate components
            if all_component_specs:
                await self._instantiate_components(
                    all_component_specs, instantiator, context, processing_stats, errors, warnings
                )

            await self._emit_manifest_signals(context, processing_stats)

            success = len(errors) == 0 or not context.strict_mode
            message = f"Manifest processing complete - {processing_stats['manifests_processed']} manifests, {processing_stats['components_instantiated']}/{processing_stats['components_discovered']} components instantiated"

            return PhaseResult(
                success=success,
                message=message,
                errors=errors,
                warnings=warnings,
                metadata=processing_stats
            )

        except Exception as e:
            error_msg = f'Critical error during manifest processing: {e}'
            logger.error(error_msg, exc_info=True)
            return PhaseResult.failure_result(
                message='Manifest processing failed',
                errors=[error_msg],
                warnings=warnings,
                metadata=processing_stats
            )

    def _get_manifest_paths(self, context) -> List[Path]:
        manifest_paths = []
        if hasattr(context.config, 'config_paths'):
            for path in context.config.config_paths:
                if path.exists() and path.suffix in ['.yaml', '.yml']:
                    manifest_paths.append(path)

        if manifest_paths:
            logger.info(f'Found {len(manifest_paths)} manifest files to process')
            for path in manifest_paths:
                logger.debug(f'  - {path}')
        else:
            logger.warning('No manifest files found in configuration paths')

        return manifest_paths

    async def _process_single_manifest(
        self, manifest_path: Path, processor: ManifestProcessor, context,
        stats: dict, errors: list, warnings: list
    ) -> Optional[ManifestProcessingResult]:
        logger.info(f'Processing manifest: {manifest_path}')
        
        try:
            manifest_data = load_yaml_robust(manifest_path)
            if not manifest_data:
                warning_msg = f'Manifest file is empty or invalid: {manifest_path}'
                warnings.append(warning_msg)
                logger.warning(warning_msg)
                return None

            result = await processor.process_manifest(manifest_path, manifest_data)
            stats['manifests_processed'] += 1

            if result.errors:
                errors.extend(result.errors)
                logger.error(f'Manifest processing errors for {manifest_path}: {result.errors}')

            if result.warnings:
                warnings.extend(result.warnings)
                logger.warning(f'Manifest processing warnings for {manifest_path}: {result.warnings}')

            if result.success:
                logger.info(f'✓ Successfully processed manifest {manifest_path}: {result.component_count} components')
            else:
                logger.error(f'✗ Failed to process manifest {manifest_path}')

            return result

        except Exception as e:
            error_msg = f'Exception processing manifest {manifest_path}: {e}'
            errors.append(error_msg)
            logger.error(error_msg, exc_info=True)
            return None

    async def _instantiate_components(
        self, component_specs: List[ComponentSpec], instantiator: ComponentInstantiator,
        context, stats: dict, errors: list, warnings: list
    ) -> None:
        logger.info(f'Instantiating {len(component_specs)} components from manifests')

        # Group components by type for ordered processing
        grouped_specs = self._group_components_by_type(component_specs)

        # Process in dependency order
        processing_order = ['shared_service', 'mechanism', 'observer', 'manager', 'composite', 'orchestration_command']
        
        for component_type in processing_order:
            if component_type not in grouped_specs:
                continue

            type_specs = grouped_specs[component_type]
            logger.info(f'Processing {len(type_specs)} {component_type} components')

            for spec in type_specs:
                try:
                    result = await instantiator.instantiate_component(spec, context)
                    
                    if result.success:
                        stats['components_instantiated'] += 1
                        logger.debug(f'✓ Instantiated {spec.component_id} ({component_type})')
                        
                        # Update health reporter if available
                        if hasattr(context, 'health_reporter') and result.component:
                            try:
                                from bootstrap.health.reporter import ComponentStatus  # ← CORRECT IMPORT
                                metadata = result.component.metadata if hasattr(result.component, 'metadata') else None
                                if metadata:
                                    context.health_reporter.add_component_status(
                                        spec.component_id,
                                        ComponentStatus.INSTANCE_REGISTERED,  # ← Use compatible enum value
                                        metadata,
                                        result.warnings
                                    )
                            except (ImportError, AttributeError) as e:
                                logger.debug(f'Could not update health reporter for {spec.component_id}: {e}')
                                logger.info(f'Component {spec.component_id} instantiated successfully')
                    else:
                        stats['instantiation_failures'] += 1
                        error_msg = f'Failed to instantiate {spec.component_id}: {result.errors}'
                        if context.strict_mode:
                            errors.extend(result.errors)
                        else:
                            warnings.extend(result.errors)
                        logger.error(error_msg)

                except Exception as e:
                    stats['instantiation_failures'] += 1
                    error_msg = f'Exception instantiating {spec.component_id}: {e}'
                    errors.append(error_msg)
                    logger.error(error_msg, exc_info=True)

        logger.info(f"Component instantiation complete: {stats['components_instantiated']} successful, {stats['instantiation_failures']} failed")

    def _group_components_by_type(self, component_specs: List[ComponentSpec]) -> Dict[str, List[ComponentSpec]]:
        grouped = {}
        for spec in component_specs:
            component_type = spec.component_type
            if component_type not in grouped:
                grouped[component_type] = []
            grouped[component_type].append(spec)
        return grouped

    async def _emit_manifest_signals(self, context, stats: dict) -> None:
        try:
            if hasattr(context, 'signal_emitter'):
                from bootstrap.signals.bootstrap_signals import MANIFEST_PROCESSING_COMPLETE
                await context.signal_emitter.emit_signal(
                    MANIFEST_PROCESSING_COMPLETE,
                    {
                        'manifests_processed': stats['manifests_processed'],
                        'components_discovered': stats['components_discovered'],
                        'components_instantiated': stats['components_instantiated'],
                        'instantiation_failures': stats['instantiation_failures'],
                        'phase': 'ManifestProcessingPhase'
                    }
                )
        except Exception as e:
            logger.warning(f'Failed to emit manifest processing signals: {e}')

    def should_skip_phase(self, context) -> tuple[bool, str]:
        if not hasattr(context.config, 'config_paths') or not context.config.config_paths:
            return (True, 'No manifest configuration paths provided')
        
        if context.global_app_config.get('skip_manifest_processing', False):
            return (True, 'Manifest processing disabled in configuration')
        
        return (False, '')