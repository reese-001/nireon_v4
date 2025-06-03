"""
Manifest Processing Phase - Load and process component manifests.

Processes all manifest files, validates schemas, and instantiates components
according to their specifications. This is where the bulk of component
creation happens during bootstrap.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

from .base_phase import BootstrapPhase, PhaseResult
from bootstrap.processors.manifest_processor import ManifestProcessor, ComponentSpec
from bootstrap.processors.component_instantiator import ComponentInstantiator
from bootstrap.bootstrap_helper.utils import load_yaml_robust

logger = logging.getLogger(__name__)

class ManifestProcessingPhase(BootstrapPhase):
    """
    Manifest Processing Phase - Component instantiation from manifests.
    
    Loads manifest files, validates schemas, extracts component specifications,
    and instantiates components using the factory system. This is the primary
    phase where user-defined components are brought into existence.
    
    Responsibilities:
    - Load and validate manifest files against schemas
    - Extract component specifications from manifests
    - Instantiate components using appropriate factories
    - Register components with certification
    - Handle both enhanced and simple manifest formats
    - Collect and report processing statistics
    """
    
    async def execute(self, context) -> PhaseResult:
        """
        Execute manifest processing and component instantiation.
        
        Loads all manifests from config paths and creates components
        according to their specifications.
        """
        logger.info("Processing manifests and instantiating components")
        
        errors = []
        warnings = []
        processing_stats = {
            'manifests_processed': 0,
            'components_instantiated': 0,
            'components_registered': 0,
            'validation_errors': 0,
            'schema_validation_errors': 0
        }
        
        try:
            # Load manifest files
            manifest_files = await self._load_manifest_files(context, errors)
            
            if not manifest_files:
                if context.config.config_paths:
                    error_msg = "No valid manifest files found despite paths being provided"
                    if context.strict_mode:
                        errors.append(error_msg)
                        return PhaseResult.failure_result(
                            message="No manifests to process",
                            errors=errors,
                            metadata=processing_stats
                        )
                    else:
                        warnings.append(error_msg)
                
                return PhaseResult.success_result(
                    message="No manifests to process",
                    warnings=warnings,
                    metadata=processing_stats
                )
            
            # Create manifest processor with schema validation
            manifest_processor = ManifestProcessor(strict_mode=context.strict_mode)
            
            # Create component instantiator
            component_instantiator = ComponentInstantiator(
                context.mechanism_factory,
                context.interface_validator,
                context.registry_manager,
                context.global_app_config
            )
            
            # Process each manifest file
            all_components = []
            for manifest_path, manifest_data in manifest_files:
                try:
                    result = await self._process_single_manifest(
                        manifest_processor,
                        component_instantiator,
                        manifest_path,
                        manifest_data,
                        context,
                        processing_stats
                    )
                    
                    all_components.extend(result.components)
                    errors.extend(result.errors)
                    warnings.extend(result.warnings)
                    
                except Exception as e:
                    error_msg = f"Critical error processing manifest {manifest_path}: {e}"
                    errors.append(error_msg)
                    logger.error(error_msg, exc_info=True)
            
            # Emit manifest processing complete signal
            await self._emit_manifest_processing_signal(context, processing_stats, all_components)
            
            success = len(errors) == 0 or not context.strict_mode
            message = f"Manifest processing complete - {processing_stats['components_instantiated']} components from {processing_stats['manifests_processed']} manifests"
            
            return PhaseResult(
                success=success,
                message=message,
                errors=errors,
                warnings=warnings,
                metadata=processing_stats
            )
            
        except Exception as e:
            error_msg = f"Critical error during manifest processing: {e}"
            logger.error(error_msg, exc_info=True)
            return PhaseResult.failure_result(
                message="Manifest processing failed",
                errors=[error_msg],
                warnings=warnings,
                metadata=processing_stats
            )
    
    async def _load_manifest_files(self, context, errors: list) -> List[tuple[Path, Dict[str, Any]]]:
        """Load manifest files from configured paths."""
        manifest_files = []
        
        for config_path in context.config.config_paths:
            path = Path(config_path)
            
            if not path.exists():
                error_msg = f"Manifest file not found: {path}"
                errors.append(error_msg)
                logger.warning(error_msg)
                continue
            
            try:
                manifest_data = load_yaml_robust(path)
                if manifest_data:
                    manifest_files.append((path, manifest_data))
                    logger.info(f"✓ Loaded manifest: {path}")
                else:
                    warning_msg = f"Manifest file is empty or invalid: {path}"
                    logger.warning(warning_msg)
                    
            except Exception as e:
                error_msg = f"Failed to load manifest {path}: {e}"
                errors.append(error_msg)
                logger.error(error_msg)
        
        return manifest_files
    
    async def _process_single_manifest(
        self,
        manifest_processor: ManifestProcessor,
        component_instantiator: 'ComponentInstantiator',
        manifest_path: Path,
        manifest_data: Dict[str, Any],
        context,
        processing_stats: Dict[str, int]
    ) -> 'SingleManifestResult':
        """Process a single manifest file."""
        logger.info(f"Processing manifest: {manifest_path}")
        
        # Process manifest and extract components
        result = await manifest_processor.process_manifest(manifest_path, manifest_data)
        processing_stats['manifests_processed'] += 1
        
        if not result.success:
            processing_stats['schema_validation_errors'] += len(result.errors)
            logger.error(f"Manifest processing failed for {manifest_path}")
            return SingleManifestResult(
                components=[],
                errors=result.errors,
                warnings=result.warnings
            )
        
        # Instantiate components from specifications
        instantiated_components = []
        instantiation_errors = []
        instantiation_warnings = []
        
        for component_spec in result.components:
            try:
                component_result = await component_instantiator.instantiate_component(
                    component_spec, context
                )
                
                if component_result.success:
                    instantiated_components.append(component_result.component)
                    processing_stats['components_instantiated'] += 1
                    
                    # Register component if not already registered by instantiator
                    if not component_result.already_registered:
                        processing_stats['components_registered'] += 1
                else:
                    instantiation_errors.extend(component_result.errors)
                    processing_stats['validation_errors'] += len(component_result.errors)
                    
                instantiation_warnings.extend(component_result.warnings)
                
            except Exception as e:
                error_msg = f"Failed to instantiate component {component_spec.component_id}: {e}"
                instantiation_errors.append(error_msg)
                logger.error(error_msg, exc_info=True)
        
        logger.info(f"✓ Manifest {manifest_path.name}: {len(instantiated_components)} components instantiated")
        
        return SingleManifestResult(
            components=instantiated_components,
            errors=result.errors + instantiation_errors,
            warnings=result.warnings + instantiation_warnings
        )
    
    async def _emit_manifest_processing_signal(
        self, 
        context, 
        processing_stats: Dict[str, int],
        all_components: List[Any]
    ) -> None:
        """Emit manifest processing completion signal."""
        try:
            from signals.bootstrap_signals import MANIFEST_PROCESSING_COMPLETE
            
            component_types = {}
            for component in all_components:
                if hasattr(component, 'metadata') and hasattr(component.metadata, 'category'):
                    category = component.metadata.category
                    component_types[category] = component_types.get(category, 0) + 1
            
            await context.signal_emitter.emit_signal(
                signal_type=MANIFEST_PROCESSING_COMPLETE,
                payload={
                    'processing_stats': processing_stats,
                    'component_types': component_types,
                    'total_components': len(all_components),
                    'run_id': context.run_id
                }
            )
            
        except Exception as e:
            # Non-critical - don't fail the phase
            logger.warning(f"Failed to emit manifest processing signal: {e}")
    
    def should_skip_phase(self, context) -> tuple[bool, str]:
        """Skip if no config paths provided or explicitly disabled."""
        if not context.config.config_paths:
            return True, "No manifest paths provided"
        
        skip_manifests = context.global_app_config.get('skip_manifest_processing', False)
        if skip_manifests:
            return True, "Manifest processing disabled in configuration"
        
        return False, ""

class SingleManifestResult:
    """Result of processing a single manifest."""
    
    def __init__(self, components: List[Any], errors: List[str], warnings: List[str]):
        self.components = components
        self.errors = errors
        self.warnings = warnings