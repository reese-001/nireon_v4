# nireon_v4/bootstrap/phases/abiogenesis_phase.py
from __future__ import annotations
import logging
import importlib
from typing import List, Dict, Any
from pathlib import Path

from .base_phase import BootstrapPhase, PhaseResult
from runtime.utils import load_yaml_robust, import_by_path

logger = logging.getLogger(__name__)

class AbiogenesisPhase(BootstrapPhase):
    """
    L0 Abiogenesis Phase - Preloads critical shared services marked with preload: true
    This phase runs before all others to ensure essential services are available.
    """
    
    PHASE_NAME = 'AbiogenesisPhase'
    EXECUTION_ORDER = -10

    def __init__(self):
        super().__init__()
        self.preloaded_services: List[str] = []

    async def execute(self, context) -> PhaseResult:
        logger.info('Executing L0 Abiogenesis Phase - Preloading critical shared services')
        
        errors = []
        warnings = []
        preload_stats = {
            'manifests_processed': 0,
            'services_found': 0,
            'services_preloaded': 0,
            'preload_failures': 0,
            'l0_abiogenesis': True
        }

        try:
            # Load manifest files to find services with preload: true
            manifest_paths = self._get_manifest_paths(context)
            if not manifest_paths:
                logger.warning('No manifest paths found for preloading services')
                return PhaseResult.success_result(
                    message='L0 Abiogenesis completed - no manifests found',
                    metadata=preload_stats
                )

            # Process each manifest for preload services
            for manifest_path in manifest_paths:
                try:
                    await self._process_manifest_for_preload(
                        manifest_path, context, preload_stats, errors, warnings
                    )
                    preload_stats['manifests_processed'] += 1
                except Exception as e:
                    error_msg = f'Failed to process manifest {manifest_path} for preloading: {e}'
                    errors.append(error_msg)
                    logger.error(error_msg, exc_info=True)

            # Report to health system
            if hasattr(context, 'health_reporter') and context.health_reporter:
                try:
                    context.health_reporter.add_phase_result(
                        phase_name=self.PHASE_NAME,
                        status='completed',
                        message=f'Preloaded {preload_stats["services_preloaded"]} critical services',
                        errors=errors,
                        warnings=warnings,
                        metadata=preload_stats
                    )
                except Exception as e:
                    logger.warning(f'Failed to update health reporter: {e}')

            # After preloading all services, create IdeaService if we have the dependencies
            await self._create_idea_service_if_possible(context, preload_stats)

            # Determine success
            success = len(errors) == 0 or not context.strict_mode
            message = f"L0 Abiogenesis complete - {preload_stats['services_preloaded']}/{preload_stats['services_found']} services preloaded"
            
            if preload_stats['services_preloaded'] > 0:
                logger.info(f"✓ {message}. Preloaded services: {', '.join(self.preloaded_services)}")
            else:
                logger.warning("⚠️ No services were preloaded - this may cause later phases to fail")

            return PhaseResult(
                success=success,
                message=message,
                errors=errors,
                warnings=warnings,
                metadata=preload_stats
            )

        except Exception as e:
            error_msg = f'Critical error during L0 Abiogenesis: {e}'
            logger.error(error_msg, exc_info=True)
            return PhaseResult.failure_result(
                message='L0 Abiogenesis failed',
                errors=[error_msg],
                warnings=warnings,
                metadata=preload_stats
            )

    async def _process_manifest_for_preload(self, manifest_path: Path, context, stats: dict, errors: list, warnings: list):
        """Process a single manifest file for services with preload: true"""
        try:
            logger.debug(f'Processing manifest for preload services: {manifest_path}')
            manifest_data = load_yaml_robust(manifest_path)
            
            if not manifest_data or not isinstance(manifest_data, dict):
                warnings.append(f'Manifest {manifest_path} is empty or invalid')
                return

            # Apply environment overrides if present
            current_env = context.global_app_config.get('environment', 'development')
            if 'environment_overrides' in manifest_data and current_env in manifest_data['environment_overrides']:
                self._apply_environment_overrides(manifest_data, manifest_data['environment_overrides'][current_env])

            # Process shared_services section
            shared_services = manifest_data.get('shared_services', {})
            for service_id, service_spec in shared_services.items():
                if not isinstance(service_spec, dict):
                    continue
                    
                # Check if this service should be preloaded
                if service_spec.get('enabled', True) and service_spec.get('preload', False):
                    stats['services_found'] += 1
                    logger.info(f'Found preload service: {service_id}')
                    
                    try:
                        # Check if already loaded
                        try:
                            existing = context.registry.get(service_id)
                            if existing is not None:
                                logger.debug(f'Service {service_id} already exists in registry, skipping preload')
                                continue
                        except Exception:
                            # Service doesn't exist, proceed with loading
                            pass

                        # Instantiate the service
                        instance = await self._instantiate_preload_service(service_id, service_spec, context)
                        
                        # Register in registry by service ID (using a simple approach)
                        # Since these are shared services, we'll add them to a simple dict if registry has one
                        if hasattr(context.registry, '_components'):
                            context.registry._components[service_id] = instance
                        
                        # Register by port type if specified
                        port_type = service_spec.get('port_type')
                        if port_type:
                            try:
                                module_path, class_name = port_type.split(':')
                                port_module = importlib.import_module(module_path)
                                port_interface = getattr(port_module, class_name)
                                context.registry.register_service_instance(port_interface, instance)
                                logger.debug(f'Registered {service_id} as {port_interface.__name__}')
                            except Exception as e:
                                logger.warning(f'Failed to register {service_id} by port type {port_type}: {e}')

                        stats['services_preloaded'] += 1
                        self.preloaded_services.append(service_id)
                        logger.info(f"✓ [L0 Abiogenesis] Preloaded critical service '{service_id}'")
                        
                    except Exception as e:
                        stats['preload_failures'] += 1
                        error_msg = f'Failed to preload service {service_id}: {e}'
                        if context.strict_mode:
                            errors.append(error_msg)
                        else:
                            warnings.append(error_msg)
                        logger.error(error_msg, exc_info=True)

        except Exception as e:
            error_msg = f'Error processing manifest {manifest_path}: {e}'
            errors.append(error_msg)
            logger.error(error_msg, exc_info=True)

    async def _instantiate_preload_service(self, service_id: str, service_spec: dict, context) -> Any:
        """Instantiate a preload service from its specification"""
        class_path = service_spec.get('class')
        if not class_path:
            raise ValueError(f'Service {service_id} missing class specification')

        try:
            # Import the service class
            service_class = import_by_path(class_path)
            
            # Get service configuration
            service_config = service_spec.get('config', {})
            
            # Try different instantiation patterns
            try:
                # Try with config parameter
                return service_class(config=service_config)
            except TypeError:
                try:
                    # Try with cfg parameter
                    return service_class(cfg=service_config)
                except TypeError:
                    try:
                        # Try with unpacked config
                        return service_class(**service_config)
                    except TypeError:
                        try:
                            # Try with no parameters
                            return service_class()
                        except TypeError:
                            # Last resort: just pass config
                            return service_class(service_config)
                            
        except Exception as e:
            logger.error(f'Failed to instantiate {service_id} from {class_path}: {e}')
            raise

    def _get_manifest_paths(self, context) -> List[Path]:
        """Get manifest file paths from context configuration"""
        manifest_paths = []
        
        if hasattr(context.config, 'config_paths'):
            for path in context.config.config_paths:
                if path.exists() and path.suffix in ['.yaml', '.yml']:
                    manifest_paths.append(path)
        
        return manifest_paths

    def _apply_environment_overrides(self, manifest_data: Dict[str, Any], env_overrides: Dict[str, Any]) -> None:
        """Apply environment-specific overrides to manifest data"""
        for section_name, section_overrides in env_overrides.items():
            if section_name in manifest_data:
                if isinstance(manifest_data[section_name], dict) and isinstance(section_overrides, dict):
                    for key, override_value in section_overrides.items():
                        if key in manifest_data[section_name]:
                            if isinstance(manifest_data[section_name][key], dict) and isinstance(override_value, dict):
                                # Handle config_override specially
                                if 'config_override' in override_value:
                                    if 'config_override' not in manifest_data[section_name][key]:
                                        manifest_data[section_name][key]['config_override'] = {}
                                    manifest_data[section_name][key]['config_override'].update(override_value['config_override'])
                                    # Apply other overrides
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

    def should_skip_phase(self, context) -> tuple[bool, str]:
        """Check if this phase should be skipped"""
        # Never skip L0 Abiogenesis - it's critical for system bootstrap
        return (False, '')

    async def _create_idea_service_if_possible(self, context, preload_stats: dict) -> None:
        try:
            from domain.ports.idea_repository_port import IdeaRepositoryPort
            from domain.ports.event_bus_port import EventBusPort
            from application.services.idea_service import IdeaService
            idea_repo = None
            event_bus = None
            try:
                # This will now succeed because IdeaRepositoryPort is preloaded from the manifest
                idea_repo = context.registry.get_service_instance(IdeaRepositoryPort)
                # This will now succeed because EventBusPort is preloaded from the manifest
                event_bus = context.registry.get_service_instance(EventBusPort)
            except Exception as e:
                logger.debug(f'Dependencies not available for IdeaService creation: {e}')
                return
            if idea_repo and event_bus:
                # This is where the service is correctly instantiated with its dependencies
                idea_service = IdeaService(repository=idea_repo, event_bus=event_bus)
                context.registry.register_service_instance(IdeaService, idea_service)
                
                # Also try to register as IdeaServicePort if it exists
                try:
                    from domain.ports.idea_service_port import IdeaServicePort
                    context.registry.register_service_instance(IdeaServicePort, idea_service)
                    logger.info("✓ [L0 Abiogenesis] Created IdeaService and registered as IdeaServicePort")
                    self.preloaded_services.append('IdeaService')
                    preload_stats['services_preloaded'] += 1
                except ImportError:
                    # IdeaServicePort might not exist, just register as IdeaService
                    logger.info("✓ [L0 Abiogenesis] Created IdeaService (IdeaServicePort interface not found)")
                    self.preloaded_services.append('IdeaService')
                    preload_stats['services_preloaded'] += 1
            else:
                logger.debug('Missing dependencies for IdeaService creation')
                
        except Exception as e:
            logger.warning(f'Failed to create IdeaService during abiogenesis: {e}')