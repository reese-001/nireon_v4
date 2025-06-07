from __future__ import annotations
import asyncio
import inspect
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type, Union
from pathlib import Path
from core.base_component import NireonBaseComponent
from core.lifecycle import ComponentMetadata, ComponentRegistryMissingError
from bootstrap.processors.manifest_processor import ComponentSpec
from bootstrap.exceptions import ComponentInstantiationError, ConfigurationError, BootstrapError
from runtime.utils import import_by_path, extract_class_name
from bootstrap.bootstrap_helper.metadata import get_default_metadata
from configs.config_utils import ConfigMerger

logger = logging.getLogger(__name__)


@dataclass
class ComponentInstantiationResult:
    success: bool
    component: Optional[Any]
    component_id: str
    errors: List[str]
    warnings: List[str]
    already_registered: bool = False
    instantiation_method: str = 'unknown'
    config_layers_applied: int = 0

    @classmethod
    def success_result(cls, component: Any, component_id: str, method: str = 'direct', config_layers: int = 0, warnings: List[str] = None, already_registered: bool = False) -> 'ComponentInstantiationResult':
        return cls(
            success=True,
            component=component,
            component_id=component_id,
            errors=[],
            warnings=warnings or [],
            instantiation_method=method,
            config_layers_applied=config_layers,
            already_registered=already_registered
        )

    @classmethod
    def failure_result(cls, component_id: str, errors: List[str], warnings: List[str] = None, method: str = 'failed') -> 'ComponentInstantiationResult':
        return cls(
            success=False,
            component=None,
            component_id=component_id,
            errors=errors,
            warnings=warnings or [],
            instantiation_method=method
        )


class ComponentInstantiator:
    def __init__(self, mechanism_factory: Any, interface_validator: Any, registry_manager: Any, global_app_config: Dict[str, Any]):
        self.mechanism_factory = mechanism_factory
        self.interface_validator = interface_validator
        self.registry_manager = registry_manager
        self.global_app_config = global_app_config
        self.strict_mode = global_app_config.get('bootstrap_strict_mode', True)
        logger.info(f'ComponentInstantiator initialized (Strict Mode: {self.strict_mode})')

    async def instantiate_component(self, component_spec: ComponentSpec, context: Any) -> ComponentInstantiationResult:
        component_id = component_spec.component_id
        logger.info(f"Attempting instantiation for component '{component_id}' (type: {component_spec.component_type}, manifest: {component_spec.manifest_type})")

        try:
            if self._is_component_already_registered(component_id, context):
                logger.info(f"Component '{component_id}' is already registered. Retrieving existing instance.")
                if not hasattr(context, 'registry') or context.registry is None:
                    err_msg = f"Context lacks 'registry' for already registered component '{component_id}'."
                    logger.error(err_msg)
                    return ComponentInstantiationResult.failure_result(component_id, errors=[err_msg])
                
                existing_component = context.registry.get(component_id)
                return ComponentInstantiationResult.success_result(
                    component=existing_component,
                    component_id=component_id,
                    method='already_registered',
                    already_registered=True
                )

            if component_spec.manifest_type == 'enhanced':
                return await self._instantiate_enhanced_component(component_spec, context)
            elif component_spec.manifest_type == 'simple':
                return await self._instantiate_simple_component(component_spec, context)
            else:
                err_msg = f"Unknown manifest type '{component_spec.manifest_type}' for component '{component_id}'."
                logger.error(err_msg)
                return ComponentInstantiationResult.failure_result(component_id, errors=[err_msg])

        except Exception as e:
            error_msg = f"Critical unexpected error during instantiation of component '{component_id}': {str(e)}"
            logger.error(error_msg, exc_info=True)
            if self.strict_mode:
                raise ComponentInstantiationError(error_msg, component_id=component_id) from e
            return ComponentInstantiationResult.failure_result(component_id, errors=[error_msg])

    async def _instantiate_enhanced_component(self, spec: ComponentSpec, context: Any) -> ComponentInstantiationResult:
        component_id = spec.component_id
        spec_data = spec.spec_data
        class_path = spec_data.get('class')
        metadata_definition_path = spec_data.get('metadata_definition')
        component_type = spec.component_type

        # For shared services, metadata_definition is optional
        if component_type == 'shared_service':
            if not class_path:
                err = f"Shared service '{component_id}' missing required 'class' field."
                logger.error(err)
                return ComponentInstantiationResult.failure_result(component_id, errors=[err], method='shared_service_validation_failed')
            
            # For shared services, we can proceed without metadata_definition
            return await self._instantiate_shared_service(spec, context)
        
        # For other enhanced components (mechanisms, observers, managers, etc.), both are required
        if not class_path or not metadata_definition_path:
            err = f"Enhanced component '{component_id}' missing 'class' or 'metadata_definition'."
            logger.error(err)
            return ComponentInstantiationResult.failure_result(component_id, errors=[err], method='enhanced_class_validation_failed')

        try:
            component_class = import_by_path(class_path)
            if not inspect.isclass(component_class):
                err = f"Path '{class_path}' for '{component_id}' did not resolve to a class."
                logger.error(err)
                return ComponentInstantiationResult.failure_result(component_id, errors=[err], method='enhanced_class_import_failed')

            canonical_metadata = import_by_path(metadata_definition_path)
            if not isinstance(canonical_metadata, ComponentMetadata):
                err = f"Metadata path '{metadata_definition_path}' for '{component_id}' did not resolve to ComponentMetadata."
                logger.error(err)
                return ComponentInstantiationResult.failure_result(component_id, errors=[err], method='enhanced_class_metadata_import_failed')

            instance_metadata = self._build_instance_metadata(canonical_metadata, component_id, spec_data)
            resolved_config, config_layers = await self._resolve_component_config(spec, context)
            instance = await self._create_instance_from_class(component_class, resolved_config, instance_metadata, context)

            if instance is None:
                err = f"Instantiation of enhanced component '{component_id}' returned None."
                logger.error(err)
                return ComponentInstantiationResult.failure_result(component_id, errors=[err], method='enhanced_class_instantiation_returned_none')

            await self._register_component_with_certification(instance, instance_metadata, context)
            logger.info(f"✓ Enhanced component '{component_id}' instantiated successfully via class '{class_path}'.")
            return ComponentInstantiationResult.success_result(instance, component_id, 'enhanced_class', config_layers)

        except Exception as e:
            error_msg = f"Failed to instantiate enhanced component '{component_id}' from class '{class_path}': {str(e)}"
            logger.error(error_msg, exc_info=True)
            return ComponentInstantiationResult.failure_result(component_id, errors=[error_msg], method='enhanced_class_exception')

    async def _instantiate_shared_service(self, spec: ComponentSpec, context: Any) -> ComponentInstantiationResult:
        component_id = spec.component_id
        spec_data = spec.spec_data
        class_path = spec_data.get('class')

        try:
            # Import the service class
            service_class = import_by_path(class_path)
            if not inspect.isclass(service_class):
                err = f"Path '{class_path}' for shared service '{component_id}' did not resolve to a class."
                logger.error(err)
                return ComponentInstantiationResult.failure_result(component_id, errors=[err], method='shared_service_class_import_failed')

            # Create metadata for the shared service
            from bootstrap.bootstrap_helper.metadata import create_service_metadata
            service_metadata = create_service_metadata(
                service_id=component_id,
                service_name=component_id,
                category='shared_service',
                description=f'Shared service: {component_id}',
                requires_initialize=False
            )

            # Check if class has its own metadata and use that instead
            if hasattr(service_class, 'METADATA_DEFINITION') and isinstance(service_class.METADATA_DEFINITION, ComponentMetadata):
                base_metadata = service_class.METADATA_DEFINITION
                import dataclasses
                service_metadata = dataclasses.replace(base_metadata, id=component_id)
            elif hasattr(service_class, 'metadata') and isinstance(service_class.metadata, ComponentMetadata):
                base_metadata = service_class.metadata
                import dataclasses
                service_metadata = dataclasses.replace(base_metadata, id=component_id)

            # Apply any metadata overrides from the manifest
            service_metadata = self._build_instance_metadata(service_metadata, component_id, spec_data)
            
            # Resolve configuration
            resolved_config, config_layers = await self._resolve_component_config(spec, context)
            
            # Create the instance
            instance = await self._create_instance_from_class(service_class, resolved_config, service_metadata, context)
            
            if instance is None:
                err = f"Instantiation of shared service '{component_id}' returned None."
                logger.error(err)
                return ComponentInstantiationResult.failure_result(component_id, errors=[err], method='shared_service_instantiation_returned_none')

            # Register the service
            await self._register_component_with_certification(instance, service_metadata, context)
            logger.info(f"✓ Shared service '{component_id}' instantiated successfully via class '{class_path}'.")
            return ComponentInstantiationResult.success_result(instance, component_id, 'shared_service', config_layers)

        except Exception as e:
            error_msg = f"Failed to instantiate shared service '{component_id}' from class '{class_path}': {str(e)}"
            logger.error(error_msg, exc_info=True)
            return ComponentInstantiationResult.failure_result(component_id, errors=[error_msg], method='shared_service_exception')

    async def _instantiate_simple_component(self, spec: ComponentSpec, context: Any) -> ComponentInstantiationResult:
        component_id = spec.component_id
        spec_data = spec.spec_data
        factory_key = spec_data.get('factory_key')
        class_path = spec_data.get('class')
        component_type = spec_data.get('type', 'unknown')

        if not factory_key and not class_path:
            err = f"Simple component '{component_id}' requires 'factory_key' or 'class'."
            logger.error(err)
            return ComponentInstantiationResult.failure_result(component_id, errors=[err], method='simple_validation_failed')

        try:
            base_metadata = self._get_base_metadata_for_simple(factory_key, class_path, component_id, component_type)
            instance_metadata = self._build_instance_metadata(base_metadata, component_id, spec_data)
            resolved_config, config_layers = await self._resolve_component_config(spec, context)

            instance = None
            method_used = 'unknown_simple'

            if factory_key and component_type == 'mechanism' and self.mechanism_factory:
                instance = await self._create_instance_from_factory(factory_key, resolved_config, instance_metadata, context)
                method_used = 'factory'
            elif class_path:
                component_class = import_by_path(class_path)
                if not inspect.isclass(component_class):
                    err = f"Path '{class_path}' for simple component '{component_id}' did not resolve to a class."
                    logger.error(err)
                    return ComponentInstantiationResult.failure_result(component_id, errors=[err], method='simple_class_import_failed')
                
                instance = await self._create_instance_from_class(component_class, resolved_config, instance_metadata, context)
                method_used = 'simple_class'
            else:
                err = f"Cannot determine instantiation method for simple component '{component_id}'."
                logger.error(err)
                return ComponentInstantiationResult.failure_result(component_id, errors=[err], method='simple_method_undetermined')

            if instance is None:
                err = f"Instantiation of simple component '{component_id}' via {method_used} returned None."
                logger.error(err)
                return ComponentInstantiationResult.failure_result(component_id, errors=[err], method=f'{method_used}_returned_none')

            await self._register_component_with_certification(instance, instance_metadata, context)
            logger.info(f"✓ Simple component '{component_id}' instantiated successfully via {method_used}.")
            return ComponentInstantiationResult.success_result(instance, component_id, method_used, config_layers)

        except Exception as e:
            error_msg = f"Failed to instantiate simple component '{component_id}': {str(e)}"
            logger.error(error_msg, exc_info=True)
            return ComponentInstantiationResult.failure_result(component_id, errors=[error_msg], method='simple_exception')

    async def _resolve_component_config(self, spec: ComponentSpec, context: Any) -> tuple[Dict[str, Any], int]:
        component_id = spec.component_id
        spec_data = spec.spec_data
        logger.debug(f"Resolving configuration for '{component_id}'")
        
        layers_applied = 0
        config = {}
        
        # Layer 1: Pydantic defaults
        pydantic_defaults = await self._get_pydantic_defaults(spec, context)
        if pydantic_defaults:
            config = ConfigMerger.merge(config, pydantic_defaults, f'{component_id}_pydantic_defaults')
            layers_applied += 1
            logger.debug(f'[{component_id}] Config after Pydantic defaults: {list(config.keys())}')
        
        # Layer 2: Config loader
        if hasattr(context, 'config_loader') and context.config_loader is not None:
            try:
                loader_config = await context.config_loader.load_component_config(spec_data, component_id, self.global_app_config)
                if loader_config:
                    config = ConfigMerger.merge(config, loader_config, f'{component_id}_loader_config')
                    layers_applied += 1
                    logger.debug(f'[{component_id}] Config after loader: {list(loader_config.keys())}')
            except Exception as e:
                logger.warning(f"Config loader failed for '{component_id}': {e}", exc_info=True)
        
        # Layer 3: Manifest config
        manifest_config = spec_data.get('config', {})
        # Fix: Ensure manifest_config is a dictionary, not a string
        if isinstance(manifest_config, str):
            logger.warning(f"[{component_id}] Manifest 'config' field is a string ('{manifest_config}'), treating as config file path. Converting to empty dict for now.")
            manifest_config = {}
        elif not isinstance(manifest_config, dict):
            logger.warning(f"[{component_id}] Manifest 'config' field is not a dict or string (type: {type(manifest_config)}), using empty dict")
            manifest_config = {}
        
        if manifest_config:
            config = ConfigMerger.merge(config, manifest_config, f'{component_id}_manifest_config')
            layers_applied += 1
            logger.debug(f'[{component_id}] Config after manifest: {list(manifest_config.keys())}')
        
        # Layer 4: Config override
        config_override = spec_data.get('config_override', {})
        # Fix: Ensure config_override is a dictionary
        if not isinstance(config_override, dict):
            logger.warning(f"[{component_id}] 'config_override' is not a dict (type: {type(config_override)}), using empty dict")
            config_override = {}
        
        if config_override:
            config = ConfigMerger.merge(config, config_override, f'{component_id}_config_override')
            layers_applied += 1
            logger.debug(f'[{component_id}] Config after override: {list(config_override.keys())}')
        
        logger.info(f"Configuration for '{component_id}' resolved with {layers_applied} layers. Final keys: {list(config.keys())}")
        return (config, layers_applied)

    async def _get_pydantic_defaults(self, spec: ComponentSpec, context: Any) -> Dict[str, Any]:
        spec_data = spec.spec_data
        class_path = spec_data.get('class')
        if not class_path:
            return {}

        try:
            component_class = import_by_path(class_path)
            return self._get_pydantic_defaults_from_class(component_class, spec.component_id)
        except Exception:
            return {}

    def _get_pydantic_defaults_from_class(self, component_class: Type, component_name: str) -> Dict[str, Any]:
        logger.debug(f"Getting Pydantic defaults for '{component_name}' from class {component_class.__name__}")

        # Check for nested ConfigModel
        if hasattr(component_class, 'ConfigModel'):
            config_model_cls = getattr(component_class, 'ConfigModel')
            if hasattr(config_model_cls, 'model_json_schema'):
                try:
                    return config_model_cls().model_dump(exclude_unset=False)
                except Exception as e:
                    logger.debug(f'Failed to get defaults from nested ConfigModel for {component_name}: {e}')

        # Check for Pydantic model on the class itself
        if hasattr(component_class, '__pydantic_model__'):
            try:
                return component_class.__pydantic_model__.model_construct().model_dump()
            except Exception as e:
                logger.debug(f'Failed to get defaults from __pydantic_model__ for {component_name}: {e}')

        return {}

    async def _create_instance_from_class(self, component_class: Type, config: Dict[str, Any], metadata: ComponentMetadata, context: Any) -> Any:
        component_id = metadata.id
        logger.debug(f"Creating instance of {component_class.__name__} for ID '{component_id}' with config keys: {list(config.keys())}")

        try:
            sig = inspect.signature(component_class.__init__)
            constructor_args = {}

            # Standard arguments
            if 'config' in sig.parameters:
                constructor_args['config'] = config
            if 'metadata_definition' in sig.parameters:
                constructor_args['metadata_definition'] = metadata

            # Dependency injection
            if hasattr(context, 'common_mechanism_deps') and context.common_mechanism_deps:
                common_deps = context.common_mechanism_deps
                if 'common_deps' in sig.parameters:
                    constructor_args['common_deps'] = common_deps
                if 'llm' in sig.parameters and hasattr(common_deps, 'llm_port'):
                    constructor_args['llm'] = common_deps.llm_port
                if 'embedding_port' in sig.parameters and hasattr(common_deps, 'embedding_port'):
                    constructor_args['embedding_port'] = common_deps.embedding_port
                if 'event_bus' in sig.parameters and hasattr(common_deps, 'event_bus'):
                    constructor_args['event_bus'] = common_deps.event_bus
                if 'registry' in sig.parameters:
                    if hasattr(common_deps, 'component_registry') and common_deps.component_registry is not None:
                        constructor_args['registry'] = common_deps.component_registry
                    elif hasattr(context, 'registry') and context.registry is not None:
                        constructor_args['registry'] = context.registry
                    else:
                        logger.warning(f"Cannot inject 'registry' for {component_id}: not found in common_deps or context.")

            logger.debug(f'Attempting to instantiate {component_class.__name__} with args: {list(constructor_args.keys())}')
            instance = component_class(**constructor_args)

            # Ensure proper ID and metadata for NireonBaseComponent
            if isinstance(instance, NireonBaseComponent):
                if not hasattr(instance, 'component_id') or instance.component_id != metadata.id:
                    logger.warning(f"Instance ID mismatch for '{metadata.id}'. Instance has '{getattr(instance, 'component_id', 'MISSING')}'. Forcing ID from metadata.")
                    object.__setattr__(instance, '_component_id', metadata.id)
                if not hasattr(instance, 'metadata') or instance.metadata.id != metadata.id:
                    logger.warning(f"Instance metadata ID mismatch for '{metadata.id}'. Forcing metadata from instantiation.")
                    object.__setattr__(instance, '_metadata_definition', metadata)

            return instance

        except TypeError as e:
            logger.error(f"TypeError during instantiation of {component_class.__name__} for '{component_id}': {e}", exc_info=True)
            # Fallback attempts
            try:
                logger.debug(f"Fallback 1: Instantiating {component_class.__name__} with config only for '{component_id}'")
                return component_class(config=config)
            except TypeError:
                try:
                    logger.debug(f"Fallback 2: Instantiating {component_class.__name__} with no args for '{component_id}'")
                    return component_class()
                except TypeError as final_e:
                    full_error_msg = f"All instantiation attempts for {component_class.__name__} ('{component_id}') failed. Final error: {final_e}"
                    logger.critical(full_error_msg, exc_info=True)
                    raise ComponentInstantiationError(full_error_msg, component_id=component_id) from final_e
        except Exception as e:
            full_error_msg = f"Unexpected error during instantiation of {component_class.__name__} for '{component_id}': {e}"
            logger.critical(full_error_msg, exc_info=True)
            raise ComponentInstantiationError(full_error_msg, component_id=component_id) from e

    async def _create_instance_from_factory(self, factory_key: str, config: Dict[str, Any], metadata: ComponentMetadata, context: Any) -> Any:
        component_id = metadata.id
        if not self.mechanism_factory:
            raise ComponentInstantiationError(f"Mechanism factory not available for '{component_id}'.", component_id=component_id)

        logger.debug(f"Creating instance for '{component_id}' via factory key '{factory_key}'.")
        try:
            return self.mechanism_factory.create_mechanism(factory_key, metadata, config)
        except Exception as e:
            err_msg = f"Factory key '{factory_key}' failed to create component '{component_id}': {str(e)}"
            logger.error(err_msg, exc_info=True)
            raise ComponentInstantiationError(err_msg, component_id=component_id) from e

    def _build_instance_metadata(self, base_metadata: ComponentMetadata, component_id: str, spec_data: Dict[str, Any]) -> ComponentMetadata:
        import dataclasses
        instance_metadata_dict = dataclasses.asdict(base_metadata)
        instance_metadata_dict['id'] = component_id

        # Apply metadata overrides from manifest
        metadata_override = spec_data.get('metadata_override', {})
        if metadata_override:
            logger.debug(f"Applying metadata overrides for '{component_id}': {list(metadata_override.keys())}")
            for key, value in metadata_override.items():
                if key in instance_metadata_dict:
                    instance_metadata_dict[key] = value
                else:
                    logger.warning(f"Unknown metadata override key '{key}' for '{component_id}'.")

        # Apply epistemic tags from manifest
        if 'epistemic_tags' in spec_data:
            tags = spec_data['epistemic_tags']
            if isinstance(tags, list) and all(isinstance(t, str) for t in tags):
                instance_metadata_dict['epistemic_tags'] = tags
            else:
                logger.warning(f"Invalid 'epistemic_tags' in manifest for '{component_id}', expected list of strings.")

        # Handle requires_initialize
        if 'requires_initialize' in metadata_override:
            if isinstance(metadata_override['requires_initialize'], bool):
                instance_metadata_dict['requires_initialize'] = metadata_override['requires_initialize']
            else:
                logger.warning(f"Invalid 'requires_initialize' override for '{component_id}', using base value.")
                instance_metadata_dict['requires_initialize'] = base_metadata.requires_initialize
        elif 'requires_initialize' not in instance_metadata_dict:
            instance_metadata_dict['requires_initialize'] = True

        return ComponentMetadata(**instance_metadata_dict)

    def _get_base_metadata_for_simple(self, factory_key: Optional[str], class_path: Optional[str], component_id: str, component_type: str) -> ComponentMetadata:
        # Try to get metadata from factory key first
        if factory_key:
            base_meta = get_default_metadata(factory_key)
            if base_meta:
                return base_meta

        # Try to get metadata from class
        if class_path:
            try:
                component_class = import_by_path(class_path)
                if hasattr(component_class, 'METADATA_DEFINITION') and isinstance(component_class.METADATA_DEFINITION, ComponentMetadata):
                    return component_class.METADATA_DEFINITION
            except Exception as e:
                logger.debug(f"Could not get class metadata for '{class_path}' of '{component_id}': {e}")

        # Fallback to generic metadata
        logger.warning(f"No specific base metadata found for simple component '{component_id}', using fallback.")
        return ComponentMetadata(
            id=component_id,
            name=extract_class_name(class_path) if class_path else component_id,
            version='0.1.0-fallback',
            category=component_type,
            description=f'Fallback metadata for simple component: {component_id}',
            requires_initialize=True
        )

    async def _register_component_with_certification(self, instance: Any, metadata: ComponentMetadata, context: Any) -> None:
        component_id = metadata.id
        if not self.registry_manager:
            raise BootstrapError(f"RegistryManager not available for registering '{component_id}'.")

        try:
            self.registry_manager.register_with_certification(instance, metadata)
            logger.debug(f"Component '{component_id}' passed to registry_manager for registration with certification.")
        except Exception as e:
            err_msg = f"RegistryManager failed to register component '{component_id}': {str(e)}"
            logger.error(err_msg, exc_info=True)
            raise ComponentInstantiationError(err_msg, component_id=component_id) from e

    def _is_component_already_registered(self, component_id: str, context: Any) -> bool:
        if not hasattr(context, 'registry') or context.registry is None:
            logger.warning(f"Cannot check if '{component_id}' is registered: context.registry is missing or None.")
            return False
        
        try:
            existing = context.registry.get(component_id)
            if existing is None:
                return False
                
            # Check if the existing component is a placeholder that should be replaced
            placeholder_classes = [
                'PlaceholderLLMPortImpl',
                'PlaceholderEmbeddingPortImpl', 
                'PlaceholderEventBusImpl',
                'PlaceholderIdeaRepositoryImpl'
            ]
            
            existing_class_name = existing.__class__.__name__
            if existing_class_name in placeholder_classes:
                logger.info(f"Component '{component_id}' exists but is a placeholder ({existing_class_name}). Will replace with real implementation.")
                return False
            
            return True
            
        except ComponentRegistryMissingError:
            return False
        except Exception as e:
            logger.warning(f"Error checking if component '{component_id}' is registered: {e}. Assuming not registered.")
            return False