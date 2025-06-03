"""
Component instantiation engine for NIREON V4 bootstrap.

This module handles the creation of component instances from ComponentSpec objects,
supporting both enhanced and simple manifest types with full configuration resolution.
"""

from __future__ import annotations
import asyncio
import inspect
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type, Union
from pathlib import Path

from application.components.base import NireonBaseComponent
from application.components.lifecycle import ComponentMetadata
from bootstrap.processors.manifest_processor import ComponentSpec
from bootstrap.bootstrap_helper.exceptions import (
    ComponentInstantiationError, 
    ConfigurationError,
    BootstrapError
)
from bootstrap.bootstrap_helper.utils import import_by_path, extract_class_name
from bootstrap.bootstrap_helper.metadata import get_default_metadata
from configs.config_utils import ConfigMerger

logger = logging.getLogger(__name__)


@dataclass
class ComponentInstantiationResult:
    """
    Result of component instantiation process.
    
    Contains the created component (if successful), error information,
    and metadata about the instantiation process.
    """
    success: bool
    component: Optional[Any]
    component_id: str
    errors: List[str]
    warnings: List[str]
    already_registered: bool = False
    instantiation_method: str = "unknown"
    config_layers_applied: int = 0
    
    @classmethod
    def success_result(
        cls, 
        component: Any, 
        component_id: str, 
        method: str = "direct",
        config_layers: int = 0,
        warnings: List[str] = None
    ) -> 'ComponentInstantiationResult':
        """Create a successful instantiation result."""
        return cls(
            success=True,
            component=component,
            component_id=component_id,
            errors=[],
            warnings=warnings or [],
            instantiation_method=method,
            config_layers_applied=config_layers
        )
    
    @classmethod
    def failure_result(
        cls, 
        component_id: str, 
        errors: List[str], 
        warnings: List[str] = None
    ) -> 'ComponentInstantiationResult':
        """Create a failed instantiation result."""
        return cls(
            success=False,
            component=None,
            component_id=component_id,
            errors=errors,
            warnings=warnings or [],
            instantiation_method="failed"
        )


class ComponentInstantiator:
    """
    Engine for instantiating components from manifest specifications.
    
    Handles both enhanced and simple component types, resolves configuration
    through the 6-layer hierarchy, and manages registration with certification.
    """
    
    def __init__(
        self, 
        mechanism_factory, 
        interface_validator, 
        registry_manager, 
        global_app_config: Dict[str, Any]
    ):
        self.mechanism_factory = mechanism_factory
        self.interface_validator = interface_validator
        self.registry_manager = registry_manager
        self.global_app_config = global_app_config
        self.strict_mode = global_app_config.get('bootstrap_strict_mode', True)
        
        logger.info("ComponentInstantiator initialized")
    
    async def instantiate_component(
        self, 
        component_spec: ComponentSpec, 
        context
    ) -> ComponentInstantiationResult:
        """
        Main entry point for component instantiation.
        
        Args:
            component_spec: Component specification from manifest
            context: Bootstrap context with registry, config loader, etc.
            
        Returns:
            ComponentInstantiationResult with created component or errors
        """
        component_id = component_spec.component_id
        logger.info(f"Instantiating component '{component_id}' (type: {component_spec.component_type}, manifest: {component_spec.manifest_type})")
        
        try:
            # Check if component already exists
            if self._is_component_already_registered(component_id, context):
                logger.info(f"Component '{component_id}' already registered, skipping instantiation")
                existing_component = context.registry.get(component_id)
                return ComponentInstantiationResult.success_result(
                    component=existing_component,
                    component_id=component_id,
                    method="already_registered"
                )
            
            # Route to appropriate instantiation method
            if component_spec.manifest_type == 'enhanced':
                return await self._instantiate_enhanced_component(component_spec, context)
            elif component_spec.manifest_type == 'simple':
                return await self._instantiate_simple_component(component_spec, context)
            else:
                return ComponentInstantiationResult.failure_result(
                    component_id=component_id,
                    errors=[f"Unknown manifest type: {component_spec.manifest_type}"]
                )
        
        except Exception as e:
            error_msg = f"Critical error instantiating component '{component_id}': {e}"
            logger.error(error_msg, exc_info=True)
            
            if self.strict_mode:
                raise ComponentInstantiationError(error_msg, component_id=component_id) from e
            
            return ComponentInstantiationResult.failure_result(
                component_id=component_id,
                errors=[error_msg]
            )
    
    async def _instantiate_enhanced_component(
        self, 
        spec: ComponentSpec, 
        context
    ) -> ComponentInstantiationResult:
        """
        Instantiate an enhanced manifest component.
        
        Enhanced components have explicit class and metadata_definition paths
        and use the full 6-layer configuration resolution.
        """
        component_id = spec.component_id
        spec_data = spec.spec_data
        
        # Extract required paths
        class_path = spec_data.get('class')
        metadata_definition_path = spec_data.get('metadata_definition')
        
        if not class_path or not metadata_definition_path:
            return ComponentInstantiationResult.failure_result(
                component_id=component_id,
                errors=[f"Enhanced component '{component_id}' missing required 'class' or 'metadata_definition'"]
            )
        
        try:
            # Import component class
            component_class = import_by_path(class_path)
            if not inspect.isclass(component_class):
                return ComponentInstantiationResult.failure_result(
                    component_id=component_id,
                    errors=[f"Path '{class_path}' did not resolve to a class"]
                )
            
            # Import canonical metadata
            canonical_metadata = import_by_path(metadata_definition_path)
            if not isinstance(canonical_metadata, ComponentMetadata):
                return ComponentInstantiationResult.failure_result(
                    component_id=component_id,
                    errors=[f"Metadata path '{metadata_definition_path}' did not resolve to ComponentMetadata"]
                )
            
            # Build instance-specific metadata
            instance_metadata = self._build_instance_metadata(canonical_metadata, component_id, spec_data)
            
            # Resolve configuration through 6-layer hierarchy
            resolved_config, config_layers = await self._resolve_component_config(spec, context)
            
            # Create component instance
            instance = await self._create_instance_from_class(
                component_class, 
                resolved_config, 
                instance_metadata,
                context
            )
            
            if instance is None:
                return ComponentInstantiationResult.failure_result(
                    component_id=component_id,
                    errors=[f"Component instantiation returned None"]
                )
            
            # Register with certification
            await self._register_component_with_certification(instance, instance_metadata, context)
            
            logger.info(f"✓ Enhanced component '{component_id}' instantiated successfully")
            return ComponentInstantiationResult.success_result(
                component=instance,
                component_id=component_id,
                method="enhanced_class",
                config_layers=config_layers
            )
        
        except Exception as e:
            error_msg = f"Failed to instantiate enhanced component '{component_id}': {e}"
            logger.error(error_msg, exc_info=True)
            return ComponentInstantiationResult.failure_result(
                component_id=component_id,
                errors=[error_msg]
            )
    
    async def _instantiate_simple_component(
        self, 
        spec: ComponentSpec, 
        context
    ) -> ComponentInstantiationResult:
        """
        Instantiate a simple manifest component.
        
        Simple components can use either factory_key or class path
        and rely on default metadata with optional overrides.
        """
        component_id = spec.component_id
        spec_data = spec.spec_data
        
        factory_key = spec_data.get('factory_key')
        class_path = spec_data.get('class')
        component_type = spec_data.get('type', 'unknown')
        
        if not factory_key and not class_path:
            return ComponentInstantiationResult.failure_result(
                component_id=component_id,
                errors=[f"Simple component '{component_id}' needs either 'factory_key' or 'class'"]
            )
        
        try:
            # Get base metadata (from factory key or class)
            base_metadata = self._get_base_metadata_for_simple(factory_key, class_path, component_id, component_type)
            
            # Build instance metadata with overrides
            instance_metadata = self._build_instance_metadata(base_metadata, component_id, spec_data)
            
            # Resolve configuration
            resolved_config, config_layers = await self._resolve_component_config(spec, context)
            
            # Create instance using appropriate method
            if factory_key and component_type == 'mechanism' and self.mechanism_factory:
                instance = await self._create_instance_from_factory(
                    factory_key, 
                    resolved_config, 
                    instance_metadata, 
                    context
                )
                method = "factory"
            elif class_path:
                component_class = import_by_path(class_path)
                instance = await self._create_instance_from_class(
                    component_class, 
                    resolved_config, 
                    instance_metadata,
                    context
                )
                method = "simple_class"
            else:
                return ComponentInstantiationResult.failure_result(
                    component_id=component_id,
                    errors=[f"Cannot determine instantiation method for '{component_id}'"]
                )
            
            if instance is None:
                return ComponentInstantiationResult.failure_result(
                    component_id=component_id,
                    errors=[f"Component instantiation returned None"]
                )
            
            # Register with certification
            await self._register_component_with_certification(instance, instance_metadata, context)
            
            logger.info(f"✓ Simple component '{component_id}' instantiated successfully")
            return ComponentInstantiationResult.success_result(
                component=instance,
                component_id=component_id,
                method=method,
                config_layers=config_layers
            )
        
        except Exception as e:
            error_msg = f"Failed to instantiate simple component '{component_id}': {e}"
            logger.error(error_msg, exc_info=True)
            return ComponentInstantiationResult.failure_result(
                component_id=component_id,
                errors=[error_msg]
            )
    
    async def _resolve_component_config(
        self, 
        spec: ComponentSpec, 
        context
    ) -> tuple[Dict[str, Any], int]:
        """
        Resolve component configuration through 6-layer hierarchy.
        
        Layers (highest to lowest precedence):
        1. Runtime adaptations (not applicable during bootstrap)
        2. Environment variables (handled by config loader)
        3. Manifest overrides (config_override in spec)
        4. Environment configs (handled by config loader)
        5. Default configs (handled by config loader)
        6. Python defaults (Pydantic model defaults)
        
        Returns:
            Tuple of (resolved_config, layers_applied_count)
        """
        component_id = spec.component_id
        spec_data = spec.spec_data
        
        logger.debug(f"Resolving configuration for '{component_id}' through 6-layer hierarchy")
        
        layers_applied = 0
        
        # Layer 6: Python defaults (Pydantic model defaults)
        pydantic_defaults = await self._get_pydantic_defaults(spec, context)
        config = pydantic_defaults.copy()
        if pydantic_defaults:
            layers_applied += 1
            logger.debug(f"[{component_id}] Applied Pydantic defaults: {list(pydantic_defaults.keys())}")
        
        # Layers 5-4: Default and environment configs (via config loader)
        if hasattr(context, 'config_loader'):
            try:
                loader_config = await context.config_loader.load_component_config(
                    spec_data, component_id, self.global_app_config
                )
                if loader_config:
                    config = ConfigMerger.merge(
                        config, 
                        loader_config, 
                        f'{component_id}_loader_config'
                    )
                    layers_applied += 1
                    logger.debug(f"[{component_id}] Applied config loader: {list(loader_config.keys())}")
            except Exception as e:
                logger.warning(f"Config loader failed for '{component_id}': {e}")
        
        # Layer 3: Manifest overrides
        manifest_config = spec_data.get('config', {})
        config_override = spec_data.get('config_override', {})
        
        if manifest_config:
            config = ConfigMerger.merge(
                config, 
                manifest_config, 
                f'{component_id}_manifest_config'
            )
            layers_applied += 1
            logger.debug(f"[{component_id}] Applied manifest config: {list(manifest_config.keys())}")
        
        if config_override:
            config = ConfigMerger.merge(
                config, 
                config_override, 
                f'{component_id}_config_override'
            )
            layers_applied += 1
            logger.debug(f"[{component_id}] Applied config override: {list(config_override.keys())}")
        
        # Layers 2-1: Environment variables and runtime adaptations are handled by the config loader
        
        logger.info(f"Configuration resolved for '{component_id}': {layers_applied} layers applied")
        return config, layers_applied
    
    async def _get_pydantic_defaults(self, spec: ComponentSpec, context) -> Dict[str, Any]:
        """Extract Pydantic model defaults from component class."""
        spec_data = spec.spec_data
        class_path = spec_data.get('class')
        
        if not class_path:
            return {}
        
        try:
            component_class = import_by_path(class_path)
            return self._extract_pydantic_defaults_from_class(component_class, spec.component_id)
        except Exception as e:
            logger.debug(f"Could not get Pydantic defaults for '{spec.component_id}': {e}")
            return {}
    
    def _extract_pydantic_defaults_from_class(self, component_class: Type, component_name: str) -> Dict[str, Any]:
        """Extract Pydantic defaults from a component class."""
        logger.debug(f"Attempting to get Pydantic defaults for '{component_name}' from class: {component_class.__name__}")
        
        # Look for nested ConfigModel
        if hasattr(component_class, 'ConfigModel'):
            config_model_cls = getattr(component_class, 'ConfigModel')
            if hasattr(config_model_cls, 'model_construct'):
                try:
                    return config_model_cls.model_construct().model_dump()
                except Exception as e:
                    logger.debug(f"Failed to get defaults from nested ConfigModel: {e}")
        
        # Look for config model in related modules
        module_path = component_class.__module__
        component_class_name = component_class.__name__
        expected_config_class_name = f'{component_class_name}Config'
        
        # Try common config module patterns
        config_module_patterns = [
            f'{module_path}.config',
            module_path,
        ]
        
        if '.' in module_path:
            parent_module = module_path.rsplit('.', 1)[0]
            config_module_patterns.append(f'{parent_module}.config')
        
        for config_module_path in config_module_patterns:
            try:
                import importlib
                config_module = importlib.import_module(config_module_path)
                if hasattr(config_module, expected_config_class_name):
                    candidate_cls = getattr(config_module, expected_config_class_name)
                    if (hasattr(candidate_cls, 'model_construct') and 
                        hasattr(candidate_cls, 'model_dump')):
                        try:
                            return candidate_cls.model_construct().model_dump()
                        except Exception as e:
                            logger.debug(f"Error getting defaults from {expected_config_class_name}: {e}")
                        break
            except ImportError:
                continue
            except Exception as e:
                logger.debug(f"Error accessing config from '{config_module_path}': {e}")
        
        logger.debug(f"No Pydantic config model found for '{component_name}'")
        return {}
    
    async def _create_instance_from_class(
        self, 
        component_class: Type, 
        config: Dict[str, Any], 
        metadata: ComponentMetadata,
        context
    ) -> Any:
        """Create component instance from class with dependency injection."""
        logger.debug(f"Creating instance of {component_class.__name__} with config keys: {list(config.keys())}")
        
        # Verify metadata ID matches
        if metadata.id != context.get('component_id', metadata.id):
            logger.warning(f"Metadata ID '{metadata.id}' may not match expected component ID")
        
        try:
            sig = inspect.signature(component_class.__init__)
            constructor_args = {}
            
            # Standard V4 component arguments
            if 'config' in sig.parameters:
                constructor_args['config'] = config
            
            if 'metadata_definition' in sig.parameters:
                constructor_args['metadata_definition'] = metadata
            
            # Dependency injection from context
            if hasattr(context, 'common_mechanism_deps') and context.common_mechanism_deps:
                common_deps = context.common_mechanism_deps
                
                if 'common_deps' in sig.parameters:
                    constructor_args['common_deps'] = common_deps
                elif 'llm' in sig.parameters and hasattr(common_deps, 'llm_port'):
                    constructor_args['llm'] = common_deps.llm_port
                elif 'embed' in sig.parameters and hasattr(common_deps, 'embedding_port'):
                    constructor_args['embed'] = common_deps.embedding_port
                elif 'embedding_port' in sig.parameters and hasattr(common_deps, 'embedding_port'):
                    constructor_args['embedding_port'] = common_deps.embedding_port
                elif 'event_bus' in sig.parameters and hasattr(common_deps, 'event_bus'):
                    constructor_args['event_bus'] = common_deps.event_bus
                elif 'registry' in sig.parameters and hasattr(common_deps, 'component_registry'):
                    constructor_args['registry'] = common_deps.component_registry
            
            logger.debug(f"Instantiating {component_class.__name__} with args: {list(constructor_args.keys())}")
            instance = component_class(**constructor_args)
            
            # Verify the instance has correct component_id
            if isinstance(instance, NireonBaseComponent):
                if instance.component_id != metadata.id:
                    logger.error(f"Instance component_id '{instance.component_id}' != metadata.id '{metadata.id}'")
                    # Force correction
                    object.__setattr__(instance, '_component_id', metadata.id)
                    object.__setattr__(instance, '_metadata_definition', metadata)
            
            return instance
        
        except TypeError as e:
            # Try fallback instantiation patterns
            logger.debug(f"Primary instantiation failed for {component_class.__name__}: {e}")
            
            # Try with just config
            try:
                return component_class(config=config)
            except TypeError:
                # Try with no args
                try:
                    return component_class()
                except TypeError as final_error:
                    logger.error(f"All instantiation attempts failed for {component_class.__name__}: {final_error}")
                    raise ComponentInstantiationError(
                        f"Could not instantiate {component_class.__name__}: {final_error}",
                        component_id=metadata.id
                    ) from final_error
    
    async def _create_instance_from_factory(
        self, 
        factory_key: str, 
        config: Dict[str, Any], 
        metadata: ComponentMetadata,
        context
    ) -> Any:
        """Create component instance using mechanism factory."""
        if not self.mechanism_factory:
            raise ComponentInstantiationError(
                f"Cannot create component '{metadata.id}' - mechanism factory not available",
                component_id=metadata.id
            )
        
        logger.debug(f"Creating instance via factory key '{factory_key}' for component '{metadata.id}'")
        
        try:
            return self.mechanism_factory.create_mechanism(factory_key, metadata, config)
        except Exception as e:
            logger.error(f"Factory creation failed for '{factory_key}': {e}")
            raise ComponentInstantiationError(
                f"Factory failed to create component '{metadata.id}': {e}",
                component_id=metadata.id
            ) from e
    
    def _build_instance_metadata(
        self, 
        base_metadata: ComponentMetadata, 
        component_id: str, 
        spec_data: Dict[str, Any]
    ) -> ComponentMetadata:
        """Build instance-specific metadata from base metadata and manifest overrides."""
        import dataclasses
        
        # Start with base metadata as dict
        instance_metadata_dict = dataclasses.asdict(base_metadata)
        
        # Override ID to match manifest
        instance_metadata_dict['id'] = component_id
        
        # Apply manifest metadata overrides
        metadata_override = spec_data.get('metadata_override', {})
        if metadata_override:
            logger.debug(f"Applying metadata overrides for '{component_id}': {metadata_override}")
            for key, value in metadata_override.items():
                if key in instance_metadata_dict:
                    instance_metadata_dict[key] = value
                elif key == 'requires_initialize' and isinstance(value, bool):
                    instance_metadata_dict[key] = value
                else:
                    logger.warning(f"Unknown metadata override key '{key}' for '{component_id}'")
        
        # Apply epistemic tags from manifest
        if 'epistemic_tags' in spec_data:
            tags = spec_data['epistemic_tags']
            if isinstance(tags, list) and all(isinstance(t, str) for t in tags):
                instance_metadata_dict['epistemic_tags'] = tags
            else:
                logger.warning(f"Invalid epistemic_tags in manifest for '{component_id}'")
        
        # Ensure requires_initialize is set
        if 'requires_initialize' not in instance_metadata_dict:
            instance_metadata_dict['requires_initialize'] = base_metadata.requires_initialize
        
        return ComponentMetadata(**instance_metadata_dict)
    
    def _get_base_metadata_for_simple(
        self, 
        factory_key: Optional[str], 
        class_path: Optional[str], 
        component_id: str, 
        component_type: str
    ) -> ComponentMetadata:
        """Get base metadata for simple component from factory key or class."""
        # Try factory key first
        if factory_key:
            base_metadata = get_default_metadata(factory_key)
            if base_metadata:
                return base_metadata
        
        # Try class metadata
        if class_path:
            try:
                component_class = import_by_path(class_path)
                if hasattr(component_class, 'METADATA_DEFINITION'):
                    return component_class.METADATA_DEFINITION
            except Exception as e:
                logger.debug(f"Could not get class metadata for '{class_path}': {e}")
        
        # Fallback to minimal metadata
        logger.warning(f"No base metadata found for '{component_id}', using fallback")
        return ComponentMetadata(
            id=component_id,
            name=extract_class_name(class_path) if class_path else component_id,
            version='1.0.0',
            category=component_type,
            description=f"Simple component: {component_id}",
            epistemic_tags=[]
        )
    
    async def _register_component_with_certification(
        self, 
        instance: Any, 
        metadata: ComponentMetadata, 
        context
    ) -> None:
        """Register component with self-certification."""
        try:
            self.registry_manager.register_with_certification(instance, metadata)
            logger.debug(f"Component '{metadata.id}' registered with certification")
        except Exception as e:
            logger.error(f"Failed to register component '{metadata.id}': {e}")
            raise ComponentInstantiationError(
                f"Registration failed for '{metadata.id}': {e}",
                component_id=metadata.id
            ) from e
    
    def _is_component_already_registered(self, component_id: str, context) -> bool:
        """Check if a component is already registered."""
        try:
            existing = context.registry.get(component_id)
            return existing is not None
        except Exception:
            return False