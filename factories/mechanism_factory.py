from __future__ import annotations
import logging
from typing import TYPE_CHECKING, Any, Dict

from core.base_component import NireonBaseComponent
from runtime.utils import import_by_path
from .dependencies import CommonMechanismDependencies

if TYPE_CHECKING:
    from core.lifecycle import ComponentMetadata

logger = logging.getLogger(__name__)

class SimpleMechanismFactory:
    def __init__(self, common_deps: CommonMechanismDependencies):
        if not isinstance(common_deps, CommonMechanismDependencies):
            raise TypeError('common_deps must be an instance of CommonMechanismDependencies')
        
        self.common_deps = common_deps
        
        # Define known factory keys and their corresponding classes
        # This allows backward compatibility for factory-based instantiation
        self.KNOWN_MECHANISMS = {
            "explorer_mechanism": "nireon.mechanisms.explorer.service:ExplorerMechanism",
            "explorer_default_key": "nireon.mechanisms.explorer.service:ExplorerMechanism",
            # Add more mechanism mappings here as needed
        }
        
        logger.info('V4 SimpleMechanismFactory initialized with mechanism mappings.')

    def create_mechanism(self, factory_key_or_class_path: str, metadata: ComponentMetadata, 
                        component_config: Dict[str, Any]) -> NireonBaseComponent:
        """
        Create mechanism from factory key or class path
        
        Args:
            factory_key_or_class_path: Either a known factory key or full class path
            metadata: Component metadata for the instance
            component_config: Resolved configuration for the component
            
        Returns:
            Instantiated mechanism component
        """
        logger.info(f"Creating mechanism for key/path: '{factory_key_or_class_path}' (component_id: {metadata.id})")
        
        actual_class_to_instantiate = None
        
        # Check if it's a known factory key
        if factory_key_or_class_path in self.KNOWN_MECHANISMS:
            class_path = self.KNOWN_MECHANISMS[factory_key_or_class_path]
            logger.debug(f"Resolved factory key '{factory_key_or_class_path}' to class path: {class_path}")
            try:
                actual_class_to_instantiate = import_by_path(class_path)
            except ImportError as e:
                logger.error(f"Could not import mechanism class '{class_path}' for factory key '{factory_key_or_class_path}': {e}")
                raise ValueError(f"Unknown mechanism or unimportable class for factory key: '{factory_key_or_class_path}'") from e
        else:
            # Assume it's a full class path
            try:
                actual_class_to_instantiate = import_by_path(factory_key_or_class_path)
                logger.debug(f"Imported mechanism class directly from path: {factory_key_or_class_path}")
            except ImportError as e:
                logger.error(f"Could not import mechanism class '{factory_key_or_class_path}': {e}")
                raise ValueError(f"Unknown mechanism or unimportable class: '{factory_key_or_class_path}'") from e
        
        # Validate that it's a mechanism class
        if not issubclass(actual_class_to_instantiate, NireonBaseComponent):
            raise ValueError(f"Class '{actual_class_to_instantiate.__name__}' is not a NireonBaseComponent")
        
        # Instantiate the mechanism with proper dependency injection
        try:
            import inspect
            sig = inspect.signature(actual_class_to_instantiate.__init__)
            constructor_args = {}
            
            # Standard arguments
            if 'config' in sig.parameters:
                constructor_args['config'] = component_config
            if 'metadata_definition' in sig.parameters:
                constructor_args['metadata_definition'] = metadata
            
            # Dependency injection
            if 'common_deps' in sig.parameters:
                constructor_args['common_deps'] = self.common_deps
            elif 'llm' in sig.parameters and self.common_deps.llm_port:
                constructor_args['llm'] = self.common_deps.llm_port
            elif 'embedding_port' in sig.parameters:
                constructor_args['embedding_port'] = self.common_deps.embedding_port
            elif 'embed' in sig.parameters:
                constructor_args['embed'] = self.common_deps.embedding_port
            elif 'event_bus' in sig.parameters and self.common_deps.event_bus:
                constructor_args['event_bus'] = self.common_deps.event_bus
            elif 'registry' in sig.parameters:
                constructor_args['registry'] = self.common_deps.component_registry
            elif 'idea_service' in sig.parameters and self.common_deps.idea_service:
                constructor_args['idea_service'] = self.common_deps.idea_service
            
            logger.debug(f"Instantiating {actual_class_to_instantiate.__name__} with args: {list(constructor_args.keys())}")
            instance = actual_class_to_instantiate(**constructor_args)
            
            # Validate the instance
            if not isinstance(instance, NireonBaseComponent):
                raise ValueError(f"Instantiated object is not a NireonBaseComponent: {type(instance)}")
            
            # Ensure component ID matches metadata
            if hasattr(instance, 'component_id') and instance.component_id != metadata.id:
                logger.warning(f"Component ID mismatch: instance.component_id='{instance.component_id}', metadata.id='{metadata.id}'. Correcting.")
                object.__setattr__(instance, '_component_id', metadata.id)
                object.__setattr__(instance, '_metadata_definition', metadata)
            
            logger.info(f"✓ Successfully created mechanism '{metadata.id}' of type {actual_class_to_instantiate.__name__}")
            return instance
            
        except Exception as e:
            error_msg = f"Failed to instantiate mechanism '{metadata.id}' using class {actual_class_to_instantiate.__name__}: {e}"
            logger.error(error_msg, exc_info=True)
            raise ValueError(error_msg) from e

    def register_mechanism_type(self, factory_key: str, class_path: str) -> None:
        """
        Register a new mechanism type with the factory
        
        Args:
            factory_key: Key to use for factory-based creation
            class_path: Full import path to the mechanism class
        """
        self.KNOWN_MECHANISMS[factory_key] = class_path
        logger.info(f"Registered mechanism type: '{factory_key}' -> '{class_path}'")

    def get_known_mechanism_types(self) -> Dict[str, str]:
        """Get all known mechanism types"""
        return dict(self.KNOWN_MECHANISMS)

    def supports_factory_key(self, factory_key: str) -> bool:
        """Check if factory supports a given key"""
        return factory_key in self.KNOWN_MECHANISMS