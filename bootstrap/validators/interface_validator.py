from __future__ import annotations
import logging
import inspect
from typing import Any, Dict, List, Optional, Tuple, Type, Protocol

from core.base_component import NireonBaseComponent
from core.lifecycle import ComponentMetadata, ComponentRegistryMissingError
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from domain.context import NireonExecutionContext

logger = logging.getLogger(__name__)

class InterfaceValidator:
    """V4 Interface and metadata validator for bootstrap components."""
    
    CORE_LIFECYCLE_METHODS = ['initialize', 'process', 'analyze', 'react', 'adapt', 'health_check', 'shutdown']
    
    def __init__(self, context: 'NireonExecutionContext'):
        self.context = context
        if self.context.component_registry is None:
            raise ValueError('InterfaceValidator requires a ComponentRegistry in the ExecutionContext.')
        self.component_registry = self.context.component_registry
        logger.info('V4 InterfaceValidator initialized.')
    
    async def validate_component(
        self, 
        instance: NireonBaseComponent, 
        expected_metadata: ComponentMetadata, 
        context: 'NireonExecutionContext',
        yaml_config_at_instantiation: Dict[str, Any],
        actual_runtime_metadata: ComponentMetadata,
        manifest_spec: Dict[str, Any]
    ) -> List[str]:
        """Validate a single component's interface and metadata."""
        
        errors: List[str] = []
        component_id = actual_runtime_metadata.id
        
        logger.debug(f'[{component_id}] Starting V4 interface and metadata validation.')
        
        # Basic type check
        if not isinstance(instance, NireonBaseComponent):
            errors.append(f"Component '{component_id}' is not an instance of NireonBaseComponent.")
            return errors
        
        # Check lifecycle methods
        for method_name in self.CORE_LIFECYCLE_METHODS:
            if not hasattr(instance, method_name) or not callable(getattr(instance, method_name)):
                errors.append(f"Component '{component_id}' is missing callable lifecycle method: '{method_name}'.")
            
            # Check for concrete implementations of key methods
            if method_name in ['process'] and not self._has_concrete_implementation(instance, f'_{method_name}_impl'):
                errors.append(f"Component '{component_id}' must provide a concrete implementation for '_{method_name}_impl'.")
        
        # Metadata validation
        if actual_runtime_metadata.id != expected_metadata.id:
            errors.append(f"ID mismatch: Instance metadata ID is '{actual_runtime_metadata.id}', but expected (from manifest/store) was '{expected_metadata.id}'. The manifest ID ('{component_id}') should be canonical.")
        
        if actual_runtime_metadata.name != expected_metadata.name:
            logger.debug(f"[{component_id}] Name mismatch: Instance metadata has '{actual_runtime_metadata.name}', expected (from manifest/store) was '{expected_metadata.name}'. This may be acceptable.")
        
        if actual_runtime_metadata.category != expected_metadata.category:
            errors.append(f"Category mismatch: instance='{actual_runtime_metadata.category}', expected='{expected_metadata.category}'.")
        
        if actual_runtime_metadata.version != expected_metadata.version:
            logger.debug(f"[{component_id}] Version mismatch: instance='{actual_runtime_metadata.version}', expected='{expected_metadata.version}'.")
        
        if set(actual_runtime_metadata.epistemic_tags) != set(expected_metadata.epistemic_tags):
            errors.append(f"Epistemic tags mismatch: instance={sorted(list(set(actual_runtime_metadata.epistemic_tags)))}, expected={sorted(list(set(expected_metadata.epistemic_tags)))}.")
        
        if actual_runtime_metadata.requires_initialize != expected_metadata.requires_initialize:
            errors.append(f"'requires_initialize' flag mismatch: instance metadata has '{actual_runtime_metadata.requires_initialize}', expected (from manifest/store) was '{expected_metadata.requires_initialize}'.")
        
        # Configuration validation
        if yaml_config_at_instantiation and not instance.config:
            errors.append(f"Component '{component_id}' was provided a non-empty configuration during instantiation but its runtime `config` attribute is empty or None.")
        
        # Initialization state validation
        if actual_runtime_metadata.requires_initialize and not instance.is_initialized:
            errors.append(f"Component '{component_id}' requires initialization, but 'is_initialized' is False after initialization phase.")
        elif not actual_runtime_metadata.requires_initialize and instance.is_initialized:
            logger.debug(f"[{component_id}] Component 'is_initialized' is True, but metadata.requires_initialize is False. This is acceptable if component initializes itself eagerly.")
        
        # Self-certification validation (with improved error handling)
        try:
            certification_data = self.component_registry.get_certification(component_id)
            if not certification_data:
                # Check if this is a type-based key that shouldn't have certification
                if self._is_type_based_key(component_id):
                    logger.debug(f"[{component_id}] Skipping certification check for type-based registry key")
                else:
                    errors.append(f"Component '{component_id}' is missing self-certification data in the registry.")
            else:
                # Validate certification data
                if certification_data.get('component_id') != component_id:
                    errors.append(f"Self-certification ID mismatch: cert has '{certification_data.get('component_id')}', expected '{component_id}'.")
                
                if certification_data.get('status') not in ['initialized', 'initializing', 'healthy', 'registered']:
                    logger.debug(f"[{component_id}] Self-certification status is '{certification_data.get('status')}'.")
        
        except (ComponentRegistryMissingError, KeyError, AttributeError) as e:
            # Check if this is a type-based key before reporting error
            if self._is_type_based_key(component_id):
                logger.debug(f"[{component_id}] Skipping certification check for type-based registry key: {e}")
            else:
                errors.append(f"Could not retrieve self-certification data for component '{component_id}': {e}")
        
        # Interface protocol validation
        if hasattr(expected_metadata, 'expected_interfaces') and expected_metadata.expected_interfaces:
            for i, expected_iface_protocol in enumerate(expected_metadata.expected_interfaces):
                if not isinstance(instance, expected_iface_protocol):
                    errors.append(f"Component '{component_id}' does not implement expected V4 interface protocol #{i+1}: '{expected_iface_protocol.__name__}' (defined in manifest/canonical metadata).")
                else:
                    logger.debug(f"[{component_id}] Implements expected V4 interface protocol: {expected_iface_protocol.__name__}")
        
        # Log results
        if errors:
            logger.warning(f'[{component_id}] Validation failed with {len(errors)} errors: {errors}')
        else:
            logger.info(f'✓ [{component_id}] Passed V4 interface and metadata validation.')
        
        return errors
    
    def _is_type_based_key(self, component_id: str) -> bool:
        """Check if a component ID appears to be a type-based registry key."""
        # Type-based keys typically contain module paths with dots
        # and class names, e.g., 'domain.ports.llm_port.LLMPort'
        if '.' in component_id and any(segment[0].isupper() for segment in component_id.split('.') if segment):
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
    
    def _has_concrete_implementation(self, instance: object, method_name: str) -> bool:
        """Check if an instance has a concrete implementation of a method."""
        
        method = getattr(instance, method_name, None)
        if not method or not callable(method):
            return False
        
        # Check if method is abstract
        if hasattr(method, '__isabstractmethod__') and method.__isabstractmethod__:
            return False
        
        # Check if method is defined in the instance's class (not inherited as abstract)
        cls = type(instance)
        if method_name in cls.__dict__:
            return True
        
        # Special handling for NireonBaseComponent _process_impl
        if isinstance(instance, NireonBaseComponent) and method_name == '_process_impl':
            base_impl = getattr(NireonBaseComponent, '_process_impl', None)
            instance_impl = getattr(cls, '_process_impl', None)
            if instance_impl is base_impl:
                return False
            return True
        
        # Check inheritance chain for abstract methods
        for base_cls in cls.__mro__[1:]:  # Skip the instance's own class
            if method_name in base_cls.__dict__:
                base_method_attr = getattr(base_cls, method_name)
                if hasattr(base_method_attr, '__isabstractmethod__') and base_method_attr.__isabstractmethod__:
                    # If base class has abstract method but instance class doesn't override it
                    if method_name not in cls.__dict__:
                        return False
                return True
        
        return False