# nireon_v4/validation/interface_validator.py
from __future__ import annotations
import logging
import inspect # For inspecting methods
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Type, Protocol

if TYPE_CHECKING:
    from nireon_v4.application.components.base import NireonBaseComponent
    from nireon_v4.application.components.lifecycle import ComponentMetadata # V4
    from nireon_v4.application.context import NireonExecutionContext # V4

logger = logging.getLogger(__name__)

class InterfaceValidator:
    # V4: NireonComponent protocol methods, more flexible than just a list
    # These are core methods expected by the runtime for lifecycle management.
    # Inheriting from NireonBaseComponent should provide these.
    CORE_LIFECYCLE_METHODS = [
        "initialize", "process", "analyze", "react", "adapt", "health_check", "shutdown"
        # "recover_from_error" is also good to have
    ]

    def __init__(self, context: NireonExecutionContext): # V4 context
        self.context = context
        if self.context.component_registry is None:
            # This check is important. The validator needs the registry.
            raise ValueError("InterfaceValidator requires a ComponentRegistry in the ExecutionContext.")
        self.component_registry = self.context.component_registry
        logger.info("V4 InterfaceValidator initialized.")

    async def validate_component(
        self,
        instance: NireonBaseComponent,
        expected_metadata: ComponentMetadata, # Metadata derived from manifest/defaults stored in BootstrapValidationData
        context: NireonExecutionContext,      # Execution context for this validation run
        yaml_config_at_instantiation: Dict[str, Any], # The config that was used to create the instance
        actual_runtime_metadata: ComponentMetadata,   # The .metadata attribute of the live instance
        manifest_spec: Dict[str, Any] # The raw component spec from the manifest
    ) -> List[str]:
        """
        Validates a V4 component instance against its expected contract and metadata.
        """
        errors: List[str] = []
        component_id = actual_runtime_metadata.id # Use the ID from the live instance's metadata

        logger.debug(f"[{component_id}] Starting V4 interface and metadata validation.")

        # 1. NireonBaseComponent Adherence (basic checks if not already guaranteed by type)
        if not isinstance(instance, NireonBaseComponent):
            errors.append(f"Component '{component_id}' is not an instance of NireonBaseComponent.")
            # If not a NireonBaseComponent, many subsequent checks might fail or be irrelevant.
            # However, V4 might allow non-NireonBaseComponent services if they adhere to specific ports.
            # For now, focusing on NireonBaseComponent derivatives for full validation.
            return errors 

        # 2. Lifecycle Method Implementation
        for method_name in self.CORE_LIFECYCLE_METHODS:
            if not hasattr(instance, method_name) or not callable(getattr(instance, method_name)):
                errors.append(f"Component '{component_id}' is missing callable lifecycle method: '{method_name}'.")
            # Check for abstract _impl methods if it's a common pattern
            if method_name in ["process"] and not self._has_concrete_implementation(instance, f"_{method_name}_impl"):
                 errors.append(f"Component '{component_id}' must provide a concrete implementation for '_{method_name}_impl'.")
        
        # 3. Metadata Consistency Checks
        #    - Compare actual_runtime_metadata (on the instance) vs. expected_metadata (from manifest + defaults)
        
        if actual_runtime_metadata.id != expected_metadata.id:
            # This should ideally be caught and corrected during instantiation by _create_component_instance_v4 or init_full_component_v4
            errors.append(
                f"ID mismatch: Instance metadata ID is '{actual_runtime_metadata.id}', "
                f"but expected (from manifest/store) was '{expected_metadata.id}'. "
                f"The manifest ID ('{component_id}') should be canonical."
            )
        
        if actual_runtime_metadata.name != expected_metadata.name:
            # Names can sometimes be dynamic, so this might be a warning or configurable strictness.
            logger.debug(
                f"[{component_id}] Name mismatch: Instance metadata has '{actual_runtime_metadata.name}', "
                f"expected (from manifest/store) was '{expected_metadata.name}'. This may be acceptable."
            )
            # errors.append(f"Name mismatch: instance='{actual_runtime_metadata.name}', expected='{expected_metadata.name}'.")

        if actual_runtime_metadata.category != expected_metadata.category:
            errors.append(
                f"Category mismatch: instance='{actual_runtime_metadata.category}', "
                f"expected='{expected_metadata.category}'."
            )
        
        if actual_runtime_metadata.version != expected_metadata.version:
             logger.debug( # Version might be updated dynamically by component, treat as warning/debug
                f"[{component_id}] Version mismatch: instance='{actual_runtime_metadata.version}', "
                f"expected='{expected_metadata.version}'."
            )

        # Compare epistemic_tags (set comparison for order insensitivity)
        if set(actual_runtime_metadata.epistemic_tags) != set(expected_metadata.epistemic_tags):
            errors.append(
                f"Epistemic tags mismatch: instance={sorted(list(set(actual_runtime_metadata.epistemic_tags)))}, "
                f"expected={sorted(list(set(expected_metadata.epistemic_tags)))}."
            )

        # Check `requires_initialize` consistency
        if actual_runtime_metadata.requires_initialize != expected_metadata.requires_initialize:
            errors.append(
                f"'requires_initialize' flag mismatch: instance metadata has '{actual_runtime_metadata.requires_initialize}', "
                f"expected (from manifest/store) was '{expected_metadata.requires_initialize}'."
            )


        # 4. Configuration State
        #    - Check if instance.config reflects the merged configuration from bootstrap.
        #    - This is a complex check. yaml_config_at_instantiation is what was passed to constructor.
        #      The instance.config might be further processed by Pydantic.
        #      A simple check: if yaml_config_at_instantiation was non-empty, instance.config should also be.
        if yaml_config_at_instantiation and not instance.config:
            errors.append(
                f"Component '{component_id}' was provided a non-empty configuration during instantiation "
                f"but its runtime `config` attribute is empty or None."
            )
        # More detailed config validation could involve Pydantic model re-validation if applicable.
        # For V4, Pydantic models on components should handle their own config validation.
        # This validator can check if `instance.config` is an instance of the expected Pydantic model
        # if the Pydantic model type can be inferred (e.g., from instance_metadata.expected_interfaces or convention).

        # 5. Initialization Status (after Phase 5 init loop)
        if actual_runtime_metadata.requires_initialize and not instance.is_initialized:
            errors.append(
                f"Component '{component_id}' requires initialization, but 'is_initialized' is False after initialization phase."
            )
        elif not actual_runtime_metadata.requires_initialize and instance.is_initialized:
            logger.debug(
                f"[{component_id}] Component 'is_initialized' is True, but metadata.requires_initialize is False. "
                f"This is acceptable if component initializes itself eagerly."
            )

        # 6. Self-Certification Data (as per NIREON V4 Configuration Guide.md, section 7)
        #    The `initialize` method of NireonBaseComponent should call `_self_certify`.
        #    Here, we check if that certification data is present and seems reasonable.
        try:
            certification_data = self.component_registry.get_certification(component_id)
            if not certification_data:
                errors.append(f"Component '{component_id}' is missing self-certification data in the registry.")
            else:
                if certification_data.get("component_id") != component_id:
                    errors.append(f"Self-certification ID mismatch: cert has '{certification_data.get('component_id')}', expected '{component_id}'.")
                if certification_data.get("status") not in ["initialized", "initializing", "healthy"]: # Adjust based on NireonBaseComponent flow
                    # If called after initialize loop, status should be 'initialized' or 'healthy'
                    logger.debug(f"[{component_id}] Self-certification status is '{certification_data.get('status')}'.")
                # Could add more checks on certification_data fields if a schema exists.
        except (ComponentRegistryMissingError, KeyError):
            errors.append(f"Could not retrieve self-certification data for component '{component_id}'.")


        # 7. Expected Interfaces (from V4 ComponentMetadata.expected_interfaces)
        #    This is more aligned with Protocol-based checks.
        if expected_metadata.expected_interfaces: # Check against the *expected* metadata
            for i, expected_iface_protocol in enumerate(expected_metadata.expected_interfaces):
                if not isinstance(instance, expected_iface_protocol): # Runtime checkable protocol
                    errors.append(
                        f"Component '{component_id}' does not implement expected V4 interface protocol #{i + 1}: "
                        f"'{expected_iface_protocol.__name__}' (defined in manifest/canonical metadata)."
                    )
                else:
                    logger.debug(f"[{component_id}] Implements expected V4 interface protocol: {expected_iface_protocol.__name__}")


        if errors:
            logger.warning(f"[{component_id}] Validation failed with {len(errors)} errors: {errors}")
        else:
            logger.info(f"✓ [{component_id}] Passed V4 interface and metadata validation.")
        
        return errors

    def _has_concrete_implementation(self, instance: object, method_name: str) -> bool:
        """Checks if a method has a concrete implementation in the instance's class hierarchy."""
        method = getattr(instance, method_name, None)
        if not method or not callable(method):
            return False
        
        # Check for @abstractmethod decorator (Python 3.3+)
        if hasattr(method, "__isabstractmethod__") and method.__isabstractmethod__:
            return False

        # Check if the method is defined in the instance's specific class or a non-ABC base
        # This logic is a bit complex due to Python's MRO.
        # A simpler check for NireonBaseComponent: if it's not from NireonBaseComponent itself, it's overridden.
        cls = type(instance)
        if method_name in cls.__dict__: # Defined in the instance's direct class
            return True
        
        # For NireonBaseComponent, if _process_impl is still the one from NireonBaseComponent, it's not overridden.
        if isinstance(instance, NireonBaseComponent) and method_name == "_process_impl":
            base_impl = getattr(NireonBaseComponent, "_process_impl", None)
            instance_impl = getattr(cls, "_process_impl", None)
            if instance_impl is base_impl: # Method is inherited directly from NireonBaseComponent without override
                 return False # It's the abstract one from base
            return True # It's overridden

        # General fallback (might not be perfect for all inheritance scenarios with ABCs)
        for base_cls in cls.__mro__[1:]: # Exclude the class itself
            if method_name in base_cls.__dict__:
                # If it's found in a base class, we need to ensure it's not an abstract method from that base
                base_method_attr = getattr(base_cls, method_name)
                if hasattr(base_method_attr, "__isabstractmethod__") and base_method_attr.__isabstractmethod__:
                    # If the version in the MRO is abstract, and it wasn't overridden by `cls`, then it's not concrete.
                    if method_name not in cls.__dict__: return False
                return True # Found a non-abstract version or it was overridden in `cls`
        return False # Should not be reached if method exists