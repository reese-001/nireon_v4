import asyncio
import importlib
import inspect
import json
import logging
from datetime import datetime, timezone
import re
from typing import Any, Dict, Optional, Type, Union
from pydantic import BaseModel # For V4 Pydantic configs

from domain.context import NireonExecutionContext
from core.lifecycle import (
    ComponentMetadata,
    ComponentRegistry,
    ComponentRegistryMissingError,
)
from core.base_component import NireonBaseComponent
from factories.dependencies import CommonMechanismDependencies # V4 common deps
# from nireon.factories.observer_factory import CommonObserverDependencies # V4
# from nireon.factories.manager_factory import CommonManagerDependencies   # V4
from domain.ports.event_bus_port import EventBusPort
from configs.config_utils import ConfigMerger # V4 ConfigMerger

from ._exceptions import BootstrapError
from .health_reporter import (
    BootstrapHealthReporter,
    ComponentStatus,
)
from ...runtime.utils import import_by_path, load_yaml_robust
from .service_resolver import _safe_register_service_instance

import dataclasses
from pathlib import Path

logger = logging.getLogger(__name__)

def _get_pydantic_defaults_v4(
    component_class_or_factory_key: Union[Type, str],
    component_name_for_logging: str
) -> Dict[str, Any]:
    """
    Attempts to get default configuration from a Pydantic model associated with a V4 component.
    It uses conventions to find the Pydantic config class.
    """
    logger.debug(f"Attempting to get Pydantic defaults for V4 component/key: '{component_name_for_logging}' (target: {component_class_or_factory_key})")
    
    config_model_cls: Optional[Type[BaseModel]] = None

    if inspect.isclass(component_class_or_factory_key):
        # Convention 1: Component class has a nested Config Pydantic model
        if hasattr(component_class_or_factory_key, "ConfigModel") and \
           inspect.isclass(getattr(component_class_or_factory_key, "ConfigModel")) and \
           issubclass(getattr(component_class_or_factory_key, "ConfigModel"), BaseModel):
            config_model_cls = getattr(component_class_or_factory_key, "ConfigModel")
            logger.debug(f"Found Pydantic ConfigModel nested in class for '{component_name_for_logging}'.")

        # Convention 2: Look for a `config.py` in the same module or a `config` submodule
        if not config_model_cls:
            module_path = component_class_or_factory_key.__module__
            component_class_name = component_class_or_factory_key.__name__
            
            # Expected Pydantic config class name (e.g., ExplorerMechanism -> ExplorerMechanismConfig)
            expected_config_class_name = f"{component_class_name}Config"
            
            potential_config_module_paths = [
                f"{module_path}.config", # e.g. mechanisms.explorer.config
                module_path # Check in the same module
            ]
            if '.' in module_path: # e.g. nireon.mechanisms.explorer
                parent_module = module_path.rsplit('.', 1)[0]
                potential_config_module_paths.append(f"{parent_module}.config") # e.g. nireon.mechanisms.config

            for config_module_path_attempt in potential_config_module_paths:
                try:
                    config_module = importlib.import_module(config_module_path_attempt)
                    if hasattr(config_module, expected_config_class_name):
                        candidate_cls = getattr(config_module, expected_config_class_name)
                        if inspect.isclass(candidate_cls) and issubclass(candidate_cls, BaseModel):
                            config_model_cls = candidate_cls
                            logger.debug(f"Found Pydantic config class '{expected_config_class_name}' in module '{config_module_path_attempt}'.")
                            break
                except ImportError:
                    logger.debug(f"Module '{config_module_path_attempt}' not found for Pydantic config of '{component_name_for_logging}'.")
                except Exception as e:
                    logger.debug(f"Error importing or accessing config from '{config_module_path_attempt}' for '{component_name_for_logging}': {e}")
            
    elif isinstance(component_class_or_factory_key, str): # It's a factory key
        # For factory keys, Pydantic defaults are less common directly tied to the key.
        # The factory itself might fetch them or they might be part of the component class it creates.
        # This function is more about the class. If it's a key, the factory should handle Pydantic.
        logger.debug(f"Input '{component_class_or_factory_key}' is a string (factory key), Pydantic defaults usually tied to the class it produces. Returning empty.")
        return {}


    if config_model_cls:
        try:
            # Create an instance with default values and dump to dict
            # .model_construct() is preferred for creating models without validation if defaults are trusted
            # If validation of defaults is needed, use `config_model_cls()`
            return config_model_cls.model_construct().model_dump()
        except Exception as e:
            logger.warning(f"Could not get defaults from Pydantic model {config_model_cls.__name__} for '{component_name_for_logging}': {e}")
            return {}
            
    logger.debug(f"No Pydantic config model found or defaults retrievable for '{component_name_for_logging}'. Using empty dict for Pydantic defaults.")
    return {}


async def _create_component_instance_v4(
    component_class: Type, # The actual class of the component
    resolved_config_for_instance: Dict[str, Any],
    instance_id: str, # The ID for this specific instance (from manifest)
    instance_metadata_object: ComponentMetadata, # The fully resolved V4 ComponentMetadata for this instance
    common_deps: Optional[Union[CommonMechanismDependencies, Any]] = None, # V4 common deps for specific types
                                                                        # Make Any for now if observer/manager deps differ
    # V4 factories should be passed if needed, or registry used by the caller
    # mechanism_factory: Optional[SimpleMechanismFactory] = None, 
    # observer_factory: Optional[SimpleObserverFactory] = None,
    # manager_factory: Optional[SimpleManagerFactory] = None,
) -> Optional[NireonBaseComponent]:
    """
    Creates a V4 component instance.
    Assumes NireonBaseComponent derivatives take (config, metadata_definition).
    Uses common_deps if the component type matches and deps are provided.
    """
    logger.debug(f"[_create_component_instance_v4] ID='{instance_id}', Class='{component_class.__name__}', Metadata.ID='{instance_metadata_object.id}'")

    if instance_metadata_object.id != instance_id:
        # This is a safeguard. The caller (init_full_component_v4) should ensure this.
        logger.error(
            f"CRITICAL PRE-INSTANTIATION MISMATCH in _create_component_instance_"
            f"instance_metadata_object.id ('{instance_metadata_object.id}') != instance_id ('{instance_id}'). "
            f"This indicates an issue in the calling logic. Forcing metadata ID to '{instance_id}'."
        )
        instance_metadata_object = dataclasses.replace(instance_metadata_object, id=instance_id)
    
    try:
        # Most V4 NireonBaseComponents should accept (config, metadata_definition)
        # Some might also accept common_deps if they are e.g. mechanisms that need LLM ports directly.
        # This needs careful handling based on component constructor signatures.

        sig = inspect.signature(component_class.__init__)
        constructor_args = {}

        if "config" in sig.parameters:
            constructor_args["config"] = resolved_config_for_instance
        if "metadata_definition" in sig.parameters:
            constructor_args["metadata_definition"] = instance_metadata_object
        
        # Add common dependencies if they are expected by the constructor
        # This part is tricky as different component types might need different deps.
        # For V4, factories are preferred for components needing complex deps.
        # If a component takes `common_deps` directly, it's likely a mechanism.
        if common_deps:
            if "common_deps" in sig.parameters and isinstance(common_deps, CommonMechanismDependencies): # Be specific
                 constructor_args["common_deps"] = common_deps
            # Add elif for other common_deps types (observer, manager) when defined
            elif "llm" in sig.parameters and hasattr(common_deps, "llm_port"):
                constructor_args["llm"] = common_deps.llm_port
            elif "embed" in sig.parameters and hasattr(common_deps, "embedding_port"):
                constructor_args["embed"] = common_deps.embedding_port
            # ... and so on for other direct dependencies if components take them.

        logger.debug(f"Attempting to instantiate '{instance_id}' of type '{component_class.__name__}' with args: {list(constructor_args.keys())}")
        instance = component_class(**constructor_args)
        
        # Post-instantiation checks specific to NireonBaseComponent
        if isinstance(instance, NireonBaseComponent):
            if not hasattr(instance, 'component_id') or instance.component_id != instance_id:
                logger.error(
                    f"CRITICAL MISMATCH: Instance '{getattr(instance, 'component_id', 'MISSING')}' ID "
                    f"after creation does not match manifest ID '{instance_id}'. Overriding on instance."
                )
                # Directly set the attribute if possible (Python allows this)
                object.__setattr__(instance, '_component_id', instance_id)
            
            if not hasattr(instance, 'metadata') or instance.metadata.id != instance_id:
                logger.error(
                    f"CRITICAL MISMATCH: Instance metadata ID '{getattr(getattr(instance, 'metadata', None), 'id', 'MISSING')}' "
                    f"after creation does not match manifest ID '{instance_id}'. Overriding on instance."
                )
                object.__setattr__(instance, '_metadata_definition', instance_metadata_object)
        return instance

    except TypeError as err:
        logger.error(
            f"Instantiation TypeError for '{instance_id}' (Class: {component_class.__name__}) "
            f"with metadata ID '{instance_metadata_object.id}'. Error: {err}",
            exc_info=True
        )
        # Add more sophisticated fallback attempts if needed, similar to V3's _create_component_instance
        # But V4 should ideally have more standardized constructors or use factories.
        if issubclass(component_class, NireonBaseComponent): # Check if it's our base
             raise BootstrapError(
                f"NireonBaseComponent derivative '{instance_id}' failed instantiation with standard V4 signature. "
                f"Constructor signature was: {sig}. Provided args: {list(constructor_args.keys())}. Error: {err}"
            ) from err
        else: # For non-NireonBaseComponent, try simpler patterns
            logger.debug(f"Primary instantiation attempt for non-NireonBaseComponent '{instance_id}' failed: {err} – trying simpler fallbacks.")
            try:
                return component_class(config=resolved_config_for_instance)
            except TypeError:
                try:
                    return component_class() # No-args constructor
                except TypeError as err_no_args:
                    logger.error(f"All instantiation attempts for '{instance_id}' failed. Initial error: {err}. No-args error: {err_no_args}", exc_info=True)
                    raise BootstrapError(f"Instantiation attempts exhausted for '{instance_id}'. Last error: {err_no_args}") from err_no_args
    except Exception as exc:
        logger.error(f"General instantiation failure for '{instance_id}': {exc}", exc_info=True)
        raise BootstrapError(f"Instantiation for '{instance_id}' failed: {exc}") from exc

async def init_full_component_v4(
    cid_manifest: str, # Component ID from the manifest
    spec: Dict[str, Any], # Component specification from the manifest
    registry: ComponentRegistry,
    event_bus: EventBusPort, # V4 EventBusPort
    run_id: str, # Current bootstrap run_id
    replay_context: bool, # Indicates if in replay mode
    common_deps: Optional[Union[CommonMechanismDependencies, Any]] = None, # V4 common deps
    global_app_config: Dict[str, Any] = None,
    health_reporter: BootstrapHealthReporter = None,
    validation_data_store: Any = None # BootstrapValidationData
    # perform_initialization is intentionally REMOVED. This function ONLY creates and registers.
) -> None:
    """
    Initializes a V4 component based on an "enhanced" manifest specification.
    This involves:
    1. Importing the component class and its canonical metadata.
    2. Merging configurations (Pydantic defaults, YAML file, manifest inline).
    3. Instantiating the component.
    4. Registering the instance with the V4 ComponentRegistry.
    It DOES NOT call the component's .initialize() method.
    """
    global_app_config = global_app_config or {}
    class_path: str = spec.get("class")
    metadata_definition_path: str = spec.get("metadata_definition") # Path to V4 ComponentMetadata definition
    instance_config_file_template: Optional[str] = spec.get("config") # Path template for instance-specific YAML
    inline_config_override_from_manifest = spec.get("config_override", {})
    is_enabled = spec.get("enabled", True)
    is_strict_mode = global_app_config.get("bootstrap_strict_mode", True)

    logger.info(f"-> Processing V4 Full Component: {cid_manifest} (Class: {class_path}, Enabled: {is_enabled})")

    # Fallback metadata for early error reporting
    base_name_for_report = class_path.split(":")[-1] if class_path and ":" in class_path else \
                           class_path.split(".")[-1] if class_path else cid_manifest
    category_from_spec = spec.get("type", "unknown_component_type") # 'type' in manifest maps to 'category' in ComponentMetadata
    
    fallback_metadata = ComponentMetadata(
        id=cid_manifest, name=base_name_for_report, version="0.0.0", category=category_from_spec
    )

    if not is_enabled:
        logger.info(f"Component '{cid_manifest}' is disabled in manifest. Skipping.")
        if health_reporter: health_reporter.add_component_status(cid_manifest, ComponentStatus.INSTANCE_REGISTERED_NO_INIT, fallback_metadata, ["Disabled in manifest"])
        return

    if not class_path or not metadata_definition_path:
        msg = f"V4 Component '{cid_manifest}' definition missing 'class' ('{class_path}') or 'metadata_definition' ('{metadata_definition_path}')."
        logger.error(msg)
        if health_reporter: health_reporter.add_component_status(cid_manifest, ComponentStatus.DEFINITION_ERROR, fallback_metadata, [msg])
        if is_strict_mode: raise BootstrapError(msg)
        return

    # 1. Import component class and its canonical metadata definition
    try:
        component_class = import_by_path(class_path)
        if not inspect.isclass(component_class): # Ensure it's a class
             raise TypeError(f"Path '{class_path}' for component '{cid_manifest}' did not resolve to a class.")
    except (ImportError, TypeError) as e_cls:
        msg = f"Failed to import/validate class '{class_path}' for component '{cid_manifest}': {e_cls}"
        logger.error(msg, exc_info=True)
        if health_reporter: health_reporter.add_component_status(cid_manifest, ComponentStatus.DEFINITION_ERROR, fallback_metadata, [msg])
        if is_strict_mode: raise BootstrapError(msg) from e_cls
        return

    try:
        canonical_metadata_definition: ComponentMetadata = import_by_path(metadata_definition_path)
        if not isinstance(canonical_metadata_definition, ComponentMetadata):
            msg = f"Imported metadata_definition '{metadata_definition_path}' for '{cid_manifest}' is not a ComponentMetadata instance (type: {type(canonical_metadata_definition)})."
            logger.error(msg)
            if health_reporter: health_reporter.add_component_status(cid_manifest, ComponentStatus.METADATA_ERROR, fallback_metadata, [msg])
            if is_strict_mode: raise BootstrapError(msg)
            return
    except ImportError as e_meta_def:
        msg = f"Failed to import canonical metadata_definition '{metadata_definition_path}' for component '{cid_manifest}': {e_meta_def}"
        logger.error(msg, exc_info=True)
        if health_reporter: health_reporter.add_component_status(cid_manifest, ComponentStatus.METADATA_ERROR, fallback_metadata, [msg])
        if is_strict_mode: raise BootstrapError(msg) from e_meta_def
        return

    # 2. Prepare instance-specific metadata (override ID, apply manifest overrides)
    instance_specific_metadata_dict = dataclasses.asdict(canonical_metadata_definition)
    instance_specific_metadata_dict["id"] = cid_manifest # Manifest ID is the instance ID
    
    manifest_meta_override = spec.get("metadata_override", {})
    if manifest_meta_override:
        logger.debug(f"[{cid_manifest}] Applying manifest metadata_override: {manifest_meta_override}")
        for key, value in manifest_meta_override.items():
            if key in instance_specific_metadata_dict:
                instance_specific_metadata_dict[key] = value
            elif key == "requires_initialize" and isinstance(value, bool):
                 instance_specific_metadata_dict[key] = value
            else:
                logger.warning(f"[{cid_manifest}] Unknown key '{key}' or invalid type in metadata_override. Ignoring.")
    
    if "epistemic_tags" in spec: # Manifest tags override canonical
        tags = spec["epistemic_tags"]
        if isinstance(tags, list) and all(isinstance(t, str) for t in tags):
            instance_specific_metadata_dict["epistemic_tags"] = tags
        else:
            logger.warning(f"[{cid_manifest}] Invalid 'epistemic_tags' in manifest. Using from canonical metadata.")
    
    if 'requires_initialize' not in instance_specific_metadata_dict:
        instance_specific_metadata_dict['requires_initialize'] = canonical_metadata_definition.requires_initialize

    try:
        instance_metadata = ComponentMetadata(**instance_specific_metadata_dict)
    except Exception as e_meta_final:
        msg = f"Failed to construct final ComponentMetadata for '{cid_manifest}': {e_meta_final}"
        logger.error(msg, exc_info=True)
        if health_reporter: health_reporter.add_component_status(cid_manifest, ComponentStatus.METADATA_CONSTRUCTION_ERROR, fallback_metadata, [msg])
        if is_strict_mode: raise BootstrapError(msg) from e_meta_final
        return

    logger.info(f"[{cid_manifest}] Instance Metadata prepared. ID='{instance_metadata.id}', Name='{instance_metadata.name}', Category='{instance_metadata.category}', ReqInit='{instance_metadata.requires_initialize}'")

    # 3. Configuration Merging
    # Pydantic defaults -> YAML file -> Inline manifest config_override
    pydantic_defaults = _get_pydantic_defaults_v4(component_class, instance_metadata.name)
    logger.debug(f"[{cid_manifest}] Pydantic defaults: {list(pydantic_defaults.keys())}")

    yaml_file_config = {}
    if instance_config_file_template:
        actual_config_path_str = instance_config_file_template.replace("{id}", cid_manifest)
        yaml_file_config = load_yaml_robust(Path(actual_config_path_str))
        # Add error handling for failed YAML load if needed
        logger.debug(f"[{cid_manifest}] Loaded YAML from '{actual_config_path_str}': {list(yaml_file_config.keys())}")
    
    merged_config_step1 = ConfigMerger.merge(pydantic_defaults, yaml_file_config, f"{cid_manifest}_pydantic_yaml")
    final_resolved_config = ConfigMerger.merge(merged_config_step1, inline_config_override_from_manifest, f"{cid_manifest}_final")
    
    logger.info(f"[{cid_manifest}] Final resolved config (top-level keys): {list(final_resolved_config.keys())}")
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"[{cid_manifest}] Full resolved config:\n{json.dumps(final_resolved_config, indent=2, default=str)}")

    if validation_data_store:
        validation_data_store.store_component_data(
            component_id=cid_manifest,
            original_metadata=instance_metadata, # This is the resolved metadata for this specific instance
            resolved_config=final_resolved_config,
            manifest_spec=spec
        )

    # 4. Instantiate the component
    try:
        instance = await _create_component_instance_v4(
            component_class=component_class,
            resolved_config_for_instance=final_resolved_config,
            instance_id=cid_manifest, # Use manifest ID
            instance_metadata_object=instance_metadata, # Pass the fully resolved instance metadata
            common_deps=common_deps
        )
    except BootstrapError as e_create:
        if health_reporter: health_reporter.add_component_status(cid_manifest, ComponentStatus.INSTANTIATION_ERROR, instance_metadata, [str(e_create)])
        if is_strict_mode: raise
        return

    if instance is None:
        msg = f"Instantiation for V4 component '{cid_manifest}' returned None."
        if health_reporter: health_reporter.add_component_status(cid_manifest, ComponentStatus.INSTANTIATION_ERROR, instance_metadata, [msg])
        if is_strict_mode: raise BootstrapError(msg)
        return

    # 5. Register the instance (NO .initialize() call here)
    try:
        _safe_register_service_instance(
            registry,
            type(instance), # Register by its actual type
            instance,
            instance.component_id, # Use the ID from the instance (should match cid_manifest)
            instance.metadata.category,
            description_for_meta=instance.metadata.description,
            requires_initialize_override=instance.metadata.requires_initialize
        )
        if health_reporter:
            health_reporter.add_component_status(
                instance.component_id,
                ComponentStatus.INSTANCE_REGISTERED_INIT_DEFERRED, # Mark as registered, init is later
                instance.metadata,
                []
            )
        logger.info(f"✓ V4 Component Instantiated & Registered: {instance.component_id} (Type: {type(instance).__name__}, Requires Init: {instance.metadata.requires_initialize})")

    except BootstrapError as e_reg:
        if health_reporter: health_reporter.add_component_status(instance.component_id, ComponentStatus.BOOTSTRAP_ERROR, instance.metadata, [str(e_reg)])
        if is_strict_mode: raise
    except Exception as exc_unexpected_reg:
        logger.error(f"Unexpected error during registration of V4 component '{instance.component_id}': {exc_unexpected_reg}", exc_info=True)
        if health_reporter: health_reporter.add_component_status(instance.component_id, ComponentStatus.BOOTSTRAP_ERROR, instance.metadata, [str(exc_unexpected_reg)])
        if is_strict_mode: raise BootstrapError(f"Unexpected registration error for '{instance.component_id}'") from exc_unexpected_reg