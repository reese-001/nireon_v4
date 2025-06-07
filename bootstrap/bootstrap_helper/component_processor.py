import asyncio
import dataclasses
import inspect
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from core.base_component import NireonBaseComponent
from core.lifecycle import (
    ComponentMetadata,
    ComponentRegistry,
    ComponentRegistryMissingError,
)
from factories.mechanism_factory import SimpleMechanismFactory
# from nireon.factories.observer_factory import SimpleObserverFactory # For V4
# from nireon.factories.manager_factory import SimpleManagerFactory   # For V4
from domain.ports.event_bus_port import EventBusPort
from validators.interface_validator import InterfaceValidator # V4 validator


from ._exceptions import BootstrapError, StepCommandError
from .health_reporter import (
    BootstrapHealthReporter,
    ComponentStatus,
)
from .metadata import get_default_metadata as get_default_metadata # V4 version
from .service_resolver import _safe_register_service_instance
from ...runtime.utils import import_by_path, load_yaml_robust
from .enhanced_components import _create_component_instance, _get_pydantic_defaults
from configs.config_utils import ConfigMerger # V4 ConfigMerger

logger = logging.getLogger(__name__)

async def process_simple_component(
    comp_def: Dict[str, Any],
    registry: ComponentRegistry,
    mechanism_factory: SimpleMechanismFactory,
    # observer_factory: SimpleObserverFactory, # Add when V4 factories exist
    # manager_factory: SimpleManagerFactory,
    health_reporter: BootstrapHealthReporter,
    run_id: str,
    global_app_config: Dict[str, Any],
    validation_data_store: Any # Type hint as BootstrapValidationData when defined
) -> None:
    """
    Processes a component definition from a "simple" style manifest for V4.
    Instantiates and registers the component. Initialization is deferred.
    """
    component_id = comp_def.get("component_id")
    factory_key = comp_def.get("factory_key") # May not be used directly if comp_def has 'class'
    component_class_path = comp_def.get("class")
    component_type = comp_def.get("type") # e.g., 'mechanism', 'observer', 'manager'
    yaml_config_override = comp_def.get("config", {}) # Config from simple manifest
    is_enabled = comp_def.get("enabled", True)
    is_strict_mode = global_app_config.get("bootstrap_strict_mode", True)

    # Use a base metadata for reporting if things go wrong early
    current_component_metadata_for_reporting = ComponentMetadata(
        id=component_id or "unknown_id_simple",
        name=factory_key or component_class_path or "UnknownNameSimple",
        version="0.0.0", # Default, will be overridden
        category=component_type or "unknown_type_simple",
        requires_initialize=True # Default, might be overridden
    )

    if not is_enabled:
        logger.info(f"Simple component '{component_id}' is disabled. Skipping.")
        health_reporter.add_component_status(
            component_id or "unknown_disabled_component",
            ComponentStatus.INSTANCE_REGISTERED_NO_INIT, # Or a more specific "SKIPPED" status
            current_component_metadata_for_reporting,
            ["Disabled in manifest"],
        )
        return

    if not all([component_id, (factory_key or component_class_path), component_type]):
        errmsg = f"Skipping simple component definition due to missing id, (factory_key or class), or type: {comp_def}"
        logger.error(errmsg)
        health_reporter.add_component_status(
            component_id or "unknown_component_def_error",
            ComponentStatus.DEFINITION_ERROR,
            current_component_metadata_for_reporting,
            [errmsg],
        )
        if is_strict_mode:
            raise BootstrapError(errmsg)
        return

    # Determine actual class to instantiate (either via factory_key or direct class path)
    target_class_or_factory_key = component_class_path if component_class_path else factory_key
    
    # Build V4 ComponentMetadata
    # In "simple" manifest, metadata might be less explicit. We might rely on defaults.
    # V4 might not have `get_default_metadata` if simple manifests are deprecated.
    # For now, assume we try to build it, potentially falling back.
    base_metadata_for_class = get_default_metadata(factory_key) # V3 style mapping
    if not base_metadata_for_class and component_class_path:
        # Attempt to get metadata from class if class_path provided (V4 style)
        try:
            cls_for_meta = import_by_path(component_class_path)
            if hasattr(cls_for_meta, "METADATA_DEFINITION"): # Convention for V4 components
                base_metadata_for_class = cls_for_meta.METADATA_DEFINITION
        except Exception:
             pass # Fall through

    if not base_metadata_for_class:
        logger.warning(f"No default or class-defined metadata found for '{target_class_or_factory_key}'. Using minimal fallback for component '{component_id}'.")
        base_metadata_for_class = ComponentMetadata(
            id=component_id, name=component_id, version="unknown", category=component_type
        )

    try:
        final_instance_metadata = _build_component_metadata(
            base_metadata_for_class, component_id, comp_def
        )
        current_component_metadata_for_reporting = final_instance_metadata
    except Exception as e_meta:
        errmsg = f"Error constructing ComponentMetadata for simple component '{component_id}': {e_meta}"
        logger.error(errmsg, exc_info=True)
        health_reporter.add_component_status(component_id, ComponentStatus.METADATA_CONSTRUCTION_ERROR, current_component_metadata_for_reporting, [errmsg])
        if is_strict_mode: raise BootstrapError(errmsg) from e_meta
        return

    # Configuration merging (Pydantic defaults -> simple manifest config)
    # For simple components, Pydantic defaults might be less common, but we check.
    pydantic_defaults = {}
    if component_class_path:
        try:
            cls_for_pydantic = import_by_path(component_class_path)
            pydantic_defaults = _get_pydantic_defaults(cls_for_pydantic, final_instance_metadata.name)
        except Exception as e_pydantic:
            logger.debug(f"Could not get Pydantic defaults for simple component '{component_id}': {e_pydantic}")
    
    final_config = ConfigMerger.merge(pydantic_defaults, yaml_config_override, f"{component_id}_simple_config_merge")

    if validation_data_store:
        validation_data_store.store_component_data(
            component_id=component_id,
            original_metadata=final_instance_metadata,
            resolved_config=final_config,
            manifest_spec=comp_def
        )

    instance: Optional[NireonBaseComponent] = None
    try:
        # Instantiation logic:
        # If 'class' is provided, try direct instantiation.
        # If 'factory_key' is provided and type is 'mechanism', use mechanism_factory.
        # (Extend for observer/manager factories when they exist for V4)
        if component_class_path:
            cls_to_instantiate = import_by_path(component_class_path)
            instance = await _create_component_instance(
                cls_to_instantiate, final_config, component_id, final_instance_metadata, None # No common_deps for direct instantiation here
            )
        elif factory_key and component_type == "mechanism" and mechanism_factory:
            instance = mechanism_factory.create_mechanism( # V4 factory usage
                factory_key, final_instance_metadata, final_config
            )
        # Add elif for observer_factory, manager_factory when they are part of V4
        else:
            errmsg = f"Cannot determine instantiation method for simple component '{component_id}'. Need 'class' or a typed 'factory_key'."
            raise BootstrapError(errmsg)

        if instance is None:
            errmsg = f"Component instantiation returned None for simple component '{component_id}' (target: {target_class_or_factory_key})"
            health_reporter.add_component_status(component_id, ComponentStatus.INSTANTIATION_ERROR, final_instance_metadata, [errmsg])
            if is_strict_mode: raise BootstrapError(errmsg)
            return

        logger.info(f"Instantiated simple component '{component_id}' (Type: {component_type}, Target: {target_class_or_factory_key})")
        
        # Registration (NO INITIALIZATION HERE)
        _safe_register_service_instance(
            registry, type(instance), instance, component_id, final_instance_metadata.category,
            description_for_meta=final_instance_metadata.description,
            requires_initialize_override=final_instance_metadata.requires_initialize
        )
        # Update health reporter status after successful registration
        health_reporter.add_component_status(
            component_id, 
            ComponentStatus.INSTANCE_REGISTERED_INIT_DEFERRED, # Mark as registered, init later
            instance.metadata, # Use metadata from the instance itself
            []
        )

    except Exception as e:
        errmsg = f"Error during instantiation/registration of simple component '{component_id}': {e}"
        logger.error(errmsg, exc_info=True)
        health_reporter.add_component_status(component_id, ComponentStatus.BOOTSTRAP_ERROR, current_component_metadata_for_reporting, [errmsg])
        if is_strict_mode: raise BootstrapError(errmsg) from e


def _build_component_metadata(
    base_metadata: ComponentMetadata, # This should be the METADATA_DEFINITION from the component class
    component_id_from_manifest: str,
    manifest_comp_definition: Dict[str, Any],
) -> ComponentMetadata:
    """
    Builds the final ComponentMetadata for an instance in V4.
    The `base_metadata` is expected to be the canonical definition from the component's code.
    Overrides from the manifest are applied.
    """
    if not isinstance(base_metadata, ComponentMetadata):
        raise TypeError(f"base_metadata for '{component_id_from_manifest}' must be ComponentMetadata, got {type(base_metadata)}")

    # Start with a copy of the base metadata from the component's definition
    instance_metadata_dict = dataclasses.asdict(base_metadata)

    # The ID from the manifest is the canonical ID for this instance
    instance_metadata_dict["id"] = component_id_from_manifest

    # Apply overrides from the manifest spec
    manifest_meta_override = manifest_comp_definition.get("metadata_override", {})
    if manifest_meta_override:
        logger.debug(f"[{component_id_from_manifest}] Applying manifest metadata_override: {manifest_meta_override}")
        for key, value in manifest_meta_override.items():
            if key in instance_metadata_dict:
                instance_metadata_dict[key] = value
            elif key == "requires_initialize" and isinstance(value, bool): # Special case for this boolean
                instance_metadata_dict[key] = value
            else:
                logger.warning(
                    f"[{component_id_from_manifest}] Unknown key '{key}' or invalid type in metadata_override for simple component. Ignoring."
                )
    
    # Explicit epistemic_tags from manifest override anything from base
    if "epistemic_tags" in manifest_comp_definition:
        tags = manifest_comp_definition["epistemic_tags"]
        if isinstance(tags, list) and all(isinstance(t, str) for t in tags):
            instance_metadata_dict["epistemic_tags"] = tags
            logger.debug(f"[{component_id_from_manifest}] Using epistemic_tags from manifest: {tags}")
        else:
            logger.warning(f"[{component_id_from_manifest}] Invalid 'epistemic_tags' in manifest, expected list of strings. Using base.")


    # Ensure 'requires_initialize' is present, defaulting from base if not overridden
    if 'requires_initialize' not in instance_metadata_dict:
        instance_metadata_dict['requires_initialize'] = base_metadata.requires_initialize

    return ComponentMetadata(**instance_metadata_dict)


async def instantiate_shared_service(
    service_key_in_manifest: str,
    service_spec_from_manifest: Dict[str, Any],
    registry: ComponentRegistry,
    event_bus: EventBusPort, # V4 EventBusPort
    global_app_config: Dict[str, Any],
    health_reporter: BootstrapHealthReporter,
    validation_data_store: Any # BootstrapValidationData
) -> None:
    """
    Instantiates and registers a shared service defined in an enhanced manifest for V4.
    Handles V4 configuration (Pydantic defaults, YAML, manifest overrides).
    Initialization is deferred.
    """
    class_path = service_spec_from_manifest.get("class")
    config_path_template = service_spec_from_manifest.get("config") # Path to YAML config file
    inline_config_override = service_spec_from_manifest.get("config_override", {}) # Inline in manifest
    is_enabled = service_spec_from_manifest.get("enabled", True)
    is_strict_mode = global_app_config.get("bootstrap_strict_mode", True)
    
    # Metadata for reporting, might be refined if METADATA_DEFINITION is found
    service_name_for_report = class_path.split(":")[-1] if class_path and ":" in class_path else \
                              class_path.split(".")[-1] if class_path else service_key_in_manifest
    base_meta_for_report = ComponentMetadata(
        id=service_key_in_manifest, name=service_name_for_report, category="shared_service", version="unknown"
    )

    if not class_path:
        msg = f"Shared service '{service_key_in_manifest}' definition missing 'class' path."
        logger.error(msg)
        health_reporter.add_component_status(service_key_in_manifest, ComponentStatus.DEFINITION_ERROR, base_meta_for_report, [msg])
        if is_strict_mode: raise BootstrapError(msg)
        return

    if not is_enabled:
        logger.info(f"Shared service '{service_key_in_manifest}' (Class: {class_path}) is disabled. Skipping.")
        health_reporter.add_component_status(service_key_in_manifest, ComponentStatus.INSTANCE_REGISTERED_NO_INIT, base_meta_for_report, ["Disabled in manifest"])
        return

    try: # Check if already registered (e.g., by build_container or previous manifest)
        existing_instance = registry.get(service_key_in_manifest)
        if existing_instance is not None:
            logger.info(f"Service '{service_key_in_manifest}' (Class: {class_path}) already in registry. Skipping manifest instantiation.")
            # Optionally update health status if not already properly reported
            return
    except (KeyError, ComponentRegistryMissingError):
        logger.info(f"Service '{service_key_in_manifest}' not found in registry by key. Proceeding with instantiation.")
    
    logger.info(f"-> Instantiating Shared Service from manifest: {service_key_in_manifest} (Class: {class_path})")

    try:
        service_class = import_by_path(class_path)
    except ImportError as e_import:
        msg = f"Failed to import class '{class_path}' for shared service '{service_key_in_manifest}': {e_import}"
        logger.error(msg, exc_info=True)
        health_reporter.add_component_status(service_key_in_manifest, ComponentStatus.DEFINITION_ERROR, base_meta_for_report, [msg])
        if is_strict_mode: raise BootstrapError(msg) from e_import
        return

    # V4 Configuration Merging:
    # 1. Pydantic defaults (if service_class is NireonBaseComponent with Pydantic config model)
    # 2. YAML file config (from service_spec_from_manifest['config'])
    # 3. Inline config from manifest (service_spec_from_manifest['config_override'])
    
    pydantic_defaults = _get_pydantic_defaults(service_class, service_name_for_report)
    
    yaml_file_config = {}
    if config_path_template:
        # Resolve {id} if present, though less common for shared services
        actual_config_path = config_path_template.replace("{id}", service_key_in_manifest)
        yaml_file_config = load_yaml_robust(Path(actual_config_path))
        if not yaml_file_config and Path(actual_config_path).is_file() and Path(actual_config_path).read_text(encoding='utf-8').strip():
             msg = f"Failed to parse non-empty config YAML '{actual_config_path}' for service '{service_key_in_manifest}'."
             logger.error(msg)
             # health_reporter... if is_strict_mode: raise ...

    # Merge: Pydantic -> YAML file
    merged_config_step1 = ConfigMerger.merge(pydantic_defaults, yaml_file_config, f"{service_key_in_manifest}_pydantic_yaml")
    # Merge: (Pydantic -> YAML file) -> Inline manifest override
    final_service_config = ConfigMerger.merge(merged_config_step1, inline_config_override, f"{service_key_in_manifest}_final")
    
    # Determine the metadata for this service instance
    # V4 services might define their own METADATA_DEFINITION static attribute
    service_instance_metadata = base_meta_for_report # Fallback
    if hasattr(service_class, "METADATA_DEFINITION") and isinstance(service_class.METADATA_DEFINITION, ComponentMetadata):
        service_instance_metadata = dataclasses.replace(service_class.METADATA_DEFINITION, id=service_key_in_manifest)
    elif hasattr(service_class, "metadata") and isinstance(service_class.metadata, ComponentMetadata): # Less common for class
         service_instance_metadata = dataclasses.replace(service_class.metadata, id=service_key_in_manifest)
    
    if validation_data_store:
        validation_data_store.store_component_data(
            component_id=service_key_in_manifest,
            original_metadata=service_instance_metadata, # This is the "expected" metadata from its definition
            resolved_config=final_service_config,
            manifest_spec=service_spec_from_manifest
        )
        
    service_instance: Optional[Any] = None
    try:
        # _create_component_instance is designed for NireonBaseComponent style.
        # Shared services might not always be NireonBaseComponent.
        # We need a more generic instantiation helper or handle cases.
        if issubclass(service_class, NireonBaseComponent):
            service_instance = await _create_component_instance(
                component_class=service_class,
                resolved_config_for_instance=final_service_config,
                instance_id=service_key_in_manifest, # This is the key from manifest
                instance_metadata_object=service_instance_metadata, # This metadata has ID set to service_key_in_manifest
                common_deps=None # Shared services usually don't take common_mechanism_deps
            )
        else: # Generic instantiation for non-NireonBaseComponent services
            # Try instantiating with common patterns, assuming they might take config or event_bus
            # This is a simplified version of V3's create_service_instance logic
            ctor_params = inspect.signature(service_class.__init__).parameters
            kwargs_for_ctor = {}
            if 'config' in ctor_params: kwargs_for_ctor['config'] = final_service_config
            if 'cfg' in ctor_params: kwargs_for_ctor['cfg'] = final_service_config # Common alias
            if 'event_bus' in ctor_params: kwargs_for_ctor['event_bus'] = event_bus
            if 'registry' in ctor_params: kwargs_for_ctor['registry'] = registry # Some services might need registry
            # Add other common dependency checks if needed (e.g., global_app_config)
            service_instance = service_class(**kwargs_for_ctor)

    except Exception as e_create:
        msg = f"Instantiation failed for shared service '{service_key_in_manifest}': {e_create}"
        logger.error(msg, exc_info=True)
        health_reporter.add_component_status(service_key_in_manifest, ComponentStatus.INSTANTIATION_ERROR, service_instance_metadata, [msg])
        if is_strict_mode: raise BootstrapError(msg) from e_create
        return

    if service_instance:
        # For shared services, the service_key_in_manifest is the primary ID.
        # If it's a NireonBaseComponent, its internal metadata.id should match.
        # The requires_initialize flag comes from its own metadata.
        final_requires_init = getattr(service_instance_metadata, 'requires_initialize', False) # Default to False for non-NireonBaseComponents

        _safe_register_service_instance(
            registry,
            service_class, # Register by its actual class type
            service_instance,
            service_key_in_manifest, # Use manifest key as the primary ID
            service_instance_metadata.category,
            description_for_meta=service_instance_metadata.description,
            requires_initialize_override=final_requires_init
        )
        logger.info(f"✓ Shared Service Instantiated and Registered: {service_key_in_manifest} -> {service_class.__name__} (Requires Init: {final_requires_init})")
        health_reporter.add_component_status(
            service_key_in_manifest,
            ComponentStatus.INSTANCE_REGISTERED_INIT_DEFERRED, # Initialization happens in Phase 5
            service_instance_metadata,
            []
        )
    else:
        msg = f"Service instantiation returned None for '{service_key_in_manifest}' using class {service_class.__name__}"
        logger.error(msg)
        health_reporter.add_component_status(service_key_in_manifest, ComponentStatus.INSTANTIATION_ERROR, service_instance_metadata, [msg])
        if is_strict_mode: raise BootstrapError(msg)


async def register_orchestration_command(
    command_id_from_manifest: str,
    command_spec_from_manifest: Dict[str, Any],
    registry: ComponentRegistry,
    health_reporter: BootstrapHealthReporter,
    global_app_config: Dict[str, Any],
) -> None:
    """
    Registers an orchestration command CLASS and its METADATA into the registry.
    Does not instantiate the command.
    """
    class_path: str = command_spec_from_manifest.get("class")
    metadata_definition_path: Optional[str] = command_spec_from_manifest.get("metadata_definition")
    is_enabled = command_spec_from_manifest.get("enabled", True)
    is_strict_mode = global_app_config.get("bootstrap_strict_mode", True)

    logger.info(f"-> Registering Orchestration Command Class: {command_id_from_manifest} (Class: {class_path}, Enabled: {is_enabled})")
    
    # Fallback metadata if things go wrong early
    base_name_for_report = class_path.split(":")[-1] if class_path and ":" in class_path else \
                           class_path.split(".")[-1] if class_path else command_id_from_manifest
    fallback_meta = ComponentMetadata(
        id=command_id_from_manifest, name=base_name_for_report, version="0.0.0",
        category="orchestration_command_definition", # Special category for class definitions
        requires_initialize=False # Command classes are not initialized like components
    )

    if not is_enabled:
        logger.info(f"Command class '{command_id_from_manifest}' is disabled. Skipping registration.")
        health_reporter.add_component_status(command_id_from_manifest, ComponentStatus.INSTANCE_REGISTERED_NO_INIT, fallback_meta, ["Disabled in manifest (class registration)"])
        return

    if not class_path:
        msg = f"Orchestration command '{command_id_from_manifest}' definition missing 'class' path."
        logger.error(msg)
        health_reporter.add_component_status(command_id_from_manifest, ComponentStatus.DEFINITION_ERROR, fallback_meta, [msg])
        if is_strict_mode: raise BootstrapError(msg)
        return

    try:
        command_class = import_by_path(class_path)
        if not callable(command_class): # Should be a class
             raise TypeError(f"Path '{class_path}' for command '{command_id_from_manifest}' did not resolve to a callable class.")
    except (ImportError, TypeError) as exc_import:
        msg = f"Unable to import/validate class '{class_path}' for command '{command_id_from_manifest}': {exc_import}"
        logger.error(msg, exc_info=True)
        health_reporter.add_component_status(command_id_from_manifest, ComponentStatus.DEFINITION_ERROR, fallback_meta, [msg])
        if is_strict_mode: raise BootstrapError(msg) from exc_import
        return

    # Load or define metadata for the command class
    command_instance_metadata: ComponentMetadata = fallback_meta
    if metadata_definition_path:
        try:
            imported_meta_obj = import_by_path(metadata_definition_path)
            if isinstance(imported_meta_obj, ComponentMetadata):
                # Override ID with manifest key, ensure category is appropriate
                command_instance_metadata = dataclasses.replace(
                    imported_meta_obj, 
                    id=command_id_from_manifest, 
                    category="orchestration_step", # V4 uses this for StepCommand type
                    requires_initialize=False # Command classes don't get initialized
                )
            else:
                logger.warning(f"Imported metadata '{metadata_definition_path}' for command '{command_id_from_manifest}' is not ComponentMetadata. Using fallback.")
        except ImportError as exc_meta_import:
            msg = f"Could not import metadata_definition '{metadata_definition_path}' for command '{command_id_from_manifest}': {exc_meta_import}."
            logger.warning(msg)
            health_reporter.add_component_status(command_id_from_manifest, ComponentStatus.METADATA_ERROR, fallback_meta, [msg + " Using fallback metadata."])
    else: # If no explicit metadata_definition, try to use a class attribute or create minimal
        if hasattr(command_class, "METADATA_DEFINITION") and isinstance(command_class.METADATA_DEFINITION, ComponentMetadata):
            command_instance_metadata = dataclasses.replace(
                command_class.METADATA_DEFINITION, 
                id=command_id_from_manifest,
                category="orchestration_step",
                requires_initialize=False
            )
        else:
            command_instance_metadata = ComponentMetadata(
                id=command_id_from_manifest, name=getattr(command_class, "__name__", command_id_from_manifest),
                version="0.1.0", category="orchestration_step",
                description=f"Dynamically registered command class: {getattr(command_class, '__name__', command_id_from_manifest)}",
                requires_initialize=False
            )
    
    try:
        # Register the CLASS ITSELF, not an instance, with its metadata
        registry.register(command_class, command_instance_metadata)
        logger.info(f"✓ Orchestration Command Class Registered: {command_instance_metadata.id} -> {command_class.__name__} (Type: {command_instance_metadata.category})")
        health_reporter.add_component_status(command_id_from_manifest, ComponentStatus.INSTANCE_REGISTERED, command_instance_metadata, []) # Using INSTANCE_REGISTERED for class defs too
    except Exception as exc_reg:
        msg = f"Error registering command class '{command_id_from_manifest}' with metadata '{command_instance_metadata.id}': {exc_reg}"
        logger.error(msg, exc_info=True)
        health_reporter.add_component_status(command_id_from_manifest, ComponentStatus.REGISTRATION_ERROR, command_instance_metadata, [msg])
        if is_strict_mode: raise BootstrapError(msg) from exc_reg